// Inkling 2 (Thinking Machines) text model.
// Written from transformers main models/inkling2/modular_inkling2.py. Every layer has both an attention KV cache
// and a packed short-conv state [k | v | attn | mlp] (last K-1 inputs of each stream).
// Attention is computed without flash attention (per-token sinks + relative position bias).

#include "models.h"

#include "../llama-kv-cache-iswa.h"
#include "../llama-kv-cache.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-recurrent.h"

#include <cfloat>
#include <type_traits>
#include <cstring>
#include <cstdlib>
#include <cmath>

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping);

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // visible iff pos_q - pos_k < n_swa
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = 1;
    }

    const std::string a = "inkling2.";
    // GB10/ink2: B70's public TQ1 (PR #32217 converter of 11/19, inkling2.tq.* keys) lacks these keys: the shapes give
    // d_rel / rel_extent / the conv kernel, the rest are Inkling 2 Small's config.json constants (as this fork's GGUFs carry)
    std::string tq_tile;
    const bool pub = ml.get_key(a + "tq.doc", tq_tile, false);
    if (pub) {
        auto meta = [&](const char * n) { return ml.get_tensor_meta(n); };
        const ggml_tensor * r0 = meta("blk.0.attn_r.weight");
        const ggml_tensor * p0 = meta("blk.0.attn_rel_proj.weight");
        const ggml_tensor * c0 = meta("blk.0.shortconv_attn.weight");
        GGML_ASSERT(r0 && p0 && c0 && "public Inkling 2 TQ1: the remapped tensors are missing (LLAMA_INK_PUBLIC=0?)");
        hparams.ink_d_rel           = (uint32_t) (r0->ne[1] / hparams.n_head(0));
        hparams.ink_rel_extent      = (uint32_t) p0->ne[0];
        hparams.n_shortconv_l_cache = (uint32_t) c0->ne[0];
        hparams.ink_log_n_floor     = 128000;
        hparams.ink_log_alpha       = 0.1f;
        hparams.ink_qk_clip         = 22.627416610717773f;   // sqrt(512)
        hparams.ink_rel_clip        = 8.0f;
        hparams.ink_res_block       = 6;
        hparams.ink_gate_softcap    = 4.0f;
        hparams.ink_up_softcap      = 25.0f;
        hparams.ink_n_vocab_real    = 200058;
    }
    ml.get_key(a + "d_rel",               hparams.ink_d_rel,           !pub);
    ml.get_key(a + "rel_extent",          hparams.ink_rel_extent,      !pub);
    ml.get_key(a + "shortconv_kernel",    hparams.n_shortconv_l_cache, !pub);
    ml.get_key(a + "log_scaling_n_floor", hparams.ink_log_n_floor,     !pub);
    ml.get_key(a + "log_scaling_alpha",   hparams.ink_log_alpha,       !pub);
    ml.get_key(a + "qk_l2_clip",          hparams.ink_qk_clip,         !pub);
    ml.get_key(a + "rel_l2_clip",         hparams.ink_rel_clip,        !pub);
    ml.get_key(a + "attn_res_block_size", hparams.ink_res_block,       !pub);
    ml.get_key(a + "mlp_gate_softcap",    hparams.ink_gate_softcap,    !pub);
    ml.get_key(a + "mlp_up_softcap",      hparams.ink_up_softcap,      !pub);
    ml.get_key(a + "unpadded_vocab_size", hparams.ink_n_vocab_real,    !pub);
    if (pub) {
        LLAMA_LOG_WARN("%s: public Inkling 2 TQ1: d_rel %u, rel_extent %u, conv %u (from the shapes), the other constants Inkling 2 Small's\n",
            __func__, hparams.ink_d_rel, hparams.ink_rel_extent, hparams.n_shortconv_l_cache);
    }

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;
    hparams.ink_n_embd_r = d_conv * (hparams.n_embd_k_gqa_max() + hparams.n_embd_v_gqa_max() + 2*hparams.n_embd);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t hd   = hparams.n_embd_head_k();
    const int64_t drel = hparams.ink_d_rel;
    const int64_t K    = hparams.n_shortconv_l_cache;
    const int64_t nff  = hparams.n_ff_exp();

    // GB10/ink2: B70's public TQ1 keeps the 201,024 padded rows but its tokenizer has 200,058 tokens: the tensors are
    // created at their stored size, the graph reads the first n_vocab rows of the output
    int64_t n_vocab_w = n_vocab;
    if (const ggml_tensor * m = this->ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str())) {
        if (m->ne[1] > n_vocab) {
            n_vocab_w = m->ne[1];
        }
    }
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_w}, 0);
    tok_norm    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"),    {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"),    {n_embd, n_vocab_w}, 0);
    res_out_q   = create_tensor(tn(LLM_TENSOR_ATTNRES_OUT_Q,   "weight"),    {n_embd}, 0);

    ink.resize(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        auto & il    = ink[i];
        const bool    full = !hparams.is_swa(i);
        const int64_t nkv  = hparams.n_head_kv(i);
        const int64_t kvw  = nkv*hd;

        if (i > 0) {
            if (full) {
                il.res_attn_q = create_tensor(tn(LLM_TENSOR_ATTNRES_ATTN_Q, "weight", i), {n_embd, 3}, 0);
            } else {
                il.res_attn_q = create_tensor(tn(LLM_TENSOR_ATTNRES_ATTN_Q, "weight", i), {n_embd}, 0);
            }
        }
        il.res_mlp_q = create_tensor(tn(LLM_TENSOR_ATTNRES_MLP_Q, "weight", i), {n_embd}, 0);

        if (full) {
            il.norm_q = create_tensor(tn(LLM_TENSOR_ATTN_Q_IN_NORM, "weight", i), {n_embd}, 0);
            il.norm_k = create_tensor(tn(LLM_TENSOR_ATTN_K_IN_NORM, "weight", i), {n_embd}, 0);
            il.norm_v = create_tensor(tn(LLM_TENSOR_ATTN_V_IN_NORM, "weight", i), {n_embd}, 0);
        } else {
            il.norm_q = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        }

        layer.wq  = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_head*hd}, 0);
        layer.wk  = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, kvw}, 0);
        layer.wv  = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, kvw}, 0);
        layer.wo  = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*hd, n_embd}, 0);
        il.wgate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, n_head*hd}, 0);
        il.wr     = create_tensor(tn(LLM_TENSOR_INK_ATTN_R,     "weight", i), {n_embd, n_head*drel}, 0);
        il.wsink  = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);
        il.q_clip   = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,      "weight", i), {hd, n_head}, 0);
        il.rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP,    "weight", i), {drel, n_head}, 0);
        il.rel_proj = create_tensor(tn(LLM_TENSOR_INK_ATTN_REL_PROJ,"weight", i), {(int64_t) hparams.ink_rel_extent, drel}, 0);

        if (full) {
            il.dc_k_s = create_tensor(tn(LLM_TENSOR_DYNCONV_K_STATIC, "weight", i), {kvw, K}, 0);
            il.dc_v_s = create_tensor(tn(LLM_TENSOR_DYNCONV_V_STATIC, "weight", i), {kvw, K}, 0);
            il.dc_k_p = create_tensor(tn(LLM_TENSOR_DYNCONV_K_PROJ,   "weight", i), {n_embd, nkv*K}, 0);
            il.dc_v_p = create_tensor(tn(LLM_TENSOR_DYNCONV_V_PROJ,   "weight", i), {n_embd, nkv*K}, 0);
        } else {
            il.sc_k = create_tensor(tn(LLM_TENSOR_INK_SHORTCONV_K, "weight", i), {K, kvw}, 0);
            il.sc_v = create_tensor(tn(LLM_TENSOR_INK_SHORTCONV_V, "weight", i), {K, kvw}, 0);
        }
        il.sc_attn = create_tensor(tn(LLM_TENSOR_INK_SHORTCONV_ATTN, "weight", i), {K, n_embd}, 0);
        il.sc_mlp  = create_tensor(tn(LLM_TENSOR_INK_SHORTCONV_MLP,  "weight", i), {K, n_embd}, 0);

        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);
        // pruned models may keep a different number of experts per layer: take it from the router shape
        int64_t n_exp_l = n_expert;
        if (const ggml_tensor * meta = this->ml->get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i).str().c_str())) {
            n_exp_l = meta->ne[1];
        }
        GGML_ASSERT(n_exp_l >= (int64_t) hparams.n_expert_used() && n_exp_l <= (int64_t) n_expert);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_exp_l}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_exp_l}, 0);
        // expert groups: the router covers n_exp_l experts, split over ffn_*_exps (group 0) and optional
        // ffn_*_exps_g1 .. _g7, each group holding one quant type and one FFN width (router rows ordered by group)
        constexpr int MG = std::remove_reference_t<decltype(il)>::MAX_GRP;
        static const llm_tensor GT[MG][3] = {
            {LLM_TENSOR_FFN_GATE_EXPS,     LLM_TENSOR_FFN_UP_EXPS,     LLM_TENSOR_FFN_DOWN_EXPS},
            {LLM_TENSOR_INK_GATE_EXPS_G1, LLM_TENSOR_INK_UP_EXPS_G1, LLM_TENSOR_INK_DOWN_EXPS_G1},
            {LLM_TENSOR_INK_GATE_EXPS_G2, LLM_TENSOR_INK_UP_EXPS_G2, LLM_TENSOR_INK_DOWN_EXPS_G2},
            {LLM_TENSOR_INK_GATE_EXPS_G3, LLM_TENSOR_INK_UP_EXPS_G3, LLM_TENSOR_INK_DOWN_EXPS_G3},
            {LLM_TENSOR_INK_GATE_EXPS_G4, LLM_TENSOR_INK_UP_EXPS_G4, LLM_TENSOR_INK_DOWN_EXPS_G4},
            {LLM_TENSOR_INK_GATE_EXPS_G5, LLM_TENSOR_INK_UP_EXPS_G5, LLM_TENSOR_INK_DOWN_EXPS_G5},
            {LLM_TENSOR_INK_GATE_EXPS_G6, LLM_TENSOR_INK_UP_EXPS_G6, LLM_TENSOR_INK_DOWN_EXPS_G6},
            {LLM_TENSOR_INK_GATE_EXPS_G7, LLM_TENSOR_INK_UP_EXPS_G7, LLM_TENSOR_INK_DOWN_EXPS_G7},
        };
        int64_t n_g[MG] = {n_exp_l};
        int64_t f_g[MG];   // per-group expert FFN width (neuron-pruned groups are narrower)
        for (int g = 0; g < MG; ++g) {
            f_g[g] = nff;
        }
        bool tq_l = false;   // TQ2T experts: strips [K*16, rows/16, E]
        if (const ggml_tensor * meta = this->ml->get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).str().c_str())) {
            tq_l = meta->type == GGML_TYPE_TQ2T;
            f_g[0] = tq_l ? meta->ne[1] * 16 : meta->ne[1];
        }
        int n_grp = 1;
        {
            int64_t rest = 0;
            for (int g = 1; g < MG; ++g) {
                const ggml_tensor * meta = this->ml->get_tensor_meta(tn(GT[g][0], "weight", i).str().c_str());
                if (!meta) {
                    break;   // groups are contiguous: g1, g2, ...
                }
                n_g[g] = meta->ne[2];
                f_g[g] = tq_l ? meta->ne[1] * 16 : meta->ne[1];
                rest += meta->ne[2];
                n_grp = g + 1;
            }
            n_g[0] = n_exp_l - rest;
            GGML_ASSERT(n_g[0] > 0 && "inkling2: expert groups larger than the router");
        }
        // TQ2T: [K*16, rows/16, E]
        const int64_t sk = tq_l ? 16 : 1;
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd*sk, f_g[0]/sk, n_g[0]}, 0);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd*sk, f_g[0]/sk, n_g[0]}, 0);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {f_g[0]*sk, n_embd/sk, n_g[0]}, 0);
        il.n_grp = n_grp;
        il.grp_n[0] = n_g[0];
        for (int g = 1; g < n_grp; ++g) {
            il.gexps[g-1][0] = create_tensor(tn(GT[g][0], "weight", i), {n_embd*sk, f_g[g]/sk, n_g[g]}, 0);
            il.gexps[g-1][1] = create_tensor(tn(GT[g][1], "weight", i), {n_embd*sk, f_g[g]/sk, n_g[g]}, 0);
            il.gexps[g-1][2] = create_tensor(tn(GT[g][2], "weight", i), {f_g[g]*sk, n_embd/sk, n_g[g]}, 0);
            il.grp_n[g] = n_g[g];
        }
        // RVQ2X8 experts: gains per group and matrix, one codebook per layer
        if (layer.ffn_gate_exps->type == GGML_TYPE_RVQ2X8) {
            il.rvq_book = create_tensor(tn(LLM_TENSOR_INK_EXPS_RVQ, "rvq_book", i), {8, 256, 2}, 0);
            for (int g = 0; g < n_grp; ++g) {
                il.rvq_gain[g][0] = create_tensor(tn(GT[g][0], "rvq_gain", i), {f_g[g], n_g[g]}, 0);
                il.rvq_gain[g][1] = create_tensor(tn(GT[g][1], "rvq_gain", i), {f_g[g], n_g[g]}, 0);
                il.rvq_gain[g][2] = create_tensor(tn(GT[g][2], "rvq_gain", i), {n_embd, n_g[g]}, 0);
                for (int m = 0; m < 3; ++m) {
                    const ggml_tensor * w = g == 0 ? (m == 0 ? layer.ffn_gate_exps : m == 1 ? layer.ffn_up_exps : layer.ffn_down_exps)
                                                   : il.gexps[g-1][m];
                    GGML_ASSERT(w->type == GGML_TYPE_RVQ2X8 && "inkling2: RVQ2X8 layers must not mix expert types");
                }
            }
        }
        // TQ2T experts: gains and input signs per group and matrix
        if (layer.ffn_gate_exps->type == GGML_TYPE_TQ2T) {
            il.tq = true;
            for (int g = 0; g < n_grp; ++g) {
                for (int m = 0; m < 3; ++m) {
                    const ggml_tensor * w = g == 0 ? (m == 0 ? layer.ffn_gate_exps : m == 1 ? layer.ffn_up_exps : layer.ffn_down_exps)
                                                   : il.gexps[g-1][m];
                    GGML_ASSERT(w->type == GGML_TYPE_TQ2T && "inkling2: TQ2T layers must not mix expert types");
                    il.tq_gain[g][m] = create_tensor(tn(GT[g][m], "tq_gain", i), {w->ne[1] * 16, n_g[g]}, 0);
                    il.tq_sign[g][m] = create_tensor(tn(GT[g][m], "tq_sign", i), {w->ne[0] / 16}, 0);
                }
            }
        }
        // per-expert NVFP4 global scales (*.scale) are loaded generically into layer.ffn_*_exps_s
    }
}

namespace {

class llm_graph_input_inkling2 : public llm_graph_input_i {
public:
    llm_graph_input_inkling2(const llama_hparams & hparams, const llama_memory_hybrid_iswa_context * mctx) :
        hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_inkling2() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (tau) {
            GGML_ASSERT(ggml_backend_buffer_is_host(tau->buffer));
            float * data = (float *) tau->data;
            for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
                const float eff = (float) (ubatch->pos[i] + 1) / (float) hparams.ink_log_n_floor;
                data[i] = 1.0f + hparams.ink_log_alpha*logf(std::max(eff, 1.0f));
            }
        }
        if (rel_idx) {
            mctx->get_attn()->get_base()->set_input_rel_idx(rel_idx, ubatch, hparams.ink_rel_extent);
        }
        if (rel_idx_swa) {
            mctx->get_attn()->get_swa()->set_input_rel_idx(rel_idx_swa, ubatch, hparams.ink_rel_extent);
        }
        if (vocab_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(vocab_mask->buffer));
            float * data = (float *) vocab_mask->data;
            for (int64_t id = 0; id < vocab_mask->ne[0]; ++id) {
                data[id] = id < (int64_t) hparams.ink_n_vocab_real ? 0.0f : -INFINITY;
            }
        }
    }

    // graph reuse: the inputs only depend on n_tokens and the two n_kv (the hybrid memory input checks the rest)
    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        const int64_t n_tokens = params.ubatch.n_tokens;
        bool res = true;
        if (tau) {
            res &= tau->ne[2] == n_tokens;
        }
        if (rel_idx) {
            res &= rel_idx->ne[0] == (int64_t) mctx->get_attn()->get_base()->get_n_kv() && rel_idx->ne[1] == n_tokens;
        }
        if (rel_idx_swa) {
            res &= rel_idx_swa->ne[0] == (int64_t) mctx->get_attn()->get_swa()->get_n_kv() && rel_idx_swa->ne[1] == n_tokens;
        }
        return res;
    }

    ggml_tensor * tau         = nullptr; // F32 [1, 1, n_tokens]
    ggml_tensor * rel_idx     = nullptr; // I32 [n_kv_base, n_tokens]
    ggml_tensor * rel_idx_swa = nullptr; // I32 [n_kv_swa,  n_tokens]
    ggml_tensor * vocab_mask  = nullptr; // F32 [n_vocab]

    const llama_hparams hparams;
    const llama_memory_hybrid_iswa_context * mctx;
};

} // namespace

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_inkling2::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_inkling2 &>(model_base);

    const int64_t hd     = hparams.n_embd_head_k();
    const int64_t drel   = hparams.ink_d_rel;
    const int64_t E      = hparams.ink_rel_extent;
    const int64_t K      = hparams.n_shortconv_l_cache;
    const int64_t d_conv = K - 1;
    const int64_t n_embd_r = hparams.n_embd_r();
    const int64_t kw_max = hparams.n_embd_k_gqa_max();
    const int64_t vw_max = hparams.n_embd_v_gqa_max();
    const float   eps    = hparams.f_norm_rms_eps;

    const int64_t off_k    = 0;
    const int64_t off_v    = d_conv*kw_max;
    const int64_t off_attn = d_conv*(kw_max + vw_max);
    const int64_t off_mlp  = d_conv*(kw_max + vw_max + n_embd);

    const auto * mctx_hyb  = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    const auto * mctx_recr = mctx_hyb->get_recr();
    const auto * mctx_attn = mctx_hyb->get_attn();
    const uint32_t kv_head = mctx_recr->get_head();

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens*n_seqs);

    bool has_full = false, has_swa = false;
    for (int il = 0; il < n_layer; ++il) {
        (hparams.is_swa(il) ? has_swa : has_full) = true;
    }
    auto inp = std::make_unique<llm_graph_input_inkling2>(hparams, mctx_hyb);
    // only create inputs the graph uses (an unused input gets no buffer)
    if (has_full) {
        inp->tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(inp->tau);
        inp->rel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_base()->get_n_kv(), n_tokens);
        ggml_set_input(inp->rel_idx);
    }
    if (has_swa) {
        inp->rel_idx_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_swa()->get_n_kv(), n_tokens);
        ggml_set_input(inp->rel_idx_swa);
    }
    const int64_t n_vocab = model.vocab.n_tokens();
    if (!cparams.embeddings && hparams.ink_n_vocab_real > 0 && (int64_t) hparams.ink_n_vocab_real < n_vocab) {
        inp->vocab_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_vocab);
        ggml_set_input(inp->vocab_mask);
    }
    ggml_tensor * tau         = inp->tau;
    ggml_tensor * rel_idx     = inp->rel_idx;
    ggml_tensor * rel_idx_swa = inp->rel_idx_swa;
    ggml_tensor * vocab_mask  = inp->vocab_mask;
    res->add_input(std::move(inp));

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();
    auto * inp_attn   = inp_hybrid->get_attn();

    // LLAMA_INK_INP_CONT (default 1): with several sequences in a ubatch (per-slot KV), every attention layer reads one
    // view per stream of the CPU inputs rel_idx and the KQ masks, and the scheduler copies each view to the GPU on its
    // own, all alive from the start of the graph (np 3, ubatch 2048: 170.8 MiB copies of rel_idx views at the 9.2 GiB
    // peak, 11/21 16:03). One CONT of each input makes a single device copy that every layer views; same values.
    static const bool inp_cont = [] { const char * v = getenv("LLAMA_INK_INP_CONT"); return !v || atoi(v) != 0; }();
    ggml_tensor * kq_mask_dev = nullptr, * kq_mask_swa_dev = nullptr;
    if (inp_cont && n_seqs > 1) {
        if (rel_idx)     { rel_idx     = ggml_cont(ctx0, rel_idx);     cb(rel_idx,     "rel_idx_dev",     -1); }
        if (rel_idx_swa) { rel_idx_swa = ggml_cont(ctx0, rel_idx_swa); cb(rel_idx_swa, "rel_idx_swa_dev", -1); }
        if (has_full) { kq_mask_dev     = ggml_cont(ctx0, inp_attn->get_kq_mask());     cb(kq_mask_dev,     "kq_mask_dev",     -1); }
        if (has_swa)  { kq_mask_swa_dev = ggml_cont(ctx0, inp_attn->get_kq_mask_swa()); cb(kq_mask_swa_dev, "kq_mask_swa_dev", -1); }
    }

    ggml_tensor * conv_rs = nullptr; // this layer's gathered packed state {n_embd_r, n_seqs}

    // causal window over [state | x] for one stream; saves the last d_conv inputs back to the cache.
    // returns sx {d_conv + n_seq_tokens, w, n_seqs} (time-major)
    auto build_window = [&](ggml_tensor * x2d, int64_t off, int il) -> ggml_tensor * {
        const int64_t w  = x2d->ne[0];
        const size_t  sz = ggml_element_size(conv_rs);
        ggml_tensor * x3 = ggml_reshape_3d(ctx0, x2d, w, n_seq_tokens, n_seqs);
        ggml_tensor * xt = ggml_transpose(ctx0, x3);
        // several tokens: a contiguous copy first, so the concat below takes the contiguous kernel (the strided one
        // took ~16 us a layer for the 4608-wide streams at 4 tokens). env LLAMA_INK_XT_CONT=0: off
        static const bool xt_cont = [] { const char * v = getenv("LLAMA_INK_XT_CONT"); return !v || atoi(v) != 0; }();
        if (xt_cont && n_seq_tokens > 1) {
            xt = ggml_cont(ctx0, xt);
        }
        ggml_tensor * state = ggml_view_3d(ctx0, conv_rs, d_conv, w, n_seqs, d_conv*sz, conv_rs->nb[1], off*sz);
        ggml_tensor * sx = ggml_concat(ctx0, state, xt, 0);
        ggml_tensor * conv_state = mctx_recr->get_r_l(il);
        // slot 0 = the final state, slot k = the state k tokens back (partial rollback for speculative decoding)
        const int64_t n_snap    = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, n_snap);
        const int64_t mem_size  = mctx_recr->get_size();
        for (int64_t slot = 0; slot < n_written; ++slot) {
            ggml_tensor * new_state = ggml_view_3d(ctx0, sx, d_conv, w, n_seqs, sx->nb[1], sx->nb[2], (sx->ne[0] - d_conv - slot)*sx->nb[0]);
            ggml_tensor * dst = ggml_view_3d(ctx0, conv_state, d_conv, w, n_seqs, d_conv*sz, n_embd_r*sz,
                                             ((slot*mem_size + kv_head)*n_embd_r + off)*sz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, dst));
        }
        return sx;
    };

    // static short conv: (conv(x) + x) / sqrt(K+1)
    // res: added to the result (res + conv), inside the conv kernel on the prefill path
    auto build_sconv = [&](ggml_tensor * x2d, ggml_tensor * kernel, int64_t off, int il, ggml_tensor * res = nullptr) -> ggml_tensor * {
        const int64_t w = x2d->ne[0];
        // prefill: the conv reads the tokens and the cached state where they are (GB10 CUDA, GGML_CUDA_INK_SCONV_MAGIC on a
        // SCALE node, srcs x, state, taps) instead of the transposed [state | x] window; the state snapshots come straight
        // from the last tokens. env LLAMA_INK_SCONV_X=0: the window path below
        static const bool sconv_x = [] { const char * v = getenv("LLAMA_INK_SCONV_X"); return !v || atoi(v) != 0; }();
        const int64_t n_snap = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        if (sconv_x && n_seq_tokens >= 64 && n_seq_tokens >= d_conv + n_snap && kernel->ne[0] == 4 && d_conv == 3 &&
            ggml_is_contiguous(x2d) && x2d->type == GGML_TYPE_F32 && conv_rs->type == GGML_TYPE_F32 && kernel->type == GGML_TYPE_F32) {
            const size_t  sz = ggml_element_size(conv_rs);
            ggml_tensor * x3 = ggml_reshape_3d(ctx0, x2d, w, n_seq_tokens, n_seqs);
            ggml_tensor * state = ggml_view_3d(ctx0, conv_rs, d_conv, w, n_seqs, d_conv*sz, conv_rs->nb[1], off*sz);
            ggml_tensor * y = ggml_scale(ctx0, x2d, 1.0f/sqrtf((float) (K + 1)));
            y->op_params[8]  = 0x494e4b53;   // GGML_CUDA_INK_SCONV_MAGIC
            y->op_params[9]  = (int32_t) d_conv;
            y->op_params[10] = (int32_t) n_seq_tokens;
            y->src[1] = state;
            y->src[2] = kernel;
            static const bool res_in = [] { const char * v = getenv("LLAMA_INK_SCONV_RES"); return !v || atoi(v) != 0; }();
            if (res && res_in && ggml_is_contiguous(res) && ggml_are_same_shape(res, x2d) && res->type == GGML_TYPE_F32) {
                y->src[3] = res;   // y = res + conv (the same sum as the ADD after it)
                res = nullptr;
            }
            ggml_build_forward_expand(gf, y);   // before the snapshots below overwrite the cache rows
            ggml_tensor * conv_state = mctx_recr->get_r_l(il);
            const int64_t mem_size = mctx_recr->get_size();
            for (int64_t slot = 0; slot < n_snap; ++slot) {
                // slot k = the last d_conv inputs k tokens back = tokens [n - d_conv - k, n - k)
                ggml_tensor * src = ggml_view_3d(ctx0, x3, w, d_conv, n_seqs, x3->nb[1], x3->nb[2], (n_seq_tokens - d_conv - slot)*x3->nb[1]);
                src = ggml_transpose(ctx0, src);   // {d_conv, w, n_seqs}
                ggml_tensor * dst = ggml_view_3d(ctx0, conv_state, d_conv, w, n_seqs, d_conv*sz, n_embd_r*sz,
                                                 ((slot*mem_size + kv_head)*n_embd_r + off)*sz);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
            }
            return res ? ggml_add(ctx0, res, y) : y;
        }
        ggml_tensor * sx = build_window(x2d, off, il);
        ggml_tensor * y  = ggml_ssm_conv(ctx0, sx, kernel); // {w, n_seq_tokens, n_seqs}
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, x2d, w, n_seq_tokens, n_seqs));
        y = ggml_scale(ctx0, y, 1.0f/sqrtf((float) (K + 1)));
        y = ggml_reshape_2d(ctx0, y, w, n_tokens);
        return res ? ggml_add(ctx0, res, y) : y;
    };

    // dynamic short conv (full layers k/v): taps = static[lag] + proj(dyn_in)[head, lag]; (sum + x) / sqrt(2K+1)
    auto build_dconv = [&](ggml_tensor * x2d, ggml_tensor * dyn_in, ggml_tensor * st, ggml_tensor * pw,
                           int64_t nkv, int64_t off, int il) -> ggml_tensor * {
        const int64_t w  = x2d->ne[0];
        const int64_t hh = w/nkv;
        ggml_tensor * sx = build_window(x2d, off, il);
        ggml_tensor * dw = build_lora_mm(pw, dyn_in);                    // {nkv*K, n_tokens}, index h*K + lag
        dw = ggml_reshape_3d(ctx0, dw, K, nkv, n_tokens);
        // GB10: the whole conv in one CUDA kernel (ggml-cuda/l2clip.cuh GGML_CUDA_INK_DCONV_MAGIC on a SCALE node, srcs x,
        // window, static taps, dynamic taps); env LLAMA_INK_DCONV_FUSE=0: the per-lag graph below (~22 kernels)
        static const bool dconv_fuse = [] { const char * v = getenv("LLAMA_INK_DCONV_FUSE"); return v && atoi(v) != 0; }();   // default off: no measured gain, ub4 ppl +0.24% (11/19 11:36)
        if (dconv_fuse && ggml_is_contiguous(x2d) && ggml_is_contiguous(dw)) {
            ggml_tensor * o = ggml_scale(ctx0, x2d, 1.0f/sqrtf((float) (2*K + 1)));
            o->op_params[8]  = 0x494e4b44;   // GGML_CUDA_INK_DCONV_MAGIC
            o->op_params[9]  = (int32_t) K;
            o->op_params[10] = (int32_t) d_conv;
            o->op_params[11] = (int32_t) nkv;
            o->op_params[12] = (int32_t) n_seq_tokens;
            o->src[1] = sx;
            o->src[2] = st;
            o->src[3] = dw;
            return o;
        }
        ggml_tensor * acc = ggml_reshape_3d(ctx0, x2d, hh, nkv, n_tokens);
        for (int64_t lag = 0; lag < K; ++lag) {
            // token t - lag lives at sx[t + d_conv - lag]
            ggml_tensor * xl = ggml_view_3d(ctx0, sx, n_seq_tokens, w, n_seqs, sx->nb[1], sx->nb[2], (d_conv - lag)*sx->nb[0]);
            xl = ggml_cont(ctx0, ggml_transpose(ctx0, xl));             // {w, n_seq_tokens, n_seqs}
            xl = ggml_reshape_3d(ctx0, xl, hh, nkv, n_tokens);
            ggml_tensor * ws = ggml_view_1d(ctx0, st, w, lag*st->nb[1]);
            ws = ggml_reshape_3d(ctx0, ws, hh, nkv, 1);
            ggml_tensor * wd = ggml_view_3d(ctx0, dw, 1, nkv, n_tokens, dw->nb[1], dw->nb[2], lag*dw->nb[0]);
            wd = ggml_cont(ctx0, wd);
            ggml_tensor * t = ggml_add(ctx0, ggml_mul(ctx0, xl, ws), ggml_mul(ctx0, xl, wd));
            acc = ggml_add(ctx0, acc, t);
        }
        acc = ggml_scale(ctx0, acc, 1.0f/sqrtf((float) (2*K + 1)));
        return ggml_reshape_2d(ctx0, acc, w, n_tokens);
    };

    // x * min(1, thr / (||x|| + 1e-6)) over dim 0
    auto l2clip = [&](ggml_tensor * x, float thr) -> ggml_tensor * {
        ggml_tensor * n = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
        n = ggml_scale_bias(ctx0, n, 1.0f/thr, 1e-6f/thr);
        n = ggml_clamp(ctx0, n, 1.0f, FLT_MAX);
        return ggml_div(ctx0, x, n);
    };

    // block attention residual: states (already rms-normed, unweighted) {n_embd, n_tokens} each; query {n_embd, nq}
    // Sb: the completed block states already stacked {n_embd, Jb, n_tokens} (or nullptr); normed: the states after them.
    // Keeping Sb stacked across layers saves re-concatenating every completed block in every layer.
    auto attn_res = [&](ggml_tensor * Sb, const std::vector<ggml_tensor *> & normed, ggml_tensor * query, int64_t nq) -> std::vector<ggml_tensor *> {
        const int64_t J = normed.size();
        ggml_tensor * S = Sb;
        for (auto * s : normed) {
            ggml_tensor * s3 = ggml_reshape_3d(ctx0, s, n_embd, 1, n_tokens);
            S = S ? ggml_concat(ctx0, S, s3, 1) : s3;                   // {n_embd, J, n_tokens}
        }
        ggml_tensor * q2 = ggml_reshape_2d(ctx0, query, n_embd, nq);
        ggml_tensor * sc = ggml_mul_mat(ctx0, q2, S);                   // {nq, J, n_tokens}
        ggml_prec_set_acc(sc, GGML_PREC_F32);
        sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));       // {J, nq, n_tokens}
        sc = ggml_soft_max_ext(ctx0, sc, nullptr, 1.0f/sqrtf((float) n_embd), 0.0f);
        // schedule sc first so the CONT below is right before the MUL_MAT that reads it: the CUDA backend fuses the
        // pair (ggml-cuda/l2clip.cu, ggml_cuda_resw_match) into one pass over S, without the transposed copy
        ggml_build_forward_expand(gf, sc);
        ggml_tensor * St = ggml_cont(ctx0, ggml_permute(ctx0, S, 1, 0, 2, 3)); // {J, n_embd, n_tokens}
        ggml_tensor * o  = ggml_mul_mat(ctx0, St, sc);                  // {n_embd, nq, n_tokens}
        ggml_prec_set_acc(o, GGML_PREC_F32);
        std::vector<ggml_tensor *> out;
        // nq == 1: o is already {n_embd, n_tokens} in memory, a reshape instead of a copy (env LLAMA_INK_RES_NOCOPY, default 1)
        static const bool nocopy = [] { const char * v = getenv("LLAMA_INK_RES_NOCOPY"); return !v || atoi(v) != 0; }();
        if (nq == 1 && nocopy && n_tokens >= 64 && ggml_is_contiguous(o)) {   // prefill only: decode was 0.4% slower without the copy
            out.push_back(ggml_reshape_2d(ctx0, o, n_embd, n_tokens));
            return out;
        }
        for (int64_t i = 0; i < nq; ++i) {
            out.push_back(ggml_cont(ctx0, ggml_view_2d(ctx0, o, n_embd, n_tokens, o->nb[2], i*o->nb[1])));
        }
        GGML_UNUSED(J);
        return out;
    };

    auto unorm = [&](ggml_tensor * x) { return ggml_rms_norm(ctx0, x, 1e-6f); };

    auto build_attention = [&](ggml_tensor * xq, ggml_tensor * xk, ggml_tensor * xv, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        const auto & L     = model.ink[il];
        const bool   full  = !hparams.is_swa(il);
        const int64_t nkv  = hparams.n_head_kv(il);
        const int64_t nh   = hparams.n_head(il);

        ggml_tensor * q = build_lora_mm(layer.wq, xq);
        ggml_tensor * k = build_lora_mm(layer.wk, xk);
        ggml_tensor * v = build_lora_mm(layer.wv, xv);
        if (full) {
            k = build_dconv(k, xk, L.dc_k_s, L.dc_k_p, nkv, off_k, il);
            v = build_dconv(v, xv, L.dc_v_s, L.dc_v_p, nkv, off_v, il);
        } else {
            k = build_sconv(k, L.sc_k, off_k, il);
            v = build_sconv(v, L.sc_v, off_v, il);
        }
        ggml_tensor * r     = build_lora_mm(L.wr, xq);
        ggml_tensor * gate  = build_lora_mm(L.wgate, xq);
        ggml_tensor * sinks = build_lora_mm(L.wsink, xq);              // {nh, n_tokens}

        q = ggml_reshape_3d(ctx0, q, hd, nh, n_tokens);
        q = l2clip(ggml_mul(ctx0, q, L.q_clip), hparams.ink_qk_clip);
        k = l2clip(ggml_reshape_3d(ctx0, k, hd, nkv, n_tokens), hparams.ink_qk_clip);
        v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nkv, n_tokens), eps);
        r = ggml_reshape_3d(ctx0, r, drel, nh, n_tokens);
        r = l2clip(ggml_mul(ctx0, r, L.rel_clip), hparams.ink_rel_clip);
        if (full) {
            q = ggml_mul(ctx0, q, tau);
            r = ggml_mul(ctx0, r, tau);
        }
        cb(q, "ink_q", il); cb(k, "ink_k", il); cb(v, "ink_v", il); cb(r, "ink_r", il);

        // GB10: one CUDA kernel for bias gather + scale + mask + sink + softmax (op_params[10] magic on a SOFT_MAX node,
        // see ggml-cuda/l2clip.cu); env LLAMA_INK_FUSED_SM=0: the plain graph below
        static const bool fused_sm = [] { const char * v = getenv("LLAMA_INK_FUSED_SM"); return !v || atoi(v) != 0; }();

        // relative logits {E, nh, n_tokens} -> gathered bias {n_kv, n_tokens, nh}
        ggml_tensor * projT = ggml_cont(ctx0, ggml_transpose(ctx0, L.rel_proj)); // {drel, E}
        ggml_tensor * rel = ggml_mul_mat(ctx0, projT, ggml_reshape_2d(ctx0, r, drel, nh*n_tokens));
        ggml_prec_set_acc(rel, GGML_PREC_F32);
        // the fused softmax reads this raw {E, nh*n_tokens} table itself (index E: the zero column); the plain graph
        // pads and permutes it to {nh, (E+1)*n_tokens} rows
        static const bool rel_raw = [] { const char * v = getenv("LLAMA_INK_REL_RAW"); return !v || atoi(v) != 0; }();
        ggml_tensor * rel_flat = rel;
        if (!(fused_sm && rel_raw)) {
            rel = ggml_reshape_3d(ctx0, rel, E, nh, n_tokens);
            rel = ggml_pad(ctx0, rel, 1, 0, 0, 0);                          // zero column at index E
            rel = ggml_cont(ctx0, ggml_permute(ctx0, rel, 1, 0, 2, 3));     // {nh, E+1, n_tokens}
            rel = ggml_reshape_2d(ctx0, rel, nh, (E + 1)*n_tokens);
        }
        ggml_tensor * idx = full ? rel_idx : rel_idx_swa;
        const int64_t n_kv = idx->ne[0];

        // store k/v, then plain softmax attention with a per-token sink column
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, k);
        ggml_build_forward_expand(gf, v);
        const bool is_swa = !full;
        GGML_ASSERT((is_swa ? inp_attn->self_k_rot_swa[il] : inp_attn->self_k_rot[il]) == nullptr);
        const auto * mctx_cur = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k, is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v, is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs(), il));
        ggml_tensor * kc = mctx_cur->get_k(ctx0, il);
        ggml_tensor * vc = mctx_cur->get_v(ctx0, il);
        GGML_ASSERT(kc->ne[2] == n_kv);
        const bool v_trans = vc->nb[1] > vc->nb[2];
        // per-slot KV (no --kv-unified): kc/vc carry one stream per sequence of the ubatch (ne[3]), the ubatch tokens
        // are stream-major (n_tps per stream); each stream's queries attend to its own cells only
        const int64_t ns    = kc->ne[3];
        GGML_ASSERT(n_tokens % ns == 0);
        const int64_t n_tps = n_tokens/ns;

        ggml_tensor * qp = ggml_permute(ctx0, q, 0, 2, 1, 3);           // {hd, n_tokens, nh}
        auto kv_stream = [&](int64_t s, ggml_tensor *& kp_s, ggml_tensor *& vp_s) {
            ggml_tensor * kcs = ns == 1 ? kc : ggml_view_3d(ctx0, kc, kc->ne[0], kc->ne[1], kc->ne[2], kc->nb[1], kc->nb[2], s*kc->nb[3]);
            ggml_tensor * vcs = ns == 1 ? vc : ggml_view_3d(ctx0, vc, vc->ne[0], vc->ne[1], vc->ne[2], vc->nb[1], vc->nb[2], s*vc->nb[3]);
            kp_s = ggml_permute(ctx0, kcs, 0, 2, 1, 3);                 // {hd, n_kv, nkv}
            vp_s = ggml_permute(ctx0, vcs, 0, 2, 1, 3);
            if (!v_trans) {
                vp_s = ggml_cont(ctx0, ggml_transpose(ctx0, vp_s));
            }
        };
        ggml_tensor * mask_all = is_swa ? (kq_mask_swa_dev ? kq_mask_swa_dev : inp_attn->get_kq_mask_swa())
                                        : (kq_mask_dev     ? kq_mask_dev     : inp_attn->get_kq_mask());

        // GB10/ink2: the attention is materialised ({n_kv, n_tokens, nh} f32 logits, bias, softmax copies), so the
        // compute buffer grows with n_kv * n_ubatch. Process the queries in chunks of LLAMA_INK_QCHUNK (default 64,
        // 0 = off): same math per query, the graph allocator reuses one chunk's intermediates for the next.
        static const int64_t q_chunk = [] { const char * v = getenv("LLAMA_INK_QCHUNK"); return v ? (int64_t) atoll(v) : (int64_t) 64; }();
        const int64_t qc = (q_chunk > 0 && n_tps > q_chunk) ? q_chunk : n_tps;


        auto attn_chunk = [&](int64_t t0, int64_t nt, ggml_tensor * kp, ggml_tensor * vp) -> ggml_tensor * {
            ggml_tensor * qp_c = nt == n_tokens ? qp : ggml_view_3d(ctx0, qp, hd, nt, nh, qp->nb[1], qp->nb[2], t0*qp->nb[1]);   // {hd, nt, nh}
            ggml_tensor * idx_c = nt == n_tokens ? ggml_reshape_1d(ctx0, idx, n_kv*n_tokens) : ggml_view_1d(ctx0, idx, n_kv*nt, t0*idx->nb[1]); // rows of the full rel table
            if (fused_sm) {
                ggml_tensor * kq = ggml_mul_mat(ctx0, kp, qp_c);                // {n_kv, nt, nh}
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                ggml_tensor * sk = nt == n_tokens ? sinks : ggml_view_2d(ctx0, sinks, nh, nt, sinks->nb[1], t0*sinks->nb[1]);
                ggml_tensor * mask = nt == n_tokens ? mask_all : ggml_view_2d(ctx0, mask_all, mask_all->ne[0], nt, mask_all->nb[1], t0*mask_all->nb[1]);
                if (mask->ne[0] != n_kv) {
                    mask = ggml_view_2d(ctx0, mask, n_kv, nt, mask->nb[1], 0);
                }
                ggml_tensor * p = ggml_soft_max_ext(ctx0, kq, mask, 1.0f/sqrtf((float) hd), 0.0f);
                p->op_params[10] = 0x494e4b53;   // GGML_CUDA_INK_SM_MAGIC
                p->op_params[11] = rel_raw ? (int32_t) E : 0;   // > 0: src[3] is the raw {E, nh*n_tokens} table
                p->src[3] = rel_raw ? rel_flat : rel;
                p->src[4] = idx_c;
                p->src[5] = sk;
                ggml_tensor * kqv = ggml_mul_mat(ctx0, vp, p);                  // {hd, nt, nh}
                return ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), hd*nh, nt);
            }
            ggml_tensor * kqb_c = ggml_get_rows(ctx0, rel, idx_c);                                        // {nh, n_kv*nt}
            kqb_c = ggml_reshape_3d(ctx0, kqb_c, nh, n_kv, nt);
            kqb_c = ggml_cont(ctx0, ggml_permute(ctx0, kqb_c, 2, 0, 1, 3));                              // {n_kv, nt, nh}

            ggml_tensor * kq = ggml_mul_mat(ctx0, kp, qp_c);                // {n_kv, nt, nh}
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_scale(ctx0, kq, 1.0f/sqrtf((float) hd));
            kq = ggml_add(ctx0, kq, kqb_c);
            ggml_tensor * sk = nt == n_tokens ? sinks : ggml_view_2d(ctx0, sinks, nh, nt, sinks->nb[1], t0*sinks->nb[1]);
            sk = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, sk, 1, nh, nt), 0, 2, 1, 3)); // {1, nt, nh}
            kq = ggml_concat(ctx0, kq, sk, 0);                              // {n_kv+1, nt, nh}
            ggml_tensor * mask = nt == n_tokens ? mask_all : ggml_view_2d(ctx0, mask_all, mask_all->ne[0], nt, mask_all->nb[1], t0*mask_all->nb[1]);
            if (mask->type != GGML_TYPE_F32) {
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F32);
            }
            mask = ggml_pad(ctx0, mask, 1, 0, 0, 0);
            kq = ggml_soft_max_ext(ctx0, kq, mask, 1.0f, 0.0f);
            ggml_tensor * p = ggml_cont(ctx0, ggml_view_3d(ctx0, kq, n_kv, nt, nh, kq->nb[1], kq->nb[2], 0));
            ggml_tensor * kqv = ggml_mul_mat(ctx0, vp, p);                  // {hd, nt, nh}
            return ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), hd*nh, nt);
        };

        // the chunks' outputs joined pairwise (a tree: each row is copied log2(chunks) times, not up to chunks times as
        // with a running concat); env LLAMA_INK_OTREE=1 (default 0, the running concat: +0.3% only, 11/19)
        static const bool o_tree = [] { const char * v = getenv("LLAMA_INK_OTREE"); return v && atoi(v) != 0; }();   // default off: no measured gain (11/19)
        ggml_tensor * o = nullptr;
        // env LLAMA_INK_FA (default 1, 0 = off; patch from the second GB10, 11/19 09:10): each stream's queries in one CUDA
        // flash-attention MMA call that adds the relative-position bias (from the raw table via idx) and the per-token
        // sink itself (ggml-cuda/fattn-common.cuh GGML_CUDA_INK_FA_MAGIC): no {n_kv, n_tokens, nh} logits, no V copy
        static const bool ink_fa = [] { const char * v = getenv("LLAMA_INK_FA"); return !v || atoi(v) != 0; }();   // default on since 11/19 10:40 (0 = off)
        // the CUDA Inkling FA kernel reads F16 K/V only (fattn.cu); a quantized cache (-ctk / -ctv q8_0) takes the softmax path
        const bool use_fa = ink_fa && fused_sm && rel_raw && !v_trans && mask_all->type == GGML_TYPE_F16 && mask_all->ne[0] == n_kv &&
                            kc->type == GGML_TYPE_F16 && vc->type == GGML_TYPE_F16;
        {
            // env LLAMA_INK_FA_DEBUG=1: print which attention path each graph takes (first 4 graphs, layers 0 and 3)
            static const bool fa_dbg = [] { const char * v = getenv("LLAMA_INK_FA_DEBUG"); return v && atoi(v) != 0; }();
            static int fa_dbg_n = 0;
            if (fa_dbg && (il == 0 || il == 3) && fa_dbg_n < 8) {
                fa_dbg_n++;
                fprintf(stderr, "ink-fa: il=%d swa=%d n_tokens=%lld ns=%lld n_kv=%lld use_fa=%d ink_fa=%d fused_sm=%d rel_raw=%d v_trans=%d mask_type=%d mask_ne0=%lld k=%d v=%d\n",
                        il, (int) is_swa, (long long) n_tokens, (long long) ns, (long long) n_kv, (int) use_fa, (int) ink_fa, (int) fused_sm,
                        (int) rel_raw, (int) v_trans, (int) mask_all->type, (long long) mask_all->ne[0], (int) kc->type, (int) vc->type);
            }
        }
        std::vector<ggml_tensor *> parts;
        for (int64_t s = 0; s < ns; ++s) {
            if (use_fa) {
                ggml_tensor * kcs = ns == 1 ? kc : ggml_view_3d(ctx0, kc, kc->ne[0], kc->ne[1], kc->ne[2], kc->nb[1], kc->nb[2], s*kc->nb[3]);
                ggml_tensor * vcs = ns == 1 ? vc : ggml_view_3d(ctx0, vc, vc->ne[0], vc->ne[1], vc->ne[2], vc->nb[1], vc->nb[2], s*vc->nb[3]);
                const int64_t t0 = s*n_tps;
                ggml_tensor * qs  = ns == 1 ? qp : ggml_view_3d(ctx0, qp, hd, n_tps, nh, qp->nb[1], qp->nb[2], t0*qp->nb[1]);
                ggml_tensor * ms  = ns == 1 ? mask_all : ggml_view_2d(ctx0, mask_all, n_kv, n_tps, mask_all->nb[1], t0*mask_all->nb[1]);
                ggml_tensor * sks = ns == 1 ? sinks : ggml_view_2d(ctx0, sinks, nh, n_tps, sinks->nb[1], t0*sinks->nb[1]);
                // env LLAMA_INK_FA_RELDOT (default 1, 0 = off): the kernel computes the bias from r and rel_proj itself (16 FMAs a
                // logit), so the {E, nh*n_tokens} table (a TF32 GEMM, 400 MB a layer at ubatch 2048) is not built or read
                static const bool reldot = [] { const char * v = getenv("LLAMA_INK_FA_RELDOT"); return !v || atoi(v) != 0; }();
                ggml_tensor * rr  = nullptr;
                if (reldot && drel == 16) {
                    ggml_tensor * r2 = ggml_reshape_2d(ctx0, ggml_is_contiguous(r) ? r : ggml_cont(ctx0, r), drel, nh*n_tokens);
                    rr = ns == 1 ? r2 : ggml_view_2d(ctx0, r2, drel, nh*n_tps, r2->nb[1], t0*nh*r2->nb[1]);
                }
                ggml_tensor * rls = rr ? rr : ns == 1 ? rel_flat : ggml_view_2d(ctx0, rel_flat, E, nh*n_tps, rel_flat->nb[1], t0*nh*rel_flat->nb[1]);
                ggml_tensor * ixs = ns == 1 ? idx : ggml_view_2d(ctx0, idx, n_kv, n_tps, idx->nb[1], t0*idx->nb[1]);
                ggml_tensor * fa  = ggml_flash_attn_ext(ctx0, qs, ggml_permute(ctx0, kcs, 0, 2, 1, 3), ggml_permute(ctx0, vcs, 0, 2, 1, 3),
                                                        ms, 1.0f/sqrtf((float) hd), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                fa->op_params[10] = 0x494e4b46;   // GGML_CUDA_INK_FA_MAGIC
                fa->op_params[11] = (int32_t) E;
                fa->op_params[12] = (int32_t) t0;
                fa->src[4] = sks;   // {nh, n_tps}
                fa->src[5] = rls;   // {E, nh*n_tps}
                fa->src[6] = ixs;   // {n_kv, n_tps}, values t_global*(E+1) + d
                if (rr) {
                    // env LLAMA_INK_FA_P16 (default 1, 0 = F32): rel_proj^T as F16 (32 KB instead of 64 KB read per logit tile)
                    static const bool p16 = [] { const char * v = getenv("LLAMA_INK_FA_P16"); return !v || atoi(v) != 0; }();   // default 1 since 11/20 16:45 (pp512 +1.0%, PPL same)
                    fa->op_params[13] = (int32_t) drel;
                    // prefill only (n_tokens >= 64): in decode the cast costs more than the halved reads (tg64 -0.5%, 11/20 18:3x)
                    fa->src[7] = p16 && n_tokens >= 64 ? ggml_cast(ctx0, projT, GGML_TYPE_F16) : projT;   // {drel, E}
                }
                parts.push_back(ggml_reshape_2d(ctx0, fa, hd*nh, n_tps));   // fa: {hd, nh, n_tps}
                continue;
            }
            ggml_tensor * kp_s = nullptr, * vp_s = nullptr;
            kv_stream(s, kp_s, vp_s);
            const int64_t t1 = (s + 1)*n_tps;
            for (int64_t t0 = s*n_tps; t0 < t1; t0 += qc) {
                parts.push_back(attn_chunk(t0, std::min(qc, t1 - t0), kp_s, vp_s));
            }
        }
        if (o_tree) {
            while (parts.size() > 1) {
                std::vector<ggml_tensor *> next;
                for (size_t k = 0; k + 1 < parts.size(); k += 2) {
                    next.push_back(ggml_concat(ctx0, parts[k], parts[k + 1], 1));
                }
                if (parts.size() % 2) {
                    next.push_back(parts.back());
                }
                parts.swap(next);
            }
            o = parts[0];
        } else {
            for (auto * oc : parts) {
                o = o ? ggml_concat(ctx0, o, oc, 1) : oc;
            }
        }
        o = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, gate));
        cb(o, "ink_attn_gated", il);
        return build_lora_mm(layer.wo, o);
    };

    auto build_moe = [&](ggml_tensor * x, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        const auto & L     = model.ink[il];
        const int64_t n_used = hparams.n_expert_used();

        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, x);    // {n_expert, n_tokens}
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        ggml_tensor * scores = ggml_sigmoid(ctx0, logits);
        // LLAMA_INK_TOPKMOE (default 1): the routing in the node order of llm_graph_context::build_moe_ffn (sigmoid,
        // reshape, bias add, argsort top-k, get_rows, sum, clamp, div, scale) so the CUDA backend runs it as one
        // topk-moe kernel (8 small kernels a layer before). The sum guard is clamp(6.1e-5) instead of + 1e-20: the sum
        // of the 8 picked sigmoids is far above both.
        static const bool topkmoe = [] { const char * v = getenv("LLAMA_INK_TOPKMOE"); return !v || atoi(v) != 0; }();
        ggml_tensor * sel;
        ggml_tensor * w;
        if (topkmoe) {
            ggml_tensor * probs3 = ggml_reshape_3d(ctx0, scores, 1, layer.ffn_gate_inp->ne[1], n_tokens);
            sel = ggml_argsort_top_k(ctx0, ggml_add(ctx0, scores, layer.ffn_exp_probs_b), n_used);          // {n_used, n_tokens}
            w = ggml_get_rows(ctx0, probs3, sel);                                                            // {1, n_used, n_tokens}
            w = ggml_reshape_2d(ctx0, w, n_used, n_tokens);
            w = ggml_div(ctx0, w, ggml_clamp(ctx0, ggml_sum_rows(ctx0, w), 6.103515625e-5f, INFINITY));
            w = ggml_reshape_3d(ctx0, w, 1, n_used, n_tokens);
            w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
            ggml_build_forward_expand(gf, w);
        } else {
        // top-k (descending, as argsort's first n_used) without sorting the whole row; env LLAMA_INK_TOPK=0: argsort
        static const bool use_topk = [] { const char * v = getenv("LLAMA_INK_TOPK"); return !v || atoi(v) != 0; }();
        sel = use_topk ? ggml_top_k(ctx0, ggml_add(ctx0, scores, layer.ffn_exp_probs_b), n_used)
                       : ggml_argsort_top_k(ctx0, ggml_add(ctx0, scores, layer.ffn_exp_probs_b), n_used); // {n_used, n_tokens}
        w = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, scores, 1, layer.ffn_gate_inp->ne[1], n_tokens), sel);   // {1, n_used, n_tokens}
        w = ggml_reshape_2d(ctx0, w, n_used, n_tokens);
        w = ggml_div(ctx0, w, ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, w), 1.0f, 1e-20f));
        w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
        w = ggml_reshape_3d(ctx0, w, 1, n_used, n_tokens);
        }
        cb(sel, "ink_moe_sel", il);

        ggml_tensor * xr = ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens);
        const float cg = hparams.ink_gate_softcap, cu = hparams.ink_up_softcap;

        if (L.rvq_book || L.tq) {
            // RVQ2X8 layers: one mul_mat_id per group with the router-global ids; the kernel owns the range [a, a+n)
            // (op_params[12] = a, op_params[14] = 'RVQG') and writes zeros for the other slots, so no per-group id
            // masks or routing-weight masks are needed. src[3] = gains, src[4] = layer codebook.
            static const bool rvq_fuse_env = [] { const char * v = getenv("LLAMA_RVQ_FUSE"); return !v || atoi(v) != 0; }();
            static const bool tq_fuse_env  = [] { const char * v = getenv("LLAMA_TQ_FUSE");  return !v || atoi(v) != 0; }();
            const bool rvq_fuse = L.tq ? tq_fuse_env : rvq_fuse_env;   // TQ2T: GLU-down + group sum in tq.cu
            ggml_tensor * dsum = nullptr;
            int64_t a = 0;
            for (int64_t gi = 0; gi < L.n_grp; ++gi) {
                ggml_tensor * we[3] = {
                    gi == 0 ? layer.ffn_gate_exps : L.gexps[gi-1][0],
                    gi == 0 ? layer.ffn_up_exps   : L.gexps[gi-1][1],
                    gi == 0 ? layer.ffn_down_exps : L.gexps[gi-1][2] };
                auto mmid = [&](int m, ggml_tensor * inp) {
                    ggml_tensor * r = ggml_mul_mat_id(ctx0, we[m], inp, sel);
                    r->op_params[12] = (int32_t) a;
                    r->op_params[14] = 0x52565147; // 'RVQG' (GGML_CUDA_RVQ_GROUP_MAGIC)
                    r->src[3] = L.tq ? L.tq_gain[gi][m] : L.rvq_gain[gi][m];
                    r->src[4] = L.tq ? L.tq_sign[gi][m] : L.rvq_book;
                    return r;
                };
                ggml_tensor * g  = mmid(0, xr);
                ggml_tensor * u  = mmid(1, xr);
                if (rvq_fuse) {
                    // down reads the softcap GLU of (g, u) itself (op_params[13] = 'GLU1', src[5] = u) and, from the
                    // second group on, takes the earlier groups' sum for the slots it does not own (src[6])
                    ggml_tensor * d = mmid(2, g);
                    d->op_params[13] = 0x474C5531; // 'GLU1' (GGML_CUDA_RVQ_GLU_MAGIC)
                    memcpy(&d->op_params[10], &cg, sizeof(float));
                    memcpy(&d->op_params[11], &cu, sizeof(float));
                    d->src[5] = u;
                    d->src[6] = dsum;
                    dsum = d;
                } else {
                    ggml_tensor * gc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 1.0f/cg)), cg);
                    ggml_tensor * uc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f/cu)), cu);
                    ggml_tensor * h  = ggml_mul(ctx0, ggml_mul(ctx0, gc, ggml_sigmoid(ctx0, g)), uc);
                    ggml_tensor * d  = mmid(2, h);
                    dsum = dsum ? ggml_add(ctx0, dsum, d) : d;
                }
                a += L.grp_n[gi];
            }
            dsum = ggml_mul(ctx0, dsum, w);
            ggml_tensor * out = nullptr;
            for (int64_t i = 0; i < n_used; ++i) {
                ggml_tensor * e = ggml_view_2d(ctx0, dsum, n_embd, n_tokens, dsum->nb[2], i*dsum->nb[1]);
                out = out ? ggml_add(ctx0, out, e) : e;
            }
            return out;
        }

        if (L.n_grp > 1) {
            // expert groups: one mul_mat_id per group (one quant type each). Slots routed to another group get the
            // skip id (op_params[15] = "MMID" marks the node): the CUDA kernels neither read weights nor compute for them
            // and leave zeros, and their routing weight is masked to 0.
            const float SKIP = 16777216.0f; // GGML_CUDA_MMID_SKIP, exact in f32
            ggml_tensor * self = ggml_cast(ctx0, sel, GGML_TYPE_F32);                // {n_used, n_tokens}
            ggml_tensor * w2   = ggml_reshape_2d(ctx0, w, n_used, n_tokens);
            ggml_tensor * dsum = nullptr;
            int64_t a = 0;
            for (int64_t gi = 0; gi < L.n_grp; ++gi) {
                const int64_t n = L.grp_n[gi];
                ggml_tensor * we_g = gi == 0 ? layer.ffn_gate_exps : L.gexps[gi-1][0];
                ggml_tensor * we_u = gi == 0 ? layer.ffn_up_exps   : L.gexps[gi-1][1];
                ggml_tensor * we_d = gi == 0 ? layer.ffn_down_exps : L.gexps[gi-1][2];
                ggml_tensor * lo = ggml_step(ctx0, ggml_scale_bias(ctx0, self,  1.0f, 0.5f - (float) a));       // id >= a
                ggml_tensor * hi = ggml_step(ctx0, ggml_scale_bias(ctx0, self, -1.0f, (float) (a + n) - 0.5f)); // id <  a+n
                ggml_tensor * m  = ggml_mul(ctx0, lo, hi);
                // m ? id - a : SKIP, computed as m*((id - a) - SKIP) + SKIP. Keep every step exact in f32: (id - a) is a
                // small integer, and for in-group slots (id - a) - SKIP lies in [-2^24, -2^24 + n), where f32 has unit
                // spacing. Folding a into the bias (-a - SKIP) would round odd offsets (spacing 2 beyond 2^24).
                ggml_tensor * loc = ggml_scale_bias(ctx0, self, 1.0f, -(float) a);
                ggml_tensor * idf = ggml_scale_bias(ctx0, ggml_mul(ctx0, m, ggml_scale_bias(ctx0, loc, 1.0f, -SKIP)), 1.0f, SKIP);
                ggml_tensor * ids = ggml_cast(ctx0, idf, GGML_TYPE_I32);
                ggml_tensor * wg  = ggml_reshape_3d(ctx0, ggml_mul(ctx0, w2, m), 1, n_used, n_tokens);

                ggml_tensor * g = ggml_mul_mat_id(ctx0, we_g, xr, ids);
                ggml_tensor * u = ggml_mul_mat_id(ctx0, we_u, xr, ids);
                g->op_params[15] = 0x4D4D4944; // "MMID": expert-group skip ids possible
                u->op_params[15] = 0x4D4D4944; // "MMID": expert-group skip ids possible
                ggml_tensor * gc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 1.0f/cg)), cg);
                ggml_tensor * uc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f/cu)), cu);
                ggml_tensor * h  = ggml_mul(ctx0, ggml_mul(ctx0, gc, ggml_sigmoid(ctx0, g)), uc);
                ggml_tensor * d  = ggml_mul_mat_id(ctx0, we_d, h, ids);
                d->op_params[15] = 0x4D4D4944; // "MMID": expert-group skip ids possible
                d = ggml_mul(ctx0, d, wg);
                dsum = dsum ? ggml_add(ctx0, dsum, d) : d;
                a += n;
            }
            ggml_tensor * out = nullptr;
            for (int64_t i = 0; i < n_used; ++i) {
                ggml_tensor * e = ggml_view_2d(ctx0, dsum, n_embd, n_tokens, dsum->nb[2], i*dsum->nb[1]);
                out = out ? ggml_add(ctx0, out, e) : e;
            }
            return out;
        }

        ggml_tensor * g  = build_lora_mm_id(layer.ffn_gate_exps, xr, sel, layer.ffn_gate_exps_s); // {n_ff, n_used, n_tokens}
        ggml_tensor * u  = build_lora_mm_id(layer.ffn_up_exps,   xr, sel, layer.ffn_up_exps_s);
        ggml_tensor * gc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 1.0f/cg)), cg);
        ggml_tensor * uc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f/cu)), cu);
        ggml_tensor * h  = ggml_mul(ctx0, ggml_mul(ctx0, gc, ggml_sigmoid(ctx0, g)), uc);
        ggml_tensor * d  = build_lora_mm_id(layer.ffn_down_exps, h, sel, layer.ffn_down_exps_s); // {n_embd, n_used, n_tokens}
        d = ggml_mul(ctx0, d, w);
        ggml_tensor * out = nullptr;
        for (int64_t i = 0; i < n_used; ++i) {
            ggml_tensor * e = ggml_view_2d(ctx0, d, n_embd, n_tokens, d->nb[2], i*d->nb[1]);
            out = out ? ggml_add(ctx0, out, e) : e;
        }
        return out;
    };

    ggml_tensor * h = build_inp_embd(model.tok_embd);
    // do not normalize raw embeddings input (encoded image embeddings), as upstream #32217
    if (ubatch.token) {
        h = build_norm(h, model.tok_norm, NULL, LLM_NORM_RMS, -1);
    }
    cb(h, "ink_embd_norm", -1);

    ggml_tensor * blocks_S = nullptr; // rms-normed completed block states, stacked {n_embd, J, n_tokens}
    auto stack = [&](ggml_tensor * S, ggml_tensor * s) {
        ggml_tensor * s3 = ggml_reshape_3d(ctx0, s, n_embd, 1, n_tokens);
        return S ? ggml_concat(ctx0, S, s3, 1) : s3;
    };

    // LLAMA_INK_SBUF (default 1): the block states live in one buffer {n_embd, nslot, n_tokens} that grows downwards from
    // its top slot; each residual mix writes only its current state into the slot right below the J completed blocks and
    // reads [current, block J-1 .. block 0] as one view, instead of concatenating all blocks again (twice a layer). The
    // softmax mix over the J+1 states does not depend on their order. The read view is taken from the write (CPY) node,
    // so it depends on it; the next write into the same slot depends on this layer's outputs (WAR-safe in graph order).
    static const bool sbuf_env = [] { const char * v = getenv("LLAMA_INK_SBUF"); return !v || atoi(v) != 0; }();
    const int64_t n_blocks_max = (n_layer + hparams.ink_res_block - 1) / hparams.ink_res_block;
    const int64_t n_slot = n_blocks_max + 2;
    // made by an op (a leaf that is only viewed gets no buffer in the scheduler); its initial contents are never read
    ggml_tensor * sbuf = sbuf_env ? ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, h, n_embd, 1, n_tokens), n_embd, n_slot, n_tokens, 1) : nullptr;
    int64_t sb_J = 0;   // completed blocks in sbuf: slots [n_slot - sb_J, n_slot)
    // write cur (and, when blk, the new block right above it) below the blocks; return the view [cur, blocks...]
    auto sb_put = [&](ggml_tensor * cur, ggml_tensor * blk) -> ggml_tensor * {
        ggml_tensor * c3 = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        int64_t nw = 1;
        if (blk) {
            c3 = ggml_concat(ctx0, c3, ggml_reshape_3d(ctx0, blk, n_embd, 1, n_tokens), 1);
            nw = 2;
        }
        const int64_t lo = n_slot - sb_J - nw;   // first written slot
        GGML_ASSERT(lo >= 0);
        ggml_tensor * dstv = ggml_view_3d(ctx0, sbuf, n_embd, nw, n_tokens, sbuf->nb[1], sbuf->nb[2], lo * sbuf->nb[1]);
        ggml_tensor * w = ggml_cpy(ctx0, c3, dstv);
        if (blk) {
            sb_J += 1;
        }
        // [cur, blocks]: sb_J + 1 slots from lo
        return ggml_view_3d(ctx0, w, n_embd, sb_J + 1, n_tokens, sbuf->nb[1], sbuf->nb[2], 0);
    };
    // attn_res over an already assembled S {n_embd, J, n_tokens}
    auto attn_res_S = [&](ggml_tensor * S, ggml_tensor * query, int64_t nq) -> std::vector<ggml_tensor *> {
        return attn_res(S, {}, query, nq);
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & L    = model.ink[il];
        const bool   full = !hparams.is_swa(il);
        const bool   boundary = il % hparams.ink_res_block == 0;

        conv_rs = build_rs(inp_hybrid->get_recr(), mctx_recr->get_r_l(il), n_embd_r, n_seqs);

        // DSpark target features: the running (partial) block state at this layer's input
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = h;
            ggml_build_forward_expand(gf, h);
        }

        ggml_tensor * block_residual = h;
        std::vector<ggml_tensor *> ins;
        if (il == 0) {
            ins = {h, h, h};
        } else {
            ins = sbuf ? attn_res_S(sb_put(unorm(h), nullptr), L.res_attn_q, full ? 3 : 1)
                       : attn_res(blocks_S, { unorm(h) }, L.res_attn_q, full ? 3 : 1);
            if (ins.size() == 1) {
                ins = {ins[0], ins[0], ins[0]};
            }
        }
        ggml_tensor * xq, * xk, * xv;
        if (full) {
            xq = build_norm(ins[0], L.norm_q, NULL, LLM_NORM_RMS, il);
            xk = build_norm(ins[1], L.norm_k, NULL, LLM_NORM_RMS, il);
            xv = build_norm(ins[2], L.norm_v, NULL, LLM_NORM_RMS, il);
        } else {
            xq = xk = xv = build_norm(ins[0], L.norm_q, NULL, LLM_NORM_RMS, il);
        }
        ggml_tensor * a = build_attention(xq, xk, xv, il);
        // within a block the residual add happens in the conv: partial = block_residual + sconv(a)
        a = build_sconv(a, L.sc_attn, off_attn, il, boundary ? nullptr : block_residual);
        cb(a, boundary ? "ink_attn_out" : "ink_partial", il);

        ggml_tensor * partial;
        ggml_tensor * new_block = nullptr;
        if (boundary) {
            // the finished block joins the stack now; the MLP's residual mix already sees it
            if (sbuf) {
                new_block = unorm(block_residual);
            } else {
                blocks_S = stack(blocks_S, unorm(block_residual));
            }
            partial = a;
        } else {
            partial = a;
        }
        ggml_tensor * m_in = sbuf ? attn_res_S(sb_put(unorm(partial), new_block), L.res_mlp_q, 1)[0]
                                  : attn_res(blocks_S, { unorm(partial) }, L.res_mlp_q, 1)[0];
        m_in = build_norm(m_in, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        ggml_tensor * m = build_moe(m_in, il);
        h = build_sconv(m, L.sc_mlp, off_mlp, il, partial);   // partial + sconv(m)
        cb(h, "l_out", il);
    }

    {
        h = sbuf ? attn_res_S(sb_put(unorm(h), nullptr), model.res_out_q, 1)[0]
                 : attn_res(blocks_S, { unorm(h) }, model.res_out_q, 1)[0];
    }

    // DSpark target feature n_layer: the final normed hidden state of every token
    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        ggml_tensor * hn = build_norm(h, model.output_norm, NULL, LLM_NORM_RMS, -1);
        res->t_layer_inp[n_layer] = hn;
        cb(hn, "layer_inp", n_layer);
        ggml_build_forward_expand(gf, hn);
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        h = ggml_get_rows(ctx0, h, inp_out_ids);
    }
    h = build_norm(h, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(h, "result_norm", -1);
    res->t_embd = h;

    if (!cparams.embeddings) {
        const int64_t n_vocab_t = model.vocab.n_tokens();
        ggml_tensor * w_out = model.output->ne[1] > n_vocab_t   // public TQ1: rows past the tokenizer are padding
            ? ggml_view_2d(ctx0, model.output, model.output->ne[0], n_vocab_t, model.output->nb[1], 0) : model.output;
        ggml_tensor * cur = build_lora_mm(w_out, h);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        const float sc = hparams.f_final_logit_softcapping;
        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/sc)), sc);
        if (vocab_mask) {
            cur = ggml_add(ctx0, cur, vocab_mask);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }
    ggml_build_forward_expand(gf, res->t_logits ? res->t_logits : h);
}
