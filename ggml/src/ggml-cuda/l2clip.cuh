#include "common.cuh"

// fused Inkling 2 l2clip: y = x / clamp(sqrt(sum_rows(x*x))*s + b, lo, hi)
// (GGML_OP_SQR, SUM_ROWS, SQRT, SCALE, CLAMP, DIV with DIV->src[0] == SQR->src[0])
bool ggml_cuda_l2clip_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_l2clip_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// fused Inkling 2 static short conv: y = (ssm_conv(sx, k) + x) * s
// (GGML_OP_SSM_CONV, [RESHAPE/VIEW], ADD(conv, x), SCALE(bias 0)); returns the number of nodes to skip or 0
int  ggml_cuda_sconv_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_sconv_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);

// several consecutive F32 -> F32 CPY nodes (e.g. Inkling 2's per-slot conv state snapshots for speculative rollback)
// in one launch; returns the number of nodes (>= 2) or 0. env GGML_CUDA_MCPY_FUSE (default 1; views between the copies are skipped)
int  ggml_cuda_mcpy_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_mcpy_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int span);

// Inkling 2 attention softmax in one pass over the logits (a GGML_OP_SOFT_MAX node with op_params[10] ==
// GGML_CUDA_INK_SM_MAGIC): p = softmax_j(kq * scale + rel[idx[t, j]][h] + mask[t, j]) with a per-(token, head) sink
// logit in the denominator. srcs: 0 kq {n_kv, nt, nh} f32, 1 mask {n_kv, >= nt} f16/f32, 3 rel {nh, R} f32 (rows
// a token's relative logits), 4 idx {n_kv * nt} i32 (rel row of each (t, j)), 5 sinks {nh, nt} f32
#define GGML_CUDA_INK_SM_MAGIC 0x494e4b53
void ggml_cuda_op_ink_softmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Inkling 2 attention residual weighting: CONT(PERMUTE(S, 1,0,2,3)) followed by MUL_MAT(that, sc) -> one kernel
// o[d, q, t] = sum_j S[d, j, t] * sc[j, q, t] reading S in place. returns true if nodes i, i+1 match. env GGML_CUDA_RESW_FUSE
bool ggml_cuda_resw_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_resw_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int  ggml_cuda_attnres_match(const ggml_cgraph * cgraph, int i);
bool ggml_cuda_attnres_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// Inkling 2 dynamic short conv in one kernel (a GGML_OP_SCALE node with op_params[8] == GGML_CUDA_INK_DCONV_MAGIC):
// out[c, n] = scale * (x[c, n] + sum_lag sx[t + d_conv - lag, c, s] * (st[c, lag] + dw[lag, c / hh, n])), n = s*nst + t.
// srcs: 0 x {w, n_tokens} f32, 1 sx (window) {d_conv + nst, w, n_seqs} f32, 2 st {w, K} f32/f16, 3 dw {K, nkv, n_tokens} f32.
// op_params: [9] K, [10] d_conv, [11] nkv, [12] n_seq_tokens
#define GGML_CUDA_INK_DCONV_MAGIC 0x494e4b44
void ggml_cuda_op_ink_dconv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// Inkling 2 static short conv straight from the tokens and the cached state (SCALE node, op_params[8] == magic, srcs x, state, taps)
#define GGML_CUDA_INK_SCONV_MAGIC 0x494e4b53
void ggml_cuda_op_ink_sconv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
