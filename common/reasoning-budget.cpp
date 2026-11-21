#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

// GB10/ink2 (opt-in, env; unset = upstream behaviour):
// LLAMA_REASONING_BUDGET_ONCE=1   after a forced end, the model may not open a new thinking block. Inkling writes the
//                                 forced <|end_message|> and then opens <|content_thinking|> again, and upstream re-arms
//                                 the budget on each new block, so the thought never ends (budget 256: 3,464 vs 3,527 chars)
// LLAMA_REASONING_LOOP=L,C,W[;L,C,W...]  force the end of the thought when an L-char string is seen C+ times in its last W
//                                 chars, for any of the rules (e.g. 20,7,400 = the receiver's relay rule; 48,4,2400 for loops
//                                 with a long period, B70's second rule); a loop-forced end also blocks a new thought
static bool ink_rb_once() {
    static const bool v = [] { const char * e = getenv("LLAMA_REASONING_BUDGET_ONCE"); return e && atoi(e) != 0; }();
    return v;
}
struct ink_rb_loop_cfg { int len = 0, count = 0, window = 0; };
static const std::vector<ink_rb_loop_cfg> & ink_rb_loop() {
    static const std::vector<ink_rb_loop_cfg> c = [] {
        std::vector<ink_rb_loop_cfg> v;
        const char * e = getenv("LLAMA_REASONING_LOOP");
        if (!e) {
            return v;
        }
        std::string spec(e);
        size_t pos = 0;
        while (pos <= spec.size()) {
            const size_t end = std::min(spec.find(';', pos), spec.size());
            ink_rb_loop_cfg r;
            if (sscanf(spec.substr(pos, end - pos).c_str(), "%d,%d,%d", &r.len, &r.count, &r.window) == 3 && r.len > 0 && r.count >= 2 &&
                    r.window >= r.len) {
                v.push_back(r);
            }
            pos = end + 1;
        }
        return v;
    }();
    return c;
}
static int ink_rb_loop_window_max() {
    static const int w = [] { int m = 0; for (const auto & r : ink_rb_loop()) { m = std::max(m, r.window); } return m; }();
    return w;
}
bool common_reasoning_budget_loop_enabled() { return !ink_rb_loop().empty(); }

// UTF-8 bytes -> code points (a cut first character is skipped, a cut last one is left out)
static std::u32string ink_rb_cpts(const std::string & s) {
    std::u32string out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 0;
        if (n == 0) { i++; continue; }
        if (i + n > s.size()) { break; }
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
        for (int k = 1; k < n; k++) { cp = (cp << 6) | (s[i + k] & 0x3f); }
        out.push_back(cp);
        i += n;
    }
    return out;
}

// the receiver's rule (run_loop.py): in the last W chars (at least L*C of them), some L-char string occurs C+ times
static bool ink_rb_looping(const std::string & tail, const ink_rb_loop_cfg & cfg) {
    std::u32string w = ink_rb_cpts(tail);
    if ((int) w.size() > cfg.window) { w.erase(0, w.size() - cfg.window); }
    if ((int) w.size() < cfg.len * cfg.count) { return false; }
    std::unordered_map<std::u32string_view, int> seen;
    const std::u32string_view v(w);
    for (size_t i = 0; i + cfg.len <= v.size(); i++) {
        if (++seen[v.substr(i, cfg.len)] >= cfg.count) { return true; }
    }
    return false;
}

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // GB10/ink2 (see ink_rb_once / ink_rb_loop)
    bool        forced_end = false; // the end of the thought was forced (budget or loop), not written by the model
    bool        loop_hit   = false; // the force came from the loop check
    std::string tail;               // last bytes of the thought, for the loop check
    int         n_tail_tok = 0;     // tokens appended to tail
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            // GB10/ink2 LLAMA_REASONING_LOOP: a looping thought is ended like an exhausted budget
            const ink_rb_loop_cfg * hit = nullptr;
            if (ctx->state == REASONING_BUDGET_COUNTING && !ink_rb_loop().empty() && ctx->vocab != nullptr) {
                ctx->tail += common_token_to_piece(ctx->vocab, token, false);
                const size_t keep = (size_t) ink_rb_loop_window_max() * 4 + 8;
                if (ctx->tail.size() > 2 * keep) { ctx->tail.erase(0, ctx->tail.size() - keep); }
                ctx->n_tail_tok++;
                for (const auto & r : ink_rb_loop()) {
                    // rules with a long window are checked every 4th token (their check hashes W strings of L chars)
                    if (r.window > 1000 && (ctx->n_tail_tok & 3) != 0) { continue; }
                    if (ink_rb_looping(ctx->tail, r)) { hit = &r; break; }
                }
            }
            if (hit) {
                {
                    const auto & cfg = *hit;
                    ctx->loop_hit = true;
                    ctx->state = utf8_complete ? REASONING_BUDGET_FORCING : REASONING_BUDGET_WAITING_UTF8;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "loop detected, forcing end sequence\n");
                    LOG_WRN("reasoning-budget: loop in the thought (LLAMA_REASONING_LOOP=%d,%d,%d), forcing its end\n", cfg.len, cfg.count, cfg.window);
                    break;
                }
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            // an end tag that is already in the prompt (e.g. "<think></think>") also ends forcing
            if (match >= 0 || ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                ctx->forced_end = true;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->forced_end && (ink_rb_once() || ctx->loop_hit)) {
                break; // GB10/ink2: the start token is masked in apply(); never re-arm
            }
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                ctx->tail.clear();
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_DONE && ctx->forced_end && (ink_rb_once() || ctx->loop_hit)) {
        // GB10/ink2: after a forced end the model may not open a new thought (single-token start tags only)
        for (const auto & seq : ctx->start_matcher.seqs) {
            if (seq.size() != 1) { continue; }
            for (size_t i = 0; i < cur_p->size; i++) {
                if (cur_p->data[i].id == seq[0]) { cur_p->data[i].logit = -INFINITY; }
            }
        }
        return;
    }

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->forced_end = false;
    ctx->loop_hit = false;
    ctx->tail.clear();
    ctx->n_tail_tok = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
