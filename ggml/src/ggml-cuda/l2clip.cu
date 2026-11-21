#include "rvq.cuh"
#include "l2clip.cuh"
#include "convert.cuh"

// one warp per row
static __global__ void l2clip_f32(const float * __restrict__ x, float * __restrict__ y, const int ncols, const int nrows,
                                  const float s, const float b, const float lo, const float hi) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + (int64_t) row*ncols;
    float       * yr = y + (int64_t) row*ncols;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < ncols; c += WARP_SIZE) {
        const float v = xr[c];
        acc += v*v;
    }
    acc = warp_reduce_sum(acc);
    const float n   = fminf(fmaxf(sqrtf(acc)*s + b, lo), hi);
    const float inv = 1.0f/n;
    for (int c = threadIdx.x; c < ncols; c += WARP_SIZE) {
        yr[c] = xr[c]/n;
    }
    GGML_UNUSED(inv);
}

bool ggml_cuda_l2clip_match(const ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * v = getenv("GGML_CUDA_L2CLIP_FUSE"); return v == nullptr || atoi(v) != 0; }();
    if (!on || i + 6 > cgraph->n_nodes) {
        return false;
    }
    const ggml_op ops[6] = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_CLAMP, GGML_OP_DIV };
    const ggml_tensor * const * n = cgraph->nodes + i;
    for (int k = 0; k < 6; ++k) {
        if (n[k]->op != ops[k]) {
            return false;
        }
    }
    const ggml_tensor * x = n[0]->src[0];
    if (n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] || n[4]->src[0] != n[3] ||
        n[5]->src[0] != x || n[5]->src[1] != n[4]) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || n[5]->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(n[5]) ||
        !ggml_are_same_shape(x, n[5]) || !ggml_is_contiguous(n[4]) || n[4]->ne[0] != 1 ||
        ggml_nrows(n[4]) != ggml_nrows(x)) {
        return false;
    }
    const int outputs[] = { i + 5 };
    const ggml_op fops[6] = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_CLAMP, GGML_OP_DIV };
    return ggml_can_fuse_subgraph(cgraph, i, 6, fops, outputs, 1);
}

void ggml_cuda_op_l2clip_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * x = n[0]->src[0];
    const ggml_tensor * dst = n[5];
    float s, b, lo, hi;
    memcpy(&s,  (const float *) n[3]->op_params + 0, sizeof(float));
    memcpy(&b,  (const float *) n[3]->op_params + 1, sizeof(float));
    memcpy(&lo, (const float *) n[4]->op_params + 0, sizeof(float));
    memcpy(&hi, (const float *) n[4]->op_params + 1, sizeof(float));
    const int ncols = x->ne[0];
    const int nrows = ggml_nrows(x);
    const dim3 block(WARP_SIZE, 4, 1);
    const dim3 grid((nrows + 3)/4, 1, 1);
    l2clip_f32<<<grid, block, 0, ctx.stream()>>>((const float *) x->data, (float *) dst->data, ncols, nrows, s, b, lo, hi);
}

// sx {nc + n_t - 1, w, n_s} time-major, k {nc, w}, x/y {w, n_t, n_s} contiguous
template <int NC>
static __global__ void sconv_f32(const float * __restrict__ sx, const float * __restrict__ k, const float * __restrict__ x,
                                 float * __restrict__ y, const int w, const int n_t, const int n_s,
                                 const int64_t sx_nb1, const int64_t sx_nb2, const float s) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= w) {
        return;
    }
    const int t  = blockIdx.y;
    const int is = blockIdx.z;
    const float * sr = sx + is*sx_nb2 + c*sx_nb1 + t;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc += sr[j]*k[c*NC + j];
    }
    const int64_t o = ((int64_t) is*n_t + t)*w + c;
    y[o] = (acc + x[o])*s;
}

int ggml_cuda_sconv_match(const ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * v = getenv("GGML_CUDA_SCONV_FUSE"); return v == nullptr || atoi(v) != 0; }();
    if (!on) {
        return 0;
    }
    const ggml_tensor * conv = cgraph->nodes[i];
    static int dbg = [] { const char * v = getenv("GGML_CUDA_SCONV_DEBUG"); return v ? atoi(v) : 0; }();
    if (dbg > 0) {
        --dbg;
        fprintf(stderr, "sconv dbg: %s", ggml_op_desc(conv));
        for (int q = i + 1; q < i + 6 && q < cgraph->n_nodes; ++q) {
            const ggml_tensor * t = cgraph->nodes[q];
            fprintf(stderr, " | %s(%s) s0=%s s1=%s", ggml_op_desc(t), t->name, t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
        }
        fprintf(stderr, "  [conv=%s]\n", conv->name);
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && j <= i + 2 && (cgraph->nodes[j]->op == GGML_OP_RESHAPE || cgraph->nodes[j]->op == GGML_OP_VIEW)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * add = cgraph->nodes[j];
    const ggml_tensor * sc  = cgraph->nodes[j + 1];
    if (add->op != GGML_OP_ADD || sc->op != GGML_OP_SCALE || add->src[0] != conv || sc->src[0] != add) {
        return 0;
    }
    // the in-between view nodes must only feed the add
    for (int v = i + 1; v < j; ++v) {
        if (add->src[1] != cgraph->nodes[v]) {
            return 0;
        }
    }
    const ggml_tensor * sx = conv->src[0];
    const ggml_tensor * kk = conv->src[1];
    const ggml_tensor * x  = add->src[1];
    const int nc = kk->ne[0];
    if (nc != 4 || sx->type != GGML_TYPE_F32 || kk->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 ||
        sx->nb[0] != sizeof(float) || !ggml_is_contiguous(kk) || !ggml_is_contiguous(x) || !ggml_is_contiguous(sc) ||
        !ggml_are_same_shape(conv, x) || ggml_get_op_params_f32(sc, 1) != 0.0f) {
        return 0;
    }
    // the view nodes in between are not computed and view a tensor outside the subgraph: check only conv/add/scale
    const int     idxs[3] = { i, j, j + 1 };
    const ggml_op ops[3]  = { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_SCALE };
    const int outputs[] = { j + 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 3, ops, outputs, 1)) {
        return 0;
    }
    return j + 1 - i;
}

void ggml_cuda_op_sconv_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * conv = cgraph->nodes[i];
    const ggml_tensor * sc   = cgraph->nodes[last];
    const ggml_tensor * add  = sc->src[0];
    const ggml_tensor * sx = conv->src[0];
    const ggml_tensor * kk = conv->src[1];
    const ggml_tensor * x  = add->src[1];
    const int w   = conv->ne[0];
    const int n_t = conv->ne[1];
    const int n_s = conv->ne[2];
    const float s = ggml_get_op_params_f32(sc, 0);
    const dim3 block(128, 1, 1);
    const dim3 grid((w + 127)/128, n_t, n_s);
    sconv_f32<4><<<grid, block, 0, ctx.stream()>>>((const float *) sx->data, (const float *) kk->data, (const float *) x->data,
        (float *) sc->data, w, n_t, n_s, sx->nb[1]/sizeof(float), sx->nb[2]/sizeof(float), s);
}

// =====================================================================================================================
// batched strided F32 copies
// =====================================================================================================================

#define MCPY_MAX 8

struct mcpy_desc {
    const char * src;
    char       * dst;
    int64_t      ne0, ne1, ne2;
    size_t       snb1, snb2, dnb1, dnb2;
};
struct mcpy_args {
    mcpy_desc c[MCPY_MAX];
};

static __global__ void mcpy_f32(const mcpy_args a) {
    const mcpy_desc & d = a.c[blockIdx.y];
    const int64_t n = d.ne0 * d.ne1 * d.ne2;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t i0 = i % d.ne0;
        const int64_t r  = i / d.ne0;
        const int64_t i1 = r % d.ne1;
        const int64_t i2 = r / d.ne1;
        *(float *) (d.dst + i2 * d.dnb2 + i1 * d.dnb1 + i0 * sizeof(float)) =
            *(const float *) (d.src + i2 * d.snb2 + i1 * d.snb1 + i0 * sizeof(float));
    }
}

static bool mcpy_ok(const ggml_tensor * n) {
    if (n->op != GGML_OP_CPY) {
        return false;
    }
    const ggml_tensor * s = n->src[0], * d = n->src[1];
    return s->type == GGML_TYPE_F32 && d->type == GGML_TYPE_F32 && s->nb[0] == sizeof(float) && d->nb[0] == sizeof(float) &&
           s->ne[3] == 1 && d->ne[3] == 1 && ggml_are_same_shape(s, d) && ggml_nelements(s) > 0;
}

static void mcpy_range(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    lo = (uintptr_t) t->data;
    hi = lo + (t->ne[2] - 1) * t->nb[2] + (t->ne[1] - 1) * t->nb[1] + t->ne[0] * sizeof(float);
}

static bool mcpy_view(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// returns the span of nodes [i, i + span) holding >= 2 CPY nodes and only views between them, or 0
int ggml_cuda_mcpy_match(const ggml_cgraph * cgraph, int i) {
    if (!ggml_cuda_rvq_knob("GGML_CUDA_MCPY_FUSE", 1) || !mcpy_ok(cgraph->nodes[i])) {
        return 0;
    }
    int idx[MCPY_MAX] = { i };
    int n = 1, last = i;
    for (int j = i + 1; j < cgraph->n_nodes && n < MCPY_MAX; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (mcpy_view(t)) {
            continue;   // the views the copies read and write through
        }
        if (!mcpy_ok(t)) {
            break;
        }
        // the batched launch runs every copy at once: no copy may read or write what another one writes
        uintptr_t slo, shi, tlo, thi;
        mcpy_range(t->src[0], slo, shi);
        mcpy_range(t->src[1], tlo, thi);
        bool clash = false;
        for (int k = 0; k < n; ++k) {
            uintptr_t dlo, dhi, rlo, rhi;
            mcpy_range(cgraph->nodes[idx[k]]->src[1], dlo, dhi);
            mcpy_range(cgraph->nodes[idx[k]]->src[0], rlo, rhi);
            clash |= (slo < dhi && dlo < shi) || (tlo < dhi && dlo < thi) || (tlo < rhi && rlo < thi);
        }
        if (clash) {
            break;
        }
        idx[n++] = j;
        last = j;
    }
    return n >= 2 ? last - i + 1 : 0;
}

void ggml_cuda_op_mcpy_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int span) {
    mcpy_args a = {};
    int64_t nmax = 0;
    int n = 0;
    for (int j = i; j < i + span && n < MCPY_MAX; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op != GGML_OP_CPY) {
            continue;
        }
        const ggml_tensor * s = t->src[0], * d = t->src[1];
        a.c[n++] = { (const char *) s->data, (char *) d->data, s->ne[0], s->ne[1], s->ne[2], s->nb[1], s->nb[2], d->nb[1], d->nb[2] };
        nmax = std::max(nmax, ggml_nelements(s));
    }
    const int nb = (int) std::min<int64_t>((nmax + 255) / 256, 1024);
    mcpy_f32<<<dim3(nb, n), 256, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

// =====================================================================================================================
// Inkling 2 attention softmax (bias gather + scale + mask + sink + softmax in one kernel)
// =====================================================================================================================

template <typename TM>
static __global__ void __launch_bounds__(256)
ink_softmax_f32(const float * __restrict__ kq, const TM * __restrict__ mask, const float * __restrict__ rel,
                const int32_t * __restrict__ idx, const float * __restrict__ sinks, float * __restrict__ dst,
                const int n_kv, const int nt, const int nh, const size_t mask_s1, const size_t sink_s1, const float scale,
                const int cache, const int E) {
    extern __shared__ float buf[];   // n_kv logits when cache != 0
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float   * row = kq + ((size_t) h * nt + t) * n_kv;
    const int32_t * ir  = idx + (size_t) t * n_kv;
    const TM      * mr  = mask + (size_t) t * mask_s1;
    float         * out = dst + ((size_t) h * nt + t) * n_kv;
    const float sink = sinks[(size_t) t * sink_s1 + h];

    auto logit = [&](int j) -> float {
        float b;
        if (E > 0) {   // raw table {E, nh * n_tokens}: row = t_abs * (E + 1) + e, e == E the zero column
            const int row = ir[j], ta = row / (E + 1), e = row - ta * (E + 1);
            b = e < E ? rel[((size_t) ta * nh + h) * E + e] : 0.0f;
        } else {
            b = rel[(size_t) ir[j] * nh + h];
        }
        return row[j] * scale + b + (float) mr[j];
    };
    auto block_max = [&](float v) -> float {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if ((tid & 31) == 0) red[tid >> 5] = v;
        __syncthreads();
        v = tid < (int) (blockDim.x >> 5) ? red[tid] : -INFINITY;
        if (tid < 32) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if (tid == 0) red[0] = v;
        }
        __syncthreads();
        v = red[0];
        __syncthreads();
        return v;
    };
    auto block_sum = [&](float v) -> float {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((tid & 31) == 0) red[tid >> 5] = v;
        __syncthreads();
        v = tid < (int) (blockDim.x >> 5) ? red[tid] : 0.0f;
        if (tid < 32) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (tid == 0) red[0] = v;
        }
        __syncthreads();
        v = red[0];
        __syncthreads();
        return v;
    };

    float mx = -INFINITY;
    for (int j = tid; j < n_kv; j += blockDim.x) {
        const float l = logit(j);
        if (cache) buf[j] = l;
        mx = fmaxf(mx, l);
    }
    mx = fmaxf(block_max(mx), sink);
    float sm = 0.0f;
    for (int j = tid; j < n_kv; j += blockDim.x) {
        sm += expf((cache ? buf[j] : logit(j)) - mx);
    }
    sm = block_sum(sm) + expf(sink - mx);
    const float inv = 1.0f / sm;
    for (int j = tid; j < n_kv; j += blockDim.x) {
        out[j] = expf((cache ? buf[j] : logit(j)) - mx) * inv;
    }
}

void ggml_cuda_op_ink_softmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kq = dst->src[0], * mask = dst->src[1], * rel = dst->src[3], * idx = dst->src[4], * sk = dst->src[5];
    GGML_ASSERT(kq->type == GGML_TYPE_F32 && ggml_is_contiguous(kq) && mask && rel && idx && sk);
    GGML_ASSERT(rel->type == GGML_TYPE_F32 && idx->type == GGML_TYPE_I32 && sk->type == GGML_TYPE_F32);
    const int n_kv = kq->ne[0], nt = kq->ne[1], nh = kq->ne[2];
    const int E = dst->op_params[11];
    GGML_ASSERT((E > 0 ? rel->ne[0] == E : rel->ne[0] == nh) && sk->ne[0] == nh && ggml_nelements(idx) == (int64_t) n_kv * nt);
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    // the logits cache in dynamic shared memory: up to the device's opt-in limit minus the kernel's static part (a
    // reduction array); past it the kernel recomputes the logits each pass. (11/19: a fixed n_kv <= 12288 cut left
    // 12257..12288 at 48 KB + the static part, past the default 48 KB limit -> "invalid argument" at n_kv > ~12.2K)
    static int smem_optin = -1;
    if (smem_optin < 0) {
        const int dev = ggml_cuda_get_device();
        int optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        const int want = optin - 1024;   // room for the static reduction array
        CUDA_CHECK(cudaFuncSetAttribute(ink_softmax_f32<half>,  cudaFuncAttributeMaxDynamicSharedMemorySize, want));
        CUDA_CHECK(cudaFuncSetAttribute(ink_softmax_f32<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, want));
        smem_optin = want;
    }
    const int cache = (size_t) n_kv * sizeof(float) <= (size_t) smem_optin;
    const size_t smem = cache ? (size_t) n_kv * sizeof(float) : 0;
    const dim3 grid(nt, nh);
    cudaStream_t st = ctx.stream();
    if (mask->type == GGML_TYPE_F16) {
        ink_softmax_f32<half><<<grid, 256, smem, st>>>((const float *) kq->data, (const half *) mask->data, (const float *) rel->data,
            (const int32_t *) idx->data, (const float *) sk->data, (float *) dst->data, n_kv, nt, nh, mask->nb[1] / sizeof(half),
            sk->nb[1] / sizeof(float), scale, cache, E);
    } else {
        ink_softmax_f32<float><<<grid, 256, smem, st>>>((const float *) kq->data, (const float *) mask->data, (const float *) rel->data,
            (const int32_t *) idx->data, (const float *) sk->data, (float *) dst->data, n_kv, nt, nh, mask->nb[1] / sizeof(float),
            sk->nb[1] / sizeof(float), scale, cache, E);
    }
    CUDA_CHECK(cudaGetLastError());
}

// =====================================================================================================================
// attention residual weighting (CONT(PERMUTE) + MUL_MAT with a short K)
// =====================================================================================================================

static __global__ void resw_f32(const char * __restrict__ S, const char * __restrict__ sc, char * __restrict__ o,
                                const int n_embd, const int J, const int nq,
                                const size_t s1, const size_t s2, const size_t c1, const size_t c2, const size_t o1, const size_t o2) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < J; ++j) {
        const float sv = *(const float *) (S + t * s2 + j * s1 + (size_t) d * sizeof(float));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < nq) {
                acc[q] += sv * *(const float *) (sc + t * c2 + q * c1 + (size_t) j * sizeof(float));
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q < nq) {
            *(float *) (o + t * o2 + q * o1 + (size_t) d * sizeof(float)) = acc[q];
        }
    }
}

bool ggml_cuda_resw_match(const ggml_cgraph * cgraph, int i) {
    if (!ggml_cuda_rvq_knob("GGML_CUDA_RESW_FUSE", 1) || i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * c = cgraph->nodes[i], * m = cgraph->nodes[i + 1];
    if (c->op != GGML_OP_CONT || m->op != GGML_OP_MUL_MAT || m->src[0] != c) {
        return false;
    }
    const ggml_tensor * p = c->src[0];
    if (p->op != GGML_OP_PERMUTE || p->op_params[0] != 1 || p->op_params[1] != 0 || p->op_params[2] != 2 || p->op_params[3] != 3) {
        return false;
    }
    const ggml_tensor * S = p->src[0], * sc = m->src[1];
    if (S->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || S->nb[0] != sizeof(float) ||
        sc->nb[0] != sizeof(float) || m->nb[0] != sizeof(float) || S->ne[3] != 1 || sc->ne[3] != 1) {
        return false;
    }
    // S {n_embd, J, nt}, St = c {J, n_embd, nt}, sc {J, nq, nt}, m {n_embd, nq, nt}
    if (c->ne[0] != S->ne[1] || sc->ne[0] != c->ne[0] || sc->ne[2] != S->ne[2] || m->ne[0] != S->ne[0] || sc->ne[1] > 4) {
        return false;
    }
    // the graph allocator may give the MUL_MAT's output the memory of S once the CONT (S's last reader in the plain
    // graph) has run; the fused kernel reads S while it writes the output, so they must not overlap
    {
        const uintptr_t s0 = (uintptr_t) S->data, s1 = s0 + ggml_nbytes(S);
        const uintptr_t m0 = (uintptr_t) m->data, m1 = m0 + ggml_nbytes(m);
        const uintptr_t c0 = (uintptr_t) sc->data, c1 = c0 + ggml_nbytes(sc);
        if ((s0 < m1 && m0 < s1) || (c0 < m1 && m0 < c1)) {
            return false;
        }
    }
    int idxs[2] = { i, i + 1 };
    ggml_op ops[2] = { GGML_OP_CONT, GGML_OP_MUL_MAT };
    const int outputs[] = { i + 1 };
    return ggml_can_fuse_subgraph_ext(cgraph, idxs, 2, ops, outputs, 1);
}

void ggml_cuda_op_resw_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * m = cgraph->nodes[i + 1];
    const ggml_tensor * S = cgraph->nodes[i]->src[0]->src[0], * sc = m->src[1];
    const int n_embd = S->ne[0], J = S->ne[1], nt = S->ne[2], nq = sc->ne[1];
    const dim3 grid((n_embd + 255) / 256, nt);
    resw_f32<<<grid, 256, 0, ctx.stream()>>>((const char *) S->data, (const char *) sc->data, (char *) m->data, n_embd, J, nq,
        S->nb[1], S->nb[2], sc->nb[1], sc->nb[2], m->nb[1], m->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

// ---- whole attention residual in one kernel (GGML_CUDA_ATTNRES_FUSE, default 1) ----
// graph: sc = MUL_MAT(q2 {n_embd, nq}, S {n_embd, J, nt}); PERMUTE; CONT; SOFT_MAX (scale) ... CONT(PERMUTE(S)); MUL_MAT(St, sc).
// The score nodes are skipped where they stand (ggml_cuda_attnres_match) and the resw node computes scores, softmax and
// the weighted sum per token: S is read from DRAM once (the second pass hits L1 / L2), scores in f32 (the plain graph
// quantizes S to q8_1 for a quantized query).
static __global__ void attnres_f32(const float * __restrict__ q, const char * __restrict__ S, char * __restrict__ o,
                                   const int n_embd, const int J, const int nq, const size_t s1, const size_t s2,
                                   const size_t o1, const size_t o2, const float scale) {
    extern __shared__ float sm_ar[];   // [nw][J*nq] partial dots, then p [J*nq]
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32, nw = blockDim.x / 32;
    const int JQ = J * nq;
    float * pp = sm_ar + nw * JQ;
    const char * St = S + (size_t) t * s2;
    for (int j = 0; j < J; ++j) {
        const float * sj = (const float *) (St + (size_t) j * s1);
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int d = tid; d < n_embd; d += blockDim.x) {
            const float v = sj[d];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (k < nq) {
                    a[k] += v * q[k * n_embd + d];
                }
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k < nq) {
                float x = a[k];
#pragma unroll
                for (int h = 16; h >= 1; h >>= 1) {
                    x += __shfl_xor_sync(0xffffffff, x, h);
                }
                if (lane == 0) {
                    sm_ar[warp * JQ + j * nq + k] = x;
                }
            }
        }
    }
    __syncthreads();
    if (tid < nq) {
        float mx = -INFINITY;
        for (int j = 0; j < J; ++j) {
            float x = 0.0f;
            for (int w = 0; w < nw; ++w) {
                x += sm_ar[w * JQ + j * nq + tid];
            }
            x *= scale;
            pp[j * nq + tid] = x;
            mx = fmaxf(mx, x);
        }
        float sum = 0.0f;
        for (int j = 0; j < J; ++j) {
            const float e = expf(pp[j * nq + tid] - mx);
            pp[j * nq + tid] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < J; ++j) {
            pp[j * nq + tid] *= inv;
        }
    }
    __syncthreads();
    for (int d = tid; d < n_embd; d += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < J; ++j) {
            const float v = *(const float *) (St + (size_t) j * s1 + (size_t) d * sizeof(float));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (k < nq) {
                    acc[k] += v * pp[j * nq + k];
                }
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k < nq) {
                *(float *) (o + (size_t) t * o2 + (size_t) k * o1 + (size_t) d * sizeof(float)) = acc[k];
            }
        }
    }
}

static bool attnres_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

// a skipped score chain waiting for its resw node: the softmax output it stands for
struct attnres_pending {
    const ggml_tensor * sm  = nullptr;
    const ggml_tensor * q2  = nullptr;
    float scale = 1.0f;
};
static attnres_pending g_attnres;

int ggml_cuda_attnres_match(const ggml_cgraph * cgraph, int i) {
    static const int on = ggml_cuda_rvq_knob("GGML_CUDA_ATTNRES_FUSE", 0);
    if (!on || i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sc = cgraph->nodes[i], * pe = cgraph->nodes[i + 1], * ct = cgraph->nodes[i + 2], * sm = cgraph->nodes[i + 3];
    if (sc->op != GGML_OP_MUL_MAT || pe->op != GGML_OP_PERMUTE || pe->src[0] != sc || ct->op != GGML_OP_CONT || ct->src[0] != pe ||
        sm->op != GGML_OP_SOFT_MAX || sm->src[0] != ct || sm->src[1] || sm->src[2]) {
        return 0;
    }
    if (pe->op_params[0] != 1 || pe->op_params[1] != 0 || pe->op_params[2] != 2 || pe->op_params[3] != 3) {
        return 0;
    }
    float scale, max_bias;
    memcpy(&scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    const ggml_tensor * q2 = sc->src[0], * S = sc->src[1];
    const int64_t n_embd = S->ne[0], J = S->ne[1], nq = q2->ne[1];
    if (S->ne[2] < 64) {   // prefill only: one block per token, too few blocks for a decode step
        return 0;
    }
    if (max_bias != 0.0f || S->type != GGML_TYPE_F32 || S->nb[0] != sizeof(float) || S->ne[3] != 1 || q2->ne[0] != n_embd ||
        nq > 4 || q2->ne[2] != 1 || q2->ne[3] != 1 || !ggml_is_contiguous(q2) || sm->type != GGML_TYPE_F32 ||
        (q2->type != GGML_TYPE_F32 && !ggml_get_to_fp32_cuda(q2->type)) || J * nq > 256) {
        return 0;
    }
    if ((on == 2 && q2->type != GGML_TYPE_F32) || (on == 3 && q2->type == GGML_TYPE_F32)) {   // diagnosis: one query kind only
        return 0;
    }
    // the resw pair that reads sm, with S unchanged on the way
    const uintptr_t s0 = (uintptr_t) S->data, s1 = s0 + ggml_nbytes(S);
    for (int k = i + 4; k + 1 < cgraph->n_nodes && k < i + 400; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op == GGML_OP_CONT && cgraph->nodes[k + 1]->op == GGML_OP_MUL_MAT && cgraph->nodes[k + 1]->src[1] == sm) {
            const ggml_tensor * m = cgraph->nodes[k + 1];
            if (!ggml_cuda_resw_match(cgraph, k) || n->src[0]->src[0] != S) {
                return 0;
            }
            // each skipped tensor has exactly its one reader in the chain
            if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
                ggml_node_get_use_count(cgraph, i + 2) != 1 || ggml_node_get_use_count(cgraph, i + 3) != 1 ||
                (sc->flags | pe->flags | ct->flags | sm->flags) & GGML_TENSOR_FLAG_OUTPUT) {
                return 0;
            }
            g_attnres.sm = sm; g_attnres.q2 = q2; g_attnres.scale = scale;
            return 4;
        }
        if (!attnres_is_view(n) && n->data) {
            const uintptr_t d0 = (uintptr_t) n->data, d1 = d0 + ggml_nbytes(n);
            if (d0 < s1 && s0 < d1) {
                return 0;   // something writes into S before the weighted sum
            }
        }
    }
    return 0;
}

bool ggml_cuda_attnres_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * m = cgraph->nodes[i + 1];
    if (g_attnres.sm == nullptr || m->src[1] != g_attnres.sm) {
        return false;
    }
    const ggml_tensor * S = cgraph->nodes[i]->src[0]->src[0], * q2 = g_attnres.q2;
    const int n_embd = S->ne[0], J = S->ne[1], nt = S->ne[2], nq = q2->ne[1];
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float> qf(ctx.pool());
    const float * qp = (const float *) q2->data;
    if (q2->type != GGML_TYPE_F32) {
        qf.alloc((size_t) n_embd * nq);
        ggml_get_to_fp32_cuda(q2->type)(q2->data, qf.ptr, (int64_t) n_embd * nq, st);
        qp = qf.ptr;
    }
    const int nth = 256;
    const size_t smem = (size_t) (nth / 32 + 1) * J * nq * sizeof(float);
    attnres_f32<<<nt, nth, smem, st>>>(qp, (const char *) S->data, (char *) m->data, n_embd, J, nq, S->nb[1], S->nb[2],
        m->nb[1], m->nb[2], g_attnres.scale);
    CUDA_CHECK(cudaGetLastError());
    g_attnres.sm = nullptr;
    return true;
}

// ---- Inkling 2 static short conv from the tokens and the cached state (see l2clip.cuh, GGML_CUDA_INK_SCONV_MAGIC) ----
// y[s][t][c] = (sum_j k[c][j] * u(t + j - D) + x[s][t][c]) * scale, u(p) = p >= 0 ? x[s][p][c] : st[s][c][D + p]
template <int D>
static __global__ void ink_sconv_x_f32(const float * __restrict__ x, const char * __restrict__ st, const float * __restrict__ k,
                                       float * __restrict__ y, const int w, const int n_t, const size_t st_nb1, const size_t st_nb2,
                                       const float scale, const float * __restrict__ res) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= w) {
        return;
    }
    const int t = blockIdx.y, s = blockIdx.z;
    const float * xs = x + (size_t) s * n_t * w;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j <= D; ++j) {
        const int p = t + j - D;
        const float v = p >= 0 ? xs[(size_t) p * w + c] : *(const float *) (st + s * st_nb2 + c * st_nb1 + (size_t) (D + p) * sizeof(float));
        acc += v * k[c * (D + 1) + j];
    }
    const size_t o = ((size_t) s * n_t + t) * w + c;
    const float v = __fmul_rn(acc + xs[(size_t) t * w + c], scale);
    y[o] = res ? __fadd_rn(res[o], v) : v;   // res + y: the separate ADD's rounding
}

void ggml_cuda_op_ink_sconv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * st = dst->src[1], * k = dst->src[2], * res = dst->src[3];
    const int D = dst->op_params[9], n_t = dst->op_params[10];
    const int w = x->ne[0];
    const int n_s = ggml_nrows(x) / n_t;
    GGML_ASSERT(D == 3 && k->ne[0] == 4 && k->ne[1] == w && ggml_is_contiguous(x) && ggml_is_contiguous(k) && ggml_is_contiguous(dst));
    GGML_ASSERT(st->type == GGML_TYPE_F32 && st->nb[0] == sizeof(float) && st->ne[0] == D && st->ne[1] == w && st->ne[2] == n_s);
    GGML_ASSERT(!res || (res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_are_same_shape(res, x)));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const dim3 grid((w + 127) / 128, n_t, n_s);
    ink_sconv_x_f32<3><<<grid, 128, 0, ctx.stream()>>>((const float *) x->data, (const char *) st->data, (const float *) k->data,
        (float *) dst->data, w, n_t, st->nb[1], st->nb[2], scale, res ? (const float *) res->data : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

// ---- Inkling 2 dynamic short conv (see l2clip.cuh, GGML_CUDA_INK_DCONV_MAGIC) ----
template <typename TS>
static __global__ void ink_dconv_f32(const float * __restrict__ x, const char * __restrict__ sx, const TS * __restrict__ st,
        const float * __restrict__ dw, float * __restrict__ out, const int w, const int n_tokens, const int K, const int d_conv,
        const int nkv, const int nst, const int64_t sx_s0, const int64_t sx_s1, const int64_t sx_s2, const int64_t st_s1,
        const float scale) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (c >= w || n >= n_tokens) {
        return;
    }
    const int s  = n / nst;
    const int t  = n - s*nst;
    const int h  = c / (w / nkv);
    const char * sxc = sx + c*sx_s1 + s*sx_s2;
    const float * dwn = dw + ((int64_t) n*nkv + h)*K;
    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const float xv = *(const float *) (sxc + (int64_t) (t + d_conv - lag)*sx_s0);
        acc += xv * ((float) st[lag*st_s1 + c] + dwn[lag]);
    }
    out[(int64_t) n*w + c] = scale * (x[(int64_t) n*w + c] + acc);
}

void ggml_cuda_op_ink_dconv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * sx = dst->src[1];
    const ggml_tensor * st = dst->src[2];
    const ggml_tensor * dw = dst->src[3];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && sx->type == GGML_TYPE_F32 && dw->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dw) && ggml_is_contiguous(dst));
    GGML_ASSERT(st->type == GGML_TYPE_F32 || st->type == GGML_TYPE_F16);
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int K = dst->op_params[9], d_conv = dst->op_params[10], nkv = dst->op_params[11], nst = dst->op_params[12];
    const int w = (int) x->ne[0];
    const int n_tokens = (int) (ggml_nelements(x) / w);
    GGML_ASSERT(w % nkv == 0 && n_tokens % nst == 0 && st->ne[0] == w && dw->ne[0] * dw->ne[1] * dw->ne[2] == (int64_t) K*nkv*n_tokens);
    const dim3 grid((w + 255)/256, n_tokens);
    cudaStream_t stream = ctx.stream();
    if (st->type == GGML_TYPE_F32) {
        ink_dconv_f32<float><<<grid, 256, 0, stream>>>((const float *) x->data, (const char *) sx->data, (const float *) st->data,
            (const float *) dw->data, (float *) dst->data, w, n_tokens, K, d_conv, nkv, nst, sx->nb[0], sx->nb[1], sx->nb[2],
            st->nb[1]/sizeof(float), scale);
    } else {
        ink_dconv_f32<half><<<grid, 256, 0, stream>>>((const float *) x->data, (const char *) sx->data, (const half *) st->data,
            (const float *) dw->data, (float *) dst->data, w, n_tokens, K, d_conv, nkv, nst, sx->nb[0], sx->nb[1], sx->nb[2],
            st->nb[1]/sizeof(half), scale);
    }
    CUDA_CHECK(cudaGetLastError());
}
