#include "kqmv.cuh"
#include "quantize.cuh"

// Q4_K block (144 B): half d, half dmin, uint8 scales[12], uint8 qs[128]
// Q5_K block (176 B): half d, half dmin, uint8 scales[12], uint8 qh[32], uint8 qs[128]
// 256 weights per block = 4 chunks of 64; chunk j: weight 64j+l (l<32) = low nibble of qs[32j+l] (sub-block 2j),
// weight 64j+32+l = high nibble (sub-block 2j+1); Q5_K adds bit 2j / 2j+1 of qh[l] as bit 4.
//
// Work split: a block of 256 threads owns one row. Lane groups of 8 lanes take one superblock per iteration; lane
// (j = (lane>>1)&3, hf = lane&1) handles 16 qs bytes = 16 weights of sub-block 2j and 16 of sub-block 2j+1.

#define KQMV_THREADS 256

static __device__ __forceinline__ void kqmv_scales(const int j, const uint32_t a, const uint32_t b, const uint32_t c,
        float & s0, float & m0, float & s1, float & m1) {
    const int k0 = 8*(2*(j & 1));
    const int k1 = k0 + 8;
    const uint32_t a0 = (a >> k0) & 0xFF, b0 = (b >> k0) & 0xFF, c0 = (c >> k0) & 0xFF;
    const uint32_t a1 = (a >> k1) & 0xFF, b1 = (b >> k1) & 0xFF, c1 = (c >> k1) & 0xFF;
    const bool hi = j >= 2;
    s0 = (float) (hi ? ((c0 & 0xF) | ((a0 >> 6) << 4)) : (a0 & 63));
    m0 = (float) (hi ? ((c0 >>  4) | ((b0 >> 6) << 4)) : (b0 & 63));
    s1 = (float) (hi ? ((c1 & 0xF) | ((a1 >> 6) << 4)) : (a1 & 63));
    m1 = (float) (hi ? ((c1 >>  4) | ((b1 >> 6) << 4)) : (b1 & 63));
}

template <bool Q5>
static __global__ void __launch_bounds__(KQMV_THREADS) kqmv_kernel(
        const char * __restrict__ W, const float * x, float * dst, const int nsb, const int64_t row_stride) {
    constexpr int BS = Q5 ? 176 : 144;
    extern __shared__ int4 kqmv_sm[];
    __shared__ float red[KQMV_THREADS/32];

    const int row = blockIdx.x;
    const int n16 = nsb*BS/16;
    const int4 * src = (const int4 *) (W + row*row_stride);
    for (int i = threadIdx.x; i < n16; i += KQMV_THREADS) {
        kqmv_sm[i] = __ldg(src + i);
    }
    // weights are constant: everything above may overlap the previous kernel; x and dst only after the wait
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sub  = lane >> 3;
    const int j    = (lane >> 1) & 3;
    const int hf   = lane & 1;

    float acc = 0.0f;
    for (int s = warp*4 + sub; s < nsb; s += KQMV_THREADS/8) {
        const uint8_t  * blk = (const uint8_t *) kqmv_sm + s*BS;
        const uint32_t * hw  = (const uint32_t *) blk;
        const uint32_t   dd  = hw[0];
        const float d  = __half2float(__ushort_as_half((unsigned short) (dd & 0xFFFF)));
        const float dm = __half2float(__ushort_as_half((unsigned short) (dd >> 16)));
        float s0, m0, s1, m1;
        kqmv_scales(j, hw[1], hw[2], hw[3], s0, m0, s1, m1);

        const int4 q4 = *(const int4 *) (blk + (Q5 ? 48 : 16) + 32*j + 16*hf);
        int4 qh4 = make_int4(0, 0, 0, 0);
        if constexpr (Q5) {
            qh4 = *(const int4 *) (blk + 16 + 16*hf);
        }
        const uint32_t qw[4] = {(uint32_t) q4.x,  (uint32_t) q4.y,  (uint32_t) q4.z,  (uint32_t) q4.w};
        const uint32_t hw4[4] = {(uint32_t) qh4.x, (uint32_t) qh4.y, (uint32_t) qh4.z, (uint32_t) qh4.w};
        const float * xb = x + s*256 + 64*j + 16*hf;

        float sl = 0.0f, sh = 0.0f, xsl = 0.0f, xsh = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float4 xl = *(const float4 *) (xb + 4*k);
            const float4 xh = *(const float4 *) (xb + 32 + 4*k);
            uint32_t lo =  qw[k]       & 0x0F0F0F0F;
            uint32_t hi = (qw[k] >> 4) & 0x0F0F0F0F;
            if constexpr (Q5) {
                lo |= ((hw4[k] >> (2*j))     & 0x01010101) << 4;
                hi |= ((hw4[k] >> (2*j + 1)) & 0x01010101) << 4;
            }
            sl += (float) (lo & 0xFF)*xl.x + (float) ((lo >> 8) & 0xFF)*xl.y + (float) ((lo >> 16) & 0xFF)*xl.z + (float) (lo >> 24)*xl.w;
            sh += (float) (hi & 0xFF)*xh.x + (float) ((hi >> 8) & 0xFF)*xh.y + (float) ((hi >> 16) & 0xFF)*xh.z + (float) (hi >> 24)*xh.w;
            xsl += xl.x + xl.y + xl.z + xl.w;
            xsh += xh.x + xh.y + xh.z + xh.w;
        }
        acc += d*(s0*sl + s1*sh) - dm*(m0*xsl + m1*xsh);
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }
    if (lane == 0) {
        red[warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < KQMV_THREADS/32; ++w) {
            a += red[w];
        }
        dst[row] = a;
    }
}

// NC = 1..8 columns (speculative verify) with q8_1 activations: the same staging, then per lane 16 qs bytes of sub-blocks
// 2j / 2j+1 as packed nibbles (Q5_K: plus the qh bit) against the matching 16 int8 of the q8_1 blocks, 8 dp4a a column.
// min term: d * dmin * m * (d8 * sum q8) from the q8_1 block's s = d8 * sum(qs), added once per sub-block (lane hf == 0).
template <bool Q5, int NC>
static __global__ void __launch_bounds__(KQMV_THREADS) kqmv8_kernel(
        const char * __restrict__ W, const block_q8_1 * y, float * dst, const int nsb, const int64_t row_stride,
        const int stride_col_y, const int stride_col_dst) {
    constexpr int BS = Q5 ? 176 : 144;
    extern __shared__ int4 kqmv_sm[];
    __shared__ float red[KQMV_THREADS/32][NC];

    const int row = blockIdx.x;
    const int n16 = nsb*BS/16;
    const int4 * src = (const int4 *) (W + row*row_stride);
    for (int i = threadIdx.x; i < n16; i += KQMV_THREADS) {
        kqmv_sm[i] = __ldg(src + i);
    }
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sub  = lane >> 3;
    const int j    = (lane >> 1) & 3;
    const int hf   = lane & 1;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    for (int s = warp*4 + sub; s < nsb; s += KQMV_THREADS/8) {
        const uint8_t  * blk = (const uint8_t *) kqmv_sm + s*BS;
        const uint32_t * hw  = (const uint32_t *) blk;
        const uint32_t   dd  = hw[0];
        const float d  = __half2float(__ushort_as_half((unsigned short) (dd & 0xFFFF)));
        const float dm = __half2float(__ushort_as_half((unsigned short) (dd >> 16)));
        float s0, m0, s1, m1;
        kqmv_scales(j, hw[1], hw[2], hw[3], s0, m0, s1, m1);
        const int4 q4 = *(const int4 *) (blk + (Q5 ? 48 : 16) + 32*j + 16*hf);
        int4 qh4 = make_int4(0, 0, 0, 0);
        if constexpr (Q5) {
            qh4 = *(const int4 *) (blk + 16 + 16*hf);
        }
        const int qw[4] = {q4.x, q4.y, q4.z, q4.w};
        const int hq[4] = {qh4.x, qh4.y, qh4.z, qh4.w};
        int lo[4], hi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            lo[k] =  qw[k]       & 0x0F0F0F0F;
            hi[k] = (qw[k] >> 4) & 0x0F0F0F0F;
            if constexpr (Q5) {
                lo[k] |= ((hq[k] >> (2*j))     & 0x01010101) << 4;
                hi[k] |= ((hq[k] >> (2*j + 1)) & 0x01010101) << 4;
            }
        }
        const float a0 = d*s0, a1 = d*s1, b0 = dm*m0, b1 = dm*m1;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1 * yb = y + c*stride_col_y + 8*s + 2*j;
            const int * p0 = (const int *) (yb[0].qs + 16*hf);   // q8_1 blocks are 36 B: 4 B alignment only
            const int * p1 = (const int *) (yb[1].qs + 16*hf);
            const int4 u0 = make_int4(p0[0], p0[1], p0[2], p0[3]);
            const int4 u1 = make_int4(p1[0], p1[1], p1[2], p1[3]);
            int dl = ggml_cuda_dp4a(lo[0], u0.x, 0);
            dl = ggml_cuda_dp4a(lo[1], u0.y, dl); dl = ggml_cuda_dp4a(lo[2], u0.z, dl); dl = ggml_cuda_dp4a(lo[3], u0.w, dl);
            int dh = ggml_cuda_dp4a(hi[0], u1.x, 0);
            dh = ggml_cuda_dp4a(hi[1], u1.y, dh); dh = ggml_cuda_dp4a(hi[2], u1.z, dh); dh = ggml_cuda_dp4a(hi[3], u1.w, dh);
            const float2 ds0 = __half22float2(yb[0].ds);
            const float2 ds1 = __half22float2(yb[1].ds);
            float v = a0*ds0.x*(float) dl + a1*ds1.x*(float) dh;
            if (hf == 0) {
                v -= b0*ds0.y + b1*ds1.y;
            }
            acc[c] += v;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float a = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, o);
        }
        if (lane == 0) {
            red[warp][c] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < NC) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < KQMV_THREADS/32; ++w) {
            a += red[w][threadIdx.x];
        }
        dst[threadIdx.x*stride_col_dst + row] = a;
    }
}

static bool kqmv_env() {
    static const bool v = [] { const char * e = getenv("GGML_KQMV"); return !e || atoi(e) != 0; }();
    return v;
}

// GGML_KQMV_NC (default 8, 0 = off): largest column count for the q8_1 kernel (2..8; pp4 75.8 -> 77.9 on TQ1b); GGML_KQMV_Q1=1: use the q8_1
// kernel for one column too (instead of the f32 one)
static int kqmv_nc_env() {
    static const int v = [] { const char * e = getenv("GGML_KQMV_NC"); return e ? atoi(e) : 8; }();
    return v;
}
static bool kqmv_q1_env() {
    static const bool v = [] { const char * e = getenv("GGML_KQMV_Q1"); return e && atoi(e) != 0; }();
    return v;
}

bool ggml_cuda_kqmv_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!kqmv_env()) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_DGX_SPARK) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ncols = src1->ne[1];
    if (ncols > 1 && (ncols > kqmv_nc_env() || ncols > 8 || dst->ne[1] != ncols)) {
        return false;
    }
    if (src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    if (ncols > 1 && (src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0)) {
        return false;
    }
    if (K % 256 != 0 || src1->ne[0] != K || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const size_t rb = ggml_row_size(src0->type, K);
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] % 16 != 0 || rb > 48*1024) {
        return false;
    }
    // x is read with float4 loads
    if (((uintptr_t) src1->data) % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_kqmv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K    = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int     nsb  = (int) (K / 256);
    const size_t  rb   = ggml_row_size(src0->type, K);
    const ggml_cuda_kernel_launch_params lp(dim3((unsigned) rows), dim3(KQMV_THREADS), rb, ctx.stream());
    const int ncols = (int) src1->ne[1];
    if (ncols > 1 || kqmv_q1_env()) {
        const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
        ggml_cuda_pool_alloc<char> q8(ctx.pool(), ncols * ne10_padded * sizeof(block_q8_1) / QK8_1);
        const int64_t s11 = src1->nb[1] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8.get(), src0->type, K, s11, s11*ncols, s11*ncols,
                               ne10_padded, ncols, 1, 1, ctx.stream());
        const int scy = (int) (ne10_padded / QK8_1);
        const int scd = (int) (dst->nb[1] / sizeof(float));
        const block_q8_1 * yq = (const block_q8_1 *) q8.get();
        float * dd = (float *) dst->data;
        const char * w = (const char *) src0->data;
        const int64_t rs = (int64_t) src0->nb[1];
#define KQMV8(Q, N) case N: ggml_cuda_kernel_launch(kqmv8_kernel<Q, N>, lp, w, yq, dd, nsb, rs, scy, scd); break;
        if (src0->type == GGML_TYPE_Q5_K) {
            switch (ncols) { KQMV8(true, 1) KQMV8(true, 2) KQMV8(true, 3) KQMV8(true, 4) KQMV8(true, 5) KQMV8(true, 6) KQMV8(true, 7) KQMV8(true, 8) default: GGML_ABORT("kqmv ncols"); }
        } else {
            switch (ncols) { KQMV8(false, 1) KQMV8(false, 2) KQMV8(false, 3) KQMV8(false, 4) KQMV8(false, 5) KQMV8(false, 6) KQMV8(false, 7) KQMV8(false, 8) default: GGML_ABORT("kqmv ncols"); }
        }
#undef KQMV8
        return;
    }
    if (src0->type == GGML_TYPE_Q5_K) {
        ggml_cuda_kernel_launch(kqmv_kernel<true>, lp, (const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, nsb, (int64_t) src0->nb[1]);
    } else {
        ggml_cuda_kernel_launch(kqmv_kernel<false>, lp, (const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, nsb, (int64_t) src0->nb[1]);
    }
}
