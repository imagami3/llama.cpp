// ink2-tq fork: B70 trellis tq2 experts on CUDA (see tq.cuh). Decode kernel checked against B70's tq_ref.py
// (~/gb10lab/trellis/kern/tq_test.cu: max rel err 5e-7, 181-194 GB/s on GB10).
#include "tq.cuh"
#include "rvq.cuh"   // GGML_CUDA_RVQ_GROUP_MAGIC

#include <mma.h>
#include <vector>
#include <chrono>

#define TQ_TILE_B 66
#define TQ_NW     4      // warps (strips) per decode block

static __device__ __forceinline__ uint32_t tq_bsum(const uint32_t s) {
    const uint32_t v = s * 34038481u + 76625530u;
    return __dp4a(v, 0x01010101u, 0u);
}

// sum of (unsigned byte of a) * (signed byte of b) + c
static __device__ __forceinline__ int tq_dp4a_us(const uint32_t a, const int b, const int c) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// (float) b for 0 <= b < 2^23 without I2F (quarter rate): the bits of 2^23 + b, minus 2^23 (exact)
static __device__ __forceinline__ float tq_u2f(const uint32_t b) {
    return __int_as_float(0x4B000000u | b) - 8388608.0f;
}

// half2 (b0 - 510, b1 - 510) for 0 <= b < 1024 without conversions: half bits 0x6400 | b = 1024 + b (exact), minus 1534
static __device__ __forceinline__ half2 tq_h2(const uint32_t b0, const uint32_t b1) {
    const uint32_t u = (0x64006400u | b0) | (b1 << 16);
    return __hsub2(*(const half2 *) &u, __half2half2(__float2half(1534.0f)));
}

// fused down-projection input (op_params[13] == 'GLU1', as in rvq.cu): x = cg*tanh(g/cg) * sigmoid(g) * cu*tanh(u/cu)
static __device__ __forceinline__ float tq_act(const float g, const float u, const float cg, const float cu) {
    return (cg * tanhf(g / cg)) * (1.0f / (1.0f + expf(-g))) * (cu * tanhf(u / cu));
}

// x' = H64(sign * x) / 8 for every 64-block of every input row; xsum[row] = sum(x'). One warp per 64-block.
// rows = n_tok * ne11 input rows (row = t * ne11 + j), read with strides nb11 / nb12. xu (same layout as x): GLU input.
static __global__ void tq_rotate(const char * __restrict__ x, const float * __restrict__ sign, float * __restrict__ xr,
                                 float * __restrict__ xsum, const int K, const int ne11, const size_t nb11, const size_t nb12,
                                 const char * __restrict__ xu = nullptr, const float cg = 1.0f, const float cu = 1.0f) {
    const int row  = blockIdx.y;
    const int b    = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b * 64 >= K) {
        return;
    }
    const size_t off = (row / ne11) * nb12 + (row % ne11) * nb11;
    const float * xi = (const float *) (x + off);
    float v0 = xi[b*64 + lane], v1 = xi[b*64 + 32 + lane];
    if (xu) {
        const float * ui = (const float *) (xu + off);
        v0 = tq_act(v0, ui[b*64 + lane], cg, cu);
        v1 = tq_act(v1, ui[b*64 + 32 + lane], cg, cu);
    }
    float a0 = v0 * sign[b*64 + lane];
    float a1 = v1 * sign[b*64 + 32 + lane];
    const float u = a0 + a1, v = a0 - a1;
    a0 = u; a1 = v;
#pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        const float o0 = __shfl_xor_sync(0xffffffff, a0, h);
        const float o1 = __shfl_xor_sync(0xffffffff, a1, h);
        const bool hi = lane & h;
        a0 = hi ? o0 - a0 : a0 + o0;
        a1 = hi ? o1 - a1 : a1 + o1;
    }
    a0 *= 0.125f;
    a1 *= 0.125f;
    float * xo = xr + (size_t) row * K;
    xo[b*64 + lane]      = a0;
    xo[b*64 + 32 + lane] = a1;
    float s = a0 + a1;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
    }
    if (xsum && lane == 0) {
        atomicAdd(xsum + row, s);
    }
}

// decode: block = TQ_NW strips (16 rows each) of one (token, slot) pair; each warp walks its strip's tiles, 32 at a time
// staged in shared memory. dst[t][s][r] = gain[r] / 147.8 * (sum_c bsum(s_rc) x'[c] - 510 * sum(x')); zeros when the
// slot's expert is not in [ida, ida + n_expert).
static __global__ void __launch_bounds__(TQ_NW * 32)
tq_mmv(const uint8_t * __restrict__ W, const half * __restrict__ gain, const float * __restrict__ xr, const float * __restrict__ xsum,
       const char * __restrict__ ids, char * __restrict__ dst, const int K, const int rows, const int n_expert, const int n_used,
       const int ne11, const size_t nb02, const size_t ids_nb0, const size_t ids_nb1, const size_t nb1, const size_t nb2, const int ida) {
    __shared__ uint32_t sm[TQ_NW][32 * TQ_TILE_B / 4 + 2];
    extern __shared__ float xs[];   // K floats
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int p = blockIdx.y, t = p / n_used, s = p % n_used;
    const int strip = blockIdx.x * TQ_NW + warp;
    float * y = (float *) (dst + t * nb2 + s * nb1);
    const uint32_t e = (uint32_t) (*(const int32_t *) (ids + t * ids_nb1 + s * ids_nb0) - ida);
    if (e >= (uint32_t) n_expert) {
        if (strip * 16 < rows && lane < 16) {
            y[strip * 16 + lane] = 0.0f;
        }
        return;
    }
    const int xrow = t * ne11 + s % ne11;
    const float * xg = xr + (size_t) xrow * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        xs[i] = xg[i];
    }
    __syncthreads();
    if (strip * 16 >= rows) {
        return;
    }
    const int nc = K / 16;
    const uint8_t * Ws = W + (size_t) e * nb02 + (size_t) strip * nc * TQ_TILE_B;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        acc[r] = 0.0f;
    }
    for (int j0 = 0; j0 < nc; j0 += 32) {
        const int nt = min(32, nc - j0);
        // the strip base and 32 tiles (2112 B) are 8-byte aligned for every width used (K multiple of 64)
        const uint2 * src = (const uint2 *) (Ws + (size_t) j0 * TQ_TILE_B);
        const int n8 = (nt * TQ_TILE_B + 7) / 8;
        __syncwarp();
        for (int i = lane; i < n8; i += 32) {
            const uint2 q = src[i];
            sm[warp][2*i] = q.x;
            sm[warp][2*i + 1] = q.y;
        }
        __syncwarp();
        if (lane < nt) {
            const int off = TQ_TILE_B * lane;
            const uint32_t * tw = sm[warp] + (off >> 2);
            uint32_t s0, SB[16];
            if (off & 2) {
                s0 = tw[0] >> 16;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    SB[k] = tw[k + 1];
                }
            } else {
                s0 = tw[0] & 0xffff;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                }
            }
            s0 &= 0x3fff;
            const float * xc = xs + (j0 + lane) * 16;
            float xv[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                xv[c] = xc[c];
            }
            // stream words: ST[0] = s0 | SB[0] << 14, ST[k] = SB[k-1] >> 18 | SB[k] << 14; row r uses ST[r], ST[r+1]
            uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                float a = 0.0f;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    a = fmaf((float) tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu), xv[c], a);
                }
                acc[r] += a;
                lo = hi;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        acc[r] = v;
    }
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r == lane) {
                v = acc[r];
            }
        }
        const int row = strip * 16 + lane;
        y[row] = __half2float(gain[(size_t) e * rows + row]) * (1.0f / 147.8f) * (v - 510.0f * xsum[xrow]);
    }
}

// decode v2: as tq_mmv, but each block rotates its own input row into shared memory (x' = H64(sign * x) / 8, sum(x')),
// so a call is one kernel (no tq_rotate, memset or xr buffer). The input row is read from L2 by every block of the slot.
static __global__ void __launch_bounds__(TQ_NW * 32)
tq_mmv2(const uint8_t * __restrict__ W, const half * __restrict__ gain, const char * __restrict__ x, const float * __restrict__ sign,
        const char * __restrict__ ids, char * __restrict__ dst, const int K, const int rows, const int n_expert, const int n_used,
        const int ne11, const size_t nb02, const size_t ids_nb0, const size_t ids_nb1, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const int ida) {
    __shared__ uint32_t sm[TQ_NW][32 * TQ_TILE_B / 4 + 2];
    __shared__ float wsum[TQ_NW];
    extern __shared__ float xs[];   // K floats
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int p = blockIdx.y, t = p / n_used, s = p % n_used;
    const int strip = blockIdx.x * TQ_NW + warp;
    float * y = (float *) (dst + t * nb2 + s * nb1);
    const uint32_t e = (uint32_t) (*(const int32_t *) (ids + t * ids_nb1 + s * ids_nb0) - ida);
    if (e >= (uint32_t) n_expert) {
        if (strip * 16 < rows && lane < 16) {
            y[strip * 16 + lane] = 0.0f;
        }
        return;
    }
    const float * xi = (const float *) (x + t * nb12 + (s % ne11) * nb11);
    float part = 0.0f;
    for (int b = warp; b < K / 64; b += TQ_NW) {
        float a0 = xi[b*64 + lane] * sign[b*64 + lane];
        float a1 = xi[b*64 + 32 + lane] * sign[b*64 + 32 + lane];
        const float u = a0 + a1, v = a0 - a1;
        a0 = u; a1 = v;
#pragma unroll
        for (int h = 16; h >= 1; h >>= 1) {
            const float o0 = __shfl_xor_sync(0xffffffff, a0, h);
            const float o1 = __shfl_xor_sync(0xffffffff, a1, h);
            const bool hi = lane & h;
            a0 = hi ? o0 - a0 : a0 + o0;
            a1 = hi ? o1 - a1 : a1 + o1;
        }
        a0 *= 0.125f;
        a1 *= 0.125f;
        xs[b*64 + lane]      = a0;
        xs[b*64 + 32 + lane] = a1;
        part += a0 + a1;
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        part += __shfl_xor_sync(0xffffffff, part, o);
    }
    if (lane == 0) {
        wsum[warp] = part;
    }
    __syncthreads();
    float xsum = 0.0f;
#pragma unroll
    for (int w = 0; w < TQ_NW; ++w) {
        xsum += wsum[w];
    }
    if (strip * 16 >= rows) {
        return;
    }
    const int nc = K / 16;
    const uint8_t * Ws = W + (size_t) e * nb02 + (size_t) strip * nc * TQ_TILE_B;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        acc[r] = 0.0f;
    }
    for (int j0 = 0; j0 < nc; j0 += 32) {
        const int nt = min(32, nc - j0);
        const uint2 * src = (const uint2 *) (Ws + (size_t) j0 * TQ_TILE_B);
        const int n8 = (nt * TQ_TILE_B + 7) / 8;
        __syncwarp();
        for (int i = lane; i < n8; i += 32) {
            const uint2 q = src[i];
            sm[warp][2*i] = q.x;
            sm[warp][2*i + 1] = q.y;
        }
        __syncwarp();
        if (lane < nt) {
            const int off = TQ_TILE_B * lane;
            const uint32_t * tw = sm[warp] + (off >> 2);
            uint32_t s0, SB[16];
            if (off & 2) {
                s0 = tw[0] >> 16;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    SB[k] = tw[k + 1];
                }
            } else {
                s0 = tw[0] & 0xffff;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                }
            }
            s0 &= 0x3fff;
            const float * xc = xs + (j0 + lane) * 16;
            float xv[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                xv[c] = xc[c];
            }
            uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                float a = 0.0f;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    a = fmaf((float) tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu), xv[c], a);
                }
                acc[r] += a;
                lo = hi;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        acc[r] = v;
    }
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r == lane) {
                v = acc[r];
            }
        }
        const int row = strip * 16 + lane;
        y[row] = __half2float(gain[(size_t) e * rows + row]) * (1.0f / 147.8f) * (v - 510.0f * xsum);
    }
}

// decode v3: block = 4 warps = SPB strips x KS K-parts (KS = 4 / SPB). Warp (strip, part) walks a contiguous range of
// tiles (multiple of 4 tiles, so 8-byte aligned) in chunks of 32 staged in smem; x' read from xr (rotated by tq_rotate,
// no xsum: each lane keeps sum(x') of its tiles, so y = gain / 147.8 * (sum bsum x' - 510 sum x')). More warps per row
// than v1/v2 when a call has only one or two experts.
template <int KS>
static __global__ void __launch_bounds__(128)
tq_mmv3(const uint8_t * __restrict__ W, const half * __restrict__ gain, const float * __restrict__ xr,
        const char * __restrict__ ids, char * __restrict__ dst, const int K, const int rows, const int n_expert, const int n_used,
        const int ne11, const size_t nb02, const size_t ids_nb0, const size_t ids_nb1, const size_t nb1, const size_t nb2, const int ida,
        const int zf, const char * __restrict__ prev) {
    constexpr int SPB = 4 / KS;
    // zf: grid x = (token, slot) pair, y = strip block, so the blocks of one strip block run together (same expert of
    // several tokens re-read from L2)
    const int bx = zf ? blockIdx.y : blockIdx.x, by = zf ? blockIdx.x : blockIdx.y;
    __shared__ uint32_t sm[4][32 * TQ_TILE_B / 4 + 2];
    __shared__ float red[4][17];
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int p = by, t = p / n_used, s = p % n_used;
    const int sl = warp / KS, part = warp % KS;
    const int strip = bx * SPB + sl;
    float * y = (float *) (dst + t * nb2 + s * nb1);
    const uint32_t e = (uint32_t) (*(const int32_t *) (ids + t * ids_nb1 + s * ids_nb0) - ida);
    if (e >= (uint32_t) n_expert) {   // not this group's slot: zero, or the earlier groups' sum (prev, same layout as dst)
        if (part == 0 && strip * 16 < rows && lane < 16) {
            y[strip * 16 + lane] = prev ? ((const float *) (prev + t * nb2 + s * nb1))[strip * 16 + lane] : 0.0f;
        }
        return;
    }
    const int nc = K / 16;
    const int L  = (((nc + KS - 1) / KS) + 3) & ~3;
    const int jb = min(nc, part * L), je = min(nc, jb + L);
    const float * xg = xr + (size_t) (t * ne11 + s % ne11) * K;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        acc[r] = 0.0f;
    }
    float sx = 0.0f;
    if (strip * 16 < rows) {
        const uint8_t * Ws = W + (size_t) e * nb02 + (size_t) strip * nc * TQ_TILE_B;
        for (int j0 = jb; j0 < je; j0 += 32) {
            const int nt = min(32, je - j0);
            const uint2 * src = (const uint2 *) (Ws + (size_t) j0 * TQ_TILE_B);
            const int n8 = (nt * TQ_TILE_B + 7) / 8;
            __syncwarp();
            for (int i = lane; i < n8; i += 32) {
                const uint2 q = src[i];
                sm[warp][2*i] = q.x;
                sm[warp][2*i + 1] = q.y;
            }
            __syncwarp();
            if (lane < nt) {
                const float4 * xc4 = (const float4 *) (xg + (j0 + lane) * 16);
                float xv[16];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 q = __ldg(xc4 + c);
                    xv[4*c] = q.x; xv[4*c + 1] = q.y; xv[4*c + 2] = q.z; xv[4*c + 3] = q.w;
                }
                const int off = TQ_TILE_B * lane;
                const uint32_t * tw = sm[warp] + (off >> 2);
                uint32_t s0, SB[16];
                if (off & 2) {
                    s0 = tw[0] >> 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = tw[k + 1];
                    }
                } else {
                    s0 = tw[0] & 0xffff;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                    }
                }
                s0 &= 0x3fff;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    sx += xv[c];
                }
                uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                    const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                    float a = 0.0f;
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        a = fmaf((float) tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu), xv[c], a);
                    }
                    acc[r] += a;
                    lo = hi;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        acc[r] = v;
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        sx += __shfl_xor_sync(0xffffffff, sx, o);
    }
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r == lane) {
                v = acc[r];
            }
        }
        red[warp][lane] = v;
    }
    if (lane == 16) {
        red[warp][16] = sx;
    }
    __syncthreads();
    if (threadIdx.x < SPB * 16) {
        const int ls = threadIdx.x / 16, r = threadIdx.x % 16;
        const int st = bx * SPB + ls;
        if (st * 16 < rows) {
            float v = 0.0f, sxt = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v   += red[ls * KS + k][r];
                sxt += red[ls * KS + k][16];
            }
            const int row = st * 16 + r;
            y[row] = __half2float(gain[(size_t) e * rows + row]) * (1.0f / 147.8f) * (v - 510.0f * sxt);
        }
    }
}

// prefill: W_rot (f16, row-major [rows][K], gain applied). One thread per (tile, tile row): 16 weights.
static __global__ void tq_dequant_f16(const uint8_t * __restrict__ W, const half * __restrict__ gain, half * __restrict__ w,
                                      const int K, const int rows) {
    const int nc = K / 16;
    const int64_t n = (int64_t) (rows / 16) * nc * 16;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int r = i & 15;
        const int64_t tile = i >> 4;
        const int strip = tile / nc, j = tile % nc;
        const uint16_t * tb = (const uint16_t *) (W + (size_t) tile * TQ_TILE_B);   // 2-byte aligned
        const uint32_t s0 = tb[0] & 0x3fff;
        // SB[k] = symbol word k = bytes 2+4k .. 5+4k
        auto SB = [&](const int k) -> uint32_t { return (uint32_t) tb[1 + 2*k] | ((uint32_t) tb[2 + 2*k] << 16); };
        const uint32_t lo = r == 0 ? (s0 | (SB(0) << 14)) : ((SB(r - 1) >> 18) | (SB(r) << 14));
        const uint32_t hi = (SB(r) >> 18) | (r < 15 ? (SB(r + 1) << 14) : 0u);
        const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
        const int row = strip * 16 + r;
        const float g = __half2float(gain[row]) * (1.0f / 147.8f);
        half2 o[8];
#pragma unroll
        for (int c = 0; c < 16; c += 2) {
            const float w0 = g * ((float) tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu) - 510.0f);
            const float w1 = g * ((float) tq_bsum((uint32_t) (ch >> (2*c + 2)) & 0x3fffu) - 510.0f);
            o[c/2] = __floats2half2_rn(w0, w1);
        }
        uint4 * dp = (uint4 *) (w + (size_t) row * K + j * 16);
        dp[0] = *(const uint4 *) &o[0];
        dp[1] = *(const uint4 *) &o[4];
    }
}

// prefill: x' (GLU when xu, then H64(sign * x) / 8) of the sorted pairs only, written as f16: xs[j][k]. One warp per 64-block.
static __global__ void tq_rotate_gather_f16(const char * __restrict__ x, const char * __restrict__ xu, const float cg, const float cu,
                                            const float * __restrict__ sign, const int32_t * __restrict__ pairs, half * __restrict__ xs,
                                            const int K, const int n_used, const int ne11, const size_t nb11, const size_t nb12) {
    const int j = blockIdx.y;
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b * 64 >= K) {
        return;
    }
    const int p = pairs[j], t = p / n_used, s = p % n_used;
    const size_t off = t * nb12 + (s % ne11) * nb11;
    const float * xi = (const float *) (x + off);
    float v0 = xi[b*64 + lane], v1 = xi[b*64 + 32 + lane];
    if (xu) {
        const float * ui = (const float *) (xu + off);
        v0 = tq_act(v0, ui[b*64 + lane], cg, cu);
        v1 = tq_act(v1, ui[b*64 + 32 + lane], cg, cu);
    }
    float a0 = v0 * sign[b*64 + lane];
    float a1 = v1 * sign[b*64 + 32 + lane];
    const float u = a0 + a1, v = a0 - a1;
    a0 = u; a1 = v;
#pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        const float o0 = __shfl_xor_sync(0xffffffff, a0, h);
        const float o1 = __shfl_xor_sync(0xffffffff, a1, h);
        const bool hi = lane & h;
        a0 = hi ? o0 - a0 : a0 + o0;
        a1 = hi ? o1 - a1 : a1 + o1;
    }
    half * xo = xs + (size_t) j * K;
    xo[b*64 + lane]      = __float2half(a0 * 0.125f);
    xo[b*64 + 32 + lane] = __float2half(a1 * 0.125f);
}

// gate + up of one group (same input rows, same pairs): x read once, rotated with both signs, two f16 outputs
static __global__ void tq_rotate_gather2_f16(const char * __restrict__ x, const float * __restrict__ sa, const float * __restrict__ sb,
                                             const int32_t * __restrict__ pairs, half * __restrict__ xa, half * __restrict__ xb,
                                             const int K, const int n_used, const int ne11, const size_t nb11, const size_t nb12) {
    const int j = blockIdx.y;
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b * 64 >= K) {
        return;
    }
    const int p = pairs[j], t = p / n_used, s = p % n_used;
    const float * xi = (const float *) (x + t * nb12 + (s % ne11) * nb11);
    const float v0 = xi[b*64 + lane], v1 = xi[b*64 + 32 + lane];
    float a0 = v0 * sa[b*64 + lane], a1 = v1 * sa[b*64 + 32 + lane];
    float c0 = v0 * sb[b*64 + lane], c1 = v1 * sb[b*64 + 32 + lane];
    { const float u = a0 + a1, v = a0 - a1; a0 = u; a1 = v; }
    { const float u = c0 + c1, v = c0 - c1; c0 = u; c1 = v; }
#pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        const float o0 = __shfl_xor_sync(0xffffffff, a0, h), o1 = __shfl_xor_sync(0xffffffff, a1, h);
        const float q0 = __shfl_xor_sync(0xffffffff, c0, h), q1 = __shfl_xor_sync(0xffffffff, c1, h);
        const bool hi = lane & h;
        a0 = hi ? o0 - a0 : a0 + o0;  a1 = hi ? o1 - a1 : a1 + o1;
        c0 = hi ? q0 - c0 : c0 + q0;  c1 = hi ? q1 - c1 : c1 + q1;
    }
    half * oa = xa + (size_t) j * K, * ob = xb + (size_t) j * K;
    oa[b*64 + lane] = __float2half(a0 * 0.125f);  oa[b*64 + 32 + lane] = __float2half(a1 * 0.125f);
    ob[b*64 + lane] = __float2half(c0 * 0.125f);  ob[b*64 + 32 + lane] = __float2half(c1 * 0.125f);
}

static __global__ void tq_scatter(const float * __restrict__ ys, const int32_t * __restrict__ pairs, char * __restrict__ dst,
                                  const int rows, const int n_used, const size_t nb1, const size_t nb2) {
    const int j = blockIdx.x;
    const int p = pairs[j];
    const int t = p / n_used, s = p % n_used;
    float * y = (float *) (dst + t * nb2 + s * nb1);
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        y[r] = ys[(int64_t) j * rows + r];
    }
}

// GGML_TQ_PF_DBG (timing only, wrong results): 1 skips the tile decode, 2 skips the MMAs
static int tq_knob_pf_dbg() {
    static const int v = [] { const char * e = getenv("GGML_TQ_PF_DBG"); return e ? atoi(e) : 0; }();
    return v;
}

// GGML_TQ_PF_NMAX: largest token tile of the fused prefill (128, or 64: experts with more tokens get tiles of 64)
static int tq_pf_nmax() {
    static const int v = [] { const char * e = getenv("GGML_TQ_PF_NMAX"); return e ? atoi(e) : 32; }();   // pp512 on TQ1: 128 636, 64 700, 32 714 (smaller tiles: more blocks a SM)
    return v;
}

// Host data of the prefill H2D copies. On GB10 the GPU can read pageable host memory, and a cudaMemcpyAsync from it may
// run after the call returns, so the source must stay alive until the stream has passed the copy: every copy source goes
// here, and the store is cleared only right after a cudaStreamSynchronize (at the start of the next prefill call).
// (Freeing the vectors right after the async copy changed TQ1 perplexity 3.2973 -> 3.8406 in the per-layer path.)
static std::vector<std::vector<char>> & tq_keep_store() {
    static std::vector<std::vector<char>> s;
    return s;
}
static const void * tq_keep(const void * p, const size_t n) {
    auto & s = tq_keep_store();
    s.emplace_back((const char *) p, (const char *) p + n);
    return s.back().data();
}
static void tq_sync_and_release(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tq_keep_store().clear();
}

// prefill, fused (few tokens per expert): 64 rows (4 strips) x N tokens a block, K in steps of 128 (8 tiles a strip; 64 at
// a K % 128 tail). The step's 4 x 8 tiles are staged in smem, each thread decodes 4 (tile, tile row) units of 16 weights
// to f16 (bsum - 510: integers, exact), wmma f16 x f16 -> f32; gain / 147.8 applied at the end.
#define TQ_PF_M  64
// GPU-sorted multi prefill (GGML_TQ_GSORT): int32 meta written by tq_gsort
#define TQM_GCNT 0    // [8] pairs of group g
#define TQM_GOFF 8    // [8] first pair of group g in the sorted pairs
#define TQM_ICNT 16   // [8] items (32-token tiles) of group g
#define TQM_IOFF 24   // [8] first item of group g
#define TQM_XOFF 32   // [8] element offset of node m's rows in xs
#define TQM_TOT  40   // pairs in all groups
#define TQM_XOFF2 48  // [8] element offset of down node m's rows in the down xs (GGML_TQ_GLUF: written by the up kernel)
#define TQM_ICNT64 56 // [8] items of 64 tokens of group g (GGML_TQ_PF_N64: experts with more than 32 tokens)
#define TQM_IOFF64 64 // [8] first such item of group g (absolute index into items)
#define TQM_N    72

// GGML_TQ_GLUF: the up (or gate) kernel of a group also makes the down kernel's input for its rows:
// xd[XOFF2[mi] + local*Kd + row] = H64(sign * act(gate, up)) / 8, the down gather's exact arithmetic
struct tq_glu_args {
    half        * xd = nullptr;
    const char  * other = nullptr;   // the group's other gate/up output (written by an earlier kernel)
    size_t        onb1 = 0, onb2 = 0;
    const float * sign = nullptr;    // the down node's signs [Kd]
    float         cg = 1.0f, cu = 1.0f;
    int           Kd = 0, mi = 0;
    int           self_is_u = 0;     // this kernel's value is act's second argument
};
#define TQ_PF_K  128
#define TQ_PF_LD (TQ_PF_K + 8)
#define TQ_PF_SEG (8 * TQ_TILE_B)   // 528 B: one strip's tiles of one step

template <int N, bool MAGIC>
static __global__ void __launch_bounds__(128)
tq_prefill_mma(const uint8_t * __restrict__ W, const half * __restrict__ gain, const half * __restrict__ xs,
               const int4 * __restrict__ items, const int32_t * __restrict__ pairs, char * __restrict__ dst, const int K, const int rows,
               const size_t nb02, const int n_used, const size_t nb1, const size_t nb2, const int dbg,
               const int * __restrict__ meta = nullptr, const int mg = 0, const int mm = 0, const int tokx = 0,
               float * __restrict__ red = nullptr, const float * __restrict__ rw = nullptr, const float * __restrict__ rsc = nullptr,
               const tq_glu_args glu = tq_glu_args()) {
    using namespace nvcuda;
    extern __shared__ __align__(32) char smem_tq[];
    uint8_t * sT = (uint8_t *) smem_tq;                          // 4 x 528 B
    float   * sg = (float *) (smem_tq + 4 * TQ_PF_SEG);          // [64]
    half    * sA = (half *) (sg + TQ_PF_M);                      // [64][LD]
    half    * sB = sA + TQ_PF_M * TQ_PF_LD;                      // [N][LD]
    float   * sC = (float *) sA;                                 // [N][64], reuses sA / sB after the last K step
    static_assert(TQ_PF_M * N * 4 <= (TQ_PF_M + N) * TQ_PF_LD * 2, "sC must fit in sA + sB");

    int4 it;
    if (meta) {   // GPU-sorted: grid.y is an upper bound of the group's items
        constexpr int ICNT_F = N == 64 ? TQM_ICNT64 : TQM_ICNT, IOFF_F = N == 64 ? TQM_IOFF64 : TQM_IOFF;
        if ((int) blockIdx.y >= meta[ICNT_F + mg]) {
            return;
        }
        it = items[meta[IOFF_F + mg] + blockIdx.y];
        pairs += meta[TQM_GOFF + mg];
        if (!tokx) {   // tokx = n_used: xs holds one row per token (gate / up input), row of pair p = p / n_used
            xs += meta[TQM_XOFF + mm];
        }
    } else {
        it = items[blockIdx.y];
    }
    const int e = it.x, j0 = it.y, nt = it.z;
    const int r0 = blockIdx.x * TQ_PF_M;
    const int tid = threadIdx.x, warp = tid / 32;
    const int nc = K / 16;

    if (tid < TQ_PF_M) {
        sg[tid] = __half2float(gain[(size_t) e * rows + r0 + tid]) * (1.0f / 147.8f);
    }

    constexpr int NF = N / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        wmma::fill_fragment(acc[f], 0.0f);
    }

    const uint8_t * Wb = W + (size_t) e * nb02 + (size_t) (r0 / 16) * nc * TQ_TILE_B;   // first strip of this row tile
    constexpr int NT = (4 * TQ_PF_SEG / 8 + 127) / 128;   // uint2 of tiles a thread (3)
    constexpr int NB = N * (TQ_PF_K / 8) / 128;           // uint4 of B a thread
    int brow[NB];
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int n = (tid + i * 128) / (TQ_PF_K / 8);
        brow[i] = n < nt ? (tokx ? pairs[j0 + n] / tokx : j0 + n) : 0;
    }
    auto load_step = [&](const int k0, uint2 (&ct)[NT], uint4 (&cb)[NB]) {
        const int jt0 = k0 / 16;
        const int nck = min(8, nc - jt0);
        const int seg8 = nck * TQ_TILE_B / 8;   // uint2 a strip this step (66 or 33)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int q = tid + i * 128;
            const int sl = q / (TQ_PF_SEG / 8), w = q % (TQ_PF_SEG / 8);
            ct[i] = make_uint2(0, 0);
            if (sl < 4 && w < seg8) {
                ct[i] = __ldg((const uint2 *) (Wb + (size_t) sl * nc * TQ_TILE_B + (size_t) jt0 * TQ_TILE_B) + w);
            }
        }
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int q = tid + i * 128;
            const int n = q / (TQ_PF_K / 8), c = q % (TQ_PF_K / 8);
            cb[i] = make_uint4(0, 0, 0, 0);
            if (n < nt && c < nck * 2) {
                cb[i] = *(const uint4 *) (xs + (size_t) brow[i] * K + k0 + c * 8);
            }
        }
    };
    uint2 ct[NT];
    uint4 cb[NB];
    load_step(0, ct, cb);
    for (int k0 = 0; k0 < K; k0 += TQ_PF_K) {
        const int nck = min(8, nc - k0 / 16);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int q = tid + i * 128;
            if (q < 4 * TQ_PF_SEG / 8) {
                ((uint2 *) sT)[q] = ct[i];
            }
        }
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int q = tid + i * 128;
            const int n = q / (TQ_PF_K / 8), c = q % (TQ_PF_K / 8);
            *(uint4 *) (sB + n * TQ_PF_LD + c * 8) = cb[i];
        }
        __syncthreads();
        if (k0 + TQ_PF_K < K) {
            load_step(k0 + TQ_PF_K, ct, cb);   // in flight during the decode and the MMAs
        }
        // decode: unit u = (strip sl, tile jt, tile row r)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = tid + i * 128;
            const int sl = u >> 7, jt = (u >> 4) & 7, r = u & 15;
            if (jt < nck && dbg != 1) {
                const uint16_t * tb = (const uint16_t *) (sT + sl * TQ_PF_SEG + jt * TQ_TILE_B);
                auto SB = [&](const int k) -> uint32_t { return (uint32_t) tb[1 + 2*k] | ((uint32_t) tb[2 + 2*k] << 16); };
                const uint32_t sbr = SB(r);
                const uint32_t lo = r == 0 ? ((tb[0] & 0x3fffu) | (sbr << 14)) : ((SB(r - 1) >> 18) | (sbr << 14));
                const uint32_t hi = (sbr >> 18) | (r < 15 ? (SB(r + 1) << 14) : 0u);
                const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                half2 o[8];
#pragma unroll
                for (int c = 0; c < 16; c += 2) {
                    const uint32_t b0 = tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu);
                    const uint32_t b1 = tq_bsum((uint32_t) (ch >> (2*c + 2)) & 0x3fffu);
                    o[c/2] = MAGIC ? tq_h2(b0, b1) : __floats2half2_rn((float) ((int) b0 - 510), (float) ((int) b1 - 510));
                }
                uint4 * dp = (uint4 *) (sA + (sl * 16 + r) * TQ_PF_LD + jt * 16);
                dp[0] = *(const uint4 *) &o[0];
                dp[1] = *(const uint4 *) &o[4];
            }
        }
        __syncthreads();
        for (int kk = 0; kk < (dbg == 2 ? 0 : nck * 16); kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, sA + (warp * 16) * TQ_PF_LD + kk, TQ_PF_LD);
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sB + (f * 16) * TQ_PF_LD + kk, TQ_PF_LD);
                wmma::mma_sync(acc[f], fa, fb, acc[f]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        wmma::store_matrix_sync(sC + (f * 16) * TQ_PF_M + warp * 16, acc[f], TQ_PF_M, wmma::mem_col_major);
    }
    __syncthreads();
    if (glu.xd) {   // one warp per token column: the 64 rows of this tile are one H64 block of the down input
        const int lane = tid & 31;
        half * xd = glu.xd + meta[TQM_XOFF2 + glu.mi];
        for (int n = warp; n < nt; n += blockDim.x / 32) {
            const int pr = pairs[j0 + n];
            const float * o = (const float *) (glu.other + (pr / n_used) * glu.onb2 + (pr % n_used) * glu.onb1) + r0;
            const float s0 = sC[n * TQ_PF_M + lane] * sg[lane], s1 = sC[n * TQ_PF_M + 32 + lane] * sg[32 + lane];
            const float o0 = o[lane], o1 = o[32 + lane];
            const float v0 = glu.self_is_u ? tq_act(o0, s0, glu.cg, glu.cu) : tq_act(s0, o0, glu.cg, glu.cu);
            const float v1 = glu.self_is_u ? tq_act(o1, s1, glu.cg, glu.cu) : tq_act(s1, o1, glu.cg, glu.cu);
            float a0 = v0 * glu.sign[r0 + lane];
            float a1 = v1 * glu.sign[r0 + 32 + lane];
            { const float u = a0 + a1, v = a0 - a1; a0 = u; a1 = v; }
#pragma unroll
            for (int h = 16; h >= 1; h >>= 1) {
                const float e0 = __shfl_xor_sync(0xffffffff, a0, h);
                const float e1 = __shfl_xor_sync(0xffffffff, a1, h);
                const bool hi = lane & h;
                a0 = hi ? e0 - a0 : a0 + e0;
                a1 = hi ? e1 - a1 : a1 + e1;
            }
            half * xo = xd + (size_t) (j0 + n) * glu.Kd + r0;
            xo[lane]      = __float2half(a0 * 0.125f);
            xo[32 + lane] = __float2half(a1 * 0.125f);
        }
    }
    for (int q = tid; q < nt * TQ_PF_M; q += blockDim.x) {
        const int n = q / TQ_PF_M, r = q % TQ_PF_M;
        const int pr = pairs[j0 + n];
        if (red) {   // GGML_TQ_REDFUSE: the MoE weighted sum, red[t][row] += (y * scale) * weight (zeroed before)
            float v = sC[n * TQ_PF_M + r] * sg[r];
            if (rsc) {
                v *= rsc[pr];
            }
            atomicAdd(red + (size_t) (pr / n_used) * rows + r0 + r, v * rw[pr]);
            continue;
        }
        float * y = (float *) (dst + (pr / n_used) * nb2 + (pr % n_used) * nb1);
        y[r0 + r] = sC[n * TQ_PF_M + r] * sg[r];
    }
}

template <int N>
static void tq_prefill_launch_dev(const int4 * items_dev, const int n_items, cudaStream_t stream,
                                  const ggml_tensor * src0, const ggml_tensor * gain, const half * xs, const int32_t * pairs,
                                  char * out, const size_t onb1, const size_t onb2, const int K, const int rows, const int n_used) {
    if (n_items == 0) {
        return;
    }
    const size_t smem = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + N) * TQ_PF_LD * 2;   // sC shares sA / sB
    static bool smem_set = false;
    if (!smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<N, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<N, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        smem_set = true;
    }
    static const int magic = [] { const char * v = getenv("GGML_TQ_MAGIC"); return v ? atoi(v) : 1; }();
    const dim3 grid(rows / TQ_PF_M, n_items);
    (magic ? tq_prefill_mma<N, true> : tq_prefill_mma<N, false>)<<<grid, 128, smem, stream>>>((const uint8_t *) src0->data, (const half *) gain->data, xs, items_dev, pairs,
                                                   out, K, rows, src0->nb[2], n_used, onb1, onb2, tq_knob_pf_dbg(), nullptr, 0, 0, 0, nullptr, nullptr, nullptr, tq_glu_args());
    CUDA_CHECK(cudaGetLastError());
}

template <int N>
static void tq_prefill_launch(const std::vector<int4> & items, ggml_cuda_pool_alloc<int4> & items_d, cudaStream_t stream,
                              const ggml_tensor * src0, const ggml_tensor * gain, const half * xs, const int32_t * pairs,
                              char * out, const size_t onb1, const size_t onb2, const int K, const int rows, const int n_used) {
    if (items.empty()) {
        return;
    }
    items_d.alloc(items.size());
    CUDA_CHECK(cudaMemcpyAsync(items_d.ptr, tq_keep(items.data(), items.size() * sizeof(int4)), items.size() * sizeof(int4),
                               cudaMemcpyHostToDevice, stream));
    tq_prefill_launch_dev<N>(items_d.ptr, (int) items.size(), stream, src0, gain, xs, pairs, out, onb1, onb2, K, rows, n_used);
}

// GB10: pinned host staging for the multi prefill (GGML_TQ_PIN, default 1): the ids come back and the pairs + every
// launch's items go up in one copy each (was 1 + 1 + up to 3 per node pageable copies). Reused per call: each call starts
// with a stream sync, so the previous call's upload is done.
static int tq_pin() {
    static const int v = [] { const char * e = getenv("GGML_TQ_PIN"); return e ? atoi(e) : 1; }();
    return v;
}
static char * tq_pin_buf(const int which, const size_t n) {
    static char * buf[2] = {nullptr, nullptr};
    static size_t cap[2] = {0, 0};
    if (n > cap[which]) {
        if (buf[which]) {
            CUDA_CHECK(cudaFreeHost(buf[which]));
        }
        cap[which] = std::max(n, (size_t) 1 << 20) * 2;
        CUDA_CHECK(cudaMallocHost(&buf[which], cap[which]));
    }
    return buf[which];
}


// prefill, the tile decode shared by up to NC chunks of 32 tokens: per K step the 64 x 128 weight tile is decoded once into
// sA, then each chunk's inputs are staged in sB and multiplied (acc of every chunk in registers). Same smem as N = 32, so
// the same occupancy, with the decode (the larger cost: skipping it doubled pp512) amortized over up to 32 * NC tokens.
template <int NC, bool MAGIC>
static __global__ void __launch_bounds__(128)
tq_prefill_mmc(const uint8_t * __restrict__ W, const half * __restrict__ gain, const half * __restrict__ xs,
               const int4 * __restrict__ items, const int32_t * __restrict__ pairs, char * __restrict__ dst, const int K, const int rows,
               const size_t nb02, const int n_used, const size_t nb1, const size_t nb2) {
    using namespace nvcuda;
    constexpr int N = 32;
    extern __shared__ __align__(32) char smem_tq[];
    uint8_t * sT = (uint8_t *) smem_tq;                          // 4 x 528 B
    float   * sg = (float *) (smem_tq + 4 * TQ_PF_SEG);          // [64]
    half    * sA = (half *) (sg + TQ_PF_M);                      // [64][LD]
    half    * sB = sA + TQ_PF_M * TQ_PF_LD;                      // [32][LD]
    float   * sC = (float *) sA;                                 // [32][64] at the end
    static_assert(TQ_PF_M * N * 4 <= (TQ_PF_M + N) * TQ_PF_LD * 2, "sC must fit in sA + sB");

    const int4 it = items[blockIdx.y];
    const int e = it.x, j0 = it.y, nt = it.z;
    const int nch = (nt + N - 1) / N;
    const int r0 = blockIdx.x * TQ_PF_M;
    const int tid = threadIdx.x, warp = tid / 32;
    const int nc = K / 16;

    if (tid < TQ_PF_M) {
        sg[tid] = __half2float(gain[(size_t) e * rows + r0 + tid]) * (1.0f / 147.8f);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NC][2];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        wmma::fill_fragment(acc[c][0], 0.0f);
        wmma::fill_fragment(acc[c][1], 0.0f);
    }
    const uint8_t * Wb = W + (size_t) e * nb02 + (size_t) (r0 / 16) * nc * TQ_TILE_B;
    constexpr int NT = (4 * TQ_PF_SEG / 8 + 127) / 128;
    auto load_tiles = [&](const int k0, uint2 (&ct)[NT]) {
        const int jt0 = k0 / 16;
        const int nck = min(8, nc - jt0);
        const int seg8 = nck * TQ_TILE_B / 8;
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int q = tid + i * 128;
            const int sl = q / (TQ_PF_SEG / 8), w = q % (TQ_PF_SEG / 8);
            ct[i] = make_uint2(0, 0);
            if (sl < 4 && w < seg8) {
                ct[i] = __ldg((const uint2 *) (Wb + (size_t) sl * nc * TQ_TILE_B + (size_t) jt0 * TQ_TILE_B) + w);
            }
        }
    };
    uint2 ct[NT];
    load_tiles(0, ct);
    for (int k0 = 0; k0 < K; k0 += TQ_PF_K) {
        const int nck = min(8, nc - k0 / 16);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int q = tid + i * 128;
            if (q < 4 * TQ_PF_SEG / 8) {
                ((uint2 *) sT)[q] = ct[i];
            }
        }
        __syncthreads();
        if (k0 + TQ_PF_K < K) {
            load_tiles(k0 + TQ_PF_K, ct);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = tid + i * 128;
            const int sl = u >> 7, jt = (u >> 4) & 7, r = u & 15;
            if (jt < nck) {
                const uint16_t * tb = (const uint16_t *) (sT + sl * TQ_PF_SEG + jt * TQ_TILE_B);
                auto SB = [&](const int k) -> uint32_t { return (uint32_t) tb[1 + 2*k] | ((uint32_t) tb[2 + 2*k] << 16); };
                const uint32_t sbr = SB(r);
                const uint32_t lo = r == 0 ? ((tb[0] & 0x3fffu) | (sbr << 14)) : ((SB(r - 1) >> 18) | (sbr << 14));
                const uint32_t hi = (sbr >> 18) | (r < 15 ? (SB(r + 1) << 14) : 0u);
                const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                half2 o[8];
#pragma unroll
                for (int c = 0; c < 16; c += 2) {
                    const uint32_t b0 = tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu);
                    const uint32_t b1 = tq_bsum((uint32_t) (ch >> (2*c + 2)) & 0x3fffu);
                    o[c/2] = MAGIC ? tq_h2(b0, b1) : __floats2half2_rn((float) ((int) b0 - 510), (float) ((int) b1 - 510));
                }
                uint4 * dp = (uint4 *) (sA + (sl * 16 + r) * TQ_PF_LD + jt * 16);
                dp[0] = *(const uint4 *) &o[0];
                dp[1] = *(const uint4 *) &o[4];
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < nch) {
                // chunk c's inputs: 32 tokens x (nck * 16) halves, 4 uint4 a thread
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int q = tid + i * 128;
                    const int n = q / (TQ_PF_K / 8), cc = q % (TQ_PF_K / 8);
                    const int jj = c * N + n;
                    uint4 v = make_uint4(0, 0, 0, 0);
                    if (jj < nt && cc < nck * 2) {
                        v = *(const uint4 *) (xs + (size_t) (j0 + jj) * K + k0 + cc * 8);
                    }
                    *(uint4 *) (sB + n * TQ_PF_LD + cc * 8) = v;
                }
                __syncthreads();   // (first chunk: also the decode of sA)
                for (int kk = 0; kk < nck * 16; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sA + (warp * 16) * TQ_PF_LD + kk, TQ_PF_LD);
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                        wmma::load_matrix_sync(fb, sB + (f * 16) * TQ_PF_LD + kk, TQ_PF_LD);
                        wmma::mma_sync(acc[c][f], fa, fb, acc[c][f]);
                    }
                }
                __syncthreads();
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c < nch) {
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                wmma::store_matrix_sync(sC + (f * 16) * TQ_PF_M + warp * 16, acc[c][f], TQ_PF_M, wmma::mem_col_major);
            }
            __syncthreads();
            const int ntc = min(N, nt - c * N);
            for (int q = tid; q < ntc * TQ_PF_M; q += blockDim.x) {
                const int n = q / TQ_PF_M, r = q % TQ_PF_M;
                const int pr = pairs[j0 + c * N + n];
                float * y = (float *) (dst + (pr / n_used) * nb2 + (pr % n_used) * nb1);
                y[r0 + r] = sC[n * TQ_PF_M + r] * sg[r];
            }
            __syncthreads();
        }
    }
}

// GGML_TQ_PF_MC: chunks of 32 tokens sharing one tile decode (default 0 = the plain N = 32 / 64 / 128 kernels; 2 or 4)
static int tq_pf_mc() {
    static const int v = [] { const char * e = getenv("GGML_TQ_PF_MC"); return e ? atoi(e) : 0; }();   // measured slower: pp512 717 -> 685 (2) / 583 (4)
    return v;
}

static void tq_prefill_launch_mc(const std::vector<int4> & items, ggml_cuda_pool_alloc<int4> & items_d, cudaStream_t stream,
                                 const ggml_tensor * src0, const ggml_tensor * gain, const half * xs, const int32_t * pairs,
                                 char * out, const size_t onb1, const size_t onb2, const int K, const int rows, const int n_used) {
    if (items.empty()) {
        return;
    }
    items_d.alloc(items.size());
    CUDA_CHECK(cudaMemcpyAsync(items_d.ptr, tq_keep(items.data(), items.size() * sizeof(int4)), items.size() * sizeof(int4),
                               cudaMemcpyHostToDevice, stream));
    const size_t smem = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + 32) * TQ_PF_LD * 2;
    static bool smem_set = false;
    if (!smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mmc<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mmc<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        smem_set = true;
    }
    const dim3 grid(rows / TQ_PF_M, items.size());
    if (tq_pf_mc() >= 4) {
        tq_prefill_mmc<4, true><<<grid, 128, smem, stream>>>((const uint8_t *) src0->data, (const half *) gain->data, xs, items_d.ptr,
            pairs, out, K, rows, src0->nb[2], n_used, onb1, onb2);
    } else {
        tq_prefill_mmc<2, true><<<grid, 128, smem, stream>>>((const uint8_t *) src0->data, (const half *) gain->data, xs, items_d.ptr,
            pairs, out, K, rows, src0->nb[2], n_used, onb1, onb2);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mul_mat_id_tq_needs_sync(const ggml_tensor * dst) {
    static const int mmv_max = [] { const char * v = getenv("GGML_TQ_MMV_MAX"); return v ? atoi(v) : GGML_CUDA_TQ_MMV_MAX_TOKENS; }();
    return dst->src[1]->ne[2] > mmv_max;
}

void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const ggml_tensor * gain = dst->src[3];
    const ggml_tensor * sign = dst->src[4];

    GGML_ASSERT(src0->type == GGML_TYPE_TQ2T && ggml_is_contiguous(src0));
    GGML_ASSERT(gain && gain->type == GGML_TYPE_F16 && ggml_is_contiguous(gain) && "TQ2T mul_mat_id needs src[3] = F16 gains");
    GGML_ASSERT(sign && sign->type == GGML_TYPE_F32 && ggml_nelements(sign) == src0->ne[0] / 16 && "TQ2T mul_mat_id needs src[4] = F32 signs [K]");
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int K        = src0->ne[0] / 16;   // strips: [K*16, rows/16, E]
    const int rows     = src0->ne[1] * 16;
    const int n_expert = src0->ne[2];
    const int n_used   = ids->ne[0];
    const int n_tok    = ids->ne[1];
    const int ne11     = src1->ne[1];
    GGML_ASSERT(K % 64 == 0 && rows % 16 == 0);
    GGML_ASSERT(gain->ne[0] == rows && gain->ne[1] == n_expert);
    GGML_ASSERT(dst->ne[0] == rows && dst->ne[1] == n_used && dst->ne[2] == n_tok);
    GGML_ASSERT(src1->ne[2] == n_tok && (ne11 == 1 || ne11 == n_used));

    cudaStream_t stream = ctx.stream();
    const int ida = dst->op_params[14] == GGML_CUDA_RVQ_GROUP_MAGIC ? dst->op_params[12] : 0;

    // fused down-projection input (op_params[13] == 'GLU1'): src1 = gate output, src[5] = up output (as in rvq.cu)
    const ggml_tensor * up = dst->op_params[13] == GGML_CUDA_RVQ_GLU_MAGIC ? dst->src[5] : nullptr;
    float cg = 1.0f, cu = 1.0f;
    if (up) {
        GGML_ASSERT(up->type == GGML_TYPE_F32 && ggml_are_same_shape(up, src1) && up->nb[1] == src1->nb[1] && up->nb[2] == src1->nb[2]);
        memcpy(&cg, &dst->op_params[10], sizeof(float));
        memcpy(&cu, &dst->op_params[11], sizeof(float));
    }
    const char * xu = up ? (const char *) up->data : nullptr;
    // accumulate mode (src[6]): slots this group does not own take src[6] instead of zero (the sum of the earlier groups)
    const ggml_tensor * prev = dst->src[6];
    if (prev) {
        GGML_ASSERT(prev->type == GGML_TYPE_F32 && ggml_are_same_shape(prev, dst) && ggml_is_contiguous(prev) && ggml_is_contiguous(dst));
    }

    // GGML_TQ_MMV3 (default 1): decode v3 (rotate kernel + K split over warps); GGML_TQ_MMV2: v2 (rotation fused)
    static const int mmv3 = [] { const char * v = getenv("GGML_TQ_MMV3"); return v ? atoi(v) : 1; }();
    static const int mmv2 = [] { const char * v = getenv("GGML_TQ_MMV2"); return v ? atoi(v) : 1; }();
    if ((mmv3 || up || prev) && !ggml_cuda_mul_mat_id_tq_needs_sync(dst)) {
        const int n_in = n_tok * ne11;
        ggml_cuda_pool_alloc<float> xr(ctx.pool(), (size_t) n_in * K);
        {
            const dim3 grid((K / 64 + 3) / 4, n_in);
            tq_rotate<<<grid, 128, 0, stream>>>((const char *) src1->data, (const float *) sign->data, xr.ptr, nullptr, K, ne11,
                                                src1->nb[1], src1->nb[2], xu, cg, cu);
            CUDA_CHECK(cudaGetLastError());
        }
        const int nc = K / 16;
        // GGML_TQ_KS: force the K split (1, 2 or 4); default from the tile count
        static const int ks_env = [] { const char * v = getenv("GGML_TQ_KS"); return v ? atoi(v) : 0; }();
        const int ks = ks_env ? ks_env : (nc >= 64 ? 2 : 1);   // tg64 on tq80c: KS 4 23.3, 2 23.9, 1 22.0
        const int spb = 4 / ks;
        // GGML_TQ_ZF (default 1): pairs innermost when several tokens (speculative verify)
        static const int zf_env = [] { const char * v = getenv("GGML_TQ_ZF"); return v ? atoi(v) : 1; }();
        const int zf = zf_env && n_tok > 1;
        const dim3 grid = zf ? dim3(n_tok * n_used, (rows / 16 + spb - 1) / spb) : dim3((rows / 16 + spb - 1) / spb, n_tok * n_used);
#define TQ_MMV3_ARGS (const uint8_t *) src0->data, (const half *) gain->data, xr.ptr, (const char *) ids->data, (char *) dst->data, \
            K, rows, n_expert, n_used, ne11, src0->nb[2], ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2], ida, zf, \
            prev ? (const char *) prev->data : nullptr
        if (ks == 4) {
            tq_mmv3<4><<<grid, 128, 0, stream>>>(TQ_MMV3_ARGS);
        } else if (ks == 2) {
            tq_mmv3<2><<<grid, 128, 0, stream>>>(TQ_MMV3_ARGS);
        } else {
            tq_mmv3<1><<<grid, 128, 0, stream>>>(TQ_MMV3_ARGS);
        }
#undef TQ_MMV3_ARGS
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mmv2 && !ggml_cuda_mul_mat_id_tq_needs_sync(dst)) {
        const dim3 grid((rows / 16 + TQ_NW - 1) / TQ_NW, n_tok * n_used);
        tq_mmv2<<<grid, TQ_NW * 32, K * sizeof(float), stream>>>(
            (const uint8_t *) src0->data, (const half *) gain->data, (const char *) src1->data, (const float *) sign->data,
            (const char *) ids->data, (char *) dst->data, K, rows, n_expert, n_used, ne11, src0->nb[2], ids->nb[0], ids->nb[1],
            src1->nb[1], src1->nb[2], dst->nb[1], dst->nb[2], ida);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (!ggml_cuda_mul_mat_id_tq_needs_sync(dst)) {
        // old decode path (GGML_TQ_MMV3=0 GGML_TQ_MMV2=0): rotate every input row once
        const int n_in = n_tok * ne11;
        ggml_cuda_pool_alloc<float> xr(ctx.pool(), (size_t) n_in * K);
        ggml_cuda_pool_alloc<float> xsum(ctx.pool(), n_in);
        CUDA_CHECK(cudaMemsetAsync(xsum.ptr, 0, n_in * sizeof(float), stream));
        {
            const dim3 grid((K / 64 + 3) / 4, n_in);
            tq_rotate<<<grid, 128, 0, stream>>>((const char *) src1->data, (const float *) sign->data, xr.ptr, xsum.ptr, K, ne11,
                                                src1->nb[1], src1->nb[2], xu, cg, cu);
            CUDA_CHECK(cudaGetLastError());
        }
        const dim3 grid((rows / 16 + TQ_NW - 1) / TQ_NW, n_tok * n_used);
        tq_mmv<<<grid, TQ_NW * 32, K * sizeof(float), stream>>>(
            (const uint8_t *) src0->data, (const half *) gain->data, xr.ptr, xsum.ptr, (const char *) ids->data, (char *) dst->data,
            K, rows, n_expert, n_used, ne11, src0->nb[2], ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2], ida);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // prefill: sort (token, slot) pairs by expert on the host, gather rotated inputs as f16, per expert f16 dequant + one
    // cuBLAS GEMM, scatter back. Skipped slots stay zero.
    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    { cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone; CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
      if (cs != cudaStreamCaptureStatusNone) { GGML_ABORT("tq: host-synced prefill path while capturing a CUDA graph (set GGML_TQ_PF_GRAPH=0)"); } }
    tq_sync_and_release(stream);

    std::vector<int32_t> count(n_expert, 0);
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const uint32_t e = *(const int32_t *) (ids_host.data() + t * ids->nb[1] + s * ids->nb[0]) - ida;
            if (e < (uint32_t) n_expert) {
                count[e]++;
            }
        }
    }
    std::vector<int32_t> start(n_expert + 1, 0);
    for (int e = 0; e < n_expert; ++e) {
        start[e + 1] = start[e] + count[e];
    }
    const int n_valid = start[n_expert];
    std::vector<int32_t> pairs(std::max(n_valid, 1));
    {
        std::vector<int32_t> fill(start.begin(), start.end() - 1);
        for (int t = 0; t < n_tok; ++t) {
            for (int s = 0; s < n_used; ++s) {
                const uint32_t e = *(const int32_t *) (ids_host.data() + t * ids->nb[1] + s * ids->nb[0]) - ida;
                if (e < (uint32_t) n_expert) {
                    pairs[fill[e]++] = t * n_used + s;
                }
            }
        }
    }
    if (n_valid < n_tok * n_used) {   // some slots belong to other groups: zero, or the earlier groups' sum
        GGML_ASSERT(ggml_is_contiguous(dst));
        if (prev) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, prev->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    }
    if (n_valid == 0) {
        return;
    }

    ggml_cuda_pool_alloc<int32_t> pairs_d(ctx.pool(), n_valid);
    CUDA_CHECK(cudaMemcpyAsync(pairs_d.ptr, tq_keep(pairs.data(), n_valid * sizeof(int32_t)), n_valid * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    ggml_cuda_pool_alloc<half>  xs(ctx.pool(), (size_t) n_valid * K);
    {
        const dim3 grid((K / 64 + 3) / 4, n_valid);
        tq_rotate_gather_f16<<<grid, 128, 0, stream>>>((const char *) src1->data, xu, cg, cu, (const float *) sign->data, pairs_d.ptr,
                                                       xs.ptr, K, n_used, ne11, src1->nb[1], src1->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    // GGML_TQ_PF_MAX: experts with at most this many tokens use the fused kernel (0: always dequant + cuBLAS)
    static const int pf_max = [] { const char * v = getenv("GGML_TQ_PF_MAX"); return v ? atoi(v) : 1 << 30; }();   // pp512 on TQ1: 64 438, 128 457, 256 462, all 464
    std::vector<int4> it32, it64, it128;
    bool any_big = false;
    for (int e = 0; e < n_expert; ++e) {
        const int n = count[e];
        if (n == 0) {
            continue;
        }
        if (n > pf_max || rows % TQ_PF_M != 0) {
            any_big = true;
        } else if (tq_pf_nmax() <= 32) {
            for (int j = 0; j < n; j += 32) {
                it32.push_back(make_int4(e, start[e] + j, std::min(32, n - j), 0));
            }
        } else if (n <= 32) {
            it32.push_back(make_int4(e, start[e], n, 0));
        } else if (n <= 64) {
            it64.push_back(make_int4(e, start[e], n, 0));
        } else if (tq_pf_nmax() <= 64) {
            for (int j = 0; j < n; j += 64) {
                it64.push_back(make_int4(e, start[e] + j, std::min(64, n - j), 0));
            }
        } else {
            for (int j = 0; j < n; j += 128) {
                it128.push_back(make_int4(e, start[e] + j, std::min(128, n - j), 0));
            }
        }
    }
    ggml_cuda_pool_alloc<int4> items32_d(ctx.pool()), items64_d(ctx.pool()), items128_d(ctx.pool());
    tq_prefill_launch<32> (it32,  items32_d,  stream, src0, gain, xs.ptr, pairs_d.ptr, (char *) dst->data, dst->nb[1], dst->nb[2], K, rows, n_used);
    tq_prefill_launch<64> (it64,  items64_d,  stream, src0, gain, xs.ptr, pairs_d.ptr, (char *) dst->data, dst->nb[1], dst->nb[2], K, rows, n_used);
    tq_prefill_launch<128>(it128, items128_d, stream, src0, gain, xs.ptr, pairs_d.ptr, (char *) dst->data, dst->nb[1], dst->nb[2], K, rows, n_used);

    // the async H2D copies above are from pageable memory: they return once the host data is staged, so the host vectors
    // may go out of scope without a stream sync (GGML_TQ_PF_SYNC=1: sync anyway, as before)
    static const int pf_sync = [] { const char * v = getenv("GGML_TQ_PF_SYNC"); return v ? atoi(v) : 0; }();
    if (!any_big) {
        if (pf_sync) {
            tq_sync_and_release(stream);
        }
        return;
    }
    ggml_cuda_pool_alloc<float> ys(ctx.pool(), (size_t) n_valid * rows);
    ggml_cuda_pool_alloc<half> wf(ctx.pool(), (size_t) rows * K);   // declared after items_d: the pool frees in reverse order
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    const int dq_blocks = (int) std::min<int64_t>(((int64_t) rows * K / 16 + 255) / 256, 2048);
    for (int e = 0; e < n_expert && any_big; ++e) {
        const int n = count[e];
        if (n == 0 || (n <= pf_max && rows % TQ_PF_M == 0)) {
            continue;
        }
        tq_dequant_f16<<<dq_blocks, 256, 0, stream>>>((const uint8_t *) src0->data + (size_t) e * src0->nb[2],
                                                     (const half *) gain->data + (size_t) e * rows, wf.ptr, K, rows);
        CUDA_CHECK(cudaGetLastError());
        // Y[rows x n] (col-major, ld rows) = W^T (W: col-major K x rows) * X (col-major K x n)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, K,
            &alpha, wf.ptr, CUDA_R_16F, K,
            xs.ptr + (size_t) start[e] * K, CUDA_R_16F, K,
            &beta, ys.ptr + (size_t) start[e] * rows, CUDA_R_32F, rows,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // scatter only the cuBLAS experts' results (the fused kernel wrote its own)
    for (int e = 0; e < n_expert; ++e) {
        const int n = count[e];
        if (n == 0 || (n <= pf_max && rows % TQ_PF_M == 0)) {
            continue;
        }
        tq_scatter<<<n, 256, 0, stream>>>(ys.ptr + (size_t) start[e] * rows, pairs_d.ptr + start[e], (char *) dst->data, rows, n_used,
                                          dst->nb[1], dst->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }
    if (pf_sync) {
        tq_sync_and_release(stream);
    }
}

// =====================================================================================================================
// decode, several expert groups in one launch (as ggml_cuda_rvq_multi / _down_multi do for RVQ2X8).
// gate/up: the 2 x n_grp gate/up nodes of a layer (same src1, ids); every slot is owned by J nodes (its group's gate and up),
// the j-th owner of slot p is the j-th node in graph order whose range [ida, ida + E) holds the slot's expert.
// down: the chained GLU-down nodes (src[6] = previous node); one owner a slot; all slots are written into the last node.

#define TQ_MM_MAX 8

struct tq_mm_desc {
    const uint8_t * W;
    const half    * gain;
    const float   * sign;
    char          * dst;
    const char    * x;
    const char    * xu;
    size_t nb02, nb1, nb2, nb11, nb12;
    int K, rows, n_expert, ida, ne11;
};

struct tq_mm_args {
    int n_m;
    tq_mm_desc m[TQ_MM_MAX];
};

static __device__ __forceinline__ int tq_mm_owner(const tq_mm_args & a, const int id, const int j) {
    int c = 0;
    for (int m = 0; m < a.n_m; ++m) {
        if ((uint32_t) (id - a.m[m].ida) < (uint32_t) a.m[m].n_expert) {
            if (c == j) {
                return m;
            }
            ++c;
        }
    }
    return -1;
}

// xr[(j * n_pairs + p) * Kmax + k] = rotated input of slot p for its j-th owner (GLU of (x, xu) when xu is set)
static __global__ void tq_rotate_mm(const tq_mm_args a, const char * __restrict__ ids, float * __restrict__ xr, const int Kmax,
                                    const int n_used, const int n_pairs, const size_t ids_nb0, const size_t ids_nb1, const float cg,
                                    const float cu, const int xh) {
    ggml_cuda_pdl_lc();   // lets the decode kernel (PDL launch) start and prefetch weights while this runs
    const int p = blockIdx.y, j = blockIdx.z, t = p / n_used, s = p % n_used;
    const int id = *(const int32_t *) (ids + t * ids_nb1 + s * ids_nb0);
    const int m  = tq_mm_owner(a, id, j);
    if (m < 0) {
        return;
    }
    const tq_mm_desc & d = a.m[m];
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b * 64 >= d.K) {
        return;
    }
    const size_t off = t * d.nb12 + (s % d.ne11) * d.nb11;
    const float * xi = (const float *) (d.x + off);
    float v0 = xi[b*64 + lane], v1 = xi[b*64 + 32 + lane];
    if (d.xu) {
        const float * ui = (const float *) (d.xu + off);
        v0 = tq_act(v0, ui[b*64 + lane], cg, cu);
        v1 = tq_act(v1, ui[b*64 + 32 + lane], cg, cu);
    }
    float a0 = v0 * d.sign[b*64 + lane];
    float a1 = v1 * d.sign[b*64 + 32 + lane];
    const float u = a0 + a1, v = a0 - a1;
    a0 = u; a1 = v;
#pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        const float o0 = __shfl_xor_sync(0xffffffff, a0, h);
        const float o1 = __shfl_xor_sync(0xffffffff, a1, h);
        const bool hi = lane & h;
        a0 = hi ? o0 - a0 : a0 + o0;
        a1 = hi ? o1 - a1 : a1 + o1;
    }
    if (xh == 2) {   // int8 inputs, one scale per 16 (= one tile column block); stored replicated x4 for dp4a
        a0 *= 0.125f; a1 *= 0.125f;
        float m0 = fabsf(a0), m1 = fabsf(a1);
#pragma unroll
        for (int h = 8; h >= 1; h >>= 1) {
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, h));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, h));
        }
        const float d0 = m0 / 127.0f, d1 = m1 / 127.0f;
        int q0 = d0 > 0.0f ? __float2int_rn(a0 / d0) : 0;
        int q1 = d1 > 0.0f ? __float2int_rn(a1 / d1) : 0;
        const size_t base = ((size_t) j * n_pairs + p) * Kmax;
        int * xo = (int *) xr + base;
        xo[b*64 + lane]      = (q0 & 0xFF) * 0x01010101;
        xo[b*64 + 32 + lane] = (q1 & 0xFF) * 0x01010101;
        int z0 = q0, z1 = q1;
#pragma unroll
        for (int h = 8; h >= 1; h >>= 1) {
            z0 += __shfl_xor_sync(0xffffffff, z0, h);
            z1 += __shfl_xor_sync(0xffffffff, z1, h);
        }
        if ((lane & 15) == 0) {
            const size_t npk = (size_t) gridDim.z * n_pairs * Kmax;    // J * n_pairs * Kmax
            float * xs = (float *) xr + npk;
            int   * xz = (int *) (xs + npk / 16);
            const size_t g0 = base / 16 + b*4 + (lane >> 4), g1 = g0 + 2;
            xs[g0] = d0; xz[g0] = z0;
            xs[g1] = d1; xz[g1] = z1;
        }
        return;
    }
    if (xh) {   // f16 inputs (half the L2 traffic of the decode kernel's x reads)
        half * xo = (half *) xr + ((size_t) j * n_pairs + p) * Kmax;
        xo[b*64 + lane]      = __float2half(a0 * 0.125f);
        xo[b*64 + 32 + lane] = __float2half(a1 * 0.125f);
        return;
    }
    float * xo = xr + ((size_t) j * n_pairs + p) * Kmax;
    xo[b*64 + lane]      = a0 * 0.125f;
    xo[b*64 + 32 + lane] = a1 * 0.125f;
}

// as tq_mmv3 (KS warps a strip, SPB = 4 / KS strips a block), the matrix picked per slot: grid x = J * nblk (j-th owner,
// strip block), y = slot; zf swaps x and y (slots innermost, several tokens)
template <int KS, bool MAGIC, bool PREF, bool XH = false, bool Q8 = false, bool V16 = false>
static __global__ void __launch_bounds__(128)
tq_mmv_mm(const tq_mm_args a, const float * __restrict__ xr, const char * __restrict__ ids, const int Kmax, const int nblk,
          const int n_used, const int n_pairs, const size_t ids_nb0, const size_t ids_nb1, const int zf) {
    constexpr int SPB = 4 / KS;
    __shared__ uint32_t sm[4][32 * TQ_TILE_B / 4 + 2];
    __shared__ float red[4][17];
    const int bx = zf ? blockIdx.y : blockIdx.x, p = zf ? blockIdx.x : blockIdx.y;
    const int j = bx / nblk, sb = bx % nblk;
    const int t = p / n_used, s = p % n_used;
    const int id = *(const int32_t *) (ids + t * ids_nb1 + s * ids_nb0);
    const int m  = tq_mm_owner(a, id, j);
    if (m < 0) {
        return;
    }
    const tq_mm_desc & d = a.m[m];
    const int rows = d.rows, K = d.K;
    if (sb * SPB * 16 >= rows) {
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int sl = warp / KS, part = warp % KS;
    const int strip = sb * SPB + sl;
    const int e  = id - d.ida;
    const int nc = K / 16;
    const int L  = (((nc + KS - 1) / KS) + 3) & ~3;
    const int jb = min(nc, part * L), je = min(nc, jb + L);
    const float * xg = xr + ((size_t) j * n_pairs + p) * Kmax;
    const half  * xgh = (const half *) xr + ((size_t) j * n_pairs + p) * Kmax;
    // Q8: per-16 scales and sums follow the J * n_pairs * Kmax int8x4 inputs (see tq_rotate_mm, xh == 2)
    const size_t npk = (size_t) ((zf ? gridDim.y : gridDim.x) / nblk) * n_pairs * Kmax;
    const float * xsc = xr + npk;
    const int   * xzs = (const int *) (xsc + npk / 16);
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        acc[r] = 0.0f;
    }
    float sx = 0.0f;
    if (strip * 16 < rows) {
        const uint8_t * Ws = d.W + (size_t) e * d.nb02 + (size_t) strip * nc * TQ_TILE_B;
        // PREF: the next chunk's tiles are loaded into registers while this chunk is computed
        constexpr int NQ = (32 * TQ_TILE_B / 8 + 31) / 32;   // 9 uint2 a lane
        uint2 pq[PREF ? (V16 ? 10 : NQ) : 1];   // V16 fills 5 uint4 = 10 uint2
        // V16: a full 32-tile chunk (2112 B = 132 x 16) at a 16 B aligned address is loaded as uint4
        bool w16 = false;
        auto load_chunk = [&](const int j1) {
            const int n1 = min(32, je - j1);
            const uint8_t * b1 = Ws + (size_t) j1 * TQ_TILE_B;
            w16 = V16 && n1 == 32 && (((uintptr_t) b1) & 15) == 0;
            if (w16) {
                const uint4 * s4 = (const uint4 *) b1;
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int q = lane + 32 * i;
                    const uint4 v = q < 132 ? __ldg(s4 + q) : make_uint4(0, 0, 0, 0);
                    pq[2*i] = make_uint2(v.x, v.y);
                    pq[2*i + 1] = make_uint2(v.z, v.w);
                }
                return;
            }
            const uint2 * s1 = (const uint2 *) b1;
            const int m8 = (n1 * TQ_TILE_B + 7) / 8;
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const int q = lane + 32 * i;
                pq[i] = q < m8 ? __ldg(s1 + q) : make_uint2(0, 0);
            }
        };
        if (PREF && jb < je) {
            load_chunk(jb);
        }
        // PDL: everything above (ids, owner, first weight chunk) may overlap tq_rotate_mm; xr only after the wait
        ggml_cuda_pdl_sync();
        for (int j0 = jb; j0 < je; j0 += 32) {
            const int nt = min(32, je - j0);
            const uint2 * src = (const uint2 *) (Ws + (size_t) j0 * TQ_TILE_B);
            const int n8 = (nt * TQ_TILE_B + 7) / 8;
            __syncwarp();
            if (PREF && w16) {
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int q = lane + 32 * i;
                    if (q < 132) {
                        sm[warp][4*q]     = pq[2*i].x;
                        sm[warp][4*q + 1] = pq[2*i].y;
                        sm[warp][4*q + 2] = pq[2*i + 1].x;
                        sm[warp][4*q + 3] = pq[2*i + 1].y;
                    }
                }
            } else if (PREF) {
#pragma unroll
                for (int i = 0; i < NQ; ++i) {
                    const int q = lane + 32 * i;
                    if (q < n8) {
                        sm[warp][2*q] = pq[i].x;
                        sm[warp][2*q + 1] = pq[i].y;
                    }
                }
            } else {
                for (int i = lane; i < n8; i += 32) {
                    const uint2 q = src[i];
                    sm[warp][2*i] = q.x;
                    sm[warp][2*i + 1] = q.y;
                }
            }
            __syncwarp();
            if (PREF && j0 + 32 < je) {
                load_chunk(j0 + 32);
            }
            if (Q8 && lane < nt) {
                // int8 x (replicated bytes), scale and sum per 16: one dp4a a weight, int accumulate per tile row
                int xq[16];
                const int4 * xq4 = (const int4 *) ((const int *) xg + (j0 + lane) * 16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 q = __ldg(xq4 + c);
                    xq[4*c] = q.x; xq[4*c + 1] = q.y; xq[4*c + 2] = q.z; xq[4*c + 3] = q.w;
                }
                const size_t gi = (((size_t) j * n_pairs + p) * Kmax) / 16 + j0 + lane;
                const float sc  = xsc[gi];
                const int   z5  = 510 * xzs[gi];
                const int off = TQ_TILE_B * lane;
                const uint32_t * tw = sm[warp] + (off >> 2);
                uint32_t s0, SB[16];
                if (off & 2) {
                    s0 = tw[0] >> 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = tw[k + 1];
                    }
                } else {
                    s0 = tw[0] & 0xffff;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                    }
                }
                s0 &= 0x3fff;
                uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                    const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                    int ia = 0;
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        const uint32_t v = ((uint32_t) (ch >> (2*c)) & 0x3fffu) * 34038481u + 76625530u;
                        ia = tq_dp4a_us(v, xq[c], ia);
                    }
                    acc[r] = fmaf(sc, (float) (ia - z5), acc[r]);
                    lo = hi;
                }
            } else if (lane < nt) {
                float xv[16];
                if (XH) {
                    const uint4 * xh4 = (const uint4 *) (xgh + (j0 + lane) * 16);
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const uint4 q = __ldg(xh4 + c);
                        const half2 * h2 = (const half2 *) &q;
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const float2 f2 = __half22float2(h2[k]);
                            xv[8*c + 2*k] = f2.x; xv[8*c + 2*k + 1] = f2.y;
                        }
                    }
                } else {
                    const float4 * xc4 = (const float4 *) (xg + (j0 + lane) * 16);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const float4 q = __ldcg(xc4 + c);   // L2: xr is written by the PDL predecessor
                        xv[4*c] = q.x; xv[4*c + 1] = q.y; xv[4*c + 2] = q.z; xv[4*c + 3] = q.w;
                    }
                }
                const int off = TQ_TILE_B * lane;
                const uint32_t * tw = sm[warp] + (off >> 2);
                uint32_t s0, SB[16];
                if (off & 2) {
                    s0 = tw[0] >> 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = tw[k + 1];
                    }
                } else {
                    s0 = tw[0] & 0xffff;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                    }
                }
                s0 &= 0x3fff;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    sx += xv[c];
                }
                uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                    const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                    float acc1 = 0.0f;
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        const uint32_t bs = tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu);
                        acc1 = fmaf(MAGIC ? tq_u2f(bs) : (float) bs, xv[c], acc1);
                    }
                    acc[r] += acc1;
                    lo = hi;
                }
            }
        }
    }
    ggml_cuda_pdl_sync();   // warps without a strip did not wait above; dst is written below
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        acc[r] = v;
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        sx += __shfl_xor_sync(0xffffffff, sx, o);
    }
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r == lane) {
                v = acc[r];
            }
        }
        red[warp][lane] = v;
    }
    if (lane == 16) {
        red[warp][16] = sx;
    }
    __syncthreads();
    if (threadIdx.x < SPB * 16) {
        const int ls = threadIdx.x / 16, r = threadIdx.x % 16;
        const int st = sb * SPB + ls;
        if (st * 16 < rows) {
            float v = 0.0f, sxt = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v   += red[ls * KS + k][r];
                sxt += red[ls * KS + k][16];
            }
            const int row = st * 16 + r;
            float * y = (float *) (d.dst + t * d.nb2 + s * d.nb1);
            y[row] = __half2float(d.gain[(size_t) e * rows + row]) * (1.0f / 147.8f) * (v - 510.0f * sxt);
        }
    }
}


// several tokens (speculative verify): the slots that picked the same expert share one block, so each tile is read once
// and decoded once per row, then multiplied with up to NP inputs. The block of a slot whose rank among the slots with the
// same expert is a multiple of NP is the leader for the next NP of them; the others exit.
template <int KS, int NP>
static __global__ void __launch_bounds__(128)
tq_mmv_mmt(const tq_mm_args a, const float * __restrict__ xr, const char * __restrict__ ids, const int Kmax, const int nblk,
           const int n_used, const int n_pairs, const size_t ids_nb0, const size_t ids_nb1, const int zf) {
    constexpr int SPB = 4 / KS;
    __shared__ uint32_t sm[4][32 * TQ_TILE_B / 4 + 2];
    __shared__ float red[4][NP][17];
    const int bx = zf ? blockIdx.y : blockIdx.x, p = zf ? blockIdx.x : blockIdx.y;
    const int j = bx / nblk, sb = bx % nblk;
    auto pid = [&](const int q) { return *(const int32_t *) (ids + (q / n_used) * ids_nb1 + (q % n_used) * ids_nb0); };
    const int id = pid(p);
    const int m  = tq_mm_owner(a, id, j);
    if (m < 0) {
        return;
    }
    const tq_mm_desc & d = a.m[m];
    const int rows = d.rows, K = d.K;
    if (sb * SPB * 16 >= rows) {
        return;
    }
    int rank = 0;
    for (int q = 0; q < p; ++q) {
        rank += pid(q) == id;
    }
    if (rank % NP != 0) {
        return;
    }
    int qs[NP];
    int nq = 0;
    for (int q = p; q < n_pairs && nq < NP; ++q) {
        if (pid(q) == id) {
            qs[nq++] = q;
        }
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int sl = warp / KS, part = warp % KS;
    const int strip = sb * SPB + sl;
    const int e  = id - d.ida;
    const int nc = K / 16;
    const int L  = (((nc + KS - 1) / KS) + 3) & ~3;
    const int jb = min(nc, part * L), je = min(nc, jb + L);
    float acc[NP][16];
    float sx[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        sx[k] = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc[k][r] = 0.0f;
        }
    }
    if (strip * 16 < rows) {
        const uint8_t * Ws = d.W + (size_t) e * d.nb02 + (size_t) strip * nc * TQ_TILE_B;
        for (int j0 = jb; j0 < je; j0 += 32) {
            const int nt = min(32, je - j0);
            const uint2 * src = (const uint2 *) (Ws + (size_t) j0 * TQ_TILE_B);
            const int n8 = (nt * TQ_TILE_B + 7) / 8;
            __syncwarp();
            for (int i = lane; i < n8; i += 32) {
                const uint2 q = src[i];
                sm[warp][2*i] = q.x;
                sm[warp][2*i + 1] = q.y;
            }
            __syncwarp();
            if (lane < nt) {
                float xv[NP][16];
#pragma unroll
                for (int k = 0; k < NP; ++k) {
                    if (k < nq) {
                        const float4 * xc4 = (const float4 *) (xr + ((size_t) j * n_pairs + qs[k]) * Kmax + (j0 + lane) * 16);
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const float4 q = __ldg(xc4 + c);
                            xv[k][4*c] = q.x; xv[k][4*c + 1] = q.y; xv[k][4*c + 2] = q.z; xv[k][4*c + 3] = q.w;
                        }
                    } else {
#pragma unroll
                        for (int c = 0; c < 16; ++c) {
                            xv[k][c] = 0.0f;
                        }
                    }
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        sx[k] += xv[k][c];
                    }
                }
                const int off = TQ_TILE_B * lane;
                const uint32_t * tw = sm[warp] + (off >> 2);
                uint32_t s0, SB[16];
                if (off & 2) {
                    s0 = tw[0] >> 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = tw[k + 1];
                    }
                } else {
                    s0 = tw[0] & 0xffff;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        SB[k] = __funnelshift_r(tw[k], tw[k + 1], 16);
                    }
                }
                s0 &= 0x3fff;
                uint32_t lo = s0 | (SB[0] << 14);
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const uint32_t hi = (SB[r] >> 18) | (r < 15 ? (SB[r + 1] << 14) : 0u);
                    const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                    float a0[NP];
#pragma unroll
                    for (int k = 0; k < NP; ++k) {
                        a0[k] = 0.0f;
                    }
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        const float bv = (float) tq_bsum((uint32_t) (ch >> (2*c)) & 0x3fffu);
#pragma unroll
                        for (int k = 0; k < NP; ++k) {
                            a0[k] = fmaf(bv, xv[k][c], a0[k]);
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NP; ++k) {
                        acc[k][r] += a0[k];
                    }
                    lo = hi;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NP; ++k) {
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float v = acc[k][r];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o);
            }
            acc[k][r] = v;
        }
        float v = sx[k];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        sx[k] = v;
        if (lane < 16) {
            float w = 0.0f;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r == lane) {
                    w = acc[k][r];
                }
            }
            red[warp][k][lane] = w;
        }
        if (lane == 16) {
            red[warp][k][16] = sx[k];
        }
    }
    __syncthreads();
    for (int q = threadIdx.x; q < nq * SPB * 16; q += blockDim.x) {
        const int k = q / (SPB * 16), ls = (q / 16) % SPB, r = q % 16;
        const int st = sb * SPB + ls;
        if (st * 16 < rows) {
            float v = 0.0f, sxt = 0.0f;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                v   += red[ls * KS + kk][k][r];
                sxt += red[ls * KS + kk][k][16];
            }
            const int row = st * 16 + r, pq = qs[k];
            float * y = (float *) (d.dst + (pq / n_used) * d.nb2 + (pq % n_used) * d.nb1);
            y[row] = __half2float(d.gain[(size_t) e * rows + row]) * (1.0f / 147.8f) * (v - 510.0f * sxt);
        }
    }
}

static int tq_knob(const char * name, const int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// GGML_TQ_XH (decode, multi-group path): the rotated inputs stored as f16 instead of f32
static int tq_xh() {
    static const int v = tq_knob("GGML_TQ_XH", 0);
    return v;
}

static bool tq_mm_node_ok(const ggml_tensor * n) {
    if (!(n->op == GGML_OP_MUL_MAT_ID && n->src[0] && n->src[0]->type == GGML_TYPE_TQ2T &&
          n->op_params[14] == GGML_CUDA_RVQ_GROUP_MAGIC && n->src[3] && n->src[4] && n->type == GGML_TYPE_F32 &&
          ggml_is_contiguous(n) && n->src[1]->type == GGML_TYPE_F32 && n->src[1]->nb[0] == sizeof(float))) {
        return false;
    }
    if (ggml_cuda_mul_mat_id_tq_needs_sync(n)) {
        // prefill: one host sort for the whole set (GGML_TQ_PF_MULTI, default 1); fused kernel only (rows % 64)
        static const int pf_multi = tq_knob("GGML_TQ_PF_MULTI", 1);
        return pf_multi && (n->src[0]->ne[1] * 16) % TQ_PF_M == 0;
    }
    return n->src[1]->ne[2] <= tq_knob("GGML_TQ_MULTI_TOK", 8);
}

// ---- prefill over a matched set: ids copied to the host once, each group's (token, slot) pairs sorted once ----

struct tq_pf_sort {
    std::vector<int32_t> count, start, pairs;
    int n_valid = 0;
};

static void tq_pf_sort_group(const std::vector<char> & ids_host, const ggml_tensor * ids, const int ida, const int n_expert, tq_pf_sort & o) {
    const int n_used = ids->ne[0], n_tok = ids->ne[1];
    o.count.assign(n_expert, 0);
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const uint32_t e = *(const int32_t *) (ids_host.data() + t * ids->nb[1] + s * ids->nb[0]) - ida;
            if (e < (uint32_t) n_expert) {
                o.count[e]++;
            }
        }
    }
    o.start.assign(n_expert + 1, 0);
    for (int e = 0; e < n_expert; ++e) {
        o.start[e + 1] = o.start[e] + o.count[e];
    }
    o.n_valid = o.start[n_expert];
    o.pairs.assign(std::max(o.n_valid, 1), 0);
    std::vector<int32_t> fill(o.start.begin(), o.start.end() - 1);
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const uint32_t e = *(const int32_t *) (ids_host.data() + t * ids->nb[1] + s * ids->nb[0]) - ida;
            if (e < (uint32_t) n_expert) {
                o.pairs[fill[e]++] = t * n_used + s;
            }
        }
    }
}

// prefill over a matched set in two phases: (1) every node's inputs are rotated + gathered (GLU of (src1, xu) for the down
// chain) into one f16 buffer, (2) the fused mma of every node writes its owned slots. All inputs are read before any output
// is written: the down chain's last output may share memory with an earlier group's gate/up output (the allocator frees
// those after their own down node), which a per-node rotate-then-mma order overwrote (TQ1 perplexity 3.2973 -> 3.8406).
// ---- GPU-sorted multi prefill (GGML_TQ_GSORT, default 1): no device -> host copy of the ids, no stream sync ----
struct tq_gs_args {
    int ng, nm, n_used, n_tok;
    int ida[TQ_MM_MAX], nexp[TQ_MM_MAX], mg[TQ_MM_MAX], mK[TQ_MM_MAX];
    int nm2, mg2[TQ_MM_MAX], mK2[TQ_MM_MAX];   // GGML_TQ_GLUF: the down nodes' groups and K (offsets XOFF2)
    int big, ib64;   // GGML_TQ_PF_N64: experts with > 32 pairs get items of 64 tokens, stored from items[ib64]
    size_t nb0, nb1;
};
#define TQ_GS_MAXE 3072   // sum of the groups' experts

// one block: count pairs per (group, expert), prefix, items of 32 tokens, fill the sorted pairs (order inside an expert is
// arbitrary: every output column depends only on its own pair), node offsets into xs
static __global__ void __launch_bounds__(1024) tq_gsort(const char * __restrict__ ids, const tq_gs_args a, int32_t * __restrict__ pairs,
                                                        int4 * __restrict__ items, int * __restrict__ meta) {
    __shared__ int cnt[TQ_GS_MAXE], st[TQ_GS_MAXE], ist[TQ_GS_MAXE];
    __shared__ int gb[TQ_MM_MAX + 1];
    const int tid = threadIdx.x;
    if (tid == 0) {
        gb[0] = 0;
        for (int g = 0; g < a.ng; ++g) {
            gb[g + 1] = gb[g] + a.nexp[g];
        }
    }
    __syncthreads();
    const int T = gb[a.ng];
    for (int q = tid; q < T; q += blockDim.x) {
        cnt[q] = 0;
    }
    __syncthreads();
    const int NP = a.n_tok * a.n_used;
    for (int p = tid; p < NP; p += blockDim.x) {
        const int t = p / a.n_used, sl = p % a.n_used;
        const int e = *(const int32_t *) (ids + t * a.nb1 + sl * a.nb0);
        for (int g = 0; g < a.ng; ++g) {
            const uint32_t u = (uint32_t) (e - a.ida[g]);
            if (u < (uint32_t) a.nexp[g]) {
                atomicAdd(&cnt[gb[g] + u], 1);
                break;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int run = 0, irun = 0, irun64 = a.ib64;
        for (int g = 0; g < a.ng; ++g) {
            meta[TQM_GOFF + g] = run;
            meta[TQM_IOFF + g] = irun;
            meta[TQM_IOFF64 + g] = irun64;
            for (int q = gb[g]; q < gb[g + 1]; ++q) {
                st[q] = run;
                run  += cnt[q];
                if (a.big && cnt[q] > 32) {   // ist: the expert's first item in its own list
                    ist[q] = irun64;
                    irun64 += (cnt[q] + 63) / 64;
                } else {
                    ist[q] = irun;
                    irun += (cnt[q] + 31) / 32;
                }
            }
            meta[TQM_GCNT + g] = run - meta[TQM_GOFF + g];
            meta[TQM_ICNT + g] = irun - meta[TQM_IOFF + g];
            meta[TQM_ICNT64 + g] = irun64 - meta[TQM_IOFF64 + g];
        }
        meta[TQM_TOT] = run;
        int x = 0;
        for (int m = 0; m < a.nm; ++m) {
            meta[TQM_XOFF + m] = x;
            x += meta[TQM_GCNT + a.mg[m]] * a.mK[m];
        }
        x = 0;
        for (int m = 0; m < a.nm2; ++m) {
            meta[TQM_XOFF2 + m] = x;
            x += meta[TQM_GCNT + a.mg2[m]] * a.mK2[m];
        }
    }
    __syncthreads();
    for (int q = tid; q < T; q += blockDim.x) {
        int g = 0;
        while (q >= gb[g + 1]) {
            ++g;
        }
        const int c = cnt[q], e = q - gb[g];
        int k = ist[q];
        const int loc = st[q] - meta[TQM_GOFF + g];
        const int tn = a.big && c > 32 ? 64 : 32;
        for (int j = 0; j < c; j += tn) {
            items[k++] = make_int4(e, loc + j, min(tn, c - j), 0);
        }
    }
    __syncthreads();
    for (int q = tid; q < T; q += blockDim.x) {
        cnt[q] = st[q];
    }
    __syncthreads();
    for (int p = tid; p < NP; p += blockDim.x) {
        const int t = p / a.n_used, sl = p % a.n_used;
        const int e = *(const int32_t *) (ids + t * a.nb1 + sl * a.nb0);
        for (int g = 0; g < a.ng; ++g) {
            const uint32_t u = (uint32_t) (e - a.ida[g]);
            if (u < (uint32_t) a.nexp[g]) {
                pairs[atomicAdd(&cnt[gb[g] + u], 1)] = p;
                break;
            }
        }
    }
}

struct tq_gx_args {
    int nm, ng, n_used;
    int tok, Kmax;   // tok = n_tok: one row per token, node m at xs + m * tok * Kmax (inputs with ne11 == 1)
    float cg, cu;
    const char  * x[TQ_MM_MAX];
    const char  * xu[TQ_MM_MAX];
    const float * sign[TQ_MM_MAX];
    int K[TQ_MM_MAX], mg[TQ_MM_MAX], ne11[TQ_MM_MAX];
    size_t nb11[TQ_MM_MAX], nb12[TQ_MM_MAX];
};

// every node's input rows of every sorted pair in one launch: row j belongs to group g (meta), each node of that group
// gets x' = H64(sign * x) / 8 (GLU of (x, xu) first for the down chain) at xs[xoff[m] + local*K]
static __global__ void tq_gather_multi(const tq_gx_args a, const int32_t * __restrict__ pairs, const int * __restrict__ meta,
                                       half * __restrict__ xs) {
    const int j = blockIdx.y;
    int g = 0, local, t, sl;
    if (a.tok) {
        local = t = j; sl = 0; g = -1;
    } else {
        if (j >= meta[TQM_TOT]) {
            return;
        }
        while (g + 1 < a.ng && j >= meta[TQM_GOFF + g + 1]) {
            ++g;
        }
        local = j - meta[TQM_GOFF + g];
        const int p = pairs[j];
        t = p / a.n_used; sl = p % a.n_used;
    }
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    for (int m = 0; m < a.nm; ++m) {
        const int K = a.K[m];
        if ((g >= 0 && a.mg[m] != g) || b * 64 >= K) {
            continue;
        }
        const size_t off = t * a.nb12[m] + (sl % a.ne11[m]) * a.nb11[m];
        const float * xi = (const float *) (a.x[m] + off);
        float v0 = xi[b*64 + lane], v1 = xi[b*64 + 32 + lane];
        if (a.xu[m]) {
            const float * ui = (const float *) (a.xu[m] + off);
            v0 = tq_act(v0, ui[b*64 + lane], a.cg, a.cu);
            v1 = tq_act(v1, ui[b*64 + 32 + lane], a.cg, a.cu);
        }
        float a0 = v0 * a.sign[m][b*64 + lane];
        float a1 = v1 * a.sign[m][b*64 + 32 + lane];
        { const float u = a0 + a1, v = a0 - a1; a0 = u; a1 = v; }
#pragma unroll
        for (int h = 16; h >= 1; h >>= 1) {
            const float o0 = __shfl_xor_sync(0xffffffff, a0, h);
            const float o1 = __shfl_xor_sync(0xffffffff, a1, h);
            const bool hi = lane & h;
            a0 = hi ? o0 - a0 : a0 + o0;
            a1 = hi ? o1 - a1 : a1 + o1;
        }
        half * xo = xs + (a.tok ? (size_t) m * a.tok * a.Kmax : (size_t) meta[TQM_XOFF + m]) + (size_t) local * K;
        xo[b*64 + lane]      = __float2half(a0 * 0.125f);
        xo[b*64 + 32 + lane] = __float2half(a1 * 0.125f);
    }
}

int ggml_cuda_tq_down_match(const ggml_cgraph * cgraph, const int i);

// GGML_TQ_GLUF: what the gate/up call leaves for the down call of the same layer (persistent buffers: the down call reuses
// the gate/up sort, its pair order inside an expert is arbitrary but must be the same one)
struct tq_gluf_state {
    int32_t * pairs = nullptr; int4 * items = nullptr; int * meta = nullptr; half * xd = nullptr;
    size_t cap_p = 0, cap_i = 0, cap_x = 0;
    const ggml_tensor * ids = nullptr, * down0 = nullptr;
    int nd = 0, mg[TQ_MM_MAX] = {}, K[TQ_MM_MAX] = {};
    int big = 0, nexp[TQ_MM_MAX] = {};   // GGML_TQ_PF_N64 items of the sort, experts of the down nodes' groups
    bool ready = false;
};
static tq_gluf_state g_gluf;

template <typename T>
static bool tq_gluf_grow(T ** p, size_t & cap, const size_t need, cudaStream_t stream) {
    if (need <= cap) {
        return true;
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
    if (st != cudaStreamCaptureStatusNone) {
        return false;   // no allocation inside a graph capture: this call goes the plain way
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (*p) {
        CUDA_CHECK(cudaFree(*p));
    }
    const size_t n = need + need / 4;
    CUDA_CHECK(cudaMalloc((void **) p, n * sizeof(T)));
    cap = n;
    return true;
}

// GGML_TQ_PF_N64 (default 1024): with at least this many tokens in the batch, experts with more than 32 pairs get tiles of 64
// tokens (the weight tile is decoded once per 64 instead of per 32 tokens; same sums, bit-identical); 0 = off
static int tq_pf_n64_min() {
    static const int v = [] { const char * e = getenv("GGML_TQ_PF_N64"); return e ? atoi(e) : 1024; }();
    return v;
}

// returns false (nothing launched) when the set does not fit the GPU-sorted path
static bool tq_pf_multi_gpu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const int n, const bool down,
                            float * red = nullptr, const float * rw = nullptr, const float * rsc = nullptr) {
    static const int on = [] { const char * v = getenv("GGML_TQ_GSORT"); return v ? atoi(v) : 1; }();
    if (!on || tq_pf_mc() > 0 || tq_pf_nmax() > 32 || n > TQ_MM_MAX) {
        return false;
    }
    const ggml_tensor * n0 = cgraph->nodes[i];
    const ggml_tensor * ids = n0->src[2];
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    if (ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (down) {
        const bool use = g_gluf.ready && g_gluf.down0 == n0 && g_gluf.nd == n && g_gluf.ids == ids;
        g_gluf.ready = false;
        if (use) {   // inputs made by the up kernels, sort of the gate/up call
            cudaStream_t stream = ctx.stream();
            const int NP = ids->ne[0] * ids->ne[1];
            if (red) {
                CUDA_CHECK(cudaMemsetAsync(red, 0, (size_t) ids->ne[1] * last->ne[0] * sizeof(float), stream));
            }
            const size_t smem32 = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + 32) * TQ_PF_LD * 2;
            const size_t smem64 = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + 64) * TQ_PF_LD * 2;
            CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem32));
            CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<64, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem64));
            for (int m = 0; m < n; ++m) {
                const ggml_tensor * d = cgraph->nodes[i + m];
                const ggml_tensor * src0 = d->src[0];
                const int rows = src0->ne[1] * 16;
                const int ne = (int) src0->ne[2];
                const dim3 grid(rows / TQ_PF_M, g_gluf.big ? ne + 1 : NP / 32 + ne + 1);
                tq_prefill_mma<32, true><<<grid, 128, smem32, stream>>>((const uint8_t *) src0->data, (const half *) d->src[3]->data,
                    g_gluf.xd, g_gluf.items, g_gluf.pairs, (char *) last->data, g_gluf.K[m], rows, src0->nb[2], (int) ids->ne[0],
                    last->nb[1], last->nb[2], tq_knob_pf_dbg(), g_gluf.meta, g_gluf.mg[m], 16 + m, 0, red, rw, rsc);
                CUDA_CHECK(cudaGetLastError());
                if (g_gluf.big) {
                    const dim3 grid64(rows / TQ_PF_M, NP / 64 + std::min(ne, NP / 33) + 1);
                    tq_prefill_mma<64, true><<<grid64, 128, smem64, stream>>>((const uint8_t *) src0->data, (const half *) d->src[3]->data,
                        g_gluf.xd, g_gluf.items, g_gluf.pairs, (char *) last->data, g_gluf.K[m], rows, src0->nb[2], (int) ids->ne[0],
                        last->nb[1], last->nb[2], tq_knob_pf_dbg(), g_gluf.meta, g_gluf.mg[m], 16 + m, 0, red, rw, rsc);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            return true;
        }
    }
    tq_gs_args a = {};
    a.n_used = ids->ne[0];
    a.n_tok  = ids->ne[1];
    a.nb0 = ids->nb[0];
    a.nb1 = ids->nb[1];
    a.nm = n;
    int Kmax = 0, etot = 0;
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const int ida = d->op_params[12];
        int g = 0;
        while (g < a.ng && a.ida[g] != ida) {
            ++g;
        }
        if (g == a.ng) {
            a.ida[g] = ida;
            a.nexp[g] = d->src[0]->ne[2];
            etot += a.nexp[g];
            a.ng++;
        }
        a.mg[m] = g;
        a.mK[m] = d->src[0]->ne[0] / 16;
        if (a.mK[m] % 64 != 0 || (d->src[0]->ne[1] * 16) % TQ_PF_M != 0) {
            return false;
        }
        Kmax = std::max(Kmax, a.mK[m]);
    }
    if (etot > TQ_GS_MAXE) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int NP = a.n_tok * a.n_used;
    const int J = n / a.ng;   // nodes per group (gate + up: 2, down: 1)

    // GGML_TQ_GLUF plan (gate/up call): the down set right after, one down node per group reading this group's two outputs
    static const int gluf_on = tq_knob("GGML_TQ_GLUF", 1);
    tq_glu_args glu[TQ_MM_MAX];   // per node m; xd set only on the later-launched node of each group
    bool plan = false;
    int jd = -1;
    if (gluf_on && !down && !red && J == 2) {
        for (int j = i + n; j < cgraph->n_nodes && j < i + n + 32; ++j) {
            const ggml_tensor * c = cgraph->nodes[j];
            if (c->op == GGML_OP_MUL_MAT_ID && c->src[0]->type == GGML_TYPE_TQ2T) {
                jd = j;
                break;
            }
        }
        const int nd = jd >= 0 ? ggml_cuda_tq_down_match(cgraph, jd) : 0;
        plan = nd == a.ng;
        a.nm2 = nd;
        size_t xneed = 0;
        for (int k = 0; k < nd && plan; ++k) {
            const ggml_tensor * dn = cgraph->nodes[jd + k];
            int mx = -1, mu = -1;
            for (int m = 0; m < n; ++m) {
                mx = cgraph->nodes[i + m] == dn->src[1] ? m : mx;
                mu = cgraph->nodes[i + m] == dn->src[5] ? m : mu;
            }
            const int Kd = dn->src[0]->ne[0] / 16;
            plan = mx >= 0 && mu >= 0 && mx != mu && a.mg[mx] == a.mg[mu] && dn->src[2] == ids && Kd % 64 == 0 &&
                   cgraph->nodes[i + mx]->src[0]->ne[1] * 16 == Kd && cgraph->nodes[i + mu]->src[0]->ne[1] * 16 == Kd &&
                   dn->src[1]->type == GGML_TYPE_F32 && dn->src[1]->ne[1] == a.n_used && dn->src[4] &&
                   ggml_nelements(dn->src[4]) == Kd;
            if (!plan) {
                break;
            }
            const int fm = std::max(mx, mu), om = std::min(mx, mu);
            const ggml_tensor * other = cgraph->nodes[i + om];
            tq_glu_args & gl = glu[fm];
            gl.other = (const char *) other->data;
            gl.onb1 = other->nb[1];
            gl.onb2 = other->nb[2];
            gl.sign = (const float *) dn->src[4]->data;
            memcpy(&gl.cg, &dn->op_params[10], sizeof(float));
            memcpy(&gl.cu, &dn->op_params[11], sizeof(float));
            gl.Kd = Kd;
            gl.mi = k;
            gl.self_is_u = fm == mu;
            a.mg2[k] = a.mg[mx];
            a.mK2[k] = Kd;
            g_gluf.mg[k] = a.mg[mx];
            g_gluf.K[k] = Kd;
            xneed = std::max(xneed, (size_t) Kd);
        }
        static size_t cap_m = 0;
        plan = plan && tq_gluf_grow(&g_gluf.pairs, g_gluf.cap_p, (size_t) NP, stream) &&
               tq_gluf_grow(&g_gluf.items, g_gluf.cap_i, (size_t) (NP / 32 + NP / 64 + 2 * etot + 16), stream) &&
               tq_gluf_grow(&g_gluf.meta, cap_m, (size_t) TQM_N, stream) &&
               tq_gluf_grow(&g_gluf.xd, g_gluf.cap_x, (size_t) NP * xneed, stream);
        if (!plan) {
            a.nm2 = 0;
            for (int m = 0; m < n; ++m) {
                glu[m] = tq_glu_args();
            }
        }
    }
    g_gluf.ready = false;
    ggml_cuda_pool_alloc<int32_t> pairs_pool(ctx.pool());
    ggml_cuda_pool_alloc<int4>    items_pool(ctx.pool());
    ggml_cuda_pool_alloc<int>     meta_pool(ctx.pool());
    int32_t * pairs_p; int4 * items_p; int * meta_p;
    if (plan) {
        pairs_p = g_gluf.pairs; items_p = g_gluf.items; meta_p = g_gluf.meta;
    } else {
        pairs_p = pairs_pool.alloc(NP); items_p = items_pool.alloc(NP / 32 + NP / 64 + 2 * etot + 16); meta_p = meta_pool.alloc(TQM_N);
    }
    ggml_cuda_pool_alloc<half>    xs(ctx.pool(), (size_t) J * NP * Kmax);
    a.big  = tq_pf_n64_min() > 0 && a.n_tok >= tq_pf_n64_min();
    a.ib64 = NP / 32 + etot + 8;
    tq_gsort<<<1, 1024, 0, stream>>>((const char *) ids->data, a, pairs_p, items_p, meta_p);
    CUDA_CHECK(cudaGetLastError());
    // GGML_TQ_ITEM_STATS=N (measurement only, syncs the stream): every N sorts with >= 256 tokens print the mean items of
    // 32 / 64 tokens a sort, the tokens (pairs) a sort and the fill of the items (pairs / (items x tile)) over all groups
    static const int istats = tq_knob("GGML_TQ_ITEM_STATS", 0);
    cudaStreamCaptureStatus istats_cs = cudaStreamCaptureStatusNone;
    if (istats) {
        CUDA_CHECK(cudaStreamIsCapturing(stream, &istats_cs));   // no sync / copy inside a CUDA graph capture
    }
    if (istats && a.n_tok >= 256 && istats_cs == cudaStreamCaptureStatusNone) {
        static double s_i32 = 0, s_i64 = 0, s_p = 0, s_hot = 0; static long s_n = 0;
        int hm[TQM_N];
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(hm, meta_p, sizeof(hm), cudaMemcpyDeviceToHost));
        long i32 = 0, i64 = 0;
        for (int g = 0; g < a.ng; ++g) { i32 += hm[TQM_ICNT + g]; i64 += hm[TQM_ICNT64 + g]; }
        // items of 32 holding a full tile: experts with >= 32 pairs (when not big)
        std::vector<int4> hi(i32 > 0 ? (size_t) i32 : 1);
        long hot = 0;
        for (int g = 0; g < a.ng; ++g) {
            const int c = hm[TQM_ICNT + g];
            if (c <= 0) continue;
            std::vector<int4> tmp(c);
            CUDA_CHECK(cudaMemcpy(tmp.data(), items_p + hm[TQM_IOFF + g], c * sizeof(int4), cudaMemcpyDeviceToHost));
            for (const int4 & it : tmp) { hot += it.z == 32; }
        }
        s_i32 += i32; s_i64 += i64; s_p += hm[TQM_TOT]; s_hot += hot; ++s_n;
        if (s_n % istats == 0) {
            const double items = (s_i32 + s_i64) / s_n, fill = (s_p / s_n) / (s_i32 / s_n * 32 + s_i64 / s_n * 64);
            fprintf(stderr, "tq_item_stats: %ld sorts (n_tok %d, groups %d): items32 %.1f items64 %.1f (full32 %.1f) pairs %.1f fill %.3f items/sort %.1f\n",
                    s_n, a.n_tok, a.ng, s_i32 / s_n, s_i64 / s_n, s_hot / s_n, s_p / s_n, fill, items);
            s_i32 = s_i64 = s_p = s_hot = 0; s_n = 0;
        }
    }

    tq_gx_args gx = {};
    gx.nm = n; gx.ng = a.ng; gx.n_used = a.n_used;
    gx.cg = 1.0f; gx.cu = 1.0f;
    static const int tokon = tq_knob("GGML_TQ_TOKX", 1);
    bool tokx = tokon && !down;
    for (int m = 0; m < n && tokx; ++m) {
        tokx = cgraph->nodes[i + m]->src[1]->ne[1] == 1;
    }
    if (tokx) {
        gx.tok = a.n_tok; gx.Kmax = Kmax;
    }
    if (down) {
        memcpy(&gx.cg, &n0->op_params[10], sizeof(float));
        memcpy(&gx.cu, &n0->op_params[11], sizeof(float));
    }
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const ggml_tensor * src1 = d->src[1];
        gx.x[m]    = (const char *) src1->data;
        gx.xu[m]   = down ? (const char *) d->src[5]->data : nullptr;
        gx.sign[m] = (const float *) d->src[4]->data;
        gx.K[m]    = a.mK[m];
        gx.mg[m]   = a.mg[m];
        gx.ne11[m] = src1->ne[1];
        gx.nb11[m] = src1->nb[1];
        gx.nb12[m] = src1->nb[2];
    }
    {
        const dim3 grid((Kmax / 64 + 3) / 4, tokx ? a.n_tok : NP);
        tq_gather_multi<<<grid, 128, 0, stream>>>(gx, pairs_p, meta_p, xs.ptr);
        CUDA_CHECK(cudaGetLastError());
    }
    if (red) {   // after the gather: the sum's buffer may be the memory of the gate / up outputs it reads
        CUDA_CHECK(cudaMemsetAsync(red, 0, (size_t) a.n_tok * last->ne[0] * sizeof(float), stream));
    }
    const size_t smem   = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + 32) * TQ_PF_LD * 2;
    const size_t smem64 = 4 * TQ_PF_SEG + TQ_PF_M * 4 + (TQ_PF_M + 64) * TQ_PF_LD * 2;
    static bool smem_set = false;
    if (!smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(tq_prefill_mma<64, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem64));
        smem_set = true;
    }
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const ggml_tensor * src0 = d->src[0];
        const int K = a.mK[m], rows = src0->ne[1] * 16;
        char * out = down ? (char *) last->data : (char *) d->data;
        const size_t onb1 = down ? last->nb[1] : d->nb[1], onb2 = down ? last->nb[2] : d->nb[2];
        const int ne = a.nexp[a.mg[m]];
        const dim3 grid(rows / TQ_PF_M, a.big ? ne + 1 : NP / 32 + ne + 1);
        const half * xm = tokx ? xs.ptr + (size_t) m * a.n_tok * Kmax : xs.ptr;
        tq_glu_args gl = glu[m];
        if (gl.other) {
            gl.xd = g_gluf.xd;
        }
        tq_prefill_mma<32, true><<<grid, 128, smem, stream>>>((const uint8_t *) src0->data, (const half *) d->src[3]->data, xm, items_p,
            pairs_p, out, K, rows, src0->nb[2], a.n_used, onb1, onb2, tq_knob_pf_dbg(), meta_p, a.mg[m], m, tokx ? a.n_used : 0,
            red, rw, rsc, gl);
        CUDA_CHECK(cudaGetLastError());
        if (a.big) {
            const dim3 grid64(rows / TQ_PF_M, NP / 64 + std::min(ne, NP / 33) + 1);
            tq_prefill_mma<64, true><<<grid64, 128, smem64, stream>>>((const uint8_t *) src0->data, (const half *) d->src[3]->data, xm,
                items_p, pairs_p, out, K, rows, src0->nb[2], a.n_used, onb1, onb2, tq_knob_pf_dbg(), meta_p, a.mg[m], m,
                tokx ? a.n_used : 0, red, rw, rsc, gl);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (plan) {
        g_gluf.ready = true;
        g_gluf.big = a.big;
        g_gluf.ids = ids;
        g_gluf.down0 = cgraph->nodes[jd];
        g_gluf.nd = a.nm2;
    }
    return true;
}

bool ggml_cuda_tq_pf_graph_ok(const ggml_tensor * dst) {
    static const int on = [] {
        const char * g = getenv("GGML_TQ_GSORT"), * v = getenv("GGML_TQ_PF_GRAPH");
        const char * mg = getenv("GGML_TQ_MAGIC");
        return (!g || atoi(g) != 0) && (!v || atoi(v) != 0) && (!mg || atoi(mg) != 0);
    }();
    return on && tq_pf_mc() == 0 && tq_pf_nmax() <= 32 && tq_mm_node_ok(dst);
}

// a host-synced prefill path must not run while a CUDA graph is being captured
static void tq_assert_not_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
    if (st != cudaStreamCaptureStatusNone) {
        GGML_ABORT("tq: host-synced prefill path while capturing a CUDA graph (set GGML_TQ_PF_GRAPH=0)");
    }
}

struct tq_pf_timer {   // GGML_TQ_TIME=1 (measurement only): host time of tq_pf_multi after its sync, printed every 96 calls
    std::chrono::steady_clock::time_point t0;
    bool on;
    tq_pf_timer() : on(tq_knob("GGML_TQ_TIME", 0) != 0) {}
    void start() { if (on) t0 = std::chrono::steady_clock::now(); }
    ~tq_pf_timer() {
        if (!on) return;
        static double acc = 0; static int cnt = 0;
        acc += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (++cnt % 96 == 0) { fprintf(stderr, "tq_pf_multi host: %.1f us/call (avg of 96)\n", acc / 96); acc = 0; }
    }
};

static void tq_pf_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const int n, const bool down) {
    static const int magic = [] { const char * v = getenv("GGML_TQ_MAGIC"); return v ? atoi(v) : 1; }();
    if (magic && tq_pf_multi_gpu(ctx, cgraph, i, n, down)) {
        return;
    }
    tq_assert_not_capturing(ctx.stream());
    tq_pf_timer tmr;
    const ggml_tensor * n0 = cgraph->nodes[i];
    const ggml_tensor * ids = n0->src[2];
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    const int n_used = ids->ne[0];
    cudaStream_t stream = ctx.stream();
    std::vector<char> ids_host(ggml_nbytes(ids));
    if (tq_pin()) {
        char * pin = tq_pin_buf(0, ggml_nbytes(ids));
        CUDA_CHECK(cudaMemcpyAsync(pin, ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        tq_sync_and_release(stream);
        tmr.start();
        memcpy(ids_host.data(), pin, ggml_nbytes(ids));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        tq_sync_and_release(stream);
    }
    float cg = 1.0f, cu = 1.0f;
    if (down) {
        memcpy(&cg, &n0->op_params[10], sizeof(float));
        memcpy(&cu, &n0->op_params[11], sizeof(float));
    }
    std::vector<int> idas, which(n);
    std::vector<tq_pf_sort> sorts;
    sorts.reserve(n);
    std::vector<size_t> poff(n), xoff(n);
    size_t ptot = 0, xtot = 0;
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const int ida = d->op_params[12];
        size_t k = 0;
        while (k < idas.size() && idas[k] != ida) {
            ++k;
        }
        if (k == idas.size()) {
            idas.push_back(ida);
            sorts.emplace_back();
            tq_pf_sort_group(ids_host, ids, ida, d->src[0]->ne[2], sorts.back());
        }
        which[m] = (int) k;
        poff[m] = ptot;
        xoff[m] = xtot;
        ptot += sorts[k].n_valid;
        xtot += (size_t) sorts[k].n_valid * (d->src[0]->ne[0] / 16);
    }
    if (ptot == 0) {
        return;
    }
    std::vector<int32_t> pall(ptot);
    for (int m = 0; m < n; ++m) {
        const tq_pf_sort & so = sorts[which[m]];
        std::copy(so.pairs.begin(), so.pairs.begin() + so.n_valid, pall.begin() + poff[m]);
    }
    // every node's launch items (N = 32 / 64 / 128), built before any upload
    std::vector<std::vector<int4>> its(3 * n);
    if (tq_pf_mc() == 0) {
        for (int m = 0; m < n; ++m) {
            const tq_pf_sort & so = sorts[which[m]];
            const int n_expert = cgraph->nodes[i + m]->src[0]->ne[2];
            std::vector<int4> & it32 = its[3*m + 0], & it64 = its[3*m + 1], & it128 = its[3*m + 2];
            for (int e = 0; e < n_expert && so.n_valid > 0; ++e) {
                const int c = so.count[e];
                if (c == 0) {
                    continue;
                } else if (tq_pf_nmax() <= 32) {
                    for (int j = 0; j < c; j += 32) {
                        it32.push_back(make_int4(e, so.start[e] + j, std::min(32, c - j), 0));
                    }
                } else if (c <= 32) {
                    it32.push_back(make_int4(e, so.start[e], c, 0));
                } else if (c <= 64) {
                    it64.push_back(make_int4(e, so.start[e], c, 0));
                } else if (tq_pf_nmax() <= 64) {
                    for (int j = 0; j < c; j += 64) {
                        it64.push_back(make_int4(e, so.start[e] + j, std::min(64, c - j), 0));
                    }
                } else {
                    for (int j = 0; j < c; j += 128) {
                        it128.push_back(make_int4(e, so.start[e] + j, std::min(128, c - j), 0));
                    }
                }
            }
        }
    }
    const bool pin = tq_pin() && tq_pf_mc() == 0;
    const size_t pbytes = (ptot * sizeof(int32_t) + 15) / 16 * 16;
    std::vector<size_t> ioff(3 * n, 0);
    size_t ibytes = 0;
    for (int q = 0; q < 3 * n; ++q) {
        ioff[q] = ibytes;
        ibytes += its[q].size() * sizeof(int4);
    }
    ggml_cuda_pool_alloc<char> up_d(ctx.pool(), pin ? pbytes + ibytes : 16);
    ggml_cuda_pool_alloc<int32_t> pairs_d(ctx.pool(), pin ? 1 : ptot);
    int32_t * pairs_ptr = pairs_d.ptr;
    if (pin) {
        char * h = tq_pin_buf(1, pbytes + ibytes);
        memcpy(h, pall.data(), ptot * sizeof(int32_t));
        for (int q = 0; q < 3 * n; ++q) {
            if (!its[q].empty()) {
                memcpy(h + pbytes + ioff[q], its[q].data(), its[q].size() * sizeof(int4));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(up_d.ptr, h, pbytes + ibytes, cudaMemcpyHostToDevice, stream));
        pairs_ptr = (int32_t *) up_d.ptr;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(pairs_d.ptr, tq_keep(pall.data(), ptot * sizeof(int32_t)), ptot * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   stream));
    }
    ggml_cuda_pool_alloc<half> xs(ctx.pool(), xtot);
    // phase 1: all inputs. GGML_TQ_GU2 (default 1): a gate / up pair of one group (same input, same pairs, same K) in one
    // kernel that reads the input once
    static const int gu2 = [] { const char * v = getenv("GGML_TQ_GU2"); return v ? atoi(v) : 1; }();
    std::vector<int> mate(n, -1);
    if (!down && gu2) {
        for (int m = 0; m < n; ++m) {
            for (int m2 = m + 1; m2 < n && mate[m] < 0; ++m2) {
                const ggml_tensor * a = cgraph->nodes[i + m], * b = cgraph->nodes[i + m2];
                if (mate[m2] < 0 && which[m2] == which[m] && a->src[1] == b->src[1] && a->src[0]->ne[0] == b->src[0]->ne[0]) {
                    mate[m] = m2; mate[m2] = m;
                }
            }
        }
    }
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const tq_pf_sort & so = sorts[which[m]];
        if (so.n_valid == 0) {
            continue;
        }
        if (mate[m] >= 0) {
            if (mate[m] < m) {
                continue;   // done with its mate
            }
            const ggml_tensor * d2 = cgraph->nodes[i + mate[m]];
            const ggml_tensor * src1 = d->src[1];
            const int K = d->src[0]->ne[0] / 16;
            const dim3 grid((K / 64 + 3) / 4, so.n_valid);
            tq_rotate_gather2_f16<<<grid, 128, 0, stream>>>((const char *) src1->data, (const float *) d->src[4]->data,
                (const float *) d2->src[4]->data, pairs_ptr + poff[m], xs.ptr + xoff[m], xs.ptr + xoff[mate[m]], K, n_used,
                src1->ne[1], src1->nb[1], src1->nb[2]);
            CUDA_CHECK(cudaGetLastError());
            continue;
        }
        const ggml_tensor * src1 = d->src[1];
        const int K = d->src[0]->ne[0] / 16;
        const dim3 grid((K / 64 + 3) / 4, so.n_valid);
        tq_rotate_gather_f16<<<grid, 128, 0, stream>>>((const char *) src1->data, down ? (const char *) d->src[5]->data : nullptr, cg, cu,
                                                       (const float *) d->src[4]->data, pairs_ptr + poff[m], xs.ptr + xoff[m], K, n_used,
                                                       src1->ne[1], src1->nb[1], src1->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }
    // phase 2: all outputs
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const tq_pf_sort & so = sorts[which[m]];
        if (so.n_valid == 0) {
            continue;
        }
        const ggml_tensor * src0 = d->src[0];
        const int K = src0->ne[0] / 16, rows = src0->ne[1] * 16, n_expert = src0->ne[2];
        char * out = down ? (char *) last->data : (char *) d->data;
        const size_t onb1 = down ? last->nb[1] : d->nb[1], onb2 = down ? last->nb[2] : d->nb[2];
        if (tq_pf_mc() > 0) {
            const int cap = 32 * (tq_pf_mc() >= 4 ? 4 : 2);
            std::vector<int4> itm;
            for (int e = 0; e < n_expert; ++e) {
                for (int j = 0; j < so.count[e]; j += cap) {
                    itm.push_back(make_int4(e, so.start[e] + j, std::min(cap, so.count[e] - j), 0));
                }
            }
            ggml_cuda_pool_alloc<int4> itm_d(ctx.pool());
            tq_prefill_launch_mc(itm, itm_d, stream, src0, d->src[3], xs.ptr + xoff[m], pairs_ptr + poff[m], out, onb1, onb2, K, rows, n_used);
            continue;
        }
        GGML_UNUSED(n_expert);
        const int32_t * pd = pairs_ptr + poff[m];
        const half * xd = xs.ptr + xoff[m];
        if (pin) {
            const int4 * ib = (const int4 *) (up_d.ptr + pbytes);
            tq_prefill_launch_dev<32> (ib + ioff[3*m + 0] / sizeof(int4), (int) its[3*m + 0].size(), stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
            tq_prefill_launch_dev<64> (ib + ioff[3*m + 1] / sizeof(int4), (int) its[3*m + 1].size(), stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
            tq_prefill_launch_dev<128>(ib + ioff[3*m + 2] / sizeof(int4), (int) its[3*m + 2].size(), stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
        } else {
            ggml_cuda_pool_alloc<int4> items32_d(ctx.pool()), items64_d(ctx.pool()), items128_d(ctx.pool());
            tq_prefill_launch<32> (its[3*m + 0], items32_d,  stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
            tq_prefill_launch<64> (its[3*m + 1], items64_d,  stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
            tq_prefill_launch<128>(its[3*m + 2], items128_d, stream, src0, d->src[3], xd, pd, out, onb1, onb2, K, rows, n_used);
        }
    }
}

int ggml_cuda_tq_gu_match(const ggml_cgraph * cgraph, const int i) {
    static const int on = tq_knob("GGML_TQ_MULTI", 1);
    if (!on) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    auto ok = [&](const ggml_tensor * n) {
        return tq_mm_node_ok(n) && n->op_params[13] != GGML_CUDA_RVQ_GLU_MAGIC && !n->src[6] && n->src[1]->ne[1] == 1 &&
               n->src[1] == first->src[1] && n->src[2] == first->src[2] && n->src[0]->ne[0] == first->src[0]->ne[0];
    };
    if (!ok(first)) {
        return 0;
    }
    int n = 1;
    while (n < TQ_MM_MAX && i + n < cgraph->n_nodes && ok(cgraph->nodes[i + n])) {
        ++n;
    }
    // every group (ida) must appear the same number of times, so each slot has J owners
    int J = 0;
    for (int a = 0; a < n; ++a) {
        int c = 0;
        for (int b = 0; b < n; ++b) {
            c += cgraph->nodes[i + b]->op_params[12] == cgraph->nodes[i + a]->op_params[12];
        }
        if (J == 0) {
            J = c;
        } else if (c != J) {
            return 0;
        }
    }
    return n >= 2 && J >= 1 && n / J >= 2 ? n : 0;
}

int ggml_cuda_tq_down_match(const ggml_cgraph * cgraph, const int i) {
    static const int on = tq_knob("GGML_TQ_DOWN_MULTI", 1);
    if (!on) {
        return 0;
    }
    auto ok = [&](const ggml_tensor * n, const ggml_tensor * prev, const ggml_tensor * first) {
        if (!tq_mm_node_ok(n) || n->op_params[13] != GGML_CUDA_RVQ_GLU_MAGIC || !n->src[5] || n->src[6] != prev) {
            return false;
        }
        const ggml_tensor * g = n->src[1], * u = n->src[5];
        if (u->type != GGML_TYPE_F32 || !ggml_are_same_shape(g, u) || g->nb[1] != u->nb[1] || g->nb[2] != u->nb[2]) {
            return false;
        }
        return !first || (n->src[2] == first->src[2] && n->src[0]->ne[1] == first->src[0]->ne[1] && n->ne[0] == first->ne[0] &&
                          memcmp(&n->op_params[10], &first->op_params[10], 2 * sizeof(int32_t)) == 0);
    };
    const ggml_tensor * first = cgraph->nodes[i];
    if (!ok(first, first->src[6], nullptr) || first->src[6]) {
        return 0;   // the chain starts with the first group (no src[6])
    }
    int n = 1;
    while (n < TQ_MM_MAX && i + n < cgraph->n_nodes && ok(cgraph->nodes[i + n], cgraph->nodes[i + n - 1], first)) {
        ++n;
    }
    if (n < 2) {
        return 0;
    }
    int idxs[TQ_MM_MAX];
    ggml_op ops[TQ_MM_MAX];
    for (int k = 0; k < n; ++k) {
        idxs[k] = i + k;
        ops[k]  = GGML_OP_MUL_MAT_ID;
    }
    const int outputs[] = { i + n - 1 };
    return ggml_can_fuse_subgraph_ext(cgraph, idxs, n, ops, outputs, 1) ? n : 0;
}

static void tq_mm_launch(ggml_backend_cuda_context & ctx, const tq_mm_args & a, const ggml_tensor * ids, const int J,
                         const float cg, const float cu) {
    const int n_used  = ids->ne[0];
    const int n_tok   = ids->ne[1];
    const int n_pairs = n_tok * n_used;
    int Kmax = 0, rmax = 0;
    for (int m = 0; m < a.n_m; ++m) {
        Kmax = std::max(Kmax, a.m[m].K);
        rmax = std::max(rmax, a.m[m].rows);
    }
    cudaStream_t stream = ctx.stream();
    // GGML_TQ_Q8 (default 0 until measured): int8 inputs with one scale per 16 for the decode kernel (one dp4a a weight)
    static const int q8 = tq_knob("GGML_TQ_Q8", 0);
    // GGML_TQ_STATS=1 (measurement only, needs GGML_CUDA_DISABLE_GRAPHS=1): weight bytes read per single-token launch
    static const int stats = tq_knob("GGML_TQ_STATS", 0);
    if (stats && n_tok == 1) {
        static double tot_bytes = 0; static long n_launch = 0; static long gcount[TQ_MM_MAX] = {0};
        static bool reg = false;
        if (!reg) { reg = true; atexit([] { fprintf(stderr, "TQ_STATS launches=%ld avg_bytes=%.3f MB  group hits:", n_launch, n_launch ? tot_bytes / n_launch / 1e6 : 0.0);
            for (int g = 0; g < TQ_MM_MAX; ++g) { fprintf(stderr, " %ld", gcount[g]); } fprintf(stderr, "\n"); }); }
        std::vector<int32_t> h(n_used);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int s = 0; s < n_used; ++s) {
            CUDA_CHECK(cudaMemcpy(&h[s], (const char *) ids->data + s * ids->nb[0], 4, cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < n_used; ++s) {
            for (int m = 0; m < a.n_m; ++m) {
                if ((uint32_t) (h[s] - a.m[m].ida) < (uint32_t) a.m[m].n_expert) {
                    tot_bytes += (double) a.m[m].K * a.m[m].rows * 66 / 256;
                    gcount[m]++;
                }
            }
        }
        n_launch++;
    }
    ggml_cuda_pool_alloc<float> xr(ctx.pool(), (size_t) J * n_pairs * Kmax * (q8 ? 18 : 16) / 16);
    {
        const dim3 grid((Kmax / 64 + 3) / 4, n_pairs, J);
        tq_rotate_mm<<<grid, 128, 0, stream>>>(a, (const char *) ids->data, xr.ptr, Kmax, n_used, n_pairs, ids->nb[0], ids->nb[1], cg, cu,
                                               q8 ? 2 : tq_xh());
        CUDA_CHECK(cudaGetLastError());
    }
    // K split from the smallest K of the set (all gate/up share K; downs differ): 2 unless a matrix has < 64 tiles
    int kmin = Kmax;
    for (int m = 0; m < a.n_m; ++m) {
        kmin = std::min(kmin, a.m[m].K);
    }
    static const int ks_env = tq_knob("GGML_TQ_KS", 0);
    const int ks   = ks_env ? ks_env : (kmin / 16 >= 64 ? 2 : (Kmax / 16 >= 64 ? 2 : 1));
    const int spb  = 4 / ks;
    const int nblk = (rmax / 16 + spb - 1) / spb;
    static const int zf_env = tq_knob("GGML_TQ_ZF", 1);
    const int zf = zf_env && n_tok > 1;
    const dim3 grid = zf ? dim3(n_pairs, J * nblk) : dim3(J * nblk, n_pairs);
#define TQ_MM_ARGS a, xr.ptr, (const char *) ids->data, Kmax, nblk, n_used, n_pairs, ids->nb[0], ids->nb[1], zf
    // GGML_TQ_MAGIC_TG (default 0): int -> float of the byte sums by bit tricks instead of I2F (same values; measured
    // tg64 27.57 -> 27.39 with it, so off); GGML_TQ_PREF (default 1): prefetch the next chunk into registers
    static const int magic = tq_knob("GGML_TQ_MAGIC_TG", 0);
    static const int pref  = tq_knob("GGML_TQ_PREF", 1);
    static const int pdl   = tq_knob("GGML_TQ_PDL", 1);
    static const int v16   = tq_knob("GGML_TQ_V16", 0);
    const int xh = tq_xh();
#define TQ_MM_PF(ks, mg) (xh ? (pref ? tq_mmv_mm<ks, mg, true, true> : tq_mmv_mm<ks, mg, false, true>) : (pref ? tq_mmv_mm<ks, mg, true> : tq_mmv_mm<ks, mg, false>))
    // GGML_TQ_MMT (default 0): with 2..4 tokens, slots that share an expert share one block (tile read once)
    static const int mmt = tq_knob("GGML_TQ_MMT", 0);   // measured slower: pp4 72.3 -> 57.8 (167 regs, few shared experts)
    if (mmt && !q8 && n_tok >= 2 && n_tok <= 4) {
        if (ks == 1) {
            tq_mmv_mmt<1, 4><<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else {
            tq_mmv_mmt<2, 4><<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (q8) {
        if (ks == 4) {
            (pref ? tq_mmv_mm<4, false, true, false, true> : tq_mmv_mm<4, false, false, false, true>)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else if (ks == 2) {
            (pref ? tq_mmv_mm<2, false, true, false, true> : tq_mmv_mm<2, false, false, false, true>)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else {
            (pref ? tq_mmv_mm<1, false, true, false, true> : tq_mmv_mm<1, false, false, false, true>)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        }
    } else if (magic) {
        if (ks == 4) {
            TQ_MM_PF(4, true)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else if (ks == 2) {
            TQ_MM_PF(2, true)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else {
            TQ_MM_PF(1, true)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        }
    } else if (pdl && pref && !xh) {
        // GGML_TQ_PDL (default 1): programmatic launch after tq_rotate_mm (see the waits in tq_mmv_mm)
        const ggml_cuda_kernel_launch_params lp(grid, dim3(128), 0, stream);
        // GGML_TQ_V16 (default 0): 16 B weight loads for full aligned chunks (measured: no gain, 215 vs 215 GB/s)
        if (v16) {
            if (ks == 4) {
                ggml_cuda_kernel_launch(tq_mmv_mm<4, false, true, false, false, true>, lp, TQ_MM_ARGS);
            } else if (ks == 2) {
                ggml_cuda_kernel_launch(tq_mmv_mm<2, false, true, false, false, true>, lp, TQ_MM_ARGS);
            } else {
                ggml_cuda_kernel_launch(tq_mmv_mm<1, false, true, false, false, true>, lp, TQ_MM_ARGS);
            }
        } else if (ks == 4) {
            ggml_cuda_kernel_launch(tq_mmv_mm<4, false, true>, lp, TQ_MM_ARGS);
        } else if (ks == 2) {
            ggml_cuda_kernel_launch(tq_mmv_mm<2, false, true>, lp, TQ_MM_ARGS);
        } else {
            ggml_cuda_kernel_launch(tq_mmv_mm<1, false, true>, lp, TQ_MM_ARGS);
        }
    } else {
        if (ks == 4) {
            TQ_MM_PF(4, false)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else if (ks == 2) {
            TQ_MM_PF(2, false)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        } else {
            TQ_MM_PF(1, false)<<<grid, 128, 0, stream>>>(TQ_MM_ARGS);
        }
    }
#undef TQ_MM_ARGS
    CUDA_CHECK(cudaGetLastError());
}

static tq_mm_desc tq_mm_fill(const ggml_tensor * d, char * dst) {
    const ggml_tensor * w = d->src[0];
    tq_mm_desc m = {};
    m.W        = (const uint8_t *) w->data;
    m.gain     = (const half *) d->src[3]->data;
    m.sign     = (const float *) d->src[4]->data;
    m.dst      = dst;
    m.x        = (const char *) d->src[1]->data;
    m.xu       = d->op_params[13] == GGML_CUDA_RVQ_GLU_MAGIC ? (const char *) d->src[5]->data : nullptr;
    m.nb02     = w->nb[2];
    m.nb1      = d->nb[1];
    m.nb2      = d->nb[2];
    m.nb11     = d->src[1]->nb[1];
    m.nb12     = d->src[1]->nb[2];
    m.K        = w->ne[0] / 16;
    m.rows     = w->ne[1] * 16;
    m.n_expert = w->ne[2];
    m.ida      = d->op_params[12];
    m.ne11     = d->src[1]->ne[1];
    GGML_ASSERT(m.K % 64 == 0 && d->src[3]->ne[0] == m.rows);
    return m;
}

void ggml_cuda_tq_gu_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const int n) {
    if (ggml_cuda_mul_mat_id_tq_needs_sync(cgraph->nodes[i])) {
        tq_pf_multi(ctx, cgraph, i, n, false);
        return;
    }
    tq_mm_args a = {};
    a.n_m = n;
    for (int m = 0; m < n; ++m) {
        a.m[m] = tq_mm_fill(cgraph->nodes[i + m], (char *) cgraph->nodes[i + m]->data);
    }
    int J = 0;
    for (int b = 0; b < n; ++b) {
        J += cgraph->nodes[i + b]->op_params[12] == cgraph->nodes[i]->op_params[12];
    }
    tq_mm_launch(ctx, a, cgraph->nodes[i]->src[2], J, 1.0f, 1.0f);
}

// prefill down + the MoE weighted sum after it (GGML_TQ_REDFUSE): false = nothing done, run both the plain way
bool ggml_cuda_tq_down_multi_red(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const int n,
                                 ggml_tensor * dst, const ggml_tensor * w, const ggml_tensor * sc) {
    static const int on = tq_knob("GGML_TQ_REDFUSE", 0);
    static const int magic = tq_knob("GGML_TQ_MAGIC", 1);
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    if (!on || !magic || !ggml_cuda_mul_mat_id_tq_needs_sync(cgraph->nodes[i]) || !ggml_is_contiguous(dst) ||
        dst->type != GGML_TYPE_F32 || dst->ne[0] != last->ne[0] || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        (sc && (sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc)))) {
        return false;
    }
    for (int m = 0; m < n; ++m) {
        if (cgraph->nodes[i + m]->src[0]->ne[1] * 16 != dst->ne[0]) {
            return false;
        }
    }
    return tq_pf_multi_gpu(ctx, cgraph, i, n, true, (float *) dst->data, (const float *) w->data, sc ? (const float *) sc->data : nullptr);
}

void ggml_cuda_tq_down_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const int n) {
    if (ggml_cuda_mul_mat_id_tq_needs_sync(cgraph->nodes[i])) {
        tq_pf_multi(ctx, cgraph, i, n, true);
        return;
    }
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    tq_mm_args a = {};
    a.n_m = n;
    for (int m = 0; m < n; ++m) {
        a.m[m] = tq_mm_fill(cgraph->nodes[i + m], (char *) last->data);
        a.m[m].nb1 = last->nb[1];
        a.m[m].nb2 = last->nb[2];
    }
    float cg, cu;
    memcpy(&cg, &cgraph->nodes[i]->op_params[10], sizeof(float));
    memcpy(&cu, &cgraph->nodes[i]->op_params[11], sizeof(float));
    tq_mm_launch(ctx, a, cgraph->nodes[i]->src[2], 1, cg, cu);
}
