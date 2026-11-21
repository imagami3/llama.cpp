#include "rvq.cuh"

#include <mma.h>

#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// =====================================================================================================================
// Device layout. The GGUF stores RVQ2X8 row-major: expert e, row r, code c (c-th 8 inputs) at e*nb02 + (r*C + c)*2.
// On GB10, every RVQ2X8 tensor is repacked once after loading (ggml_backend_cuda_rvq_repack) into a K-tile-major
// layout per expert: tiles of RVQ_KT = 16 codes (128 inputs); tile kt holds, for every row, its nck codes contiguously:
//   code (r, c) at e*nb02 + (kt*rows*16 + r*nck + c%16)*2,  kt = c/16, nck = 16 (full tiles) or C%16 (last tile).
// A decode block that owns one K tile then reads one contiguous rows*32 B region instead of 32 B pieces 1152 B apart:
// measured 126 -> 190 GB/s on GB10. The byte size is unchanged.
// =====================================================================================================================

#define RVQ_KT 16

static std::mutex                      rvq_mtx;
static std::unordered_set<const void *> rvq_tiled;   // data pointers of repacked tensors

// decode: per (input, row chunk) arrival counters for the in-kernel K-tile reduction. Zero between launches: the last
// block of every (input, row chunk) resets its counter. Allocated once with the first repack (never inside a capture).
#define RVQ_CNT_N (1 << 16)
static int * rvq_cnt = nullptr;

static __device__ __forceinline__ size_t rvq_tiled_idx(const int r, const int c, const int rows, const int C) {
    const int kt  = c / RVQ_KT;
    const int nck = min(RVQ_KT, C - kt * RVQ_KT);
    return (size_t) kt * rows * RVQ_KT + (size_t) r * nck + (c % RVQ_KT);
}

static __global__ void rvq_repack_kernel(const uint16_t * __restrict__ src, uint16_t * __restrict__ dst, const int rows, const int C,
                                         const int64_t n_exp) {
    const int64_t per = (int64_t) rows * C;
    const int64_t n   = per * n_exp;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t e = i / per, w = i % per;
        const int r = w / C, c = w % C;
        dst[e * per + rvq_tiled_idx(r, c, rows, C)] = src[i];
    }
}

void ggml_backend_cuda_rvq_repack(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_RVQ2X8 && ggml_is_contiguous(t));
    {
        std::lock_guard<std::mutex> lock(rvq_mtx);
        if (rvq_tiled.count(t->data)) {
            return;
        }
    }
    const int K = t->ne[0], rows = t->ne[1], E = t->ne[2] * t->ne[3];
    GGML_ASSERT(K % 64 == 0);
    const int C = K / 8;
    const size_t per = (size_t) rows * C * 2;
    const int bat = std::max<int>(1, (int) std::min<size_t>(E, (256u << 20) / per));   // <= 256 MB scratch
    {
        std::lock_guard<std::mutex> lock(rvq_mtx);
        if (!rvq_cnt) {
            CUDA_CHECK(cudaMalloc(&rvq_cnt, RVQ_CNT_N * sizeof(int)));
            CUDA_CHECK(cudaMemset(rvq_cnt, 0, RVQ_CNT_N * sizeof(int)));
        }
    }
    void * tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, per * bat));
    for (int e0 = 0; e0 < E; e0 += bat) {
        const int nb = std::min(bat, E - e0);
        char * p = (char *) t->data + per * e0;
        CUDA_CHECK(cudaMemcpy(tmp, p, per * nb, cudaMemcpyDeviceToDevice));
        rvq_repack_kernel<<<1024, 256>>>((const uint16_t *) tmp, (uint16_t *) p, rows, C, nb);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(tmp));
    std::lock_guard<std::mutex> lock(rvq_mtx);
    rvq_tiled.insert(t->data);
}

static void rvq_check_tiled(const ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(rvq_mtx);
    if (!rvq_tiled.count(t->data)) {
        GGML_ABORT("RVQ2X8 tensor %s was not repacked for CUDA (call ggml_backend_cuda_rvq_repack after loading)", t->name);
    }
}

// =====================================================================================================================
// decode (<= GGML_CUDA_RVQ_MMV_MAX_TOKENS tokens): LUT, split over K tiles (B70's XeStrata idea).
// block = (K tile, row chunk, x set). T[c][i] = book_i . x[8c .. 8c+8) for the tile's 16 chunks and both books (f32), then
// every row of every expert that uses this x (all slots of a token for gate/up, one slot for down) costs 2 lookups a code.
// Partial sums per K tile go to a buffer; rvq_lut_reduce sums them and applies the gain.
// =====================================================================================================================

#define RVQ_LUT_UNROLL 4

// fused input activation of the down projection (op_params[13] == 'GLU1'): x = cg*tanh(g/cg) * sigmoid(g) * cu*tanh(u/cu),
// g = src1 (gate output), u = src[5] (up output, same layout), cg / cu = op_params[10] / [11] (float bits)
static __device__ __forceinline__ float rvq_act(const float g, const float u, const float cg, const float cu) {
    return (cg * tanhf(g / cg)) * (1.0f / (1.0f + expf(-g))) * (cu * tanhf(u / cu));
}

static __global__ void __launch_bounds__(256)
rvq_lut_partial(const char * __restrict__ codes, const float * __restrict__ book, const char * __restrict__ x,
                const char * __restrict__ xu, const float cg, const float cu,
                const char * __restrict__ ids, float * __restrict__ part,
                const int K, const int rows, const int n_expert, const int n_used, const int ne11, const int n_pairs,
                const size_t nb02, const size_t nb11, const size_t nb12,
                const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block, const int ida, const int zfast,
                const half * __restrict__ gain, const char * __restrict__ prev, char * __restrict__ dst,
                const size_t nb1, const size_t nb2, const int n_kt, const int n_rc, int * __restrict__ cnt) {
    __shared__ float T[RVQ_KT * 512];   // per chunk: book0 entries 0..255, book1 entries 256..511
    __shared__ float sx[RVQ_KT * 8];
    __shared__ uint32_t se[GGML_CUDA_RVQ_MAX_USED];
    __shared__ int      so[GGML_CUDA_RVQ_MAX_USED];   // slots in ascending expert order (zfast): all tokens' blocks walk experts alike
    __shared__ int      s_nown;                       // slots this group owns; they come first in so[]
    __shared__ int      s_last;

    // zfast: grid (input, K tile, row chunk) so the blocks of all tokens for one K tile run together and the codes of
    // experts shared between tokens come from L2; else grid (K tile, row chunk, input)
    const int kt  = zfast ? blockIdx.y : blockIdx.x;
    const int z   = zfast ? blockIdx.x : blockIdx.z;
    const int rc  = zfast ? blockIdx.z : blockIdx.y;
    const int t   = z / ne11;
    const int j   = z % ne11;
    const int tid = threadIdx.x;
    const int C   = K / 8;
    const int c0  = kt * RVQ_KT;
    const int nck = min(RVQ_KT, C - c0);
    const int n_ps = ne11 == 1 ? n_used : 1;

    const float * xr = (const float *) (x + t * nb12 + j * nb11) + c0 * 8;
    if (xu) {
        const float * ur = (const float *) (xu + t * nb12 + j * nb11) + c0 * 8;
        for (int i = tid; i < nck * 8; i += blockDim.x) {
            sx[i] = rvq_act(xr[i], ur[i], cg, cu);
        }
    } else {
        for (int i = tid; i < nck * 8; i += blockDim.x) {
            sx[i] = xr[i];
        }
    }
    if (tid < n_ps) {
        const int s = ne11 == 1 ? tid : j;
        se[tid] = *(const int32_t *) (ids + t * nb_ids1 + s * nb_ids0) - ida;
    }
    __syncthreads();
    if (tid < n_ps) {
        // owned slots first (expert groups: most slots belong to another group), then the rest; zfast sorts the owned
        // ones by expert. Blocks whose rows all fall on non-owned slots leave before building the table.
        const bool own = se[tid] < (uint32_t) n_expert;
        int rank = 0, n_own = 0;
        for (int k = 0; k < n_ps; ++k) {
            const bool ok = se[k] < (uint32_t) n_expert;
            n_own += ok;
            if (own) {
                rank += ok && (zfast ? (se[k] < se[tid] || (se[k] == se[tid] && k < tid)) : k < tid);
            } else {
                rank += ok || k < tid;
            }
        }
        so[rank] = tid;
        if (tid == 0) {
            s_nown = n_own;
        }
    }
    __syncthreads();
    if (cnt && kt == 0) {
        // fused reduction: the K-tile-0 block of each row chunk writes the rows of slots this group does not own
        // (zero, or the earlier groups' sum in accumulate mode)
        const int w0 = max(rc * rows_per_block, s_nown * rows);
        const int w1 = min(n_ps * rows, (rc + 1) * rows_per_block);
        for (int v = w0 + tid; v < w1; v += blockDim.x) {
            const int ps = so[v / rows], r = v % rows;
            const int s  = ne11 == 1 ? ps : j;
            ((float *) (dst + t * nb2 + s * nb1))[r] = prev ? ((const float *) (prev + t * nb2 + s * nb1))[r] : 0.0f;
        }
    }
    if (rc * rows_per_block >= s_nown * rows) {
        return;   // uniform over the block
    }
    for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
        const int c = idx >> 9, i = idx & 511;
        const float4 * b = (const float4 *) (book + i * 8);
        const float4 * v = (const float4 *) (sx + c * 8);
        const float4 b0 = __ldg(b), b1 = __ldg(b + 1), v0 = v[0], v1 = v[1];
        T[idx] = b0.x*v0.x + b0.y*v0.y + b0.z*v0.z + b0.w*v0.w + b1.x*v1.x + b1.y*v1.y + b1.z*v1.z + b1.w*v1.w;
    }
    __syncthreads();

    const int nv = s_nown * rows;
    const int v0 = rc * rows_per_block;
    const int v1 = min(nv, v0 + rows_per_block);
    const size_t tile_off = (size_t) c0 * rows * 2;   // bytes: tiles before this one are full
    if (nck == RVQ_KT) {
        for (int vb = v0 + tid; vb < v1; vb += blockDim.x * RVQ_LUT_UNROLL) {
            uint4 q[RVQ_LUT_UNROLL][2];
            int   pv[RVQ_LUT_UNROLL];
#pragma unroll
            for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {   // issue all loads first
                const int v = vb + u * blockDim.x;
                pv[u] = -1;
                if (v < v1) {
                    const int ps = so[v / rows], r = v % rows;
                    const uint32_t e = se[ps];
                    if (e < (uint32_t) n_expert) {
                        const uint4 * cr = (const uint4 *) (codes + e * nb02 + tile_off + (size_t) r * 32);
                        q[u][0] = __ldg(cr); q[u][1] = __ldg(cr + 1);
                        pv[u] = (t * n_used + (ne11 == 1 ? ps : j)) * rows + r;
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                if (pv[u] < 0) {
                    continue;
                }
                const uint32_t w[8] = {q[u][0].x, q[u][0].y, q[u][0].z, q[u][0].w, q[u][1].x, q[u][1].y, q[u][1].z, q[u][1].w};
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t a = w[k] & 0xffff, b = w[k] >> 16;
                    const float * Ta = T + (2*k) * 512;
                    const float * Tb = Ta + 512;
                    acc += Ta[a >> 8] + Ta[256 + (a & 255)] + Tb[b >> 8] + Tb[256 + (b & 255)];
                }
                part[(size_t) kt * n_pairs * rows + pv[u]] = acc;
            }
        }
    } else {
        for (int v = v0 + tid; v < v1; v += blockDim.x) {
            const int ps = so[v / rows], r = v % rows;
            const uint32_t e = se[ps];
            if (e >= (uint32_t) n_expert) {
                continue;   // reduce writes 0 for skipped slots
            }
            const uint16_t * cr = (const uint16_t *) (codes + e * nb02 + tile_off) + (size_t) r * nck;
            float acc = 0.0f;
            for (int c = 0; c < nck; ++c) {
                const uint32_t a = cr[c];
                acc += T[c * 512 + (a >> 8)] + T[c * 512 + 256 + (a & 255)];
            }
            part[(size_t) kt * n_pairs * rows + (t * n_used + (ne11 == 1 ? ps : j)) * rows + r] = acc;
        }
    }
    if (!cnt) {
        return;
    }
    // fused reduction: the last of the n_kt blocks of this (input, row chunk) to finish sums the partials of its rows
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const int old = atomicAdd(&cnt[z * n_rc + rc], 1);
        s_last = old == n_kt - 1;
        if (s_last) {
            cnt[z * n_rc + rc] = 0;   // ready for the next launch
        }
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    for (int v = v0 + tid; v < v1; v += blockDim.x) {
        const int ps = so[v / rows], r = v % rows;
        const uint32_t e = se[ps];
        const int s = ne11 == 1 ? ps : j;
        const int p = t * n_used + s;
        float acc = 0.0f;
        for (int k = 0; k < n_kt; ++k) {
            acc += __ldcg(&part[((size_t) k * n_pairs + p) * rows + r]);
        }
        ((float *) (dst + t * nb2 + s * nb1))[r] = acc * __half2float(gain[(size_t) e * rows + r]);
    }
}

static __global__ void rvq_lut_reduce(const float * __restrict__ part, const half * __restrict__ gain, const char * __restrict__ ids,
                                      const char * __restrict__ prev, float * __restrict__ dst, const int rows, const int n_expert, const int n_used, const int n_pairs,
                                      const int n_kt, const size_t nb_ids0, const size_t nb_ids1, const size_t nb1, const size_t nb2,
                                      const int ida) {
    const int p = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const int t = p / n_used, s = p % n_used;
    const uint32_t e = *(const int32_t *) (ids + t * nb_ids1 + s * nb_ids0) - ida;
    float * y = (float *) ((char *) dst + t * nb2 + s * nb1);
    if (e >= (uint32_t) n_expert) {   // other group: zero, or the previous groups' result (accumulate mode, src[6])
        y[r] = prev ? ((const float *) (prev + t * nb2 + s * nb1))[r] : 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int kt = 0; kt < n_kt; ++kt) {
        acc += part[((size_t) kt * n_pairs + p) * rows + r];
    }
    y[r] = acc * __half2float(gain[(size_t) e * rows + r]);
}

// =====================================================================================================================
// prefill helpers
// =====================================================================================================================

// W[r][k] (f16, row-major [rows][K]) = gain[r] * (b1[i1] + b2[i2]), codes in the tiled layout
static __global__ void rvq_dequant_f16(const uint16_t * __restrict__ codes, const half * __restrict__ gain,
                                       const float * __restrict__ book, half * __restrict__ w, const int K, const int rows) {
    __shared__ float sb[2 * 256 * 8];
    for (int i = threadIdx.x; i < 2 * 256 * 8; i += blockDim.x) {
        sb[i] = book[i];
    }
    __syncthreads();
    const int C = K / 8;
    const int64_t n = (int64_t) rows * C;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int r = i / C, c = i % C;
        const uint32_t code = codes[rvq_tiled_idx(r, c, rows, C)];
        const float gr = __half2float(gain[r]);
        const float * b1 = sb + (code >> 8) * 8;
        const float * b2 = sb + 2048 + (code & 255) * 8;
        half2 o[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            o[jj] = __floats2half2_rn(gr * (b1[2*jj] + b2[2*jj]), gr * (b1[2*jj+1] + b2[2*jj+1]));
        }
        *(uint4 *) (w + i * 8) = *(const uint4 *) o;
    }
}

// xs[j][k] = f16(x[token(j)][slot(j) % ne11][k]) for the sorted pair list
static __global__ void rvq_gather_f16(const char * __restrict__ x, const char * __restrict__ xu, const float cg, const float cu,
                                      const int32_t * __restrict__ pairs, half * __restrict__ xs,
                                      const int K, const int n_used, const int ne11, const size_t nb11, const size_t nb12) {
    const int j = blockIdx.x;
    const int p = pairs[j];
    const int t = p / n_used, s = p % n_used;
    const float * xr = (const float *) (x + t * nb12 + (s % ne11) * nb11);
    if (xu) {
        const float * ur = (const float *) (xu + t * nb12 + (s % ne11) * nb11);
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            xs[(int64_t) j * K + k] = __float2half(rvq_act(xr[k], ur[k], cg, cu));
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            xs[(int64_t) j * K + k] = __float2half(xr[k]);
        }
    }
}

// dst[token(j)][slot(j)][:] = ys[j][:]
static __global__ void rvq_scatter(const float * __restrict__ ys, const int32_t * __restrict__ pairs, char * __restrict__ dst,
                                   const int rows, const int n_used, const size_t nb1, const size_t nb2) {
    const int j = blockIdx.x;
    const int p = pairs[j];
    const int t = p / n_used, s = p % n_used;
    float * y = (float *) (dst + t * nb2 + s * nb1);
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        y[r] = ys[(int64_t) j * rows + r];
    }
}

// =====================================================================================================================
// prefill, fused: block = (work item = expert + tile of <= 32 of its sorted tokens, tile of 64 rows). Per K step of 64 the
// codes of the 64 rows (8 contiguous codes a row in the tiled layout) are decoded into shared memory (f16, gain applied)
// and multiplied with the token tile on tensor cores (wmma 16x16x16, f32 accumulate).
// =====================================================================================================================

#define RVQ_PF_M 64
#define RVQ_PF_K 128   // one full code tile (16 codes) a step; a last step of 64 when K % 128 == 64
#define RVQ_PF_LD (RVQ_PF_K + 8)   // padded smem row (272 B) against bank conflicts

// N = token tile (32, 64 or 128); 4 warps, each 16 rows x N tokens. Dynamic shared memory:
//   book f16 [512*8] | A f16 [64][128] | B f16 [N][128] | C f32 [N][64] (col-major [tok][row]) | gains f16 [64]
template <int N>
static __global__ void __launch_bounds__(128)
rvq_prefill_mma(const char * __restrict__ codes, const half * __restrict__ gain, const float * __restrict__ book,
                const half * __restrict__ xs, const int4 * __restrict__ items, float * __restrict__ ys,
                const int K, const int rows, const size_t nb02) {
    using namespace nvcuda;
    extern __shared__ __align__(32) char smem_pf[];
    half  * sb = (half *) smem_pf;
    half  * sA = sb + 512 * 8;
    half  * sB = sA + RVQ_PF_M * RVQ_PF_LD;
    float * sC = (float *) (sB + N * RVQ_PF_LD);
    half  * sg = (half *) (sC + RVQ_PF_M * N);

    const int4 it = items[blockIdx.y];   // (grid: x = row tile, y = item, so the row tiles of one expert run together and share its inputs in L2) x = expert, y = first sorted pair, z = n tokens
    const int e = it.x, j0 = it.y, nt = it.z;
    const int r0 = blockIdx.x * RVQ_PF_M;
    const int tid = threadIdx.x, warp = tid / 32;
    const int C = K / 8;

    for (int i = tid; i < 512 * 8; i += blockDim.x) {
        sb[i] = __float2half(book[i]);   // E4M3 values: exact in f16
    }
    if (tid < RVQ_PF_M) {
        sg[tid] = gain[(size_t) e * rows + r0 + tid];
    }
    __syncthreads();

    constexpr int NF = N / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        wmma::fill_fragment(acc[f], 0.0f);
    }

    // each thread owns one 8-code half of one row's 16-code tile (64 rows x 2 halves = 128 threads): one 16 B load a step,
    // and N*16/128 16 B loads of B. Loads of step s+1 are issued before the tensor-core work of step s.
    static_assert(RVQ_PF_M * 2 == 128, "one row half a thread");
    const uint16_t * cexp = (const uint16_t *) (codes + (size_t) e * nb02);
    const int ar = tid >> 1, ah = tid & 1;
    constexpr int NB = N * (RVQ_PF_K / 8) / 128;   // B uint4 a thread
    auto load_step = [&](const int k0, uint4 & ca, uint4 (&cb)[NB]) {
        const int kt  = k0 / RVQ_PF_K;
        const int nck = min(16, C - kt * 16);
        ca = make_uint4(0, 0, 0, 0);
        if (ah * 8 < nck) {
            ca = __ldg((const uint4 *) (cexp + (size_t) kt * rows * 16 + (size_t) (r0 + ar) * nck + ah * 8));
        }
        const int kc = nck;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int q = tid + i * 128;
            const int n = q / (RVQ_PF_K / 8), c = q % (RVQ_PF_K / 8);
            cb[i] = make_uint4(0, 0, 0, 0);
            if (n < nt && c < kc) {
                cb[i] = *(const uint4 *) (xs + (size_t) (j0 + n) * K + k0 + c * 8);
            }
        }
    };
    uint4 ca, cb[NB];
    load_step(0, ca, cb);
    for (int k0 = 0; k0 < K; k0 += RVQ_PF_K) {
        const int kc = min(RVQ_PF_K, K - k0) / 8;   // codes this step (16, or 8 at a 64-wide tail)
        // A: decode this thread's 8 codes (half2 math, gain applied); zero when past the tail
        {
            const half2 g2 = __half2half2(sg[ar]);
            const uint32_t cw[4] = {ca.x, ca.y, ca.z, ca.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t code = (cw[i >> 1] >> (16 * (i & 1))) & 0xffff;
                uint4 o = make_uint4(0, 0, 0, 0);
                if (ah * 8 + i < kc) {
                    const uint4 u1 = *(const uint4 *) (sb + (code >> 8) * 8);
                    const uint4 u2 = *(const uint4 *) (sb + (256 + (code & 255)) * 8);
                    const half2 * a1 = (const half2 *) &u1;
                    const half2 * a2 = (const half2 *) &u2;
                    half2 * oo = (half2 *) &o;
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        oo[jj] = __hmul2(g2, __hadd2(a1[jj], a2[jj]));
                    }
                }
                *(uint4 *) (sA + ar * RVQ_PF_LD + (ah * 8 + i) * 8) = o;
            }
        }
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int q = tid + i * 128;
            const int n = q / (RVQ_PF_K / 8), c = q % (RVQ_PF_K / 8);
            *(uint4 *) (sB + n * RVQ_PF_LD + c * 8) = cb[i];
        }
        __syncthreads();
        if (k0 + RVQ_PF_K < K) {
            load_step(k0 + RVQ_PF_K, ca, cb);   // in flight during the MMAs
        }
        for (int kk = 0; kk < kc * 8; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, sA + (warp * 16) * RVQ_PF_LD + kk, RVQ_PF_LD);
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sB + (f * 16) * RVQ_PF_LD + kk, RVQ_PF_LD);
                wmma::mma_sync(acc[f], fa, fb, acc[f]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        wmma::store_matrix_sync(sC + (f * 16) * RVQ_PF_M + warp * 16, acc[f], RVQ_PF_M, wmma::mem_col_major);
    }
    __syncthreads();
    for (int q = tid; q < nt * RVQ_PF_M; q += blockDim.x) {
        const int n = q / RVQ_PF_M, r = q % RVQ_PF_M;
        ys[(size_t) (j0 + n) * rows + r0 + r] = sC[n * RVQ_PF_M + r];
    }
}

template <int N>
static void rvq_prefill_launch(const std::vector<int4> & items, ggml_cuda_pool_alloc<int4> & items_d, cudaStream_t stream,
                               const ggml_tensor * src0, const ggml_tensor * gain, const ggml_tensor * book,
                               const half * xs, float * ys, const int K, const int rows) {
    if (items.empty()) {
        return;
    }
    items_d.alloc(items.size());
    CUDA_CHECK(cudaMemcpyAsync(items_d.ptr, items.data(), items.size() * sizeof(int4), cudaMemcpyHostToDevice, stream));
    const size_t smem = 512 * 8 * 2 + RVQ_PF_M * RVQ_PF_LD * 2 + N * RVQ_PF_LD * 2 + RVQ_PF_M * N * 4 + RVQ_PF_M * 2;
    static bool smem_set = false;
    if (!smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute(rvq_prefill_mma<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        smem_set = true;
    }
    const dim3 grid(rows / RVQ_PF_M, items.size());
    rvq_prefill_mma<N><<<grid, 128, smem, stream>>>((const char *) src0->data, (const half *) gain->data, (const float *) book->data,
        xs, items_d.ptr, ys, K, rows, src0->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

// =====================================================================================================================

bool ggml_cuda_mul_mat_id_rvq_needs_sync(const ggml_tensor * dst) {
    // GGML_RVQ_MMV_MAX: largest token count for the LUT decode path (default GGML_CUDA_RVQ_MMV_MAX_TOKENS)
    static const int mmv_max = [] { const char * v = getenv("GGML_RVQ_MMV_MAX"); return v ? atoi(v) : GGML_CUDA_RVQ_MMV_MAX_TOKENS; }();
    return dst->src[1]->ne[2] > mmv_max || dst->src[2]->ne[0] > GGML_CUDA_RVQ_MAX_USED;
}

void ggml_cuda_mul_mat_id_rvq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const ggml_tensor * gain = dst->src[3];
    const ggml_tensor * book = dst->src[4];

    GGML_ASSERT(src0->type == GGML_TYPE_RVQ2X8 && ggml_is_contiguous(src0));
    GGML_ASSERT(gain && gain->type == GGML_TYPE_F16 && ggml_is_contiguous(gain) && "RVQ2X8 mul_mat_id needs src[3] = F16 gains");
    GGML_ASSERT(book && book->type == GGML_TYPE_F32 && ggml_nelements(book) == 2*256*8 && "RVQ2X8 mul_mat_id needs src[4] = codebook");
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    rvq_check_tiled(src0);

    const int K        = src0->ne[0];
    const int rows     = src0->ne[1];
    const int n_expert = src0->ne[2];
    const int n_used   = ids->ne[0];
    const int n_tok    = ids->ne[1];
    const int ne11     = src1->ne[1];
    GGML_ASSERT(K % 64 == 0);
    GGML_ASSERT(gain->ne[0] == rows && gain->ne[1] == n_expert);
    GGML_ASSERT(dst->ne[0] == rows && dst->ne[1] == n_used && dst->ne[2] == n_tok);

    cudaStream_t stream = ctx.stream();

    // group-range mode (op_params[14] == 'RVQG'): ids are router-global, this group owns [op_params[12], +n_expert);
    // other ids are skipped (zeros). Otherwise ids are local and ids >= n_expert are skipped.
    const int ida = dst->op_params[14] == GGML_CUDA_RVQ_GROUP_MAGIC ? dst->op_params[12] : 0;

    // fused down-projection input (op_params[13] == 'GLU1'): src1 = gate output, src[5] = up output
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

    if (!ggml_cuda_mul_mat_id_rvq_needs_sync(dst)) {
        const int n_pairs = n_tok * n_used;
        const int n_kt    = (K / 8 + RVQ_KT - 1) / RVQ_KT;
        const int n_ps    = ne11 == 1 ? n_used : 1;
        const int nv      = n_ps * rows;
        const int nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        static const int lut_bps = [] { const char * v = getenv("GGML_RVQ_LUT_BPS"); return v ? atoi(v) : 2; }();   // blocks per SM
        static const int lut_min = [] { const char * v = getenv("GGML_RVQ_LUT_MINROWS"); return v ? atoi(v) : 2048; }();
        const int want    = std::max(1, (lut_bps * nsm + n_kt * n_tok * ne11 - 1) / (n_kt * n_tok * ne11));
        const int rpb     = std::max(lut_min, (nv + want - 1) / want);
        const int n_rc    = (nv + rpb - 1) / rpb;
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) n_kt * n_pairs * rows);
        static const int zfast = [] { const char * v = getenv("GGML_RVQ_LUT_ZFAST"); return v ? atoi(v) : 1; }();
        const bool zf = zfast && n_tok > 1;
        const dim3 grid = zf ? dim3(n_tok * ne11, n_kt, n_rc) : dim3(n_kt, n_rc, n_tok * ne11);
        // GGML_RVQ_FUSE_REDUCE (default 0): the last K-tile block of each row chunk does the reduction (no second kernel).
        // Measured slower on GB10 (the one reducing block per row chunk is a serial tail): 2304 rows, 2 slots 34.6 -> 53.4 us
        static const int fred = [] { const char * v = getenv("GGML_RVQ_FUSE_REDUCE"); return v ? atoi(v) : 0; }();
        int * cnt = fred && rvq_cnt && n_tok * ne11 * n_rc <= RVQ_CNT_N ? rvq_cnt : nullptr;
        rvq_lut_partial<<<grid, 256, 0, stream>>>(
            (const char *) src0->data, (const float *) book->data, (const char *) src1->data, xu, cg, cu, (const char *) ids->data, part.ptr,
            K, rows, n_expert, n_used, ne11, n_pairs, src0->nb[2], src1->nb[1], src1->nb[2], ids->nb[0], ids->nb[1], rpb, ida, zf ? 1 : 0,
            (const half *) gain->data, prev ? (const char *) prev->data : nullptr, (char *) dst->data, dst->nb[1], dst->nb[2], n_kt, n_rc, cnt);
        CUDA_CHECK(cudaGetLastError());
        if (cnt) {
            return;
        }
        const dim3 grid2((rows + 255) / 256, n_pairs);
        rvq_lut_reduce<<<grid2, 256, 0, stream>>>(part.ptr, (const half *) gain->data, (const char *) ids->data,
            prev ? (const char *) prev->data : nullptr, (float *) dst->data,
            rows, n_expert, n_used, n_pairs, n_kt, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2], ida);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // prefill: sort (token, slot) pairs by expert on the host, gather inputs as f16; per expert either the fused kernel
    // (few tokens) or f16 dequant + one cuBLAS GEMM (many tokens); scatter back. Skipped slots stay zero.
    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

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
    CUDA_CHECK(cudaMemcpyAsync(pairs_d.ptr, pairs.data(), n_valid * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    ggml_cuda_pool_alloc<half>  xs(ctx.pool(), (size_t) n_valid * K);
    ggml_cuda_pool_alloc<float> ys(ctx.pool(), (size_t) n_valid * rows);

    rvq_gather_f16<<<n_valid, 256, 0, stream>>>((const char *) src1->data, xu, cg, cu, pairs_d.ptr, xs.ptr, K, n_used, ne11, src1->nb[1], src1->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    static const int rvq_pf = [] { const char * v = getenv("GGML_RVQ_PREFILL"); return v ? atoi(v) : 1; }();   // 0 cublas, 1 fused
    static const int pf_cut = [] { const char * v = getenv("GGML_RVQ_PF_MAX"); return v ? atoi(v) : 128; }();   // above: cuBLAS (sweep on GB10: 128 best)
    const int pf_max = (rvq_pf == 1 && rows % RVQ_PF_M == 0) ? pf_cut : 0;
    // token tiles: <= 32 -> N 32, <= 64 -> N 64, else tiles of 128
    std::vector<int4> it32, it64, it128;
    for (int e = 0; e < n_expert; ++e) {
        const int n = count[e];
        if (n == 0 || n > pf_max) {
            continue;
        }
        if (n <= 32) {
            it32.push_back(make_int4(e, start[e], n, 0));
        } else if (n <= 64) {
            it64.push_back(make_int4(e, start[e], n, 0));
        } else {
            for (int j = 0; j < n; j += 128) {
                it128.push_back(make_int4(e, start[e] + j, std::min(128, n - j), 0));
            }
        }
    }
    ggml_cuda_pool_alloc<int4> items32_d(ctx.pool()), items64_d(ctx.pool()), items128_d(ctx.pool());
    rvq_prefill_launch<32> (it32,  items32_d,  stream, src0, gain, book, xs.ptr, ys.ptr, K, rows);
    rvq_prefill_launch<64> (it64,  items64_d,  stream, src0, gain, book, xs.ptr, ys.ptr, K, rows);
    rvq_prefill_launch<128>(it128, items128_d, stream, src0, gain, book, xs.ptr, ys.ptr, K, rows);
    bool any_big = false;
    for (int e = 0; e < n_expert; ++e) {
        any_big = any_big || count[e] > pf_max;
    }
    ggml_cuda_pool_alloc<half> wf(ctx.pool());   // declared after items_d: the pool frees in reverse order
    if (any_big) {
        wf.alloc((size_t) rows * K);
        cublasHandle_t handle = ctx.cublas_handle();
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        const float alpha = 1.0f, beta = 0.0f;
        const int dq_blocks = std::min<int64_t>(((int64_t) rows * (K / 8) + 255) / 256, 1024);
        for (int e = 0; e < n_expert; ++e) {
            const int n = count[e];
            if (n == 0 || n <= pf_max) {
                continue;
            }
            rvq_dequant_f16<<<dq_blocks, 256, 0, stream>>>(
                (const uint16_t *) ((const char *) src0->data + (size_t) e * src0->nb[2]), (const half *) gain->data + (size_t) e * rows,
                (const float *) book->data, wf.ptr, K, rows);
            CUDA_CHECK(cudaGetLastError());
            // Y[rows x n] (col-major, ld rows) = W^T (W: col-major K x rows) * X (col-major K x n)
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, K,
                &alpha, wf.ptr, CUDA_R_16F, K,
                xs.ptr + (size_t) start[e] * K, CUDA_R_16F, K,
                &beta, ys.ptr + (size_t) start[e] * rows, CUDA_R_32F, rows,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));   // host vectors (pairs, items) must outlive the async copies

    rvq_scatter<<<n_valid, 256, 0, stream>>>(ys.ptr, pairs_d.ptr, (char *) dst->data, rows, n_used, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

// =====================================================================================================================
// decode, several gate/up matrices in one launch: the expert groups of a layer (and gate with up) share x and ids, so a
// block builds the 16 x 512 table of its K tile once and serves every owned (matrix, slot) row of its row chunk.
// Each (matrix, slot) pair a token owns is a segment of rows_m virtual rows; a block covers rows_per_block of them.
// =====================================================================================================================

#define RVQ_MM_MAX 8
#define RVQ_MM_SEG (RVQ_MM_MAX * 16)

struct rvq_mm_desc {
    const char * codes;
    const half * gain;
    char       * dst;
    size_t       nb02, nb1, nb2, part_off;
    int          rows, n_expert, ida;
};
struct rvq_mm_args {
    rvq_mm_desc m[RVQ_MM_MAX];
    int         n_m;
};

static __global__ void __launch_bounds__(256)
rvq_lut_multi(const rvq_mm_args a, const float * __restrict__ book, const char * __restrict__ x, const char * __restrict__ ids,
              float * __restrict__ part, const int K, const int n_used, const int n_pairs, const size_t nb12,
              const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block, const int tfast, const int same_ids) {
    __shared__ float T[RVQ_KT * 512];
    __shared__ float sx[RVQ_KT * 8];
    __shared__ int   sg_m[RVQ_MM_SEG], sg_s[RVQ_MM_SEG], sg_e[RVQ_MM_SEG], sg_v[RVQ_MM_SEG + 1];
    __shared__ int   s_nseg;

    // tfast (several tokens): grid (token, row chunk, K tile) so all blocks of one K tile run together and the codes of
    // experts shared between tokens come from L2; else grid (K tile, row chunk, token)
    const int kt  = tfast ? blockIdx.z : blockIdx.x;
    const int rc  = blockIdx.y;
    const int t   = tfast ? blockIdx.x : blockIdx.z;
    const int tid = threadIdx.x;
    const int C   = K / 8;
    const int c0  = kt * RVQ_KT;
    const int nck = min(RVQ_KT, C - c0);

    if (tid == 0) {
        int ns = 0, v = 0;
        const int t_ids = same_ids ? 0 : t;   // timing experiment only (GGML_RVQ_DBG_SAMEIDS): wrong results
        for (int m = 0; m < a.n_m; ++m) {
            for (int s = 0; s < n_used && ns < RVQ_MM_SEG; ++s) {
                const uint32_t e = *(const int32_t *) (ids + t_ids * nb_ids1 + s * nb_ids0) - a.m[m].ida;
                if (e < (uint32_t) a.m[m].n_expert) {
                    sg_m[ns] = m; sg_s[ns] = s; sg_e[ns] = e; sg_v[ns] = v;
                    v += a.m[m].rows;
                    ++ns;
                }
            }
        }
        sg_v[ns] = v;
        s_nseg   = ns;
    }
    __syncthreads();
    const int nv = sg_v[s_nseg];
    const int v0 = rc * rows_per_block;
    if (v0 >= nv) {
        return;   // uniform over the block
    }
    const int v1 = min(nv, v0 + rows_per_block);

    const float * xr = (const float *) (x + t * nb12) + c0 * 8;
    for (int i = tid; i < nck * 8; i += blockDim.x) {
        sx[i] = xr[i];
    }
    __syncthreads();
    for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
        const int c = idx >> 9, i = idx & 511;
        const float4 * b = (const float4 *) (book + i * 8);
        const float4 * w = (const float4 *) (sx + c * 8);
        const float4 b0 = __ldg(b), b1 = __ldg(b + 1), w0 = w[0], w1 = w[1];
        T[idx] = b0.x*w0.x + b0.y*w0.y + b0.z*w0.z + b0.w*w0.w + b1.x*w1.x + b1.y*w1.y + b1.z*w1.z + b1.w*w1.w;
    }
    __syncthreads();

    const int nseg = s_nseg;
    int g = 0;   // segment of the first row this thread handles; rows only grow, so the search walks forward
    for (int vb = v0 + tid; vb < v1; vb += blockDim.x * RVQ_LUT_UNROLL) {
        uint4 q[RVQ_LUT_UNROLL][2];
        size_t po[RVQ_LUT_UNROLL];
        bool   ok[RVQ_LUT_UNROLL];
        int    nk[RVQ_LUT_UNROLL];
        const uint16_t * cp[RVQ_LUT_UNROLL];
#pragma unroll
        for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {   // issue all loads first
            const int v = vb + u * blockDim.x;
            ok[u] = v < v1;
            if (ok[u]) {
                while (g + 1 < nseg && sg_v[g + 1] <= v) {
                    ++g;
                }
                const rvq_mm_desc & d = a.m[sg_m[g]];
                const int r = v - sg_v[g];
                const char * base = d.codes + (size_t) sg_e[g] * d.nb02 + (size_t) c0 * d.rows * 2;
                po[u] = d.part_off + ((size_t) kt * n_pairs + t * n_used + sg_s[g]) * d.rows + r;
                nk[u] = nck;
                if (nck == RVQ_KT) {
                    const uint4 * cr = (const uint4 *) (base + (size_t) r * 32);
                    q[u][0] = __ldg(cr); q[u][1] = __ldg(cr + 1);
                } else {
                    cp[u] = (const uint16_t *) base + (size_t) r * nck;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
            if (!ok[u]) {
                continue;
            }
            float acc = 0.0f;
            if (nk[u] == RVQ_KT) {
                const uint32_t w[8] = {q[u][0].x, q[u][0].y, q[u][0].z, q[u][0].w, q[u][1].x, q[u][1].y, q[u][1].z, q[u][1].w};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t lo = w[k] & 0xffff, hi = w[k] >> 16;
                    const float * Ta = T + (2*k) * 512;
                    const float * Tb = Ta + 512;
                    acc += Ta[lo >> 8] + Ta[256 + (lo & 255)] + Tb[hi >> 8] + Tb[256 + (hi & 255)];
                }
            } else {
                for (int c = 0; c < nk[u]; ++c) {
                    const uint32_t c2 = cp[u][c];
                    acc += T[c * 512 + (c2 >> 8)] + T[c * 512 + 256 + (c2 & 255)];
                }
            }
            part[po[u]] = acc;
        }
    }
}

// rvq_lut_multi with each block over ktg consecutive K tiles: a thread keeps its rows' sums in registers across the
// tiles (rebuilding the table per tile), so the K-split partials written to and read back from memory shrink ktg-fold
// (4 tokens: ~26 MB a layer at ktg 1). A thread owns rows v0 + j*blockDim + tid, j < RPT. grid (token, row chunk, K group).
template <int RPT>
static __global__ void __launch_bounds__(256)
rvq_lut_multi_g(const rvq_mm_args a, const float * __restrict__ book, const char * __restrict__ x, const char * __restrict__ ids,
                float * __restrict__ part, const int K, const int n_used, const int n_pairs, const size_t nb12,
                const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block, const int ktg, const int n_kt) {
    __shared__ float T[RVQ_KT * 512];
    __shared__ float sx[RVQ_KT * 8];
    __shared__ int   sg_m[RVQ_MM_SEG], sg_s[RVQ_MM_SEG], sg_e[RVQ_MM_SEG], sg_v[RVQ_MM_SEG + 1];
    __shared__ int   s_nseg;

    const int t   = blockIdx.x;
    const int rc  = blockIdx.y;
    const int kg  = blockIdx.z;
    const int tid = threadIdx.x;
    const int C   = K / 8;

    if (tid == 0) {
        int ns = 0, v = 0;
        for (int m = 0; m < a.n_m; ++m) {
            for (int s = 0; s < n_used && ns < RVQ_MM_SEG; ++s) {
                const uint32_t e = *(const int32_t *) (ids + t * nb_ids1 + s * nb_ids0) - a.m[m].ida;
                if (e < (uint32_t) a.m[m].n_expert) {
                    sg_m[ns] = m; sg_s[ns] = s; sg_e[ns] = e; sg_v[ns] = v;
                    v += a.m[m].rows;
                    ++ns;
                }
            }
        }
        sg_v[ns] = v;
        s_nseg   = ns;
    }
    __syncthreads();
    const int nv = sg_v[s_nseg];
    const int v0 = rc * rows_per_block;
    if (v0 >= nv) {
        return;   // uniform over the block
    }
    const int v1   = min(nv, v0 + rows_per_block);
    const int nseg = s_nseg;

    int   rg[RPT], rr[RPT];
    float acc[RPT];
    {
        int g = 0;
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int v = v0 + j * blockDim.x + tid;
            acc[j] = 0.0f;
            rg[j]  = -1;
            rr[j]  = 0;
            if (v < v1) {
                while (g + 1 < nseg && sg_v[g + 1] <= v) {
                    ++g;
                }
                rg[j] = g;
                rr[j] = v - sg_v[g];
            }
        }
    }

    const float * xt = (const float *) (x + t * nb12);
    const int kt_end = min(n_kt, (kg + 1) * ktg);
    for (int kt = kg * ktg; kt < kt_end; ++kt) {
        const int c0  = kt * RVQ_KT;
        const int nck = min(RVQ_KT, C - c0);
        __syncthreads();   // the previous tile's table is no longer read
        for (int i = tid; i < nck * 8; i += blockDim.x) {
            sx[i] = xt[c0 * 8 + i];
        }
        __syncthreads();
        for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
            const int c = idx >> 9, i = idx & 511;
            const float4 * b = (const float4 *) (book + i * 8);
            const float4 * w = (const float4 *) (sx + c * 8);
            const float4 b0 = __ldg(b), b1 = __ldg(b + 1), w0 = w[0], w1 = w[1];
            T[idx] = b0.x*w0.x + b0.y*w0.y + b0.z*w0.z + b0.w*w0.w + b1.x*w1.x + b1.y*w1.y + b1.z*w1.z + b1.w*w1.w;
        }
        __syncthreads();
        if (nck == RVQ_KT) {
#pragma unroll
            for (int j0 = 0; j0 < RPT; j0 += RVQ_LUT_UNROLL) {
                uint4 q[RVQ_LUT_UNROLL][2];
#pragma unroll
                for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {   // issue the loads first
                    const int j = j0 + u;
                    if (rg[j] >= 0) {
                        const rvq_mm_desc & d = a.m[sg_m[rg[j]]];
                        const uint4 * cr = (const uint4 *) (d.codes + (size_t) sg_e[rg[j]] * d.nb02 + (size_t) c0 * d.rows * 2 +
                                                            (size_t) rr[j] * 32);
                        q[u][0] = __ldg(cr); q[u][1] = __ldg(cr + 1);
                    }
                }
#pragma unroll
                for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                    const int j = j0 + u;
                    if (rg[j] < 0) {
                        continue;
                    }
                    const uint32_t w[8] = {q[u][0].x, q[u][0].y, q[u][0].z, q[u][0].w, q[u][1].x, q[u][1].y, q[u][1].z, q[u][1].w};
                    float ts = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const uint32_t lo = w[k] & 0xffff, hi = w[k] >> 16;
                        const float * Ta = T + (2*k) * 512;
                        const float * Tb = Ta + 512;
                        ts += Ta[lo >> 8] + Ta[256 + (lo & 255)] + Tb[hi >> 8] + Tb[256 + (hi & 255)];
                    }
                    acc[j] += ts;
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < RPT; ++j) {
                if (rg[j] < 0) {
                    continue;
                }
                const rvq_mm_desc & d = a.m[sg_m[rg[j]]];
                const uint16_t * cr = (const uint16_t *) (d.codes + (size_t) sg_e[rg[j]] * d.nb02 + (size_t) c0 * d.rows * 2) +
                                      (size_t) rr[j] * nck;
                float ts = 0.0f;
                for (int c = 0; c < nck; ++c) {
                    const uint32_t c2 = cr[c];
                    ts += T[c * 512 + (c2 >> 8)] + T[c * 512 + 256 + (c2 & 255)];
                }
                acc[j] += ts;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
        if (rg[j] < 0) {
            continue;
        }
        const rvq_mm_desc & d = a.m[sg_m[rg[j]]];
        part[d.part_off + ((size_t) kg * n_pairs + t * n_used + sg_s[rg[j]]) * d.rows + rr[j]] = acc[j];
    }
}

// several tokens, two per block: the table holds both tokens' entries side by side (float2), so a code row of an
// expert either token uses is read once and one 8-byte lookup serves both. The segments are the union of the pair's
// (matrix, expert); each carries the slot of each token that uses it (-1: not used). Sums run in the same order as
// rvq_lut_multi's, so the results are bitwise the same. grid (pair, row chunk, K tile); dynamic shared memory 64 KB.
static __global__ void __launch_bounds__(256)
rvq_lut_multi_pair(const rvq_mm_args a, const float * __restrict__ book, const char * __restrict__ x, const char * __restrict__ ids,
                   float * __restrict__ part, const int K, const int n_used, const int n_pairs, const int n_tok, const size_t nb12,
                   const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block) {
    extern __shared__ float2 T2[];                    // RVQ_KT * 512
    __shared__ float sx[2][RVQ_KT * 8];
    __shared__ int   sg_m[RVQ_MM_SEG], sg_e[RVQ_MM_SEG], sg_s0[RVQ_MM_SEG], sg_s1[RVQ_MM_SEG], sg_v[RVQ_MM_SEG + 1];
    __shared__ int   s_nseg;

    const int t0  = 2 * blockIdx.x;
    const int t1  = t0 + 1 < n_tok ? t0 + 1 : -1;
    const int rc  = blockIdx.y;
    const int kt  = blockIdx.z;
    const int tid = threadIdx.x;
    const int C   = K / 8;
    const int c0  = kt * RVQ_KT;
    const int nck = min(RVQ_KT, C - c0);

    if (tid == 0) {
        int ns = 0, v = 0;
        for (int m = 0; m < a.n_m; ++m) {
            const int seg0 = ns;
            for (int s = 0; s < n_used && ns < RVQ_MM_SEG; ++s) {   // token t0's experts in slot order
                const uint32_t e = *(const int32_t *) (ids + t0 * nb_ids1 + s * nb_ids0) - a.m[m].ida;
                if (e < (uint32_t) a.m[m].n_expert) {
                    sg_m[ns] = m; sg_e[ns] = e; sg_s0[ns] = s; sg_s1[ns] = -1; sg_v[ns] = v;
                    v += a.m[m].rows;
                    ++ns;
                }
            }
            if (t1 >= 0) {
                for (int s = 0; s < n_used; ++s) {                   // token t1: join a shared expert, else append
                    const uint32_t e = *(const int32_t *) (ids + t1 * nb_ids1 + s * nb_ids0) - a.m[m].ida;
                    if (e >= (uint32_t) a.m[m].n_expert) {
                        continue;
                    }
                    int g = -1;
                    for (int k = seg0; k < ns; ++k) {
                        if (sg_e[k] == (int) e && sg_s1[k] < 0) { g = k; break; }
                    }
                    if (g >= 0) {
                        sg_s1[g] = s;
                    } else if (ns < RVQ_MM_SEG) {
                        sg_m[ns] = m; sg_e[ns] = e; sg_s0[ns] = -1; sg_s1[ns] = s; sg_v[ns] = v;
                        v += a.m[m].rows;
                        ++ns;
                    }
                }
            }
        }
        sg_v[ns] = v;
        s_nseg   = ns;
    }
    __syncthreads();
    const int nv = sg_v[s_nseg];
    const int v0 = rc * rows_per_block;
    if (v0 >= nv) {
        return;   // uniform over the block
    }
    const int v1 = min(nv, v0 + rows_per_block);

    {
        const float * x0 = (const float *) (x + t0 * nb12) + c0 * 8;
        const float * x1 = (const float *) (x + (t1 >= 0 ? t1 : t0) * nb12) + c0 * 8;
        for (int i = tid; i < nck * 8; i += blockDim.x) {
            sx[0][i] = x0[i];
            sx[1][i] = x1[i];
        }
    }
    __syncthreads();
    for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
        const int c = idx >> 9, i = idx & 511;
        const float4 * b = (const float4 *) (book + i * 8);
        const float4 b0 = __ldg(b), b1 = __ldg(b + 1);
        const float4 * w = (const float4 *) (sx[0] + c * 8);
        const float4 * u = (const float4 *) (sx[1] + c * 8);
        const float4 w0 = w[0], w1 = w[1], u0 = u[0], u1 = u[1];
        T2[idx] = make_float2(b0.x*w0.x + b0.y*w0.y + b0.z*w0.z + b0.w*w0.w + b1.x*w1.x + b1.y*w1.y + b1.z*w1.z + b1.w*w1.w,
                              b0.x*u0.x + b0.y*u0.y + b0.z*u0.z + b0.w*u0.w + b1.x*u1.x + b1.y*u1.y + b1.z*u1.z + b1.w*u1.w);
    }
    __syncthreads();

    const int nseg = s_nseg;
    int g = 0;
    for (int vb = v0 + tid; vb < v1; vb += blockDim.x * RVQ_LUT_UNROLL) {
        uint4 q[RVQ_LUT_UNROLL][2];
        int   gs[RVQ_LUT_UNROLL], rr[RVQ_LUT_UNROLL];
        bool  ok[RVQ_LUT_UNROLL];
        const uint16_t * cp[RVQ_LUT_UNROLL];
#pragma unroll
        for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
            const int v = vb + u * blockDim.x;
            ok[u] = v < v1;
            if (ok[u]) {
                while (g + 1 < nseg && sg_v[g + 1] <= v) {
                    ++g;
                }
                const rvq_mm_desc & d = a.m[sg_m[g]];
                const int r = v - sg_v[g];
                gs[u] = g; rr[u] = r;
                const char * base = d.codes + (size_t) sg_e[g] * d.nb02 + (size_t) c0 * d.rows * 2;
                if (nck == RVQ_KT) {
                    const uint4 * cr = (const uint4 *) (base + (size_t) r * 32);
                    q[u][0] = __ldg(cr); q[u][1] = __ldg(cr + 1);
                } else {
                    cp[u] = (const uint16_t *) base + (size_t) r * nck;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
            if (!ok[u]) {
                continue;
            }
            float acc0 = 0.0f, acc1 = 0.0f;
            if (nck == RVQ_KT) {
                const uint32_t w[8] = {q[u][0].x, q[u][0].y, q[u][0].z, q[u][0].w, q[u][1].x, q[u][1].y, q[u][1].z, q[u][1].w};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t lo = w[k] & 0xffff, hi = w[k] >> 16;
                    const float2 * Ta = T2 + (2*k) * 512;
                    const float2 * Tb = Ta + 512;
                    const float2 p0 = Ta[lo >> 8], p1 = Ta[256 + (lo & 255)], p2 = Tb[hi >> 8], p3 = Tb[256 + (hi & 255)];
                    acc0 += p0.x + p1.x + p2.x + p3.x;
                    acc1 += p0.y + p1.y + p2.y + p3.y;
                }
            } else {
                for (int c = 0; c < nck; ++c) {
                    const uint32_t c2 = cp[u][c];
                    const float2 p0 = T2[c * 512 + (c2 >> 8)], p1 = T2[c * 512 + 256 + (c2 & 255)];
                    acc0 += p0.x + p1.x;
                    acc1 += p0.y + p1.y;
                }
            }
            const int gg = gs[u];
            const rvq_mm_desc & d = a.m[sg_m[gg]];
            if (sg_s0[gg] >= 0) {
                part[d.part_off + ((size_t) kt * n_pairs + t0 * n_used + sg_s0[gg]) * d.rows + rr[u]] = acc0;
            }
            if (sg_s1[gg] >= 0) {
                part[d.part_off + ((size_t) kt * n_pairs + t1 * n_used + sg_s1[gg]) * d.rows + rr[u]] = acc1;
            }
        }
    }
}

// one block row per (matrix, pair): owned slots get sum over K tiles * gain, the others zero
static __global__ void rvq_lut_multi_reduce(const rvq_mm_args a, const float * __restrict__ part, const char * __restrict__ ids,
                                            const int n_used, const int n_pairs, const int n_kt,
                                            const size_t nb_ids0, const size_t nb_ids1) {
    const rvq_mm_desc & d = a.m[blockIdx.z];
    const int p = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= d.rows) {
        return;
    }
    const int t = p / n_used, s = p % n_used;
    const uint32_t e = *(const int32_t *) (ids + t * nb_ids1 + s * nb_ids0) - d.ida;
    float * y = (float *) (d.dst + t * d.nb2 + s * d.nb1);
    if (e >= (uint32_t) d.n_expert) {
        y[r] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < n_kt; ++k) {
        acc += part[d.part_off + ((size_t) k * n_pairs + p) * d.rows + r];
    }
    y[r] = acc * __half2float(d.gain[(size_t) e * d.rows + r]);
}

static std::mutex                           rvq_knob_mtx;
static std::map<std::string, int>          rvq_knob_ctl;   // values from the control file
static int                                 rvq_ctl_epoch = 0;

int ggml_cuda_rvq_knob(const char * name, const int def) {
    std::lock_guard<std::mutex> lock(rvq_knob_mtx);
    const auto it = rvq_knob_ctl.find(name);
    if (it != rvq_knob_ctl.end()) {
        return it->second;
    }
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

int ggml_cuda_rvq_ctl_poll() {
    static const char * path = getenv("GGML_RVQ_CTL");
    if (!path) {
        return 0;
    }
    static std::string last;
    FILE * f = fopen(path, "r");
    std::string cur;
    if (f) {
        char buf[4096];
        const size_t n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
        cur.assign(buf, n);
    }
    std::lock_guard<std::mutex> lock(rvq_knob_mtx);
    if (cur != last) {
        last = cur;
        rvq_knob_ctl.clear();
        size_t p = 0;
        while (p < cur.size()) {
            size_t q = cur.find('\n', p);
            if (q == std::string::npos) {
                q = cur.size();
            }
            const std::string line = cur.substr(p, q - p);
            const size_t eq = line.find('=');
            if (eq != std::string::npos && eq > 0) {
                rvq_knob_ctl[line.substr(0, eq)] = atoi(line.c_str() + eq + 1);
            }
            p = q + 1;
        }
        ++rvq_ctl_epoch;
        fprintf(stderr, "rvq_ctl: epoch %d:", rvq_ctl_epoch);
        for (const auto & kv : rvq_knob_ctl) {
            fprintf(stderr, " %s=%d", kv.first.c_str(), kv.second);
        }
        fprintf(stderr, "\n");
    }
    return rvq_ctl_epoch;
}

// largest token count for the grouped launches (GGML_RVQ_MULTI_TOK); with several tokens they use the token-fastest grid
static int rvq_multi_max_tok() {
    return ggml_cuda_rvq_knob("GGML_RVQ_MULTI_TOK", 8);
}
// grid order for several tokens (GGML_RVQ_MULTI_TFAST, default 1)
static int rvq_multi_tfast(const int n_tok) {
    return n_tok > 1 && ggml_cuda_rvq_knob("GGML_RVQ_MULTI_TFAST", 1);
}

static bool rvq_multi_ok(const ggml_tensor * n, const ggml_tensor * first) {
    if (n->op != GGML_OP_MUL_MAT_ID || !n->src[0] || n->src[0]->type != GGML_TYPE_RVQ2X8 ||
        n->op_params[14] != GGML_CUDA_RVQ_GROUP_MAGIC || n->op_params[13] == GGML_CUDA_RVQ_GLU_MAGIC || n->src[6] ||
        !n->src[3] || !n->src[4] || n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n)) {
        return false;
    }
    if (first) {
        return n->src[1] == first->src[1] && n->src[2] == first->src[2] && n->src[4] == first->src[4] &&
               n->src[0]->ne[0] == first->src[0]->ne[0];
    }
    const ggml_tensor * x = n->src[1];
    // single-token decode only: with several tokens (speculative verify) the per-token blocks re-read shared experts and
    // the grouped path measured slower than the per-group zfast path (C, 3 drafts: code 30.7 vs 32.4 tok/s)
    return n->src[2]->ne[1] <= rvq_multi_max_tok() &&
           x->type == GGML_TYPE_F32 && x->ne[1] == 1 && x->nb[0] == sizeof(float) &&
           n->src[2]->ne[0] <= GGML_CUDA_RVQ_MAX_USED && !ggml_cuda_mul_mat_id_rvq_needs_sync(n);
}

int ggml_cuda_rvq_multi_match(const ggml_cgraph * cgraph, int i) {
    const int on = ggml_cuda_rvq_knob("GGML_RVQ_MULTI", 1);
    if (!on) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    if (!rvq_multi_ok(first, nullptr)) {
        return 0;
    }
    int n = 1;
    while (n < RVQ_MM_MAX && i + n < cgraph->n_nodes && rvq_multi_ok(cgraph->nodes[i + n], first)) {
        ++n;
    }
    return n >= 2 ? n : 0;
}

void ggml_cuda_rvq_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n) {
    static int dbg = [] { const char * v = getenv("GGML_RVQ_MULTI_DEBUG"); return v ? atoi(v) : 0; }();
    if (dbg > 0) {
        --dbg;
        fprintf(stderr, "rvq_multi: node %d, %d matrices:", i, n);
        for (int m = 0; m < n; ++m) {
            fprintf(stderr, " %s(rows %lld, ida %d)", cgraph->nodes[i + m]->name, (long long) cgraph->nodes[i + m]->src[0]->ne[1], cgraph->nodes[i + m]->op_params[12]);
        }
        fprintf(stderr, "\n");
    }
    const ggml_tensor * n0  = cgraph->nodes[i];
    const ggml_tensor * x   = n0->src[1];
    const ggml_tensor * ids = n0->src[2];
    const ggml_tensor * book = n0->src[4];
    const int K      = n0->src[0]->ne[0];
    const int n_used = ids->ne[0];
    const int n_tok  = ids->ne[1];
    const int n_pairs = n_tok * n_used;
    const int n_kt   = (K / 8 + RVQ_KT - 1) / RVQ_KT;
    // K tiles a block (rvq_lut_multi_g); 1: rvq_lut_multi, a block a tile
    const int ktg    = std::max(1, ggml_cuda_rvq_knob(n_tok > 1 ? "GGML_RVQ_KTG_T" : "GGML_RVQ_KTG", n_tok > 1 ? 4 : 1));
    const int n_kg   = (n_kt + ktg - 1) / ktg;

    rvq_mm_args a = {};
    a.n_m = n;
    size_t off = 0;
    int max_rows = 0, sum_rows = 0;
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const ggml_tensor * w = d->src[0];
        rvq_check_tiled(w);
        GGML_ASSERT(d->src[3]->type == GGML_TYPE_F16 && d->src[3]->ne[0] == w->ne[1]);
        a.m[m].codes    = (const char *) w->data;
        a.m[m].gain     = (const half *) d->src[3]->data;
        a.m[m].dst      = (char *) d->data;
        a.m[m].nb02     = w->nb[2];
        a.m[m].nb1      = d->nb[1];
        a.m[m].nb2      = d->nb[2];
        a.m[m].part_off = off;
        a.m[m].rows     = w->ne[1];
        a.m[m].n_expert = w->ne[2];
        a.m[m].ida      = d->op_params[12];
        off += (size_t) n_kg * n_pairs * w->ne[1];
        max_rows = std::max(max_rows, (int) w->ne[1]);
        sum_rows += w->ne[1];
    }
    // owned virtual rows per token: a slot belongs to one expert range (ida, n_expert); matrices of the same range (gate
    // and up of a group) all own it. If the ranges are disjoint the bound is n_used * the largest per-range row sum.
    int range_rows = 0;
    bool disjoint = true;
    for (int m = 0; m < n; ++m) {
        int rr = 0;
        for (int m2 = 0; m2 < n; ++m2) {
            const rvq_mm_desc & p = a.m[m], & q = a.m[m2];
            if (p.ida == q.ida && p.n_expert == q.n_expert) {
                rr += q.rows;
            } else if (p.ida < q.ida + q.n_expert && q.ida < p.ida + p.n_expert) {
                disjoint = false;
            }
        }
        range_rows = std::max(range_rows, rr);
    }
    const int rpb_env = ggml_cuda_rvq_knob(n_tok > 1 ? "GGML_RVQ_MULTI_RPB_T" : "GGML_RVQ_MULTI_RPB", n_tok > 1 ? 4096 : 2048);
    const int rpb   = std::max(256, rpb_env);
    const int nvmax = n_used * (disjoint ? range_rows : sum_rows);
    const int n_rc  = (nvmax + rpb - 1) / rpb;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> part(ctx.pool(), off);
    if (n_tok > 1 && ktg == 1 && ggml_cuda_rvq_knob("GGML_RVQ_MULTI_PAIR", 0)) {
        // two tokens a block (rvq_lut_multi_pair): a pair's union owns up to twice one token's rows
        static bool attr = false;
        const int smem = RVQ_KT * 512 * (int) sizeof(float2);
        if (!attr) {
            CUDA_CHECK(cudaFuncSetAttribute(rvq_lut_multi_pair, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr = true;
        }
        const int n_rc2 = (2 * nvmax + rpb - 1) / rpb;
        const dim3 gridp((n_tok + 1) / 2, n_rc2, n_kt);
        rvq_lut_multi_pair<<<gridp, 256, smem, stream>>>(a, (const float *) book->data, (const char *) x->data, (const char *) ids->data,
            part.ptr, K, n_used, n_pairs, n_tok, x->nb[2], ids->nb[0], ids->nb[1], rpb);
    } else if (ktg > 1) {
        const int rpb_g = std::min(rpb, 16 * 256);
        const int n_rc_g = (nvmax + rpb_g - 1) / rpb_g;
        const dim3 gridg(n_tok, n_rc_g, n_kg);
        if (rpb_g <= 8 * 256) {
            rvq_lut_multi_g<8><<<gridg, 256, 0, stream>>>(a, (const float *) book->data, (const char *) x->data, (const char *) ids->data,
                part.ptr, K, n_used, n_pairs, x->nb[2], ids->nb[0], ids->nb[1], rpb_g, ktg, n_kt);
        } else {
            rvq_lut_multi_g<16><<<gridg, 256, 0, stream>>>(a, (const float *) book->data, (const char *) x->data, (const char *) ids->data,
                part.ptr, K, n_used, n_pairs, x->nb[2], ids->nb[0], ids->nb[1], rpb_g, ktg, n_kt);
        }
    } else {
        const int tf = rvq_multi_tfast(n_tok);
        const dim3 grid = tf ? dim3(n_tok, n_rc, n_kt) : dim3(n_kt, n_rc, n_tok);
        rvq_lut_multi<<<grid, 256, 0, stream>>>(a, (const float *) book->data, (const char *) x->data, (const char *) ids->data,
            part.ptr, K, n_used, n_pairs, x->nb[2], ids->nb[0], ids->nb[1], rpb, tf, ggml_cuda_rvq_knob("GGML_RVQ_DBG_SAMEIDS", 0));
    }
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid2((max_rows + 255) / 256, n_pairs, n);
    rvq_lut_multi_reduce<<<grid2, 256, 0, stream>>>(a, part.ptr, (const char *) ids->data, n_used, n_pairs, n_kg, ids->nb[0], ids->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

// =====================================================================================================================
// decode, the down projections of all expert groups of a layer in one launch. Every (token, slot) has exactly one owner
// group; a block (K tile, row chunk, token*slot) finds it in the prologue, builds the table from that group's softcap-GLU
// input, and leaves at once if its K tile lies past that group's width. The reduce writes every slot of the last node.
// =====================================================================================================================

struct rvq_dn_desc {
    const char * codes;
    const half * gain;
    const char * g;
    const char * u;
    size_t       nb02, nb11, nb12;
    int          K, n_expert, ida;
};
struct rvq_dn_args {
    rvq_dn_desc m[RVQ_MM_MAX];
    int         n_m;
};

static __device__ __forceinline__ int rvq_dn_owner(const rvq_dn_args & a, const int id, uint32_t & e) {
    for (int m = 0; m < a.n_m; ++m) {
        const uint32_t le = (uint32_t) (id - a.m[m].ida);
        if (le < (uint32_t) a.m[m].n_expert) {
            e = le;
            return m;
        }
    }
    return -1;
}

static __global__ void __launch_bounds__(256)
rvq_lut_down_multi(const rvq_dn_args a, const float * __restrict__ book, const char * __restrict__ ids, float * __restrict__ part,
                   const int rows, const int n_used, const int n_pairs, const float cg, const float cu,
                   const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block, const int tfast) {
    __shared__ float T[RVQ_KT * 512];
    __shared__ float sx[RVQ_KT * 8];

    const int kt  = tfast ? blockIdx.z : blockIdx.x;
    const int rc  = blockIdx.y;
    const int z   = tfast ? blockIdx.x : blockIdx.z;   // t * n_used + j
    const int t   = z / n_used;
    const int j   = z % n_used;
    const int tid = threadIdx.x;

    const int id = *(const int32_t *) (ids + t * nb_ids1 + j * nb_ids0);
    uint32_t e = 0;
    const int m = rvq_dn_owner(a, id, e);
    if (m < 0) {
        return;
    }
    const rvq_dn_desc & d = a.m[m];
    const int C   = d.K / 8;
    const int c0  = kt * RVQ_KT;
    if (c0 >= C) {
        return;   // this group is narrower than the widest one
    }
    const int nck = min(RVQ_KT, C - c0);
    const int v0  = rc * rows_per_block;
    const int v1  = min(rows, v0 + rows_per_block);

    const float * gr = (const float *) (d.g + t * d.nb12 + j * d.nb11) + c0 * 8;
    const float * ur = (const float *) (d.u + t * d.nb12 + j * d.nb11) + c0 * 8;
    for (int i = tid; i < nck * 8; i += blockDim.x) {
        sx[i] = rvq_act(gr[i], ur[i], cg, cu);
    }
    __syncthreads();
    for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
        const int c = idx >> 9, i = idx & 511;
        const float4 * b = (const float4 *) (book + i * 8);
        const float4 * w = (const float4 *) (sx + c * 8);
        const float4 b0 = __ldg(b), b1 = __ldg(b + 1), w0 = w[0], w1 = w[1];
        T[idx] = b0.x*w0.x + b0.y*w0.y + b0.z*w0.z + b0.w*w0.w + b1.x*w1.x + b1.y*w1.y + b1.z*w1.z + b1.w*w1.w;
    }
    __syncthreads();

    const char * base = d.codes + (size_t) e * d.nb02 + (size_t) c0 * rows * 2;
    float * pp = part + ((size_t) kt * n_pairs + z) * rows;
    if (nck == RVQ_KT) {
        for (int vb = v0 + tid; vb < v1; vb += blockDim.x * RVQ_LUT_UNROLL) {
            uint4 q[RVQ_LUT_UNROLL][2];
#pragma unroll
            for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                const int v = vb + u * blockDim.x;
                if (v < v1) {
                    const uint4 * cr = (const uint4 *) (base + (size_t) v * 32);
                    q[u][0] = __ldg(cr); q[u][1] = __ldg(cr + 1);
                }
            }
#pragma unroll
            for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                const int v = vb + u * blockDim.x;
                if (v >= v1) {
                    continue;
                }
                const uint32_t w[8] = {q[u][0].x, q[u][0].y, q[u][0].z, q[u][0].w, q[u][1].x, q[u][1].y, q[u][1].z, q[u][1].w};
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t lo = w[k] & 0xffff, hi = w[k] >> 16;
                    const float * Ta = T + (2*k) * 512;
                    const float * Tb = Ta + 512;
                    acc += Ta[lo >> 8] + Ta[256 + (lo & 255)] + Tb[hi >> 8] + Tb[256 + (hi & 255)];
                }
                pp[v] = acc;
            }
        }
    } else {
        for (int v = v0 + tid; v < v1; v += blockDim.x) {
            const uint16_t * cr = (const uint16_t *) base + (size_t) v * nck;
            float acc = 0.0f;
            for (int c = 0; c < nck; ++c) {
                const uint32_t c2 = cr[c];
                acc += T[c * 512 + (c2 >> 8)] + T[c * 512 + 256 + (c2 & 255)];
            }
            pp[v] = acc;
        }
    }
}

// rvq_lut_down_multi over ktg consecutive K tiles a block, the rows' sums kept in registers across the tiles (see
// rvq_lut_multi_g). grid (token*slot, row chunk, K group).
template <int RPT>
static __global__ void __launch_bounds__(256)
rvq_lut_down_multi_g(const rvq_dn_args a, const float * __restrict__ book, const char * __restrict__ ids, float * __restrict__ part,
                     const int rows, const int n_used, const int n_pairs, const float cg, const float cu,
                     const size_t nb_ids0, const size_t nb_ids1, const int rows_per_block, const int ktg) {
    __shared__ float T[RVQ_KT * 512];
    __shared__ float sx[RVQ_KT * 8];

    const int z   = blockIdx.x;            // t * n_used + j
    const int rc  = blockIdx.y;
    const int kg  = blockIdx.z;
    const int t   = z / n_used;
    const int j   = z % n_used;
    const int tid = threadIdx.x;

    const int id = *(const int32_t *) (ids + t * nb_ids1 + j * nb_ids0);
    uint32_t e = 0;
    const int m = rvq_dn_owner(a, id, e);
    if (m < 0) {
        return;
    }
    const rvq_dn_desc & d = a.m[m];
    const int C    = d.K / 8;
    const int n_kt = (C + RVQ_KT - 1) / RVQ_KT;
    if (kg * ktg >= n_kt) {
        return;   // this group is narrower than the widest one
    }
    const int v0 = rc * rows_per_block;
    const int v1 = min(rows, v0 + rows_per_block);
    const float * gr = (const float *) (d.g + t * d.nb12 + j * d.nb11);
    const float * ur = (const float *) (d.u + t * d.nb12 + j * d.nb11);

    float acc[RPT];
#pragma unroll
    for (int q = 0; q < RPT; ++q) {
        acc[q] = 0.0f;
    }
    const int kt_end = min(n_kt, (kg + 1) * ktg);
    for (int kt = kg * ktg; kt < kt_end; ++kt) {
        const int c0  = kt * RVQ_KT;
        const int nck = min(RVQ_KT, C - c0);
        __syncthreads();
        for (int i = tid; i < nck * 8; i += blockDim.x) {
            sx[i] = rvq_act(gr[c0 * 8 + i], ur[c0 * 8 + i], cg, cu);
        }
        __syncthreads();
        for (int idx = tid; idx < nck * 512; idx += blockDim.x) {
            const int c = idx >> 9, i = idx & 511;
            const float4 * b = (const float4 *) (book + i * 8);
            const float4 * w = (const float4 *) (sx + c * 8);
            const float4 b0 = __ldg(b), b1 = __ldg(b + 1), w0 = w[0], w1 = w[1];
            T[idx] = b0.x*w0.x + b0.y*w0.y + b0.z*w0.z + b0.w*w0.w + b1.x*w1.x + b1.y*w1.y + b1.z*w1.z + b1.w*w1.w;
        }
        __syncthreads();
        const char * base = d.codes + (size_t) e * d.nb02 + (size_t) c0 * rows * 2;
        if (nck == RVQ_KT) {
#pragma unroll
            for (int q0 = 0; q0 < RPT; q0 += RVQ_LUT_UNROLL) {
                uint4 qq[RVQ_LUT_UNROLL][2];
#pragma unroll
                for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                    const int v = v0 + (q0 + u) * blockDim.x + tid;
                    if (v < v1) {
                        const uint4 * cr = (const uint4 *) (base + (size_t) v * 32);
                        qq[u][0] = __ldg(cr); qq[u][1] = __ldg(cr + 1);
                    }
                }
#pragma unroll
                for (int u = 0; u < RVQ_LUT_UNROLL; ++u) {
                    const int v = v0 + (q0 + u) * blockDim.x + tid;
                    if (v >= v1) {
                        continue;
                    }
                    const uint32_t w[8] = {qq[u][0].x, qq[u][0].y, qq[u][0].z, qq[u][0].w, qq[u][1].x, qq[u][1].y, qq[u][1].z, qq[u][1].w};
                    float ts = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const uint32_t lo = w[k] & 0xffff, hi = w[k] >> 16;
                        const float * Ta = T + (2*k) * 512;
                        const float * Tb = Ta + 512;
                        ts += Ta[lo >> 8] + Ta[256 + (lo & 255)] + Tb[hi >> 8] + Tb[256 + (hi & 255)];
                    }
                    acc[q0 + u] += ts;
                }
            }
        } else {
#pragma unroll
            for (int q = 0; q < RPT; ++q) {
                const int v = v0 + q * blockDim.x + tid;
                if (v >= v1) {
                    continue;
                }
                const uint16_t * cr = (const uint16_t *) base + (size_t) v * nck;
                float ts = 0.0f;
                for (int c = 0; c < nck; ++c) {
                    const uint32_t c2 = cr[c];
                    ts += T[c * 512 + (c2 >> 8)] + T[c * 512 + 256 + (c2 & 255)];
                }
                acc[q] += ts;
            }
        }
    }
    float * pp = part + ((size_t) kg * n_pairs + z) * rows;
#pragma unroll
    for (int q = 0; q < RPT; ++q) {
        const int v = v0 + q * blockDim.x + tid;
        if (v < v1) {
            pp[v] = acc[q];
        }
    }
}

static __global__ void rvq_lut_down_reduce(const rvq_dn_args a, const float * __restrict__ part, const char * __restrict__ ids,
                                           char * __restrict__ dst, const int rows, const int n_used, const int n_pairs,
                                           const size_t nb_ids0, const size_t nb_ids1, const size_t nb1, const size_t nb2,
                                           const int ktg) {
    const int p = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const int t = p / n_used, s = p % n_used;
    const int id = *(const int32_t *) (ids + t * nb_ids1 + s * nb_ids0);
    uint32_t e = 0;
    const int m = rvq_dn_owner(a, id, e);
    float * y = (float *) (dst + t * nb2 + s * nb1);
    if (m < 0) {
        y[r] = 0.0f;
        return;
    }
    const int n_kt = ((a.m[m].K / 8 + RVQ_KT - 1) / RVQ_KT + ktg - 1) / ktg;
    float acc = 0.0f;
    for (int k = 0; k < n_kt; ++k) {
        acc += part[((size_t) k * n_pairs + p) * rows + r];
    }
    y[r] = acc * __half2float(a.m[m].gain[(size_t) e * rows + r]);
}

static bool rvq_down_ok(const ggml_tensor * n, const ggml_tensor * prev_node, const ggml_tensor * first) {
    if (n->op != GGML_OP_MUL_MAT_ID || !n->src[0] || n->src[0]->type != GGML_TYPE_RVQ2X8 ||
        n->op_params[14] != GGML_CUDA_RVQ_GROUP_MAGIC || n->op_params[13] != GGML_CUDA_RVQ_GLU_MAGIC ||
        !n->src[3] || !n->src[4] || !n->src[5] || n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) ||
        n->src[6] != prev_node) {
        return false;
    }
    const ggml_tensor * g = n->src[1], * u = n->src[5];
    if (n->src[2]->ne[1] > rvq_multi_max_tok()) {
        return false;   // see rvq_multi_ok
    }
    if (g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || g->nb[0] != sizeof(float) || !ggml_are_same_shape(g, u) ||
        g->nb[1] != u->nb[1] || g->nb[2] != u->nb[2] || g->ne[1] != n->src[2]->ne[0] || n->src[2]->ne[0] > GGML_CUDA_RVQ_MAX_USED ||
        ggml_cuda_mul_mat_id_rvq_needs_sync(n)) {
        return false;
    }
    if (first) {
        return n->src[2] == first->src[2] && n->src[4] == first->src[4] && n->src[0]->ne[1] == first->src[0]->ne[1] &&
               memcmp(&n->op_params[10], &first->op_params[10], 2 * sizeof(int32_t)) == 0 &&
               ggml_are_same_shape(n, first) && n->nb[1] == first->nb[1] && n->nb[2] == first->nb[2];
    }
    return true;
}

int ggml_cuda_rvq_down_match(const ggml_cgraph * cgraph, int i) {
    const int on = ggml_cuda_rvq_knob("GGML_RVQ_DOWN_MULTI", 1);
    if (!on) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    if (!rvq_down_ok(first, nullptr, nullptr)) {
        return 0;
    }
    int n = 1;
    while (n < RVQ_MM_MAX && i + n < cgraph->n_nodes && rvq_down_ok(cgraph->nodes[i + n], cgraph->nodes[i + n - 1], first)) {
        ++n;
    }
    if (n < 2) {
        return 0;
    }
    // the earlier links of the chain must feed only the next link
    int idxs[RVQ_MM_MAX];
    ggml_op ops[RVQ_MM_MAX];
    for (int k = 0; k < n; ++k) {
        idxs[k] = i + k;
        ops[k]  = GGML_OP_MUL_MAT_ID;
    }
    const int outputs[] = { i + n - 1 };
    return ggml_can_fuse_subgraph_ext(cgraph, idxs, n, ops, outputs, 1) ? n : 0;
}

void ggml_cuda_rvq_down_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n) {
    const ggml_tensor * n0   = cgraph->nodes[i];
    const ggml_tensor * last = cgraph->nodes[i + n - 1];
    const ggml_tensor * ids  = n0->src[2];
    const int rows    = n0->src[0]->ne[1];
    const int n_used  = ids->ne[0];
    const int n_tok   = ids->ne[1];
    const int n_pairs = n_tok * n_used;
    float cg, cu;
    memcpy(&cg, &n0->op_params[10], sizeof(float));
    memcpy(&cu, &n0->op_params[11], sizeof(float));

    rvq_dn_args a = {};
    a.n_m = n;
    int kmax = 0;
    for (int m = 0; m < n; ++m) {
        const ggml_tensor * d = cgraph->nodes[i + m];
        const ggml_tensor * w = d->src[0];
        rvq_check_tiled(w);
        a.m[m].codes    = (const char *) w->data;
        a.m[m].gain     = (const half *) d->src[3]->data;
        a.m[m].g        = (const char *) d->src[1]->data;
        a.m[m].u        = (const char *) d->src[5]->data;
        a.m[m].nb02     = w->nb[2];
        a.m[m].nb11     = d->src[1]->nb[1];
        a.m[m].nb12     = d->src[1]->nb[2];
        a.m[m].K        = w->ne[0];
        a.m[m].n_expert = w->ne[2];
        a.m[m].ida      = d->op_params[12];
        kmax = std::max(kmax, (int) w->ne[0]);
    }
    const int n_kt = (kmax / 8 + RVQ_KT - 1) / RVQ_KT;
    const int rpb_env = ggml_cuda_rvq_knob(n_tok > 1 ? "GGML_RVQ_DOWN_RPB_T" : "GGML_RVQ_DOWN_RPB", 2048);
    const int rpb  = std::max(256, rpb_env);
    const int n_rc = (rows + rpb - 1) / rpb;

    cudaStream_t stream = ctx.stream();
    // K tiles a block (rvq_lut_down_multi_g); 1: rvq_lut_down_multi
    const int ktg  = std::max(1, ggml_cuda_rvq_knob(n_tok > 1 ? "GGML_RVQ_DKTG_T" : "GGML_RVQ_DKTG", n_tok > 1 ? 2 : 1));
    const int n_kg = (n_kt + ktg - 1) / ktg;
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) n_kg * n_pairs * rows);
    if (ktg > 1) {
        const int rpb_g  = std::min(rpb, 16 * 256);
        const int n_rc_g = (rows + rpb_g - 1) / rpb_g;
        const dim3 gridg(n_pairs, n_rc_g, n_kg);
        if (rpb_g <= 8 * 256) {
            rvq_lut_down_multi_g<8><<<gridg, 256, 0, stream>>>(a, (const float *) n0->src[4]->data, (const char *) ids->data, part.ptr,
                rows, n_used, n_pairs, cg, cu, ids->nb[0], ids->nb[1], rpb_g, ktg);
        } else {
            rvq_lut_down_multi_g<16><<<gridg, 256, 0, stream>>>(a, (const float *) n0->src[4]->data, (const char *) ids->data, part.ptr,
                rows, n_used, n_pairs, cg, cu, ids->nb[0], ids->nb[1], rpb_g, ktg);
        }
    } else {
        const int tf = rvq_multi_tfast(n_tok);
        const dim3 grid = tf ? dim3(n_pairs, n_rc, n_kt) : dim3(n_kt, n_rc, n_pairs);
        rvq_lut_down_multi<<<grid, 256, 0, stream>>>(a, (const float *) n0->src[4]->data, (const char *) ids->data, part.ptr,
            rows, n_used, n_pairs, cg, cu, ids->nb[0], ids->nb[1], rpb, tf);
    }
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid2((rows + 255) / 256, n_pairs);
    rvq_lut_down_reduce<<<grid2, 256, 0, stream>>>(a, part.ptr, (const char *) ids->data, (char *) last->data, rows, n_used, n_pairs,
        ids->nb[0], ids->nb[1], last->nb[1], last->nb[2], ktg);
    CUDA_CHECK(cudaGetLastError());
}
