#pragma once
#include "common.cuh"

// ink2-tq fork: MUL_MAT_ID with B70's trellis tq2 experts (b70/trellis/ref/tq_ref.py).
//   src[0] tiles  TQ2T [K*16, rows/16, E]  (per expert: rows/16 strips x K/16 tiles of 66 bytes; tile = 16x16 weights row-major,
//                 bytes 0-1 = start state s_0 (14 bits), bytes 2..65 = 255 2-bit symbols; s_t = (s_{t-1} >> 2) | (b_t << 12))
//   src[1] x      F32  [K, ne11, n_tok]
//   src[2] ids    I32  [n_used, n_tok]
//   src[3] gain   F16  [rows, E]
//   src[4] sign   F32  [K]           (x' = H64(sign * x) / 8 per 64-block; W_rot[r, c] = gain[r] * value(s_rc))
//   value(s) = (bytesum(s * 34038481 + 76625530) - 510) / 147.8
// op_params[14] == GGML_CUDA_RVQ_GROUP_MAGIC: ids are router-global and this group owns [op_params[12], +E) (others: zeros)

// up to this many tokens the decode kernel (no host sync, CUDA-graph safe); above: f16 dequant + cuBLAS
#define GGML_CUDA_TQ_MMV_MAX_TOKENS 8

void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_mul_mat_id_tq_needs_sync(const ggml_tensor * dst);
// prefill node that the GPU-sorted multi path handles without a host sync (CUDA graphs may stay on)
bool ggml_cuda_tq_pf_graph_ok(const ggml_tensor * dst);

// decode: one launch for the gate/up nodes of all expert groups of a layer (same src1 and ids), and one for the chained
// GLU-down nodes (src[6] = previous node; only the last node's output is written). match returns n (>= 2) or 0.
int  ggml_cuda_tq_gu_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_tq_gu_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n);
int  ggml_cuda_tq_down_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_tq_down_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n);
bool ggml_cuda_tq_down_multi_red(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n,
                                 ggml_tensor * dst, const ggml_tensor * w, const ggml_tensor * sc);
