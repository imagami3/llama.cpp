#pragma once

#include "common.cuh"

// GB10 single-token matvec for Q4_K / Q5_K weights with f32 activations (no q8_1 quantization).
// One CUDA block per row: the row is staged into shared memory with 16-byte loads before the PDL grid-dependency
// wait, so the weight read of this kernel overlaps the tail of the previous one. Env GGML_KQMV=0 disables it.
bool ggml_cuda_kqmv_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_kqmv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
