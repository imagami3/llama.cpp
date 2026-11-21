#pragma once
#include "common.cuh"

// ink2-rvq fork: MUL_MAT_ID with RVQ2X8 experts (B70/XeStrata format).
//   src[0] codes  RVQ2X8 [K, rows, E]  (one uint16 per 8 consecutive inputs, code = i1<<8 | i2)
//   src[1] x      F32    [K, ne11, n_tok]
//   src[2] ids    I32    [n_used, n_tok]  (ids >= GGML_CUDA_MMID_SKIP: slot belongs to another expert group -> zeros)
//   src[3] gain   F16    [rows, E]
//   src[4] book   F32    [8, 256, 2]      (w[r][8c+j] = gain[r] * (book[0][i1][j] + book[1][i2][j]))

// up to this many tokens the LUT decode kernel is used (no host sync, CUDA-graph safe); above: the prefill path
#define GGML_CUDA_RVQ_MMV_MAX_TOKENS 8

// MUL_MAT_ID op_params[14] marker: ids are router-global and the group starts at op_params[12]
#define GGML_CUDA_RVQ_GROUP_MAGIC 0x52565147
// op_params[13] marker: down projection with fused softcap-GLU input (src1 = gate, src[5] = up, caps in op_params[10..11])
#define GGML_CUDA_RVQ_GLU_MAGIC 0x474C5531

// max experts used per token supported by the LUT decode kernel
#define GGML_CUDA_RVQ_MAX_USED 32

void ggml_cuda_mul_mat_id_rvq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_mul_mat_id_rvq_needs_sync(const ggml_tensor * dst);

// decode: one launch for several RVQ gate/up MUL_MAT_ID nodes that share src1 and ids (the expert groups of a layer).
// i .. i+n-1 must be such nodes; returns n (>= 2) when they can be fused, else 0.
// runtime knobs: env NAME (default def), overridable while running from the file named by GGML_RVQ_CTL ("NAME=VALUE"
// lines). ggml_cuda_rvq_ctl_poll() re-reads that file when it changes and returns an epoch that bumps on every change
// (the CUDA graphs are re-captured then), so one process can A/B kernel variants without reloading the model.
int  ggml_cuda_rvq_knob(const char * name, int def);
int  ggml_cuda_rvq_ctl_poll();
int  ggml_cuda_rvq_multi_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_rvq_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n);

// decode: the chained GLU down projections of a layer's expert groups (each node takes the previous one's sum, src[6])
// in one launch; returns the number of chained nodes (>= 2) at i, else 0. Only the last node's output is written.
int  ggml_cuda_rvq_down_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_rvq_down_multi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n);
