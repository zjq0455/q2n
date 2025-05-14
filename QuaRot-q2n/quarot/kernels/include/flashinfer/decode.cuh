#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "math.cuh"
#include "cp_async.cuh"
#include "layout.cuh"
#include "page.cuh"
#include "rope.cuh"
#include "state.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"
#include "quantization.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

namespace {

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <size_t vec_size, size_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> apply_llama_rope(
    const T *x,
    const vec_t<float, vec_size> &freq,
    size_t offset,
    float scale = 0.f,
    float zero = 0.f
  ) {
  constexpr size_t head_dim = vec_size * bdx;
  // vec / permuted_vec must be float type. Thus has cast_load()
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(
    quant::get_ptr(x, threadIdx.x * vec_size),
    scale,
    zero
  );
  permuted_vec.cast_load(
    quant::get_ptr(x, ((threadIdx.x * vec_size < head_dim / 2)
                                  ? threadIdx.x * vec_size + head_dim / 2
                                  : threadIdx.x * vec_size - head_dim / 2)),
    scale, zero
  );

#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * freq[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos +
             ((threadIdx.x * vec_size < head_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
  return vec;
}

/*!
 * \brief Load k tile from smem and compute qk.
 * \tparam rotary_mode The rotary mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache.
 * \param compute_stage_idx A integer indicates the compute stage index in
 *   the pipeline
 * \param num_heads A integer indicates the number of heads
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param x A float indicates the thread-local result of qk
 */
template <RotaryMode rotary_mode, size_t vec_size, size_t bdx, typename T>
__device__ __forceinline__ void compute_qk(
  const T *smem, const vec_t<float, vec_size> &q_vec,
  const vec_t<float, vec_size> &freq, size_t kv_idx,
  size_t compute_stage_idx, size_t num_heads,
  float sm_scale, float &x,
  float scale=0.f, float zero=0.f
) {
  size_t tx = threadIdx.x;
  vec_t<float, vec_size> k_vec;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
    // apply rotary embedding for all rows in k matrix of kv-cache
    k_vec = apply_llama_rope<vec_size, bdx>(
      smem, freq, kv_idx,
      scale, zero
    );
  } else {
    // do not apply rotary embedding
    k_vec.cast_load(
      quant::get_ptr(smem, tx * vec_size),
      scale, zero
    );
  }
  x = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    x += q_vec[i] * k_vec[i] * sm_scale;
  }
#pragma unroll
  for (size_t offset = bdx / 2; offset > 0; offset /= 2) {
    x += math::shfl_xor_sync(x, offset);
  }
}

/*!
 * \brief Load v tile from shared memory and update partial state.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam T A template type indicates the input data type
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \param smem A pointer to the start of shared memory
 * \param x A float indicates the pre-softmax logits
 * \param kv_shared_offset An array of size_t indicates the k/v tiles offset in shared
 *   memory of different pipeline stages.
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param pred_guard A boolean indicates whether the current thread is in the valid range
 * \param s The flashattention state to be updated
 */
template <size_t vec_size, size_t bdx, typename T, bool norm_on_the_fly>
__device__ __forceinline__ void update_partial_state(
  const T *smem, const float x,
  size_t compute_stage_idx, bool pred_guard,
  state_t<vec_size, norm_on_the_fly> &s,
  float scale=0.f, float zero=0.f
) {
  vec_t<float, vec_size> v_vec;
  size_t tx = threadIdx.x;
  v_vec.cast_load(
    quant::get_ptr(smem, tx * vec_size),
    scale, zero
  );
  if (pred_guard) {
    s.merge(v_vec, x);
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \param s The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <size_t vec_size, size_t bdx, size_t bdy, bool norm_on_the_fly>
__device__ __forceinline__ void sync_state(state_t<vec_size, norm_on_the_fly> &s, float *smem,
                                           float *smem_md) {
  constexpr size_t head_dim = bdx * vec_size;
  auto block = cg::this_thread_block();
  size_t tx = threadIdx.x, ty = threadIdx.y;
  s.o.store(smem + ty * head_dim + tx * vec_size);
  smem_md[ty * 2] = s.m;
  smem_md[ty * 2 + 1] = s.d;
  block.sync();
  s.init();
#pragma unroll
  for (size_t j = 0; j < bdy; ++j) {
    float mj = smem_md[j * 2], dj = smem_md[j * 2 + 1];
    vec_t<float, vec_size> oj;
    oj.load(smem + j * head_dim + tx * vec_size);
    s.merge(oj, mj, dj);
  }
}

template <size_t bdx, size_t bdy>
__device__ __forceinline__ void load_chunk_param(
  half2* quantParamK,
  half2* quantParamV,
  half2* smem_quantParamK,
  half2* smem_quantParamV,
  size_t iter,
  size_t kv_chunk_size
){
  constexpr size_t iterBound = bdx * bdy * 16 / 2 / sizeof(half2) / bdy; // 8
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t offset = tx + (ty % (bdy / 2)) * bdx;
  // Here we avoid illegal memory address by restricting the alignment of quantParamK to 16bytes.
  bool predGuard = (iter % iterBound == 0) && (iter * bdy + offset * 16 / sizeof(half2) < kv_chunk_size);
  if(predGuard){
    half2* src_ptr = (ty < bdy / 2) ? quantParamK : quantParamV;
    half2* dst_ptr = (ty < bdy / 2) ? smem_quantParamK : smem_quantParamV;
    *((float4*)(dst_ptr) + offset) = *((float4*)(src_ptr) + offset);
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single
 * sequence, fused with RoPE.
 * \tparam layout The layout of k/v matrices (NHD or HND)
 * \tparam cooperative Whether to use cooperative kernel or not
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \tparam rotary_mode The rotary mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeInQ A template type indicates the Query input data type
 * \tparam DTypeIn A template type indicates the K/V input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q [num_heads, head_dim] The query matrix
 * \param k [seq_len, num_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_heads, head_dim] The value matrix in kv-cache
 * \param o [num_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param kv_info The tensor info of k/v matrices
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param head_dim A integer indicates the head dimension
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 * \param quantParamK A pointer to packed {Scale, Zero} for K, aligned for 16bytes.
 * \param quantParamV A pointer to packed {Scale, Zero} for V, aligned for 16bytes.
 */
template <QKVLayout layout, bool cooperative, bool norm_on_the_fly, RotaryMode rotary_mode,
          size_t vec_size, size_t bdx, size_t bdy, typename DTypeInQ, typename DTypeIn, typename DTypeOut>
__global__ void SingleDecodeWithKVCacheKernel(
  DTypeInQ *__restrict__ q, DTypeIn *__restrict__ k,
  DTypeIn *__restrict__ v, DTypeOut *__restrict__ o,
  float *__restrict__ tmp,
  tensor_info_t<layout> kv_info, float sm_scale,
  float rope_inv_scale, float rope_inv_theta,
  size_t kv_chunk_size,
  half2* __restrict__ quantParamK,
  half2* __restrict__ quantParamV
){
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  sm_scale *= math::log2e;

  constexpr size_t num_stages_smem = 4;
  constexpr size_t head_dim = bdx * vec_size;
  size_t head_idx = blockIdx.y;
  size_t kv_chunk_idx = blockIdx.x;
  size_t num_kv_chunks = gridDim.x;
  size_t num_heads = gridDim.y;
  size_t seq_len = kv_info.kv_len;

  static_assert(bdx * bdy == 128);
  // Enough smem to sync partial states within a threadblock
  static_assert(num_stages_smem >= sizeof(float) / quant::size_of_type<DTypeIn>() / 2);
  // NOTE: use uint8_t as smem type. Take attention to the pointer offset.
  __shared__ uint8_t smem[static_cast<size_t>(2 * num_stages_smem * bdy * head_dim * quant::size_of_type<DTypeIn>())];
  DTypeIn *k_smem = reinterpret_cast<DTypeIn*>(smem);
  DTypeIn *v_smem = quant::get_ptr(k_smem, num_stages_smem * bdy * head_dim);
  __shared__ float smem_md[2 * bdy];

  // Define smem for quantization parameter
  // We first load them into smem to reduce the latency.
  constexpr size_t param_chunk_size = bdx * bdy * 16 / 2 / sizeof(half2);
  __shared__ half2 smem_quantParam[param_chunk_size * 2];
  half2* quantParamK_smem = smem_quantParam;
  half2* quantParamV_smem = quantParamK_smem + param_chunk_size;

  size_t tx = threadIdx.x, ty = threadIdx.y;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      freq[i] =
          rope_inv_scale *
          powf(rope_inv_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_llama_rope<vec_size, bdx>(quant::get_ptr(q, head_idx * head_dim), freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(quant::get_ptr(q, head_idx * head_dim + tx * vec_size));
  }
  block.sync();

  size_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  size_t chunk_end = chunk_start + kv_chunk_size;

  size_t tx_mem = tx % 4;
  size_t ty_mem = ty * 2 + tx / 4;
  // preload k tiles and v tiles
  size_t producer_kv_idx_base = chunk_start + ty_mem;
  constexpr size_t vec_bits = quant::size_of_type<DTypeIn>() * vec_size * 8 * 2;
  // Half warps do not load
  // 0.5 * 16 * 8 = 64 bits per thread -> * 2 = 128bits 

#pragma unroll
  for (size_t iter = 0; iter < num_stages_smem; ++iter) {
    size_t producer_kv_idx = producer_kv_idx_base;
    cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(k_smem, (iter * bdy + ty_mem) * head_dim + tx_mem * vec_size * 2),
        quant::get_ptr(k, kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx_mem * vec_size * 2)),
        (producer_kv_idx < chunk_end) && (ty_mem < bdy));
    cp_async::commit_group();
    cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(v_smem, (iter * bdy + ty_mem) * head_dim + tx_mem * vec_size * 2),
        quant::get_ptr(v, kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx_mem * vec_size * 2)),
        (producer_kv_idx < chunk_end) && (ty_mem < bdy));
    cp_async::commit_group();
    producer_kv_idx_base += bdy;
  }

  // pipelining k/v tiles loading and state updating
  size_t consumer_kv_idx_base = chunk_start + ty, stage_idx = 0;
  state_t<vec_size, norm_on_the_fly> s_partial;
  float x = 0.f;
  float2 paramK, paramV;

#pragma unroll 4
  for (size_t iter = 0; iter < (kv_chunk_size + bdy - 1) / bdy; ++iter) {
    size_t producer_kv_idx = producer_kv_idx_base, consumer_kv_idx = consumer_kv_idx_base;
    // load K quantization parameter
    load_chunk_param<bdx, bdy>(
      quantParamK + head_idx * seq_len + chunk_start + iter * bdy,
      quantParamV + head_idx * seq_len + chunk_start + iter * bdy,
      quantParamK_smem,
      quantParamV_smem,
      iter, kv_chunk_size
    );
    block.sync();
    constexpr size_t iterBound = bdx * bdy * 16 / 2 / sizeof(half2) / bdy;
    paramK = __half22float2(quantParamK_smem[iter % iterBound * bdy + ty]);
    paramV = __half22float2(quantParamV_smem[iter % iterBound * bdy + ty]);
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<rotary_mode, vec_size, bdx>(
      quant::get_ptr(k_smem, (stage_idx * bdy + ty) * head_dim),
      q_vec, freq,
      consumer_kv_idx, stage_idx, num_heads, sm_scale, x,
      paramK.x, paramK.y
    );
    block.sync();
    // load k
    cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(k_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * 2),
        quant::get_ptr(k, kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx_mem * vec_size * 2)),
        (producer_kv_idx < chunk_end) && (ty_mem < bdy));
    cp_async::commit_group();
    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_partial_state<vec_size, bdx>(
      quant::get_ptr(v_smem, (stage_idx * bdy + ty) * head_dim),
      x, stage_idx,
      consumer_kv_idx < chunk_end, s_partial,
      paramV.x, paramV.y
    );
    block.sync();

    // load v
    cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(v_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * 2),
        quant::get_ptr(v, kv_info.get_kv_elem_offset(producer_kv_idx, head_idx, tx_mem * vec_size * 2)),
        (producer_kv_idx < chunk_end) && (ty_mem < bdy));
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += bdy;
    consumer_kv_idx_base += bdy;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy>(s_partial, reinterpret_cast<float *>(smem), smem_md);

  if constexpr (cooperative) {
    // update tmp buffer
    s_partial.o.store(tmp + (head_idx * num_kv_chunks + kv_chunk_idx) * head_dim + tx * vec_size);
    float *tmp_md = tmp + num_heads * num_kv_chunks * head_dim;
    tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2] = s_partial.m;
    tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2 + 1] = s_partial.d;
    grid.sync();

    // sync global states
    if (kv_chunk_idx == 0) {
      state_t<vec_size, norm_on_the_fly> s_global;
#pragma unroll 4
      for (size_t iter = 0; iter < (num_kv_chunks + bdy - 1) / bdy; ++iter) {
        size_t kv_chunk_idx = iter * bdy + ty;
        if (kv_chunk_idx < num_kv_chunks) {
          s_partial.m = tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2];
          s_partial.d = tmp_md[(head_idx * num_kv_chunks + kv_chunk_idx) * 2 + 1];
          s_partial.o.load(tmp + (head_idx * num_kv_chunks + kv_chunk_idx) * head_dim +
                           tx * vec_size);
          s_global.merge(s_partial);
        }
      }
      block.sync();
      // sync partial state of all warps inside a threadblock
      sync_state<vec_size, bdx, bdy>(s_global, reinterpret_cast<float *>(smem), smem_md);
      s_global.normalize();
      s_global.o.cast_store(o + head_idx * head_dim + tx * vec_size);
      tmp[head_idx] = s_global.m;
      tmp[num_heads + head_idx] = s_global.d;
    }
  } else {
    s_partial.normalize();
    s_partial.o.cast_store(o + head_idx * head_dim + tx * vec_size);
  }
}

template <typename DType, typename IdType>
__forceinline__ __device__ void AdvancePageIterator(
  paged_kv_t<DType, IdType> paged_kv,
  size_t *kv_idx_base,
  size_t *valid_page_size,
  size_t &producer_valid_page_size,
  size_t &producer_entry_base,
  size_t &producer_page_iter,
  size_t &producer_page_idx,
  size_t cur_page_indptr_begin,
  size_t cur_page_indptr_end,
  size_t batch_idx,
  size_t stage_idx
) {
  const size_t ty = threadIdx.y;
  if (producer_entry_base >= producer_valid_page_size) {
    producer_entry_base = 0;
    producer_page_iter += 1;
    if (producer_page_iter < cur_page_indptr_end) {
      producer_page_idx = paged_kv.indices[producer_page_iter];
      producer_valid_page_size = paged_kv.get_valid_page_size(batch_idx, producer_page_iter);
    } else {
      producer_valid_page_size = 0;
    }
  }
  // Used for position consumer embedding
  kv_idx_base[stage_idx] = producer_entry_base + ty + (producer_page_iter - cur_page_indptr_begin) * paged_kv.page_size;
  valid_page_size[stage_idx] = producer_valid_page_size;
}

/*!
 * \brief FlashAttention decoding cuda kernel with PagedKVCcache for batch requests,
 *   fused with RoPE.
 * \tparam rotary_mode The rotary mode
 * \tparam norm_on_the_fly Whether to normalize on the fly or not
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeInQ A template type indicates the Query input data type
 * \tparam DTypeIn A template type indicates the KV Cache input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_heads, head_dim] The query matrix
 * \param paged_kv The PagedKVCache data structure
 * \param o [num_heads, head_dim] The output matrix
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_inv_scale A floating number indicate the multiplicative inverse
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_inv_theta A floating number indicate the multiplicative inverse
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template<
  RotaryMode rotary_mode,
  bool norm_on_the_fly,
  size_t vec_size,
  size_t bdx,
  size_t bdy,
  size_t FoldFactor,
  typename DTypeInQ,
  typename DTypeIn,
  typename DTypeOut,
  typename IdType
>
__global__ void BatchDecodeWithPagedKVCacheKernel(
  DTypeInQ *__restrict__ q,
  paged_kv_t<DTypeIn, IdType> paged_kv,
  DTypeOut *__restrict__ o,
  float sm_scale,
  float rope_inv_scale,
  float rope_inv_theta
) {
  auto block = cg::this_thread_block();
  sm_scale *= math::log2e;

  constexpr size_t num_stages_smem = 4;
  constexpr size_t head_dim = bdx * vec_size;
  size_t batch_idx = blockIdx.x;
  size_t head_idx = blockIdx.y;
  size_t num_heads = gridDim.y;
  
  // [cur_page_indptr_begin, cur_page_indptr_end)
  size_t cur_page_indptr_begin = paged_kv.indptr[batch_idx], cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
  size_t cur_last_page_offset = paged_kv.last_page_offset[batch_idx]; // Perfectly equal to seq_len here
  size_t seq_len = (cur_page_indptr_end - cur_page_indptr_begin - 1) * paged_kv.page_size + cur_last_page_offset;

  static_assert(bdx * bdy == 128);
  // Use uint8_t to allocate data as bytes
  static_assert(num_stages_smem >= sizeof(float) / quant::size_of_type<DTypeIn>() / 2);
  __shared__ uint8_t smem[
    static_cast<size_t>(2 * num_stages_smem * bdy * head_dim * quant::size_of_type<DTypeIn>())
  ];
  DTypeIn *k_smem = reinterpret_cast<DTypeIn*>(smem);
  DTypeIn *v_smem = quant::get_ptr(k_smem, num_stages_smem * bdy * head_dim);

  __shared__ float smem_md[2 * bdy];

  size_t tx = threadIdx.x, ty = threadIdx.y;
  // Remap the mem loading pointer
  size_t tx_mem = tx % (bdx / FoldFactor), ty_mem = ty * FoldFactor + tx / (bdx / FoldFactor);

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_inv_scale *
                __powf(rope_inv_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
    // apply rotary embedding to q matrix
    q_vec = apply_llama_rope<vec_size, bdx>(
      quant::get_ptr(q, (batch_idx * num_heads + head_idx) * head_dim),
      freq,
      seq_len - 1
    );
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(
      quant::get_ptr(q, (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size)
    );
  }
  block.sync();

  // preload k/v tiles
  size_t producer_entry_base = 0, stage_idx = 0;
  constexpr size_t vec_bits = quant::size_of_type<DTypeIn>() * vec_size * 8 * FoldFactor; // load K/V in 128bits
  static_assert(vec_bits == 128, "We want to highly utilize the memory bandwidth");

  size_t producer_page_iter = cur_page_indptr_begin;
  size_t producer_page_idx = paged_kv.indices[producer_page_iter];
  size_t producer_valid_page_size = paged_kv.get_valid_page_size(batch_idx, producer_page_iter);
  size_t kv_idx_base[num_stages_smem]{0};
  size_t valid_page_size[num_stages_smem]{0};
#pragma unroll
  for (size_t iter = 0; iter < num_stages_smem; ++iter) {
    AdvancePageIterator(
      paged_kv,
      kv_idx_base,
      valid_page_size,
      producer_valid_page_size,
      producer_entry_base,
      producer_page_iter,
      producer_page_idx,
      cur_page_indptr_begin,
      cur_page_indptr_end,
      batch_idx,
      stage_idx
    );
    bool producer_pred_guard = (producer_entry_base + ty_mem < producer_valid_page_size) &&
                               (producer_page_iter < cur_page_indptr_end) &&
                               (ty_mem < bdy);
    cp_async::pred_load<vec_bits, true>(
      quant::get_ptr(k_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * FoldFactor),
      quant::get_ptr(paged_kv.data, paged_kv.get_k_elem_offset(producer_page_idx, head_idx, producer_entry_base + ty_mem, tx_mem * vec_size * FoldFactor)),
      producer_pred_guard
    );
    cp_async::commit_group();
    cp_async::pred_load<vec_bits, true>(
      quant::get_ptr(v_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * FoldFactor),
      quant::get_ptr(paged_kv.data, paged_kv.get_v_elem_offset(producer_page_idx, head_idx, producer_entry_base + ty_mem, tx_mem * vec_size * FoldFactor)),
      producer_pred_guard
    );
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_entry_base += bdy;
  }

  state_t<vec_size, norm_on_the_fly> s;
  float x = 0.f;
  size_t consumer_kv_idx_base = 0;

  for (size_t consumer_page_iter = cur_page_indptr_begin; consumer_page_iter < cur_page_indptr_end;
       ++consumer_page_iter) {
    size_t consumer_valid_page_size = valid_page_size[stage_idx];
    size_t consumer_kv_page_idx = paged_kv.indices[consumer_page_iter];

#pragma unroll 4
    for (size_t iter = 0; iter < (consumer_valid_page_size + bdy - 1) / bdy; ++iter) {
      // NOTE: Maybe we should open page_size >= bdy for fully utilization
      consumer_kv_idx_base = kv_idx_base[stage_idx];
      size_t consumer_kv_entry = iter * bdy + ty;

      bool consumer_pred_guard = (consumer_kv_entry < consumer_valid_page_size);
      AdvancePageIterator(
        paged_kv,
        kv_idx_base,
        valid_page_size,
        producer_valid_page_size,
        producer_entry_base,
        producer_page_iter,
        producer_page_idx,
        cur_page_indptr_begin,
        cur_page_indptr_end,
        batch_idx,
        stage_idx
      );
      bool producer_pred_guard = (producer_entry_base + ty_mem < producer_valid_page_size) &&
                                 (producer_page_iter < cur_page_indptr_end) &&
                                 (ty_mem < bdy);
      // loading scales
      float2 paramK, paramV;
      if(consumer_pred_guard){
        paramK = __half22float2(
          quant::get_ptr(paged_kv.param, paged_kv.get_param_k_elem_offset(consumer_kv_page_idx, head_idx, consumer_kv_entry))[0]
        );
        paramV = __half22float2(
          quant::get_ptr(paged_kv.param, paged_kv.get_param_v_elem_offset(consumer_kv_page_idx, head_idx, consumer_kv_entry))[0]
        );
      }

      cp_async::wait_group<2 * num_stages_smem - 1>();
      block.sync();
      compute_qk<rotary_mode, vec_size, bdx>(
        quant::get_ptr(k_smem, (stage_idx * bdy + ty) * head_dim), 
        q_vec, freq, 
        consumer_kv_idx_base, stage_idx, num_heads,
        sm_scale, x,
        paramK.x, paramK.y
      );
      block.sync();

      // load k tiles
      cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(k_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * FoldFactor),
        quant::get_ptr(paged_kv.data, paged_kv.get_k_elem_offset(producer_page_idx, head_idx, producer_entry_base + ty_mem, tx_mem * vec_size * FoldFactor)),
        producer_pred_guard
      );
      cp_async::commit_group();

      // update m/d/o states
      cp_async::wait_group<2 * num_stages_smem - 1>();
      block.sync();
      update_partial_state<vec_size, bdx>(
        quant::get_ptr(v_smem, (stage_idx * bdy + ty) * head_dim), 
        x, stage_idx, consumer_pred_guard, s,
        paramV.x, paramV.y
      );
      block.sync();

      // load v tiles
      cp_async::pred_load<vec_bits, true>(
        quant::get_ptr(v_smem, (stage_idx * bdy + ty_mem) * head_dim + tx_mem * vec_size * FoldFactor),
        quant::get_ptr(paged_kv.data, paged_kv.get_v_elem_offset(producer_page_idx, head_idx, producer_entry_base + ty_mem, tx_mem * vec_size * FoldFactor)),
        producer_pred_guard
      );
      cp_async::commit_group();

      stage_idx = (stage_idx + 1) % num_stages_smem;
      producer_entry_base += bdy;
    }
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync partial state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy>(s, reinterpret_cast<float *>(smem), smem_md);
  s.normalize();

  // update global states
  s.o.cast_store(o + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single sequence
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \param q The query matrix, shape: [num_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_heads, head_dim]
 *   for NHD layout, [num_heads, head_dim, seq_len] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_heads, head_dim]
 *   for NHD layout, [num_heads, head_dim, seq_len] for HND layout
 * \param o The output matrix, shape: [num_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_heads A integer indicates the number of heads
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param layout The layout of q/k/v matrices.
 * \param rotary_mode The rotary mode
 * \param rope_scale A floating point number indicate the scaling ratio
 *   used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 */
template <typename DTypeInQ, typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(
  DTypeInQ *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
  size_t num_heads, size_t seq_len, size_t head_dim,
  QKVLayout layout = QKVLayout::kNHD,
  RotaryMode rotary_mode = RotaryMode::kNone,
  float rope_scale = 1.f, float rope_theta = 1e4,
  cudaStream_t stream = nullptr, size_t dev_id = 0,
  half2* quantParamK = nullptr,
  half2* quantParamV = nullptr
) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;
  constexpr bool norm_on_the_fly = false;

  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));

  SWITCH_HEAD_DIM(
      head_dim, HEAD_DIM,
      {SWITCH_ROTARY_MODE(
          rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, QKV_LAYOUT, {
            constexpr size_t vec_size = std::max(static_cast<size_t>(16 / quant::size_of_type<DTypeIn>()), HEAD_DIM / 32) / 2;
            constexpr size_t bdx = HEAD_DIM / vec_size;
            constexpr size_t bdy = 128 / bdx;
            tensor_info_t<QKV_LAYOUT> kv_info(1, seq_len, num_heads, head_dim);
            if (seq_len <= 128) {
              // no need to use cooperative kernel
              auto kernel =
                  SingleDecodeWithKVCacheKernel<QKV_LAYOUT, false, norm_on_the_fly, ROTARY_MODE,
                                                vec_size, bdx, bdy, DTypeInQ, DTypeIn, DTypeOut>;
              dim3 nblks = dim3(1, num_heads);
              dim3 nthrs = dim3(bdx, bdy);
              void *args[] = {(void *)&q,
                              (void *)&k,
                              (void *)&v,
                              (void *)&o,
                              (void *)&tmp,
                              (void *)&kv_info,
                              (void *)&sm_scale,
                              (void *)&rope_inv_scale,
                              (void *)&rope_inv_theta,
                              (void *)&seq_len,
                              (void *)&quantParamK,
                              (void *)&quantParamV
                              };
              FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args, 0, stream));
            } else {
              // use cooperative kernel
              auto kernel =
                  SingleDecodeWithKVCacheKernel<QKV_LAYOUT, true, norm_on_the_fly, ROTARY_MODE,
                                                vec_size, bdx, bdy, DTypeInQ, DTypeIn, DTypeOut>;
              int num_blocks_per_sm = 0;
              int num_sm = 0;
              FLASHINFER_CUDA_CALL(
                  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
              FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                                                 kernel, 128, 0));
              size_t max_num_blks = size_t(num_blocks_per_sm) * size_t(num_sm);
              size_t max_num_kv_chunks = max_num_blks / num_heads;
              size_t kv_chunk_size =
                  max((seq_len + max_num_kv_chunks - 1UL) / max_num_kv_chunks,
                      min(128UL, max(16UL, seq_len / max(1UL, (128UL / num_heads)))));
              dim3 nblks = dim3((seq_len + kv_chunk_size - 1) / kv_chunk_size, num_heads);
              assert(nblks.x > 0 && nblks.y > 0);
              dim3 nthrs = dim3(bdx, bdy);
              void *args[] = {(void *)&q,
                              (void *)&k,
                              (void *)&v,
                              (void *)&o,
                              (void *)&tmp,
                              (void *)&kv_info,
                              (void *)&sm_scale,
                              (void *)&rope_inv_scale,
                              (void *)&rope_inv_theta,
                              (void *)&kv_chunk_size,
                              (void *)&quantParamK,
                              (void *)&quantParamV
                              };
              FLASHINFER_CUDA_CALL(
                  cudaLaunchCooperativeKernel((void *)kernel, nblks, nthrs, args, 0, stream));
            }
          })})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for batched requests
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [batch_size, num_heads, head_dim] The query matrix
 * \param paged_kv The paged kv cache data structure
 * \param o [batch_size, num_heads, head_dim] The output matrix
 * \param rotary_mode The rotary mode
 * \param rope_scale A floating point number indicate the scaling ratio
 *   used in RoPE Interpolation.
 * \param rope_theta A floating point number indicate the "theta" used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \param dev_id The device id
 */
template <typename DTypeInQ, typename DTypeIn, typename DTypeOut, typename IdType, size_t FoldFactor = 2>
cudaError_t BatchDecodeWithPagedKVCache(
  DTypeInQ *q,
  paged_kv_t<DTypeIn, IdType> paged_kv,
  DTypeOut *o,
  RotaryMode rotary_mode = RotaryMode::kNone,
  float rope_scale = 1.f,
  float rope_theta = 1e4,
  cudaStream_t stream = nullptr,
  size_t dev_id = 0
) {
  const float sm_scale = 1.f / std::sqrt(float(paged_kv.head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;
  constexpr bool norm_on_the_fly = false;

  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  SWITCH_HEAD_DIM(
      paged_kv.head_dim, HEAD_DIM, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
        // NOTE: Here we keep calculation in float same.
        // reduce memory movement by disabling warps.
        constexpr size_t vec_size = std::max(static_cast<size_t>(16 / quant::size_of_type<DTypeIn>() / FoldFactor), HEAD_DIM / 32);
        constexpr size_t bdx = HEAD_DIM / vec_size;
        constexpr size_t bdy = 128 / bdx;
        dim3 nblks(paged_kv.batch_size, paged_kv.num_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel = BatchDecodeWithPagedKVCacheKernel<ROTARY_MODE, norm_on_the_fly, vec_size, bdx,
                                                        bdy, FoldFactor, DTypeInQ, DTypeIn, DTypeOut, IdType>;
        void *args[] = {(void *)&q,        (void *)&paged_kv,       (void *)&o,
                        (void *)&sm_scale, (void *)&rope_inv_scale, (void *)&rope_inv_theta};
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args, 0, stream));
      })});

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
