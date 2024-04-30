// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <cassert>
#include <cstring>
#include <vector>

#include "common.cuh"
#include "common.h"
#include "debug.h"

#include "fmoe_expert_kernel.h"

namespace nvinfer1 {
namespace plugin {

__global__ void ScatterMappingKernel(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                                     int* acc_histogram) {
  int idx = threadIdx.x;
  extern __shared__ int his[];
  if (idx < num_expert + 1) his[idx] = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    /*if (gate_idx[i] < 0 || gate_idx[i] > num_expert) return;*/
    auto old = atomicAdd(&his[gate_idx[i] + 1], 1);
    mapping[i] = old;
  }

  __syncthreads();

  // acc his
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_expert; i++) his[i + 1] += his[i];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    mapping[i] += his[gate_idx[i]];
  }

  if (idx < num_expert + 1) acc_histogram[idx] = his[idx];
}
int ComputeScatterMapping(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                          int* acc_histogram, cudaStream_t stream) {
  int block_size = 0;
  if (idx_num < 1024)
    block_size = 256;
  else if (idx_num < 4096)
    block_size = 512;
  else
    block_size = 1024;
  /*printf("block_size=%d, idx_num=%d, num_expert=%d\n", block_size, idx_num, num_expert);*/
  /*print_data(gate_idx, idx_num, "gate_idx");*/
  ScatterMappingKernel<<<1, block_size, (num_expert + 1) * sizeof(int), stream>>>(gate_idx, num_expert, idx_num,
                                                                                  mapping, acc_histogram);
  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

__global__ void ScatterMappingKernel(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                                     int* inverse_mapping, int* acc_histogram) {
  int idx = threadIdx.x;
  extern __shared__ int his[];
  if (idx < num_expert + 1) his[idx] = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    /*if (gate_idx[i] < 0 || gate_idx[i] > num_expert) return;*/
    auto old = atomicAdd(&his[gate_idx[i] + 1], 1);
    mapping[i] = old;
  }

  __syncthreads();

  // acc his
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_expert; i++) his[i + 1] += his[i];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    mapping[i] += his[gate_idx[i]];
    inverse_mapping[mapping[i]] = i;
  }

  if (idx < num_expert + 1) acc_histogram[idx] = his[idx];
}

int ComputeScatterMapping(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                          int* inverse_mapping, int* acc_histogram, cudaStream_t stream) {
  int block_size = 0;
  if (idx_num < 1024)
    block_size = 256;
  else if (idx_num < 4096)
    block_size = 512;
  else
    block_size = 1024;
  /*printf("block_size=%d, idx_num=%d, num_expert=%d\n", block_size, idx_num, num_expert);*/
  /*print_data(gate_idx, idx_num, "gate_idx");*/
  ScatterMappingKernel<<<1, block_size, (num_expert + 1) * sizeof(int), stream>>>(
      gate_idx, num_expert, idx_num, mapping, inverse_mapping, acc_histogram);
  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

__global__ void IntToCharKernel(const int* weight_int32, const int volume, char* weight_int8) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  weight_int8[idx] = (char)weight_int32[idx];
}

int TransformInt32ToInt8(const int* weight_int32, const int volume, char* weight_int8, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = (volume + block_size - 1) / block_size;

  IntToCharKernel<<<grid_size, block_size, 0, stream>>>(weight_int32, volume, weight_int8);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

template <typename T, int TPB>
__global__ void Trans2CharRowScatterMapping(const T* fp_data, const int* mapping, const int idim, char* char_data,
                                            T* scales_data) {
  int tidx = threadIdx.x;
  // if (tidx >= idim) return;

  fp_data += blockIdx.x * idim;
  char_data += mapping[blockIdx.x] * idim;
  
  T s_max = (T)0.0;
  
  for (int i = tidx; i < idim; i += TPB) {
    T tmp_abs = fabsf(fp_data[i]);
    if (s_max < tmp_abs) {
      // printf("grid:%d thread:%d change from %f to %f \n", blockIdx.x,tidx, s_max[tidx], fp32_data[i]);
      s_max = tmp_abs;
    }
  }

  using BlockReduce = cub::BlockReduce<T, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T dim_max =  BlockReduce(temp_storage).Reduce(s_max, cub::Max());
  
  if (tidx == 0) {
    scales_data[mapping[blockIdx.x]] = 255.0 / 2.0 / dim_max;
  }
  __syncthreads();

  // quantization
  for (int i = tidx; i < idim; i += blockDim.x) {
    bool positive = (fp_data[i] > (T)0.0);
    T tmp = fp_data[i] * scales_data[mapping[blockIdx.x]];
    if (tmp <= -128)
      char_data[i] = -128;
    else if (tmp >= 127)
      char_data[i] = 127;
    else
      char_data[i] = (signed char)(tmp + (positive ? 1 : -1) * 0.5);
  }
}

int QuantizedScatterMappingCopy(const float* input, const int* mapping, const int S, const int idim,
                                char* input_buffer_int8, float* scale, cudaStream_t stream) {
  if (input == nullptr || input_buffer_int8 == nullptr || scale == nullptr || mapping == nullptr) return -1;

  const int block_1d_size = 256;
  int grid_1d_size = S;

  Trans2CharRowScatterMapping<float, block_1d_size><<<grid_1d_size, block_1d_size, 0, stream>>>(
      input, mapping, idim, input_buffer_int8, scale);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int QuantizedScatterMappingCopy(const half* input, const int* mapping, const int S, const int idim,
                                char* input_buffer_int8, half* scale, cudaStream_t stream) {
  if (input == nullptr || input_buffer_int8 == nullptr || scale == nullptr || mapping == nullptr) return -1;

  const int block_1d_size = 256;
  int grid_1d_size = S;

  Trans2CharRowScatterMapping<half, block_1d_size><<<grid_1d_size, block_1d_size, 0, stream>>>(
      input, mapping, idim, input_buffer_int8, scale);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

// int QuantizedScatterMappingCopy(const float* input, const int* mapping, const int S, const int idim, signed char*
// input_buffer_int8, float* scale, cudaStream_t stream) {

//   return QuantizedScatterMappingCopyTpl(input, mapping, S, idim, input_buffer_int8, scale, stream);
// }

// int QuantizedScatterMappingCopy(const half* input, const int* mapping, const int S, const int idim, signed char*
// input_buffer_int8, half* scale, cudaStream_t stream) {
//   return QuantizedScatterMappingCopyTpl(input, mapping, S, idim, input_buffer_int8, scale, stream);
// }

template <typename T>
__global__ void ScatterMappingCopyKernel(const T* input, const int* mapping, const int dim, const int numel,
                                         T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int s = idx / dim;
  int i = idx % dim;

  int mapping_idx = mapping[s];

  output[mapping_idx * dim + i] = input[idx];
}

template <typename T>
int ComputeScatterMappingCopyTpl(const T* input, const int* mapping, const int S, const int dim, T* output,
                                 cudaStream_t stream) {
  auto numel = S * dim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  ScatterMappingCopyKernel<T><<<grid_size, block_size, 0, stream>>>(input, mapping, dim, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeScatterMappingCopy(const float* input, const int* mapping, const int S, const int dim, float* output,
                              cudaStream_t stream) {
  return ComputeScatterMappingCopyTpl(input, mapping, S, dim, output, stream);
}

int ComputeScatterMappingCopy(const half* input, const int* mapping, const int S, const int dim, half* output,
                              cudaStream_t stream) {
  return ComputeScatterMappingCopyTpl(input, mapping, S, dim, output, stream);
}

template <typename T, int TPB>
__global__ void DequantizedBiasSiluAndQuantizeKernel(const int* input_int32, const T* input_scale,
                                                     const T* weight_scale, const T* weight_bias,
                                                     const int hidden_dims, char* char_data, T* scale) {
  int tidx = threadIdx.x;
  T temp = 0.0;
  T dequantized = 0.0;
  if (tidx >= hidden_dims) return;

  input_int32 += blockIdx.x * hidden_dims;
  char_data += blockIdx.x * hidden_dims;
  // input_scale += blockIdx.x;

  T s_max = (T)0.0;

  for (int i = tidx; i < hidden_dims; i += blockDim.x) {
    //  when calculate fp16, overload operator "/" in common.cuh
    dequantized = (input_int32[i]) / input_scale[blockIdx.x] / weight_scale[i] + weight_bias[i];
    temp = fabsf(dequantized * sigmoid(dequantized));
    // if (blockIdx.x==0 && i==1) {
    //   printf("input:%d scale:%f  wscale:%f  wbias:%f\n",input_int32[i], (float)input_scale[blockIdx.x], (float)weight_scale[i],
    //   (float)weight_bias[i]); printf("dequantized:%f fabsf:%f\n", (float)dequantized, (float)temp);
    // }

    if (s_max < temp) {
      // printf("grid:%d thread:%d change from %f to %f \n", blockIdx.x,tidx, s_max[tidx], fp32_data[i]);
      s_max = temp;
    }
  }

  
  using BlockReduce = cub::BlockReduce<T, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T dim_max =  BlockReduce(temp_storage).Reduce(s_max, cub::Max());

  if (tidx == 0) {
    // if (s_max[0] < s_max[1]) {
    //   s_max[0] = s_max[1];
    // }
    // printf("max_abs = %f\n", s_max[0]);
    scale[blockIdx.x] = 255.0 / 2.0 / dim_max;
  }
  __syncthreads();

  // quantization
  for (int i = tidx; i < hidden_dims; i += blockDim.x) {
    dequantized = input_int32[i] / input_scale[blockIdx.x] / weight_scale[i] + weight_bias[i];
    temp = dequantized * sigmoid(dequantized) * scale[blockIdx.x];
    if (temp <= -128)
      char_data[i] = -128;
    else if (temp >= 127)
      char_data[i] = 127;
    else
      char_data[i] = (signed char)(temp + (temp > 0 ? 1 : -1) * 0.5f);
  }
}

int DequantizedBiasSiluAndQuantize(const int* input_int32, const float* input_scale, const float* weight_scale,
                                   const float* weight_bias, const int m, const int hidden_dims,
                                   char* Layer2_input_int8, float* Layer2_scale, cudaStream_t stream) {
  if (input_int32 == nullptr || input_scale == nullptr || weight_scale == nullptr || weight_bias == nullptr ||
      Layer2_input_int8 == nullptr || Layer2_scale == nullptr)
    return -1;

  const int block_1d_size = 256;
  int grid_1d_size = m;

  DequantizedBiasSiluAndQuantizeKernel<float, block_1d_size><<<grid_1d_size, block_1d_size, 0, stream>>>(
      input_int32, input_scale, weight_scale, weight_bias, hidden_dims, Layer2_input_int8, Layer2_scale);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int DequantizedBiasSiluAndQuantize(const int* input_int32, const half* input_scale, const half* weight_scale,
                                   const half* weight_bias, const int m, const int hidden_dims,
                                   char* Layer2_input_int8, half* Layer2_scale, cudaStream_t stream) {
  if (input_int32 == nullptr || input_scale == nullptr || weight_scale == nullptr || weight_bias == nullptr ||
  Layer2_input_int8 == nullptr || Layer2_scale == nullptr)
  return -1;

  const int block_1d_size = 256;
  int grid_1d_size = m;

  DequantizedBiasSiluAndQuantizeKernel<half, block_1d_size><<<grid_1d_size, block_1d_size, 0, stream>>>(
  input_int32, input_scale, weight_scale, weight_bias, hidden_dims, Layer2_input_int8, Layer2_scale);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

template <typename T>
__global__ void DequantizedBiasGatherMappingKernel(const int* input_int32, const T* input_scale,
                                                   const T* weight_scale, const T* weight_bias,
                                                   const int* inverse_mapping, const int idim, const int acc_idex,
                                                   const int numel, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int s = idx / idim;
  int i = idx % idim;

  int inverse_mapping_idx = inverse_mapping[s + acc_idex];

  output[inverse_mapping_idx * idim + i] =
  (input_int32[idx] / weight_scale[i]) / input_scale[s] + weight_bias[i];  // overload operator"/" (int / half)

  // if (blockIdx.x==0) {
  //     printf("result %f input:%d scale:%f  wscale:%f  wbias:%f\n", (float)output[inverse_mapping_idx * idim + i],
  //     input_int32[idx], (float)input_scale[s], (float)weight_scale[i],
  //     (float)weight_bias[i]);
  // }
}

int DequantizedBiasGatherMapping(const int* input_int32, const float* input_scale, const float* weight_scale,
                                 const float* weight_bias, const int m, const int idim, const int acc_idex,
                                 const int* inverse_mapping, float* output, cudaStream_t stream) {
  auto numel = m * idim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  DequantizedBiasGatherMappingKernel<<<grid_size, block_size, 0, stream>>>(
      input_int32, input_scale, weight_scale, weight_bias, inverse_mapping, idim, acc_idex, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

int DequantizedBiasGatherMapping(const int* input_int32, const half* input_scale, const half* weight_scale,
                                 const half* weight_bias, const int m, const int idim, const int acc_idex,
                                 const int* inverse_mapping, half* output, cudaStream_t stream) {
  auto numel = m * idim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  DequantizedBiasGatherMappingKernel<<<grid_size, block_size, 0, stream>>>(
      input_int32, input_scale, weight_scale, weight_bias, inverse_mapping, idim, acc_idex, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

template <typename T>
__global__ void BiasSiluKernel(const T* input, const T* bias, const int N, const int dim, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    int bias_idx = idx % dim;
    auto tmp = input[idx] + bias[bias_idx];
    output[idx] = tmp * sigmoid(tmp);
  }
}

template <typename T>
int ComputeBiasSiluTpl(const T* input, const T* bias, const int N, const int dim, T* output, cudaStream_t stream) {
  constexpr int block_size = 512;
  const int grid_size = (N + block_size - 1) / block_size;
  BiasSiluKernel<T><<<grid_size, block_size, 0, stream>>>(input, bias, N, dim, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

int ComputeBiasSilu(const float* input, const float* bias, const int N, const int dim, float* output,
                    cudaStream_t stream) {
  return ComputeBiasSiluTpl<float>(input, bias, N, dim, output, stream);
}

int ComputeBiasSilu(const half* input, const half* bias, const int N, const int dim, half* output,
                    cudaStream_t stream) {
  return ComputeBiasSiluTpl<half>(input, bias, N, dim, output, stream);
}

template <typename T>
__global__ void BiasKernel(const T* input, const T* bias, const int N, const int dim, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    int bias_idx = idx % dim;
    output[idx] = input[idx] + bias[bias_idx];
  }
}

template <typename T>
int ComputeBiasTpl(const T* input, const T* bias, const int N, const int dim, T* output, cudaStream_t stream) {
  constexpr int block_size = 512;
  const int grid_size = (N + block_size - 1) / block_size;
  BiasKernel<T><<<grid_size, block_size, 0, stream>>>(input, bias, N, dim, output);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeBias(const float* input, const float* bias, const int N, const int dim, float* output, cudaStream_t stream) {
  return ComputeBiasTpl<float>(input, bias, N, dim, output, stream);
}

int ComputeBias(const half* input, const half* bias, const int N, const int dim, half* output, cudaStream_t stream) {
  return ComputeBiasTpl<half>(input, bias, N, dim, output, stream);
}

template <typename T>
__global__ void GatherrMappingCopyKernel(const T* input, const int* mapping, const int dim, const int numel,
                                         T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int s = idx / dim;
  int i = idx % dim;

  int mapping_idx = mapping[s];
  output[idx] = input[mapping_idx * dim + i];
}

template <typename T>
int ComputeGatherMappingCopyTpl(const T* input, const int* mapping, const int S, const int dim, T* output,
                                cudaStream_t stream) {
  auto numel = S * dim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  GatherrMappingCopyKernel<T><<<grid_size, block_size, 0, stream>>>(input, mapping, dim, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeGatherrMappingCopy(const float* input, const int* mapping, const int S, const int dim, float* output,
                              cudaStream_t stream) {
  return ComputeGatherMappingCopyTpl(input, mapping, S, dim, output, stream);
}

int ComputeGatherrMappingCopy(const half* input, const int* mapping, const int S, const int dim, half* output,
                              cudaStream_t stream) {
  return ComputeGatherMappingCopyTpl(input, mapping, S, dim, output, stream);
}

}  // namespace plugin
}  // namespace nvinfer1