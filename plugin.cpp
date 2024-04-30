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

#include "fmoe_expert_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common.h"
#include "cublas_common.h"
#include "fmoe_expert_kernel.h"
#include "serialize.hpp"

#include "debug.h"

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
PluginFieldCollection FMoEExpertPluginCreator::FC_{};
std::vector<PluginField> FMoEExpertPluginCreator::plugin_attributes_;

REGISTER_TENSORRT_PLUGIN(FMoEExpertPluginCreator);

template <class T>
int ComputeFmoeExpert(const T* input, const int* gate_idx, const int input_volume, const int S, const int num_expert,
                        const int idim, const int hidden_units, const T* w1_weight_ptr, const T* w1_bias_ptr,
                        const T* w2_weight_ptr, const T* w2_bias_ptr, std::vector<int>& v_acc_his, void* workspace,
                        T* output, cudaStream_t stream, std::shared_ptr<CudaStreamManager> csm_ptr) {
  auto mapping_size = alignTo<int>(S, kAlignment);
  auto his_size = alignTo<int>(num_expert + 1, kAlignment);
  auto input_buffer_size = alignTo<int>(input_volume, kAlignment);

  int* mapping = static_cast<int*>(workspace);
  int* acc_histogram = mapping + mapping_size;

  int status = -1;
  status = ComputeScatterMapping(gate_idx, num_expert, S, mapping, acc_histogram, stream);
  if (status != 0) {
    LOG(ERROR) << "compute_scatter_mapping error!" << endl;
    return status;
  }

  // print_data(gate_idx, S, "gate_idx");
  // print_data(mapping, S, "mapping");
  // print_data(acc_histogram, num_expert+1, "acc_histogram");
  // print_data(w1_weight_ptr, idim, "w1_weight_ptr");
  // print_data(w2_weight_ptr, idim, "w2_weight_ptr");
  // cout << "====================" << endl;

  // const size_t word_size = getElementSize(data_type_);
  const size_t word_size = sizeof(T);

  // get buffer from workspace
  // float* input_buffer = reinterpret_cast<float*>(acc_histogram + his_size);
  // float* hidden_buffer = input_buffer + input_buffer_size;
  T* input_buffer = reinterpret_cast<T*>(acc_histogram + his_size);
  T* hidden_buffer = input_buffer + input_buffer_size;

  status = ComputeScatterMappingCopy(input, mapping, S, idim, input_buffer, stream);
  if (status != 0) {
    LOG(ERROR) << "ComputeScatterMappingCopy error!" << endl;
    return status;
  }
  // print_data(input_buffer, 10, "reorder_input0");
  // print_data(input_buffer + idim_, 10, "reorder_input1");

  int* h_acc_his = v_acc_his.data();
  cudaMemcpyAsync(h_acc_his, acc_histogram, sizeof(int) * (num_expert + 1), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  for (int i = 0; i < num_expert; i++) {
    auto cur_stream = csm_ptr->Stream(i);
    auto handle = csm_ptr->CublasHandle(i);
    int m = h_acc_his[i + 1] - h_acc_his[i];
    if (m == 0) continue;

    // float* input_buffer_ptr = input_buffer + h_acc_his[i] * idim;
    // float* hidden_buffer_ptr = hidden_buffer + h_acc_his[i] * hidden_units;
    auto input_buffer_ptr = input_buffer + h_acc_his[i] * idim;
    auto hidden_buffer_ptr = hidden_buffer + h_acc_his[i] * hidden_units;

    // Weights offset
    auto w_offset = i * idim * hidden_units;
    auto cur_w1_weight_ptr = w1_weight_ptr + w_offset;
    auto cur_w1_bias_ptr = w1_bias_ptr + i * hidden_units;
    auto cur_w2_weight_ptr = w2_weight_ptr + w_offset;
    auto cur_w2_bias_ptr = w2_bias_ptr + i * idim;

    // w1 gemm, tmp => output
    CUBLAS_CHECK(cublasGemm(handle, transa, transb, m, hidden_units, idim, 1.0f, input_buffer_ptr, cur_w1_weight_ptr,
                            0.0f, hidden_buffer_ptr));

    // w1 bias + activate, tmp2
    status = ComputeBiasSilu(hidden_buffer_ptr, cur_w1_bias_ptr, m * hidden_units, hidden_units, hidden_buffer_ptr,
                             cur_stream);
    if (status != 0) {
      LOG(ERROR) << "ComputeBiasSilu error!" << endl;
      return status;
    }

    // print_data(hidden_buffer_ptr, 10, "silu");

    // w2 gemm tmp2 => tmp1
    CUBLAS_CHECK(cublasGemm(handle, transa, transb, m, idim, hidden_units, 1.0f, hidden_buffer_ptr, cur_w2_weight_ptr,
                            0.0f, input_buffer_ptr));

    // print_data(input_buffer_ptr, 10, "input_buffer_ptr");
    // w2 bias tmp1
    status = ComputeBias(input_buffer_ptr, cur_w2_bias_ptr, m * idim, idim, input_buffer_ptr, cur_stream);
    if (status != 0) {
      LOG(ERROR) << "ComputeBias error!" << endl;
      return status;
    }

    // print_data(input_buffer_ptr, 10, "w2");
    // cout << "=================" << endl;
  }

  csm_ptr->SyncAllStream();
  // print_data(output, 512, "output");
  // print_data(input_buffer, 512, "input_buffer");
  // print_data(mapping, 2, "mapping");
  status = ComputeGatherrMappingCopy(input_buffer, mapping, S, idim, output, stream);
  if (status != 0) {
    LOG(ERROR) << "ComputeGatherrMappingCopy error!" << endl;
    return status;
  }
  // print_data(output, 10, "output");
  // cout << "=================" << endl;

  return status;
}

template <class T>
int ComputeFmoeExpertInt8(const T* input, const int* gate_idx, const int input_volume, const int weight1_volume,
                             const int weight2_volume, const int S, const int num_expert, const int idim,
                             const int hidden_units, const T* w1_bias_ptr,
                             const char* weight1_int8, const T* weight1_scale,
                             const T* w2_bias_ptr, const char* weight2_int8, const T* weight2_scale,
                             std::vector<int>& v_acc_his, void* workspace, T* output, cudaStream_t stream,
                             std::shared_ptr<CudaStreamManager> csm_ptr) {
  // const size_t word_size = sizeof(T);
  auto input_buffer_size = alignTo<int>(input_volume, kAlignment);

  // caculate sizes
  auto mapping_size = alignTo<int>(S, kAlignment);
  auto inverse_mapping_size = alignTo<int>(S, kAlignment);
  auto his_size = alignTo<int>(num_expert + 1, kAlignment);
  // auto weight1_int8_size = alignTo<int>(weight1_volume, kAlignment);
  // auto weight2_int8_size = alignTo<int>(weight2_volume, kAlignment);
  // auto weight1_scale_size = alignTo<int>(num_expert * hidden_units, kAlignment);
  // auto weight2_scale_size = alignTo<int>(num_expert * idim, kAlignment);
  auto Layer1_In_buffer_int8_size = alignTo<int>(input_volume, kAlignment);        // Input reorder & quantize
  auto Layer1_In_scale_size = alignTo<int>(S, kAlignment);                         // Layer1 input scale
  auto Layer1_Out_buffer_int32_size = alignTo<int>(S * hidden_units, kAlignment);  // MM out1

  auto Layer2_In_buffer_int8_size = alignTo<int>(S * hidden_units, kAlignment);  // Out1 scaling and silu & quantize
  auto Layer2_In_scale_size = alignTo<int>(S, kAlignment);                       // Layer2 input scale
  auto Layer2_Out_buffer_int32_size = alignTo<int>(input_volume, kAlignment);    // MM out2

  // caculate address
  int* mapping = static_cast<int*>(workspace);
  int* inverse_mapping = mapping + mapping_size;
  int* acc_histogram = inverse_mapping + inverse_mapping_size;
  int* Layer1_Out_buffer_int32 = acc_histogram + his_size;
  int* Layer2_Out_buffer_int32 = Layer1_Out_buffer_int32 + Layer1_Out_buffer_int32_size;

  // char* weight1_int8 = reinterpret_cast<char*>(Layer2_Out_buffer_int32 + Layer2_Out_buffer_int32_size);
  // char* weight2_int8 = weight1_int8 + weight1_int8_size;
  char* Layer1_In_buffer_int8 = reinterpret_cast<char*>(Layer2_Out_buffer_int32 + Layer2_Out_buffer_int32_size);
  char* Layer2_In_buffer_int8 = Layer1_In_buffer_int8 + Layer1_In_buffer_int8_size;

  T* Layer1_In_scale = reinterpret_cast<T*>(Layer2_In_buffer_int8 + Layer2_In_buffer_int8_size);
  T* Layer2_In_scale = Layer1_In_scale + Layer1_In_scale_size;
  // float* weight1_scale = Layer2_In_scale + Layer1_In_scale_size;
  // float* weight2_scale = weight1_scale + weight1_scale_size;

  // step 0: Check all variables(for debug)
  int status = -1;
  // print_data(weight1_int8, 10, 10, "weight");
  // // printf("%d -- %d --%d", mapping_size, his_size, weight1_int8_size);
  // // printf("mapping_size = %d", mapping_size);

  // step 1: Compute reeordered idex: gate_idx -> mapping & acc_histogram
  status = ComputeScatterMapping(gate_idx, num_expert, S, mapping, inverse_mapping, acc_histogram, stream);
  if (status != 0) {
    LOG(ERROR) << "compute_scatter_mapping error!" << endl;
    return status;
  }
  // printf("volum = %d\n",input_volume);
  // print_data(gate_idx, S, "gate_idx");
  // print_data(mapping, S, "mapping");
  // print_data(inverse_mapping, S, "inverse_mapping");
  // print_data(acc_histogram, num_expert+1, "acc_histogram");
  // cout << "====================" << endl;
  // cout << first_enqueue << endl;
  // cout << layer_name <<endl;

  // step 2: transform int32 weight to int8
  // print_data(w1_weight_ptr, 10, 1, "w1_weight_ptr");
  // if (first_enqueue) {
  //   // status = TransformInt32ToInt8(w1_weight_int32_ptr, weight1_volume, weight1_int8, stream);
  //   // if (status != 0) {
  //   //   LOG(ERROR) << "Transform weight1(int32 to int 8) error!" << endl;
  //   //   return status;
  //   // }
  //   // status = TransformInt32ToInt8(w2_weight_int32_ptr, weight2_volume, weight2_int8, stream);
  //   // if (status != 0) {
  //   //   LOG(ERROR) << "Transform weight2(int32 to int 8) error!" << endl;
  //   //   return status;
  //   // }
  //   status = QuantizeWeight(w1_weight_ptr, num_expert, hidden_units, idim, weight1_int8, weight1_scale, stream);
  //   if (status != 0) {
  //     LOG(ERROR) << "Quantize weight1 error!" << endl;
  //     return status;
  //   }
  //   status = QuantizeWeight(w2_weight_ptr, num_expert, idim, hidden_units, weight2_int8, weight2_scale, stream);
  //   if (status != 0) {
  //     LOG(ERROR) << "Quantize weight2 error!" << endl;
  //     return status;
  //   }
  // }
  // cudaStreamSynchronize(stream);
  // print_data(weight1_int8, 10, 1, "weight1_int8");
  // print_data(w2_weight_int32_ptr, 5, "w2_weight_int32_ptr");

  // step 3: mapping and quantize input
  status = QuantizedScatterMappingCopy(input, mapping, S, idim, Layer1_In_buffer_int8, Layer1_In_scale, stream);
  if (status != 0) {
    LOG(ERROR) << "QuantizedScatterMappingCopy error!" << endl;
    return status;
  }
  // print_data(input,35,"input");
  // print_data(Layer1_In_buffer_int8,5,7,"int8");
  // print_data(Layer1_In_scale_dtype,7,"scale");

  // step 4: MOE caculation
  int* h_acc_his = v_acc_his.data();
  cudaMemcpyAsync(h_acc_his, acc_histogram, sizeof(int) * (num_expert + 1), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;

  for (int i = 0; i < num_expert; i++) {
    auto cur_stream = csm_ptr->Stream(i);
    auto handle = csm_ptr->CublasHandle(i);
    int m = h_acc_his[i + 1] - h_acc_his[i];
    if (m == 0) continue;

    // step 4.0: Prepare for workspace
    auto w_offset = i * idim * hidden_units;
    auto cur_weight1_int8_ptr = weight1_int8 + w_offset;
    // auto cur_w1_weight_scale_ptr = w1_weight_scale_ptr + i * hidden_units;
    // auto cur_w1_bias_ptr = w1_bias_ptr + i * hidden_units;
    auto cur_weight2_int8_ptr = weight2_int8 + w_offset;
    // auto cur_w2_weight_scale_ptr = w2_weight_scale_ptr + i * idim;
    // auto cur_w2_bias_ptr = w2_bias_ptr + i * idim;
    auto Layer1_In_buffer_int8_ptr = Layer1_In_buffer_int8 + h_acc_his[i] * idim;
    auto Layer1_Out_buffer_int32_ptr = Layer1_Out_buffer_int32 + h_acc_his[i] * hidden_units;

    // print_data(Layer1_Out_buffer_int32, 28 ,"Layer1_Out_buffer_int32");
    // print_data(Layer1_Out_buffer_int32_ptr, 10 ,"Layer1_Out_buffer_int32——ptr");
    // print_data(cur_weight1_int8_ptr, 5,4, "cur_weight1_int8_ptr");
    // print_data(Layer1_In_buffer_int8_ptr, 5,4, "Layer1_In_buffer_int8_ptr");
    // print_data(cur_weight2_int8_ptr, 5,4, "cur_weight2_int8_ptr");
    // print_data(w1_weight_scale_ptr, 12, "w1_weight_scale_ptr");
    // print_data(cur_w1_weight_scale_ptr, 4, "cur_w1_weight_scale_ptr");
    // print_data(w1_bias_ptr, 12, "w1_bias_ptr");
    // print_data(cur_w1_bias_ptr, 4, "cur_w1_bias_ptr");

    // print_data(input, 5, "input");
    // print_data(cur_weight2_int8_ptr, 4, "cur_weight2_int8_ptr");
    // print_data(w2_weight_scale_ptr, 15, "w2_weight_scale_ptr");
    // print_data(cur_w2_weight_scale_ptr, 5, "cur_w2_weight_scale_ptr");
    // print_data(w2_bias_ptr, 15, "w2_bias_ptr");
    // print_data(cur_w2_bias_ptr, 5, "cur_w2_bias_ptr");

    // cout << "====================expert:" << i << endl;

    // step 4.1: compute L1 out int32
    CUBLAS_CHECK(cublasGemm(handle, transa, transb, m, hidden_units, idim, 1, Layer1_In_buffer_int8_ptr,
                            cur_weight1_int8_ptr, 0, Layer1_Out_buffer_int32_ptr));

    // print_data(Layer1_Out_buffer_int32_ptr, 10, 2, "Layer1_Out_buffer_int32_ptr");
    // cout << "m = " << m << endl;
    // cout << "hidden_units = " << hidden_units << endl;
    // cout << "idim = " << idim << endl;

    // step 4.2: Dequantize + BiasSilu + Quantize
    auto Layer1_In_scale_ptr_expert = Layer1_In_scale + h_acc_his[i];
    const T* w1_weight_scale_ptr_expert = weight1_scale + i * hidden_units;
    auto w1_bias_ptr_expert = w1_bias_ptr + i * hidden_units;
    auto Layer2_In_buffer_int8_expert = Layer2_In_buffer_int8 + h_acc_his[i] * hidden_units;
    auto Layer2_In_scale_expert = Layer2_In_scale + h_acc_his[i];

    status = DequantizedBiasSiluAndQuantize(Layer1_Out_buffer_int32_ptr, Layer1_In_scale_ptr_expert,
                                            w1_weight_scale_ptr_expert, w1_bias_ptr_expert, m, hidden_units,
                                            Layer2_In_buffer_int8_expert, Layer2_In_scale_expert, cur_stream);
    if (status != 0) {
      LOG(ERROR) << "DequantizedBiasSilu error!" << endl;
      return status;
    }

    // if (i < num_expert) {
    //   print_data(cur_w1_weight_scale_ptr, hidden_units, "cur_w1_weight_scale_ptr");
    //   print_data(Layer1_In_scale, 7, "Layer1_In_scale");
    //   print_data(w1_bias_ptr_expert, hidden_units, "w1_bias");
      // print_data(Layer1_Out_buffer_int32_ptr, hidden_units,m, "Layer1_Out_buffer_int32_ptr");
    //   print_data(Layer1_Out_buffer_int32_ptr, Layer1_In_scale_ptr_expert, w1_weight_scale_ptr_expert, hidden_units,m,
    //   "Layer1_Out_buffer_float_ptr");
      // print_data(Layer1_In_buffer_int8_ptr, idim,m, "Layer1_In_buffer_int8_ptr");
    //   // print_data(cur_weight1_int8_ptr, idim,hidden_units, "cur_weight1_int8_ptr");
    //   // print_data(Layer1_Out_buffer_int32, hidden_units,7, "Layer1_Out_buffer_int32_ptr");
    // }
    // print_data(Layer2_In_buffer_int8, hidden_units, 7, "Layer2_In_buffer_int8");
    // print_data(Layer2_In_scale, 7, "Layer2_In_scale");
    // print_data(Layer2_In_buffer_int8_expert,  hidden_units, m, "Layer2_In_buffer_float");

    // step 4.3: compute L2 out int32
    auto Layer2_Out_buffer_int32_ptr = Layer2_Out_buffer_int32 + h_acc_his[i] * idim;
    const T* w2_weight_scale_ptr_expert = weight2_scale + i * idim;
    auto w2_bias_ptr_expert = w2_bias_ptr + i * idim;
    CUBLAS_CHECK(cublasGemm(handle, transa, transb, m, idim, hidden_units, 1, Layer2_In_buffer_int8_expert,
                            cur_weight2_int8_ptr, 0, Layer2_Out_buffer_int32_ptr));
    // print_data(Layer2_In_buffer_int8_expert, hidden_units, m, "Layer2_In_buffer_int8_expert");
    // print_data(cur_weight2_int8_ptr, hidden_units, idim, "cur_weight2_int8_ptr");
    // print_data(Layer2_Out_buffer_int32, idim, 7, "Layer2_Out_buffer_int32");
    // print_data(Layer2_Out_buffer_int32_ptr, 10, 1, "Layer2_Out_buffer_int32_ptr");
    // print_data(Layer2_In_scale_expert, m, "Layer2_In_scale_expert");
    // print_data(w2_weight_scale_ptr_expert, idim, "w2_weight_scale_ptr_expert");
    // print_data(Layer2_Out_buffer_int32_ptr, Layer2_In_scale_expert, w2_weight_scale_ptr_expert, idim,m,
    // "Layer2_Out_buffer_float_ptr");

    // step 4.4: Dequantize + Bias + ReverseMapping to Output
    status =
        DequantizedBiasGatherMapping(Layer2_Out_buffer_int32_ptr, Layer2_In_scale_expert, w2_weight_scale_ptr_expert,
                                     w2_bias_ptr_expert, m, idim, h_acc_his[i], inverse_mapping, output, cur_stream);
    if (status != 0) {
      LOG(ERROR) << "DequantizedBiasGatherMapping error!" << endl;
      return status;
    }
  }
  // print_data(output, 10, "output");
  csm_ptr->SyncAllStream();
  return 0;
}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name,
                                   const nvinfer1::Weights& w1_weight, const nvinfer1::Weights& w2_weight,
                                   const nvinfer1::Weights& w1_bias, const nvinfer1::Weights& w2_bias,
                                   const nvinfer1::DataType type, const int num_expert,
                                   const int idim, const int hidden_units, const int act_type, bool alloc_data)
    : layer_name_(name),
      data_type_(type),
      num_expert_(num_expert),
      idim_(idim),
      hidden_units_(hidden_units),
      act_type_(act_type),
      alloc_data_(alloc_data) {
  v_acc_his_.resize(num_expert_ + 1);
  cuda_stream_manager_.reset(new CudaStreamManager());
  cuda_stream_manager_->Init();

  if (data_type_ == nvinfer1::DataType::kFLOAT) {
    w1_weight_.reset(new FmoeWeightsBiasFloat32(w1_weight, w1_bias, num_expert_, hidden_units_, idim_));
    w2_weight_.reset(new FmoeWeightsBiasFloat32(w2_weight, w2_bias, num_expert_, idim_, hidden_units_));
  } else if (data_type_ == nvinfer1::DataType::kHALF) {
    w1_weight_.reset(new FmoeWeightsBiasFloat16(w1_weight, w1_bias, num_expert_, hidden_units_, idim_));
    w2_weight_.reset(new FmoeWeightsBiasFloat16(w2_weight, w2_bias, num_expert_, idim_, hidden_units_));
  } else if (data_type_ == nvinfer1::DataType::kINT8) {
    w1_weight_.reset(new FmoeWeightsBiasInt8(w1_weight, w1_bias, num_expert_, hidden_units_, idim_));
    w2_weight_.reset(new FmoeWeightsBiasInt8(w2_weight, w2_bias, num_expert_, idim_, hidden_units_));
  } else {
    LOG(ERROR) << "invalid type!!!" << endl;
  }
}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name, std::shared_ptr<FmoeWeightsBias> w1_weight,
                                   std::shared_ptr<FmoeWeightsBias> w2_weight,
                                   const nvinfer1::DataType type, const int num_expert,
                                   const int idim, const int hidden_units, const int act_type, bool alloc_data,
                                   std::shared_ptr<CudaStreamManager> cuda_stream_manager)
    : layer_name_(name),
      data_type_(type),
      num_expert_(num_expert),
      idim_(idim),
      hidden_units_(hidden_units),
      act_type_(act_type),
      alloc_data_(alloc_data),
      cuda_stream_manager_(cuda_stream_manager) {
  v_acc_his_.resize(num_expert_ + 1);

  w1_weight_.reset(w1_weight.get());
  w2_weight_.reset(w2_weight.get());

  if (alloc_data_) {
    w1_weight_->allocDeviceData();
    w2_weight_->allocDeviceData();
    w1_weight_->copyToDevice();
    w2_weight_->copyToDevice();
  }
}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name, const void* data, size_t length) : layer_name_(name) {
  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &num_expert_);
  deserialize_value(&data, &length, &idim_);
  deserialize_value(&data, &length, &hidden_units_);
  deserialize_value(&data, &length, &act_type_);
  deserialize_value(&data, &length, &alloc_data_);

  const char* d = static_cast<const char*>(data);

  if (data_type_ == DataType::kFLOAT) {
    w1_weight_.reset(new FmoeWeightsBiasFloat32(d, num_expert_, hidden_units_, idim_));  // use shared ptr to copy
    w2_weight_.reset(new FmoeWeightsBiasFloat32(d, num_expert_, idim_, hidden_units_));
  } else if (data_type_ == DataType::kHALF) {
    w1_weight_.reset(new FmoeWeightsBiasFloat16(d, num_expert_, hidden_units_, idim_));
    w2_weight_.reset(new FmoeWeightsBiasFloat16(d, num_expert_, idim_, hidden_units_));
  } else {
    w1_weight_.reset(new FmoeWeightsBiasInt8(d, num_expert_, hidden_units_, idim_));
    w2_weight_.reset(new FmoeWeightsBiasInt8(d, num_expert_, idim_, hidden_units_));
  }

  w1_weight_->allocDeviceData();
  w2_weight_->allocDeviceData();
  w1_weight_->copyToDevice();
  w2_weight_->copyToDevice();

  const void* tmp_ptr = static_cast<void*>(const_cast<char*>(d));
  int tmp = 0;
  deserialize_value(&tmp_ptr, &length, &tmp);
  deserialize_value(&tmp_ptr, &length, &tmp);
  deserialize_value(&tmp_ptr, &length, &tmp);

  alloc_data_ = true;
  v_acc_his_.resize(num_expert_ + 1);
  cuda_stream_manager_.reset(new CudaStreamManager());
  cuda_stream_manager_->Init();
}

//  FMoEExpertPlugin::~FMoEExpertPlugin() {
//   // cout<<first_enqueue_<<endl;
//   if (!first_enqueue_) {
//     cout<<"freeing!!!-----------------------"<<endl;
//     // print_data(weight1_int8_,10,1,"weight1_int8_");

//     // CUDA_CHECK(cudaFree(weight1_int8_));
//     // CUDA_CHECK(cudaFree(weight2_int8_));
//     // CUDA_CHECK(cudaFree(weight1_scale_));
//     // CUDA_CHECK(cudaFree(weight2_scale_));
//   }
//  }

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FMoEExpertPlugin::clone() const TRTNOEXCEPT {
  IPluginV2DynamicExt* ret;
  // if (data_type_ == DataType::kFLOAT || data_type_ == DataType::kHALF) {
    ret = new FMoEExpertPlugin(layer_name_, w1_weight_, w2_weight_, data_type_, num_expert_,
                               idim_, hidden_units_, act_type_, alloc_data_, cuda_stream_manager_);
    ret->setPluginNamespace(namespace_.c_str());

  return ret;
}

DimsExprs FMoEExpertPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                                IExprBuilder& exprBuilder) TRTNOEXCEPT {
  assert(nbInputs == 2);
  return inputs[0];
}

bool FMoEExpertPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                                 int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);
  assert(pos >= 0 && pos < nbInputs + nbOutputs);

  // input, gate_idx, w1_weight, w1_bias, w2_weight, w2_bias
  const PluginTensorDesc& in_out = inOut[pos];

  if (pos == 1) return (in_out.type == DataType::kINT32) && (in_out.format == TensorFormat::kLINEAR);

  if (data_type_ != DataType::kINT8)
    return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);
  else
    return (in_out.type == DataType::kFLOAT || in_out.type == DataType::kHALF) &&
           (in_out.format == TensorFormat::kLINEAR);

  return false;
}

void FMoEExpertPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
                                       const DynamicPluginTensorDesc* outputs, int nbOutputs) TRTNOEXCEPT {
  // Validate input arguments
  assert(nbInputs == 2);
  assert(nbOutputs == 1);
  // assert(data_type_ == inputs[0].desc.type);
  assert((data_type_ != DataType::kINT8 && data_type_ == inputs[0].desc.type) || data_type_ == DataType::kINT8);
}

size_t FMoEExpertPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
                                          int nbOutputs) const TRTNOEXCEPT {
  const size_t int8_size = getElementSize(DataType::kINT8);
  const size_t int32_size = getElementSize(DataType::kINT32);
  const size_t word_size = getElementSize(inputs[0].type);

  const int input_volume = volume(inputs[0].dims);
  const int S = input_volume / idim_;

  auto mapping_size = alignTo<int>(S, kAlignment);
  auto inverse_mapping_size = alignTo<int>(S, kAlignment);
  auto his_size = alignTo<int>(num_expert_ + 1, kAlignment);
  // auto weight1_int8_size = alignTo<int>(weight1_volume, kAlignment);
  // auto weight2_int8_size = alignTo<int>(weight2_volume, kAlignment);
  // auto weight1_scale_size = alignTo<int>(weight1_volume / idim_, kAlignment);
  // auto weight2_scale_size = alignTo<int>(weight2_volume / hidden_units_, kAlignment);

  auto Layer1_In_buffer_int8_size = alignTo<int>(input_volume, kAlignment);         // Input reorder & quantize
  auto Layer1_In_scale_dtype_size = alignTo<int>(S, kAlignment);                    // Layer1 input scale
  auto Layer1_Out_buffer_int32_size = alignTo<int>(S * hidden_units_, kAlignment);  // MM out1

  auto Layer2_In_buffer_int8_size = alignTo<int>(S * hidden_units_, kAlignment);  // Out1 scaling and silu & quantize
  auto Layer2_In_scale_dtype_size = alignTo<int>(S, kAlignment);                  // Layer2 input scale
  auto Layer2_Out_buffer_int32_size = alignTo<int>(input_volume, kAlignment);     // MM out2

  // printf("mark:%d-------------------\n",input_volume);
  // printf("mark:%d-------------------\n",idim_);
  // printf("mark:%d-------------------\n",weight1_volume);
  // printf("mark:%d-------------------\n",weight2_volume);
  const int workspace_size =
      (mapping_size + inverse_mapping_size + his_size) * sizeof(int)           // mapping_size & acc_histogram_size
      + (Layer1_In_buffer_int8_size + Layer2_In_buffer_int8_size) * int8_size  // int8 buffer size
      + (Layer1_In_scale_dtype_size + Layer2_In_scale_dtype_size) * word_size  // scale size
      + (Layer1_Out_buffer_int32_size + Layer2_Out_buffer_int32_size) * int32_size;  // mm out int32 size

  // printf("mark:%d-------------------\n",workspace_size);
  return workspace_size;
}

int FMoEExpertPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                              const void* const* inputs, void* const* outputs, void* workspace,
                              cudaStream_t stream) TRTNOEXCEPT {
  if (!alloc_data_) return 0;
  const int input_volume = volume(inputDesc[0].dims);
  const int weight1_volume = num_expert_ * hidden_units_ * idim_;
  const int weight2_volume = num_expert_ * hidden_units_ * idim_;
  const int S = input_volume / idim_;

  // float* p = new float[w1_weight_.count];
  // cudaMemcpy(p, static_cast<float*>(w1_weight_dev_.get()), sizeof(float) * w1_weight_.count, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 20; i++) {
  //   printf("%f\n", p[i]);
  // }
  // free(p);

  int status = -1;

  if (data_type_ == DataType::kFLOAT) {
    // Our plugin outputs only one tensor
    const float* input = static_cast<const float*>(inputs[0]);
    const int* gate_idx = static_cast<const int*>(inputs[1]);
    // print_data(gate_idx, 20, "-----------this is gate_idx]:\n");

    // const float* w1_weight_ptr = static_cast<const float*>(inputs[2]);

    // const int* w1_weight_int32_ptr = static_cast<const int*>(inputs[3]);
    // const float* w1_weight_scale_ptr = static_cast<const float*>(inputs[4]);
    // const float* w1_bias_ptr = static_cast<const float*>(inputs[2]);

    // print_data(w1_weight_ptr, 20, "-----------this is weight");
    // print_data(w1_weight_int32_ptr, 20, "-----------this is weight_int32");
    // print_data(w1_weight_scale_ptr, 5, "-----------this is weight_scale");

    // const float* w2_weight_ptr = static_cast<const float*>(inputs[4]);

    // const int* w2_weight_int32_ptr = static_cast<const int*>(inputs[7]);
    // const float* w2_weight_scale_ptr = static_cast<const float*>(inputs[8]);
    // const float* w2_bias_ptr = static_cast<const float*>(inputs[3]);

    const float* w1_weight_ptr = static_cast<const float*>(w1_weight_->d_values.get());
    const float* w2_weight_ptr = static_cast<const float*>(w2_weight_->d_values.get());
    const float* w1_bias_ptr = static_cast<const float*>(w1_weight_->d_bias.get());
    const float* w2_bias_ptr = static_cast<const float*>(w2_weight_->d_bias.get());

    float* output = static_cast<float*>(outputs[0]);
    status = ComputeFmoeExpert(input, gate_idx, input_volume, S, num_expert_, idim_, hidden_units_, w1_weight_ptr,
                                 w1_bias_ptr, w2_weight_ptr, w2_bias_ptr, v_acc_his_, workspace, output, stream,
                                 cuda_stream_manager_);

  } else if (data_type_ == DataType::kHALF) {
    // printf("--------------------------------------");
    // printf(typeid(inputs[0]).name());

    const half* input = static_cast<const half*>(inputs[0]);
    const int* gate_idx = static_cast<const int*>(inputs[1]);
    // const half* w1_weight_ptr = static_cast<const half*>(inputs[2]);
    // const half* w1_bias_ptr = static_cast<const half*>(inputs[2]);

    // const half* w2_weight_ptr = static_cast<const half*>(inputs[4]);
    // const half* w2_bias_ptr = static_cast<const half*>(inputs[3]);

    const half* w1_weight_ptr = static_cast<const half*>(w1_weight_->d_values.get());
    const half* w2_weight_ptr = static_cast<const half*>(w2_weight_->d_values.get());
    const half* w1_bias_ptr = static_cast<const half*>(w1_weight_->d_bias.get());
    const half* w2_bias_ptr = static_cast<const half*>(w2_weight_->d_bias.get());

    half* output = static_cast<half*>(outputs[0]);

    status = ComputeFmoeExpert(input, gate_idx, input_volume, S, num_expert_, idim_, hidden_units_, w1_weight_ptr,
                                 w1_bias_ptr, w2_weight_ptr, w2_bias_ptr, v_acc_his_, workspace, output, stream,
                                 cuda_stream_manager_);

  } else if (data_type_ == DataType::kINT8) {
    // Our plugin outputs only one tensor

    const int* gate_idx = static_cast<const int*>(inputs[1]);
    // print_data(gate_idx, 20, "-----------this is gate_idx]:\n");

    // const float* w1_weight_ptr = static_cast<const float*>(inputs[2]);

    // const int* w1_weight_int32_ptr = static_cast<const int*>(inputs[3]);
    // const float* w1_weight_scale_ptr = static_cast<const float*>(inputs[4]);

    // print_data(w1_weight_ptr, 20, "-----------this is weight");
    // print_data(w1_weight_int32_ptr, 20, "-----------this is weight_int32");
    // print_data(w1_weight_scale_ptr, 5, "-----------this is weight_scale");

    // const float* w2_weight_ptr = static_cast<const float*>(inputs[4]);

    // const int* w2_weight_int32_ptr = static_cast<const int*>(inputs[7]);
    // const float* w2_weight_scale_ptr = static_cast<const float*>(inputs[8]);

    const char* w1_weight_int8 = static_cast<const char*>(w1_weight_->d_values.get());
    const char* w2_weight_int8 = static_cast<const char*>(w2_weight_->d_values.get());


    // status = ComputeFmoeExpert(input, gate_idx, input_volume, S, num_expert_, idim_, hidden_units_, w1_weight_ptr,
    //                              w1_bias_ptr, w2_weight_ptr, w2_bias_ptr, v_acc_his_, workspace, output, stream,
    //                              cuda_stream_manager_);
    // cout<<first_enqueue_<<endl;
    // if (first_enqueue_) {
    //   CUDA_CHECK(cudaMalloc((char**)&weight1_int8_, weight1_volume * sizeof(char)));
    //   CUDA_CHECK(cudaMalloc((char**)&weight2_int8_, weight2_volume * sizeof(char)));
    //   CUDA_CHECK(cudaMalloc((float**)&weight1_scale_, num_expert_ * hidden_units_ * sizeof(float)));
    //   CUDA_CHECK(cudaMalloc((float**)&weight2_scale_, num_expert_ * idim_ * sizeof(float)));
    //   status = QuantizeWeight(w1_weight_ptr, num_expert_, hidden_units_, idim_, weight1_int8_,
    //                           weight1_scale_, stream);
    //   if (status != 0) {
    //     LOG(ERROR) << "Quantize weight1 error!" << endl;
    //     return status;
    //   }
    //   status = QuantizeWeight(w2_weight_ptr, num_expert_, idim_, hidden_units_, weight2_int8_,
    //                           weight2_scale_, stream);
    //   if (status != 0) {
    //     LOG(ERROR) << "Quantize weight2 error!" << endl;
    //     return status;
    //   }
    // //   print_data(w1_weight_ptr,4,2,"w1_weight_ptr first");
    // //   print_data(weight1_int8_,4,2,"weight1_int8_ first");
    // //   print_data(weight1_scale_,8,1,"weight1_scale_ first");
    // //   print_data(weight2_int8_,10,1,"weight2_int8_ first");
    // //   print_data(weight2_scale_,10,1,"weight2_scale_ first");
    // }
    // status = ComputeFmoeExpert(input, gate_idx, input_volume, S, num_expert_, idim_, hidden_units_, w1_weight_ptr,
    //                              w1_bias_ptr, w2_weight_ptr, w2_bias_ptr, v_acc_his_, workspace, output, stream,
    //                              cuda_stream_manager_);
    if (inputDesc[0].type == DataType::kFLOAT) {
      const float* input = static_cast<const float*>(inputs[0]);
      const float* weight1_scale = static_cast<const float*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                             w1_weight_.get())->d_fp32_scale.get());
      const float* weight2_scale = static_cast<const float*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                             w2_weight_.get())->d_fp32_scale.get());
      const float* w1_bias_ptr = static_cast<const float*>(w1_weight_->d_bias.get());
      const float* w2_bias_ptr = static_cast<const float*>(w2_weight_->d_bias.get());
      float* output = static_cast<float*>(outputs[0]);
      status = ComputeFmoeExpertInt8(input, gate_idx, input_volume, weight1_volume, weight2_volume, S, num_expert_,
                                        idim_, hidden_units_, w1_bias_ptr, w1_weight_int8, weight1_scale,
                                        w2_bias_ptr, w2_weight_int8, weight2_scale, v_acc_his_,
                                        workspace, output, stream, cuda_stream_manager_);
    } else {
      const half* input = static_cast<const half*>(inputs[0]);
      const half* weight1_scale = static_cast<const half*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                           w1_weight_.get())->d_fp16_scale.get());
      const half* weight2_scale = static_cast<const half*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                           w2_weight_.get())->d_fp16_scale.get());
      const half* w1_bias_ptr = static_cast<const half*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                         w1_weight_.get())->d_fp16_bias.get());
      const half* w2_bias_ptr = static_cast<const half*>(dynamic_cast<FmoeWeightsBiasInt8*>(
                                                         w2_weight_.get())->d_fp16_bias.get());
      half* output = static_cast<half*>(outputs[0]);
      status = ComputeFmoeExpertInt8(input, gate_idx, input_volume, weight1_volume, weight2_volume, S, num_expert_,
                                        idim_, hidden_units_, w1_bias_ptr, w1_weight_int8, weight1_scale,
                                        w2_bias_ptr, w2_weight_int8, weight2_scale, v_acc_his_,
                                        workspace, output, stream, cuda_stream_manager_);
    }

    // print_data(weight1_int8_,10,1,"weight1_int8_ ");
    // first_enqueue_ = false;
  }

  return status;
}

// IPluginV2Ext Methods
DataType FMoEExpertPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRTNOEXCEPT {
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* FMoEExpertPlugin::getPluginType() const TRTNOEXCEPT { return FMOE_EXPERT_NAME; }

const char* FMoEExpertPlugin::getPluginVersion() const TRTNOEXCEPT { return FMOE_EXPERT_VERSION; }

int FMoEExpertPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

int FMoEExpertPlugin::initialize() TRTNOEXCEPT {
  return 0;
}

void FMoEExpertPlugin::terminate() TRTNOEXCEPT {}

size_t FMoEExpertPlugin::getSerializationSize() const TRTNOEXCEPT {
  int scale_size = num_expert_ * hidden_units_ + num_expert_ * idim_;

  return sizeof(data_type_) + sizeof(num_expert_) + sizeof(idim_) + sizeof(hidden_units_) + sizeof(act_type_) +
         sizeof(alloc_data_) + 2 * num_expert_ * hidden_units_ * idim_ * getElementSize(data_type_) +
         ((data_type_ == DataType::kINT8) ? scale_size * (sizeof(float) + sizeof(half)) : 0) +
         // bias_size
         scale_size * ((data_type_ == DataType::kINT8) ? (sizeof(float) + sizeof(half)) : getElementSize(data_type_)) +
         3 * sizeof(int);
}

void FMoEExpertPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_expert_);
  serialize_value(&buffer, idim_);
  serialize_value(&buffer, hidden_units_);
  serialize_value(&buffer, act_type_);
  serialize_value(&buffer, alloc_data_);

  char* d = static_cast<char*>(buffer);

  w1_weight_->serialize(d);
  w2_weight_->serialize(d);

  int tmp = 0;
  void* tmp_ptr = static_cast<void*>(d);
  serialize_value(&tmp_ptr, tmp);
  serialize_value(&tmp_ptr, tmp);
  serialize_value(&tmp_ptr, tmp);
}

void FMoEExpertPlugin::destroy() TRTNOEXCEPT {
  // delete this;
}

void FMoEExpertPlugin::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* FMoEExpertPlugin::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

///////////////////////

FMoEExpertPluginCreator::FMoEExpertPluginCreator() {
  FC_.nbFields = plugin_attributes_.size();
  FC_.fields = plugin_attributes_.data();
}

const char* FMoEExpertPluginCreator::getPluginName() const TRTNOEXCEPT { return FMOE_EXPERT_NAME; }

const char* FMoEExpertPluginCreator::getPluginVersion() const TRTNOEXCEPT { return FMOE_EXPERT_VERSION; }

const PluginFieldCollection* FMoEExpertPluginCreator::getFieldNames() TRTNOEXCEPT { return &FC_; }

IPluginV2* FMoEExpertPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT {
  LOG(INFO) << "Creating FMoEExpertPlugin...\n";

  int type_id = -1;
  nvinfer1::Weights w1_weight;
  nvinfer1::Weights w2_weight;
  nvinfer1::Weights w1_bias;
  nvinfer1::Weights w2_bias;
  int num_expert = 0, idim = 0, hidden_units = 0, act_type = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("w1_weight") == 0) {
      w1_weight = *static_cast<Weights*>(const_cast<void*>(fc->fields[i].data));
      LOG(INFO) << "Building w1_weight: ";
    }
    if (field_name.compare("w2_weight") == 0) {
      w2_weight = *static_cast<Weights*>(const_cast<void*>(fc->fields[i].data));
      LOG(INFO) << "Building w2_weight";
    }
    if (field_name.compare("w1_bias") == 0) {
      w1_bias = *static_cast<Weights*>(const_cast<void*>(fc->fields[i].data));
      LOG(INFO) << "Building w1_bias";
    }
    if (field_name.compare("w2_bias") == 0) {
      w2_bias = *static_cast<Weights*>(const_cast<void*>(fc->fields[i].data));
      LOG(INFO) << "Building w2_bias";
    }
    if (field_name.compare("data_type") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building type_id: " << type_id;
    }
    if (field_name.compare("num_expert") == 0) {
      num_expert = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building num_expert: " << num_expert;
    }
    if (field_name.compare("idim") == 0) {
      idim = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building idim " << idim;
    }
    if (field_name.compare("hidden_units") == 0) {
      hidden_units = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building hidden_units " << hidden_units;
    }
    if (field_name.compare("act_type") == 0) {
      act_type = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building act_type " << act_type;
    }
  }

  if (type_id < 0 || type_id > 2) {
    LOG(ERROR) << "fmoe: invalid type_id " << type_id;
    return nullptr;
  }

  DataType type = static_cast<DataType>(type_id);

  LOG(INFO) << "Building the Plugin...\n";

  bool alloc_data = false;
  FMoEExpertPlugin* p = new FMoEExpertPlugin(name, w1_weight, w2_weight, w1_bias, w2_bias,
                                             type, num_expert, idim, hidden_units, act_type, alloc_data);
  return p;
}

IPluginV2* FMoEExpertPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                      size_t serialLength) TRTNOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FMoEExpertPlugin::destroy()
  return new FMoEExpertPlugin(name, serialData, serialLength);
}

void FMoEExpertPluginCreator::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* FMoEExpertPluginCreator::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

}  // namespace plugin
}  // namespace nvinfer1