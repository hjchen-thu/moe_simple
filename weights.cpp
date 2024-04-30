#include <cuda_fp16.h>

#include "fmoe_weights.h"

namespace nvinfer1 {
namespace plugin {

FmoeWeightsBiasFloat32::FmoeWeightsBiasFloat32(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias,
                                               const int num_expert, const int hidden_units, const int idim) {
  assert(weights.type == nvinfer1::DataType::kFLOAT);
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  auto dest_buf = new float[this->align_size];
  this->values = dest_buf;
  std::copy_n(static_cast<const float*>(weights.values), this->actual_size, dest_buf);

  auto bias_buf = new float[this->align_bias_size];
  this->bias = bias_buf;
  std::copy_n(static_cast<const float*>(bias.values), this->actual_bias_size, bias_buf);
  // for (int i = 0; i < 100; i++) printf("%f\n", static_cast<const float*>(this->values)[i]);
}

FmoeWeightsBiasFloat32::FmoeWeightsBiasFloat32(const char*& buffer, const int num_expert, const int hidden_units,
                                               const int idim) {
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  const auto values_size = this->align_size * sizeof(float);
  auto destBuf = new char[values_size];
  this->values = destBuf;

  std::copy_n(buffer, this->actual_size * sizeof(float), destBuf);
  buffer += values_size;

  const auto bias_size = this->align_bias_size * sizeof(float);
  auto bias_buf = new char[bias_size];
  this->bias = bias_buf;

  std::copy_n(buffer, this->actual_bias_size * sizeof(float), bias_buf);
  buffer += bias_size;
}

void FmoeWeightsBiasFloat32::allocDeviceData() {
  void* cudaMem{nullptr};
  // weighs elements:dims.d[0] * dims.d[1] * dims.d[2], bias elements:dims.d[0] * dims.d[1]
  int values_size = this->align_size * sizeof(float);
  CUASSERT(cudaMalloc(&cudaMem, values_size));
  this->d_values.reset(cudaMem);

  void* bias_device_mem{nullptr};
  int bias_size = this->align_bias_size * sizeof(float);
  CUASSERT(cudaMalloc(&bias_device_mem, bias_size));
  this->d_bias.reset(bias_device_mem);
}

void FmoeWeightsBiasFloat32::copyToDevice() {
  int values_size = this->actual_size * sizeof(float);
  CUASSERT(cudaMemcpy(this->d_values.get(), this->values, values_size, cudaMemcpyHostToDevice));

  int bias_size = this->actual_bias_size * sizeof(float);
  CUASSERT(cudaMemcpy(this->d_bias.get(), this->bias, bias_size, cudaMemcpyHostToDevice));
}

void FmoeWeightsBiasFloat32::serialize(char*& buffer) {
  int values_size = this->actual_size * sizeof(float);
  std::copy_n(static_cast<char*>(const_cast<void*>(this->values)), values_size, buffer);
  buffer += values_size;

  int bias_size = this->actual_bias_size * sizeof(float);
  std::copy_n(reinterpret_cast<char*>(this->bias), bias_size, buffer);
  buffer += bias_size;
}

FmoeWeightsBiasFloat16::FmoeWeightsBiasFloat16(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias,
                                       const int num_expert, const int hidden_units, const int idim) {
  assert(weights.type == nvinfer1::DataType::kFLOAT);
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  this->values = new half[this->align_size];

  const auto s = static_cast<const float*>(weights.values);
  auto d = static_cast<half*>(const_cast<void*>(this->values));

  for (auto it = 0; it < this->actual_size; it++) {  // convert to fp16
          d[it] = __float2half(s[it]);
  }

  this->bias = new half[this->align_bias_size];

  const auto s_bias = static_cast<const float*>(bias.values);
  auto d_bias = static_cast<half*>(this->bias);

  for (auto it = 0; it < this->actual_bias_size; it++) {
          d_bias[it] = __float2half(s_bias[it]);
  }
  // for (int i = 0; i < 100; i++) printf("%f\n", static_cast<const float*>(this->values)[i]);
}

FmoeWeightsBiasFloat16::FmoeWeightsBiasFloat16(const char*& buffer, const int num_expert, const int hidden_units,
                                       const int idim) {
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  const auto values_size = this->align_size * sizeof(half);
  auto destBuf = new char[values_size];
  this->values = destBuf;

  std::copy_n(buffer, this->actual_size * sizeof(half), destBuf);
  buffer += values_size;

  const auto bias_size = this->align_bias_size * sizeof(half);
  auto bias_buf = new char[bias_size];
  this->bias = reinterpret_cast<half*>(bias_buf);

  std::copy_n(buffer, this->actual_bias_size * sizeof(half), bias_buf);
  buffer += bias_size;
}

void FmoeWeightsBiasFloat16::allocDeviceData() {
  void* cudaMem{nullptr};
  int values_size = this->align_size * sizeof(half);
  CUASSERT(cudaMalloc(&cudaMem, values_size));
  this->d_values.reset(cudaMem);

  void* bias_device_mem{nullptr};
  int bias_size = this->align_bias_size * sizeof(half);
  CUASSERT(cudaMalloc(&bias_device_mem, bias_size));
  this->d_bias.reset(bias_device_mem);
}

void FmoeWeightsBiasFloat16::copyToDevice() {
  int values_size = this->actual_size * sizeof(half);
  CUASSERT(cudaMemcpy(this->d_values.get(), this->values, values_size, cudaMemcpyHostToDevice));

  int bias_size = this->actual_bias_size * sizeof(half);
  CUASSERT(cudaMemcpy(this->d_bias.get(), this->bias, bias_size, cudaMemcpyHostToDevice));
}

void FmoeWeightsBiasFloat16::serialize(char*& buffer) {
  int values_size = this->actual_size * sizeof(half);
  std::copy_n(static_cast<char*>(const_cast<void*>(this->values)), values_size, buffer);

  buffer += values_size;

  int bias_size = this->actual_bias_size  * sizeof(half);
  std::copy_n(reinterpret_cast<char*>(this->bias), bias_size, buffer);

  buffer += bias_size;
}

void FmoeWeightsBiasInt8::trans2CharRowFp32(const float* src, char* char_data, float* scale, const int dim) {
  float s_max = 0.0f;
  for (int i = 0; i < dim; i++) {
    s_max = std::max(s_max, fabsf(src[i]));
  }

  scale[0] = 255.0 / 2.0 / s_max;
  for (int i = 0; i < dim; i++) {
    float tmp = src[i] * scale[0];
    if (tmp <= -128) char_data[i] = -128;
    else if (tmp >= 127) char_data[i] = 127;
    else char_data[i] = (signed char)(tmp + (src[i] > 0 ? 1 : -1) * 0.5);
  }
}

void FmoeWeightsBiasInt8::trans2CharRowFp16(const float* src, char* char_data, half* scale, const int dim) {
  half s_max = 0.0;
  for (int i = 0; i < dim; i++) {
    s_max = std::max(s_max, (half)fabs(src[i]));
  }
  scale[0] = (half)(255.0 / 2.0 / s_max);
}

FmoeWeightsBiasInt8::FmoeWeightsBiasInt8(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias,
                                 const int num_expert, const int hidden_units, const int idim) {
  assert(weights.type == nvinfer1::DataType::kFLOAT);
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  this->values = new char[this->align_size];
  this->fp32_scale = new float[this->align_bias_size];
  this->fp16_scale = new half[this->align_bias_size];

  // gLogVerbose << "Float Weights(Host) => Half Array(Host)\n";
  float* fp32_src = static_cast<float*>(const_cast<void*>(weights.values));
  char* fp32_d = static_cast<char*>(const_cast<void*>(this->values));
  float* fp32_s = this->fp32_scale;
  for (auto it = 0; it < this->actual_size; it += idim) {
    trans2CharRowFp32(fp32_src, fp32_d, fp32_s, idim);
    fp32_src += idim;
    fp32_d += idim;
    fp32_s++;
  }
  float* fp16_src = static_cast<float*>(const_cast<void*>(weights.values));
  char* fp16_d = static_cast<char*>(const_cast<void*>(this->values));
  half* fp16_s = this->fp16_scale;

  for (auto it = 0; it < this->actual_size; it += idim) {
    trans2CharRowFp16(fp16_src, fp16_d, fp16_s, idim);
    fp16_src += idim;
    fp16_d += idim;
    fp16_s++;
  }

  auto fp32_bias_buf = new float[this->align_bias_size];
  this->bias = fp32_bias_buf;
  std::copy_n(static_cast<const float*>(bias.values), this->actual_bias_size, fp32_bias_buf);

  this->fp16_bias = new half[num_expert*hidden_units];

  const auto s_bias = static_cast<const float*>(bias.values);
  auto d_bias = static_cast<half*>(this->fp16_bias);

  for (auto it = 0; it < this->actual_bias_size; it++) {
          d_bias[it] = __float2half(s_bias[it]);
  }
  // for (int i = 0; i < 20; i++) printf("%f\n", this->scale[i]);
}

FmoeWeightsBiasInt8::FmoeWeightsBiasInt8(const char*& buffer, const int num_expert, const int hidden_units,
                                 const int idim) {
  this->actual_size = num_expert * hidden_units * idim;
  this->align_size = alignTo<int>(this->actual_size, kAlignment);

  this->actual_bias_size = num_expert * hidden_units;
  this->align_bias_size = alignTo<int>(this->actual_bias_size, kAlignment);

  int values_size = this->align_size;
  auto destBuf = new char[values_size];
  this->values = destBuf;

  std::copy_n(buffer, this->actual_size, destBuf);
  buffer += values_size;

  auto fp32_scale_buf = new float[this->align_bias_size];
  this->fp32_scale = fp32_scale_buf;
  std::copy_n(reinterpret_cast<float*>(const_cast<char*>(buffer)), this->actual_bias_size, fp32_scale_buf);
  buffer = buffer + this->align_bias_size * sizeof(float);

  auto fp16_scale_buf = new half[this->align_bias_size];
  this->fp16_scale = fp16_scale_buf;
  std::copy_n(reinterpret_cast<half*>(const_cast<char*>(buffer)), this->actual_bias_size, fp16_scale_buf);
  buffer = buffer + this->align_bias_size * sizeof(half);

  const auto bias_size = this->align_bias_size * sizeof(float);
  auto bias_buf = new char[bias_size];
  this->bias = bias_buf;

  std::copy_n(buffer, this->actual_bias_size * sizeof(float), bias_buf);

  buffer += bias_size;

  const auto fp16_bias_size = this->align_bias_size * sizeof(half);
  auto fp16_bias_buf = new char[fp16_bias_size];
  this->fp16_bias = reinterpret_cast<half*>(fp16_bias_buf);

  std::copy_n(buffer, this->actual_bias_size * sizeof(half), fp16_bias_buf);

  buffer += fp16_bias_size;
}

void FmoeWeightsBiasInt8::allocDeviceData() {
  int weights_elements = this->align_size;
  int bias_elements = this->align_bias_size;

  void* values_device_mem{nullptr};
  int values_size = weights_elements * sizeof(char);
  CUASSERT(cudaMalloc(&values_device_mem, values_size));
  this->d_values.reset(values_device_mem);

  void* fp32_scale_device_mem{nullptr};
  int fp32_scale_size = bias_elements * sizeof(float);
  CUASSERT(cudaMalloc(&fp32_scale_device_mem, fp32_scale_size));
  this->d_fp32_scale.reset(fp32_scale_device_mem);

  void* fp16_scale_device_mem{nullptr};
  int fp16_scale_size = bias_elements * sizeof(half);
  CUASSERT(cudaMalloc(&fp16_scale_device_mem, fp16_scale_size));
  this->d_fp16_scale.reset(fp16_scale_device_mem);

  void* bias_device_mem{nullptr};
  int bias_size = bias_elements * sizeof(float);
  CUASSERT(cudaMalloc(&bias_device_mem, bias_size));
  this->d_bias.reset(bias_device_mem);

  half* fp16_bias_device_mem{nullptr};
  int fp16_bias_size = bias_elements * sizeof(half);
  CUASSERT(cudaMalloc(&fp16_bias_device_mem, fp16_bias_size));
  this->d_fp16_bias.reset(fp16_bias_device_mem);
}

void FmoeWeightsBiasInt8::copyToDevice() {
  int weights_elements = this->actual_size;
  int bias_elements = this->actual_bias_size;

  int values_size = weights_elements * sizeof(char);
  CUASSERT(cudaMemcpy(this->d_values.get(), this->values, values_size, cudaMemcpyHostToDevice));

  int fp32_scale_size = bias_elements * sizeof(float);
  CUASSERT(cudaMemcpy(this->d_fp32_scale.get(), this->fp32_scale, fp32_scale_size, cudaMemcpyHostToDevice));

  int fp16_scale_size = bias_elements * sizeof(half);
  CUASSERT(cudaMemcpy(this->d_fp16_scale.get(), this->fp16_scale, fp16_scale_size, cudaMemcpyHostToDevice));

  int bias_size = bias_elements * sizeof(float);
  CUASSERT(cudaMemcpy(this->d_bias.get(), this->bias, bias_size, cudaMemcpyHostToDevice));

  int fp16_bias_size = bias_elements * sizeof(half);
  CUASSERT(cudaMemcpy(this->d_fp16_bias.get(), this->fp16_bias, fp16_bias_size, cudaMemcpyHostToDevice));
}

void FmoeWeightsBiasInt8::serialize(char*& buffer) {
  int weights_elements = this->actual_size;
  int bias_elements = this->actual_bias_size;

  int values_size = weights_elements * sizeof(char);
  std::copy_n(static_cast<char*>(const_cast<void*>(this->values)), values_size, buffer);

  buffer += values_size;

  int fp32_scale_size = bias_elements * sizeof(float);
  std::copy_n(reinterpret_cast<char*>(this->fp32_scale), fp32_scale_size, buffer);
  buffer += fp32_scale_size;

  int fp16_scale_size = bias_elements * sizeof(half);
  std::copy_n(reinterpret_cast<char*>(this->fp16_scale), fp16_scale_size, buffer);
  buffer += fp16_scale_size;

  int bias_size = bias_elements  * sizeof(float);
  std::copy_n(static_cast<char*>(const_cast<void*>(this->bias)), bias_size, buffer);
  buffer += bias_size;

  int fp16_bias_size = bias_elements  * sizeof(half);
  std::copy_n(reinterpret_cast<char*>(this->fp16_bias), fp16_bias_size, buffer);
  buffer += fp16_bias_size;
}

}  // namespace nvinfer1
}  // namespace plugin