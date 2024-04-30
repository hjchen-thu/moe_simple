#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include "common.h"
#include "cuda_stream_manager.h"

namespace nvinfer1 {
namespace plugin {

class FmoeWeightsBias {
 public:
  virtual void copyToDevice() = 0;
  virtual void serialize(char*& buffer) = 0;
  virtual void allocDeviceData() = 0;  // alloc device data

  void* values;
  void* bias;
  cuda_unique_ptr<void> d_values;
  cuda_unique_ptr<void> d_bias;

  int actual_size;
  int align_size;
  int actual_bias_size;
  int align_bias_size;
};

class FmoeWeightsBiasFloat32 : public FmoeWeightsBias {
 public:
  FmoeWeightsBiasFloat32(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias,
                     const int num_expert,  // convert nv weights to our type
                     const int hidden_units, const int idim);
  FmoeWeightsBiasFloat32(const char*& buffer, const int num_expert, const int hidden_units,
                     const int idim);  // actually every second construct function is used to deserialize
  void copyToDevice() override;
  void serialize(char*& buffer) override;
  void allocDeviceData() override;
//  protected:
  ~FmoeWeightsBiasFloat32() {
    delete[] static_cast<float*>(this->values);
    delete[] static_cast<float*>(this->bias);
  }
};

class FmoeWeightsBiasFloat16 : public FmoeWeightsBias {
 public:
  FmoeWeightsBiasFloat16(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias, const int num_expert,
                     const int hidden_units, const int idim);
  FmoeWeightsBiasFloat16(const char*& buffer, const int num_expert, const int hidden_units,
                     const int idim);
  void copyToDevice() override;
  void serialize(char*& buffer) override;
  void allocDeviceData() override;
//  protected:
  ~FmoeWeightsBiasFloat16() {
    delete[] static_cast<half*>(this->values);
    delete[] static_cast<half*>(this->bias);
  }
};

class FmoeWeightsBiasInt8 : public FmoeWeightsBias {  // each object stores both fp32 and fp16 scales
 public:
  FmoeWeightsBiasInt8(const nvinfer1::Weights& weights, const nvinfer1::Weights& bias, const int num_expert,
                     const int hidden_units, const int idim);
  FmoeWeightsBiasInt8(const char*& buffer, const int num_expert, const int hidden_units,
                     const int idim);
  void copyToDevice() override;
  void serialize(char*& buffer) override;
  void allocDeviceData() override;

  void trans2CharRowFp32(const float* src, char* char_data, float* scale, const int dim);
  void trans2CharRowFp16(const float* src, char* char_data, half* scale, const int dim);

 public:
  float* fp32_scale;
  half* fp16_scale;
  cuda_unique_ptr<void> d_fp32_scale;
  cuda_unique_ptr<void> d_fp16_scale;

  half* fp16_bias;
  cuda_unique_ptr<void> d_fp16_bias;
//  protected:
  ~FmoeWeightsBiasInt8() {
    delete[] static_cast<char*>(this->values);
    delete[] static_cast<float*>(this->bias);
    delete[] this->fp16_bias;
    delete[] this->fp32_scale;
    delete[] this->fp16_scale;
    delete[] this->fp16_bias;
  }
};

}  // namespace plugin
}  // namespace nvinfer1