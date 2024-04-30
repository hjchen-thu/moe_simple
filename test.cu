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

#include <iostream>

#include "cuda_profiler_api.h"

#include "common/test_common.h"
#include "common/test_timer.h"

#include "plugin/fmoe_expert_plugin/fmoe_expert_kernel.h"

using namespace nvinfer1;
using namespace plugin;
using namespace std;

int loop = 10;
// int loop = 1;

class FMoEPluginTest : public ::testing::Test {
 protected:
  FMoEPluginTest() {}
  virtual ~FMoEPluginTest() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

template <class T>
void torch_fmoe_forward(torch::Tensor input, torch::Tensor gate_idx, int num_expert, torch::Tensor w1_weight,
                        torch::Tensor w1_bias, torch::Tensor w2_weight, torch::Tensor w2_bias, torch::Tensor output) {
  auto seq_len = gate_idx.numel();
  auto dim = input.sizes()[1];

  auto mapping = torch::zeros_like(gate_idx);
  mapping = torch::_cast_Int(mapping);

  auto reorder_input = torch::zeros_like(input);

  // gate_idx to his
  // his[0] = 0
  int his[num_expert + 1] = {0};
  int* gate_idx_ptr = gate_idx.data_ptr<int>();
  int* mapping_ptr = mapping.data_ptr<int>();
  for (int i = 0; i < seq_len; i++) {
    auto idx = gate_idx_ptr[i] + 1;
    auto old = his[idx];
    his[idx]++;
    mapping_ptr[i] = old;
  }

  // acc his
  for (int i = 0; i < num_expert; i++) his[i + 1] += his[i];

  for (int i = 0; i < seq_len; i++) mapping_ptr[i] += his[gate_idx_ptr[i]];  // expert_offset(his) + no_in_each_expert(mapping_ptr)

  // for (int i = 0; i < seq_len; i++)
  // printf("%d ", mapping_ptr[i]);
  // printf("\n");

  // for (int i = 0; i < num_expert+1; i++)
  // printf("%d ", his[i]);
  // printf("\n");

  // scatter copy
  auto input_ptr = input.data_ptr<T>();
  auto reorder_input_ptr = reorder_input.data_ptr<T>();
  for (int i = 0; i < seq_len; i++) {
    auto a = input_ptr + i * dim;
    auto b = reorder_input_ptr + mapping_ptr[i] * dim;
    memcpy(b, a, dim * sizeof(T));
  }

  for (int i = 0; i < num_expert; i++) {
    int m = his[i + 1] - his[i];
    if (m == 0) continue;

    auto tmp = reorder_input.slice(0, his[i], his[i + 1]);
    // print_tensor(reorder_input, "reorder_input", false);
    // print_tensor(tmp, "tmp", false);
    // print_tensor(w1_weight[i], "w1_weight[i]", false);
    // print_tensor(w1_bias[i], "w1_bias[i]", false);
    auto w1 = w1_weight[i].transpose(1, 0);
    auto tmp2 = torch::matmul(tmp, w1);

    // if (i < num_expert) {
    //   cout << "\nexpert:" << i <<"\n";
    //   // cout <<"\ninput:\n"<< tmp;
    //   // cout <<"\nweight:\n"<< w1;
    //   cout <<"\nmul:\n"<< tmp2;
    // }

    // print_tensor(tmp2, "w1_weight", true);

    tmp2 = tmp2 + w1_bias[i];

    // cout <<"\nbias:\n"<< tmp2;
    tmp2 = torch::silu(tmp2);

    // print_tensor(tmp2, "tmp2", true);
    // cout <<"\nsilu:\n"<< tmp2;
    // print_tensor(tmp2, "silu", true);

    // print_tensor(tmp2, "tmp2", true);
    // print_tensor(w2_weight[i], "w2_weight[i]", false);
    // print_tensor(w2_bias[i], "w2_bias[i]", false);
    // cout << "\nexpert:" << i <<"\n";
    auto w2 = w2_weight[i].transpose(1, 0);
    auto tmp_output = torch::matmul(tmp2, w2);

    // print_tensor(tmp_output, "tmp_output", true);
    // cout <<"\nmul:\n"<< tmp_output;
    tmp_output += w2_bias[i];
    // cout <<"\nbias:\n"<< tmp_output;

    memcpy(tmp.data_ptr(), tmp_output.data_ptr(), sizeof(T) * tmp_output.numel());
    // print_tensor(tmp, "w2", true);
  }

  // scatter copy
  auto output_ptr = output.data_ptr<T>();
  for (int i = 0; i < seq_len; i++) {
    auto a = reorder_input_ptr + mapping_ptr[i] * dim;
    auto b = output_ptr + i * dim;
    memcpy(b, a, dim * sizeof(T));
  }
  // cout <<"\noutput:\n"<< output_ptr;
  // print_tensor(output, "output", true);
}

// void quantize_along_dim(const torch::Tensor weight_float, torch::Tensor* const weight_quant,
//                         torch::Tensor* const weight_scalse, const int dim = -1 , const int num_bits = 8)
// {
//   auto int_interval = pow(2, num_bits) - 1;

//   // auto

//   // return 8;
// }

void fmoe_test(string engine_file, nvinfer1::DataType data_type, nvinfer1::DataType input_type) {
  int num_expert = 16;
  int dim = 256;
  int hidden_units = 512;
  int act_type = 0;

  // int num_expert = 3;
  // int dim = 4;
  // int hidden_units = 8;
  // int act_type = 0;

  // trans weight
  auto w1_weight = torch::randn({num_expert, hidden_units, dim});
  auto w1_bias = torch::randn({num_expert, hidden_units});

  auto w2_weight = torch::randn({num_expert, dim, hidden_units});
  auto w2_bias = torch::randn({num_expert, dim});

  // auto temp = torch::randn({1, 2, 3});
  auto w1_weight_scale = 255 / get<0>(torch::max(torch::abs(w1_weight), -1)) / 2;  // {num_expert, hidden_units}
  auto w1_weight_int32_f = (w1_weight.permute({2, 0, 1}) * w1_weight_scale).permute({1, 2, 0});
  auto w1_weight_int32 = w1_weight_int32_f.round().clip(-128, 127).to(torch::kI32);  // {num_expert, hidden_units, dim}
  // print_tensor(w1_weight_int32, "w1_weight_int32", true);
  // auto w1_weight_int32 = w1_weight_int32_f.round().clip(-128,127).to(torch::kI8);

  auto w2_weight_scale = 255 / get<0>(torch::max(torch::abs(w2_weight), -1)) / 2;
  auto w2_weight_int32_f = (w2_weight.permute({2, 0, 1}) * w2_weight_scale).permute({1, 2, 0});
  auto w2_weight_int32 = w2_weight_int32_f.round().clip(-128, 127).to(torch::kI32);
  // auto w2_weight_int32 = w2_weight_int32_f.round().clip(-128,127).to(torch::kI8);
  // auto ww = torch::quantize_per_tensor()
  // cout << w1_weight;
  // cout << w1_weight_int32;
  // cout << w1_weight_scale;
  // cout << w2_weight;
  // cout << w2_weight_int32;
  // cout << w2_weight_scale;

  // auto
  // step1. build trt engine
  {
    auto builder = makeShared(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto network = makeShared(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto config = makeShared(builder->createBuilderConfig());

    // the input type is float.
    // if enable fp16, input will be casted to fp16 firstly
    auto input = network->addInput("input", DataType::kFLOAT, Dims2{-1, dim});
    auto gate_idx = network->addInput("gate_idx", DataType::kINT32, Dims2{1, -1});

    nvinfer1::Weights w_w1_weight = nvinfer1::Weights{DataType::kFLOAT, w1_weight.data_ptr<float>(), (int)w1_weight.numel()};
    nvinfer1::Weights w_w2_weight = nvinfer1::Weights{DataType::kFLOAT, w2_weight.data_ptr<float>(), (int)w2_weight.numel()};
    // nvinfer1::Weights w_w1_bias = nvinfer1::Weights{DataType::kFLOAT,
    // w1_bias.data_ptr<float>(), (int)w1_bias.numel()};

    // nvinfer1::Weights w_w2_weight = nvinfer1::Weights{DataType::kFLOAT, w2_weight.data_ptr<float>(),
    // (int)w2_weight.numel()}; nvinfer1::Weights w_w2_bias = nvinfer1::Weights{DataType::kFLOAT,
    // w2_bias.data_ptr<float>(), (int)w2_bias.numel()};

    // add relu to cast input to fp16.
    auto relu1 = network->addActivation(*input, ActivationType::kRELU)->getOutput(0);

    // add constant
    // auto w_w1_weight = addConstant(network, w1_weight);
    // for (int i = 0; i < 100; i++) {
    //   printf("%f\n", static_cast<float*>(const_cast<void *>(w_w1_weight.values))[i]);
    // }
    // auto w_w1_bias = addConstant(network, w1_bias);
    // auto w_w2_weight = addConstant(network, w2_weight);
    // auto w_w2_bias = addConstant(network, w2_bias);
    nvinfer1::Weights w_w1_bias = nvinfer1::Weights{DataType::kFLOAT, w1_bias.data_ptr<float>(), (int)w1_bias.numel()};
    nvinfer1::Weights w_w2_bias = nvinfer1::Weights{DataType::kFLOAT, w2_bias.data_ptr<float>(), (int)w2_bias.numel()};
    // int 8 parameters
    // auto w_w1_weight_int32 = addConstant(network, w1_weight_int32, DataType::kINT32);
    // auto w_w1_weight_scale = addConstant(network, w1_weight_scale);
    // auto w_w2_weight_int32 = addConstant(network, w2_weight_int32, DataType::kINT32);
    // auto w_w2_weight_scale = addConstant(network, w2_weight_scale);

    // auto w_w1_weight_int8 = addConstant(network, w1_weight_int32, DataType::kINT8);
    // auto w_w2_weight_int8 = addConstant(network, w1_weight_int8, DataType::kINT8);

    // auto wtemp = addConstant(network, torch::randn({num_expert, hidden_units, dim}));

    // auto outputs = AddFMoEExpertPlugin(network.get(),
    //                                    {input, gate_idx, w_w1_weight, w_w1_bias, w_w2_weight, w_w2_bias},
    //                                    data_type, num_expert, dim, hidden_units, act_type);
    // printf("mark!!!!-----------------------\n");

    ITensorVector outputs = AddFMoEExpertPlugin(network.get(), {relu1, gate_idx},
                                       w_w1_weight, w_w2_weight, w_w1_bias, w_w2_bias,
                                       data_type, num_expert, dim, hidden_units, act_type);

    auto relu2 = network->addActivation(*outputs[0], ActivationType::kRELU)->getOutput(0);

    // outputs = AddFMoEExpertPlugin(network.get(), {outputs[0], gate_idx, w_w1_weight, w_w1_bias, w_w2_weight,
    // w_w2_bias}, data_type, num_expert, dim, hidden_units, act_type);

    // auto relu2 = network->addActivation(*outputs[0], ActivationType::kRELU);

    network->markOutput(*relu2);

    auto profile = builder->createOptimizationProfile();
    {
      Dims2 min_profile{1, dim};
      Dims2 opt_profile{2000, dim};
      Dims2 max_profile{20000, dim};
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_profile);
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, opt_profile);
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_profile);
    }
    {
      Dims2 min_profile{1, 1};
      Dims2 opt_profile{1, 2000};
      Dims2 max_profile{1, 20000};
      profile->setDimensions(gate_idx->getName(), OptProfileSelector::kMIN, min_profile);
      profile->setDimensions(gate_idx->getName(), OptProfileSelector::kOPT, opt_profile);
      profile->setDimensions(gate_idx->getName(), OptProfileSelector::kMAX, max_profile);
    }
    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(10);
    config->setMaxWorkspaceSize(1 * (1 << 30));
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    if (data_type == nvinfer1::DataType::kHALF) {
      config->setFlag(BuilderFlag::kFP16);
    }

    // data_type=int8 support input_type=float or half
    // Despite setting the STRICT_TYPES mode, TensorRT still chooses the fastest implementation
    // path based on the speed of the enqueue function.
    // Here, the input type of the FMoEPlugin is forcibly set to Half for unit testing purposes.
    if (input_type == nvinfer1::DataType::kHALF) {
      config->setFlag(BuilderFlag::kFP16);
      relu1->setType(nvinfer1::DataType::kHALF);
      relu2->setType(nvinfer1::DataType::kFLOAT);
    }

    auto engine = makeShared(builder->buildEngineWithConfig(*network, *config));

    save_engine(*engine, engine_file);
  }

  const int N = 6;
  int seq_len_arr[N] = {2, 100, 1500, 5000, 10000, 18000};
  // const int N = 1;
  // int seq_len_arr[N] = {7,};

  for (int n = 0; n < N; n++) {
    auto seq_len = seq_len_arr[n];

    // generate gate_idx, {0, num_expert - 1}
    auto gate_idx = torch::randint(0, num_expert, {1, seq_len}, torch::kInt32);
    // cout << gate_idx << endl;

    // gate_idx = torch::_cast_Int(gate_idx);
    // cout << gate_idx << endl;
    // print_tensor(gate_idx, "gate_idx", true);
    // print_tensor(w1_weight, "w1_weight", true);
    // print_tensor(w2_weight, "w2_weight", true);
    // assert(0);

    auto input = torch::randn({seq_len, dim});
    // cout << input << endl;
    // // cout << w1_weight << endl;
    // cout << w1_weight_int32 << endl;
    // cout << w1_weight_scale << endl;
    // cout << w1_bias << endl;
    // cout << w2_weight << endl;
    // cout << w2_weight_int32 << endl;
    // cout << w2_weight_scale << endl;
    // cout << w2_bias << endl;

    auto output = torch::zeros_like(input);

    input = input.relu();
    torch_fmoe_forward<float>(input, gate_idx, num_expert, w1_weight, w1_bias, w2_weight, w2_bias, output);
    output = output.relu();

    auto trt_input = input.clone();
    auto trt_gate_idx = gate_idx.clone();
    auto trt_output = torch::zeros_like(output);

    auto trt_input_ptr = TensorToCuda(trt_input);
    auto trt_gate_idx_ptr = TensorToCuda(trt_gate_idx);
    auto trt_output_ptr = TensorToCuda(trt_output);

    // step3: trt infer
    {
      auto engine = makeShared(load_engine(engine_file));

      auto context = engine->createExecutionContext();
      context->setOptimizationProfile(0);

      Dims2 input_dim{seq_len, dim};
      Dims2 gate_idx_dim{1, seq_len};
      context->setBindingDimensions(0, input_dim);
      context->setBindingDimensions(1, gate_idx_dim);
      if (!context->allInputDimensionsSpecified()) {
        LOG(ERROR) << "allInputDimensionsSpecified error" << endl;
        assert(0);
      }

      void* device_bindings[3] = {trt_input_ptr.get(), trt_gate_idx_ptr.get(), trt_output_ptr.get()};
      bool ret = context->enqueueV2(device_bindings, nullptr, nullptr);

      string local_time_name = std::to_string(seq_len) + "x" + std::to_string(dim);
      for (int i = 0; i < loop; i++) {
        g_timer.begin(local_time_name.c_str());
        bool ret = context->enqueueV2(device_bindings, nullptr, nullptr);
        g_timer.acc(local_time_name.c_str());
      }
    }
    // CudaToTensor(trt_input_ptr, trt_input);
    // CudaToTensor(trt_gate_idx_ptr, trt_gate_idx);
    CudaToTensor(trt_output_ptr, trt_output);
    // step4: compare
    {
      output = output.cpu();
      trt_output = trt_output.cpu();
      
      // float* result = trt_output.data_ptr<float>();
      // for (int i = 0; i < 100; i++) printf("%f\n", result[i]);
      // print_tensor(trt_output, "trt_output", true);
      // half* result = reinterpret_cast<half*>(trt_output.data_ptr<float>());
      // for (int i = 0; i < 1000; i++) printf("%f ", (float)result[i]);
      print_tensor(output, "output", true);
      print_tensor(trt_output, "trt_output", true);
      float diff = 0.05;
      if (data_type == DataType::kHALF)
        diff = 5;
      else if (data_type == DataType::kINT8)
        diff = 30;
      EXPECT_TRUE(test_compare(output.data_ptr<float>(), trt_output.data_ptr<float>(), output.numel(), diff));
    }
    // if (n == 1)
    // break;
    // return;
  }

  g_timer.report();
  g_timer.clear();
}

// TEST_F(FMoEPluginTest, FLOAT) {
//   string engine_file = ENGINE_FILE_NAME;
//   auto data_type = DataType::kFLOAT;
//   fmoe_test(engine_file, data_type);
//   CUDA_CHECK(cudaDeviceReset());

//   cudaDeviceSynchronize();
//   cudaProfilerStop();
// }

// TEST_F(FMoEPluginTest, HALF) {
//   string engine_file = ENGINE_FILE_NAME;
//   auto data_type = DataType::kHALF;
//   fmoe_test(engine_file, data_type);
//   cudaDeviceSynchronize();
//   cudaProfilerStop();
// }

TEST_F(FMoEPluginTest, FLOAT) {
  string engine_file = ENGINE_FILE_NAME;
  auto data_type = DataType::kFLOAT;
  auto input_type = DataType::kFLOAT;
  fmoe_test(engine_file, data_type, input_type);
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

TEST_F(FMoEPluginTest, HALF) {
  string engine_file = ENGINE_FILE_NAME;
  auto data_type = DataType::kHALF;
  auto input_type = DataType::kHALF;
  fmoe_test(engine_file, data_type, input_type);
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

TEST_F(FMoEPluginTest, INT8_FLOAT) {
  string engine_file = ENGINE_FILE_NAME;
  auto data_type = DataType::kINT8;
  auto input_type = DataType::kFLOAT;
  fmoe_test(engine_file, data_type, input_type);
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

TEST_F(FMoEPluginTest, INT8_HALF) {
  string engine_file = ENGINE_FILE_NAME;
  auto data_type = DataType::kINT8;
  auto input_type = DataType::kHALF;
  fmoe_test(engine_file, data_type, input_type);
  cudaDeviceSynchronize();
  cudaProfilerStop();
}