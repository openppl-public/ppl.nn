// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/nn/engines/arm/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/batch_normalization_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_batch_normalization.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/batch_normalization.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

BatchNormalizationOp::BatchNormalizationOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeBatchNormalization(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode BatchNormalizationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode BatchNormalizationOp::SelectFormat(const InputOutputInfo& info,
                                           std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                           std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = selected_output_formats->at(0) =
        info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_formats->at(i) = ppl::common::DATAFORMAT_NDARRAY;
    }
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode BatchNormalizationOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder fusion_builder;
    auto fb_fusion_data = ppl::nn::pmx::arm::CreateFusionDataDirect(fusion_builder, (fuse_relu_ ? 1 : 0), &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(fusion_builder, ppl::nn::pmx::arm::PrivateDataType_FusionData, fb_fusion_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(fusion_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeBatchNormalizationParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(fusion_builder.GetBufferPointer(), fusion_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_BatchNormalizationParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode BatchNormalizationOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::BatchNormalizationParam>();
    ppl::nn::pmx::onnx::DeserializeBatchNormalizationParam(*fb_op_param->value_as_BatchNormalizationParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_fusion_data = arm_op_data->value_as_FusionData();
    fuse_relu_ = arm_fusion_data->fuse_relu();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_fusion_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_fusion_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* BatchNormalizationOp::CreateKernelImpl() const {
    auto kernel = CreateKernelImplWithParam<BatchNormalizationKernel>(param_.get());
    if (kernel) {
        kernel->SetFuseReLU(fuse_relu_);
    }
    return kernel;
}

}}} // namespace ppl::nn::arm
