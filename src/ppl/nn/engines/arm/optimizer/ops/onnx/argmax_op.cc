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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/argmax_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_argmax.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/argmax.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ArgMaxOp::ArgMaxOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeArgMax(info, param_.get());
    };

    infer_type_func_ = [](InputOutputInfo* info) {
        info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(DATATYPE_INT64);
    };
}

RetCode ArgMaxOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ArgMaxOp::SelectDataType(const InputOutputInfo& info,
                                 std::vector<ppl::common::datatype_t>* selected_input_types,
                                 std::vector<ppl::common::datatype_t>* selected_output_types,
                                 const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    selected_output_types->at(0) = DATATYPE_INT64;
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ArgMaxOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeArgMaxParam(*param_.get(), &op_builder);
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ArgMaxParam, fb_param.Union(), 0);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ArgMaxOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::ArgMaxParam>();
    ppl::nn::pmx::onnx::DeserializeArgMaxParam(*fb_op_param->value_as_ArgMaxParam(), param_.get());
    common_param_.output_types[0] = DATATYPE_INT64;
    common_param_.output_formats[0] = DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

#endif

KernelImpl* ArgMaxOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ArgMaxKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
