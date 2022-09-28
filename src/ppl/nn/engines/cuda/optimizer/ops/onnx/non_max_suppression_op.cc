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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/non_max_suppression_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/non_max_suppression_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_non_max_suppression.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_non_max_suppression.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/non_max_suppression.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode NonMaxSupressionOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<NonMaxSuppressionParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

NonMaxSupressionOp::NonMaxSupressionOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        // prefer fp32 version for precision
        auto shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        shape0->SetDataType(DATATYPE_FLOAT32);

        auto shape1 = info->GetInput<TensorImpl>(1)->GetShape();
        shape1->SetDataType(DATATYPE_FLOAT32);

        if (info->GetInputCount() > 2) {
            auto shape2 = info->GetInput<TensorImpl>(2)->GetShape();
            shape2->SetDataType(DATATYPE_INT64);
        }
        if (info->GetInputCount() > 3) {
            auto shape3 = info->GetInput<TensorImpl>(3)->GetShape();
            shape3->SetDataType(DATATYPE_FLOAT32);
        }
        if (info->GetInputCount() > 4) {
            auto shape4 = info->GetInput<TensorImpl>(4)->GetShape();
            shape4->SetDataType(DATATYPE_FLOAT32);
        }
        auto shape = info->GetOutput<TensorImpl>(0)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return RC_SUCCESS;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        int64_t max_output_boxes_per_class = 0;
        if (info->GetInputCount() >= 3) {
            auto status = info->GetInput<TensorImpl>(2)->CopyToHost(&max_output_boxes_per_class);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy max output boxes failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        return onnx::ReshapeNonMaxSuppression(info, max_output_boxes_per_class);
    };
}

RetCode NonMaxSupressionOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* NonMaxSupressionOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<NonMaxSuppressionKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode NonMaxSupressionOp::SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::onnx::SerializeNonMaxSuppressionParam(param_, &builder);
        auto fb_op_param = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_NonMaxSuppressionParam, fb_param.Union());
        pmx::onnx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode NonMaxSupressionOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::onnx::GetOpParam(base);
        auto fb_argmax_param = fb_op_param->value_as_NonMaxSuppressionParam();
        pmx::onnx::DeserializeNonMaxSuppressionParam(*fb_argmax_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
