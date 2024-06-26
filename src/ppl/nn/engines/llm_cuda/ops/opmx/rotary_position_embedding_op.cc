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

#include "rotary_position_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/opmx/rotary_position_embedding_kernel.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

RetCode RotaryPositionEmbeddingOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode RotaryPositionEmbeddingOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::opmx::RotaryPositionEmbeddingParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* RotaryPositionEmbeddingOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<RotaryPositionEmbeddingKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode RotaryPositionEmbeddingOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = opmx::CreateRotaryPositionEmbeddingParam(builder, 
        param_.get()->bypass_key,
        param_.get()->rotary_dim,
        param_.get()->theta,
        param_.get()->max_position_embeddings,
        (ppl::nn::llm::cuda::pmx::RotaryPositionEmbeddingScalingType)param_.get()->scaling_type,
        param_.get()->scaling_factor);
    auto fb_op_param = opmx::CreateOpParam(builder, opmx::OpParamType_RotaryPositionEmbeddingParam, fb_param.Union());
    opmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode RotaryPositionEmbeddingOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = opmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_RotaryPositionEmbeddingParam();
    param_ = make_shared<ppl::nn::opmx::RotaryPositionEmbeddingParam>();
    param_.get()->bypass_key                = fb_param->bypass_key();
    param_.get()->rotary_dim                = fb_param->rotary_dim();
    param_.get()->theta                     = fb_param->theta();
    param_.get()->max_position_embeddings   = fb_param->max_position_embeddings();
    param_.get()->scaling_type              = fb_param->scaling_type();
    param_.get()->scaling_factor            = fb_param->scaling_factor();

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::opmx
