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

#include "ppl/nn/engines/cuda/kernels/onnx/split_kernel.h"

#include "cudakernel/memory/split.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode SplitKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    int32_t dim_count = input->GetShape()->GetDimCount();

    // Opt 1
    if (ctx->IsLastConsumerOfInput(0) && input->GetType() == TENSORTYPE_NORMAL &&
        ctx->GetOutputCount() == 1 &&
        input->GetShape()->CalcElementsIncludingPadding() ==
            ctx->GetOutput<TensorImpl>(0)->GetShape()->CalcElementsIncludingPadding()) {
        auto output = ctx->GetOutput<TensorImpl>(0);
        output->TransferBufferFrom(input);
        return ppl::common::RC_SUCCESS;
    }

    // Opt 2
    int32_t real_axis = (param_->axis + dim_count) % dim_count; 
    if (ctx->GetOutputCount() == 3) {
        auto output0 = ctx->GetOutput<TensorImpl>(0);
        auto output1 = ctx->GetOutput<TensorImpl>(1);
        auto output2 = ctx->GetOutput<TensorImpl>(2);
        auto status = PPLCUDAAlignedSplit3ForwardImp(GetStream(),
            input->GetShape(), input->GetBufferPtr(),
            real_axis,
            output0->GetShape(), output0->GetBufferPtr(),
            output1->GetShape(), output1->GetBufferPtr(),
            output2->GetShape(), output2->GetBufferPtr());
        if (ppl::common::RC_SUCCESS == status) {
            return status;
        }
    }

    // Fallback
    dst_dims_.resize(ctx->GetOutputCount());
    dst_list_.resize(ctx->GetOutputCount());

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto output = ctx->GetOutput<TensorImpl>(i);
        dst_list_[i] = output->GetBufferPtr();
        const TensorShape& output_shape = *output->GetShape();
        if(output_shape.CalcElementsExcludingPadding() < output_shape.CalcElementsIncludingPadding()) {
            cudaMemset(dst_list_[i], 0, output_shape.CalcBytesIncludingPadding());
        }
        dst_dims_[i] = output->GetShape()->GetDims();
    }

    ppl::common::RetCode status = PPLCUDASplitForwardImp(
        GetStream(), real_axis, input->GetShape(), input->GetBufferPtr(),
        ctx->GetOutputCount(), dst_dims_.data(), dst_list_.data());
    return status;
}

}}} // namespace ppl::nn::cuda
