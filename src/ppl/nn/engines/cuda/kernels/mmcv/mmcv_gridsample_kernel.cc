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

#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_gridsample_kernel.h"

#include "cudakernel/nn/mmcv_gridsample.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode MMCVGridSampleKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    MMCVGridSampleKernelParam param_kernel_;
    param_kernel_.align_corners = param_->align_corners;
    param_kernel_.interpolation_mode = param_->interpolation_mode;
    param_kernel_.padding_mode = param_->padding_mode;
    ppl::common::RetCode status =
        PPLCUDAMMCVGridSampleForwardImp(GetStream(), input0->GetShape(), input0->GetBufferPtr(), input1->GetShape(),
                                        input1->GetBufferPtr(), output->GetShape(), output->GetBufferPtr(), param_kernel_);
    return status;
}

}}} // namespace ppl::nn::cuda
