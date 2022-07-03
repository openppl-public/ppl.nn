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

#include "ppl/nn/engines/x86/kernels/onnx/one_hot_kernel.h"
#include "ppl/kernel/x86/int64/one_hot.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode OneHotKernel::DoExecute(KernelExecContext* ctx) {
    return ppl::common::RC_UNSUPPORTED;
}
bool OneHotKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto indices_tensor = ctx.GetInput<TensorImpl>(0);
    auto depth_tensor = ctx.GetInput<TensorImpl>(1);
    auto value_tensor = ctx.GetInput<TensorImpl>(2);
    if(!indices_tensor || !value_tensor || !depth_tensor) return false;
    // value = [off_value, on_value]
    if(value_tensor->GetShape()->GetElementsExcludingPadding() != 2)
        return false;
    // indices in [-depth, depth-1]
    // auto indices_ptr = indices_tensor->GetBufferPtr<int64_t>();
    // auto depth = depth_tensor->GetBufferPtr<int64_t>()[0];
    // uint64_t indices_size = indices_tensor->GetShape()->GetElementsExcludingPadding();
    // for(uint64_t i=0; i<indices_size; i++){
    //     if(indices_ptr[i] < -depth || indices_ptr[i] > depth-1){
    //         return false;
    //     }
    //     indices_ptr[i] = indices_ptr[i] >= 0 ? 
    //                     indices_ptr[i] : indices_ptr[i] + depth;
    // }
    return true;
}

}}} // namespace ppl::nn::x86
