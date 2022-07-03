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
    PPLNN_X86_REQUIRED_INPUT(indices_tensor, 0);
    PPLNN_X86_REQUIRED_INPUT(depth_tensor, 1);
    PPLNN_X86_REQUIRED_INPUT(value_tensor, 2);
    PPLNN_X86_REQUIRED_OUTPUT(y, 0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [indices]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(indices_tensor);
    PPLNN_X86_DEBUG_TRACE("Input [depth]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(depth_tensor);
    PPLNN_X86_DEBUG_TRACE("Input [value]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(value_tensor);

    PPLNN_X86_DEBUG_TRACE("axis: %d\n", param_->axis);
    const auto isa = GetISA();
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", isa);

    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);

    // const auto data_type = indices_tensor->GetShape()->GetDataType();
    const auto* dst_shape = y->GetShape();
    const auto* indices_shape = indices_tensor->GetShape();
    const auto data_type = ctx->GetOutput<TensorImpl>(0)->GetShape()->GetDataType(); // decide on value type
    const auto data_format = ctx->GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (data_type == ppl::common::DATATYPE_INT64) {
            const auto* value = value_tensor->GetBufferPtr<int64_t>();
            const auto depth = value_tensor->GetBufferPtr<int64_t>()[0];
            return ppl::kernel::x86::one_hot_ndarray_int64(indices_tensor->GetBufferPtr<const int64_t>(), indices_shape, 
                                                       y->GetBufferPtr<int64_t>(), value[1],value[0], depth, param_->axis);
        } 
        else if (data_type == ppl::common::DATATYPE_FLOAT32) {
            const auto* value = value_tensor->GetBufferPtr<float>();
            const auto depth = value_tensor->GetBufferPtr<float>()[0];
            return ppl::kernel::x86::one_hot_ndarray_fp32(indices_tensor->GetBufferPtr<const int64_t>(), indices_shape, 
                                                       y->GetBufferPtr<float>(), value[1],value[0], depth, param_->axis);
        }
        else {
            LOG(ERROR) << "unsupported data type " << ppl::common::GetDataTypeStr(data_type) << ".";
        }
    } else {
        LOG(ERROR) << "unsupported data format " << ppl::common::GetDataFormatStr(data_format) << ".";
    }
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
