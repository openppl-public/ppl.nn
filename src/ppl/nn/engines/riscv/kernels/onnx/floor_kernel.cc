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

#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/riscv/kernels/onnx/floor_kernel.h"

#include "ppl/kernel/riscv/fp16/floor.h"
#include "ppl/kernel/riscv/fp32/floor.h"

namespace ppl { namespace nn { namespace riscv {

ppl::common::RetCode FloorKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_RISCV_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_RISCV_DEBUG_TRACE("Input [input]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_RISCV_DEBUG_TRACE("Output [output]:\n");
    PPL_RISCV_TENSOR_PRINT_DEBUG_MSG(output);

    const auto data_type = input->GetShape()->GetDataType();
    const auto data_format = input->GetShape()->GetDataFormat();
    if (ppl::common::DATATYPE_FLOAT32 == data_type) {
        return ppl::kernel::riscv::floor_fp32(input->GetShape(), input->GetBufferPtr<float>(),
                                              output->GetBufferPtr<float>());
    } else if (ppl::common::DATATYPE_FLOAT16 == data_type) {
        return ppl::kernel::riscv::floor_fp16(input->GetShape(), input->GetBufferPtr<__fp16>(),
                                              output->GetBufferPtr<__fp16>());
    } else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::nn::riscv