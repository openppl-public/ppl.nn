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

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/riscv/optimizer/ops/mmcv/mmcv_gridsample_op.h"
#include "ppl/nn/engines/riscv/kernels/mmcv/mmcv_gridsample_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_gridsample.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode MMCVGridSampleOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVGridSample(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    if (options.engine_options->forward_precision == DATATYPE_FLOAT16) {
        format_ = DATAFORMAT_N8CX;
    } else if (options.engine_options->forward_precision == DATATYPE_FLOAT32) {
        format_ = DATAFORMAT_N4CX;
    }

    return RC_SUCCESS;
}

RetCode MMCVGridSampleOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                       vector<dataformat_t>* selected_output_formats) {
    if (DATAFORMAT_N8CX == format_) {
        selected_input_formats->at(0) = DATAFORMAT_N8CX;
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    } else if (DATAFORMAT_N4CX == format_) {
        selected_input_formats->at(0) = DATAFORMAT_N4CX;
        selected_output_formats->at(0) = DATAFORMAT_N4CX;
    }
    return RC_SUCCESS;
}

RetCode MMCVGridSampleOp::SelectDataType(const InputOutputInfo& info, datatype_t forward_precision,
                                         std::vector<datatype_t>* selected_input_data_types,
                                         std::vector<datatype_t>* selected_output_data_types) {
    selected_input_data_types->at(0) = forward_precision;
    selected_input_data_types->at(1) = DATATYPE_FLOAT32;
    selected_output_data_types->at(0) = forward_precision;

    return RC_SUCCESS;
}

KernelImpl* MMCVGridSampleOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVGridSampleKernel>(param_.get());
}

}}}; // namespace ppl::nn::riscv