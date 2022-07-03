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

#include "ppl/nn/oputils/onnx/reshape_one_hot.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeOneHot(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const OneHotParam*>(arg);
    const TensorShape& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape(); //indices
    const auto* depth_ptr = info->GetInput<TensorImpl>(1)->GetBufferPtr<const int64_t>(); //depth
    const int64_t depth = depth_ptr[0];
    uint32_t fixed_axis = // [-r, r-1]
        param->axis >= 0 ? param->axis : param->axis + info->GetInput<TensorImpl>(0)->GetShape()->GetDimCount();
    
    std::vector<int64_t> output_dim(in_shape0.GetDimCount() + 1); // add one dimension in axis
    // [3,4,2,5] axis = 2 depth = 10
    // [3,4,10,2,5]

    for (uint32_t i = 0; i < fixed_axis; ++i) {
        output_dim[i] = in_shape0.GetDim(i);
    }
    output_dim[fixed_axis] = depth;
    for (uint32_t i = fixed_axis+1; i < output_dim.size(); ++i) {
        output_dim[i] = in_shape0.GetDim(i-1);
    }

    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
