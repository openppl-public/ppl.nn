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

#ifndef __ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_

#include <string.h> // for memcpy
#include "ppl/kernel/x86/common/internal_include.h"
// #include "ppl/kernel/x86/common/transpose/transpose_common.h"
#include <algorithm>
namespace ppl { namespace kernel { namespace x86 {


template <typename eT>
static ppl::common::RetCode one_hot_ndarray_common(
    const int64_t *indices,
    const ppl::nn::TensorShape *indices_shape,
    eT *dst,
    const eT on_value,
    const eT off_value,
    const int64_t depth,
    const int32_t axis)
{
    uint32_t indices_rank = indices_shape->GetDimCount();
    uint32_t real_axis = axis < 0 ? axis + indices_rank : axis;
    uint64_t outer_dim = indices_shape->GetElementsToDimensionExcludingPadding(real_axis);
    uint64_t inner_dim = indices_shape->GetElementsFromDimensionExcludingPadding(real_axis);
    uint64_t axis_dim = depth;
    uint64_t stride = axis_dim * inner_dim;

    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < outer_dim; i++) {
        eT *dst_base = dst + i * axis_dim * inner_dim; 
        std::fill(dst_base, dst_base + stride, off_value);
        for (uint64_t k = 0; k < inner_dim; ++k) {
            int64_t idx = indices[i * inner_dim + k];
            idx = idx < 0 ? idx + depth : idx;
            if(idx < 0 || idx >= depth){
                return ppl::common::RC_INVALID_VALUE;
            }
            eT *p_dst = dst_base + k;
            p_dst[idx * inner_dim] = on_value;
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_