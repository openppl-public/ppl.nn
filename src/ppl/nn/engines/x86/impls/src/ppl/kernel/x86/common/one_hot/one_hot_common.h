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
#include "ppl/kernel/x86/common/transpose/transpose_common.h"
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
    const auto indices_size =  indices_shape->GetElementsExcludingPadding();
    const auto dst_size = indices_size * depth;
    std::fill(dst, dst + dst_size, off_value);
    eT* dst_ptr = dst;
    // indices = [2,3]
    // depth = 4
    // dst = [2,3,4]

    // parallel optimize
    for(uint64_t i=0; i<indices_size; i++){
        dst_ptr[i] = on_value;
        dst_ptr += depth;
    }

    // axis = 1
    // transpose dst = [2,4,3]
    // TODO: compute dst_shape_tmp[2,3,4], dst_shape[2,4,3]
    // perm = [0,1,...,r-1], swap(r-1, axis)
    const ppl::nn::TensorShape *dst_shape_tmp = indices_shape;
    const ppl::nn::TensorShape *dst_shape = indices_shape;
    // int32_t* perm;
    std::vector<int32_t> perm;
    return transpose_ndarray<eT>(dst_shape_tmp, dst_shape, perm.data(), dst, dst);
    // return ppl::common::RC_UNSUPPORTED; 
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_TILE_TILE_COMMON_H_