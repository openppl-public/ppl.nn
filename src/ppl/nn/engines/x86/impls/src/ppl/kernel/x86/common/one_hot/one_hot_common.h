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
#include "ppl/kernel/x86/common/internal_include.h"
namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
static inline ppl::common::RetCode cast2int64_kernel(const eT *src, int64_t* &dst, 
                                                     const uint64_t length)
{
    dst = (int64_t*)malloc(length*sizeof(int64_t));
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < length; i++) {
        dst[i] = (int64_t)src[i];
    }
    return ppl::common::RC_SUCCESS;
}

static inline int64_t* cast2int64(const ppl::nn::TensorShape *src_shape, const void *src, bool& has_cast)
{
    auto src_type = src_shape->GetDataType();
    if(src_type == ppl::common::DATATYPE_INT64) return (int64_t*)src;
    const uint64_t length = src_shape->GetElementsIncludingPadding();
    int64_t* dst = nullptr;
    has_cast = true;
    switch (src_type)
    {
    case ppl::common::DATATYPE_FLOAT64:
        cast2int64_kernel<double>((double*)src, dst, length);
        break;
    case ppl::common::DATATYPE_FLOAT32:
        cast2int64_kernel<float>((float*)src, dst, length);
        break;
    case ppl::common::DATATYPE_INT32:
        cast2int64_kernel<int32_t>((int32_t*)src, dst, length);
        break;
    case ppl::common::DATATYPE_INT16:
        cast2int64_kernel<int16_t>((int16_t*)src, dst, length);
        break;
    case ppl::common::DATATYPE_INT8:
        cast2int64_kernel<int8_t>((int8_t*)src, dst, length);
        break;
    case ppl::common::DATATYPE_UINT32:
        cast2int64_kernel<uint32_t>((uint32_t*)src, dst, length);
        break;
    case ppl::common::DATATYPE_UINT16:
        cast2int64_kernel<uint16_t>((uint16_t*)src, dst, length);
        break;
    case ppl::common::DATATYPE_UINT8:
        cast2int64_kernel<uint8_t>((uint8_t*)src, dst, length);
        break;
    default:
        has_cast = false;
        break;
    }
    return dst;
}
template <typename eT>
static ppl::common::RetCode one_hot_ndarray_common(
    const void *indices,
    const ppl::nn::TensorShape *indices_shape,
    const void *depth,
    const ppl::nn::TensorShape *depth_shape,
    const eT *values,
    eT *dst,
    const int32_t axis)
{
    bool indices_cast = false;
    bool depth_cast = false;
    int64_t *real_indices = cast2int64(indices_shape, indices, indices_cast);
    int64_t *real_depth = cast2int64(depth_shape, depth, depth_cast);
    if(!real_indices || !real_depth) return ppl::common::RC_INVALID_VALUE;
    int64_t depth_val = *real_depth;
    uint32_t indices_rank = indices_shape->GetDimCount();
    uint32_t real_axis = axis < 0 ? axis + indices_rank + 1 : axis;
    uint64_t outer_dim = indices_shape->GetElementsToDimensionExcludingPadding(real_axis);
    uint64_t inner_dim = indices_shape->GetElementsFromDimensionExcludingPadding(real_axis);
    uint64_t axis_dim = depth_val;
    uint64_t stride = axis_dim * inner_dim;
    bool index_valid = true;
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < outer_dim; i++) {
        if(!index_valid) continue;
        eT *dst_base = dst + i * axis_dim * inner_dim; 
        std::fill(dst_base, dst_base + stride, values[0]);
        for (uint64_t k = 0; k < inner_dim; ++k) {
            int64_t idx = real_indices[i * inner_dim + k];
            idx = idx < 0 ? idx + depth_val : idx;
            if(idx < 0 || idx >= depth_val){
                index_valid = false;
                continue;
            }
            eT *p_dst = dst_base + k;
            p_dst[idx * inner_dim] = values[1];
        }
    }
    if(indices_cast)  free(real_indices);
    if(depth_cast)  free(real_depth);
    if(!index_valid) return ppl::common::RC_INVALID_VALUE;
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_ONE_HOT_ONE_HOT_COMMON_H_