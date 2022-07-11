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

#include "ppl/kernel/x86/common/one_hot/one_hot_common.h"
namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode one_hot_ndarray_fp32(
    const void *indices,
    const ppl::nn::TensorShape *indices_shape,
    const void *depth,
    const ppl::nn::TensorShape *depth_shape,
    const float *values,
    float *dst,
    const int32_t axis
    )
{
    return one_hot_ndarray_common<float>( indices,indices_shape,
                                            depth, depth_shape,
                                            values, dst, axis);
}


}}}; // namespace ppl::kernel::x86
