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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_PLAIN_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_PLAIN_CUDA_DEVICE_H_

#include "ppl/nn/engines/cuda/cuda_device.h"
#if CUDA_VERSION >= 11040
#include "ppl/common/cuda/cuda_plain_async_allocator.h"
#endif

namespace ppl { namespace nn { namespace cuda {

class PlainCudaDevice final : public CudaDevice {
public:
    virtual ~PlainCudaDevice();

    ppl::common::RetCode Init(uint32_t device_id, ppl::common::NcclParam* tp_nccl_param);

    using CudaDevice::Realloc;
    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    void Free(BufferDesc*) override;

private:
#if CUDA_VERSION >= 11040
    ppl::common::CudaPlainAsyncAllocator allocator_;
#endif
};

}}} // namespace ppl::nn::cuda

#endif
