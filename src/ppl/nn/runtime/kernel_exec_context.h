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

#ifndef _ST_HPC_PPL_NN_RUNTIME_KERNEL_EXEC_CONTEXT_H_
#define _ST_HPC_PPL_NN_RUNTIME_KERNEL_EXEC_CONTEXT_H_

#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn {

/**
   @class KernelExecContext
   @brief kernel execution context
*/
class KernelExecContext final : public InputOutputInfo {
public:
    void SetProfilingFlag(bool is_profiling_enabled) {
        is_profiling_enabled_ = is_profiling_enabled;
    }
    bool IsProfilingEnabled() const {
        return is_profiling_enabled_;
    }
    void SetEdgeLastConsumerList(const std::vector<nodeid_t>* l) {
        edge_last_consumer_ = l;
    }
    bool IsLastConsumerOfInput(uint32_t idx) const {
        auto eid = node_->GetInput(idx);
        return (edge_last_consumer_->at(eid) == node_->GetId());
    }

private:
    bool is_profiling_enabled_ = false;
    const std::vector<nodeid_t>* edge_last_consumer_ = nullptr;
};

}} // namespace ppl::nn

#endif
