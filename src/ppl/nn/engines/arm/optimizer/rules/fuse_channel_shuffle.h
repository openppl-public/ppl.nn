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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_RULES_FUSE_CHANNEL_SHUFFLE_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_RULES_FUSE_CHANNEL_SHUFFLE_H_

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"
#include "ppl/nn/engines/arm/optimizer/opt_rule.h"

namespace ppl { namespace nn { namespace arm {

class FuseChannelShuffleRule : public OptRule {
public:
    FuseChannelShuffleRule() {
        SetLevel(OPT_RULE_LEVEL_1);
        SetTag("BeforeLayoutOptimize");
        SetName("FuseChannelShuffle");
    };
    virtual ~FuseChannelShuffleRule() {}

    bool Apply(const OptKernelOptions& options) override;
};

}}} // namespace ppl::nn::arm

#endif
