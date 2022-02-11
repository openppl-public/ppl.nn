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

#include "ppl/nn/models/onnx/parsers/onnx/parse_reduce_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseReduceParam(const ::onnx::NodeProto& pb_node, const map<string, uint64_t>&, void* arg,
                                      ir::Node* node, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ReduceParam*>(arg);

    if (node->GetType().name == "ReduceSum")  {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceSum;
    } else if (node->GetType().name == "ReduceMax")  {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceMax;
    } else if (node->GetType().name == "ReduceMin")  {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceMin;
    } else if (node->GetType().name == "ReduceProd")  {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceProd;
    } else if (node->GetType().name == "ReduceMean")  {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceMean;
    } else {
        param->reduce_type = ppl::nn::common::ReduceParam::ReduceUnknown;
    }
    
    param->axes = utils::GetNodeAttrsByKey<int32_t>(pb_node, "axes");
    int keepdims = utils::GetNodeAttrByKey<int>(pb_node, "keepdims", 1);
    param->keep_dims = (keepdims != 0);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
