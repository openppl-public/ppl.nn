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

#include "ppl/common/file_mapping.h"
#include "ppl/nn/utils/utils.h"
#include "ppl/nn/models/onnx/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx { namespace utils {

template <typename T>
static vector<T> GetIntsByKey(const ::onnx::NodeProto& node, const char* key) {
    vector<T> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.ints_size());
            for (int32_t j = 0; j < attribute.ints_size(); j++) {
                result[j] = attribute.ints(j);
            }
            break;
        }
    }
    return result;
}

template <>
vector<int32_t> GetNodeAttrsByKey<int32_t>(const ::onnx::NodeProto& node, const char* key) {
    return GetIntsByKey<int32_t>(node, key);
}

template <>
vector<uint32_t> GetNodeAttrsByKey<uint32_t>(const ::onnx::NodeProto& node, const char* key) {
    return GetIntsByKey<uint32_t>(node, key);
}

template <>
vector<float> GetNodeAttrsByKey<float>(const ::onnx::NodeProto& node, const char* key) {
    vector<float> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.floats_size());
            for (int32_t j = 0; j < attribute.floats_size(); j++) {
                result[j] = attribute.floats(j);
            }
            break;
        }
    }
    return result;
}

template <>
vector<string> GetNodeAttrsByKey<string>(const ::onnx::NodeProto& node, const char* key) {
    vector<string> result;
    for (int32_t i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            result.resize(attribute.strings_size());
            for (int32_t j = 0; j < attribute.strings_size(); j++) {
                result[j] = attribute.strings(j);
            }
            break;
        }
    }
    return result;
}

template <>
int32_t GetAttrValue<int32_t>(const ::onnx::AttributeProto& attribute) {
    return attribute.i();
}

template <>
uint32_t GetAttrValue<uint32_t>(const ::onnx::AttributeProto& attribute) {
    return attribute.i();
}

template <>
int64_t GetAttrValue<int64_t>(const ::onnx::AttributeProto& attribute) {
    return attribute.i();
}

template <>
string GetAttrValue<string>(const ::onnx::AttributeProto& attribute) {
    return attribute.s();
}

template <>
float GetAttrValue<float>(const ::onnx::AttributeProto& attribute) {
    return attribute.f();
}

const ::onnx::TensorProto* GetTensorProtoByKey(const ::onnx::NodeProto& node, const char* key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const ::onnx::AttributeProto& attribute = node.attribute(i);
        if (attribute.name() == key) {
            return &attribute.t();
        }
    }
    return nullptr;
}

datatype_t ConvertOnnxDataTypeToPplDataType(int32_t onnx_data_type) {
    static constexpr datatype_t dt_map[] = {
        DATATYPE_UNKNOWN, // unknown
        DATATYPE_FLOAT32, // float32
        DATATYPE_UINT8, // uint8
        DATATYPE_INT8, // int8
        DATATYPE_UINT16, // uint16
        DATATYPE_INT16, // int16
        DATATYPE_INT32, // int32
        DATATYPE_INT64, // int64
        DATATYPE_UNKNOWN, // String
        DATATYPE_BOOL, // bool
        DATATYPE_FLOAT16, // float16
        DATATYPE_FLOAT64, // float64
        DATATYPE_UINT32, // uint32
        DATATYPE_UINT64, // uint64
        DATATYPE_COMPLEX64, // complex64
        DATATYPE_COMPLEX128, // complex128
        DATATYPE_BFLOAT16, // bfloat16
    };
    static constexpr int32_t onnx_data_type_max = sizeof(dt_map) / sizeof(datatype_t);

    if (onnx_data_type >= onnx_data_type_max) {
        return DATATYPE_UNKNOWN;
    }

    return dt_map[onnx_data_type];
}

int32_t ConvertPplDataTypeToOnnxDataType(datatype_t dt) {
    static constexpr int32_t dt_map[] = {
        ::onnx::TensorProto_DataType_UNDEFINED, // DATATYPE_UNKNOWN
        ::onnx::TensorProto_DataType_UINT8, // DATATYPE_UINT8
        ::onnx::TensorProto_DataType_UINT16, // DATATYPE_UINT16
        ::onnx::TensorProto_DataType_UINT32, // DATATYPE_UINT32
        ::onnx::TensorProto_DataType_UINT64, // DATATYPE_UINT64
        ::onnx::TensorProto_DataType_FLOAT16, // DATATYPE_FLOAT16
        ::onnx::TensorProto_DataType_FLOAT, // DATATYPE_FLOAT32
        ::onnx::TensorProto_DataType_UNDEFINED, // DATATYPE_FLOAT64
        ::onnx::TensorProto_DataType_BFLOAT16, // DATATYPE_BFLOAT16
        ::onnx::TensorProto_DataType_UNDEFINED, // DATATYPE_INT4B
        ::onnx::TensorProto_DataType_INT8, // DATATYPE_INT8
        ::onnx::TensorProto_DataType_INT16, // DATATYPE_INT16
        ::onnx::TensorProto_DataType_INT32, // DATATYPE_INT32
        ::onnx::TensorProto_DataType_INT64, // DATATYPE_INT64
        ::onnx::TensorProto_DataType_BOOL, // DATATYPE_BOOL
        ::onnx::TensorProto_DataType_COMPLEX64, // DATATYPE_COMPLEX64
        ::onnx::TensorProto_DataType_COMPLEX128, // DATATYPE_COMPLEX128
    };
    static constexpr int32_t data_type_max = sizeof(dt_map) / sizeof(int32_t);

    if (dt >= data_type_max) {
        return ::onnx::TensorProto_DataType_UNDEFINED;
    }

    return dt_map[dt];
}

static RetCode LoadExternalData(const ::onnx::TensorProto& pb_tensor, const char* model_file_dir, string* data) {
    if (!model_file_dir) {
        LOG(ERROR) << "`model_file_dir` is null while there are external data of tensor[" << pb_tensor.name() << "].";
        return RC_INVALID_VALUE;
    }

    const string* location = nullptr;
    const string* checksum = nullptr;
    uint64_t offset = 0, length = UINT64_MAX;
    for (int i = 0; i < pb_tensor.external_data_size(); ++i) {
        const string& key = pb_tensor.external_data(i).key();
        const string& value = pb_tensor.external_data(i).value();
        if (key == "location") {
            location = &value;
        } else if (key == "offset") {
            offset = std::stoll(value);
        } else if (key == "length") {
            length = std::stoll(value);
        } else if (key == "checksum") {
            checksum = &value;
        } else {
            LOG(ERROR) << "unrecognized key[" << key << "] in `external_data` field of TensorProto.";
            return RC_UNSUPPORTED;
        }
    }

    if (!location) {
        LOG(ERROR) << "cannot find `location` of tensor[" << pb_tensor.name() << "] in `external_data` field.";
        return RC_NOT_FOUND;
    }
    if (location->empty()) {
        LOG(ERROR) << "`location` is empty of tensor[" << pb_tensor.name() << "].";
        return RC_INVALID_VALUE;
    }
    if (checksum) {
        LOG(WARNING) << "skip checksum checking.";
    }

    FileMapping fm;
    const string full_path = string(model_file_dir) + "/" + *location;
    auto status = fm.Init(full_path.c_str(), FileMapping::READ, offset, length);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "mapping file[" << *location << "] in dir[" << model_file_dir
                   << "] error: " << fm.GetErrorMessage();
        return status;
    }

    data->assign(fm.GetData(), fm.GetSize());
    return RC_SUCCESS;
}

static RetCode LoadInternalData(const ::onnx::TensorProto& pb_tensor, datatype_t ppl_data_type, string* data) {
    const int32_t onnx_data_type = pb_tensor.data_type();
    const uint32_t elem_size = GetSizeOfDataType(ppl_data_type);
    if (onnx_data_type == ::onnx::TensorProto_DataType_FLOAT) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.float_data_size() > 0) {
            data->assign((const char*)pb_tensor.float_data().data(), pb_tensor.float_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_DOUBLE) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.double_data_size() > 0) {
            data->assign((const char*)pb_tensor.double_data().data(), pb_tensor.double_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_INT32 ||
               onnx_data_type == ::onnx::TensorProto_DataType_UINT32) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int32_data_size() > 0) {
            data->assign((const char*)pb_tensor.int32_data().data(), pb_tensor.int32_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_INT64) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int64_data().size() > 0) {
            data->assign((const char*)pb_tensor.int64_data().data(), pb_tensor.int64_data().size() * elem_size);
        }
    } else if (onnx_data_type == ::onnx::TensorProto_DataType_INT8 ||
               onnx_data_type == ::onnx::TensorProto_DataType_UINT8) {
        if (!pb_tensor.raw_data().empty()) {
            *data = pb_tensor.raw_data();
        } else if (pb_tensor.int32_data().size() > 0) { // int8/uint8 may be stored in `int32_data`
            data->assign((const char*)pb_tensor.int32_data().data(), pb_tensor.int32_data().size());
        }
    } else {
        auto onnx_pb_type = (::onnx::TensorProto_DataType)onnx_data_type;
        LOG(ERROR) << "unsupported onnx data type[" << ::onnx::TensorProto_DataType_Name(onnx_pb_type) << "] of tensor["
                   << pb_tensor.name() << "]";
        return RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

RetCode ParseTensorProto(const ::onnx::TensorProto& pb_tensor, const char* model_file_dir, string* data,
                         ir::Shape* shape) {
    const int32_t onnx_data_type = pb_tensor.data_type();
    const datatype_t ppl_data_type = utils::ConvertOnnxDataTypeToPplDataType(onnx_data_type);

    shape->data_type = ppl_data_type;
    shape->data_format = DATAFORMAT_NDARRAY; // default data format
    for (int j = 0; j < pb_tensor.dims_size(); ++j) {
        auto dim = pb_tensor.dims(j);
        shape->dims.push_back(dim);
    }

    RetCode status;
    if (pb_tensor.data_location() == ::onnx::TensorProto_DataLocation_EXTERNAL) {
        status = LoadExternalData(pb_tensor, model_file_dir, data);
    } else {
        status = LoadInternalData(pb_tensor, ppl_data_type, data);
    }

    return status;
}

RetCode PackTensorProto(const void* data, uint64_t len, datatype_t data_type, const vector<int64_t>& dims,
                        ::onnx::TensorProto* pb_tensor) {
    pb_tensor->set_data_type(utils::ConvertPplDataTypeToOnnxDataType(data_type));

    if (data_type == DATATYPE_FLOAT32) {
        auto base = (const float*)data;
        const uint32_t nr = len / GetSizeOfDataType(data_type);
        for (uint32_t i = 0; i < nr; ++i) {
            pb_tensor->add_float_data(base[i]);
        }
    } else if (data_type == DATATYPE_INT32) {
        auto base = (const int32_t*)data;
        const uint32_t nr = len / GetSizeOfDataType(data_type);
        for (uint32_t i = 0; i < nr; ++i) {
            pb_tensor->add_int32_data(base[i]);
        }
    } else if (data_type == DATATYPE_INT64) {
        auto base = (const int64_t*)data;
        const uint32_t nr = len / GetSizeOfDataType(data_type);
        for (uint32_t i = 0; i < nr; ++i) {
            pb_tensor->add_int64_data(base[i]);
        }
    } else {
        LOG(ERROR) << "unsupported data type[" << GetDataTypeStr(data_type) << "]";
        return RC_UNSUPPORTED;
    }

    for (auto x = dims.begin(); x != dims.end(); ++x) {
        pb_tensor->add_dims(*x);
    }

    return RC_SUCCESS;
}

ir::Edge* AddNewInitializer(ir::GraphTopo* topo, ir::GraphData* data, const string& key, const void* value, uint64_t valuelen) {
    auto edge_ret = topo->AddEdge(key);
    if (!edge_ret.first) {
        LOG(ERROR) << "add new initializer[" << key << "] failed.";
        return nullptr;
    }
    auto edge = edge_ret.first;
    auto eid = edge->GetId();

    topo->MarkAsConstant(eid);

    auto constant_ret = data->constants.insert(make_pair(eid, ir::Constant()));
    constant_ret.first->second.data.assign((const char*)value, valuelen);

    auto shape_ret = data->shapes.insert(make_pair(eid, ir::Shape()));
    shape_ret.first->second.data_type = DATATYPE_FLOAT32;
    shape_ret.first->second.data_format = DATAFORMAT_NDARRAY;
    shape_ret.first->second.dims.push_back(1);

    return edge;
}

void ResolveExtraInputs(ir::GraphTopo* current, ir::Node* parent_node, ir::GraphTopo* parent_graph) {
    for (uint32_t i = 0; i < current->GetExtraInputCount(); ++i) {
        auto edge = current->GetEdge(current->GetExtraInput(i));

        auto ret_pair = parent_graph->AddEdge(edge->GetName());
        auto edge_of_parent = ret_pair.first;

        if (ret_pair.second) {
            parent_graph->MarkAsExtraInput(edge_of_parent->GetId());
        }

        edge_of_parent->AddConsumer(parent_node->GetId());
        parent_node->AddExtraInput(edge_of_parent->GetId());
    }
}

}}}} // namespace ppl::nn::onnx::utils
