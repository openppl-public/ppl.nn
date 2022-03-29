// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CUDAOPPARAMS_PPL_NN_PMX_CUDA_H_
#define FLATBUFFERS_GENERATED_CUDAOPPARAMS_PPL_NN_PMX_CUDA_H_

#include "flatbuffers/flatbuffers.h"

namespace ppl {
namespace nn {
namespace pmx {
namespace cuda {

struct FuseAttrs;

struct ConvAlgoInfo;
struct ConvAlgoInfoBuilder;

struct ConvFusionInfo;
struct ConvFusionInfoBuilder;

struct ConvParam;
struct ConvParamBuilder;

struct OpParam;
struct OpParamBuilder;

enum OpParamType : uint8_t {
  OpParamType_NONE = 0,
  OpParamType_ConvParam = 1,
  OpParamType_MIN = OpParamType_NONE,
  OpParamType_MAX = OpParamType_ConvParam
};

inline const OpParamType (&EnumValuesOpParamType())[2] {
  static const OpParamType values[] = {
    OpParamType_NONE,
    OpParamType_ConvParam
  };
  return values;
}

inline const char * const *EnumNamesOpParamType() {
  static const char * const names[3] = {
    "NONE",
    "ConvParam",
    nullptr
  };
  return names;
}

inline const char *EnumNameOpParamType(OpParamType e) {
  if (flatbuffers::IsOutRange(e, OpParamType_NONE, OpParamType_ConvParam)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesOpParamType()[index];
}

template<typename T> struct OpParamTypeTraits {
  static const OpParamType enum_value = OpParamType_NONE;
};

template<> struct OpParamTypeTraits<ppl::nn::pmx::cuda::ConvParam> {
  static const OpParamType enum_value = OpParamType_ConvParam;
};

bool VerifyOpParamType(flatbuffers::Verifier &verifier, const void *obj, OpParamType type);
bool VerifyOpParamTypeVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types);

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) FuseAttrs FLATBUFFERS_FINAL_CLASS {
 private:
  float clip_min_;
  float clip_max_;
  float leaky_alpha_;

 public:
  FuseAttrs()
      : clip_min_(0),
        clip_max_(0),
        leaky_alpha_(0) {
  }
  FuseAttrs(float _clip_min, float _clip_max, float _leaky_alpha)
      : clip_min_(flatbuffers::EndianScalar(_clip_min)),
        clip_max_(flatbuffers::EndianScalar(_clip_max)),
        leaky_alpha_(flatbuffers::EndianScalar(_leaky_alpha)) {
  }
  float clip_min() const {
    return flatbuffers::EndianScalar(clip_min_);
  }
  float clip_max() const {
    return flatbuffers::EndianScalar(clip_max_);
  }
  float leaky_alpha() const {
    return flatbuffers::EndianScalar(leaky_alpha_);
  }
};
FLATBUFFERS_STRUCT_END(FuseAttrs, 12);

struct ConvAlgoInfo FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConvAlgoInfoBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ALGO_TYPE = 4,
    VT_ALGO_NAME = 6,
    VT_TILES = 8,
    VT_KID = 10,
    VT_SPLITK = 12,
    VT_SPLITF = 14,
    VT_IS_INITIALIZER_WEIGHT = 16,
    VT_HAS_BIAS = 18
  };
  const flatbuffers::String *algo_type() const {
    return GetPointer<const flatbuffers::String *>(VT_ALGO_TYPE);
  }
  const flatbuffers::String *algo_name() const {
    return GetPointer<const flatbuffers::String *>(VT_ALGO_NAME);
  }
  const flatbuffers::Vector<int32_t> *tiles() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_TILES);
  }
  int32_t kid() const {
    return GetField<int32_t>(VT_KID, 0);
  }
  int32_t splitk() const {
    return GetField<int32_t>(VT_SPLITK, 1);
  }
  int32_t splitf() const {
    return GetField<int32_t>(VT_SPLITF, 1);
  }
  int32_t is_initializer_weight() const {
    return GetField<int32_t>(VT_IS_INITIALIZER_WEIGHT, 1);
  }
  int32_t has_bias() const {
    return GetField<int32_t>(VT_HAS_BIAS, 1);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ALGO_TYPE) &&
           verifier.VerifyString(algo_type()) &&
           VerifyOffset(verifier, VT_ALGO_NAME) &&
           verifier.VerifyString(algo_name()) &&
           VerifyOffset(verifier, VT_TILES) &&
           verifier.VerifyVector(tiles()) &&
           VerifyField<int32_t>(verifier, VT_KID) &&
           VerifyField<int32_t>(verifier, VT_SPLITK) &&
           VerifyField<int32_t>(verifier, VT_SPLITF) &&
           VerifyField<int32_t>(verifier, VT_IS_INITIALIZER_WEIGHT) &&
           VerifyField<int32_t>(verifier, VT_HAS_BIAS) &&
           verifier.EndTable();
  }
};

struct ConvAlgoInfoBuilder {
  typedef ConvAlgoInfo Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_algo_type(flatbuffers::Offset<flatbuffers::String> algo_type) {
    fbb_.AddOffset(ConvAlgoInfo::VT_ALGO_TYPE, algo_type);
  }
  void add_algo_name(flatbuffers::Offset<flatbuffers::String> algo_name) {
    fbb_.AddOffset(ConvAlgoInfo::VT_ALGO_NAME, algo_name);
  }
  void add_tiles(flatbuffers::Offset<flatbuffers::Vector<int32_t>> tiles) {
    fbb_.AddOffset(ConvAlgoInfo::VT_TILES, tiles);
  }
  void add_kid(int32_t kid) {
    fbb_.AddElement<int32_t>(ConvAlgoInfo::VT_KID, kid, 0);
  }
  void add_splitk(int32_t splitk) {
    fbb_.AddElement<int32_t>(ConvAlgoInfo::VT_SPLITK, splitk, 1);
  }
  void add_splitf(int32_t splitf) {
    fbb_.AddElement<int32_t>(ConvAlgoInfo::VT_SPLITF, splitf, 1);
  }
  void add_is_initializer_weight(int32_t is_initializer_weight) {
    fbb_.AddElement<int32_t>(ConvAlgoInfo::VT_IS_INITIALIZER_WEIGHT, is_initializer_weight, 1);
  }
  void add_has_bias(int32_t has_bias) {
    fbb_.AddElement<int32_t>(ConvAlgoInfo::VT_HAS_BIAS, has_bias, 1);
  }
  explicit ConvAlgoInfoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ConvAlgoInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConvAlgoInfo>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConvAlgoInfo> CreateConvAlgoInfo(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> algo_type = 0,
    flatbuffers::Offset<flatbuffers::String> algo_name = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> tiles = 0,
    int32_t kid = 0,
    int32_t splitk = 1,
    int32_t splitf = 1,
    int32_t is_initializer_weight = 1,
    int32_t has_bias = 1) {
  ConvAlgoInfoBuilder builder_(_fbb);
  builder_.add_has_bias(has_bias);
  builder_.add_is_initializer_weight(is_initializer_weight);
  builder_.add_splitf(splitf);
  builder_.add_splitk(splitk);
  builder_.add_kid(kid);
  builder_.add_tiles(tiles);
  builder_.add_algo_name(algo_name);
  builder_.add_algo_type(algo_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<ConvAlgoInfo> CreateConvAlgoInfoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *algo_type = nullptr,
    const char *algo_name = nullptr,
    const std::vector<int32_t> *tiles = nullptr,
    int32_t kid = 0,
    int32_t splitk = 1,
    int32_t splitf = 1,
    int32_t is_initializer_weight = 1,
    int32_t has_bias = 1) {
  auto algo_type__ = algo_type ? _fbb.CreateString(algo_type) : 0;
  auto algo_name__ = algo_name ? _fbb.CreateString(algo_name) : 0;
  auto tiles__ = tiles ? _fbb.CreateVector<int32_t>(*tiles) : 0;
  return ppl::nn::pmx::cuda::CreateConvAlgoInfo(
      _fbb,
      algo_type__,
      algo_name__,
      tiles__,
      kid,
      splitk,
      splitf,
      is_initializer_weight,
      has_bias);
}

struct ConvFusionInfo FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConvFusionInfoBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_TYPES = 4,
    VT_INPUT_INDS = 6,
    VT_FUSE_ATTRS = 8,
    VT_CHANNEL_SIZE = 10,
    VT_CHANNEL_OFFSET = 12,
    VT_CONCAT_EDGE_ID = 14
  };
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *types() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_TYPES);
  }
  const flatbuffers::Vector<int32_t> *input_inds() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_INPUT_INDS);
  }
  const flatbuffers::Vector<const ppl::nn::pmx::cuda::FuseAttrs *> *fuse_attrs() const {
    return GetPointer<const flatbuffers::Vector<const ppl::nn::pmx::cuda::FuseAttrs *> *>(VT_FUSE_ATTRS);
  }
  int32_t channel_size() const {
    return GetField<int32_t>(VT_CHANNEL_SIZE, -1);
  }
  int32_t channel_offset() const {
    return GetField<int32_t>(VT_CHANNEL_OFFSET, -1);
  }
  int32_t concat_edge_id() const {
    return GetField<int32_t>(VT_CONCAT_EDGE_ID, -1);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_TYPES) &&
           verifier.VerifyVector(types()) &&
           verifier.VerifyVectorOfStrings(types()) &&
           VerifyOffset(verifier, VT_INPUT_INDS) &&
           verifier.VerifyVector(input_inds()) &&
           VerifyOffset(verifier, VT_FUSE_ATTRS) &&
           verifier.VerifyVector(fuse_attrs()) &&
           VerifyField<int32_t>(verifier, VT_CHANNEL_SIZE) &&
           VerifyField<int32_t>(verifier, VT_CHANNEL_OFFSET) &&
           VerifyField<int32_t>(verifier, VT_CONCAT_EDGE_ID) &&
           verifier.EndTable();
  }
};

struct ConvFusionInfoBuilder {
  typedef ConvFusionInfo Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_types(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> types) {
    fbb_.AddOffset(ConvFusionInfo::VT_TYPES, types);
  }
  void add_input_inds(flatbuffers::Offset<flatbuffers::Vector<int32_t>> input_inds) {
    fbb_.AddOffset(ConvFusionInfo::VT_INPUT_INDS, input_inds);
  }
  void add_fuse_attrs(flatbuffers::Offset<flatbuffers::Vector<const ppl::nn::pmx::cuda::FuseAttrs *>> fuse_attrs) {
    fbb_.AddOffset(ConvFusionInfo::VT_FUSE_ATTRS, fuse_attrs);
  }
  void add_channel_size(int32_t channel_size) {
    fbb_.AddElement<int32_t>(ConvFusionInfo::VT_CHANNEL_SIZE, channel_size, -1);
  }
  void add_channel_offset(int32_t channel_offset) {
    fbb_.AddElement<int32_t>(ConvFusionInfo::VT_CHANNEL_OFFSET, channel_offset, -1);
  }
  void add_concat_edge_id(int32_t concat_edge_id) {
    fbb_.AddElement<int32_t>(ConvFusionInfo::VT_CONCAT_EDGE_ID, concat_edge_id, -1);
  }
  explicit ConvFusionInfoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ConvFusionInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConvFusionInfo>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConvFusionInfo> CreateConvFusionInfo(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> types = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> input_inds = 0,
    flatbuffers::Offset<flatbuffers::Vector<const ppl::nn::pmx::cuda::FuseAttrs *>> fuse_attrs = 0,
    int32_t channel_size = -1,
    int32_t channel_offset = -1,
    int32_t concat_edge_id = -1) {
  ConvFusionInfoBuilder builder_(_fbb);
  builder_.add_concat_edge_id(concat_edge_id);
  builder_.add_channel_offset(channel_offset);
  builder_.add_channel_size(channel_size);
  builder_.add_fuse_attrs(fuse_attrs);
  builder_.add_input_inds(input_inds);
  builder_.add_types(types);
  return builder_.Finish();
}

inline flatbuffers::Offset<ConvFusionInfo> CreateConvFusionInfoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *types = nullptr,
    const std::vector<int32_t> *input_inds = nullptr,
    const std::vector<ppl::nn::pmx::cuda::FuseAttrs> *fuse_attrs = nullptr,
    int32_t channel_size = -1,
    int32_t channel_offset = -1,
    int32_t concat_edge_id = -1) {
  auto types__ = types ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*types) : 0;
  auto input_inds__ = input_inds ? _fbb.CreateVector<int32_t>(*input_inds) : 0;
  auto fuse_attrs__ = fuse_attrs ? _fbb.CreateVectorOfStructs<ppl::nn::pmx::cuda::FuseAttrs>(*fuse_attrs) : 0;
  return ppl::nn::pmx::cuda::CreateConvFusionInfo(
      _fbb,
      types__,
      input_inds__,
      fuse_attrs__,
      channel_size,
      channel_offset,
      concat_edge_id);
}

struct ConvParam FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConvParamBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ALGO_INFO = 4,
    VT_FUSE_INFO = 6,
    VT_GENE_CODE = 8
  };
  const ppl::nn::pmx::cuda::ConvAlgoInfo *algo_info() const {
    return GetPointer<const ppl::nn::pmx::cuda::ConvAlgoInfo *>(VT_ALGO_INFO);
  }
  const ppl::nn::pmx::cuda::ConvFusionInfo *fuse_info() const {
    return GetPointer<const ppl::nn::pmx::cuda::ConvFusionInfo *>(VT_FUSE_INFO);
  }
  const flatbuffers::Vector<uint8_t> *gene_code() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_GENE_CODE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ALGO_INFO) &&
           verifier.VerifyTable(algo_info()) &&
           VerifyOffset(verifier, VT_FUSE_INFO) &&
           verifier.VerifyTable(fuse_info()) &&
           VerifyOffset(verifier, VT_GENE_CODE) &&
           verifier.VerifyVector(gene_code()) &&
           verifier.EndTable();
  }
};

struct ConvParamBuilder {
  typedef ConvParam Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_algo_info(flatbuffers::Offset<ppl::nn::pmx::cuda::ConvAlgoInfo> algo_info) {
    fbb_.AddOffset(ConvParam::VT_ALGO_INFO, algo_info);
  }
  void add_fuse_info(flatbuffers::Offset<ppl::nn::pmx::cuda::ConvFusionInfo> fuse_info) {
    fbb_.AddOffset(ConvParam::VT_FUSE_INFO, fuse_info);
  }
  void add_gene_code(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> gene_code) {
    fbb_.AddOffset(ConvParam::VT_GENE_CODE, gene_code);
  }
  explicit ConvParamBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ConvParam> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConvParam>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConvParam> CreateConvParam(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<ppl::nn::pmx::cuda::ConvAlgoInfo> algo_info = 0,
    flatbuffers::Offset<ppl::nn::pmx::cuda::ConvFusionInfo> fuse_info = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> gene_code = 0) {
  ConvParamBuilder builder_(_fbb);
  builder_.add_gene_code(gene_code);
  builder_.add_fuse_info(fuse_info);
  builder_.add_algo_info(algo_info);
  return builder_.Finish();
}

inline flatbuffers::Offset<ConvParam> CreateConvParamDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<ppl::nn::pmx::cuda::ConvAlgoInfo> algo_info = 0,
    flatbuffers::Offset<ppl::nn::pmx::cuda::ConvFusionInfo> fuse_info = 0,
    const std::vector<uint8_t> *gene_code = nullptr) {
  auto gene_code__ = gene_code ? _fbb.CreateVector<uint8_t>(*gene_code) : 0;
  return ppl::nn::pmx::cuda::CreateConvParam(
      _fbb,
      algo_info,
      fuse_info,
      gene_code__);
}

struct OpParam FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef OpParamBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VALUE_TYPE = 4,
    VT_VALUE = 6
  };
  ppl::nn::pmx::cuda::OpParamType value_type() const {
    return static_cast<ppl::nn::pmx::cuda::OpParamType>(GetField<uint8_t>(VT_VALUE_TYPE, 0));
  }
  const void *value() const {
    return GetPointer<const void *>(VT_VALUE);
  }
  template<typename T> const T *value_as() const;
  const ppl::nn::pmx::cuda::ConvParam *value_as_ConvParam() const {
    return value_type() == ppl::nn::pmx::cuda::OpParamType_ConvParam ? static_cast<const ppl::nn::pmx::cuda::ConvParam *>(value()) : nullptr;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_VALUE_TYPE) &&
           VerifyOffset(verifier, VT_VALUE) &&
           VerifyOpParamType(verifier, value(), value_type()) &&
           verifier.EndTable();
  }
};

template<> inline const ppl::nn::pmx::cuda::ConvParam *OpParam::value_as<ppl::nn::pmx::cuda::ConvParam>() const {
  return value_as_ConvParam();
}

struct OpParamBuilder {
  typedef OpParam Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_value_type(ppl::nn::pmx::cuda::OpParamType value_type) {
    fbb_.AddElement<uint8_t>(OpParam::VT_VALUE_TYPE, static_cast<uint8_t>(value_type), 0);
  }
  void add_value(flatbuffers::Offset<void> value) {
    fbb_.AddOffset(OpParam::VT_VALUE, value);
  }
  explicit OpParamBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<OpParam> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<OpParam>(end);
    return o;
  }
};

inline flatbuffers::Offset<OpParam> CreateOpParam(
    flatbuffers::FlatBufferBuilder &_fbb,
    ppl::nn::pmx::cuda::OpParamType value_type = ppl::nn::pmx::cuda::OpParamType_NONE,
    flatbuffers::Offset<void> value = 0) {
  OpParamBuilder builder_(_fbb);
  builder_.add_value(value);
  builder_.add_value_type(value_type);
  return builder_.Finish();
}

inline bool VerifyOpParamType(flatbuffers::Verifier &verifier, const void *obj, OpParamType type) {
  switch (type) {
    case OpParamType_NONE: {
      return true;
    }
    case OpParamType_ConvParam: {
      auto ptr = reinterpret_cast<const ppl::nn::pmx::cuda::ConvParam *>(obj);
      return verifier.VerifyTable(ptr);
    }
    default: return true;
  }
}

inline bool VerifyOpParamTypeVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types) {
  if (!values || !types) return !values && !types;
  if (values->size() != types->size()) return false;
  for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
    if (!VerifyOpParamType(
        verifier,  values->Get(i), types->GetEnum<OpParamType>(i))) {
      return false;
    }
  }
  return true;
}

inline const ppl::nn::pmx::cuda::OpParam *GetOpParam(const void *buf) {
  return flatbuffers::GetRoot<ppl::nn::pmx::cuda::OpParam>(buf);
}

inline const ppl::nn::pmx::cuda::OpParam *GetSizePrefixedOpParam(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<ppl::nn::pmx::cuda::OpParam>(buf);
}

inline bool VerifyOpParamBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<ppl::nn::pmx::cuda::OpParam>(nullptr);
}

inline bool VerifySizePrefixedOpParamBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<ppl::nn::pmx::cuda::OpParam>(nullptr);
}

inline void FinishOpParamBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<ppl::nn::pmx::cuda::OpParam> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedOpParamBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<ppl::nn::pmx::cuda::OpParam> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace cuda
}  // namespace pmx
}  // namespace nn
}  // namespace ppl

#endif  // FLATBUFFERS_GENERATED_CUDAOPPARAMS_PPL_NN_PMX_CUDA_H_
