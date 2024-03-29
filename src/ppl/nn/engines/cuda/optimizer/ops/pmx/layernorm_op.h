#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_CAFFE_LAYERNORM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_CAFFE_LAYERNORM_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/params/pmx/layer_norm_param.h"

namespace ppl { namespace nn { namespace cuda {

class LayerNormOp final : public CudaOptKernel {
public:
    LayerNormOp(const ir::Node* node);
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode Init(const OptKernelOptions&) override;
    ppl::common::RetCode Finalize(const OptKernelOptions& options) override;
    void* GetParam() override {
        return (void*)&param_;
    };

private:
    ppl::nn::pmx::LayerNormParam param_;
};

}}} // namespace ppl::nn::cuda

#endif
