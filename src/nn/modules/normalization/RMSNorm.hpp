#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class RMSNorm : public Module {
public:
    // GGML supports only last dimension reduction norm
    // weight: optional pre-built weight parameter (e.g. an
    // PermutedParameter); a plain Parameter is created when null.
    RMSNorm(int64_t dim, float eps = 1e-5f, bool elementwise_affine = true, std::shared_ptr<Parameter> weight = nullptr);

    Tensor forward(Scope scope, Tensor x);

private:
    float eps_;
    bool elementwise_affine_;
};
