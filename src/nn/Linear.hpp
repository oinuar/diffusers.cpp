#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class Linear : public Module {
public:
    // weight: optional pre-built weight parameter (e.g. an
    // PermutedParameter); a plain Parameter is created when null.
    Linear(int64_t in_features, int64_t out_features, bool bias = true, std::shared_ptr<Parameter> weight = nullptr);
    
    Tensor forward(Scope scope, Tensor x);

    std::shared_ptr<Parameter> weight() const;

    std::shared_ptr<Parameter> bias() const;

private:
    bool bias_;
};
