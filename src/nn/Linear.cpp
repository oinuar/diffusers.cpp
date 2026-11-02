#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

Linear::Linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    std::shared_ptr<Parameter> weight
) : bias_(bias)
{
    modules["weight"] = weight ? std::move(weight) : std::make_shared<Parameter>(Tensor::Shape({out_features, in_features}));

    if (bias_)
        modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({out_features}));
}

Tensor Linear::forward(Scope scope, Tensor x) {
    auto w = weight()->forward(scope);

    // Weight is logically shaped [out_features, in_features] (PyTorch),
    // but stored in GGML's native reversed layout {in_features, out_features}.
    // Since ggml_mul_mat() already performs Aᵀ * B on its first operand,
    // explicitly transposing the weight would transpose it twice.
    auto y = scope.runtime().mul_mat(*w, *x);

    if (bias_) {
        auto b = bias()->forward(scope);

        y = scope.runtime().add(y, *b);
    }

    Tensor::Shape shape = x.shape();
    shape[shape.rank() - 1] = w.shape()[0];

    return Tensor(y, shape);
}

std::shared_ptr<Parameter> Linear::weight() const {
    return std::static_pointer_cast<Parameter>(modules.at("weight"));
}

std::shared_ptr<Parameter> Linear::bias() const {
    if (bias_)
        return std::static_pointer_cast<Parameter>(modules.at("bias"));

    return nullptr;
}
