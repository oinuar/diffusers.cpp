#pragma once

#include "nn/Parameter.hpp"

// A weight whose loaded values are permuted. The permutation happens
// once at load time.
class PermutedParameter : public Parameter {
public:
    enum class Axis {
        Rows,      // 2-D weight: output row j = input row perm[j]
        Elements   // 1-D weight: element j = element perm[j]
    };

    PermutedParameter(
        const Tensor::Shape& shape,
        Axis axis,
        std::vector<size_t> perm,
        const ggml_type& dtype = Tensor::DType<float>::value)
        : Parameter(shape, dtype), axis_(axis), perm_(std::move(perm))
    {
    }

    Axis axis() const {
        return axis_;
    }

    const std::vector<size_t>& perm() const {
        return perm_;
    }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    Axis axis_;
    std::vector<size_t> perm_;
};
