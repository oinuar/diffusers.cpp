#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"

class Flux2FusedLinearIn : public Module {
public:
    Flux2FusedLinearIn(
        int64_t dim,
        int64_t inner_dim,
        bool bias
    ) :
        dim_(dim),
        inner_dim_(inner_dim)
    {
        // Sharding patch for the reference Flux2FeedForward's fused
        // linear_in Linear (dim -> inner * 2, whose output the SwiGLU
        // chunks into (gate | up)): we keep gate and up as separate
        // projections. The math is identical: row-slicing a projection
        // is the same projection. The split is shardable: on the meta
        // device the fused output cannot be split, because a windowed
        // view of a column-parallel (S(0)) tensor cannot carry the
        // shard (each device holds a different half of the features,
        // so the SwiGLU product would be cross-device). With separate
        // gate/up projections the meta plans them column-parallel
        // (weight S(1) -> output S(0)), the SwiGLU product stays S(0),
        // and the feedforward's output projection can run row-parallel
        // (weight S(0) + activation S(0) -> PARTIAL -> AllReduce -> R)
        // instead of fully replicated.
        modules["gate"] = std::make_shared<Linear>(dim, inner_dim, bias);
        modules["up"] = std::make_shared<Linear>(dim, inner_dim, bias);
    }

    std::tuple<Tensor, Tensor> forward(Scope scope, Tensor x) {
        auto gate = this->gate()->forward(scope, x);
        auto up = this->up()->forward(scope, x);

        return {gate, up};
    }

    int64_t dim() const {
        return dim_;
    }

    int64_t inner_dim() const {
        return inner_dim_;
    }

    std::shared_ptr<Linear> gate() const {
        return std::static_pointer_cast<Linear>(modules.at("gate"));
    }

    std::shared_ptr<Linear> up() const {
        return std::static_pointer_cast<Linear>(modules.at("up"));
    }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    int64_t dim_;
    int64_t inner_dim_;
};
