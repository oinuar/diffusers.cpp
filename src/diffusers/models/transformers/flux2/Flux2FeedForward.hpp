#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Runtime.hpp"
#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"

class Flux2FeedForward : public Module {
private:
    Flux2FeedForward(
        int64_t dim,
        int64_t dim_out,
        float mult,
        int64_t inner_dim,
        bool bias
    ) :
        sharded_(Scope::active() && Scope::runtime().use_sharding())
    {
        if (sharded_) {
            // Sharding patch: the reference's fused linear_in Linear
            // is replaced by split gate/up projections whose weight/bias
            // Parameters carry split configurations (see
            // Parameter::Split): the loaders take each slice from the
            // fused checkpoint tensors.
            auto gate = std::make_shared<Linear>(dim, inner_dim, bias);

            gate->weight()->split_rows({"linear_in", "weight"}, 0, inner_dim);

            if (bias)
                gate->bias()->split_cols({"linear_in", "bias"}, 0, inner_dim);

            modules["gate"] = gate;

            auto up = std::make_shared<Linear>(dim, inner_dim, bias);

            up->weight()->split_rows({"linear_in", "weight"}, inner_dim, inner_dim * 2);

            if (bias)
                up->bias()->split_cols({"linear_in", "bias"}, inner_dim, inner_dim * 2);

            modules["up"] = up;
        } else {
            // Reference form: one fused projection over (gate | up)
            // features; the SwiGLU chunks its output.
            modules["linear_in"] = std::make_shared<Linear>(dim, inner_dim * 2, bias);
        }

        modules["act_fn"] = std::make_shared<Flux2SwiGLU>();
        modules["linear_out"] = std::make_shared<Linear>(inner_dim, dim_out, bias);
    }

public:
    Flux2FeedForward(
        int64_t dim,
        std::optional<int64_t> dim_out = std::nullopt,
        float mult = 3.0,
        std::optional<int64_t> inner_dim = std::nullopt,
        bool bias = false
    ) :
        Flux2FeedForward(
            dim,
            dim_out.value_or(dim),
            mult,
            inner_dim.value_or(dim * mult),
            bias
        )
    {
    }

    Tensor forward(Scope scope, Tensor x) {
        if (sharded_ != scope.runtime().use_sharding())
            throw std::runtime_error(
                "Flux2FeedForward: built for " + std::string(sharded_ ? "sharded" : "non-sharded") +
                " execution but the active scope is " + std::string(scope.runtime().use_sharding() ? "sharded" : "non-sharded"));

        auto act_fn = std::static_pointer_cast<Flux2SwiGLU>(modules["act_fn"]);
        auto linear_out = std::static_pointer_cast<Linear>(modules["linear_out"]);

        if (sharded_) {
            // Sharding patch: the reference's fused linear_in is
            // replaced by split gate/up projections (see the split
            // Parameter configurations in the constructor), so the
            // reference's chunk of its output does not apply.
            auto gate = std::static_pointer_cast<Linear>(modules["gate"]);
            auto up = std::static_pointer_cast<Linear>(modules["up"]);

            x = act_fn->forward(scope, gate->forward(scope, x), up->forward(scope, x));
        } else {
            auto linear_in = std::static_pointer_cast<Linear>(modules["linear_in"]);

            x = linear_in->forward(scope, x);
            x = act_fn->forward(scope, x);
        }

        x = linear_out->forward(scope, x);

        return x;
    }

private:
    bool sharded_;
};
