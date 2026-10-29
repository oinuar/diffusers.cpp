#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"

class Flux2FusedAttentionOutput : public Module {
public:
    Flux2FusedAttentionOutput(
        int64_t inner_dim,
        int64_t mlp_hidden_dim,
        int64_t out_dim,
        bool bias)
        :
        out_dim_(out_dim),
        inner_dim_(inner_dim),
        mlp_hidden_dim_(mlp_hidden_dim)
    {
        // Sharding patch for the reference's fused to_out Linear:
        // the reference fuses the attention output + MLP output
        // projections into one Linear over the concatenated inputs; we
        // keep them separate and add the outputs: the attention
        // output leaves flash_attn_ext heads-sharded (S(0) once
        // flattened) and must go through a row-parallel projection
        // (weight S(0) + activation S(0) -> PARTIAL -> AllReduce -> R)
        // -- the meta backend forbids a concat along the sharded axis,
        // so the fused form (concat + one Linear) is infeasible there.
        // By linearity of the matmul, W @ [attn | mlp] == W_attn @ attn + W_mlp @ mlp.
        modules["attn"] = std::make_shared<Linear>(inner_dim_, out_dim_, bias);
        modules["mlp"] = std::make_shared<Linear>(mlp_hidden_dim_, out_dim_, false);
    }

    Tensor forward(Scope scope, Tensor hidden_states, Tensor mlp_hidden_states) {
        hidden_states = attn()->forward(scope, hidden_states);
        hidden_states = hidden_states + mlp()->forward(scope, mlp_hidden_states);

        return hidden_states;
    }

    int64_t inner_dim() const {
        return inner_dim_;
    }

    int64_t mlp_hidden_dim() const {
        return mlp_hidden_dim_;
    }

    std::shared_ptr<Linear> attn() const {
        return std::static_pointer_cast<Linear>(modules.at("attn"));
    }

    std::shared_ptr<Linear> mlp() const {
        return std::static_pointer_cast<Linear>(modules.at("mlp"));
    }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    int64_t out_dim_;
    int64_t inner_dim_;
    int64_t mlp_hidden_dim_;
};
