#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"
#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"
#include "diffusers/models/transformers/flux2/Flux2FusedQKVProjection.hpp"
#include "diffusers/models/transformers/flux2/Flux2FusedAttentionOutput.hpp"

template <class AttnOp>
class Flux2ParallelSelfAttention : public Module {
public:
    Flux2ParallelSelfAttention(
        int64_t query_dim,
        int64_t heads = 8,
        int64_t dim_head = 64,
        float dropout = 0.0,
        bool bias = false,
        bool out_bias = true,
        float eps = 1e-5,
        std::optional<int64_t> out_dim = std::nullopt,
        bool elementwise_affine = true,
        float mlp_ratio = 4.0,
        int64_t mlp_mult_factor = 2)
    {
        sharded_ = Scope::sharded();

        head_dim_ = dim_head;
        inner_dim_ = out_dim.value_or(dim_head * heads);
        query_dim_ = query_dim;
        out_dim_ = out_dim.value_or(query_dim);
        heads_ = out_dim ? *out_dim / dim_head : heads;

        use_bias_ = bias;
        dropout_ = dropout;

        mlp_ratio_ = mlp_ratio;
        mlp_hidden_dim_ = int(query_dim * mlp_ratio_);
        mlp_mult_factor_ = mlp_mult_factor;

        // Fused QKV projections + MLP input projection
        if (sharded_) {
            // Sharding patch: the reference's fused Linear is replaced
            // by split-projection modules (see Flux2FusedQKVProjection
            // / Flux2FusedAttentionOutput).
            modules["to_qkv_mlp_proj"] = std::make_shared<Flux2FusedQKVProjection>(query_dim_, inner_dim_, mlp_hidden_dim_, mlp_mult_factor, bias);
            modules["to_out"] = std::make_shared<Flux2FusedAttentionOutput>(inner_dim_, mlp_hidden_dim_, out_dim_, out_bias);
        } else {
            // Reference form: the reference fuses the QKV + MLP input
            // projections into one Linear, and the attention output +
            // MLP output projections into another.
            auto mlp_out = static_cast<int64_t>(mlp_hidden_dim_ * mlp_mult_factor_);

            modules["to_qkv_mlp_proj"] = std::make_shared<Linear>(query_dim_, 3 * inner_dim_ + mlp_out, bias);
            modules["to_out"] = std::make_shared<Linear>(inner_dim_ + mlp_hidden_dim_, out_dim_, out_bias);
        }

        modules["mlp_act_fn"] = std::make_shared<Flux2SwiGLU>();

        // QK Norm
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
    }

    Tensor forward(
        Scope scope,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::pair<std::shared_ptr<Flux2PosEmbed>, Tensor>> image_rotary_emb = std::nullopt
    ) {
        if (sharded_ != Scope::sharded())
            throw std::runtime_error(
                "Flux2ParallelSelfAttention: built for " + std::string(sharded_ ? "sharded" : "reference") +
                " execution but the active scope is " + std::string(Scope::sharded() ? "sharded" : "reference"));

        Tensor query, key, value;
        Tensor mlp_hidden_states;
        std::optional<Tensor> mlp_gate;
        std::optional<Tensor> mlp_up;

        if (sharded_) {
            // Sharding patch: the reference's fused projection is
            // replaced by split row-parallel projections (see
            // Flux2FusedQKVProjection), so the reference's split/chunk
            // of its output does not apply.
            auto to_qkv_mlp_proj = std::static_pointer_cast<Flux2FusedQKVProjection>(modules["to_qkv_mlp_proj"]);

            auto [q, k, v, mlp_gate_, mlp_up_] = to_qkv_mlp_proj->forward(scope, hidden_states);

            query = q;
            key = k;
            value = v;
            mlp_gate = mlp_gate_;
            mlp_up = mlp_up_;
        } else {
            // Parallel in (QKV + MLP in) projection
            auto to_qkv_mlp_proj = std::static_pointer_cast<Linear>(modules["to_qkv_mlp_proj"]);

            hidden_states = to_qkv_mlp_proj->forward(scope, hidden_states);

            auto parts = hidden_states.split_with_sizes({3 * inner_dim_, static_cast<int64_t>(mlp_hidden_dim_ * mlp_mult_factor_)}, -1);

            auto qkv_chunks = parts[0].chunk(3, -1);

            query = qkv_chunks[0];
            key = qkv_chunks[1];
            value = qkv_chunks[2];

            mlp_hidden_states = parts[1];
        }

        query = query.unflatten(-1, {heads_, -1});
        key = key.unflatten(-1, {heads_, -1});
        value = value.unflatten(-1, {heads_, -1});

        auto norm_q = std::static_pointer_cast<RMSNorm>(modules["norm_q"]);
        auto norm_k = std::static_pointer_cast<RMSNorm>(modules["norm_k"]);

        query = norm_q->forward(scope, query);
        key = norm_k->forward(scope, key);

        if (image_rotary_emb) {
            query = image_rotary_emb->first->forward(scope, query, image_rotary_emb->second);
            key = image_rotary_emb->first->forward(scope, key, image_rotary_emb->second);
        }

        AttnOp dispatch_attention_fn;

        hidden_states = dispatch_attention_fn(
            scope,
            query,
            key,
            value,
            attention_mask
        );

        hidden_states = hidden_states.flatten(2, 3);

        auto mlp_act_fn = std::static_pointer_cast<Flux2SwiGLU>(modules["mlp_act_fn"]);

        // Handle the feedforward (FF) logic
        if (sharded_) {
            mlp_hidden_states = mlp_act_fn->forward(scope, *mlp_gate, *mlp_up);

            // Concatenate and parallel output projection
            auto to_out = std::static_pointer_cast<Flux2FusedAttentionOutput>(modules["to_out"]);

            hidden_states = to_out->forward(scope, hidden_states, mlp_hidden_states);
        } else {
            mlp_hidden_states = mlp_act_fn->forward(scope, mlp_hidden_states);

            // Concatenate and parallel output projection
            auto to_out = std::static_pointer_cast<Linear>(modules["to_out"]);

            hidden_states = to_out->forward(scope, Tensor::cat({hidden_states, mlp_hidden_states}, -1));
        }

        return hidden_states;
    }

private:
    int64_t head_dim_;
    int64_t inner_dim_;
    int64_t query_dim_;
    int64_t out_dim_;
    int64_t heads_;

    bool use_bias_;
    float dropout_;

    float mlp_ratio_;
    int64_t mlp_hidden_dim_;
    int64_t mlp_mult_factor_;
    bool sharded_;
};
