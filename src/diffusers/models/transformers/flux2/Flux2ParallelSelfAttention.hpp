#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"
#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"

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
        sharded_ = Scope::active() && Scope::runtime().use_sharding();

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

        //
        // MROPE head layout (see Flux2PosEmbed): the q/k projections
        // and the QK-norm weights carry a permutation configuration
        // (Parameter::Permutation) -- the loader permutes them into the
        // ggml MROPE (NEOX) layout once at load time, so attention runs
        // the whole head dimension through one MROPE op. The q/k biases
        // are permuted the same way as their weights. v, the MLP part
        // and the output projections stay in the original layout (the
        // attention output is invariant under the same permutation of
        // q and k).
        //
        auto head_perm = Flux2PosEmbed::mrope_permutation(head_dim_, heads_);

        // Fused QKV projections + MLP input projection
        if (sharded_) {
            // Sharding patch: the reference's fused to_qkv_mlp_proj and
            // to_out Linears are replaced by separate projections whose
            // weight/bias Parameters carry split configurations (see
            // Parameter::Split): the loaders take each slice from the
            // fused checkpoint tensors.
            auto mlp_out = mlp_hidden_dim_ * mlp_mult_factor_;

            auto q = std::make_shared<Linear>(query_dim_, inner_dim_, bias);

            q->weight()->split_rows({"to_qkv_mlp_proj", "weight"}, 0, inner_dim_);
            q->weight()->permute_rows(head_perm);

            if (bias) {
                q->bias()->split_cols({"to_qkv_mlp_proj", "bias"}, 0, inner_dim_);
                q->bias()->permute_elements(head_perm);
            }

            modules["q"] = q;

            auto k = std::make_shared<Linear>(query_dim_, inner_dim_, bias);

            k->weight()->split_rows({"to_qkv_mlp_proj", "weight"}, inner_dim_, 2 * inner_dim_);
            k->weight()->permute_rows(head_perm);

            if (bias) {
                k->bias()->split_cols({"to_qkv_mlp_proj", "bias"}, inner_dim_, 2 * inner_dim_);
                k->bias()->permute_elements(head_perm);
            }

            modules["k"] = k;

            auto v = std::make_shared<Linear>(query_dim_, inner_dim_, bias);

            v->weight()->split_rows({"to_qkv_mlp_proj", "weight"}, 2 * inner_dim_, 3 * inner_dim_);

            if (bias)
                v->bias()->split_cols({"to_qkv_mlp_proj", "bias"}, 2 * inner_dim_, 3 * inner_dim_);

            modules["v"] = v;

            auto mlp_in_gate = std::make_shared<Linear>(query_dim_, mlp_hidden_dim_, bias);

            mlp_in_gate->weight()->split_rows({"to_qkv_mlp_proj", "weight"}, 3 * inner_dim_, 3 * inner_dim_ + mlp_hidden_dim_);

            if (bias)
                mlp_in_gate->bias()->split_cols({"to_qkv_mlp_proj", "bias"}, 3 * inner_dim_, 3 * inner_dim_ + mlp_hidden_dim_);

            modules["mlp_in_gate"] = mlp_in_gate;

            auto mlp_in_up = std::make_shared<Linear>(query_dim_, mlp_hidden_dim_, bias);

            mlp_in_up->weight()->split_rows({"to_qkv_mlp_proj", "weight"}, 3 * inner_dim_ + mlp_hidden_dim_, 3 * inner_dim_ + mlp_out);

            if (bias)
                mlp_in_up->bias()->split_cols({"to_qkv_mlp_proj", "bias"}, 3 * inner_dim_ + mlp_hidden_dim_, 3 * inner_dim_ + mlp_out);

            modules["mlp_in_up"] = mlp_in_up;

            auto attn = std::make_shared<Linear>(inner_dim_, out_dim_, out_bias);

            attn->weight()->split_cols({"to_out", "weight"}, 0, inner_dim_);

            if (out_bias)
                attn->bias()->split_cols({"to_out", "bias"}, 0, out_dim_);

            modules["attn"] = attn;

            auto mlp = std::make_shared<Linear>(mlp_hidden_dim_, out_dim_, false);

            mlp->weight()->split_cols({"to_out", "weight"}, inner_dim_, inner_dim_ + mlp_hidden_dim_);

            modules["mlp"] = mlp;
        } else {
            // Reference form: the reference fuses the QKV + MLP input
            // projections into one Linear, and the attention output +
            // MLP output projections into another.
            auto mlp_out = static_cast<int64_t>(mlp_hidden_dim_ * mlp_mult_factor_);

            // MROPE row permutation over the fused projection rows:
            // the q and k row blocks get the per-head P, the v block
            // and the MLP rows are identity.
            std::vector<size_t> qkv_mlp_perm(3 * inner_dim_ + mlp_out);

            for (int64_t r = 0; r < inner_dim_; ++r) {
                qkv_mlp_perm[r] = head_perm[r];
                qkv_mlp_perm[inner_dim_ + r] = inner_dim_ + head_perm[r];
            }

            for (int64_t r = 2 * inner_dim_; r < 3 * inner_dim_ + mlp_out; ++r)
                qkv_mlp_perm[r] = r;

            auto to_qkv_mlp_proj = std::make_shared<Linear>(query_dim_, 3 * inner_dim_ + mlp_out, bias);
            to_qkv_mlp_proj->weight()->permute_rows(qkv_mlp_perm);

            if (bias)
                to_qkv_mlp_proj->bias()->permute_elements(qkv_mlp_perm);

            modules["to_qkv_mlp_proj"] = to_qkv_mlp_proj;
            modules["to_out"] = std::make_shared<Linear>(inner_dim_ + mlp_hidden_dim_, out_dim_, out_bias);
        }

        modules["mlp_act_fn"] = std::make_shared<Flux2SwiGLU>();

        // QK Norm
        auto norm_q_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
        norm_q_weight->permute_elements(Flux2PosEmbed::mrope_permutation(head_dim_));
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine, norm_q_weight);

        auto norm_k_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
        norm_k_weight->permute_elements(Flux2PosEmbed::mrope_permutation(head_dim_));
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine, norm_k_weight);
    }

    Tensor forward(
        Scope scope,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::pair<std::shared_ptr<Flux2PosEmbed>, Tensor>> image_rotary_emb = std::nullopt
    ) {
        if (sharded_ != scope.runtime().use_sharding())
            throw std::runtime_error(
                "Flux2ParallelSelfAttention: built for " + std::string(sharded_ ? "sharded" : "non-sharded") +
                " execution but the active scope is " + std::string(scope.runtime().use_sharding() ? "sharded" : "non-sharded"));

        Tensor query, key, value;
        Tensor mlp_hidden_states;
        std::optional<Tensor> mlp_gate;
        std::optional<Tensor> mlp_up;

        if (sharded_) {
            // Sharding patch: the reference's fused projection is
            // replaced by split row-parallel projections (see the split
            // Parameter configurations in the constructor), so the
            // reference's split/chunk of its output does not apply.
            auto q = std::static_pointer_cast<Linear>(modules["q"]);
            auto k = std::static_pointer_cast<Linear>(modules["k"]);
            auto v = std::static_pointer_cast<Linear>(modules["v"]);
            auto mlp_in_gate = std::static_pointer_cast<Linear>(modules["mlp_in_gate"]);
            auto mlp_in_up = std::static_pointer_cast<Linear>(modules["mlp_in_up"]);

            query = q->forward(scope, hidden_states);
            key = k->forward(scope, hidden_states);
            value = v->forward(scope, hidden_states);
            mlp_gate = mlp_in_gate->forward(scope, hidden_states);
            mlp_up = mlp_in_up->forward(scope, hidden_states);
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

            // Parallel output projection: the reference's fused to_out
            // over [attn | mlp] is split into two row-parallel
            // projections whose outputs are added (see the split
            // Parameter configurations in the constructor).
            auto to_out_attn = std::static_pointer_cast<Linear>(modules["attn"]);
            auto to_out_mlp = std::static_pointer_cast<Linear>(modules["mlp"]);

            hidden_states = to_out_attn->forward(scope, hidden_states) + to_out_mlp->forward(scope, mlp_hidden_states);
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
