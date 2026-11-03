#pragma once

#include "nn/Module.hpp"
#include "nn/ModuleList.hpp"
#include "nn/Linear.hpp"
#include "nn/Dropout.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"

template <class AttnOp>
class Flux2Attention : public Module {
public:
    Flux2Attention(
        int64_t query_dim,
        int64_t heads = 8,
        int64_t dim_head = 64,
        float dropout = 0.0,
        bool bias = false,
        std::optional<int64_t> added_kv_proj_dim = std::nullopt,
        bool added_proj_bias = true,
        bool out_bias = true,
        float eps = 1e-5,
        std::optional<int64_t> out_dim = std::nullopt,
        bool elementwise_affine = true
    )
    {
        head_dim_ = dim_head;
        inner_dim_ = out_dim.value_or(dim_head * heads);
        query_dim_ = query_dim;
        out_dim_ = out_dim.value_or(query_dim);
        heads_ = out_dim ? *out_dim / dim_head : heads;

        use_bias_ = bias;
        dropout_ = dropout;

        added_kv_proj_dim_ = added_kv_proj_dim;
        added_proj_bias_ = added_proj_bias;

        //
        // MROPE head layout (see Flux2PosEmbed): the q/k projections
        // and the QK-norm weights carry a permutation configuration
        // (Parameter::Permutation) -- the loader permutes them into the
        // ggml MROPE (NEOX) layout once at load time, so attention runs
        // the whole head dimension through one MROPE op. The q/k biases
        // are permuted the same way as their weights. v and the output
        // projections stay in the original layout (the attention output
        // is invariant under the same permutation of q and k).
        //
        auto qk_rows_perm = Flux2PosEmbed::mrope_permutation(head_dim_, heads_);
        auto norm_perm = Flux2PosEmbed::mrope_permutation(head_dim_);

        auto to_q = std::make_shared<Linear>(query_dim_, inner_dim_, bias);
        to_q->weight()->permute_rows(qk_rows_perm);

        if (bias)
            to_q->bias()->permute_elements(qk_rows_perm);

        modules["to_q"] = to_q;

        auto to_k = std::make_shared<Linear>(query_dim_, inner_dim_, bias);
        to_k->weight()->permute_rows(qk_rows_perm);

        if (bias)
            to_k->bias()->permute_elements(qk_rows_perm);

        modules["to_k"] = to_k;

        modules["to_v"] = std::make_shared<Linear>(query_dim_, inner_dim_, bias);

        // QK Norm
        auto norm_q_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
        norm_q_weight->permute_elements(norm_perm);
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine, norm_q_weight);

        auto norm_k_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
        norm_k_weight->permute_elements(norm_perm);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine, norm_k_weight);

        // Output projection
        modules["to_out"] = std::shared_ptr<ModuleList>(new ModuleList({
            std::make_shared<Linear>(inner_dim_, out_dim_, out_bias),
            std::make_shared<Dropout>(dropout_)
        }));
        
        if (added_kv_proj_dim_) {
            auto norm_added_q_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
            norm_added_q_weight->permute_elements(norm_perm);
            modules["norm_added_q"] = std::make_shared<RMSNorm>(dim_head, eps, true, norm_added_q_weight);

            auto norm_added_k_weight = std::make_shared<Parameter>(Tensor::Shape({dim_head}));
            norm_added_k_weight->permute_elements(norm_perm);
            modules["norm_added_k"] = std::make_shared<RMSNorm>(dim_head, eps, true, norm_added_k_weight);

            auto add_q_proj = std::make_shared<Linear>(added_kv_proj_dim_.value(), inner_dim_, added_proj_bias_);
            add_q_proj->weight()->permute_rows(qk_rows_perm);

            if (added_proj_bias_)
                add_q_proj->bias()->permute_elements(qk_rows_perm);

            modules["add_q_proj"] = add_q_proj;

            auto add_k_proj = std::make_shared<Linear>(added_kv_proj_dim_.value(), inner_dim_, added_proj_bias_);
            add_k_proj->weight()->permute_rows(qk_rows_perm);

            if (added_proj_bias_)
                add_k_proj->bias()->permute_elements(qk_rows_perm);

            modules["add_k_proj"] = add_k_proj;

            modules["add_v_proj"] = std::make_shared<Linear>(added_kv_proj_dim_.value(), inner_dim_, added_proj_bias_);
            modules["to_add_out"] = std::make_shared<Linear>(inner_dim_, query_dim_, out_bias);
        }
    }

    std::tuple<Tensor, std::optional<Tensor>> forward(
        Scope scope,
        Tensor hidden_states,
        std::optional<Tensor> encoder_hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::pair<std::shared_ptr<Flux2PosEmbed>, Tensor>> image_rotary_emb = std::nullopt
    ) {
        // _get_qkv_projections(...) when use_fused_projections=false
        auto to_q = std::static_pointer_cast<Linear>(modules["to_q"]);
        auto to_k = std::static_pointer_cast<Linear>(modules["to_k"]);
        auto to_v = std::static_pointer_cast<Linear>(modules["to_v"]);

        auto query = to_q->forward(scope, hidden_states);
        auto key = to_k->forward(scope, hidden_states);
        auto value = to_v->forward(scope, hidden_states);

        std::optional<Tensor> encoder_query, encoder_key, encoder_value;

        if (encoder_hidden_states && added_kv_proj_dim_) {
            auto add_q_proj = std::static_pointer_cast<Linear>(modules["add_q_proj"]);
            auto add_k_proj = std::static_pointer_cast<Linear>(modules["add_k_proj"]);
            auto add_v_proj = std::static_pointer_cast<Linear>(modules["add_v_proj"]);

            encoder_query = add_q_proj->forward(scope, encoder_hidden_states.value());
            encoder_key = add_k_proj->forward(scope, encoder_hidden_states.value());
            encoder_value = add_v_proj->forward(scope, encoder_hidden_states.value());
        }
        // end of _get_qkv_projections(...)

        query = query.unflatten(-1, {heads_, -1});
        key = key.unflatten(-1, {heads_, -1});
        value = value.unflatten(-1, {heads_, -1});

        auto norm_q = std::static_pointer_cast<RMSNorm>(modules["norm_q"]);
        auto norm_k = std::static_pointer_cast<RMSNorm>(modules["norm_k"]);

        query = norm_q->forward(scope, query);
        key = norm_k->forward(scope, key);

        if (added_kv_proj_dim_) {
            encoder_query = encoder_query.value().unflatten(-1, {heads_, -1});
            encoder_key = encoder_key.value().unflatten(-1, {heads_, -1});
            encoder_value = encoder_value.value().unflatten(-1, {heads_, -1});

            auto norm_added_q = std::static_pointer_cast<RMSNorm>(modules["norm_added_q"]);
            auto norm_added_k = std::static_pointer_cast<RMSNorm>(modules["norm_added_k"]);

            encoder_query = norm_added_q->forward(scope, encoder_query.value());
            encoder_key = norm_added_k->forward(scope, encoder_key.value());

            query = Tensor::cat({encoder_query.value(), query}, 1);
            key = Tensor::cat({encoder_key.value(), key}, 1);
            value = Tensor::cat({encoder_value.value(), value}, 1);
        }

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

        if (encoder_hidden_states) {
            auto to_add_out = std::static_pointer_cast<Linear>(modules["to_add_out"]);

            auto parts = hidden_states.split_with_sizes({
                encoder_hidden_states.value().shape()[1],
                hidden_states.shape()[1] - encoder_hidden_states.value().shape()[1]
            }, 1);
            
            encoder_hidden_states = parts.at(0);
            hidden_states = parts.at(1);

            encoder_hidden_states = to_add_out->forward(scope, encoder_hidden_states.value());
        }

        auto to_out = std::static_pointer_cast<ModuleList>(modules["to_out"]);

        hidden_states = std::static_pointer_cast<Linear>((*to_out)[0])->forward(scope, hidden_states);
        hidden_states = std::static_pointer_cast<Dropout>((*to_out)[1])->forward(scope, hidden_states);

        if (encoder_hidden_states)
            return {hidden_states, encoder_hidden_states.value()};

        return {hidden_states, std::nullopt};
    }

private:
    int64_t head_dim_;
    int64_t inner_dim_;
    int64_t query_dim_;
    int64_t out_dim_;
    int64_t heads_;

    bool use_bias_;
    float dropout_;

    std::optional<int64_t> added_kv_proj_dim_;
    bool added_proj_bias_;
};
