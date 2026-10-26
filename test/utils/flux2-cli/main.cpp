#include "../TestCLI.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"
#include "diffusers/models/transformers/flux2/Flux2FeedForward.hpp"
#include "diffusers/models/transformers/flux2/Flux2Modulation.hpp"
#include "diffusers/models/transformers/flux2/Flux2TimestepGuidanceEmbeddings.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"
#include "diffusers/models/transformers/flux2/Flux2Attention.hpp"
#include "diffusers/models/transformers/flux2/Flux2ParallelSelfAttention.hpp"
#include "diffusers/models/transformers/flux2/Flux2SingleTransformerBlock.hpp"
#include "diffusers/models/transformers/flux2/Flux2TransformerBlock.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"

#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"

#include <numeric>

class TestFlux2CLI : public TestCLI {
public:
    TestFlux2CLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Computation<std::vector<Tensor>> compute(Context& context) {

        if (args_.get(0) == "Flux2SwiGLU") {
            Computation<void> computation({&context});

            auto gate = args_.get_one<Tensor>("--gate", {computation.desc()->context()});
            auto up = args_.get_one<Tensor>("--up", {computation.desc()->context()});

            Flux2SwiGLU model;

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, gate, up);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2FeedForward") {
            Computation<void> computation({&context});

            auto dim = args_.get_one<int64_t>("--dim");
            auto dim_out = args_.get_optional<int64_t>("--dim_out");
            auto mult = args_.get_optional<float>("--mult").value_or(3.0);
            auto inner_dim = args_.get_optional<int64_t>("--inner_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->context()});

            Flux2FeedForward model(dim, dim_out, mult, inner_dim, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, x);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2Modulation") {
            Computation<void> computation({&context});

            auto dim = args_.get_one<int64_t>("--dim");
            auto mod_param_sets = args_.get_optional<int64_t>("--mod_param_sets").value_or(2);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto temb = args_.get_one<Tensor>("--temb", {computation.desc()->context()});

            Flux2Modulation model(dim, mod_param_sets, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, temb);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2TimestepGuidanceEmbeddings") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(true);
            auto timestep = args_.get_one<Tensor>("--timestep", {computation.desc()->context()});
            auto guidance = args_.get_optional<Tensor>("--guidance", {computation.desc()->context()});

            Flux2TimestepGuidanceEmbeddings model(in_channels, embedding_dim, bias, guidance_embeds);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, timestep, guidance);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2PosEmbed") {
            Computation<void> computation({&context});

            auto theta = args_.get_one<int64_t>("--theta");
            auto axes_dim = args_.get_many<int64_t>("--axes_dim");
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->context()});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

            Flux2PosEmbed model(theta, axes_dim);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, x, position_ids);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2Attention") {
            Computation<void> computation({&context});

            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_optional<int64_t>("--heads").value_or(8);
            auto dim_head = args_.get_optional<int64_t>("--dim_head").value_or(64);
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto added_kv_proj_dim = args_.get_optional<int64_t>("--added_kv_proj_dim");
            auto added_proj_bias = args_.get_optional<bool>("--added_proj_bias").value_or(true);
            auto out_bias = args_.get_optional<bool>("--out_bias").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5);
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {computation.desc()->context()});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {computation.desc()->context()});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2Attention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                added_kv_proj_dim,
                added_proj_bias,
                out_bias,
                eps,
                out_dim,
                elementwise_affine
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                auto [y1, y2] = model.forward(scope, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb);
                std::vector<Tensor> results;

                results.push_back(y1);

                if (y2)
                    results.push_back(*y2);

                return results;
            });
            
            return result;
        }

        if (args_.get(0) == "Flux2ParallelSelfAttention") {
            Computation<void> computation({&context});

            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_optional<int64_t>("--heads").value_or(8);
            auto dim_head = args_.get_optional<int64_t>("--dim_head").value_or(64);
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto out_bias = args_.get_optional<bool>("--out_bias").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5);
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(4.0);
            auto mlp_mult_factor = args_.get_optional<int64_t>("--mlp_mult_factor").value_or(2);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {computation.desc()->context()});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2ParallelSelfAttention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                out_bias,
                eps,
                out_dim,
                elementwise_affine,
                mlp_ratio,
                mlp_mult_factor
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states, attention_mask, image_rotary_emb);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2SingleTransformerBlock") {
            Computation<void> computation({&context});

            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {computation.desc()->context()});
            auto temb_mod = args_.get_one<Tensor>("--temb_mod", {computation.desc()->context()});
            auto split_hidden_states = args_.get_optional<bool>("--split_hidden_states").value_or(false);
            auto text_seq_len = args_.get_optional<int64_t>("--text_seq_len");
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2SingleTransformerBlock<ScaledDotProductAttention<FlashAttentionOp>> model(
                dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                bias
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                auto [y1, y2] = model.forward(
                    scope,
                    hidden_states,
                    encoder_hidden_states,
                    temb_mod,
                    image_rotary_emb,
                    split_hidden_states,
                    text_seq_len
                );

                std::vector<Tensor> results;

                results.push_back(y1);

                if (y2)
                    results.push_back(*y2);

                return results;
            });
            
            return result;
        }

        if (args_.get(0) == "Flux2TransformerBlock") {
            Computation<void> computation({&context});

            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {computation.desc()->context()});
            auto temb_mod_img = args_.get_one<Tensor>("--temb_mod_img", {computation.desc()->context()});
            auto temb_mod_txt = args_.get_one<Tensor>("--temb_mod_txt", {computation.desc()->context()});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2TransformerBlock<ScaledDotProductAttention<FlashAttentionOp>> model(
                dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                bias
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                auto [y1, y2] = model.forward(
                    scope,
                    hidden_states,
                    encoder_hidden_states,
                    temb_mod_img,
                    temb_mod_txt,
                    image_rotary_emb
                );

                return {y1, y2};
            });
            
            return result;
        }

        if (args_.get(0) == "Flux2Transformer2DModel") {
            Computation<void> computation({&context});

            Flux2Transformer2DModel::Config config;

            config.patch_size = args_.get_optional<int64_t>("--patch_size").value_or(config.patch_size);
            config.in_channels = args_.get_optional<int64_t>("--in_channels").value_or(config.in_channels);
            config.out_channels = args_.get_optional<int64_t>("--out_channels");
            config.num_layers = args_.get_optional<int64_t>("--num_layers").value_or(config.num_layers);
            config.num_single_layers = args_.get_optional<int64_t>("--num_single_layers").value_or(config.num_single_layers);
            config.attention_head_dim = args_.get_optional<int64_t>("--attention_head_dim").value_or(config.attention_head_dim);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.joint_attention_dim = args_.get_optional<int64_t>("--joint_attention_dim").value_or(config.joint_attention_dim);
            config.timestep_guidance_channels = args_.get_optional<int64_t>("--timestep_guidance_channels").value_or(config.timestep_guidance_channels);
            config.mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(config.mlp_ratio);
            auto axes_dims_rope = args_.get_many<int64_t>("--axes_dims_rope");
            config.rope_theta = args_.get_optional<int64_t>("--rope_theta").value_or(config.rope_theta);
            config.eps = args_.get_optional<float>("--eps").value_or(config.eps);
            config.guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(config.guidance_embeds);

            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {computation.desc()->context()});
            auto timestep = args_.get_one<Tensor>("--timestep", {computation.desc()->context()});
            auto img_ids = args_.get_one<Tensor>("--img_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});
            auto txt_ids = args_.get_one<Tensor>("--txt_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});
            auto guidance = args_.get_optional<Tensor>("--guidance", {computation.desc()->context()});
            auto num_ref_tokens = args_.get_optional<int64_t>("--num_ref_tokens").value_or(0);
            auto ref_fixed_timestep = args_.get_optional<float>("--ref_fixed_timestep").value_or(0.0f);

            if (!axes_dims_rope.empty())
                config.axes_dims_rope = axes_dims_rope;

            Flux2Transformer2DModel model(config);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(
                    scope,
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    img_ids,
                    txt_ids,
                    guidance,
                    //std::nullopt,   // kv_cache
                    //std::nullopt,   // kv_cache_mode
                    num_ref_tokens,
                    ref_fixed_timestep
                );
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Flux2KleinPipeline_pack_latents" ||
            args_.get(0) == "Flux2KleinPipeline_unpack_latents" ||
            args_.get(0) == "Flux2KleinPipeline_patchify_latents" ||
            args_.get(0) == "Flux2KleinPipeline_unpatchify_latents") {
            Computation<void> computation({&context});
            auto latents = args_.get_one<Tensor>("--latents", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                if (args_.get(0) == "Flux2KleinPipeline_pack_latents")
                    return Flux2KleinPipeline::pack_latents(latents);
                
                if (args_.get(0) == "Flux2KleinPipeline_unpack_latents")
                    return Flux2KleinPipeline::unpack_latents(
                        latents,
                        args_.get_one<int>("--packed_h"),
                        args_.get_one<int>("--packed_w")
                    );
                
                if (args_.get(0) == "Flux2KleinPipeline_patchify_latents")
                    return Flux2KleinPipeline::patchify_latents(
                        latents,
                        args_.get_one<int>("--channels"),
                        args_.get_one<int>("--packed_h"),
                        args_.get_one<int>("--packed_w")
                    );

                return Flux2KleinPipeline::unpatchify_latents(
                    latents,
                    args_.get_one<int>("--channels"),
                    args_.get_one<int>("--packed_h"),
                    args_.get_one<int>("--packed_w")
                );
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0).rfind("Flux2KleinPipeline", 0) == 0) {
            Flux2Transformer2DModel::Config transformer_config;
            {
                transformer_config.patch_size = args_.get_optional<int64_t>("--transformer-patch_size").value_or(transformer_config.patch_size);
                transformer_config.in_channels = args_.get_optional<int64_t>("--transformer-in_channels").value_or(transformer_config.in_channels);
                transformer_config.out_channels = args_.get_optional<int64_t>("--transformer-out_channels");
                transformer_config.num_layers = args_.get_optional<int64_t>("--transformer-num_layers").value_or(transformer_config.num_layers);
                transformer_config.num_single_layers = args_.get_optional<int64_t>("--transformer-num_single_layers").value_or(transformer_config.num_single_layers);
                transformer_config.attention_head_dim = args_.get_optional<int64_t>("--transformer-attention_head_dim").value_or(transformer_config.attention_head_dim);
                transformer_config.num_attention_heads = args_.get_optional<int64_t>("--transformer-num_attention_heads").value_or(transformer_config.num_attention_heads);
                transformer_config.joint_attention_dim = args_.get_optional<int64_t>("--transformer-joint_attention_dim").value_or(transformer_config.joint_attention_dim);
                transformer_config.timestep_guidance_channels = args_.get_optional<int64_t>("--transformer-timestep_guidance_channels").value_or(transformer_config.timestep_guidance_channels);
                transformer_config.mlp_ratio = args_.get_optional<float>("--transformer-mlp_ratio").value_or(transformer_config.mlp_ratio);
                auto axes_dims_rope = args_.get_many<int64_t>("--transformer-axes_dims_rope");
                transformer_config.rope_theta = args_.get_optional<int64_t>("--transformer-rope_theta").value_or(transformer_config.rope_theta);
                transformer_config.eps = args_.get_optional<float>("--transformer-eps").value_or(transformer_config.eps);
                transformer_config.guidance_embeds = args_.get_optional<bool>("--transformer-guidance_embeds").value_or(transformer_config.guidance_embeds);

                if (!axes_dims_rope.empty())
                    transformer_config.axes_dims_rope = axes_dims_rope;
            }

            AutoencoderKLFlux2::Config vae_config;
            {
                vae_config.in_channels = args_.get_optional<int64_t>("--vae-in_channels").value_or(vae_config.in_channels);
                vae_config.out_channels = args_.get_optional<int64_t>("--vae-out_channels").value_or(vae_config.out_channels);
                auto block_out_channels = args_.get_many<int64_t>("--vae-block_out_channels");
                vae_config.layers_per_block = args_.get_optional<int64_t>("--vae-layers_per_block").value_or(vae_config.layers_per_block);
                vae_config.latent_channels = args_.get_optional<int64_t>("--vae-latent_channels").value_or(vae_config.latent_channels);
                vae_config.norm_num_groups = args_.get_optional<int64_t>("--vae-norm_num_groups").value_or(vae_config.norm_num_groups);
                vae_config.sample_size = args_.get_optional<int64_t>("--vae-sample_size").value_or(vae_config.sample_size);
                vae_config.force_upcast = args_.get_optional<bool>("--vae-force_upcast").value_or(vae_config.force_upcast);
                vae_config.use_quant_conv = args_.get_optional<bool>("--vae-use_quant_conv").value_or(vae_config.use_quant_conv);
                vae_config.use_post_quant_conv = args_.get_optional<bool>("--vae-use_post_quant_conv").value_or(vae_config.use_post_quant_conv);
                vae_config.mid_block_add_attention = args_.get_optional<bool>("--vae-mid_block_add_attention").value_or(vae_config.mid_block_add_attention);
                vae_config.batch_norm_eps = args_.get_optional<float>("--vae-batch_norm_eps").value_or(vae_config.batch_norm_eps);
                vae_config.batch_norm_momentum = args_.get_optional<float>("--vae-batch_norm_momentum").value_or(vae_config.batch_norm_momentum);
                vae_config.patch_size = std::make_tuple(
                    args_.get_optional<int64_t>("--vae-patch_size-0").value_or(std::get<0>(vae_config.patch_size)),
                    args_.get_optional<int64_t>("--vae-patch_size-1").value_or(std::get<1>(vae_config.patch_size))
                );

                if (!block_out_channels.empty())
                    vae_config.block_out_channels = block_out_channels;
            }

            Qwen3Config qwen_config;
            {
                qwen_config.vocab_size = args_.get_optional<int64_t>("--text_encoder-vocab_size").value_or(qwen_config.vocab_size);
                qwen_config.hidden_size = args_.get_optional<int64_t>("--text_encoder-hidden_size").value_or(qwen_config.hidden_size);
                qwen_config.intermediate_size = args_.get_optional<int64_t>("--text_encoder-intermediate_size").value_or(qwen_config.intermediate_size);
                qwen_config.num_hidden_layers = args_.get_optional<int64_t>("--text_encoder-num_hidden_layers").value_or(qwen_config.num_hidden_layers);
                qwen_config.num_attention_heads = args_.get_optional<int64_t>("--text_encoder-num_attention_heads").value_or(qwen_config.num_attention_heads);
                qwen_config.num_key_value_heads = args_.get_optional<int64_t>("--text_encoder-num_key_value_heads").value_or(qwen_config.num_key_value_heads);
                qwen_config.max_position_embeddings = args_.get_optional<int64_t>("--text_encoder-max_position_embeddings").value_or(qwen_config.max_position_embeddings);
            }

            auto tokenizer_dir = args_.get_one<std::string>("--tokenizer_dir");
            Computation<void> computation({&context});

            Flux2Transformer2DModel transformer(transformer_config);
            {
                CreateParametersVisitor create_parameters(context, args_, "transformer");
                RethrowVisitor visitor(create_parameters);
                transformer.accept(visitor);
                visitor.rethrow();
            }

            AutoencoderKLFlux2 vae(vae_config);
            {
                CreateParametersVisitor create_parameters(context, args_, "vae");
                RethrowVisitor visitor(create_parameters);
                vae.accept(visitor);
                visitor.rethrow();
            }

            Qwen3ForCausalLM text_encoder(qwen_config, std::vector<size_t>{9, 18, 27});
            {
                CreateParametersVisitor create_parameters(context, args_, "text_encoder");
                RethrowVisitor visitor(create_parameters);
                text_encoder.accept(visitor);
                visitor.rethrow();
            }

            auto tokenizer = Qwen2TokenizerFast::from_pretrained(tokenizer_dir);

            Flux2KleinPipeline pipeline(
                std::move(transformer),
                std::move(vae),
                std::move(text_encoder),
                std::move(tokenizer)
            );

            if (args_.get(0) == "Flux2KleinPipeline_vae_encode") {
                auto batch = args_.get_one<int>("--batch");
                auto images = args_.get_many<Image>("--images");
                auto tile_size = args_.get_optional<int>("--encode_tile_size").value_or(0);

                // Mirrors the condition-image preprocessing in the Python
                // pipeline's __call__: resize to the target area and crop
                // to the VAE's spatial multiple.
                for (auto& image : images)
                    image = Flux2KleinPipeline::preprocess_reference_image(image, pipeline.vae_scale_factor() * 2);

                auto image_latents = pipeline.encode_images(computation, images, batch, tile_size);

                if (!image_latents)
                    throw std::runtime_error("Flux2KleinPipeline_vae_encode: expected at least one image");

                return Computation<Tensor>::all(*image_latents);
            }

            if (args_.get(0) == "Flux2KleinPipeline_text_encoder") {
                auto batch = args_.get_one<int>("--batch");
                auto prompt = args_.get_one<std::string>("--prompt");
                auto max_sequence_length = args_.get_one<int>("--max_sequence_length");

                auto result = computation.scope([&](Scope scope) -> Tensor {
                    return pipeline.encode_prompt(scope, batch, prompt, max_sequence_length);
                });

                return Computation<Tensor>::all(result);
            }

            if (args_.get(0) == "Flux2KleinPipeline_denoise") {
                auto batch = args_.get_one<int>("--batch");
                auto timestep = args_.get_one<float>("--timestep");
                auto dt = args_.get_one<float>("--dt");
                auto init_latents = args_.get_one<Tensor>("--init_latents", {computation.desc()->context()});
                auto prompt_embeds = args_.get_one<Tensor>("--prompt_embeds", {computation.desc()->context()});
                auto img_ids = args_.get_one<Tensor>("--img_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});
                auto txt_ids = args_.get_one<Tensor>("--txt_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

                auto image_latents = args_.get_optional<Tensor>("--image_latents", {computation.desc()->context()});
                auto image_latent_ids = args_.get_optional<Tensor>("--image_latent_ids", {computation.desc()->context(), Tensor::DType<int32_t>::value});

                auto denoised = computation.scope([&](Scope scope) -> Tensor {
                    auto timestep_tensor = scope.context().create<float>({batch}, [batch, timestep](std::mt19937&) {
                        return std::vector<float>(batch, timestep);
                    });
    
                    auto dt_tensor = scope.context().create<float>({}, [batch, dt](std::mt19937&) {
                        return std::vector<float>{dt};
                    });

                    return pipeline.denoise(
                        scope,
                        init_latents,
                        prompt_embeds,
                        img_ids,
                        txt_ids,
                        image_latents,
                        image_latent_ids,
                        timestep_tensor,
                        dt_tensor);
                });

                return Computation<Tensor>::all(denoised);
            }

            if (args_.get(0) == "Flux2KleinPipeline_vae_decode") {
                auto packed_h = args_.get_one<int>("--packed_h");
                auto packed_w = args_.get_one<int>("--packed_w");
                auto latents = args_.get_one<Tensor>("--latents", {computation.desc()->context()});
                auto tile_size = args_.get_optional<int>("--decode_tile_size").value_or(0);

                auto result = pipeline.decode(Computation<Tensor>(latents, computation.desc()), packed_h, packed_w, tile_size);

                return Computation<Tensor>::all(result);
            }

            if (args_.get(0) == "Flux2KleinPipeline_call") {
                Flux2KleinPipeline::GenerationOptions options;

                options.prompt = args_.get_one<std::string>("--prompt");
                options.height = args_.get_one<int>("--height");
                options.width = args_.get_one<int>("--width");
                options.num_inference_steps = args_.get_one<int>("--num_inference_steps");
                options.max_sequence_length = args_.get_one<int>("--max_sequence_length");
                options.decode_tile_size = args_.get_optional<int>("--decode_tile_size").value_or(options.decode_tile_size);
                options.encode_tile_size = args_.get_optional<int>("--encode_tile_size").value_or(options.encode_tile_size);
                if (auto init_latents = args_.get_optional<std::string>("--init_latents"))
                    options.init_latents = std::move(
                        ArgumentParser::parser<Tensor>::TensorParser("--init_latents", *init_latents).parse().second);

                // The raw decoded values are converted to RGB images on the
                // CPU side in run() (Computation<Image> is not supported).
                return Computation<Tensor>::all(pipeline(context, context, context, std::move(options)));
            }

            throw std::runtime_error("Unknown command: " + args_.get(0));
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

    virtual size_t get_graph_size() const {
        if (args_.get(0).rfind("Flux2KleinPipeline", 0) == 0)
            return 65536;
        
        return TestCLI::get_graph_size();
    }

    // Flux2KleinPipeline_call returns the raw decoded values (B, 3, H, W).
    // Computation<Image> is not supported, so the conversion to RGB images
    // (H, W, 3) is performed on the CPU side after execution.
    virtual int run(Backend& backend, Computation<std::vector<Tensor>> computation, ShardingRuntime* sharding = nullptr) override {
        if (args_.get(0) == "Flux2KleinPipeline_call") {
            std::mt19937 rng;
            //ProgressBar progress("Testing");
            auto results = ExecutionRuntime::Default.run(backend, rng, computation, sharding);

            if (results.size() != 1)
                throw std::runtime_error("Flux2KleinPipeline_call: expected exactly one result tensor");

            auto& decoded = results.front();
            auto data = ExecutionRuntime::Default.read<float>(decoded);

            auto images = Flux2KleinPipeline::to_images(decoded.shape(), std::move(data));

            for (const auto& image : images) {
                std::vector<float> pixels(image.pixels().size());

                for (size_t i = 0; i < pixels.size(); ++i)
                    pixels[i] = static_cast<float>(image.pixels()[i]);

                print_tensor_like(pixels, {
                    (int64_t)image.height(),
                    (int64_t)image.width(),
                    (int64_t)image.channels()
                });
            }

            return EXIT_SUCCESS;
        }

        return TestCLI::run(backend, computation, sharding);
    }

private:
    class CreateParametersVisitor : public TestCLI::CreateParametersVisitor {
    public:
        CreateParametersVisitor(Context& context, const ArgumentParser& args, const std::string& prefix = "")
            : TestCLI::CreateParametersVisitor(context, args, prefix)
        {}

        void visit(Flux2FusedQKVProjection& to_qkv_mlp_proj, std::vector<std::string> path) override {
            ModulePath module_path("-", "--param");
            auto weight_path = module_path(path, prefix(), {"weight"});

            auto q_weight = to_qkv_mlp_proj.q()->weight();
            auto k_weight = to_qkv_mlp_proj.k()->weight();
            auto v_weight = to_qkv_mlp_proj.v()->weight();
            auto mlp_in_gate_weight = to_qkv_mlp_proj.mlp_in_gate()->weight();
            auto mlp_in_up_weight = to_qkv_mlp_proj.mlp_in_up()->weight();

            auto inner = to_qkv_mlp_proj.inner_dim();
            auto mlp_out = to_qkv_mlp_proj.mlp_out_dim();

            // The MLP input projection is split into gate and up, each
            // with mlp_out / 2 output features: the first half of the
            // fused MLP rows is the gate, the second half the up.
            auto mlp_hidden = mlp_in_gate_weight->shape()[0];

            // The fused weight is (3 * inner + mlp_out) x query_dim, so rows
            // of the flattened data are query_dim elements long.
            auto query_dim = q_weight->shape()[1];

            if (auto type = quantization().resolve(dot_name(path) + ".weight", prefix()); type && *type != GGML_TYPE_F32) {
                // Quantized weight: the file holds the raw bytes of the
                // fused tensor. The rows are contiguous in memory, so the
                // split into q/k/v/mlp_in is a plain byte-range slice.
                auto data = get_param_bytes(weight_path);

                auto row_bytes = static_cast<size_t>(ggml_row_size(*type, query_dim));

                if (data.size() != row_bytes * (3 * inner + mlp_out))
                    throw std::runtime_error(
                        "Flux2FusedQKVProjection: quantized weight '" + weight_path +
                        "' has " + std::to_string(data.size()) + " bytes, expected " +
                        std::to_string(row_bytes * (3 * inner + mlp_out)));

                q_weight->set(create_quantized(weight_path + "-q-weight", q_weight->shape(), *type,
                    slice_rows_bytes(data, *type, query_dim, 0, inner)));

                k_weight->set(create_quantized(weight_path + "-k-weight", k_weight->shape(), *type,
                    slice_rows_bytes(data, *type, query_dim, inner, 2 * inner)));

                v_weight->set(create_quantized(weight_path + "-v-weight", v_weight->shape(), *type,
                    slice_rows_bytes(data, *type, query_dim, 2 * inner, 3 * inner)));

                mlp_in_gate_weight->set(create_quantized(weight_path + "-mlp_in_gate-weight", mlp_in_gate_weight->shape(), *type,
                    slice_rows_bytes(data, *type, query_dim, 3 * inner, 3 * inner + mlp_hidden)));

                mlp_in_up_weight->set(create_quantized(weight_path + "-mlp_in_up-weight", mlp_in_up_weight->shape(), *type,
                    slice_rows_bytes(data, *type, query_dim, 3 * inner + mlp_hidden, 3 * inner + mlp_out)));

                return;
            }

            auto weight_value = get_param(weight_path);
            ArgumentParser::parser<Tensor>::TensorParser parser(weight_path, weight_value);

            auto [shape, data] = parser.parse();

            q_weight->set(context().create<float>(q_weight->shape(), [=](std::mt19937&) {
                return slice_rows(data, query_dim, 0, inner);
            }).name(weight_path + "-q-weight"));

            k_weight->set(context().create<float>(k_weight->shape(), [=](std::mt19937&) {
                return slice_rows(data, query_dim, inner, 2 * inner);
            }).name(weight_path + "-k-weight"));

            v_weight->set(context().create<float>(v_weight->shape(), [=](std::mt19937&) {
                return slice_rows(data, query_dim, 2 * inner, 3 * inner);
            }).name(weight_path + "-v-weight"));

            mlp_in_gate_weight->set(context().create<float>(mlp_in_gate_weight->shape(), [=](std::mt19937&) {
                return slice_rows(data, query_dim, 3 * inner, 3 * inner + mlp_hidden);
            }).name(weight_path + "-mlp_in_gate-weight"));

            mlp_in_up_weight->set(context().create<float>(mlp_in_up_weight->shape(), [=](std::mt19937&) {
                return slice_rows(data, query_dim, 3 * inner + mlp_hidden, 3 * inner + mlp_out);
            }).name(weight_path + "-mlp_in_up-weight"));
        }

        void visit(Flux2FusedAttentionOutput& to_out, std::vector<std::string> path) override {
            ModulePath module_path("-", "--param");

            auto weight_path = module_path(path, prefix(), {"weight"});

            auto attn_weight = to_out.attn()->weight();
            auto mlp_weight = to_out.mlp()->weight();

            auto inner = to_out.inner_dim();
            auto mlp_hidden = to_out.mlp_hidden_dim();

            // The fused weight is (out_dim) x (inner + mlp_hidden) in the
            // PyTorch (out_features, in_features) layout.
            auto out_dim = attn_weight->shape()[0];

            if (auto type = quantization().resolve(dot_name(path) + ".weight", prefix()); type && *type != GGML_TYPE_F32) {
                // Quantized weight: the file holds the raw bytes of the
                // fused tensor. Splitting the columns (attn | mlp) must
                // fall on quantization block boundaries.
                auto data = get_param_bytes(weight_path);

                auto row_bytes = static_cast<size_t>(ggml_row_size(*type, inner + mlp_hidden));

                if (data.size() != row_bytes * out_dim)
                    throw std::runtime_error(
                        "Flux2FusedAttentionOutput: quantized weight '" + weight_path +
                        "' has " + std::to_string(data.size()) + " bytes, expected " +
                        std::to_string(row_bytes * out_dim));

                attn_weight->set(create_quantized(weight_path + "-attn-weight", attn_weight->shape(), *type,
                    slice_cols_bytes(data, *type, out_dim, inner + mlp_hidden, 0, inner)));

                mlp_weight->set(create_quantized(weight_path + "-mlp-weight", mlp_weight->shape(), *type,
                    slice_cols_bytes(data, *type, out_dim, inner + mlp_hidden, inner, inner + mlp_hidden)));

                return;
            }

            auto weight_value = get_param(weight_path);
            ArgumentParser::parser<Tensor>::TensorParser weight_parser(weight_path, weight_value);

            auto [weight_shape, weight_data] = weight_parser.parse();

            attn_weight->set(context().create<float>(attn_weight->shape(), [=](std::mt19937&) {
                return slice_cols(weight_data, out_dim, inner + mlp_hidden, 0, inner);
            }).name(weight_path + "-attn-weight"));

            mlp_weight->set(context().create<float>(mlp_weight->shape(), [=](std::mt19937&) {
                return slice_cols(weight_data, out_dim, inner + mlp_hidden, inner, inner + mlp_hidden);
            }).name(weight_path + "-mlp-weight"));

            auto attn_bias = to_out.attn()->bias();

            if (attn_bias) {
                auto bias_path = module_path(path, prefix(), {"bias"});
                auto bias_value = get_param(bias_path);
                ArgumentParser::parser<Tensor>::TensorParser bias_parser(bias_path, bias_value);

                auto [bias_shape, bias_data] = bias_parser.parse();

                attn_bias->set(context().create<float>(attn_bias->shape(), [=](std::mt19937&) {
                    return bias_data;
                }).name(bias_path));
            }
        }

        void visit(Flux2FeedForward& feed_forward, std::vector<std::string> path) override {
            ModulePath module_path("-", "--param");
            auto weight_path = module_path(path, prefix(), {"linear_in", "weight"});

            auto gate_weight = feed_forward.linear_in_gate()->weight();
            auto up_weight = feed_forward.linear_in_up()->weight();

            // The fused linear_in weight is (inner * 2) x dim (PyTorch
            // layout): the first inner rows are the gate, the last inner
            // rows the up projection.
            auto inner = gate_weight->shape()[0];
            auto dim = gate_weight->shape()[1];

            if (auto type = quantization().resolve(dot_name(path) + ".linear_in.weight", prefix()); type && *type != GGML_TYPE_F32) {
                // Quantized weight: the file holds the raw bytes of the
                // fused tensor. The rows are contiguous in memory, so the
                // split into gate/up is a plain byte-range slice.
                auto data = get_param_bytes(weight_path);

                auto row_bytes = static_cast<size_t>(ggml_row_size(*type, dim));

                if (data.size() != row_bytes * (inner * 2))
                    throw std::runtime_error(
                        "Flux2FeedForward: quantized weight '" + weight_path +
                        "' has " + std::to_string(data.size()) + " bytes, expected " +
                        std::to_string(row_bytes * (inner * 2)));

                gate_weight->set(create_quantized(weight_path + "-gate-weight", gate_weight->shape(), *type,
                    slice_rows_bytes(data, *type, dim, 0, inner)));

                up_weight->set(create_quantized(weight_path + "-up-weight", up_weight->shape(), *type,
                    slice_rows_bytes(data, *type, dim, inner, inner * 2)));
            }
            else {
                auto weight_value = get_param(weight_path);
                ArgumentParser::parser<Tensor>::TensorParser parser(weight_path, weight_value);

                auto [shape, data] = parser.parse();

                gate_weight->set(context().create<float>(gate_weight->shape(), [=](std::mt19937&) {
                    return slice_rows(data, dim, 0, inner);
                }).name(weight_path + "-gate-weight"));

                up_weight->set(context().create<float>(up_weight->shape(), [=](std::mt19937&) {
                    return slice_rows(data, dim, inner, inner * 2);
                }).name(weight_path + "-up-weight"));
            }

            auto gate_bias = feed_forward.linear_in_gate()->bias();

            if (gate_bias) {
                auto bias_path = module_path(path, prefix(), {"linear_in", "bias"});

                // The fused bias is 1-D of length inner * 2: the first
                // inner elements belong to the gate, the last inner to
                // the up projection.
                if (auto type = quantization().resolve(dot_name(path) + ".linear_in.bias", prefix()); type && *type != GGML_TYPE_F32) {
                    auto data = get_param_bytes(bias_path);

                    auto up_bias = feed_forward.linear_in_up()->bias();

                    gate_bias->set(create_quantized(bias_path + "-gate-bias", gate_bias->shape(), *type,
                        slice_cols_bytes(data, *type, 1, inner * 2, 0, inner)));

                    up_bias->set(create_quantized(bias_path + "-up-bias", up_bias->shape(), *type,
                        slice_cols_bytes(data, *type, 1, inner * 2, inner, inner * 2)));
                }
                else {
                    auto bias_value = get_param(bias_path);
                    ArgumentParser::parser<Tensor>::TensorParser bias_parser(bias_path, bias_value);

                    auto [bias_shape, bias_data] = bias_parser.parse();

                    auto up_bias = feed_forward.linear_in_up()->bias();

                    gate_bias->set(context().create<float>(gate_bias->shape(), [=](std::mt19937&) {
                        return slice_cols(bias_data, 1, inner * 2, 0, inner);
                    }).name(bias_path + "-gate-bias"));

                    up_bias->set(context().create<float>(up_bias->shape(), [=](std::mt19937&) {
                        return slice_cols(bias_data, 1, inner * 2, inner, inner * 2);
                    }).name(bias_path + "-up-bias"));
                }
            }

            auto linear_out = feed_forward.linear_out();

            auto out_weight_path = path;
            out_weight_path.push_back("linear_out");
            out_weight_path.push_back("weight");
            TestCLI::CreateParametersVisitor::visit(*linear_out->weight(), std::move(out_weight_path));

            if (auto out_bias = linear_out->bias()) {
                auto out_bias_path = path;
                out_bias_path.push_back("linear_out");
                out_bias_path.push_back("bias");
                TestCLI::CreateParametersVisitor::visit(*out_bias, std::move(out_bias_path));
            }
        }

    private:
        // PyTorch: x[start:end]
        template <typename T>
        static std::vector<T> slice_rows(
            const std::vector<T>& x,
            size_t cols,
            size_t start,
            size_t end)
        {
            return {
                x.begin() + start * cols,
                x.begin() + end * cols
            };
        }

        // PyTorch: x[:, start:end]
        template <typename T>
        static std::vector<T> slice_cols(
            const std::vector<T>& x,
            size_t rows,
            size_t cols,
            size_t start,
            size_t end)
        {
            std::vector<T> out;
            out.reserve(rows * (end - start));

            for (size_t r = 0; r < rows; ++r) {
                out.insert(
                    out.end(),
                    x.begin() + r * cols + start,
                    x.begin() + r * cols + end);
            }

            return out;
        }
        // The raw-bytes counterpart of slice_rows: rows of a 2-D quantized
        // tensor are contiguous, each row is ggml_row_size bytes long.
        static std::vector<std::byte> slice_rows_bytes(
            const std::vector<std::byte>& x,
            ggml_type type,
            size_t cols,
            size_t start,
            size_t end)
        {
            auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));

            return {
                x.begin() + start * row_bytes,
                x.begin() + end * row_bytes
            };
        }

        // The raw-bytes counterpart of slice_cols: the column range must
        // fall on quantization block boundaries, so the slice is a
        // per-row byte range.
        static std::vector<std::byte> slice_cols_bytes(
            const std::vector<std::byte>& x,
            ggml_type type,
            size_t rows,
            size_t cols,
            size_t start,
            size_t end)
        {
            auto blck = static_cast<size_t>(ggml_blck_size(type));

            if (start % blck != 0 || (end - start) % blck != 0)
                throw std::runtime_error(
                    "The quantized fused weight column range [" + std::to_string(start) + ", " +
                    std::to_string(end) + ") is not aligned to the quantization block size " +
                    std::to_string(blck));

            auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));
            auto block_bytes = static_cast<size_t>(ggml_type_size(type));
            auto slice_offset = start / blck * block_bytes;
            auto slice_bytes = (end - start) / blck * block_bytes;

            std::vector<std::byte> out;
            out.reserve(rows * slice_bytes);

            for (size_t r = 0; r < rows; ++r)
                out.insert(
                    out.end(),
                    x.begin() + r * row_bytes + slice_offset,
                    x.begin() + r * row_bytes + slice_offset + slice_bytes);

            return out;
        }

    };
};

int main(int argc, char** argv) {
    TestFlux2CLI cli(argc, argv);
    return cli.main();
}
