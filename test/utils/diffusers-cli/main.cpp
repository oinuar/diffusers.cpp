#include "../TestCLI.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

#include "diffusers/models/normalization/AdaLayerNormContinuous.hpp"
#include "diffusers/models/normalization/SpatialNorm.hpp"
#include "diffusers/models/resnet/Upsample2D.hpp"
#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/downsampling/Downsample2D.hpp"
#include "diffusers/models/embeddings/TimestepEmbedding.hpp"
#include "diffusers/models/embeddings/Timesteps.hpp"
#include "diffusers/models/attention_processor/Attention.hpp"
#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"


#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"
#include "diffusers/models/unets/unet2d/DownEncoderBlock2D.hpp"
#include "diffusers/models/unets/unet2d/UpDecoderBlock2D.hpp"

#include "diffusers/models/autoencoders/vae/Decoder.hpp"
#include "diffusers/models/autoencoders/vae/Encoder.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"

#include <numeric>

class TestDiffusersCLI : public TestCLI {
public:
    TestDiffusersCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Computation<std::vector<Tensor>> compute(Context& context) {

        if (args_.get(0) == "AdaLayerNormContinuous") {
            Computation<void> computation({&context});

            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto conditioning_embedding_dim = args_.get_one<int64_t>("--conditioning_embedding_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto conditioning_embedding = args_.get_one<Tensor>("--conditioning_embedding", {computation.desc()->context()});

            AdaLayerNormContinuous<> model(embedding_dim, conditioning_embedding_dim, elementwise_affine, eps, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states, conditioning_embedding);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "SpatialNorm") {
            Computation<void> computation({&context});

            auto f_channels = args_.get_one<int64_t>("--f_channels");
            auto zq_channels = args_.get_one<int64_t>("--zq_channels");
            auto f = args_.get_one<Tensor>("--f", {computation.desc()->context()});
            auto zq = args_.get_one<Tensor>("--zq", {computation.desc()->context()});

            SpatialNorm model(f_channels, zq_channels);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, f, zq);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Upsample2D") {
            Computation<void> computation({&context});

            auto channels = args_.get_one<int64_t>("--channels");
            auto use_conv = args_.get_optional<bool>("--use_conv").value_or(false);
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto use_conv_transpose = args_.get_optional<bool>("--use_conv_transpose").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});

            Upsample2D model(channels, use_conv, out_channels, use_conv_transpose);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Downsample2D") {
            Computation<void> computation({&context});

            auto channels = args_.get_one<int64_t>("--channels");
            auto use_conv = args_.get_optional<bool>("--use_conv").value_or(false);
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto padding = args_.get_optional<int64_t>("--padding").value_or(1);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});

            Downsample2D model(channels, use_conv, out_channels, padding);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "ResnetBlock2D") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto conv_shortcut = args_.get_optional<int64_t>("--conv_shortcut");
            auto temb_channels = args_.get_optional<int64_t>("--temb_channels");
            auto groups = args_.get_optional<int64_t>("--groups").value_or(32);
            auto groups_out = args_.get_optional<int64_t>("--groups_out");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6f);
            auto kernel_size = args_.get_optional<int64_t>("--kernel_size").value_or(3);
            auto output_scale_factor = args_.get_optional<int64_t>("--output_scale_factor").value_or(1);
            auto use_in_shortcut = args_.get_optional<bool>("--use_in_shortcut");
            auto conv_shortcut_bias = args_.get_optional<bool>("--conv_shortcut_bias").value_or(true);
            auto conv_2d_out_channels = args_.get_optional<int64_t>("--conv_2d_out_channels");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto temb = args_.get_optional<Tensor>("--temb", {computation.desc()->context()});

            ResnetBlock2D<SiLU> model(
                in_channels,
                out_channels,
                conv_shortcut,
                0.0f, // dropout
                temb_channels,
                groups,
                groups_out,
                true, // pre_norm
                eps,
                false, // skip_time_act
                kernel_size,
                output_scale_factor,
                use_in_shortcut,
                false, // down
                false, // up
                conv_shortcut_bias,
                conv_2d_out_channels
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states, temb);
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "Decoder") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_optional<int64_t>("--in_channels").value_or(3);
            auto out_channels = args_.get_optional<int64_t>("--out_channels").value_or(3);
            auto block_out_channels = args_.get_many<int64_t>("--block_out_channels");
            auto layers_per_block = args_.get_optional<int>("--layers_per_block").value_or(2);
            auto norm_num_groups = args_.get_optional<int>("--norm_num_groups").value_or(32);
            auto mid_block_add_attention = args_.get_optional<bool>("--mid_block_add_attention").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});
            auto latent_embeds = args_.get_optional<Tensor>("--latent_embeds", {computation.desc()->context()});

            Decoder model(
                in_channels,
                out_channels,
                block_out_channels,
                layers_per_block,
                norm_num_groups,
                mid_block_add_attention
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, sample, latent_embeds);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Encoder") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_optional<int64_t>("--in_channels").value_or(3);
            auto out_channels = args_.get_optional<int64_t>("--out_channels").value_or(3);
            auto block_out_channels = args_.get_many<int64_t>("--block_out_channels");
            auto layers_per_block = args_.get_optional<int>("--layers_per_block").value_or(2);
            auto norm_num_groups = args_.get_optional<int>("--norm_num_groups").value_or(32);
            auto double_z = args_.get_optional<bool>("--double_z").value_or(true);
            auto mid_block_add_attention = args_.get_optional<bool>("--mid_block_add_attention").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});

            Encoder model(
                in_channels,
                out_channels,
                block_out_channels,
                layers_per_block,
                norm_num_groups,
                double_z,
                mid_block_add_attention
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, sample);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "AutoencoderKLFlux2" ||
            args_.get(0) == "AutoencoderKLFlux2_encode_tiled" ||
            args_.get(0) == "AutoencoderKLFlux2_decode_tiled") {
            Computation<void> computation({&context});

            AutoencoderKLFlux2::Config config;

            config.in_channels = args_.get_optional<int64_t>("--in_channels").value_or(config.in_channels);
            config.out_channels = args_.get_optional<int64_t>("--out_channels").value_or(config.out_channels);
            auto block_out_channels = args_.get_many<int64_t>("--block_out_channels");
            config.layers_per_block = args_.get_optional<int64_t>("--layers_per_block").value_or(config.layers_per_block);
            config.latent_channels = args_.get_optional<int64_t>("--latent_channels").value_or(config.latent_channels);
            config.norm_num_groups = args_.get_optional<int64_t>("--norm_num_groups").value_or(config.norm_num_groups);
            config.sample_size = args_.get_optional<int64_t>("--sample_size").value_or(config.sample_size);
            config.force_upcast = args_.get_optional<bool>("--force_upcast").value_or(config.force_upcast);
            config.use_quant_conv = args_.get_optional<bool>("--use_quant_conv").value_or(config.use_quant_conv);
            config.use_post_quant_conv = args_.get_optional<bool>("--use_post_quant_conv").value_or(config.use_post_quant_conv);
            config.mid_block_add_attention = args_.get_optional<bool>("--mid_block_add_attention").value_or(config.mid_block_add_attention);
            config.batch_norm_eps = args_.get_optional<float>("--batch_norm_eps").value_or(config.batch_norm_eps);
            config.batch_norm_momentum = args_.get_optional<float>("--batch_norm_momentum").value_or(config.batch_norm_momentum);
            config.patch_size = std::make_tuple(
                args_.get_optional<int64_t>("--patch_size-0").value_or(std::get<0>(config.patch_size)),
                args_.get_optional<int64_t>("--patch_size-1").value_or(std::get<1>(config.patch_size))
            );
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});
            auto sample_posterior = args_.get_optional<bool>("--sample_posterior").value_or(false);
            auto tile_size = args_.get_optional<int64_t>("--tile_size");

            if (!block_out_channels.empty())
                config.block_out_channels = block_out_channels;

            AutoencoderKLFlux2 model(config);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            Computation<Tensor> result;

            if (args_.get(0) == "AutoencoderKLFlux2_encode_tiled")
                result = Computation<Tensor>::unit("VAE encode", model.encode_tiled(Computation<Tensor>(sample, computation.desc()), *tile_size, *tile_size));

            else if (args_.get(0) == "AutoencoderKLFlux2_decode_tiled")
                result = Computation<Tensor>::unit("VAE decode", model.decode_tiled(Computation<Tensor>(sample, computation.desc()), *tile_size, *tile_size));

            else
                result = computation.scope([&](Scope scope) -> Tensor {
                    return model.forward(scope, sample, sample_posterior);
                });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "TimestepEmbedding") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto time_embed_dim = args_.get_one<int64_t>("--time_embed_dim");
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto cond_proj_dim = args_.get_optional<int64_t>("--cond_proj_dim");
            auto sample_proj_bias = args_.get_optional<bool>("--sample_proj_bias").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});
            auto condition = args_.get_optional<Tensor>("--condition", {computation.desc()->context()});

            TimestepEmbedding<> model(in_channels, time_embed_dim, out_dim, cond_proj_dim, sample_proj_bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, sample, condition);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Timesteps") {
            Computation<void> computation({&context});

            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto flip_sin_to_cos = args_.get_one<bool>("--flip_sin_to_cos");
            auto downscale_freq_shift = args_.get_one<float>("--downscale_freq_shift");
            auto scale = args_.get_optional<float>("--scale").value_or(1.0);
            auto timesteps = args_.get_one<Tensor>("--timesteps", {computation.desc()->context()});

            Timesteps model(num_channels, flip_sin_to_cos, downscale_freq_shift, scale);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, timesteps);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Attention") {
            Computation<void> computation({&context});

            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_one<int64_t>("--heads");
            auto dim_head = args_.get_one<int64_t>("--dim_head");
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0f);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto residual_connection = args_.get_optional<bool>("--residual_connection").value_or(false);
            auto norm_num_groups = args_.get_optional<int64_t>("--norm_num_groups");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto rescale_output_factor = args_.get_optional<float>("--rescale_output_factor").value_or(1.0f);
            auto upcast_softmax = args_.get_optional<bool>("--upcast_softmax").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});

            Attention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                residual_connection,
                norm_num_groups,
                eps,
                rescale_output_factor,
                upcast_softmax
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states);
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "UNetMidBlock2D") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto temb_channels = args_.get_optional<int64_t>("--temb_channels");
            auto num_layers = args_.get_optional<int64_t>("--num_layers").value_or(1);
            auto resnet_eps = args_.get_optional<float>("--resnet_eps").value_or(1e-6);
            auto output_scale_factor = args_.get_optional<float>("--output_scale_factor").value_or(1.0f);
            auto attention_head_dim = args_.get_optional<int64_t>("--attention_head_dim").value_or(1);
            auto resnet_groups = args_.get_optional<int64_t>("--resnet_groups").value_or(32);
            auto add_attention = args_.get_optional<bool>("--add_attention").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});
            auto temb = args_.get_optional<Tensor>("--temb", {computation.desc()->context()});

            UNetMidBlock2D model(
                in_channels,
                temb_channels,
                0.0f, // dropout
                num_layers, 
                resnet_eps,
                resnet_groups,
                std::nullopt, // attn_groups
                true, // resnet_pre_norm
                add_attention,
                attention_head_dim,
                output_scale_factor
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, sample, temb);
            });

            return Computation<Tensor>::all(result);
        }
        
        if (args_.get(0) == "DownEncoderBlock2D") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto num_layers = args_.get_optional<int64_t>("--num_layers").value_or(1);
            auto resnet_groups = args_.get_optional<int64_t>("--resnet_groups").value_or(32);
            auto output_scale_factor = args_.get_optional<float>("--output_scale_factor").value_or(1.0f);
            auto add_downsample = args_.get_optional<bool>("--add_downsample").value_or(true);
            auto downsample_padding = args_.get_optional<int64_t>("--downsample_padding").value_or(1);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});

            DownEncoderBlock2D model(
                in_channels,
                out_channels,
                0.0f, // dropout
                num_layers,
                1e-6, // resnet_eps
                resnet_groups,
                true, // resnet_pre_norm
                output_scale_factor,
                add_downsample,
                downsample_padding
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "UpDecoderBlock2D") {
            Computation<void> computation({&context});

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto num_layers = args_.get_optional<int64_t>("--num_layers").value_or(1);
            auto resnet_groups = args_.get_optional<int64_t>("--resnet_groups").value_or(32);
            auto output_scale_factor = args_.get_optional<float>("--output_scale_factor").value_or(1.0f);
            auto add_upsample = args_.get_optional<bool>("--add_upsample").value_or(true);
            auto temb_channels = args_.get_optional<int64_t>("--temb_channels");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {computation.desc()->context()});
            auto temb = args_.get_optional<Tensor>("--temb", {computation.desc()->context()});

            UpDecoderBlock2D model(
                in_channels,
                out_channels,
                std::nullopt, // resolution_idx
                0.0f, // dropout
                num_layers,
                1e-6, // resnet_eps
                resnet_groups,
                true, // resnet_pre_norm
                output_scale_factor,
                add_upsample,
                temb_channels
            );

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, hidden_states, temb);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "FlowMatchEulerDiscreteScheduler_step") {
            auto num_train_timesteps = args_.get_optional<int>("--num_train_timesteps").value_or(1000);
            auto shift = args_.get_optional<float>("--shift").value_or(1.0f);
            auto use_dynamic_shifting = args_.get_optional<bool>("--use_dynamic_shifting").value_or(false);
            auto shift_terminal = args_.get_optional<float>("--shift_terminal");
            auto invert_sigmas = args_.get_optional<bool>("--invert_sigmas").value_or(false);
            auto time_shift_type = args_.get_optional<std::string>("--time_shift_type").value_or("exponential");

            FlowMatchEulerDiscreteScheduler flow_match_scheduler(
                num_train_timesteps,
                shift,
                use_dynamic_shifting,
                shift_terminal,
                invert_sigmas,
                time_shift_type
            );

            auto num_inference_steps = args_.get_one<int>("--num_inference_steps");
            auto mu = args_.get_one<float>("--mu");

            auto schedule = flow_match_scheduler.schedule(num_inference_steps, mu);

            Computation<void> computation({&context});

            auto index = args_.get_one<int>("--index");
            auto model_output = args_.get_one<Tensor>("--model_output", {computation.desc()->context()});
            auto sample = args_.get_one<Tensor>("--sample", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                auto dt = scope.context().value<float>(
                    {1},
                    [dt = schedule[index].dt](std::mt19937&) { return std::vector<float>{dt}; }
                );

                auto next_sample = flow_match_scheduler.integrate(scope, model_output, sample, dt);

                return next_sample;
            });

            return Computation<Tensor>::all(result);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

    virtual size_t get_graph_size() const {
        // The tiled VAE commands run the full VAE once per tile.
        if (args_.get(0) == "AutoencoderKLFlux2_encode_tiled" ||
            args_.get(0) == "AutoencoderKLFlux2_decode_tiled")
            return 16384;

        return TestCLI::get_graph_size();
    }
};

int main(int argc, char** argv) {
    TestDiffusersCLI cli(argc, argv);

    if (cli.args().get(0) == "FlowMatchEulerDiscreteScheduler_schedule") {
        auto num_train_timesteps = cli.args().get_optional<int>("--num_train_timesteps").value_or(1000);
        auto shift = cli.args().get_optional<float>("--shift").value_or(1.0f);
        auto use_dynamic_shifting = cli.args().get_optional<bool>("--use_dynamic_shifting").value_or(false);
        auto shift_terminal = cli.args().get_optional<float>("--shift_terminal");
        auto invert_sigmas = cli.args().get_optional<bool>("--invert_sigmas").value_or(false);
        auto time_shift_type = cli.args().get_optional<std::string>("--time_shift_type").value_or("exponential");

        FlowMatchEulerDiscreteScheduler flow_match_scheduler(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            shift_terminal,
            invert_sigmas,
            time_shift_type
        );

        auto num_inference_steps = cli.args().get_one<int>("--num_inference_steps");
        auto mu = cli.args().get_one<float>("--mu");

        auto schedule = flow_match_scheduler.schedule(num_inference_steps, mu);

        cli.print_tensor_like(schedule.timesteps(), {(int64_t)schedule.timesteps().size()});
        cli.print_tensor_like(schedule.sigmas(), {(int64_t)schedule.sigmas().size()});

        return EXIT_SUCCESS;
    }

    return cli.main();
}
