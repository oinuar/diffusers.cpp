#pragma once

#include "nn/Module.hpp"
#include "ggml/Computation.hpp"
#include "diffusers/models/autoencoders/vae/DiagonalGaussianDistribution.hpp"
#include <filesystem>

class BatchNorm2d;

class AutoencoderKLFlux2 : public Module {
public:
    struct Config {
        static Config from_file(const std::filesystem::path& path);

        int64_t in_channels = 3;
        int64_t out_channels = 3;

        std::vector<int64_t> block_out_channels = {128, 256, 512, 512};

        int64_t layers_per_block = 2;
        int64_t latent_channels = 32;
        int64_t norm_num_groups = 32;
        int64_t sample_size = 1024;

        bool force_upcast = true;
        bool use_quant_conv = true;
        bool use_post_quant_conv = true;
        bool mid_block_add_attention = true;

        float batch_norm_eps = 1e-4f;
        float batch_norm_momentum = 0.1f;

        std::tuple<int64_t, int64_t> patch_size = {2, 2};
    };

    explicit AutoencoderKLFlux2(const Config& config);

    DiagonalGaussianDistribution encode(Scope& scope, Tensor x);

    Tensor decode(Scope& scope, Tensor z);

    //
    // Tiled variants of decode()/encode(), mirroring diffusers'
    // tiled_decode()/_tiled_encode(): the input is split into overlapping
    // tiles (3/4 stride), each tile is run through the full VAE in its
    // own scope, and the overlaps are blended with linear ramps (a
    // quarter of the tile size).
    //
    // This bounds the live memory of the full-resolution VAE activations
    // to the tile size regardless of the image size: the tile count is
    // not bounded by a single context's capacity.
    //
    // Returns every tile's computation (row-major) followed by the
    // computation of the final blended result: the caller can group the
    // vector into an execution unit (Computation::unit), so the
    // ExecutionRuntime runs the whole tiled pass as one progress step.
    //
    // tile_h/tile_w are the tile sizes in pixels (they must be divisible
    // by the VAE's scale factor).
    //
    std::vector<Computation<Tensor>> decode_tiled(Computation<Tensor> z, int64_t tile_h, int64_t tile_w);

    // As decode_tiled(), for encode(): the final computation is the
    // blended latent mode (B, latent_channels, H / scale, W / scale).
    std::vector<Computation<Tensor>> encode_tiled(Computation<Tensor> sample, int64_t tile_h, int64_t tile_w);

    Tensor forward(Scope scope, Tensor sample, bool sample_posterior = false);
    
    std::shared_ptr<BatchNorm2d> bn() const;

    int64_t scale_factor() const {
        return 1LL << (block_out_channels_.size() - 1);
    }

    int64_t latent_channels() const {
        return latent_channels_;
    }

    float batch_norm_eps() const {
        return batch_norm_eps_;
    }

private:
    bool use_quant_conv_;
    bool use_post_quant_conv_;
    int64_t out_channels_;
    int64_t latent_channels_;
    float batch_norm_eps_;
    std::vector<int64_t> block_out_channels_;

    AutoencoderKLFlux2(
        int64_t in_channels,
        int64_t out_channels,
        const std::vector<int64_t>& block_out_channels,
        int64_t layers_per_block,
        int64_t latent_channels,
        int64_t norm_num_groups,
        int64_t sample_size,
        bool force_upcast,
        bool use_quant_conv,
        bool use_post_quant_conv,
        bool mid_block_add_attention,
        float batch_norm_eps,
        float batch_norm_momentum,
        std::tuple<int64_t, int64_t> patch_size
    );
};
