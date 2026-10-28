#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/autoencoders/vae/Encoder.hpp"
#include "diffusers/models/autoencoders/vae/Decoder.hpp"
#include "nn/modules/BatchNorm2d.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

AutoencoderKLFlux2::AutoencoderKLFlux2(const AutoencoderKLFlux2::Config& config) : AutoencoderKLFlux2(
    config.in_channels,
    config.out_channels,
    config.block_out_channels,
    config.layers_per_block,
    config.latent_channels,
    config.norm_num_groups,
    config.sample_size,
    config.force_upcast,
    config.use_quant_conv,
    config.use_post_quant_conv,
    config.mid_block_add_attention,
    config.batch_norm_eps,
    config.batch_norm_momentum,
    config.patch_size
) {

}

AutoencoderKLFlux2::AutoencoderKLFlux2(
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
) : use_quant_conv_(use_quant_conv),
    use_post_quant_conv_(use_post_quant_conv),
    out_channels_(out_channels),
    latent_channels_(latent_channels),
    batch_norm_eps_(batch_norm_eps),
    block_out_channels_(block_out_channels)
{
    modules["encoder"] = std::make_shared<Encoder>(
        in_channels,
        latent_channels, // out_channels
        block_out_channels, 
        layers_per_block,
        norm_num_groups,
        true, // double_z
        mid_block_add_attention
    );

    modules["decoder"] = std::make_shared<Decoder>(
        latent_channels, // in_channels
        out_channels,
        block_out_channels,
        layers_per_block,
        norm_num_groups,
        mid_block_add_attention
    );

    const auto patch_h = std::get<0>(patch_size);
    const auto patch_w = std::get<1>(patch_size);
    
    modules["bn"] = std::make_shared<BatchNorm2d>(
        patch_h * patch_w * latent_channels,
        batch_norm_eps,
        batch_norm_momentum
    );

    if (use_quant_conv_)
        modules["quant_conv"] = std::make_shared<Conv2d>(
            2 * latent_channels, // in_channels
            2 * latent_channels, // out_channels
            1, // kernel_size
            1, // stride
            0 // padding
        );

    if (use_post_quant_conv_)
        modules["post_quant_conv"] = std::make_shared<Conv2d>(
            latent_channels, // in_channels
            latent_channels, // out_channels
            1 // kernel_size
        );
}

Tensor AutoencoderKLFlux2::forward(Scope scope, Tensor sample, bool sample_posterior) {
    auto x = sample;
    auto posterior = encode(scope, x);

    Tensor z;

    if (sample_posterior)
        z = posterior.sample(scope);
    else
        z = posterior.mode();
    
    auto dec = decode(scope, z);

    return dec;
}

DiagonalGaussianDistribution AutoencoderKLFlux2::encode(Scope& scope, Tensor x) {
    auto encoder = std::static_pointer_cast<Encoder>(modules["encoder"]);

    auto enc = encoder->forward(scope, x);

    if (use_quant_conv_) {
        auto quant_conv = std::static_pointer_cast<Conv2d>(modules["quant_conv"]);

        enc = quant_conv->forward(scope, enc);
    }

    auto posterior = DiagonalGaussianDistribution(scope, enc);

    return posterior;
}

Tensor AutoencoderKLFlux2::decode(Scope& scope, Tensor z) {
    if (use_post_quant_conv_) {
        auto post_quant_conv = std::static_pointer_cast<Conv2d>(modules["post_quant_conv"]);

        z = post_quant_conv->forward(scope, z);
    }

    auto decoder = std::static_pointer_cast<Decoder>(modules["decoder"]);

    auto dec = decoder->forward(scope, z);

    return dec;
}

//
// Blend the first `extent` columns of b with the last `extent` columns of
// a with a linear ramp (diffusers' blend_h):
//
//   b[:, :, :, x] = a[:, :, :, -extent + x] * (1 - w[x])
//                 + b[:, :, :, x]           * w[x]
//
// where w[x] = x / extent, the diffusers' ramp.
static Tensor blend_h(Scope& scope, const Tensor& a, const Tensor& b, int64_t extent) {
    extent = std::min({a.shape()[3], b.shape()[3], extent});

    if (extent <= 0)
        return b;

    // w[x] = x / extent (diffusers: (1 - x / blend_extent), x / blend_extent).
    auto w = Scope::context().create<float>(
        {1, 1, 1, extent},
        [extent](std::mt19937&) {
            std::vector<float> data(extent);

            for (int64_t x = 0; x < extent; ++x)
                data[x] = static_cast<float>(x) / static_cast<float>(extent);

            return data;
        });

    auto a_tail = a[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(a.shape()[3] - extent, std::nullopt)
    }];

    auto b_head = b[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(0, extent)
    }];

    auto head = a_tail * (1.0f - w) + b_head * w;

    if (extent >= b.shape()[3])
        return head;

    auto b_rest = b[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(extent, std::nullopt)
    }];

    return Tensor::cat({head, b_rest}, -1);
}

// The blend_h counterpart along the height axis (diffusers' blend_v).
static Tensor blend_v(Scope& scope, const Tensor& a, const Tensor& b, int64_t extent) {
    extent = std::min({a.shape()[2], b.shape()[2], extent});

    if (extent <= 0)
        return b;

    auto w = Scope::context().create<float>(
        {1, 1, extent, 1},
        [extent](std::mt19937&) {
            std::vector<float> data(extent);

            for (int64_t y = 0; y < extent; ++y)
                data[y] = static_cast<float>(y) / static_cast<float>(extent);

            return data;
        });

    auto a_tail = a[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(a.shape()[2] - extent, std::nullopt),
        Tensor::Slice::all()
    }];

    auto b_head = b[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(0, extent),
        Tensor::Slice::all()
    }];

    auto head = a_tail * (1.0f - w) + b_head * w;

    if (extent >= b.shape()[2])
        return head;

    auto b_rest = b[{
        Tensor::Slice::all(),
        Tensor::Slice::all(),
        Tensor::Slice::range(extent, std::nullopt),
        Tensor::Slice::all()
    }];

    return Tensor::cat({head, b_rest}, -2);
}

//
// Tiled forward, mirroring diffusers' tiled_decode()/_tiled_encode():
//
// - The input (B, C, H, W) is tiled with a 3/4 stride, so adjacent tiles
//   overlap by a quarter of the tile size (clipped at the input borders).
// - Each tile is passed through `fn` (the full VAE) in its own graph
//   scope, so no VAE graph ever contains more than one tile at a time:
//   the tile count is not bounded by a single context's capacity, and
//   the live VAE activations are one tile plus the finished tiles.
// - Each tile is blended with the tile above (blend_v) and the tile to
//   the left (blend_h) with a linear ramp over the blend extent. Like
//   diffusers, the blending is in place: the blended tile replaces its
//   raw tile, so every blend sees the already blended neighbors -- here
//   as the state cells the earlier tile scopes persisted.
// - A final scope keeps the first band_h x band_w output region of each
//   blended tile and concatenates the bands; the quarters dropped on
//   the right and the bottom are the overlap regions that feed the next
//   tiles' blending.
//
// The tile size is given in the input space of the method (pixels for
// encode, latent units for decode); the blend extent and the band are
// given in the output space.
//
template <class Fn>
static std::vector<Computation<Tensor>> tiled_forward(
    Computation<Tensor> input,
    int64_t tile_h,
    int64_t tile_w,
    int64_t blend_extent_h,
    int64_t blend_extent_w,
    int64_t band_h,
    int64_t band_w,
    Fn fn)
{
    if (tile_h <= 0 || tile_w <= 0)
        throw std::invalid_argument("tiled_forward(): tile size must be positive");

    // 3/4 stride (diffusers: overlap_size = int(size * (1 - 0.25))).
    const int64_t stride_h = tile_h * 3 / 4;
    const int64_t stride_w = tile_w * 3 / 4;

    if (stride_h <= 0 || stride_w <= 0)
        throw std::invalid_argument("tiled_forward(): tile size too small (3/4 of it must be >= 1)");

    // The input every tile scope reads. Materialized into the state
    // context on the first dereference (it is a scope output), so all
    // the tile scopes slice the same persistent tensor.
    const Tensor in = *input;

    const int64_t H = in.shape()[2];
    const int64_t W = in.shape()[3];

    // One tile covers the whole input: the untiled path is exact (the
    // tile slice would be the identity and there are no overlaps to
    // blend).
    if (H <= tile_h && W <= tile_w)
        return {
            input.scope([&fn](Scope scope, Tensor in) -> Tensor {
                return fn(scope, in);
            })
        };

    // One scope per tile: the VAE pass on the tile slice plus the blends
    // with the (already blended) up and left neighbors. The scope's
    // output is the blended tile; it is persisted into the state context
    // when a neighbor or the final scope dereferences it.
    std::vector<std::vector<Computation<Tensor>>> rows;

    for (int64_t i = 0; i < H; i += stride_h) {
        const size_t row_i = rows.size();
        std::vector<Computation<Tensor>> row;

        for (int64_t j = 0; j < W; j += stride_w) {
            const size_t col_j = row.size();

            auto up = (row_i > 0)
                ? std::make_optional(*rows[row_i - 1][col_j])
                : std::nullopt;

            auto left = (col_j > 0)
                ? std::make_optional(*row[col_j - 1])
                : std::nullopt;

            row.push_back(input.scope([&, i, j](Scope scope, Tensor in) -> Tensor {
                auto tile = in[{
                    Tensor::Slice::all(),
                    Tensor::Slice::all(),
                    Tensor::Slice::range(i, i + tile_h),
                    Tensor::Slice::range(j, j + tile_w)
                }];

                auto out = fn(scope, tile);

                if (up)
                    out = blend_v(scope, *up, out, blend_extent_h);

                if (left)
                    out = blend_h(scope, *left, out, blend_extent_w);

                return out;
            }));
        }

        rows.push_back(std::move(row));
    }

    // The final scope: from each blended tile keep the top-left band
    // (diffusers: row_limit = tile - blend_extent; clipped border tiles
    // keep whatever they have left) and concatenate the bands into the
    // full output.
    auto result = input.scope([&, band_h, band_w](Scope scope) -> Tensor {
        std::vector<Tensor> out_rows;

        for (size_t i = 0; i < rows.size(); ++i) {
            std::vector<Tensor> bands;

            for (size_t j = 0; j < rows[i].size(); ++j) {
                const Tensor tile = *rows[i][j];

                const int64_t keep_h = std::min(tile.shape()[2], band_h);
                const int64_t keep_w = std::min(tile.shape()[3], band_w);

                bands.push_back(tile[{
                    Tensor::Slice::all(),
                    Tensor::Slice::all(),
                    Tensor::Slice::range(0, keep_h),
                    Tensor::Slice::range(0, keep_w)
                }]);
            }

            out_rows.push_back(Tensor::cat(bands, -1));
        }

        return Tensor::cat(out_rows, -2);
    });

    // Every tile's computation (row-major) followed by the final
    // computation: the caller can group them into an execution unit
    // (Computation::unit), so the ExecutionRuntime runs the whole tiled
    // pass as one progress step.
    std::vector<Computation<Tensor>> computations;

    for (auto& row : rows)
        for (auto& tile : row)
            computations.push_back(std::move(tile));

    computations.push_back(std::move(result));
    
    return computations;
}

std::vector<Computation<Tensor>> AutoencoderKLFlux2::decode_tiled(Computation<Tensor> z, int64_t tile_h, int64_t tile_w) {
    const int64_t scale = scale_factor();

    if (tile_h % scale != 0 || tile_w % scale != 0)
        throw std::invalid_argument("decode_tiled(): tile size must be divisible by the VAE's scale factor");

    // Tile geometry in latent units (the input space of decode()).
    const int64_t latent_tile_h = tile_h / scale;
    const int64_t latent_tile_w = tile_w / scale;

    // Blend extent and row limit in pixel units (the output space).
    //
    // diffusers: blend_extent = int(tile_sample_min_size * 0.25),
    //            row_limit    = tile_sample_min_size - blend_extent.
    const int64_t blend_h = tile_h / 4;
    const int64_t blend_w = tile_w / 4;
    const int64_t band_h = tile_h - blend_h;
    const int64_t band_w = tile_w - blend_w;

    return tiled_forward(std::move(z), latent_tile_h, latent_tile_w, blend_h, blend_w, band_h, band_w,
        [this](Scope& s, const Tensor& tile) -> Tensor {
            return decode(s, tile);
        });
}

std::vector<Computation<Tensor>> AutoencoderKLFlux2::encode_tiled(Computation<Tensor> sample, int64_t tile_h, int64_t tile_w) {
    const int64_t scale = scale_factor();

    if (tile_h % scale != 0 || tile_w % scale != 0)
        throw std::invalid_argument("encode_tiled(): tile size must be divisible by the VAE's scale factor");

    const int64_t latent_tile_h = tile_h / scale;
    const int64_t latent_tile_w = tile_w / scale;

    // Blend extent and row limit in latent units (the output space).
    //
    // diffusers: blend_extent = int(tile_latent_min_size * 0.25),
    //            row_limit    = tile_latent_min_size - blend_extent.
    const int64_t blend_h = latent_tile_h / 4;
    const int64_t blend_w = latent_tile_w / 4;
    const int64_t band_h = latent_tile_h - blend_h;
    const int64_t band_w = latent_tile_w - blend_w;

    return tiled_forward(std::move(sample), tile_h, tile_w, blend_h, blend_w, band_h, band_w,
        [this](Scope& s, const Tensor& tile) -> Tensor {
            return encode(s, tile).mode();
        });
}

std::shared_ptr<BatchNorm2d> AutoencoderKLFlux2::bn() const {
    return std::static_pointer_cast<BatchNorm2d>(modules.at("bn"));
}

template <typename T>
static void read(const json& j, const char* key, T& value) {
    if (!j.contains(key) || j[key].is_null())
        return;

    value = j[key].get<T>();
}

AutoencoderKLFlux2::Config AutoencoderKLFlux2::Config::from_file(const std::filesystem::path& path) {
    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Failed to open configuration file: " + path.string());

    json j;
    file >> j;

    Config cfg;

    read(j, "in_channels", cfg.in_channels);
    read(j, "out_channels", cfg.out_channels);

    read(j, "block_out_channels", cfg.block_out_channels);

    read(j, "layers_per_block", cfg.layers_per_block);
    read(j, "latent_channels", cfg.latent_channels);
    read(j, "norm_num_groups", cfg.norm_num_groups);
    read(j, "sample_size", cfg.sample_size);

    read(j, "force_upcast", cfg.force_upcast);
    read(j, "use_quant_conv", cfg.use_quant_conv);
    read(j, "use_post_quant_conv", cfg.use_post_quant_conv);
    read(j, "mid_block_add_attention", cfg.mid_block_add_attention);

    read(j, "batch_norm_eps", cfg.batch_norm_eps);
    read(j, "batch_norm_momentum", cfg.batch_norm_momentum);

    if (j.contains("patch_size") && !j["patch_size"].is_null()) {
        const auto& p = j["patch_size"];
        if (!p.is_array() || p.size() != 2) {
            throw std::runtime_error("patch_size must be an array of length 2");
        }

        cfg.patch_size = std::make_tuple(
            p[0].get<int64_t>(),
            p[1].get<int64_t>()
        );
    }

    return cfg;
}
