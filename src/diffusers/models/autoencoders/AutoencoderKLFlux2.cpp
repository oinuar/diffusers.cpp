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
    auto posterior = encode(scope, sample);

    auto z = sample_posterior ? posterior.sample(scope) : posterior.mode();

    return decode(scope, z);
}

DiagonalGaussianDistribution AutoencoderKLFlux2::encode(Scope& scope, Tensor x) {
    auto encoder = std::static_pointer_cast<Encoder>(modules["encoder"]);

    auto enc = encoder->forward(scope, x);

    if (use_quant_conv_) {
        auto quant_conv = std::static_pointer_cast<Conv2d>(modules["quant_conv"]);

        enc = quant_conv->forward(scope, enc);
    }

    return DiagonalGaussianDistribution(scope, enc);
}

Tensor AutoencoderKLFlux2::decode(Scope& scope, Tensor z) {
    if (use_post_quant_conv_) {
        auto post_quant_conv = std::static_pointer_cast<Conv2d>(modules["post_quant_conv"]);

        z = post_quant_conv->forward(scope, z);
    }

    auto decoder = std::static_pointer_cast<Decoder>(modules["decoder"]);

    return decoder->forward(scope, z);
}

// Blend the first `extent` of b's `axis` (2 = height, 3 = width) with
// the last `extent` of a's `axis` with a linear ramp (diffusers'
// blend_v for axis 2, blend_h for axis 3):
//
//   b[:, :, x, :] = a[:, :, -extent + x, :] * (1 - w[x])
//                  + b[:, :, x, :]           * w[x],   w[x] = x / extent
//
// The neighbors agree on the size of every dimension but `axis` (they
// share a grid row or column), so the ramp broadcasts over the blended
// region.
static Tensor blend(Scope& scope, const Tensor& a, const Tensor& b, int64_t extent, int64_t axis) {
    extent = std::min({a.shape()[axis], b.shape()[axis], extent});

    if (extent <= 0)
        return b;

    Tensor::Shape w_shape{1, 1, 1, 1};
    w_shape[axis] = extent;

    auto w = Scope::context().create<float>(
        w_shape,
        [extent](std::mt19937&) {
            std::vector<float> data(extent);

            for (int64_t x = 0; x < extent; ++x)
                data[x] = static_cast<float>(x) / static_cast<float>(extent);

            return data;
        });

    // A range slice on `axis`, all the other dimensions as-is.
    auto axis_range = [axis](std::optional<int64_t> start, std::optional<int64_t> stop) {
        std::vector<Tensor::Slice> s(4, Tensor::Slice::all());
        s[axis] = Tensor::Slice::range(start, stop);
        return s;
    };

    auto head = a[axis_range(a.shape()[axis] - extent, std::nullopt)] * (1.0f - w) + b[axis_range(0, extent)] * w;

    if (extent >= b.shape()[axis])
        return head;

    return Tensor::cat({head, b[axis_range(extent, std::nullopt)]}, axis - 4);
}

//
// Tiled forward, mirroring diffusers' tiled_decode()/_tiled_encode():
//
// - The input (B, C, H, W) is tiled with a 3/4 stride, so adjacent tiles
//   overlap by a quarter of the tile size (clipped at the input borders).
// - The equal-sized tiles form the rectangles of the grid (the tile
//   height depends only on the row, the width only on the column, and
//   each of them takes at most three values). Every rectangle runs in
//   its own repeated scope (Computation::fold): a single graph -- the
//   tile extraction, the VAE pass on the tile, and the state update --
//   is re-executed once per tile. The tile position is a function of
//   the loop clock (the get_rows index tensors and the slot one-hot are
//   per-iteration bindings, re-written before every execution), so the
//   VAE graph holds one tile per rectangle, no matter how many tiles
//   there are: the tile count is not bounded by a single context's
//   capacity, and the live VAE activations are the rectangle's tiles
//   plus the finished tiles.
// - Each tile is blended with the tile above (axis 2) and the tile to
//   the left (axis 3) with a linear ramp over the blend extent. The
//   blending is in place, so every blend sees the already blended
//   neighbors.
// - A final scope keeps the first band_h x band_w output region of each
//   blended tile (clipped to the true output extent on the borders) and
//   concatenates the bands; the quarters dropped on the right and the
//   bottom are the overlap regions that feed the next tiles' blending.
//
// The tile size is given in the input space of fn (pixels for encode,
// latent units for decode). out_fn() maps an input-space (h, w) to the
// (B, Cout, h', w') output the network produces for an (h, w) input;
// the blend extent and the band are derived from the full tile's output
// (diffusers: blend_extent = int(tile * 0.25),
// row_limit = tile - blend_extent).
//
template <class Fn, class OutFn>
static Computation<Tensor> tiled_forward(
    Computation<Tensor> input,
    int64_t tile_h,
    int64_t tile_w,
    OutFn out_fn,
    Fn fn)
{
    if (tile_h <= 0 || tile_w <= 0)
        throw std::invalid_argument("tiled_forward(): tile size must be positive");

    // 3/4 stride (diffusers: overlap_size = int(size * (1 - 0.25))).
    const int64_t stride_h = tile_h * 3 / 4;
    const int64_t stride_w = tile_w * 3 / 4;

    if (stride_h <= 0 || stride_w <= 0)
        throw std::invalid_argument("tiled_forward(): tile size too small (3/4 of it must be >= 1)");

    // The input every tile reads. Materialized into the state context on
    // the first dereference (it is a scope output), so all the tiles
    // slice the same persistent tensor.
    const Tensor in = *input;

    const int64_t B = in.shape()[0];
    const int64_t C_in = in.shape()[1];
    const int64_t H = in.shape()[2];
    const int64_t W = in.shape()[3];

    // One tile covers the whole input: the untiled path is exact (the
    // tile slice would be the identity and there are no overlaps to
    // blend).
    if (H <= tile_h && W <= tile_w)
        return input.scope([&fn](Scope scope, Tensor in) -> Tensor {
            return fn(scope, in);
        });

    // The tile grid: R x C tiles (row-major), the border tiles clipped
    // at the input borders (the input is zero-padded for the extraction
    // only).
    const int64_t R = (H - 1) / stride_h + 1;
    const int64_t C = (W - 1) / stride_w + 1;
    const int64_t H_pad = (R - 1) * stride_h + tile_h;
    const int64_t W_pad = (C - 1) * stride_w + tile_w;
    const size_t N = static_cast<size_t>(R) * static_cast<size_t>(C);

    // The blend extent and band in the output space of the full tile.
    const Tensor::Shape tile_out = out_fn(tile_h, tile_w);
    const int64_t blend_h = tile_out[2] / 4;
    const int64_t blend_w = tile_out[3] / 4;
    const int64_t band_h = tile_out[2] - blend_h;
    const int64_t band_w = tile_out[3] - blend_w;

    // The full image's output shape (the final scope's band crop clips
    // to it).
    const Tensor::Shape full_out = out_fn(H, W);
    const int64_t out_h = full_out[2];
    const int64_t out_w = full_out[3];

    // Zero-pad the input to the padded grid (one scope; the padded tensor
    // is a state cell the tile scopes read as a leaf). Skipped when the
    // grid already covers the input exactly.
    Tensor in_pad = in;

    if (H_pad != H || W_pad != W)
        in_pad = *input.scope([&, B, C_in, H, W, H_pad, W_pad](Scope scope, Tensor in) -> Tensor {
            auto out = in;

            if (H_pad != H)
                out = Tensor::cat({out, Tensor::zeros({B, C_in, H_pad - H, W})}, -2);

            if (W_pad != W)
                out = Tensor::cat({out, Tensor::zeros({B, C_in, H_pad, W_pad - W})}, -1);

            return out;
        });

    // A class: a rectangle of the grid, every tile of it the same input
    // size (tile_h x tile_w) and output shape out_fn(tile_h, tile_w).
    struct Class {
        int64_t i0, j0, rows, cols;
        int64_t tile_h, tile_w;
        Tensor::Shape out;
    };

    // The classes, row-major over the grid. The tile height (width)
    // depends only on the row (column) and is non-increasing along it,
    // so the rows (columns) of one height (width) are consecutive.
    std::vector<Class> classes;
    classes.reserve(9);

    for (int64_t i = 0; i < R; ) {
        const int64_t h = std::min(tile_h, H - i * stride_h);
        int64_t rows = 1;

        while (i + rows < R && std::min(tile_h, H - (i + rows) * stride_h) == h)
            rows++;

        for (int64_t j = 0; j < C; ) {
            const int64_t w = std::min(tile_w, W - j * stride_w);
            int64_t cols = 1;

            while (j + cols < C && std::min(tile_w, W - (j + cols) * stride_w) == w)
                cols++;

            classes.push_back({i, j, rows, cols, h, w, out_fn(h, w)});
            j += cols;
        }

        i += rows;
    }

    // The finished grid (row-major over the R x C tile grid).
    std::vector<Tensor> tiles(N);

    for (const auto& cls : classes) {
        // The class's state cells (one zero-initialized cell per tile),
        // overwritten by the class's output on every execution of its
        // repeated scope below.
        const size_t count = static_cast<size_t>(cls.rows) * static_cast<size_t>(cls.cols);

        std::vector<Tensor> cells(count);

        for (size_t k = 0; k < count; ++k) {
            int64_t numel = 1;

            for (int64_t d = 0; d < cls.out.rank(); ++d)
                numel *= cls.out[d];

            cells[k] = *input.state(cls.out, [numel](std::mt19937&) {
                return std::vector<float>(numel, 0.0f);
            });
        }

        Computation<std::vector<Tensor>> grid{cells, input.desc()};

        grid.fold(count, [&, cls, in_pad, B, C_in, stride_h, stride_w](
            Scope scope, size_t& iter, std::vector<Tensor> cells) -> std::vector<Tensor> {

            const size_t count = static_cast<size_t>(cls.rows) * static_cast<size_t>(cls.cols);

            // The get_rows index tensors (per-iteration: the providers
            // read the loop clock). The index tensor must be
            // [n_rows, ne2, ne3, 1] in ggml order: the tile's row
            // (column) indices, repeated over the (B, C_in) batch/channel
            // dimensions.
            auto row_idx = scope.context().value<int32_t>(
                {1, B, C_in, cls.tile_h},
                [&iter, B, C_in, cls, stride_h](std::mt19937&) {
                    const int64_t i = (cls.i0 + static_cast<int64_t>(iter) / cls.cols) * stride_h;
                    std::vector<int32_t> data(B * C_in * cls.tile_h);

                    for (int64_t b = 0; b < B; ++b)
                        for (int64_t c = 0; c < C_in; ++c)
                            for (int64_t r = 0; r < cls.tile_h; ++r)
                                data[(b * C_in + c) * cls.tile_h + r] = static_cast<int32_t>(i + r);

                    return data;
                });

            auto col_idx = scope.context().value<int32_t>(
                {1, B, C_in, cls.tile_w},
                [&iter, B, C_in, cls, stride_w](std::mt19937&) {
                    const int64_t j = (cls.j0 + static_cast<int64_t>(iter) % cls.cols) * stride_w;
                    std::vector<int32_t> data(B * C_in * cls.tile_w);

                    for (int64_t b = 0; b < B; ++b)
                        for (int64_t c = 0; c < C_in; ++c)
                            for (int64_t r = 0; r < cls.tile_w; ++r)
                                data[(b * C_in + c) * cls.tile_w + r] = static_cast<int32_t>(j + r);

                    return data;
                });

            // The slot one-hot (per-iteration): marks the class cell
            // this execution fills.
            auto slot = scope.context().value<float>(
                {static_cast<int64_t>(count)},
                [&iter, count](std::mt19937&) {
                    std::vector<float> data(count, 0.0f);
                    data[iter] = 1.0f;
                    return data;
                });

            // Extract the tile (B, C_in, cls.tile_h, cls.tile_w) from
            // the padded input: get_rows gathers the tile's rows, then
            // its columns (get_rows always gathers the second dimension;
            // the transposes put the target axis in place). get_rows
            // also requires its input contiguous along the first
            // dimension: the intermediate transpose is materialized.
            auto rows = Tensor(scope.runtime().get_rows(*in_pad, *row_idx), {B, C_in, cls.tile_h, W_pad});
            auto cols = Tensor(scope.runtime().get_rows(*rows.transpose(2, 3).contiguous(), *col_idx), {B, C_in, cls.tile_w, cls.tile_h});
            auto tile = cols.transpose(2, 3);

            auto out = fn(scope, tile);

            // Update the class's cells: the slot's cell takes the new
            // tile, every other cell passes through unchanged (grid +
            // (out - grid) * mask is exact on the passthrough slots: the
            // mask is 0 and x + 0 == x).
            std::vector<Tensor> next(count);

            for (size_t m = 0; m < count; ++m)
                next[m] = cells[m] + (out - cells[m]) * slot[{m}];

            return next;
        });

        for (int64_t i = 0; i < cls.rows; ++i)
            for (int64_t j = 0; j < cls.cols; ++j)
                tiles[static_cast<size_t>((cls.i0 + i) * C + (cls.j0 + j))] = cells[static_cast<size_t>(i) * cls.cols + j];
    }

    // The final scope: blend the grid in place (every blend sees the
    // already blended up and left neighbors), keep the first band_h x
    //   band_w region of each blended tile -- clipped to the true output
    //   extent on the borders -- and concatenate the bands into the full
    //   output.
    Computation<std::vector<Tensor>> grid{tiles, input.desc()};

    return grid.scope([R, C, blend_h, blend_w, band_h, band_w, out_h, out_w](
        Scope scope, std::vector<Tensor> tiles) -> Tensor {

        // The (i, j) tile of the row-major grid.
        const auto at = [C](int64_t i, int64_t j) -> size_t {
            return static_cast<size_t>(i * C + j);
        };

        std::vector<Tensor> blended(tiles);

        for (int64_t i = 0; i < R; ++i)
            for (int64_t j = 0; j < C; ++j) {
                if (i > 0)
                    blended[at(i, j)] = blend(scope, blended[at(i - 1, j)], blended[at(i, j)], blend_h, 2);

                if (j > 0)
                    blended[at(i, j)] = blend(scope, blended[at(i, j - 1)], blended[at(i, j)], blend_w, 3);
            }

        std::vector<Tensor> bands(R);

        for (int64_t i = 0; i < R; ++i) {
            const int64_t keep_h = std::min(band_h, out_h - i * band_h);

            std::vector<Tensor> row(C);

            for (int64_t j = 0; j < C; ++j) {
                const int64_t keep_w = std::min(band_w, out_w - j * band_w);

                row[j] = blended[at(i, j)][{
                    Tensor::Slice::all(),
                    Tensor::Slice::all(),
                    Tensor::Slice::range(0, keep_h),
                    Tensor::Slice::range(0, keep_w)
                }];
            }

            bands[i] = Tensor::cat(row, -1);
        }

        auto result = Tensor::cat(bands, -2);

        // Tensor::cat() of a single tensor returns it as-is: single-band
        // rows or columns are then bare views of state cells. Give the
        // scope a real compute output.
        if (!result.is_contiguous())
            result = result.contiguous();

        return result;
    });
}

Computation<Tensor> AutoencoderKLFlux2::decode_tiled(Computation<Tensor> z, int64_t tile_h, int64_t tile_w) {
    const int64_t scale = scale_factor();

    if (tile_h % scale != 0 || tile_w % scale != 0)
        throw std::invalid_argument("decode_tiled(): tile size must be divisible by the VAE's scale factor");

    // The input's batch size (the out_fn() shapes below).
    const int64_t B = z->shape()[0];

    // The tiles in latent units (the input space of decode()); the
    // output is in pixels.
    return tiled_forward(std::move(z), tile_h / scale, tile_w / scale,
        [this, B, scale](int64_t h, int64_t w) {
            return Tensor::Shape{B, out_channels_, h * scale, w * scale};
        },
        [this](Scope& s, const Tensor& tile) -> Tensor {
            return decode(s, tile);
        });
}

Computation<Tensor> AutoencoderKLFlux2::encode_tiled(Computation<Tensor> sample, int64_t tile_h, int64_t tile_w) {
    const int64_t scale = scale_factor();

    if (tile_h % scale != 0 || tile_w % scale != 0)
        throw std::invalid_argument("encode_tiled(): tile size must be divisible by the VAE's scale factor");

    // The input's batch size (the out_fn() shapes below).
    const int64_t B = sample->shape()[0];

    // The tiles in pixels (the input space of encode()); the output is
    // in latent units.
    return tiled_forward(std::move(sample), tile_h, tile_w,
        [this, B, scale](int64_t h, int64_t w) {
            return Tensor::Shape{B, latent_channels_, (h + scale - 1) / scale, (w + scale - 1) / scale};
        },
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
