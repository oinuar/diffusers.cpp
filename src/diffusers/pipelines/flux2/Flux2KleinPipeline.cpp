#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/modules/BatchNorm2d.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Context.hpp"
#include "ggml/Computation.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "ProgressBar.hpp"
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <optional>


// 4D txt_ids: (B, L, 4) -> [0, 0, 0, l]
static Tensor prepare_txt_ids(Scope scope, int batch, int64_t seq_len) {
    // matches:
    //  t = torch.arange(1)
    //  h = torch.arange(1)
    //  w = torch.arange(1)
    //  l = torch.arange(L)
    //  coords = torch.cartesian_prod(t, h, w, l)
    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>({batch, seq_len, 4},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * seq_len * 4, 0);
            for (int b = 0; b < batch; ++b) {
                for (int l = 0; l < seq_len; ++l) {
                    int32_t* row = &ids[(size_t(b) * seq_len + l) * 4];
                    row[0] = 0; // T
                    row[1] = 0; // H
                    row[2] = 0; // W
                    row[3] = l; // L
                }
            }
            return std::move(ids);
        });
}

// 4D img_ids: (B, N, 4) -> [0, y, x, 0]
static Tensor prepare_img_ids(Scope scope, int batch, int packed_h, int packed_w) {
    int64_t N = int64_t(packed_h) * packed_w;

    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>({batch, N, 4},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * N * 4, 0);
            for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < packed_h; ++y) {
                    for (int x = 0; x < packed_w; ++x) {
                        int32_t* row = &ids[(size_t(b) * N + y * packed_w + x) * 4];
                        row[0] = 0; // T
                        row[1] = y; // H
                        row[2] = x; // W
                        row[3] = 0; // L
                    }
                }
            }
            return std::move(ids);
        });
}

// 4D image_latent_ids for img2img: (B, N_ref, 4) -> [scale + i*scale, y, x, 0]
static Tensor prepare_ref_image_ids(Scope scope, int batch, const std::vector<Image>& images, int vae_multiple) {
    int64_t total_tokens = 0;

    for (const auto& image : images) {
        const int packed_h = static_cast<int>(image.height() / vae_multiple);
        const int packed_w = static_cast<int>(image.width() / vae_multiple);

        total_tokens += int64_t(packed_h) * packed_w;
    }

    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>(
        {batch, total_tokens, 4},
        [=, &images](std::mt19937&) {
            std::vector<int32_t> ids(
                size_t(batch) * total_tokens * 4,
                0);

            for (int b = 0; b < batch; ++b) {
                int64_t offset = 0;

                for (size_t i = 0; i < images.size(); ++i) {
                    const auto& image = images[i];

                    const int packed_h = static_cast<int>(image.height() / vae_multiple);
                    const int packed_w = static_cast<int>(image.width() / vae_multiple);

                    const int32_t t =
                        10 +
                        10 * static_cast<int32_t>(i);

                    for (int y = 0; y < packed_h; ++y) {
                        for (int x = 0; x < packed_w; ++x) {
                            int32_t* row =
                                &ids[
                                    (size_t(b) * total_tokens +
                                     offset) * 4
                                ];

                            row[0] = t;
                            row[1] = y;
                            row[2] = x;
                            row[3] = 0;

                            ++offset;
                        }
                    }
                }
            }

            return ids;
        });
}

// Python compute_empirical_mu
static float compute_empirical_mu(int64_t image_seq_len, int num_steps) {
    const float a1 = 8.73809524e-05f, b1 = 1.89833333f;
    const float a2 = 0.00016927f, b2 = 0.45666666f;
    if (image_seq_len > 4300) {
        return a2 * (float)image_seq_len + b2;
    }
    float m_200 = a2 * (float)image_seq_len + b2;
    float m_10 = a1 * (float)image_seq_len + b1;
    float a = (m_200 - m_10) / 190.0f;
    float b = m_200 - 200.0f * a;
    return a * (float)num_steps + b;
}

template <class T>
static Computation<Tensor> make_packed_latents(Computation<T> computation, int batch, int packed_h, int packed_w, int num_latent_channels) {
    const int64_t token_dim = int64_t(num_latent_channels) * 4; // C * 4
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    return computation.state({batch, int64_t(packed_h) * packed_w, token_dim},
        [=](std::mt19937& rng) {
            std::vector<float> noise(count);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (float& v : noise) v = normal(rng);
            return std::move(noise);
        });
}

template <class T>
static Computation<Tensor> make_init_latents(Computation<T> computation, int batch, int packed_h, int packed_w, int num_latent_channels, std::vector<float>&& init_latents) {
    const int64_t token_dim = int64_t(num_latent_channels) * 4; // C * 4

    return computation.state({batch, int64_t(packed_h) * packed_w, token_dim},
        [init_latents = std::move(init_latents)](std::mt19937&) {
            return init_latents;
        });
}

std::vector<Image> Flux2KleinPipeline::to_images(const Tensor::Shape& shape, std::vector<float>&& data) {
    std::vector<Image> images;

    auto batch = shape[0];
    auto h = (size_t)shape[2];
    auto w = (size_t)shape[3];
    auto plane = w * h;
    constexpr size_t channels = 3;

    images.reserve(batch);

    for (int b = 0; b < batch; ++b) {
        std::vector<uint8_t> pixels(plane * channels, 0);
        const float* src = data.data() + size_t(b) * channels * plane;

        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                for (size_t c = 0; c < channels; ++c) {
                    float v = src[c * plane + y * w + x];
                    v = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
                    pixels[(y * w + x) * channels + c] = static_cast<uint8_t>(std::lrint(v * 255.0f));
                }

        images.emplace_back(w, h, channels, std::move(pixels));
    }

    return std::move(images);
}

// HWC uint8 -> CHW float32 in [-1, 1].
//
// Matches VaeImageProcessor.preprocess(): (image / 255.0) * 2.0 - 1.0,
// mapping [0, 255] to [-1, 1].
static std::vector<float> image_to_tensor_data(const Image& img) {
    const size_t width = img.width();
    const size_t height = img.height();
    const size_t channels = img.channels();

    if (channels != 3)
        throw std::invalid_argument(
            "image_to_tensor_data(): expected RGB image with 3 channels, got " +
            std::to_string(channels));

    const auto& pixels = img.pixels();

    const size_t expected_size = width * height * channels;
    if (pixels.size() != expected_size)
        throw std::invalid_argument(
            "image_to_tensor_data(): pixel buffer size does not match image dimensions");

    const size_t plane = width * height;

    std::vector<float> data(3 * plane);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t src = (y * width + x) * 3;
            const size_t dst = y * width + x;

            data[0 * plane + dst] =
                2.0f * (static_cast<float>(pixels[src + 0]) / 255.0f) - 1.0f;

            data[1 * plane + dst] =
                2.0f * (static_cast<float>(pixels[src + 1]) / 255.0f) - 1.0f;

            data[2 * plane + dst] =
                2.0f * (static_cast<float>(pixels[src + 2]) / 255.0f) - 1.0f;
        }
    }

    return data;
}

Image Flux2KleinPipeline::preprocess_reference_image(const Image& image, int multiple, double target_area) {
    if (image.channels() != 3)
        throw std::invalid_argument(
            "preprocess_reference_image(): expected RGB image");

    const double width = static_cast<double>(image.width());
    const double height = static_cast<double>(image.height());
    const double area = width * height;

    // Flux2/Klein normalizes the input image to approximately
    // one megapixel while preserving the original aspect ratio.
    //
    // Do not upscale images that are already <= target area.
    const double scale = std::min(1.0, std::sqrt(target_area / area));

    auto target_width = static_cast<size_t>(std::round(width * scale));
    auto target_height = static_cast<size_t>(std::round(height * scale));

    // The VAE operates on dimensions that must be divisible by
    // its spatial scale factor, and Flux2 subsequently performs
    // 2x2 latent patchification.
    target_width = std::max<size_t>(multiple, (target_width / multiple) * multiple);
    target_height = std::max<size_t>(multiple, (target_height / multiple) * multiple);

    // resize_and_crop() preserves the aspect ratio and center-crops
    // to the exact dimensions required by the VAE.
    return image.resize_and_crop(target_width, target_height);
}

Flux2KleinPipeline::Flux2KleinPipeline(
    Flux2Transformer2DModel&& transformer,
    AutoencoderKLFlux2&& vae,
    Qwen3ForCausalLM&& text_encoder,
    Qwen2TokenizerFast&& tokenizer
) : transformer_(std::move(transformer)),
    vae_(std::move(vae)),
    text_encoder_(std::move(text_encoder)),
    tokenizer_(std::move(tokenizer))
{
}

Computation<Tensor> Flux2KleinPipeline::operator()(
    Context& vae_context,
    Context& text_encoder_context,
    Context& transformer_context,
    GenerationOptions&& options)
{
    Computation<void> computation({
        &vae_context,
        &text_encoder_context,
        &transformer_context
    });

    if (options.height % vae_.scale_factor() != 0 || options.width % vae_.scale_factor() != 0)
        throw std::runtime_error("height/width must be divisible by VAE's scale factor");

    auto vae_multiple = vae_.scale_factor() * 2;

    auto target_height = (options.height / vae_multiple) * vae_multiple;
    auto target_width = (options.width / vae_multiple) * vae_multiple;

    auto latent_height = 2 * (target_height / vae_multiple);
    auto latent_width = 2 * (target_width / vae_multiple);

    auto packed_h = latent_height / 2;
    auto packed_w = latent_width / 2;

    auto batch = options.num_images_per_prompt;

    auto image_seq_len = static_cast<int64_t>(packed_h) * packed_w;

    // 1. Preprocess reference images.
    for (auto &img : options.images)
        img = preprocess_reference_image(img, vae_multiple);

    // 2. Encode images to latents.
    auto image_latents = encode_images(computation, options.images, batch, options.encode_tile_size);

    // 3. Encode the prompt into text embeddings.
    auto prompt_embeds = computation
        .scope([&](Scope scope) -> Tensor {
            return encode_prompt(scope, batch, options.prompt, options.max_sequence_length);
        });

    // 4. Generate RoPE embeddings.
    auto img_ids = prepare_img_ids(computation.desc()->context(), batch, packed_h, packed_w);
    auto txt_ids = prepare_txt_ids(computation.desc()->context(), batch, options.max_sequence_length);
    auto image_latent_ids = options.images.empty()
        ? std::nullopt
        : std::make_optional(prepare_ref_image_ids(computation.desc()->context(), batch, options.images, vae_.scale_factor() * 2));

    // 5. Create initial latents.
    auto latents = options.init_latents
        ? make_init_latents(computation, batch, packed_h, packed_w, vae_.latent_channels(), std::move(*options.init_latents))
        : make_packed_latents(computation, batch, packed_h, packed_w, vae_.latent_channels());

    auto mu = compute_empirical_mu(image_seq_len, options.num_inference_steps);
    const int n = options.num_inference_steps;

    std::vector<float> sigmas(n);
    for (int i = 0; i < n; ++i)
        sigmas[i] =
            static_cast<float>(1.0 - static_cast<double>(i) / n);

    // Create a shared schedule since Computation is executed lazily and this must be alive
    // during whole computation.
    auto schedule = std::make_shared<Schedule>(scheduler_.schedule(n, mu, std::move(sigmas)));

    // 6. Denoise in latent space.
    latents = latents.fold(schedule->size(), [&, schedule](Scope scope, size_t& i, Tensor latents) -> Tensor {
        auto timestep = scope.context().value<float>({batch}, [batch, schedule, &i](std::mt19937&) {
            return std::vector<float>(batch, (*schedule)[i].timestep);
        });

        auto dt = scope.context().value<float>({}, [schedule, &i](std::mt19937&) {
            return std::vector<float>{(*schedule)[i].dt};
        });

        return denoise(
            scope,
            latents,
            *prompt_embeds,
            img_ids,
            txt_ids,
            image_latents ? std::make_optional(**image_latents) : std::nullopt,
            image_latent_ids,
            timestep,
            dt);
    });

    // 7. Decode latents to pixels.
    auto decoded = decode(latents, packed_h, packed_w, options.decode_tile_size);

    return decoded;
}

std::optional<Computation<Tensor>> Flux2KleinPipeline::encode_images(Computation<void> computation, const std::vector<Image>& images, int batch, int tile_size) {
    if (images.empty())
        return std::nullopt;

    const int vae_multiple = vae_.scale_factor() * 2;

    std::vector<Computation<Tensor>> packed_imgs;
    packed_imgs.reserve(images.size());

    for (auto& img : images) {
        // The pixels of each reference image are a state cell, so the
        // tiled encode can run every tile in its own graph scope.
        auto img_tensor = computation.state(
            {1, 3, static_cast<int64_t>(img.height()), static_cast<int64_t>(img.width())},
            [img](std::mt19937&) {
                return image_to_tensor_data(img);
            });

        // (B, C, H / scale, W / scale)
        auto mode = (tile_size > 0)
            ? vae_.encode_tiled(img_tensor, tile_size, tile_size)
            : img_tensor.scope([this](Scope scope, Tensor sample) -> Tensor {
                return this->vae_.encode(scope, sample).mode();
            });

        auto packed_h = img.height() / vae_multiple;
        auto packed_w = img.width() / vae_multiple;

        // 1. Patchify: (B, C, H, W) -> (B, 4C, H/2, W/2)
        // 2. Apply BN normalization to the patchified latents.
        // 3. Pack: (B, 4C, H/2, W/2) -> (B, H/2 * W/2, 4C)
        auto packed = mode.scope([this, packed_h, packed_w](Scope scope, Tensor mode) -> Tensor {
            auto patched = patchify_latents(mode, vae_.latent_channels(), packed_h, packed_w);

            auto bn_mean = (*vae_.bn()->running_mean())->reshape({1, -1, 1, 1});
            auto bn_std = sqrt((*vae_.bn()->running_var())->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
            patched = (patched - bn_mean) / bn_std;

            return pack_latents(patched);
        });

        packed_imgs.push_back(std::move(packed));
    }

    // Concatenate the reference images along the token dimension:
    //
    // (1, N1, 4C)
    // (1, N2, 4C)
    //       ↓
    // (1, N1 + N2, 4C)
    //
    // and repeat for num_images_per_prompt.
    return computation.scope([&, batch](Scope scope) -> Tensor {
        std::vector<Tensor> packed;
        packed.reserve(packed_imgs.size());

        for (auto& p : packed_imgs)
            packed.push_back(*p);

        auto image_latents = Tensor::cat(packed, /*axis=*/1);

        if (batch > 1) {
            auto shape = image_latents.shape();
            shape[0] = batch;
            image_latents = image_latents.repeat(shape);
        }

        return image_latents;
    });
}

Tensor Flux2KleinPipeline::encode_prompt(Scope scope, int batch, const std::string& prompt, size_t max_sequence_length) {
    // Apply chat template to prompt
    std::string text = tokenizer_.apply_chat_template(
        {{"user", prompt}},
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false
    );

    std::vector<int> mask;
    size_t num_real_tokens;
    auto tokens = tokenizer_.encode(text, max_sequence_length, &mask, &num_real_tokens);

    auto input_ids = scope.context().create<int32_t>({batch, (int64_t)tokens.size()},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * tokens.size());
            for (auto b=0; b<batch; ++b) std::copy(tokens.begin(), tokens.end(), ids.begin() + b * tokens.size());
            return std::move(ids);
        });

    auto attention_mask = scope.context().create<float>({batch, (int64_t)mask.size()},
        [=](std::mt19937&) {
            std::vector<float> m(size_t(batch) * mask.size());
            for (auto b=0; b<batch; ++b) std::copy(mask.begin(), mask.end(), m.begin() + b * mask.size());
            return std::move(m);
        });

    text_encoder_.forward(
        scope,
        input_ids,
        attention_mask,
        std::nullopt, // position_ids
        std::nullopt, // past_key_values
        std::nullopt, // inputs_embeds
        std::nullopt, // labels
        false, // use_cache
        0 // logits_to_keep
    );

    auto l9 = text_encoder_.hidden_states()->at(9);
    auto l18 = text_encoder_.hidden_states()->at(18);
    auto l27 = text_encoder_.hidden_states()->at(27);

    if (!l9 || !l18 || !l27)
        throw std::runtime_error("Hidden state extraction failed");

    // Stack along a new dimension (axis=1) -> (B, 3, seq, hidden)
    auto prompt_embeds = Tensor::stack({l9, l18, l27}, /*axis=*/1);

    // Permute to (B, seq, 3, hidden)
    prompt_embeds = prompt_embeds.permute({0, 2, 1, 3});

    // Reshape to (B, seq, 3 * hidden)
    int64_t hidden_dim = l9.shape()[2]; // Assuming (B, seq, hidden)
    prompt_embeds = prompt_embeds.reshape({batch, (int64_t)tokens.size(), 3 * hidden_dim});

    return prompt_embeds;
}

Tensor Flux2KleinPipeline::denoise(
    Scope scope,
    const Tensor& latents,
    const Tensor& prompt_embeds,
    const Tensor& img_ids,
    const Tensor& txt_ids,
    const std::optional<Tensor>& image_latents,
    const std::optional<Tensor>& image_latent_ids,
    const Tensor& timestep,
    const Tensor& dt)
{
    auto latent_model_input = latents;
    auto latent_image_ids = img_ids;
    int64_t num_ref_tokens = 0;

    if (image_latents) {
        if (!image_latent_ids)
            throw std::invalid_argument(
                "denoise(): image_latent_ids is required when image_latents is provided");

        latent_model_input = Tensor::cat({latents, *image_latents}, /*axis=*/1);
        latent_image_ids = Tensor::cat({img_ids, *image_latent_ids}, /*axis=*/1);
        num_ref_tokens = image_latents->shape()[1];
    }

    auto noise_pred = transformer_.forward(scope,
        /*hidden_states=*/          latent_model_input,
        /*encoder_hidden_states=*/  prompt_embeds,
        /*timestep=*/               timestep / 1000.0f,
        /*img_ids=*/                latent_image_ids,
        /*txt_ids=*/                txt_ids,
        /*guidance=*/               std::nullopt,
        /*num_ref_tokens=*/         num_ref_tokens
    );

    // Slice noise_pred to original latents shape if img2img
    if (num_ref_tokens > 0)
        noise_pred = noise_pred[{Tensor::Slice::all(), Tensor::Slice::range(0, latents.shape()[1])}];

    // Integrate latents over dt
    return scheduler_.integrate(scope, noise_pred, latents, dt);
}

Computation<Tensor> Flux2KleinPipeline::decode(Computation<Tensor> latents, int packed_h, int packed_w, int tile_size) {
    auto z = latents.scope([&](Scope scope, Tensor latents) -> Tensor {
        // 1. Unpack to patch grid (B, 4C, ph, pw)
        auto z_packed = unpack_latents(latents, packed_h, packed_w);

        // 2. BN-unnormalize
        auto bn_mean = (*vae_.bn()->running_mean())->reshape({1, -1, 1, 1});
        auto bn_std = sqrt((*vae_.bn()->running_var())->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
        z_packed = z_packed * bn_std + bn_mean;

        // 3. unpatchify to spatial latents (B, C, 2ph, 2pw)
        auto z = unpatchify_latents(z_packed, vae_.latent_channels(), packed_h, packed_w);

        // 4. VAE decode (non-tiled).
        if (tile_size == 0)
            return vae_.decode(scope, z);

        return z;
    });

    // 2. VAE decode (tiled).
    if (tile_size > 0)
        return vae_.decode_tiled(z, tile_size, tile_size);   // (B, 3, H, W)

    return z;
}

Flux2KleinPipeline Flux2KleinPipeline::from_pretrained(Context& vae_context, Context& text_encoder_context, Context& transformer_context, const std::filesystem::path& path) {
    // 1. Initialize models from config
    Qwen3ForCausalLM text_encoder(Qwen3Config::from_file(path / "text_encoder" / "config.json"), std::vector<size_t>{9, 18, 27});
    Flux2Transformer2DModel transformer(Flux2Transformer2DModel::Config::from_file(path / "transformer" / "config.json"));
    AutoencoderKLFlux2 vae(AutoencoderKLFlux2::Config::from_file(path / "vae" / "config.json"));

    // 2. Load VAE
    {
        GGUFLoaderVisitor vae_loader(vae_context, path / "vae");
        RethrowVisitor vae_visitor(vae_loader);
        vae.accept(vae_visitor);
        vae_visitor.rethrow();
        vae_loader.validate();
    }

    // 3. Load text encoder
    {
        GGUFLoaderVisitor text_encoder_loader(text_encoder_context, path / "text_encoder");
        RethrowVisitor text_encoder_visitor(text_encoder_loader);
        text_encoder.accept(text_encoder_visitor);
        text_encoder_visitor.rethrow();
        text_encoder_loader.validate();
    }

    // 4. Load transformer
    {
        GGUFLoaderVisitor transformer_loader(transformer_context, path / "transformer");
        RethrowVisitor transformer_visitor(transformer_loader);
        transformer.accept(transformer_visitor);
        transformer_visitor.rethrow();
        transformer_loader.validate();
    };

    // 5. Load tokenizer
    auto tokenizer = Qwen2TokenizerFast::from_pretrained(path / "tokenizer");

    // 6. Construct the pipeline
    Flux2KleinPipeline pipeline(
        std::move(transformer),
        std::move(vae),
        std::move(text_encoder),
        std::move(tokenizer)
    );

    return std::move(pipeline);
}

// 1. Unpacks from (B, ph*pw, 4C) to (B, 4C, ph, pw)
Tensor Flux2KleinPipeline::unpack_latents(Tensor packed, int packed_h, int packed_w) {
    const int64_t B = packed.shape()[0];
    const int64_t C4 = packed.shape()[2];
    auto t = packed.permute({0, 2, 1}).contiguous();
    return t.reshape({B, C4, packed_h, packed_w});
}

// 2. Unpatchifies from (B, 4C, ph, pw) to (B, C, 2ph, 2pw)
// Uses a 4D interleaving trick to avoid 5D/6D tensors
Tensor Flux2KleinPipeline::unpatchify_latents(Tensor patched, int channels, int packed_h, int packed_w) {
    const int64_t B = patched.shape()[0];
    const int64_t C = channels;
    const int64_t N = packed_h * packed_w;

    auto t = patched.reshape({B, 4 * C, N});
    t = t.reshape({B, 2 * C, 2, N});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B * C, 2, packed_h, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, C, 2 * packed_h, 2 * packed_w});

    return t;
}

// 3. Packs from (B, 4C, ph, pw) to (B, ph*pw, 4C)
Tensor Flux2KleinPipeline::pack_latents(Tensor latents) {
    const int64_t B = latents.shape()[0];
    const int64_t C4 = latents.shape()[1];
    const int64_t ph = latents.shape()[2];
    const int64_t pw = latents.shape()[3];
    const int64_t N = ph * pw;

    auto t = latents.reshape({B, C4, N});
    return t.permute({0, 2, 1}).contiguous();
}

// 4. Patchifies from (B, C, 2ph, 2pw) to (B, 4C, ph, pw)
// Inverse of the 4D interleaving trick
Tensor Flux2KleinPipeline::patchify_latents(Tensor latents, int channels, int packed_h, int packed_w) {
    const int64_t B = latents.shape()[0];
    const int64_t C = channels;
    const int64_t N = packed_h * packed_w;

    auto t = latents.reshape({B * C, packed_h, 2, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B, 2 * C, N, 2});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 4 * C, N});
    t = t.reshape({B, 4 * C, packed_h, packed_w});

    return t;
}
