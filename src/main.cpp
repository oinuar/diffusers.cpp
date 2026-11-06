#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Computation.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include "ggml/ShardingRuntime.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "ProgressBar.hpp"
#include "Image.hpp"
#include <iostream>
#include <filesystem>

static std::vector<Image> run(
    const std::vector<std::pair<Context*, Backend*>>& backends,
    Computation<Tensor> computation,
    ShardingRuntime* runtime = nullptr
) {
    ProgressBar progress("Flux2Klein");
    std::mt19937 rng;

    auto decoded = ExecutionRuntime::Default.run(backends, rng, computation, runtime, &progress);
    auto data = ExecutionRuntime::Default.read<float>(decoded);

    auto images = Flux2KleinPipeline::to_images(decoded.shape(), std::move(data));

    return std::move(images);
}

int main(int argc, char** argv) {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto meta = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
    Device gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
    Backend gpu_backend(gpu);

    //meta.tensor_split({2, 1});

    Backend backend(meta);
    ShardingRuntime runtime(ExecutionRuntime::Default, meta);
    Scope scope(runtime);

    /*Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend backend(cpu);*/

    Context vae_context(6000);
    Context context(6000);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(vae_context, context, context, "../utils/convert-model/models/black-forest-labs/FLUX.2-klein-9B"));

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.width = 512;
    options.height = 512;

    auto computation = pipeline(std::move(options));

    auto images = run({{&context, &backend}, {&vae_context, &gpu_backend}}, computation, &runtime);

    images[0].save("test.png");

    return EXIT_SUCCESS;
}
