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

static std::vector<Image> run(Backend& backend, Computation<Tensor> computation, ShardingRuntime* runtime = nullptr) {
    ProgressBar progress("Flux2Klein");
    std::mt19937 rng;

    auto decoded = ExecutionRuntime::Default.run(backend, rng, computation, runtime, &progress);
    auto data = ExecutionRuntime::Default.read<float>(decoded);

    auto images = Flux2KleinPipeline::to_images(decoded.shape(), std::move(data));

    return std::move(images);
}

int main(int argc, char** argv) {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto meta = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);

    //meta.tensor_split({2, 1});

    Backend backend(meta);
    ShardingRuntime runtime(ExecutionRuntime::Default, meta);
    Scope scope(runtime);

    /*Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend backend(cpu);*/

    Context context(65536);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(context, context, context, "../utils/convert-model/models/black-forest-labs/FLUX.2-klein-9B"));

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.decode_tile_size = 64;
    options.width = 512;
    options.height = 512;

    auto computation = pipeline(context, context, context, std::move(options));

    auto images = run(backend, computation, &runtime);

    images[0].save("test.png");

    return EXIT_SUCCESS;
}
