# Diffusers Pipelines in C++

Run [Hugging Face diffusers](https://github.com/huggingface/diffusers) pipelines in pure C++, powered by [ggml](https://github.com/ggml-org/ggml). No Python and no PyTorch at build or inference time: models load from GGUF files, can be quantized to fit in less memory, and can be split across multiple CPU or GPU devices at runtime.

Every ported pipeline is a faithful port — it mirrors the Python diffusers architecture module by module and matches it 1:1 numerically, verified by a cross-check test suite against the pinned Python reference.

## Features

- **No Python at runtime.** Tokenizer, text encoder, transformer, VAE, and scheduler all run in C++; Python is used only by the optional test suite and the one-off model conversion tool.
- **Multi-device sharding out of the box.** Tensors are split automatically across multiple CPU or GPU devices — something Python diffusers does not support.
- **GGUF model format.** Each model component loads from its own `model.gguf`, with per-tensor or per-component quantization (e.g. Q8_0) to reduce memory footprint.
- **Numerical parity with diffusers.** Ported modules mirror the Python architecture and match it 1:1, verified component by component and end to end.
- **A library of diffusers-style modules.** Attention, normalization, embeddings, transformer blocks, VAEs, and schedulers — a reusable C++ module hierarchy mirroring the diffusers decomposition.
- **Broad backend support.** CPU out of the box; CUDA, Metal, HIP, Vulkan, OpenCL, SYCL, and MUSA via ggml, enabled at configure time.
- **Inference only.** No training, no gradients.

## Supported Pipelines

| Pipeline | Status | Components |
|----------|--------|------|
| [Flux2KleinPipeline](src/diffusers/pipelines/flux2/Flux2KleinPipeline.hpp) | Ported | [Flux2Transformer2DModel](src/diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp), [Qwen3ForCausalLM](src/transformers/models/qwen3/Qwen3ForCausalLM.hpp), [AutoencoderKLFlux2](src/diffusers/models/autoencoders/AutoencoderKLFlux2.hpp), [Qwen2TokenizerFast](src/transformers/models/qwen2/Qwen2TokenizerFast.hpp) |
| Ideogram 4.0 | Planned | — |
| Wan 2.2 | Planned | — |

A pipeline is **Ported** when: all required modules are ported; each is numerically verified against the Python reference, component by component and end to end; a feasible multi-device sharding plan exists (the test suite passes with `N_DEVICES=2`); and a quantization config ships for the model.

## Getting Started

### Prerequisites

- CMake 3.18+ and a C++17 compiler
- [docker](https://www.docker.com/get-started/) — build sandbox for compiling the Rust part of the [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) dependency (a local `cargo` works too — see below)
- [uv](https://docs.astral.sh/uv/) — for the test suite only; not needed when building with `-DDIFFUSERS_BUILD_TESTS=OFF`

### Build

```bash
git clone https://github.com/oinuar/diffusers-cpp
cd diffusers-cpp
git submodule update --init --recursive

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARGO_EXECUTABLE="$PWD/../utils/cargo.sh" -DDIFFUSERS_CUDA=ON   # or OFF for CPU-only
cmake --build . -j$(nproc)
```

`-DCARGO_EXECUTABLE` points at `utils/cargo.sh`, which runs the Rust build in a Docker sandbox; point it at a local `cargo` instead if you prefer. Add `-DDIFFUSERS_BUILD_TESTS=OFF` to skip the test suite entirely.

GPU backends are provided by ggml and enabled at configure time:

| Flag | Backend |
|------|---------|
| `-DDIFFUSERS_CUDA=ON` | NVIDIA CUDA (cuBLAS + mmq) |
| `-DDIFFUSERS_METAL=ON` | Apple Metal (BLAS + compute), default on Apple platforms |
| `-DDIFFUSERS_HIP=ON` | AMD ROCm/HIP (rocBLAS) |
| `-DDIFFUSERS_VULKAN=ON` | Vulkan |
| `-DDIFFUSERS_OPENCL=ON` | OpenCL |
| `-DDIFFUSERS_SYCL=ON` | Intel oneAPI SYCL |
| `-DDIFFUSERS_MUSA=ON` | Moore Threads MUSA |

The test suite is verified on CPU (single device and sharded multi-device); GPU backends are forwarded to ggml as-is.

### Getting a Model

`utils/convert-model` downloads a Hugging Face model repository and converts each component folder (e.g. `transformer/`, `vae/`, `text_encoder/`) into its own `model.gguf` under `utils/convert-model/models/`:

```bash
cd utils/convert-model
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python src/convert_hf_safetensors_to_gguf.py black-forest-labs/FLUX.2-klein-9B --type @flux2-klein-9b-q8.json
```

`--type` controls quantization: a single type name (e.g. `Q8_0`) for every tensor, a JSON object of tensor-name patterns to types, a per-component JSON object, or `@file.json`. Omit it to keep the original dtypes. `flux2-klein-9b-q8.json` ships a ready-made Q8_0 config for FLUX.2-klein-9B.

## Architecture

- `src/ggml/Tensor.hpp` — C++ tensor type and operations on top of ggml (PyTorch-style shapes).
- `src/nn/` — the `Module` base class and core primitives (`Linear`, `Conv2d`, `LayerNorm`, …). Modules form a tree of children; operations such as weight loading traverse it with the visitor pattern.
- `src/diffusers/`, `src/transformers/` — the ported diffusers modules and transformer models.
- `src/ggml/GGUFLoaderVisitor.hpp` — loads the per-component `model.gguf` files into the module tree.
- `src/ggml/Computation.hpp`, `ExecutionRuntime.hpp` — build a graph of tensors and execute it on the chosen device(s).
- `src/ggml/ShardingRuntime.hpp`, `MetaDevice.hpp` — plan how tensors are split across multiple devices.
- `test/` — the Python cross-check test suite: each test drives a small C++ CLI (`test/utils/`) and compares its output against the Python reference implementation.

## Testing

The test suite is the fidelity check: each test runs the C++ implementation and the Python reference with identical inputs (random weights, real model shapes) and compares the outputs.

- One ctest entry per Python test method. The Python reference runs in a uv-managed venv (created automatically at build time) against pinned diffusers/transformers commits — see `test/requirements.txt`.
- Tests use tiny random models; GGUF files are never involved.

```bash
ctest --test-dir build -j8 --output-on-failure
```

| Environment variable | Effect |
|----------------------|--------|
| `N_DEVICES=2` | Run on two CPU devices through the sharding runtime |
| `USE_GPU=true` | Run on the enabled GPU backend |
| `TENSOR_SPLITS=0.5,0.5` | Per-device tensor-split factors (even split by default) |

## License

[MIT](LICENSE)
