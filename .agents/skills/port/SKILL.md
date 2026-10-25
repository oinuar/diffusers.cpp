---
name: port
description: Use when a Python reference (diffusers / transformers / torch) is turned into a working, tested C++ port.
---
# Porting Guide

How to take a Python reference (diffusers / transformers / torch) and turn it into a working, tested C++ port.

Two invariants for everything you port:

1. **Mirror the Python implementation module by module** — class names, structure, formulas, config schemas, tensor names.
2. **Replace the Python/PyTorch runtime in a few systematic ways** — see [Runtime mapping](#runtime-mapping).

[Flux2KleinPipeline](src/diffusers/pipelines/flux2/Flux2KleinPipeline.hpp) is the complete worked example. Nothing counts as ported until its tests pass against the pinned Python reference.

## Layout

| What | Where |
|---|---|
| Pinned Python reference (diffusers, transformers commits) | `test/requirements.txt` |
| Tensor operators | `src/ggml/Tensor.{hpp,cpp}` |
| NN primitives (`Linear`, `Conv2d`, `LayerNorm`, …) | `src/nn/` |
| Ported diffusers modules | `src/diffusers/models/` |
| Ported transformer models | `src/transformers/models/` |
| Ported pipelines | `src/diffusers/pipelines/` |
| Cross-check tests (mirror the source paths) | `test/src/` |
| Test CLIs | `test/utils/` |
| Model conversion and quantization configs | `utils/convert-model/` |

Work bottom-up: **tensor ops → modules → pipeline → sharding → shipping**. Every step below ends with a test command — a unit is done when that command is green.

## 1. Port the tensor ops you need

Most `forward` methods translate line for line, because the tensor operator syntax is deliberately as close as possible to Python/PyTorch ([syntax reference](#tensor-operators)). If the Python code uses an op that `Tensor.hpp` doesn't have:

1. Implement it in `Tensor.cpp` as a ggml mapping, matching torch semantics exactly — broadcasting rules. Remember the op **builds the graph**: it returns a new tensor node in the current `Context`, nothing is computed at call time.
2. Add `test/src/ggml/Tensor/<op>.py` modeled on test/src/ggml/Tensor/<op>.py: feed the same random inputs to the `test-tensor-cli` binary and to torch, and compare. Cover same-shape, trailing/singleton broadcasting, scalar, and cross-rank cases.
3. Run it: `ctest --test-dir build -j8 --output-on-failure -R "src-ggml-Tensor-<op>"` (ctest names mirror the test path).

## 2. Port a module

Work from the Python class (e.g. something under `diffusers/src/diffusers/models/...` in the pinned reference).

### 2.1 Mirror the class skeleton

```cpp
class MyNorm : public Module {
public:
    struct Config {
        static Config from_file(const std::filesystem::path& path);  // parse config.json
        int64_t dim = ...;                                           // defaults = the HF defaults
        float eps = 1e-5f;
    };

    MyNorm(const Config& cfg) {
        // child name = the Python attribute name (it defines the tensor-name path)
        modules["weight"] = std::make_shared<Parameter>(Tensor::Shape{cfg.dim});
    }

    Tensor forward(Scope scope, Tensor x) {
        auto weight = weight()->forward(scope);   // Parameter returns its loaded tensor
        return (x * rsqrt(x.sum(-1, true) / dim_ + eps_)) * weight;
    }

    std::shared_ptr<Parameter> weight() const {
        return std::static_pointer_cast<Parameter>(modules.at("weight"));
    }

private:
    int64_t dim_;
    float eps_;
};
```

- **Class name = Python class name.** The model must be constructible from the same upstream `config.json` schema (`Config::from_file`), so keep the config fields and their defaults.
- **Child names define the tensor-name path.** The `modules` map key is the Python attribute name; the dotted path (e.g. `transformer_blocks.0.attn.q.weight`) is how the GGUF loader matches parameters — keep it identical to upstream.
- **`Parameter` holds a declared shape and dtype — no data.** Data is bound at load time by the visitor; a ported model is therefore fully constructible from config before any weights exist.
- **Weight loading is free.** `from_pretrained` walks the module tree with `GGUFLoaderVisitor`, which resolves each `Parameter` by name, validates its shape, and streams the bytes from the component's `model.gguf`. No per-module loading code.

### 2.2 Implement `forward`

Translate the Python `forward` line by line with the [tensor operator syntax](#tensor-operators). Rules:

- Signature is `forward(Scope scope, ...)` — the `Scope` is the compute-tensor allocation unit; pass it to every sub-module call.
- Return a `Tensor` built from ops; no in-place mutation, no autograd.
- Call sub-modules through the typed accessors (`static_pointer_cast`), mirroring `self.sub_module(...)` calls.

### 2.3 Handle fused weights, if any

If the Python class fuses several logical projections into one weight (one `nn.Linear` serving q/k/v, or gate/up), **do not mirror the fusion**: keep separate `Linear` children and add a dedicated `visit()` overload in `GGUFLoaderVisitor` that slices the fused GGUF tensor at load time.

- Row slices use `read_rows` (rows are contiguous in the file — one seek+read). Example: [`Flux2FusedQKVProjection`](src/diffusers/models/transformers/flux2/Flux2FusedQKVProjection.hpp) — Python's `to_qkv_mlp_proj` (one weight of shape `(3·inner + mlp_out) × query_dim`) becomes separate `q`, `k`, `v`, `mlp_in_gate`, and `mlp_in_up` projections.
- Column slices use `read_cols` (one seek+read per row; boundaries must be aligned to the quantization block size so quantized types like Q8_0 slice without dequantization). Example: [`Flux2FusedAttentionOutput`](src/diffusers/models/transformers/flux2/Flux2FusedAttentionOutput.hpp) — Python's `to_out` (one projection over the concatenated `[attn; mlp]` features) becomes `attn` / `mlp`; [`Flux2FeedForward`](src/diffusers/models/transformers/flux2/Flux2FeedForward.hpp) splits `linear_in` into `linear_in_gate` / `linear_in_up`, including the 1-D fused bias.

The math is identical — a row or column slice of a projection *is* that projection. The decomposition exists for sharding (step 4).

### 2.4 Test it

1. Add `test/src/<same path as the module>/<ClassName>.py` with a `unittest.TestCase` subclass (see e.g. `test/src/diffusers/models/transformers/flux2/`). Build the class with the smallest config that keeps the real shape structure, generate random weights with a seeded `torch.Generator`, pass them to the C++ CLI via `--param` options (or as quantized raw bytes via `--quant` when testing with a quantization spec), and compare against the Python reference.
2. Run it: `ctest --test-dir build -j8 --output-on-failure -R "<path-derived name>"`, e.g. `-R "src-diffusers-models-transformers-flux2-MyNorm"`.
3. If the C++ CLI crashes, the test writes a reproduction Python script to `build/test-repro/` — run it to reproduce the exact failure.

Each test directory is driven by its own CLI: `test/src/ggml/Tensor/` → `test-tensor-cli`, `test/src/nn/` → `test-nn-cli`, `test/src/diffusers/` → `test-diffusers-cli`, `test/src/diffusers/flux2` → `test-flux2-cli`, `test/src/transformers/` → `test-qwen3-cli`.

## 3. Port the pipeline

### 3.1 Mirror the composition

The pipeline class holds the same components as the Python one. `from_pretrained` (see `Flux2KleinPipeline::from_pretrained` for example):

1. Constructs each model from its component's `config.json` (`Config::from_file`).
2. Loads each component's `model.gguf` by walking the module tree: `GGUFLoaderVisitor` wrapped in a `RethrowVisitor`, then `accept(...)`, `rethrow()`, and `validate()` (reports checkpoint tensors that were never loaded).
3. Loads the tokenizer from the component's `tokenizer/` folder.

### 3.2 Translate `__call__` into `operator()`

- Map the numbered steps of Python's `__call__` one to one; keep the numbered step comments.
- `operator()` takes one `Context` per component plus a `GenerationOptions` struct that mirrors the `__call__` arguments (with the Python defaults) and **returns a `Computation<Tensor>`** — it builds the graph, it does not run it.
- Use `computation.scope(...)` for per-step sub-graphs; random noise comes from `computation.state(...)` with `std::mt19937` (a different RNG stream than torch — deterministic runs and tests pass random values as arguments instead).
- The eager denoising loop becomes a `fold` over a precomputed `Schedule` held in a `std::shared_ptr` (the computation executes lazily, so the schedule must outlive the construction).
- Public Python methods (`encode_prompt`, `prepare_image_latents`, …) become same-named pipeline methods that build their part of the graph.

### 3.3 Decide what to drop

Port the core inference path and drop the Python surface the C++ runtime cannot express: the `DiffusionPipeline` base class and module registration, LoRA and other loaders, callbacks, progress bars, interrupts, compile/XLA hooks, `attention_kwargs`, and `output_type`/`return_dict` variants. Options the ported model cannot activate (e.g. guidance for a distilled model) may stay in the options struct, marked unused. Keep input validation: C++ throws where Python raises, and may be stricter where Python only warns — but preserve Python's rounding behavior for valid inputs. Graphs produce tensors only: pixel conversion, image post-processing, and output dataclasses happen after execution, outside the graph.

### 3.4 Test it

1. Test the pipeline's public methods first (one test each: prompt encoding, image encoding, one denoising step, decoding, the latent pack/unpack helpers), exactly like the module tests.
2. Then add the end-to-end `test_call` with a config in the real model's shape class: feed the same `init_latents` to both sides (noise is not comparable across RNGs) and compare the decoded pixels — exact (`rtol=1e-4`) for dequantized weights, or a small pixel tolerance (e.g. 2.5/255) when running with the quantization config.

## 4. Make it shardable

Run the suite with two devices: `N_DEVICES=2 ctest --test-dir build -j8 --output-on-failure`. If the planner reports an infeasible plan (it prints the reason), fix the model with these rules:

- **Row-parallel projections must be direct outputs.** On the meta device, every q/k/v must be the direct output of its own row-parallel `Linear` (weight `S(1)` → output `S(0)`): the sharded heads that `flash_attn` hard-requires cannot be carried through a windowed view of a fused output — a view inherits a shard only when it covers the sharded axis in full from its start. This is why fused weights are split at load time (step 2.3).
- **Elementwise products must stay on one axis.** A SwiGLU gate/up product whose inputs come from a *fused* column-parallel output is cross-device (each device holds a different half of the features). Split into separate `gate` / `up` column-parallel projections: each output is `S(0)`, the product stays `S(0)`, and the MLP output projection can run row-parallel (`S(0)` → `P` → AllReduce → `R`).
- **Keep ops in sharded states.** Some algebraic rewrites break planning: `rsqrt` is implemented as `pow(x, -0.5)`, not `1.0f / sqrt(x)` — the scalar-broadcast division would lift the scalar to a replicated tensor, and `R / S(a)` cannot be executed on the meta backend.
- **Head axes must be divisible by the device count** — `flash_attn` shards along the heads axis.

## 5. Ship the model

1. Write a quantization config for the model in `utils/convert-model/` (JSON: one entry per component, each a single type name or a per-tensor pattern map — see `flux2-klein-9b-q8.json` for example).
2. Convert the Hugging Face repo: `uv run python src/convert_hf_safetensors_to_gguf.py <org/model> --type @<config>.json` (see the [README](README.md#getting-a-model)).

## Done

A pipeline is **Ported** when all required modules are ported; each is numerically verified against the Python reference, component by component and end to end; a feasible multi-device sharding plan exists (the suite passes with `N_DEVICES=2`); and a quantization config ships for the model. See the [status definition in the README](README.md#supported-pipelines).

## Reference

### Runtime mapping

| Python (diffusers) | C++ (this project) |
|---|---|
| `pipe(...)` runs eagerly, returns an output dataclass | `pipeline(contexts, options)` returns a `Computation<Tensor>`; it executes later, on CPU, GPU, or sharded |
| `module(...)` eager forward | `module.forward(scope, ...)`: builds into the graph; the `Scope` allocates and frees compute tensors |
| `from_pretrained()` loads safetensors from the HF hub | `from_pretrained()` loads per-component `model.gguf` files via the `GGUFLoaderVisitor`; `Parameter`s are declared shapes filled at load time |
| `torch.Generator` / `randn_tensor` noise | `std::mt19937` via `computation.state(...)` — a different RNG stream; pass initial latents for deterministic runs |
| `.to(device)`, `torch_dtype`, device workarounds | The device is chosen at runtime by the execution engine; no device or dtype bookkeeping inside modules |
| `PIL.Image` | `Image` (stb) |

### Tensor operators

`src/ggml/Tensor.{hpp,cpp}` is a thin layer over ggml. Shapes are in PyTorch order (first index = outermost / slowest dimension). `Slice` mirrors Python slicing exactly: `:` → `Slice::all()`, `i` → `Slice::index(i)`, `start:stop:step` → `Slice::range(start, stop, step)`, `None` → `Slice::none()`, `...` → `Slice::ellipsis()`.

| Python (PyTorch) | C++ |
|---|---|
| `a + b`, `a - b`, `a * b`, `a / b`, `-a` | `a + b`, `a - b`, `a * b`, `a / b`, `-a` |
| `a + 2.0`, `2.0 / a` | `a + 2.0f`, `2.0f / a` (scalars on either side) |
| `a[i]` | `a[i]` |
| `a[:, 1:3]`, `a[None, ...]` | `a[{Slice::all(), Slice::range(1, 3)}]`, `a[{Slice::none(), Slice::ellipsis()}]` |
| `a.reshape(-1, 4)` | `a.reshape({-1, 4})` |
| `a.permute(0, 2, 1)`, `a.squeeze(0)`, `a.unsqueeze(1)` | same names, same arguments |
| `a.narrow(1, 0, n)`, `a.chunk(2, dim=1)`, `a.split(4, dim=1)`, `a.split_with_sizes(...)` | same names |
| `a.expand(...)`, `a.repeat(...)`, `a.contiguous()` | same names |
| `torch.zeros(s)`, `torch.ones(s)`, `torch.full(s, v)`, `torch.arange(...)` | `Tensor::zeros(s)`, `Tensor::ones(s)`, `Tensor::full<T>(s, v)`, `Tensor::arange(...)` |
| `torch.cat(list, dim)`, `torch.stack(list, dim)` | `Tensor::cat(list, dim)`, `Tensor::stack(list, dim)` |
| `torch.abs/sqrt/exp/log/sin/cos`, `torch.pow` | `abs(a)`, `sqrt(a)`, `exp(a)`, `log(a)`, `sin(a)`, `cos(a)`, `pow(a, b)`, `rsqrt(a)` |
| `a.clamp(min, max)`, `a.sum(dim, keepdim)`, `a.mean(dim, keepdim)` | same |
