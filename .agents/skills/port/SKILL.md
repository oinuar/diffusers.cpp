---
name: port
description: Use when a Python reference (diffusers / transformers / torch) is turned into a working, tested C++ port.
---
# Porting Guide

How to take a Python reference (diffusers / transformers / torch) and turn it into a working, tested C++ port.

Two invariants for everything you port:

1. **Mirror the Python implementation module by module** — class names, structure, formulas, config schemas, tensor names.
2. **Replace the Python/PyTorch runtime in a few systematic ways** — see [Runtime mapping](#runtime-mapping).

[Flux2KleinPipeline](src/diffusers/pipelines/flux2/Flux2KleinPipeline.hpp) is the complete worked example. A port is **complete** when the single-device and the multi-device paths both work, a quantization config exists for the model, and all tests pass — see [Done](#done).

## Port flow

1. [Study the Python source code](#1-study-the-python-source-code)
2. [Analyze what is already implemented in C++](#2-analyze-what-is-already-implemented-in-c)
3. [Plan the tensor operators, modules, and porting order](#3-plan-the-port)
4. [Port the single-device CPU path 1:1, with tests](#4-port-the-single-device-cpu-path-11)
5. [Check the single-device suite passes (N_DEVICES=1)](#5-check-the-single-device-suite-passes)
6. [Run the sharded multi-device path (N_DEVICES=2)](#6-run-the-sharded-multi-device-path)
7. [Plan the sharded path, if the plan is infeasible](#7-plan-the-sharded-path-if-the-plan-is-infeasible)
8. [Check the sharding plan is optimal](#8-check-the-sharding-plan-is-optimal)
9. [Analyze the quantization](#9-analyze-the-quantization)
10. [Write the quantization config](#10-write-the-quantization-config)

Steps 1–3 are reading and planning; step 4 is the port proper, worked bottom-up: **tensor ops → modules → pipeline**; steps 5–8 turn it into a working, optimized multi-device port; steps 9–10 make it shippable. Every implementation step ends with a test command — a unit is done when that command is green.

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
| Quantization quality measurement | `test/utils/quant_quality.py` |
| Model conversion and quantization configs | `utils/convert-model/` |

## 1. Study the Python source code

The Python reference is pinned in `test/requirements.txt` (diffusers and transformers at specific commits) and installed in the test venv (`test/.venv`) — read the source exactly as the tests run against it.

Read the pipeline and everything it touches, and take notes on:

- **The pipeline's `__call__`** — the numbered steps, every option and its Python default (they become the `GenerationOptions` fields), and the components the pipeline holds.
- **Each component model** — the `config.json` schema (fields and HF defaults), the module tree (child attribute names — they define the tensor-name paths), and every `forward` formula.
- **The scheduler and tokenizer** — the step/sampling functions and prompt processing.
- **The torch op set** — list every op the port will need and check it against the [tensor operators](#tensor-operators) table; anything missing is work for [step 4.1](#41-tensor-operators).

The output of this step is the reference for the whole port: per component, the module tree with tensor names, the config schemas, and the op list.

## 2. Analyze what is already implemented in C++

Before writing anything, find out what already exists — sibling pipelines share components:

| Layer | Where | What to look for |
|---|---|---|
| Tensor operators | `src/ggml/Tensor.{hpp,cpp}` | the ops from step 1's op list |
| NN primitives | `src/nn/` | `Linear`, `Conv2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Embedding`, attention |
| Ported modules | `src/diffusers/models/`, `src/transformers/models/` | e.g. the flux2 transformer blocks, the Qwen3 text encoder, the Flux2 VAE |
| Ported pipelines | `src/diffusers/pipelines/` | earlier ports to copy the shape from |
| Tests | `test/src/` | mirrors the source paths — a module with a green test is already ported; reuse it, do not re-port it |

Diff the step 1 notes against this: the work list is the missing ops, the missing modules, and the missing components/pipeline.

## 3. Plan the port

Plan bottom-up, in dependency order — a module can only be ported (and tested) once the ops and sub-modules it uses exist:

1. **Tensor operators first.** Each missing op: its name and the exact torch semantics to match (broadcasting, keepdim, dtype).
2. **Modules leaves-first.** Normalizations and embeddings → attention and feed-forward → transformer blocks → component models (text encoder, transformer, VAE).
3. **The pipeline last**, once every component is green.

Write the plan down (e.g. one todo per unit) so progress is trackable. Keep the plan 1:1 — step 4 ports the single-device CPU path only; sharding and quantization are handled later (steps 6–10), never up front.

## 4. Port the single-device CPU path 1:1

Port the step 3 plan in order: tensor ops → modules → pipeline. Two rules for this step:

- **1:1 mirror** — the port is the Python implementation with the [runtime replaced](#runtime-mapping); no restructuring for sharding or memory. Sharding-driven adjustments come later as [module patches](#module-patches) (step 7), applied only when the graph runs through the sharding runtime.
- **Test every unit as it lands** — each unit below ends with a ctest command; the unit is done when that command is green.

### 4.1 Tensor operators

Most `forward` methods translate line for line, because the tensor operator syntax is deliberately as close as possible to Python/PyTorch ([syntax reference](#tensor-operators)). If the Python code uses an op that `Tensor.hpp` doesn't have:

1. Implement it in `Tensor.cpp` as a ggml mapping, matching torch semantics exactly — broadcasting rules. Remember the op **builds the graph**: it returns a new tensor node in the current `Context`, nothing is computed at call time.
2. Add `test/src/ggml/Tensor/<op>.py` modeled on test/src/ggml/Tensor/<op>.py: feed the same random inputs to the `test-tensor-cli` binary and to torch, and compare. Cover same-shape, trailing/singleton broadcasting, scalar, and cross-rank cases.
3. Run it: `ctest --test-dir build -j8 --output-on-failure -R "src-ggml-Tensor-<op>"` (ctest names mirror the test path).

### 4.2 Modules

Work from the Python class (e.g. something under `diffusers/src/diffusers/models/...` in the pinned reference).

#### 4.2.1 Mirror the class skeleton

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

#### 4.2.2 Implement `forward`

Translate the Python `forward` line by line with the [tensor operator syntax](#tensor-operators). Rules:

- Signature is `forward(Scope scope, ...)` — the `Scope` is the compute-tensor allocation unit; pass it to every sub-module call.
- Return a `Tensor` built from ops; no in-place mutation, no autograd.
- Call sub-modules through the typed accessors (`static_pointer_cast`), mirroring `self.sub_module(...)` calls.

#### 4.2.3 Test it

1. Add `test/src/<same path as the module>/<ClassName>.py` with a `unittest.TestCase` subclass (see e.g. `test/src/diffusers/models/transformers/flux2/`). Build the class with the smallest config that keeps the real shape structure, generate random weights with a seeded `torch.Generator`, pass them to the C++ CLI via `--param` options (or as quantized raw bytes via `--quant` when testing with a quantization spec), and compare against the Python reference.
2. Run it: `ctest --test-dir build -j8 --output-on-failure -R "<path-derived name>"`, e.g. `-R "src-diffusers-models-transformers-flux2-MyNorm"`.
3. If the C++ CLI crashes, the test writes a reproduction Python script to `build/test-repro/` — run it to reproduce the exact failure.

Each test directory is driven by its own CLI: `test/src/ggml/Tensor/` → `test-tensor-cli`, `test/src/nn/` → `test-nn-cli`, `test/src/diffusers/` → `test-diffusers-cli`, `test/src/diffusers/flux2` → `test-flux2-cli`, `test/src/transformers/` → `test-qwen3-cli`.

### 4.3 The pipeline

#### 4.3.1 Mirror the composition

The pipeline class holds the same components as the Python one. `from_pretrained` (see `Flux2KleinPipeline::from_pretrained` for example):

1. Constructs each model from its component's `config.json` (`Config::from_file`).
2. Loads each component's `model.gguf` by walking the module tree: `GGUFLoaderVisitor` wrapped in a `RethrowVisitor`, then `accept(...)`, `rethrow()`, and `validate()` (reports checkpoint tensors that were never loaded).
3. Loads the tokenizer from the component's `tokenizer/` folder.

#### 4.3.2 Translate `__call__` into `operator()`

- Map the numbered steps of Python's `__call__` one to one; keep the numbered step comments.
- `operator()` takes one `Context` per component plus a `GenerationOptions` struct that mirrors the `__call__` arguments (with the Python defaults) and **returns a `Computation<Tensor>`** — it builds the graph, it does not run it.
- Use `computation.scope(...)` for per-step sub-graphs; random noise comes from `computation.state(...)` with `std::mt19937` (a different RNG stream than torch — deterministic runs and tests pass random values as arguments instead).
- The eager denoising loop becomes a `fold` over a precomputed `Schedule` held in a `std::shared_ptr` (the computation executes lazily, so the schedule must outlive the construction).
- Public Python methods (`encode_prompt`, `prepare_image_latents`, …) become same-named pipeline methods that build their part of the graph.

#### 4.3.3 Decide what to drop

Port the core inference path and drop the Python surface the C++ runtime cannot express: the `DiffusionPipeline` base class and module registration, LoRA and other loaders, callbacks, progress bars, interrupts, compile/XLA hooks, `attention_kwargs`, and `output_type`/`return_dict` variants. Options the ported model cannot activate (e.g. guidance for a distilled model) may stay in the options struct, marked unused. Keep input validation: C++ throws where Python raises, and may be stricter where Python only warns — but preserve Python's rounding behavior for valid inputs. Graphs produce tensors only: pixel conversion, image post-processing, and output dataclasses happen after execution, outside the graph.

#### 4.3.4 Test it

1. Test the pipeline's public methods first (one test each: prompt encoding, image encoding, one denoising step, decoding, the latent pack/unpack helpers), exactly like the module tests.
2. Then add the end-to-end `test_call` with a config in the real model's shape class: feed the same `init_latents` to both sides (noise is not comparable across RNGs) and compare the decoded pixels — exact (`rtol=1e-4`) for dequantized weights, or a small pixel tolerance (e.g. 2.5/255) when running with the quantization config.

## 5. Check the single-device suite passes

With the whole port in place, run the full suite on a single CPU device (the default environment, `N_DEVICES=1`):

```bash
ctest --test-dir build -j8 --output-on-failure
```

Every test must be green before moving on — each one runs the C++ CLI and the pinned Python reference with identical inputs and compares the outputs. Fix failures at the unit level (re-run the single test with `-R "<name>"`): never proceed to sharding on a red single-device suite.

## 6. Run the sharded multi-device path

Run the same suite through the sharding runtime on two CPU devices:

```bash
N_DEVICES=2 ctest --test-dir build -j8 --output-on-failure
```

What to expect:

- Every sharded run prints the allocation plan to stderr (the format is described in [step 8](#8-check-the-sharding-plan-is-optimal)).
- An infeasible plan makes the run throw `run(): the allocation plan is infeasible: <reason>` plus a trace dump, and the test fails.
- A few tests are single-device only (e.g. the raw `flash_attn` kernel path) and skip themselves when `N_DEVICES > 1` — a skip is not a failure.

If the suite is green, go to [step 8](#8-check-the-sharding-plan-is-optimal) and check that the plan is optimal. If it fails with an infeasible plan, go to [step 7](#7-plan-the-sharded-path-if-the-plan-is-infeasible).

## 7. Plan the sharded path, if the plan is infeasible

Read the infeasibility reason the planner printed (it names the op or tensor that cannot be planned, followed by a trace dump) and fix the model against the meta backend's planning rules — usually with a module patch:

- **A view of a sharded tensor inherits no shard** unless it covers the sharded axis in full, starting from its beginning. Consequence: the sharded q/k/v that `flash_attn` requires cannot be read out of a fused QKV output — replace the fused projection with direct row-parallel `Linear`s (weight `S(1)` → output `S(0)`) (e.g. `Flux2FusedQKVProjection`).
- **Elementwise products must stay on one axis.** A product of two column-parallel (`S(0)`) tensors is cross-device when the halves live on different devices — e.g. a SwiGLU `gate * up` read out of a fused `gate | up` output. Split into separate `gate` / `up` column-parallel projections: each output is `S(0)`, the product stays `S(0)`, and the following row-parallel projection can reduce it (`S(0)` → `P` → AllReduce → `R`).
- **Concat along a sharded axis is forbidden** by the meta backend. A fused output projection over `[attn; mlp]` inputs cannot stay one `Linear` in sharding mode — split it into two row-parallel projections and add the outputs (linearity of the matmul; e.g. `Flux2FusedAttentionOutput`).
- **Keep ops in sharded states.** Some algebraic rewrites break planning: `rsqrt` is implemented as `pow(x, -0.5)`, not `1.0f / sqrt(x)` — the scalar-broadcast division would lift the scalar to a replicated tensor, and `R / S(a)` cannot be executed on the meta backend.
- **`flash_attn` shards along the heads axis** — the head count must be divisible by the device count.

Re-run `N_DEVICES=2 ctest --test-dir build -j8 --output-on-failure` after each patch; repeat until the plan is feasible and the suite is green, then check the plan in [step 8](#8-check-the-sharding-plan-is-optimal).

### Module patches

The reference form of a module stays 1:1 with Python; an adjustment the sharding planner forces (this step) is a **module patch** applied only when the graph is traced through a sharding runtime, never unconditionally. Every current patch splits a fused weight; the pattern generalizes to any adjustment.

**Two forms under one name.** The patched parent keeps both forms under the same child names, so the module tree and the checkpoint tensor names are Python's in both modes:

- **Reference form (default):** plain `Linear` children, exactly like Python's `nn.Linear` (e.g. `attn.to_qkv_mlp_proj` = `Linear(query_dim, 3·inner + mlp_out)`); `forward` calls them and `split` / `chunk`s their output, mirroring Python line by line. The generic `GGUFLoaderVisitor` traversal loads the fused tensors as-is.
- **Patch form:** a dedicated patch module holds the split children instead. Examples: [`Flux2FusedQKVProjection`](src/diffusers/models/transformers/flux2/Flux2FusedQKVProjection.hpp) replaces `to_qkv_mlp_proj` with `q` / `k` / `v` / `mlp_in_gate` / `mlp_in_up` Linears; [`Flux2FusedAttentionOutput`](src/diffusers/models/transformers/flux2/Flux2FusedAttentionOutput.hpp) replaces `to_out` with `attn` / `mlp` Linears; in `Flux2FeedForward` the `linear_in` child is a [`Flux2FusedLinearIn`](src/diffusers/models/transformers/flux2/Flux2FusedLinearIn.hpp) with `gate` / `up` Linears instead of a fused `Linear`.

**Mode selection.** `Scope::runtime().use_sharding()` is true only when the active scope is traced through a `ShardingRuntime` (meta backend); it is queryable at construction time (before any scope exists it reports not sharded) and during `forward`. The constructor records it (`sharded_ = Scope::runtime().use_sharding()`), builds the mode-appropriate children, and `forward` branches on the same query, throwing a clear error if the module was built for the other mode.

**Patch module anatomy** (see the three examples): `Linear` children named after the split projections, with typed accessors; a `forward` that performs the split computation (one GEMM per projection, returned as a tuple); and an `accept()` that always dispatches to the module-specific visit (`visitor.visit(*this, std::move(path))`), never to the generic traversal.

**Loader plumbing** — one overload per patch type, in three places:

1. `src/nn/Visitor.hpp` declares the no-op `virtual void visit(<Patch>&, std::vector<std::string>) {}`; `RethrowVisitor` wraps it with `visitWithCatch`.
2. `GGUFLoaderVisitor` implements it: resolve the fused tensor at the patch's own path (`module_path(path, {}, {"weight"})`), validate its shape, and fill each split child from a slice:
   - row slices via `read_rows` (rows are contiguous in the file — one seek+read), e.g. `to_qkv_mlp_proj` → `q` / `k` / `v` / `mlp_in_gate` / `mlp_in_up`;
   - column slices via `read_cols` (one seek+read per row; boundaries must align to the quantization block size so quantized types like Q8_0 slice without dequantization), e.g. `to_out` → `attn` / `mlp`;
   - a fused 1-D bias is column-sliced as well (`linear_in.bias` → `gate.bias` / `up.bias`).
3. The test CLI's `CreateParametersVisitor` mirrors the overload: the test still passes the fused tensor under the Python name (`--param-linear_in-weight`) and the visit slices it — F32 rows/cols, or raw quantized bytes (`slice_*_bytes`) when the quantization spec resolves a non-F32 type.

The split math is identical to the reference — a row or column slice of a projection *is* that projection — so a patch changes where the computation lives, never what it computes. `GGUFLoaderVisitor` may reference only patch modules; every real module loads through the generic traversal.

## 8. Check the sharding plan is optimal

A feasible plan is not automatically a good one. Every sharded run prints the plan to stderr — inspect a passing run with `N_DEVICES=2 ctest --test-dir build -V -R "<name>"`, or run the test's CLI binary directly:

```
=== plan (total cost 1234) ===
  [id] op or tensor name: <produced> --<bridge>--> <required>
  ...
meta device callback table (ggml_backend_meta_split_state per static tensor):
  <name>  axis 1    ne=[..., ...]
Total traced tensor memory (sum, not peak): X (sharded) + Y (replicated) = Z
```

Each line is one traced tensor in execution order. `R` = replicated on every device, `S(a)` = sharded along axis `a`, `P` = partial (one slice per device); a `P --AllReduce--> R` bridge is the collective that sums the partials. The callback table holds the per-device split of every static (weight) tensor — `MIRRORED` means the weight is duplicated on every device.

The planner's DP prices a candidate by the sum of its inputs' preference ranks (`S` < `P` < `R`), so **a fully sharded path — every input in some `S(a)` — always wins whenever it is feasible**. Check the printed plan against that, and restructure (usually with another [module patch](#module-patches)) while any of these hold:

- **A large tensor is produced `R` or `P` although `S(a)` is possible** — something in the graph forces the unsharded state (a view, a forbidden concat, a cross-axis product); find it from the plan's execution order and restructure.
- **More AllReduce bridges than necessary** — the canonical minimum is one per row-parallel block. A tensor reduced to `R` and then consumed replicated by several readers, or reduced more than once, is a restructure candidate.
- **A large weight is `MIRRORED` in the callback table** — that duplicates the weight on every device; it should be sharded along an axis the meta backend allows.
- **Uneven per-device splits** — the `ne=[...]` vectors should be roughly even (the default split is even; `TENSOR_SPLITS` overrides it); a lopsided split is a load imbalance.

After each restructure, re-run the same test and compare `total cost`, the bridge count, and the replicated bytes — a restructure is only a win if the numbers improve and the `N_DEVICES=2` suite stays green. Repeat until the plan is fully sharded where the rules allow, the bridges are minimal, and the weights are sharded.

## 9. Analyze the quantization

The goal: the most compressed model whose output still matches the Python reference within the end-to-end tolerance (the pipeline test allows a max pixel difference of 2.5/255).

The tool is [test/utils/quant_quality.py](test/utils/quant_quality.py). It builds the same tiny pipeline as the end-to-end `test_call`, runs it once with the original float32 weights (the reference), and again under every candidate spec: each parameter is quantized and dequantized exactly the way `utils/convert-model` does, and the resulting float32 values are loaded back into the Python reference, so both sides see identical values. For every spec it reports, against the float32 reference:

- the text encoder's prompt-embedding error (relative L2 + max abs), computed with identical token inputs;
- the transformer's first denoising-step noise prediction error (identical latents and float32 embeddings, so the transformer error is isolated from the encoder's);
- the final image error (max / mean absolute pixel difference, 0–255 scale);
- the storage the GGUF would take (MiB and % of float32).

Run it from the `test/` directory:

```bash
uv run python utils/quant_quality.py
```

Iterate on candidate specs the way the script does: start from a baseline (e.g. `Q8_0` everywhere except the attention norm weights), then test tensor group by tensor group — modulation, input embedders and timestep embeddings, output projections, attention projections, feed-forwards, the single-stream blocks, the text encoder's embedding table / attention / MLPs, the VAE — and promote a group to `BF16`/`F32` only when the measured error says it is sensitive. Expectations the script's numbers confirm: the VAE stays float (it decodes pixels), norm weights (q/k norms, layernorms) stay `F32`, the big GEMM weights take `Q8_0`; embeddings, modulation, and input/output projections are usually the sensitive ones. The table decides what stays float, not taste.

## 10. Write the quantization config

1. Write the winning spec from step 9 into `utils/convert-model/` (JSON: one entry per component, each a single type name or a per-tensor pattern map — see `flux2-klein-9b-q8.json` for example).
2. Convert the Hugging Face repo: `uv run python src/convert_hf_safetensors_to_gguf.py <org/model> --type @<config>.json` (see the [README](README.md#getting-a-model)).

When validating the spec end-to-end in the test suite, feed the C++ side the raw quantized bytes via `--quant` (through `TestCase.quant_params`) and run the Python reference with the dequantized values, so both sides compute with exactly the same numbers.

## Done

A port is **complete** when all of the following hold:

1. **The single-device CPU path works** — the full test suite passes with the default `N_DEVICES=1` (steps 4–5).
2. **The multi-device path works** — the suite passes with `N_DEVICES=2` under a feasible, near-optimal sharding plan (steps 6–8).
3. **A quantization config exists** for the model in `utils/convert-model/` (steps 9–10).
4. **All tests pass** — every unit numerically verified against the pinned Python reference, component by component and end to end.

A pipeline meeting all four is marked **Ported** — see the [status definition in the README](README.md#supported-pipelines).

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
