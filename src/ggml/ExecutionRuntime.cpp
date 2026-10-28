#include "ggml/ExecutionRuntime.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Computation.hpp"
#include "ggml/ShardingRuntime.hpp"
#include "ProgressBar.hpp"
#include <chrono>
#include <thread>

ExecutionRuntime ExecutionRuntime::Default;

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f GB", double(bytes) / (1024.0 * 1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f MB", double(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes) + " B";
}

static void print_buffer(const Device& device, size_t bytes, const char* usage) {
    if (bytes == 0)
        return;

    std::cerr << "Allocated buffer to "
              << ggml_backend_dev_name(*device)
              << " of size "
              << format_bytes(bytes)
              << usage
              << std::endl;
}

void ExecutionRuntime::run(Backend& backend, std::mt19937& rng, ComputationDescription& desc, ShardingRuntime* sharding, ProgressBar* progress) const {
    // The execution point: all the module forwards have been called —
    // all the low-level ggml tensor chains exist in their contexts.

    auto buft = backend.device().buffer_type();

    std::vector<ggml_cgraph*> graphs;
    graphs.reserve(desc.scopes.size());

    if (progress)
        progress->push("Initializing", desc.scopes.size());

    // Build one graph per scope, built in the scope's context (its memory
    // comes from the context's pool): the body's tensor chain was
    // already recorded there at build time; ggml_build_forward_expand
    // expands it into the graph (the nodes, and the leaves — the
    // bound inputs of this context and the pre-allocated weights /
    // state tensors). On a non-meta backend the save copies are nodes
    // of the graph as well.
    //
    // The graphs must be built BEFORE the meta buffers are allocated:
    // the meta backend is brittle about this order.
    for (auto& r : desc.scopes) {
        auto* gf = ggml_new_graph_custom(**r.context, r.context->capacity(), /*grads=*/false);

        // Build graph for every output Tensor.
        for (const auto& out : r.outputs)
            ggml_build_forward_expand(gf, *out);

        graphs.push_back(gf);

        if (progress)
            progress->next();
    }

    if (progress)
        progress->pop();

    // The meta backend plans the whole trace before any meta buffer is
    // allocated: the plan's callback states are what the meta buffer
    // allocation reads (the per-device split of every static tensor).
    if (sharding != nullptr) {
        auto& plan = sharding->plan();

        if (plan.infeasible)
            throw std::runtime_error(
                "run(): the allocation plan is infeasible: " + plan.infeasible_reason + "\n\n" + sharding->dump_trace());

        std::cerr << plan.to_string();

        // The plan counts EVERY traced tensor (the weights plus every
        // intermediate value of every scope): it is the sum of the
        // tensor sizes, NOT the memory that gets allocated -- the
        // weights buffer holds the static tensors, and every scope
        // buffer holds only the peak of simultaneously-live tensors
        // (the liveness-aware graph allocation).
        auto total_sharded_bytes = 0.0;
        auto total_replicated_bytes = 0.0;

        for (auto& node : plan.nodes) {
            auto bytes = ggml_nbytes(sharding->raw_of()[node.id]);

            if (node.produced.type == ShardingRuntime::Dist::R)
                total_replicated_bytes += bytes;
            else
                total_sharded_bytes += bytes;
        }

        std::cerr << "Total traced tensor memory (sum, not peak): "
                  << format_bytes(total_sharded_bytes)
                  << " (sharded) + "
                  << format_bytes(total_replicated_bytes)
                  << " (replicated) = "
                  << format_bytes(total_sharded_bytes + total_replicated_bytes)
                  << std::endl;
    }

    // Allocate the pinned contexts (the weights): they stay resident for
    // the whole run -- and every later run -- every scope reads them.
    // The buffer is owned by the context, so a re-run (or another
    // computation sharing the weights) skips the allocation.
    for (auto& pin : desc.pinned) {
        if (pin == nullptr)
            continue;

        pin->allocate(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        print_buffer(backend.device(), pin->allocated_size(), " (weights)");
    }

    // Allocate the state context that is shared between all the scopes:
    // it stays resident for the whole run -- and every later run -- it
    // carries every value the computation dereferenced (the state cells
    // they were materialized into), and the caller reads the
    // computation's final value from it.
    if (desc.state) {
        desc.state->allocate(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        print_buffer(backend.device(), desc.state->allocated_size(), " (compute)");
    }

    //for (auto& r : desc.scopes)
        //if (r.context)
            //r.context->allocate(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (progress)
        progress->push("Executing", desc.scopes.size());

    // The weights: the one-shot bindings are streamed once into the
    // weights buffer.
    for (auto& pin : desc.pinned)
        bind(rng, *pin, /*once_only=*/true);

    // The state's one-shot bindings: written into the allocated state
    // tensors before any execution reads them.
    if (desc.state)
        bind(rng, *desc.state, /*once_only=*/true);

    // The execution units: the scopes of a unit run as one named
    // progress step (one increment per scope execution of the unit)
    // instead of a separate "Computing" progress bar per scope.
    std::vector<int> unit_of(desc.scopes.size(), -1);
    std::vector<int> unit_total(desc.units.size(), 0);

    for (size_t u = 0; u < desc.units.size(); ++u) {
        for (size_t i = desc.units[u].begin; i < desc.units[u].end; ++i) {
            unit_of[i] = static_cast<int>(u);

            // Count the same scope executions the loop below performs.
            if (desc.scopes[i].context)
                unit_total[u] += desc.scopes[i].repeat ? static_cast<int>(desc.scopes[i].repeat->count) : 1;
        }
    }

    // The unit whose progress bar is open (-1: none).
    int unit = -1;

    for (auto i = 0; i < desc.scopes.size(); ++i) {
        auto& r = desc.scopes[i];
        auto* gf = graphs[i];

        if (!r.context)
            continue;

        // Open / close the progress bar of the scope's execution unit.
        auto scope_unit = unit_of[i];

        if (scope_unit != unit) {
            if (progress && unit >= 0)
                progress->pop();

            if (progress && scope_unit >= 0)
                progress->push(desc.units[scope_unit].name, unit_total[scope_unit]);

            unit = scope_unit;
        }

        // Allocate this scope's scratch context on demand: at any point
        // in time only the weights, the state, and at most one scope
        // context are resident. The tensors are unassigned first: a
        // re-run finds a clean context (the pointers of the previous
        // run's freed buffers would otherwise leak into the new
        // allocation). 
        //
        // Liveness-aware allocation (the graph allocator): a tensor's
        // storage is freed as soon as the graph stops consuming it, so
        // the buffer holds the peak of simultaneously-live tensors
        // instead of every tensor of the graph.
        r.context->allocate_graph(gf, buft);
        print_buffer(backend.device(), r.context->allocated_size(), " (graph)");

        // A PARTIAL (row-parallel) tensor on a device with a zero-sized
        // slice is never computed (the meta backend disables the compute
        // of every zero-slice tensor), so its storage keeps the stale
        // bytes the allocator left there; the AllReduce's in-place
        // 0.0f-scale zeroing of such a device cannot clear a stale NaN
        // (0.0f * NaN == NaN), and the butterfly sum would add the stale
        // values into the real result. Seed the storage with zero now,
        // while the graph allocation is fresh.
        if (sharding != nullptr)
            sharding->zero_partials();

        // The graph allocator (gallocr) treats a scope's bound inputs as
        // ordinary tensors: it may reuse their storage in place (an
        // in-place op writes its output over a leaf's buffer) and it frees
        // a leaf once its last consumer is allocated. For a single
        // execution that is harmless (each input is read once), but a
        // repeat scope re-executes the same graph, and the once-only
        // inputs are NOT re-written on every iteration -- so from the
        // second execution on they would be read back corrupted.
        // Snapshot the once-only inputs now, and restore them before every
        // execution below.
        std::vector<std::pair<ggml_tensor*, std::vector<std::byte>>> snapshots;

        // The bound inputs (once): written into the freshly allocated graph
        // tensors (ggml_backend_tensor_set).
        bind(rng, *r.context, /*once_only=*/true, r.repeat ? &snapshots : nullptr);

        // Handle repeating computation with re-binding and state carrying.
        if (r.repeat) {
            // A unit scope advances the unit's progress bar below.
            if (progress && scope_unit < 0)
                progress->push("Computing (" + std::to_string(ggml_graph_n_nodes(gf)) + " graph nodes)", r.repeat->count);

            // The graph is a single allocation with multiple computations:
            // compute the SAME graph repeatedly. The loop state is carried
            // by the saves: the body's output is written back into the
            // loop's state cells after every execution (a node of the
            // graph on a non-meta backend, a host-side copy on the meta
            // backend).

            for (r.repeat->iter = 0; r.repeat->iter < r.repeat->count; ++r.repeat->iter) {
                // The re-bindable inputs: rewritten on every (re-)execution.
                bind(rng, *r.context, /*once_only=*/false);

                // Restore the once-only inputs the graph allocator may have
                // reused in place during the previous execution.
                for (auto& [t, data] : snapshots)
                    ggml_backend_tensor_set(t, data.data(), 0, data.size());

                if (ggml_backend_graph_compute(*backend, gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("run(): ggml_backend_graph_compute failed");

                // Copy feedback tensors.
                for (const auto& save : r.saves) {
                    if (save.is_feedback)
                        copy(save.src, save.dst);
                }

                if (progress)
                    progress->next();
            }

            if (progress && scope_unit < 0)
                progress->pop();
        }

        // Otherwise, handle singular computation.
        else {
            if (progress && scope_unit < 0)
                progress->push("Computing (" + std::to_string(ggml_graph_n_nodes(gf)) + " graph nodes)", 1);

            // The re-bindable inputs of a single-execution scope have no
            // iter clock to read, so they are written once, right before the
            // (only) execution.
            bind(rng, *r.context, /*once_only=*/false);

            if (ggml_backend_graph_compute(*backend, gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("run(): ggml_backend_graph_compute failed");

            if (progress)
                progress->next();

            if (progress && scope_unit < 0)
                progress->pop();
        }

        // Copy state tensors.
        for (const auto& save : r.saves)
            if (!save.is_feedback)
                copy(save.src, save.dst);

        // The scope is done: free its buffer. Nothing in the scope
        // context leaks outside -- the values that survive are in the
        // state context -- so every scope context is disposable.
        r.context->release();
    }

    if (progress && unit >= 0)
        progress->pop();

    if (progress)
        progress->pop();
}

void ExecutionRuntime::bind(std::mt19937& rng, Context& context, bool once_only, std::vector<std::pair<ggml_tensor*, std::vector<std::byte>>>* snapshots) const {
    for (auto& [tensor, binding] : context.bindings()) {
        if (binding.once != once_only)
            continue;

        // Dead tensors (outside the graph) are not allocated by the
        // liveness-aware graph allocator: nothing reads their value,
        // so skip the write.
        if ((*tensor)->buffer == nullptr)
            continue;

        auto bytes = binding.provider(rng);

        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::runtime_error(std::string("bind(): size mismatch for '") + ggml_get_name(*tensor) + "'");

        ggml_backend_tensor_set(*tensor, bytes.data(), 0, bytes.size());

        // Store bound bytes to snapshot for later restoring.
        if (snapshots)
            snapshots->emplace_back(*tensor, std::move(bytes));
    }
}

void ExecutionRuntime::copy(const Tensor& src, const Tensor& dst) const {
    // A tensor in a temporary context is copied into the state context
    // (a save) or back into the loop state of the same graph (a repeat
    // feedback): the value is read from the source tensor and written
    // into the destination tensor.
    size_t nbytes = ggml_nbytes(*src);
    std::vector<std::byte> data(nbytes);
    ggml_backend_tensor_get(*src, data.data(), 0, nbytes);
    ggml_backend_tensor_set(*dst, data.data(), 0, nbytes);

    // std::cerr << "[copy] " << src.name() << " -> " << dst.name() << std::endl;
}

// -------------------------------------------------------------------------
// Tensor creation / initialization
// -------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::new_tensor(
    ggml_type type,
    int n_dims,
    const int64_t* ne
) {
    return ggml_new_tensor(*Scope::context(), type, n_dims, ne);
}

ggml_tensor* ExecutionRuntime::new_tensor_1d(
    ggml_type type,
    int64_t ne0
) {
    return ggml_new_tensor_1d(*Scope::context(), type, ne0);
}

void ExecutionRuntime::set_input(
    ggml_tensor* tensor
) {
    ggml_set_input(tensor);
}

void ExecutionRuntime::set_param(
    ggml_tensor* tensor
) {
    // There is no GGML counterpart for this.
}

void ExecutionRuntime::set_output(
    ggml_tensor* tensor
) {
    ggml_set_output(tensor);
}

ggml_tensor* ExecutionRuntime::fill(
    ggml_tensor* tensor,
    float value
) {
    return ggml_fill(*Scope::context(), tensor, value);
}

// -----------------------------------------------------------------------------
// Copy / cast
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::cont(
    ggml_tensor* tensor
) {
    return ggml_cont(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::dup(
    ggml_tensor* tensor
) {
    return ggml_dup(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::cast(
    ggml_tensor* tensor,
    ggml_type type
) {
    return ggml_cast(*Scope::context(), tensor, type);
}

ggml_tensor* ExecutionRuntime::cpy(
    ggml_tensor* src,
    ggml_tensor* dst
) {
    return ggml_cpy(*Scope::context(), src, dst);
}

// -----------------------------------------------------------------------------
// Unary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::sqrt(
    ggml_tensor* tensor
) {
    return ggml_sqrt(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::exp(
    ggml_tensor* tensor
) {
    // GGML_OP_EXP has no split-state rule in the meta backend; the
    // elementwise GGML_UNARY(EXP) does (handle_generic carries the src
    // state over). Same computation, meta-planable.
    return ggml_unary(*Scope::context(), tensor, GGML_UNARY_OP_EXP);
}

ggml_tensor* ExecutionRuntime::log(
    ggml_tensor* tensor
) {
    return ggml_log(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::sin(
    ggml_tensor* tensor
) {
    return ggml_sin(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::cos(
    ggml_tensor* tensor
) {
    return ggml_cos(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::sigmoid(
    ggml_tensor* tensor
) {
    return ggml_sigmoid(*Scope::context(), tensor);
}

// -----------------------------------------------------------------------------
// Binary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::add(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_add(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::sub(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_sub(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::mul(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::div(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_div(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Scalar operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::scale(
    ggml_tensor* tensor,
    float value
) {
    return ggml_scale(*Scope::context(), tensor, value);
}

ggml_tensor* ExecutionRuntime::clamp(
    ggml_tensor* tensor,
    float min,
    float max
) {
    return ggml_clamp(*Scope::context(), tensor, min, max);
}

// -----------------------------------------------------------------------------
// Matrix operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::mul_mat(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul_mat(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Reshape
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::reshape_1d(
    ggml_tensor* tensor,
    int64_t ne0
) {
    return ggml_reshape_1d(*Scope::context(), tensor, ne0);
}

ggml_tensor* ExecutionRuntime::reshape_2d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1
) {
    return ggml_reshape_2d(*Scope::context(), tensor, ne0, ne1);
}

ggml_tensor* ExecutionRuntime::reshape_3d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2
) {
    return ggml_reshape_3d(*Scope::context(), tensor, ne0, ne1, ne2);
}

ggml_tensor* ExecutionRuntime::reshape_4d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3
) {
    return ggml_reshape_4d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        ne3
    );
}

// -----------------------------------------------------------------------------
// Permute / transpose
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::permute(
    ggml_tensor* tensor,
    int axis0,
    int axis1,
    int axis2,
    int axis3
) {
    return ggml_permute(
        *Scope::context(),
        tensor,
        axis0,
        axis1,
        axis2,
        axis3
    );
}

// -----------------------------------------------------------------------------
// Views
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::view_1d(
    ggml_tensor* tensor,
    int64_t ne0,
    size_t offset
) {
    return ggml_view_1d(
        *Scope::context(),
        tensor,
        ne0,
        offset
    );
}

ggml_tensor* ExecutionRuntime::view_2d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    size_t nb1,
    size_t offset
) {
    return ggml_view_2d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        nb1,
        offset
    );
}

ggml_tensor* ExecutionRuntime::view_3d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    size_t nb1,
    size_t nb2,
    size_t offset
) {
    return ggml_view_3d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        nb1,
        nb2,
        offset
    );
}

ggml_tensor* ExecutionRuntime::view_4d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    size_t offset
) {
    return ggml_view_4d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        ne3,
        nb1,
        nb2,
        nb3,
        offset
    );
}

// -----------------------------------------------------------------------------
// Repeat
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::repeat(
    ggml_tensor* tensor,
    ggml_tensor* target
) {
    return ggml_repeat(*Scope::context(), tensor, target);
}

// -----------------------------------------------------------------------------
// Concatenation
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::concat(
    ggml_tensor* a,
    ggml_tensor* b,
    int dim
) {
    return ggml_concat(*Scope::context(), a, b, dim);
}

// -----------------------------------------------------------------------------
// Reduction
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::sum_rows(
    ggml_tensor* tensor
) {
    return ggml_sum_rows(*Scope::context(), tensor);
}

ggml_tensor * ExecutionRuntime::flash_attn_ext(
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* mask,
    float scale,
    float max_bias,
    float logit_softcap)
{
    return ggml_flash_attn_ext(*Scope::context(), q, k, v, mask, scale, max_bias, logit_softcap);
}

ggml_tensor * ExecutionRuntime::conv_2d_direct(
    ggml_tensor* a,
    ggml_tensor* b,
    int s0,
    int s1,
    int p0,
    int p1,
    int d0,
    int d1)
{
    return ggml_conv_2d_direct(*Scope::context(), a, b, s0, s1, p0, p1, d0, d1);
}

ggml_tensor* ExecutionRuntime::get_rows(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return ggml_get_rows(*Scope::context(), a, b);
}

ggml_tensor* ExecutionRuntime::norm(
    ggml_tensor* a,
    float eps) 
{
    return ggml_norm(*Scope::context(), a, eps);
}

ggml_tensor* ExecutionRuntime::rms_norm(
    ggml_tensor* a,
    float eps)
{
    return ggml_rms_norm(*Scope::context(), a, eps);
}

ggml_tensor* ExecutionRuntime::rope_ext(
    ggml_tensor* a,
    ggml_tensor* b,
    ggml_tensor* c,
    int n_dims,
    int mode,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow) 
{
    return ggml_rope_ext(*Scope::context(), a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

ggml_tensor* ExecutionRuntime::pool_2d(
    ggml_tensor* a,
    ggml_op_pool op,
    int k0,
    int k1,
    int s0,
    int s1,
    float p0,
    float p1) 
{
    return ggml_pool_2d(*Scope::context(), a, op, k0, k1, s0, s1, p0, p1);
}

ggml_tensor* ExecutionRuntime::interpolate(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    uint32_t mode) 
{
    return ggml_interpolate(*Scope::context(), a, ne0, ne1, ne2, ne3, mode);
}

ggml_tensor* ExecutionRuntime::upscale(
    ggml_tensor* a,
    int scale_factor,
    ggml_scale_mode mode)
{
    return ggml_upscale(*Scope::context(), a, scale_factor, mode);
}
