#pragma once

#include "ggml/Runtime.hpp"
#include "ggml/Computation.hpp"
#include <utility>

class Device;
class ShardingRuntime;
class Backend;
class ProgressBar;

class ExecutionRuntime : public Runtime {
public:
    static ExecutionRuntime Default;

    // Executes the computation in one pass and returns its final value.
    //
    // backends maps every pinned (weights) context to the backend the
    // weights are loaded on. Each scope's graph runs on the backend of
    // the weights it consumes: a scope whose graph leaves (directly, or
    // through a view) all live in the pinned contexts of a single
    // backend runs on that backend; a scope with no pinned leaves (one
    // that only moves state values around) runs on the CPU; a scope
    // whose leaves span several backends is an error -- the user must
    // restructure the computation so every scope's weights live on one
    // backend.
    //
    // All the state values live on the CPU for the whole run: the state
    // context is allocated on the CPU, a state cell a scope's graph
    // reads is bound to a scope-local input of the scope context (the
    // binding copies the cell's current value before every execution of
    // the scope), and the values the scopes persist (the saves) are
    // copied back into the state context after every execution. Since
    // no state tensor crosses a scope boundary inside a graph, the
    // scopes may run on different backends.
    //
    // Memory management is part of the execution: the pinned (weights)
    // contexts and the state context are allocated once, on the first
    // run, and stay resident for the lifetime of their contexts -- the
    // weights are shared across all the computations, the state carries
    // the values across the scopes and the runs. Every scope context is
    // allocated right before its graph executes and freed right after:
    // nothing in a scope context leaks outside. The values that survive
    // are exactly the ones the computation dereferences (operator*):
    // they are materialized into the state context at the point of the
    // dereference -- including the value this method returns, which the
    // caller reads after the scope contexts are freed -- and allocated
    // here with the rest of the state context.
    template <class T>
    T run(
        const std::vector<std::pair<Context*, Backend*>>& backends,
        std::mt19937& rng,
        Computation<T> computation,
        ShardingRuntime* sharding = nullptr,
        ProgressBar* progress = nullptr
    ) const {
        // The caller still holds the returned value after run() returns,
        // when the scope contexts are already freed: the final value is
        // an implicit dereference, materialized into the state context
        // like every other dereferenced value.
        auto value = *computation;

        run(backends, rng, *computation.desc(), sharding, progress);

        return value;
    }

    template<class T>
    std::vector<T> read(const Tensor& tensor) {
        constexpr auto expected = Tensor::DType<T>::value;

        if (tensor.dtype() != expected)
            throw std::invalid_argument("read(): dtype mismatch '" + std::string(ggml_get_name(*tensor)) + "': expected " + std::string(ggml_type_name(expected)) + ", but got " + std::string(ggml_type_name(tensor.dtype())));

        std::vector<T> data(
            ggml_nelements(*tensor)
        );

        if (data.size() * sizeof(T) != ggml_nbytes(*tensor))
            throw std::invalid_argument("read(): data size mismatch '" + std::string(ggml_get_name(*tensor)) + "': expected " + std::to_string(data.size() * sizeof(T)) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_get(
            *tensor,
            data.data(),
            0,
            ggml_nbytes(*tensor)
        );

        return std::move(data);
    }

    // -------------------------------------------------------------------------
    // Tensor creation / initialization
    // -------------------------------------------------------------------------

    ggml_tensor* new_tensor(
        ggml_type type,
        int n_dims,
        const int64_t* ne
    ) override;

    ggml_tensor* new_tensor_1d(
        ggml_type type,
        int64_t ne0
    ) override;

    void set_input(
        ggml_tensor* tensor
    ) override;

    void set_param(
        ggml_tensor* tensor
    ) override;

    void set_output(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* fill(
        ggml_tensor* tensor,
        float value
    ) override;

    // Copy / cast
    ggml_tensor* cont(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* dup(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* cast(
        ggml_tensor* tensor,
        ggml_type type
    ) override;

    ggml_tensor* cpy(
        ggml_tensor* src,
        ggml_tensor* dst
    ) override;

    // Unary
    ggml_tensor* sqrt(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* exp(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* log(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* sin(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* cos(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* sigmoid(
        ggml_tensor* tensor
    ) override;

    // Binary
    ggml_tensor* add(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* sub(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* mul(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* div(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    // Scalar
    ggml_tensor* scale(
        ggml_tensor* tensor,
        float value
    ) override;

    ggml_tensor* clamp(
        ggml_tensor* tensor,
        float min,
        float max
    ) override;

    // Matrix
    ggml_tensor* mul_mat(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    // Reshape
    ggml_tensor* reshape_1d(
        ggml_tensor* tensor,
        int64_t ne0
    ) override;

    ggml_tensor* reshape_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1
    ) override;

    ggml_tensor* reshape_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2
    ) override;

    ggml_tensor* reshape_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3
    ) override;

    // Permute / transpose
    ggml_tensor* permute(
        ggml_tensor* tensor,
        int axis0,
        int axis1,
        int axis2,
        int axis3
    ) override;

    // Views
    ggml_tensor* view_1d(
        ggml_tensor* tensor,
        int64_t ne0,
        size_t offset
    ) override;

    ggml_tensor* view_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        size_t nb1,
        size_t offset
    ) override;

    ggml_tensor* view_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        size_t nb1,
        size_t nb2,
        size_t offset
    ) override;

    ggml_tensor* view_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb1,
        size_t nb2,
        size_t nb3,
        size_t offset
    ) override;

    // Repeat
    ggml_tensor* repeat(
        ggml_tensor* tensor,
        ggml_tensor* target
    ) override;

    // Concatenation
    ggml_tensor* concat(
        ggml_tensor* a,
        ggml_tensor* b,
        int dim
    ) override;

    // Reduction
    ggml_tensor* sum_rows(
        ggml_tensor* tensor
    ) override;

    ggml_tensor * flash_attn_ext(
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* mask,
        float scale,
        float max_bias,
        float logit_softcap) override;

    ggml_tensor * conv_2d_direct(
        ggml_tensor* a,
        ggml_tensor* b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) override;

    ggml_tensor* get_rows(
        ggml_tensor* a,
        ggml_tensor* b) override;

    ggml_tensor* norm(
        ggml_tensor* a,
        float eps) override;

    ggml_tensor* rms_norm(
        ggml_tensor* a,
        float eps) override;

    ggml_tensor* rope_ext(
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
        float beta_slow) override;

    ggml_tensor* rope_multi(
        ggml_tensor* a,
        ggml_tensor* b,
        ggml_tensor* c,
        int n_dims,
        int sections[GGML_MROPE_SECTIONS],
        int mode,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) override;

    ggml_tensor* pool_2d(
        ggml_tensor* a,
        ggml_op_pool op,
        int k0,
        int k1,
        int s0,
        int s1,
        float p0,
        float p1) override;

    ggml_tensor* interpolate(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        uint32_t mode) override;

    ggml_tensor* upscale(
        ggml_tensor* a,
        int scale_factor,
        ggml_scale_mode mode) override;

private:
    void bind(std::mt19937& rng, Context& context, bool once_only, std::vector<std::pair<ggml_tensor*, std::vector<std::byte>>>* snapshots = nullptr) const;
    void copy(const Tensor& src, const Tensor& dst) const;
    void run(
        const std::vector<std::pair<Context*, Backend*>>& backends,
        std::mt19937& rng,
        ComputationDescription& desc,
        ShardingRuntime* sharding,
        ProgressBar* progress
    ) const;
};
