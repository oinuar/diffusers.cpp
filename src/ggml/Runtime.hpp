#pragma once

#include <cstddef>
#include <cstdint>
#include <ggml.h>

class Runtime {
public:
    virtual ~Runtime() = default;

    // True when the runtime shards tensors across multiple devices
    // (meta device): modules built under such a runtime use the
    // sharding-adjusted form of their graphs instead of the 1:1
    // reference form.
    virtual bool sharded() const { return false; }

    // True when the scope-boundary saves must be copied through host
    // memory instead of a CPY node of the producing scope's graph.
    //
    // The sharding runtime on an all-CPU meta device answers true: the
    // meta backend treats the meta buffer as a host buffer (every
    // underlying device is a host), and its graph splitter skips the
    // nodes that are views of host buffers -- so a CPY node whose
    // destination (the state cell or the loop state) lives in the
    // buffer would never run. Such a copy must fall back to a
    // host-side copy.
    virtual bool host_copy_saves() const { return false; }

    // -------------------------------------------------------------------------
    // Tensor creation / initialization
    // -------------------------------------------------------------------------

    virtual ggml_tensor* new_tensor(
        ggml_type type,
        int n_dims,
        const int64_t* ne
    ) = 0;

    virtual ggml_tensor* new_tensor_1d(
        ggml_type type,
        int64_t ne0
    ) = 0;

    virtual void set_input(
        ggml_tensor* tensor
    ) = 0;

    virtual void set_param(
        ggml_tensor* tensor
    ) = 0;

    virtual void set_output(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* fill(
        ggml_tensor* tensor,
        float value
    ) = 0;

    // -------------------------------------------------------------------------
    // Copy / cast
    // -------------------------------------------------------------------------

    virtual ggml_tensor* cont(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* dup(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* cast(
        ggml_tensor* tensor,
        ggml_type type
    ) = 0;

    virtual ggml_tensor* cpy(
        ggml_tensor* src,
        ggml_tensor* dst
    ) = 0;

    // -------------------------------------------------------------------------
    // Unary arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* sqrt(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* exp(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* log(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* sin(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* cos(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* sigmoid(
        ggml_tensor* tensor
    ) = 0;

    // -------------------------------------------------------------------------
    // Binary arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* add(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* sub(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* mul(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* div(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    // -------------------------------------------------------------------------
    // Scalar arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* scale(
        ggml_tensor* tensor,
        float value
    ) = 0;

    virtual ggml_tensor* clamp(
        ggml_tensor* tensor,
        float min,
        float max
    ) = 0;

    // -------------------------------------------------------------------------
    // Matrix operations
    // -------------------------------------------------------------------------

    virtual ggml_tensor* mul_mat(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    // -------------------------------------------------------------------------
    // Reshape
    // -------------------------------------------------------------------------

    virtual ggml_tensor* reshape_1d(
        ggml_tensor* tensor,
        int64_t ne0
    ) = 0;

    virtual ggml_tensor* reshape_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1
    ) = 0;

    virtual ggml_tensor* reshape_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2
    ) = 0;

    virtual ggml_tensor* reshape_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3
    ) = 0;

    // -------------------------------------------------------------------------
    // Permute / transpose
    // -------------------------------------------------------------------------

    virtual ggml_tensor* permute(
        ggml_tensor* tensor,
        int axis0,
        int axis1,
        int axis2,
        int axis3
    ) = 0;

    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------

    virtual ggml_tensor* view_1d(
        ggml_tensor* tensor,
        int64_t ne0,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        size_t nb1,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        size_t nb1,
        size_t nb2,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb1,
        size_t nb2,
        size_t nb3,
        size_t offset
    ) = 0;

    // -------------------------------------------------------------------------
    // Repeat / broadcast
    // -------------------------------------------------------------------------

    virtual ggml_tensor* repeat(
        ggml_tensor* tensor,
        ggml_tensor* target
    ) = 0;

    // -------------------------------------------------------------------------
    // Concatenation
    // -------------------------------------------------------------------------

    virtual ggml_tensor* concat(
        ggml_tensor* a,
        ggml_tensor* b,
        int dim
    ) = 0;

    // -------------------------------------------------------------------------
    // Reduction
    // -------------------------------------------------------------------------

    virtual ggml_tensor* sum_rows(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor * flash_attn_ext(
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* mask,
        float scale,
        float max_bias,
        float logit_softcap) = 0;

    virtual ggml_tensor * conv_2d_direct(
        ggml_tensor* a,
        ggml_tensor* b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) = 0;

    virtual ggml_tensor* get_rows(
        ggml_tensor* a,
        ggml_tensor* b) = 0;

    virtual ggml_tensor* norm(
        ggml_tensor* a,
        float eps) = 0;

    virtual ggml_tensor* rms_norm(
        ggml_tensor* a,
        float eps) = 0;

    virtual ggml_tensor* rope_ext(
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
        float beta_slow) = 0;

    virtual ggml_tensor* pool_2d(
        ggml_tensor* a,
        ggml_op_pool op,
        int k0,
        int k1,
        int s0,
        int s1,
        float p0,
        float p1) = 0;

    virtual ggml_tensor* interpolate(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        uint32_t mode) = 0;

    virtual ggml_tensor* upscale(
        ggml_tensor* a,
        int scale_factor,
        ggml_scale_mode mode) = 0;
};
