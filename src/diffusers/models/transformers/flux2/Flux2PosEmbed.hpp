#pragma once

#include "nn/Module.hpp"
#include "ggml/Context.hpp"
#include "ggml/Runtime.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

//
// Flux2's per-axis rotary embedding, computed with a single
// GGML_ROPE_TYPE_MROPE op.
//
// The reference rotates each 32-dim axis with its own position column,
// rotating adjacent pairs (2l, 2l+1) within the axis. MROPE instead
// pairs (p, p + D/2) with the position of the sector p/sectors-per-axis.
// The two are identical after the even/odd (NEOX) head-dim permutation
// P, applied to the q/k weights once at load time (PermutedParameter):
//
//   MROPE pair j          = (x[j], x[j + D/2])   in MROPE layout
//   Flux2 pair (2j, 2j+1) = the same two values  in the original layout
//   sector j/(d_a/2)      = axis j/(d_a/2)
//
// with the per-pair frequency factor
//
//   ff_j = theta^(2l/d_a - 2j/D),   j = 16a + l  (D=128)
//
// so that MROPE's angle pos_a * theta^(-2j/D) / ff_j equals the
// reference's per-axis angle pos_a * theta^(-2l/d_a).
//
// Everything downstream (attention scores, v, output projection) is
// invariant under the same P on q and k, so no other op changes.
//
class Flux2PosEmbed : public Module {
public:
    Flux2PosEmbed(
        int64_t theta,
        const std::vector<int64_t>& axes_dim)
        : theta_(theta), axes_dim_(axes_dim)
    {
    }

    // x: (batch, seq, heads, head_dim) in the MROPE head layout (the
    // q/k projection weights are pre-permuted at load time, see
    // PermutedParameter).
    //
    // position_ids: (4, seq) int32 -- the four position blocks
    // [T, H, W, L] transposed to one block per axis (the joint
    // (seq, 4) ids, transposed and flattened to (4 * seq,)).
    //
    // Returns (batch, seq, heads, head_dim): the rope output is the
    // assembled head dimension (no slice/concat ops).
    Tensor forward(Scope scope, Tensor x, Tensor position_ids) {
        auto D = x.shape()[3];

        if (D % 2 != 0)
            throw std::runtime_error("Flux2PosEmbed: head_dim must be even");

        // Pairs per axis; unused axes stay 0 (MROPE sections).
        int sections[4] = {0, 0, 0, 0};

        int64_t sum_axes = 0;

        for (size_t i = 0; i < axes_dim_.size(); ++i) {
            if (i >= 4 || axes_dim_[i] <= 0 || axes_dim_[i] % 2 != 0)
                throw std::runtime_error("Flux2PosEmbed: axes_dim must have at most 4 even, positive entries");

            sections[i] = axes_dim_[i] / 2;
            sum_axes += axes_dim_[i];
        }

        if (sum_axes != D)
            throw std::runtime_error("Flux2PosEmbed: sum(axes_dim) must equal head_dim");

        // Per-pair frequency factors (static per model): pair j of axis
        // a (j = start_a + l, start_a = sum of the earlier axes' pair
        // counts) must rotate with the reference's per-axis angle
        // pos_a * theta^(-2l/d_a), while MROPE computes
        // pos_a * theta^(-2j/D) / ff_j -- hence ff_j = theta^(2l/d_a - 2j/D).
        auto ff = scope.context().create<float>({D / 2}, [=, axes_dim = axes_dim_, theta = theta_](std::mt19937&) {
            std::vector<float> factors(D / 2);
            int64_t start = 0;

            for (size_t a = 0; a < axes_dim.size(); ++a) {
                auto d_a = axes_dim[a];

                for (int64_t l = 0; l < d_a / 2; ++l) {
                    auto j = start + l;

                    factors[j] = powf(
                        (float)theta,
                        2.0f * (float)l / (float)d_a - 2.0f * (float)j / (float)D);
                }

                start += d_a / 2;
            }

            return factors;
        });

        // (4, seq) -> (4 * seq,): one MROPE position block per axis.
        auto pos = position_ids.flatten();

        auto rope = scope.runtime().rope_multi(
            *x,
            *pos,
            *ff,

            D,
            sections,

            GGML_ROPE_TYPE_MROPE,

            0,        // n_ctx_orig
            theta_,   // freq_base
            1.0f,     // freq_scale
            0.0f,     // ext_factor
            1.0f,     // attn_factor
            0.0f,     // beta_fast
            0.0f      // beta_slow
        );

        return Tensor(rope, x.shape());
    }

    //
    // The head-dim permutation P between the reference's adjacent
    // layout (pairs (2l, 2l+1)) and the ggml MROPE (NEOX) layout
    // (pairs (p, p + D/2)): the MROPE layout position p holds the
    // original head-dim position P(p).
    //
    static std::vector<size_t> mrope_permutation(int64_t head_dim) {
        std::vector<size_t> perm(head_dim);

        for (int64_t p = 0; p < head_dim; ++p)
            perm[p] = p < head_dim / 2 ? 2 * p : 2 * (p - head_dim / 2) + 1;

        return perm;
    }

    //
    // The same, applied per head block over a flattened
    // (heads * head_dim) dimension (e.g. the q/k projection rows).
    //
    static std::vector<size_t> mrope_permutation(int64_t head_dim, int64_t num_heads) {
        auto block = mrope_permutation(head_dim);
        std::vector<size_t> perm(num_heads * head_dim);

        for (int64_t h = 0; h < num_heads; ++h)
            for (int64_t p = 0; p < head_dim; ++p)
                perm[h * head_dim + p] = h * head_dim + block[p];

        return perm;
    }

private:
    int64_t theta_;
    std::vector<int64_t> axes_dim_;
};
