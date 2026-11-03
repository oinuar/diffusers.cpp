#pragma once

#include "nn/Module.hpp"
#include "nn/Visitor.hpp"
#include "ggml/Tensor.hpp"
#include "ggml/Runtime.hpp"
#include <cstddef>
#include <optional>
#include <vector>

class Parameter : public Module {
public:
    // A split configuration: this parameter is a slice of a larger
    // "fused" checkpoint tensor. A reference model may store several
    // projections in one tensor (e.g. a fused QKV + MLP input
    // projection); a sharded restructuring replaces the fused
    // projection with separate Linears, and each of their weight/bias
    // Parameters carries the split configuration that tells the
    // loaders (GGUFLoaderVisitor, the test CLIs) which slice of the
    // fused tensor to read.
    struct Split {
        enum class Axis {
            Rows,  // the slice is the row range [start, end)
            Cols   // the slice is the column range [start, end)
        };

        Axis axis;
        size_t start;
        size_t end;
        // The fused tensor's path, as path segments, relative to the
        // parent of the module that owns this parameter. A Linear
        // registered as modules["q"] of module M references M's fused
        // tensor "to_qkv_mlp_proj.weight" as
        // {"to_qkv_mlp_proj", "weight"}.
        std::vector<std::string> fused;
    };

    // A permutation configuration: the parameter's values are loaded
    // as a permutation of the checkpoint's values, the permutation
    // being applied once at load time by the loaders (GGUFLoaderVisitor,
    // the test CLIs). Used to store a weight in a layout the ops
    // prefer (e.g. the ggml MROPE head layout for the q/k projections
    // and QK norms -- see Flux2PosEmbed).
    struct Permutation {
        enum class Axis {
            Rows,     // 2-D weight: output row j = input row perm[j]
            Elements  // 1-D weight: output element j = input element perm[j]
        };

        Axis axis;
        std::vector<size_t> perm;
    };

    Parameter(const Tensor::Shape& shape, const ggml_type& dtype = Tensor::DType<float>::value)
        : shape_(shape), dtype_(dtype), tensor_(), split_(), permutation_()
    {
    }

    // Declare this parameter as the rows [start, end) of the fused tensor.
    void split_rows(const std::vector<std::string>& fused, size_t start, size_t end) {
        split_ = Split{Split::Axis::Rows, start, end, fused};
    }

    // Declare this parameter as the columns [start, end) of the fused tensor.
    void split_cols(const std::vector<std::string>& fused, size_t start, size_t end) {
        split_ = Split{Split::Axis::Cols, start, end, fused};
    }

    // The split configuration, when this parameter is a slice of a fused tensor.
    const std::optional<Split>& split() const {
        return split_;
    }

    // Declare this parameter's values are loaded with its rows permuted:
    // output row j = input row perm[j].
    void permute_rows(std::vector<size_t> perm) {
        permutation_ = Permutation{Permutation::Axis::Rows, std::move(perm)};
    }

    // Declare this parameter's values are loaded with its elements
    // permuted: output element j = input element perm[j].
    void permute_elements(std::vector<size_t> perm) {
        permutation_ = Permutation{Permutation::Axis::Elements, std::move(perm)};
    }

    // The permutation configuration, when this parameter's values are
    // loaded permuted (see Permutation).
    const std::optional<Permutation>& permutation() const {
        return permutation_;
    }

    // The full module path of the fused tensor this parameter is a slice
    // of (see Split::fused), given this parameter's own module path, as
    // the visitors receive it. The parameter's path is
    // [..., parent, module, param], where module is the module whose
    // `modules` map holds this parameter (for a Linear's weight/bias
    // that's the Linear's name); the fused path is relative to that
    // module's parent, so drop the last two elements and append the
    // fused segments. The visitors join the result with their own
    // module-path separator (dot for the GGUF loader, dash for the
    // test CLIs).
    std::vector<std::string> split_fused_path(const std::vector<std::string>& path) const {
        if (!split_)
            throw std::runtime_error("Parameter has no split configuration");

        if (path.size() < 2)
            throw std::runtime_error(
                "Split configuration: the parameter path must name the owning module and the parameter");

        auto fused_path = std::vector<std::string>(path.begin(), path.end() - 2);
        fused_path.insert(fused_path.end(), split_->fused.begin(), split_->fused.end());

        return fused_path;
    }

    Tensor forward(Scope scope) {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter. Did you forget to set it?");

        scope.runtime().set_param(*tensor_);

        return tensor_;
    }
    
    void set(Tensor tensor) {
        tensor_ = tensor;
    }

    Tensor operator *() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter");

        return tensor_;
    }

    const Tensor* operator ->() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter");

        return &tensor_;
    }

    const Tensor::Shape& shape() const {
        return shape_;
    }

    ggml_type dtype() const {
        return dtype_;
    }

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    ggml_type dtype_;
    Tensor tensor_;
    std::optional<Split> split_;
    std::optional<Permutation> permutation_;
};
