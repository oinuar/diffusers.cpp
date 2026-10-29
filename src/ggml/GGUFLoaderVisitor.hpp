#pragma once

#include "nn/Visitor.hpp"
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>
#include <ggml.h>
#include <gguf.h>

class Context;

class GGUFLoaderVisitor : public Visitor {
public:
    // One tensor from the GGUF file: its type, 2-D layout and the byte
    // offset of its data in the file.
    struct TensorInfo {
        ggml_type type;
        int64_t rows;   // ne[1]
        int64_t cols;   // ne[0] (row length, fastest dimension)
        std::streamoff offset;
    };

    GGUFLoaderVisitor(Context& context, const std::filesystem::path& path);
    ~GGUFLoaderVisitor();

    void visit(Parameter& parameter, std::vector<std::string> path) override;
    void visit(Flux2FusedQKVProjection& to_qkv_mlp_proj, std::vector<std::string> path) override;
    void visit(Flux2FusedAttentionOutput& to_out, std::vector<std::string> path) override;
    void visit(Flux2FusedLinearIn& linear_in, std::vector<std::string> path) override;

    void validate() const;

    size_t size() const {
        return lookup_.size();
    }

private:
    // Looks up the tensor by its dotted module path and describes where
    // its raw bytes live in the GGUF file. The tensor is removed from
    // lookup_ so that validate() can report tensors that were not loaded.
    TensorInfo load_tensor(const std::string& name);

    Context& context_;
    gguf_context* gguf_ctx_;
    std::shared_ptr<std::ifstream> file_;
    std::unordered_map<std::string, size_t> lookup_;
};
