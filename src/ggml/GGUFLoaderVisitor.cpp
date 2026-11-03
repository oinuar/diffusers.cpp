#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Context.hpp"
#include "nn/Parameter.hpp"
#include "nn/ModulePath.hpp"
#include <string>
#include <fstream>
#include <iostream>

// A one-shot provider that streams rows [start, end) of a 2-D GGUF tensor
// from the file. The rows are contiguous in the file, so it is a single
// seek + read of exactly the requested bytes.
static Context::Provider<std::byte> read_rows(
    const std::shared_ptr<std::ifstream>& file,
    const GGUFLoaderVisitor::TensorInfo& tensor,
    size_t start,
    size_t end,
    const std::string& name)
{
    auto row_bytes = ggml_row_size(tensor.type, tensor.cols);
    auto offset = tensor.offset + static_cast<std::streamoff>(start * row_bytes);
    auto n_bytes = (end - start) * row_bytes;

    return [file, offset, n_bytes, name](std::mt19937&) {
        std::vector<std::byte> buf(n_bytes);

        file->seekg(offset, file->beg);

        if (!*file)
            throw std::runtime_error("Error while loading Tensor '" + name + "': seek failed");

        file->read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n_bytes));

        if (file->gcount() != static_cast<std::streamsize>(n_bytes))
            throw std::runtime_error("Error while loading Tensor '" + name + "': read failed");

        return buf;
    };
}

// A one-shot provider that streams columns [start, end) of every row of a
// 2-D GGUF tensor from the file: one seek + read per row. The slice
// boundaries must fall on quantization block boundaries, so that
// quantized types (Q8_0, ...) are sliced without dequantization.
static Context::Provider<std::byte> read_cols(
    const std::shared_ptr<std::ifstream>& file,
    const GGUFLoaderVisitor::TensorInfo& tensor,
    size_t start,
    size_t end,
    const std::string& name)
{
    auto blck = static_cast<size_t>(ggml_blck_size(tensor.type));
    auto block_bytes = static_cast<size_t>(ggml_type_size(tensor.type));

    if (start % blck != 0 || (end - start) % blck != 0)
        throw std::runtime_error(
            "Error while loading Tensor '" + name +
            "': column range [" + std::to_string(start) + ", " + std::to_string(end) +
            ") is not aligned to the quantization block size (" + std::to_string(blck) + ")");

    auto row_bytes = ggml_row_size(tensor.type, tensor.cols);
    auto slice_offset = start / blck * block_bytes;
    auto slice_bytes = (end - start) / blck * block_bytes;

    return [
        file,
        offset = tensor.offset,
        rows = static_cast<size_t>(tensor.rows),
        row_bytes,
        slice_offset,
        slice_bytes,
        name
    ](std::mt19937&) {
        std::vector<std::byte> row(slice_bytes);
        std::vector<std::byte> buf;
        buf.reserve(rows * slice_bytes);

        for (size_t r = 0; r < rows; ++r) {
            file->seekg(offset + static_cast<std::streamoff>(r * row_bytes + slice_offset), file->beg);

            if (!*file)
                throw std::runtime_error("Error while loading Tensor '" + name + "': seek failed");

            file->read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(slice_bytes));

            if (file->gcount() != static_cast<std::streamsize>(slice_bytes))
                throw std::runtime_error("Error while loading Tensor '" + name + "': read failed");

            buf.insert(buf.end(), row.begin(), row.end());
        }

        return buf;
    };
}

// Reorders the raw bytes of a freshly loaded weight according to its
// permutation configuration (see Parameter::Permutation):
//
//   Rows:     output row j = input row perm[j] -- a pure byte reorder,
//             exact for every storage type.
//   Elements: output element j = input element perm[j] within each row
//             -- a fixed-size element reorder, exact for F32/F16/BF16.
//
// The providers are one-shot (weight load), so the buffer is permuted
// once, right after the read.
static Context::Provider<std::byte> permute_parameter(
    const Context::Provider<std::byte>& provider,
    ggml_type type,
    int64_t rows,
    int64_t cols,
    const Parameter::Permutation& permutation,
    const std::string& name)
{
    size_t element_bytes = 0;

    if (permutation.axis == Parameter::Permutation::Axis::Elements) {
        switch (type) {
            case GGML_TYPE_F32:    element_bytes = 4; break;
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:   element_bytes = 2; break;
            default:
                throw std::runtime_error(
                    "Error while loading Tensor '" + name +
                    "': element permutation is not supported for type " +
                    std::string(ggml_type_name(type)));
        }
    }

    return [=](std::mt19937& rng) {
        auto buf = provider(rng);
        auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));

        if (permutation.perm.size() != static_cast<size_t>(permutation.axis == Parameter::Permutation::Axis::Rows ? rows : cols))
            throw std::runtime_error(
                "Error while loading Tensor '" + name +
                "': permutation size mismatch");

        std::vector<std::byte> out(buf.size());

        if (permutation.axis == Parameter::Permutation::Axis::Rows) {
            for (size_t r = 0; r < rows; ++r) {
                if (r >= permutation.perm.size() || permutation.perm[r] >= static_cast<size_t>(rows))
                    throw std::runtime_error(
                        "Error while loading Tensor '" + name + "': permutation out of range");

                std::copy(
                    buf.begin() + permutation.perm[r] * row_bytes,
                    buf.begin() + (permutation.perm[r] + 1) * row_bytes,
                    out.begin() + r * row_bytes);
            }
        } else {
            for (size_t r = 0; r < rows; ++r) {
                for (size_t j = 0; j < cols; ++j) {
                    if (permutation.perm[j] >= static_cast<size_t>(cols))
                        throw std::runtime_error(
                            "Error while loading Tensor '" + name + "': permutation out of range");

                    std::copy(
                        buf.begin() + r * row_bytes + permutation.perm[j] * element_bytes,
                        buf.begin() + r * row_bytes + (permutation.perm[j] + 1) * element_bytes,
                        out.begin() + r * row_bytes + j * element_bytes);
                }
            }
        }

        return out;
    };
}

// Like Context::create, but for a tensor of an arbitrary GGUF type (e.g.
// a quantized type) that is streamed from the file by a one-shot provider.
static Tensor create_weight(
    Context& context,
    const std::string& name,
    const Tensor::Shape& shape,
    ggml_type type,
    const Context::Provider<std::byte>& provider)
{
    Scope scope(context);
    auto tensor = Tensor::empty(shape, type).input();
    tensor.name(name);
    context.bind<std::byte>(tensor, provider, /*once=*/true);
    return tensor;
}

static std::optional<std::filesystem::path> find_first_gguf(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
        return std::nullopt;

    if (!std::filesystem::is_directory(path))
        return path;

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            return entry.path();
    }

    return std::nullopt;
}

GGUFLoaderVisitor::GGUFLoaderVisitor(Context& context, const std::filesystem::path& path)
    : context_(context), gguf_ctx_(nullptr), file_(std::make_shared<std::ifstream>()), lookup_()
{
    auto gguf_path = find_first_gguf(path);

    if (!gguf_path)
        throw std::runtime_error(
            "No such GGUF file in path: " + path.string());

    file_->open(*gguf_path, std::ifstream::in | std::ifstream::binary);

    if (!file_->is_open())
        throw std::runtime_error("Failed to open GGUF file: " + gguf_path->string());

    //
    // Parse GGUF metadata only.
    //
    // No ggml context is requested here. Therefore
    // gguf_ctx_ does not own any ggml tensors.
    //
    gguf_ctx_ = gguf_init_from_file(
        gguf_path->c_str(),
        {
            /* .no_alloc = */ true,
            /* .ctx      = */ nullptr,
        });

    if (!gguf_ctx_)
        throw std::runtime_error(
            "Failed to initialize GGUF file: " +
            gguf_path->string());

    auto n_tensors = gguf_get_n_tensors(gguf_ctx_);
    lookup_.reserve(n_tensors);

    for (auto i = 0; i < n_tensors; ++i) {
        auto name = gguf_get_tensor_name(gguf_ctx_, i);

        if (!name)
            throw std::runtime_error(
                "GGUF tensor has no name");

        lookup_[name] = i;
    }
}

GGUFLoaderVisitor::~GGUFLoaderVisitor() {
    gguf_free(gguf_ctx_);
}

void GGUFLoaderVisitor::validate() const {
    std::string message;

    for (auto& [name, _] : lookup_) {
        if (!message.empty())
            message += "\n";

        message += "  - Tensor exists in checkpoint but was not loaded: " + name;
    }

    //if (!message.empty())
        //throw std::runtime_error("Error while validating the checkpoint:\n" + message);
}

GGUFLoaderVisitor::TensorInfo GGUFLoaderVisitor::load_tensor(const std::string& name) {
    auto it = lookup_.find(name);

    if (it == std::end(lookup_))
        throw std::runtime_error("Error while loading Tensor '" + name + "': Tensor not found");

    auto tensor_id = it->second;
    lookup_.erase(it);

    auto type = gguf_get_tensor_type(gguf_ctx_, tensor_id);
    auto ne = gguf_get_tensor_ne(gguf_ctx_, tensor_id);

    auto offset =
        gguf_get_data_offset(gguf_ctx_) +
        gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    return { type, ne[1], ne[0], offset };
}

GGUFLoaderVisitor::TensorBytes GGUFLoaderVisitor::load_tensor_bytes(
    const std::string& model_path,
    const Tensor::Shape& parameter_shape)
{
    auto it = lookup_.find(model_path);

    if (it == std::end(lookup_))
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Tensor not found");

    auto tensor_id = it->second;
    lookup_.erase(it);

    auto type = gguf_get_tensor_type(gguf_ctx_, tensor_id);
    auto ne = gguf_get_tensor_ne(gguf_ctx_, tensor_id);
    int n_dims = 0;

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (ne[i] > 1 || (i == 0 && ne[0] == 1)) {
            n_dims = i + 1;
        }
    }

    if (n_dims == 0)
        throw std::runtime_error(
            "GGUF tensor has invalid dimensions");

    Tensor::Shape gguf_shape(n_dims);

    for (auto r = 0; r < gguf_shape.rank(); ++r)
        gguf_shape[r] = ne[gguf_shape.rank() - 1 - r];

    if (parameter_shape != gguf_shape)
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter shape mismatch: expected " + parameter_shape.to_string() + ", got " + gguf_shape.to_string());

    const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    // Contiguous storage: a row of the fastest dimension, times the
    // outer dimensions.
    auto n_bytes = static_cast<size_t>(ggml_row_size(type, gguf_shape[gguf_shape.rank() - 1]));

    for (auto r = 0; r + 1 < gguf_shape.rank(); ++r)
        n_bytes *= gguf_shape[r];

    return {
        type,
        gguf_get_tensor_name(gguf_ctx_, tensor_id),
        [n_bytes, offs, file = file_, model_path](std::mt19937&) {
            std::vector<std::byte> buf(n_bytes);

            file->seekg(offs, file->beg);

            if (!*file)
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': seek failed");

            // Read the tensor data
            file->read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

            if (file->gcount() != static_cast<std::streamsize>(buf.size()))
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': read failed");

            return buf;
        }
    };
}

void GGUFLoaderVisitor::visit(Parameter& parameter, std::vector<std::string> path) {
    ModulePath module_path;

    // A split parameter (see Parameter::Split): its bytes live in a
    // fused checkpoint tensor, not at its own path.
    auto load = parameter.split()
        ? load_split(parameter, path, *parameter.split())
        : load_tensor_bytes(module_path(path), parameter.shape());

    auto provider = load.provider;

    // A permuted parameter (see Parameter::Permutation): load the raw
    // GGUF bytes, then reorder them according to the configuration.
    // A split parameter is permuted after the slice.
    if (auto permutation = parameter.permutation()) {
        auto shape = parameter.shape();
        auto cols = shape[shape.rank() - 1];
        auto rows = 1;

        for (auto r = 0; r + 1 < shape.rank(); ++r)
            rows *= shape[r];

        provider = permute_parameter(provider, load.type, rows, cols, *permutation, load.name);
    }

    Scope scope(context_);
    auto tensor = Tensor::empty(parameter.shape(), load.type);

    tensor.name(load.name);

    context_.bind<std::byte>(tensor, provider, /*once=*/true);

    parameter.set(tensor);
}

GGUFLoaderVisitor::TensorBytes GGUFLoaderVisitor::load_split(
    const Parameter& parameter,
    const std::vector<std::string>& path,
    const Parameter::Split& split)
{
    ModulePath module_path;

    auto fused_name = module_path(parameter.split_fused_path(path));

    auto fused = load_fused(fused_name);

    // The slice is named after its fused tensor and the parameter:
    // "...to_qkv_mlp_proj.weight-q-weight".
    auto name = fused_name + "-" + path[path.size() - 2] + "-" + path[path.size() - 1];

    auto shape = parameter.shape();

    if (split.axis == Parameter::Split::Axis::Rows) {
        if (shape.rank() != 2)
            throw std::runtime_error(
                "Error while loading Tensor '" + module_path(path) +
                "': a row split requires a 2-D parameter, got " + shape.to_string());

        if (split.end > static_cast<size_t>(fused.rows))
            throw std::runtime_error(
                "Error while loading Tensor '" + module_path(path) +
                "': the row split [" + std::to_string(split.start) + ", " + std::to_string(split.end) +
                ") of '" + fused_name + "' is out of bounds (the fused tensor has " +
                std::to_string(fused.rows) + " rows)");

        auto expected = Tensor::Shape({static_cast<int64_t>(split.end - split.start), fused.cols});

        if (shape != expected)
            throw std::runtime_error(
                "Error while loading Tensor '" + module_path(path) +
                "': Parameter shape mismatch: expected " + expected.to_string() +
                ", got " + shape.to_string());

        return { fused.type, name, read_rows(file_, fused, split.start, split.end, name) };
    }

    if (shape.rank() > 2)
        throw std::runtime_error(
            "Error while loading Tensor '" + module_path(path) +
            "': a column split requires a 1-D or 2-D parameter, got " + shape.to_string());

    if (split.end > static_cast<size_t>(fused.cols))
        throw std::runtime_error(
            "Error while loading Tensor '" + module_path(path) +
            "': the column split [" + std::to_string(split.start) + ", " + std::to_string(split.end) +
            ") of '" + fused_name + "' is out of bounds (the fused tensor has " +
            std::to_string(fused.cols) + " columns)");

    auto expected = shape.rank() == 2
        ? Tensor::Shape({fused.rows, static_cast<int64_t>(split.end - split.start)})
        : Tensor::Shape({static_cast<int64_t>(split.end - split.start)});

    if (shape != expected)
        throw std::runtime_error(
            "Error while loading Tensor '" + module_path(path) +
            "': Parameter shape mismatch: expected " + expected.to_string() +
            ", got " + shape.to_string());

    return { fused.type, name, read_cols(file_, fused, split.start, split.end, name) };
}

GGUFLoaderVisitor::TensorInfo GGUFLoaderVisitor::load_fused(const std::string& name) {
    if (auto it = fused_.find(name); it != fused_.end())
        return it->second;

    auto info = load_tensor(name);
    auto result = fused_.emplace(name, info);

    return result.first->second;
}