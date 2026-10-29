#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Context.hpp"
#include "nn/Parameter.hpp"
#include "nn/ModulePath.hpp"
#include "diffusers/models/transformers/flux2/Flux2FusedQKVProjection.hpp"
#include "diffusers/models/transformers/flux2/Flux2FusedAttentionOutput.hpp"
#include "diffusers/models/transformers/flux2/Flux2FusedLinearIn.hpp"
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

void GGUFLoaderVisitor::visit(Parameter& parameter, std::vector<std::string> path) {
    ModulePath module_path;
    auto model_path = module_path(path);
    auto it = lookup_.find(model_path);

    if (it == std::end(lookup_))
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Tensor not found");

    auto tensor_name = it->first;
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

    auto name = gguf_get_tensor_name(gguf_ctx_, tensor_id);

    const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    Tensor::Shape expected_shape(n_dims);

    for (auto r = 0; r < expected_shape.rank(); ++r)
        expected_shape[r] = ne[expected_shape.rank() - 1 - r];

    if (parameter.shape() != expected_shape)
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter shape mismatch: expected " + parameter.shape().to_string() + ", got " + expected_shape.to_string());

    Scope scope(context_);
    auto tensor = Tensor::empty(expected_shape, type);

    tensor.name(name);

    auto n_bytes = ggml_nbytes(*tensor);

    context_.bind<std::byte>(tensor,
        [n_bytes, expected_shape, offs, file = file_, model_path = std::move(model_path)](std::mt19937&) {
            std::vector<std::byte> buf(n_bytes);

            // std::cerr << "BIND " << tensor_name.c_str() << " " << expected_shape.to_string() << " " << ggml_type_name(type) << std::endl;

            file->seekg(offs, file->beg);
            
            if (!*file)
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': seek failed");

            // Read the tensor data
            file->read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

            if (file->gcount() != static_cast<std::streamsize>(buf.size()))
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': read failed");

            return buf;
        }, /*once=*/true);

    // std::cerr << "LOAD " << tensor_name.c_str() << " " << expected_shape.to_string() << " " << ggml_type_name(type) << std::endl;
    parameter.set(tensor);
}

void GGUFLoaderVisitor::visit(Flux2FusedQKVProjection& to_qkv_mlp_proj, std::vector<std::string> path) {
    ModulePath module_path;
    auto weight_path = module_path(path, {}, {"weight"});

    auto weight = load_tensor(weight_path);

    auto q_weight = to_qkv_mlp_proj.q()->weight();
    auto k_weight = to_qkv_mlp_proj.k()->weight();
    auto v_weight = to_qkv_mlp_proj.v()->weight();
    auto mlp_in_gate_weight = to_qkv_mlp_proj.mlp_in_gate()->weight();
    auto mlp_in_up_weight = to_qkv_mlp_proj.mlp_in_up()->weight();

    auto inner = to_qkv_mlp_proj.inner_dim();
    auto mlp_out = to_qkv_mlp_proj.mlp_out_dim();

    // The MLP input projection is split into gate and up, each with
    // mlp_out / 2 output features: the first half of the fused MLP rows
    // is the gate, the second half the up projection (the reference
    // chunks the fused output down the middle in Flux2SwiGLU).
    auto mlp_hidden = mlp_in_gate_weight->shape()[0];

    // The fused weight is (3 * inner + mlp_out) x query_dim, so rows
    // of the flattened data are query_dim elements long.
    auto query_dim = q_weight->shape()[1];

    if (weight.rows != 3 * inner + mlp_out || weight.cols != query_dim)
        throw std::runtime_error(
            "Error while loading Tensor '" + weight_path +
            "': Parameter shape mismatch: expected " +
            Tensor::Shape({3 * inner + mlp_out, query_dim}).to_string() +
            ", got " + Tensor::Shape({weight.rows, weight.cols}).to_string());

    q_weight->set(create_weight(context_, weight_path + "-q-weight", q_weight->shape(), weight.type,
        read_rows(file_, weight, 0, inner, weight_path + "-q-weight")));

    k_weight->set(create_weight(context_, weight_path + "-k-weight", k_weight->shape(), weight.type,
        read_rows(file_, weight, inner, 2 * inner, weight_path + "-k-weight")));

    v_weight->set(create_weight(context_, weight_path + "-v-weight", v_weight->shape(), weight.type,
        read_rows(file_, weight, 2 * inner, 3 * inner, weight_path + "-v-weight")));

    mlp_in_gate_weight->set(create_weight(context_, weight_path + "-mlp_in_gate-weight", mlp_in_gate_weight->shape(), weight.type,
        read_rows(file_, weight, 3 * inner, 3 * inner + mlp_hidden, weight_path + "-mlp_in_gate-weight")));

    mlp_in_up_weight->set(create_weight(context_, weight_path + "-mlp_in_up-weight", mlp_in_up_weight->shape(), weight.type,
        read_rows(file_, weight, 3 * inner + mlp_hidden, 3 * inner + mlp_out, weight_path + "-mlp_in_up-weight")));
}

void GGUFLoaderVisitor::visit(Flux2FusedAttentionOutput& to_out, std::vector<std::string> path) {
    ModulePath module_path;

    auto weight_path = module_path(path, {}, {"weight"});

    auto weight = load_tensor(weight_path);

    auto attn_weight = to_out.attn()->weight();
    auto mlp_weight = to_out.mlp()->weight();

    auto inner = to_out.inner_dim();
    auto mlp_hidden = to_out.mlp_hidden_dim();

    // The fused weight is (out_dim) x (inner + mlp_hidden) in the
    // PyTorch (out_features, in_features) layout.
    auto out_dim = attn_weight->shape()[0];

    if (weight.rows != out_dim || weight.cols != inner + mlp_hidden)
        throw std::runtime_error(
            "Error while loading Tensor '" + weight_path +
            "': Parameter shape mismatch: expected " +
            Tensor::Shape({out_dim, inner + mlp_hidden}).to_string() +
            ", got " + Tensor::Shape({weight.rows, weight.cols}).to_string());

    attn_weight->set(create_weight(context_, weight_path + "-attn-weight", attn_weight->shape(), weight.type,
        read_cols(file_, weight, 0, inner, weight_path + "-attn-weight")));

    mlp_weight->set(create_weight(context_, weight_path + "-mlp-weight", mlp_weight->shape(), weight.type,
        read_cols(file_, weight, inner, inner + mlp_hidden, weight_path + "-mlp-weight")));

    auto attn_bias = to_out.attn()->bias();

    if (attn_bias) {
        auto bias_path = path;
        bias_path.push_back("bias");
        visit(*attn_bias, std::move(bias_path));
    }
}

void GGUFLoaderVisitor::visit(Flux2FusedLinearIn& linear_in, std::vector<std::string> path) {
    ModulePath module_path;
    auto weight_path = module_path(path, {}, {"weight"});

    auto weight = load_tensor(weight_path);

    auto gate_weight = linear_in.gate()->weight();
    auto up_weight = linear_in.up()->weight();

    // The fused linear_in weight is (inner * 2) x dim (PyTorch layout).
    // The first inner rows are the gate projection, the last inner rows
    // the up projection (the reference chunks the fused output down the
    // middle in Flux2SwiGLU).
    auto inner = linear_in.inner_dim();
    auto dim = linear_in.dim();

    if (weight.rows != inner * 2 || weight.cols != dim)
        throw std::runtime_error(
            "Error while loading Tensor '" + weight_path +
            "': Parameter shape mismatch: expected " +
            Tensor::Shape({inner * 2, dim}).to_string() +
            ", got " + Tensor::Shape({weight.rows, weight.cols}).to_string());

    gate_weight->set(create_weight(context_, weight_path + "-gate-weight", gate_weight->shape(), weight.type,
        read_rows(file_, weight, 0, inner, weight_path + "-gate-weight")));

    up_weight->set(create_weight(context_, weight_path + "-up-weight", up_weight->shape(), weight.type,
        read_rows(file_, weight, inner, inner * 2, weight_path + "-up-weight")));

    auto gate_bias = linear_in.gate()->bias();

    if (gate_bias) {
        auto bias_path = module_path(path, {}, {"bias"});
        auto bias = load_tensor(bias_path);

        auto up_bias = linear_in.up()->bias();

        // The fused bias is 1-D of length inner * 2; the first inner
        // elements belong to the gate, the last inner to the up.
        gate_bias->set(create_weight(context_, bias_path + "-gate-bias", gate_bias->shape(), bias.type,
            read_cols(file_, bias, 0, inner, bias_path + "-gate-bias")));

        up_bias->set(create_weight(context_, bias_path + "-up-bias", up_bias->shape(), bias.type,
            read_cols(file_, bias, inner, inner * 2, bias_path + "-up-bias")));
    }
}