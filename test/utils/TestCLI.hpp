#pragma once

#include "ggml/Context.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Backend.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/ShardingRuntime.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include "nn/Visitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/ModulePath.hpp"
#include "ProgressBar.hpp"
#include "./QuantSpec.hpp"
#include "./ArgumentParser.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cstring>
#include <iterator>

class TestCLI {
public:
    int main() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

        ggml_backend_load_all();

        auto n_devices = args_.get_optional<size_t>("--runner-n_devices").value_or(1);
        auto use_gpu = args_.get_optional<bool>("--runner-use_gpu").value_or(false);

        Context weights_context(get_graph_size());

        // If more than one device, use Meta device.
        if (n_devices > 1) {
            auto tensor_splits = args_.get_many<float>("--runner-tensor_split");

            if (use_gpu)
                throw std::runtime_error("Multi-GPU tests are not supported");

            Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
            std::vector<ggml_backend_dev_t> devices;

            for (auto i = 0; i < n_devices; ++i)
                devices.push_back(*cpu);

            MetaDevice meta(std::move(devices));
            Backend backend(meta);

            // Sharding is only relevant for the Meta device: the other modes
            // of execution run the whole graph on one device and skip the
            // sharding entirely.
            ShardingRuntime runtime(ExecutionRuntime::Default, meta);

            // Activate sharding runtime for all subsequent scopes.
            Scope scope(runtime);

            // Set tensor split coefficients.
            meta.tensor_split(tensor_splits);

            return run({{&weights_context, &backend}}, compute(weights_context), &runtime);
        }

        // Use single GPU device.
        if (use_gpu) {
            Device gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
            Backend backend(gpu);

            return run({{&weights_context, &backend}}, compute(weights_context));
        }

        Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
        Backend backend(cpu);


        return run({{&weights_context, &backend}}, compute(weights_context));
    }

    virtual Computation<std::vector<Tensor>> compute(Context& weights_context) = 0;

    virtual size_t get_graph_size() const {
        return GGML_DEFAULT_GRAPH_SIZE;
    }

    const ArgumentParser& args() const {
        return args_;
    }

    template <typename T>
    void print_tensor_like(const std::vector<T>& data, const Tensor::Shape& shape, std::ostream& stream=std::cout) const {
        size_t expected = 1;
        for (auto i = 0; i < shape.rank(); ++i)
            expected *= shape[i];

        if (expected != data.size())
            throw std::runtime_error("tensor data size does not match shape, expected " + std::to_string(expected) + ", but got " + std::to_string(data.size()));

        std::cerr << "output shape: " << shape.to_string() << std::endl;

        size_t index = 0;
        print_tensor_recursively(data, shape, 0, index, stream);
        stream << std::endl;
    }

protected:
    ArgumentParser args_;

    TestCLI(int argc, char** argv) : args_(argc, argv) {}

public:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Context& context, const ArgumentParser& args, const std::string& prefix = "")
            : context_(context), args_(args), prefix_(prefix)
        {}
        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            ModulePath module_path("-", "--param");

            // A split parameter (see Parameter::Split): its values live
            // in a fused tensor, not at its own path -- load the fused
            // tensor and take the configured slice.
            if (auto split = parameter.split()) {
                visit_split(parameter, path, *split, module_path);
                return;
            }

            auto joined_path = module_path(path, prefix_);

            // Quantized parameter (see --quant): the value is a file with
            // the raw bytes of the type resolved by the quantization spec,
            // like a tensor in a GGUF file from utils/convert-model.
            if (auto type = quantization().resolve(dot_name(path), prefix_); type && *type != parameter.dtype()) {
                auto bytes = get_param_bytes(joined_path);

                // A permuted parameter (see Parameter::Permutation): the
                // raw bytes are reordered at load time, mirroring the
                // GGUF loader.
                if (auto permutation = parameter.permutation())
                    bytes = permute_bytes(bytes, *type, parameter.shape(), *permutation, joined_path);

                auto tensor = create_quantized(joined_path, parameter.shape(), *type, std::move(bytes));

                parameter.set(tensor.name(joined_path));
                return;
            }

            // Plain parameter: a tensor literal (inline or in the file the
            // --param option points to), parsed as floats and bound as the
            // parameter's dtype.
            auto value = get_param(joined_path);

            ArgumentParser::parser<Tensor>::TensorParser parser(joined_path, value, parameter.dtype());

            Tensor::Shape shape;
            std::vector<float> data;

            try {
                std::tie(shape, data) = parser.parse();
            } catch (const std::runtime_error& error) {
                throw std::runtime_error("invalid argument " + joined_path + ": " + error.what());
            }

            // A permuted parameter (see Parameter::Permutation): the
            // values are reordered at load time, mirroring the GGUF
            // loader.
            if (auto permutation = parameter.permutation())
                data = permute_values(data, shape, *permutation, joined_path);

            auto tensor = ArgumentParser::parser<Tensor>::create(context_, shape, std::move(data), parameter.dtype());

            parameter.set(tensor.name(joined_path));
        }

        std::string get_param(const std::string& path) {
            auto value = args_.get_one<std::string>(path);

            // Read tensor value from file
            std::error_code ec;
            if (std::filesystem::is_regular_file(value, ec)) {
                std::ifstream file(value);
                if (!file)
                    throw std::runtime_error(
                        "Failed to open parameter file: " + value);

                std::stringstream buffer;
                buffer << file.rdbuf();

                return buffer.str();
            }

            // Otherwise, read inline tensor
            return value;
        }

        /** @brief The quantization spec parsed from the --quant argument
         *  (same grammar as the --type option of utils/convert-model).
         *  Parsed lazily, once per visitor.
         */
        const QuantSpec& quantization() {
            if (!quant_spec_)
                quant_spec_ = QuantSpec::parse(args_.get_optional<std::string>("--quant").value_or(""));

            return *quant_spec_;
        }

        /** @brief Reads the raw bytes of a quantized parameter from the
         *  file the --param option points to.
         */
        std::vector<std::byte> get_param_bytes(const std::string& path) const {
            auto value = args_.get_one<std::string>(path);

            std::error_code ec;

            if (!std::filesystem::is_regular_file(value, ec) || ec)
                throw std::runtime_error(
                    "Quantized parameter '" + path + "' must be a file with raw bytes, got: " + value);

            std::ifstream file(value, std::ios::binary);

            if (!file)
                throw std::runtime_error(
                    "Failed to open quantized parameter file: " + value);

            std::string raw{
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>()};

            std::vector<std::byte> bytes(raw.size());
            std::memcpy(bytes.data(), raw.data(), raw.size());

            return bytes;
        }

        /** @brief Creates a tensor of the given (quantized) type and binds
         *  the raw bytes to it. Mirrors the way GGUFLoaderVisitor streams
         *  quantized tensors from a GGUF file.
         */
        Tensor create_quantized(const std::string& name, const Tensor::Shape& shape, ggml_type type, std::vector<std::byte>&& bytes) const {
            Scope scope(context_);
            auto tensor = Tensor::empty(shape, type).input();
            tensor.name(name);

            auto expected = ggml_nbytes(*tensor);

            if (bytes.size() != expected)
                throw std::runtime_error(
                    "Quantized parameter '" + name + "': expected " +
                    std::to_string(expected) + " bytes of " + ggml_type_name(type) +
                    " with shape " + shape.to_string() + ", got " +
                    std::to_string(bytes.size()));

            context_.bind<std::byte>(tensor, [bytes = std::move(bytes)](std::mt19937&) {
                return std::move(bytes);
            }, /*once=*/true);

            return tensor;
        }

        // The tensor name inside its component (dotted, without the
        // component prefix), the name quantization patterns match against.
        static std::string dot_name(const std::vector<std::string>& path) {
            ModulePath module_path;
            return module_path(path);
        }

        Context& context() {
            return context_;
        }

        const std::string& prefix() const {
            return prefix_;
        }

    private:
        // Loads a split parameter (see Parameter::Split) from the fused
        // tensor the split configuration points to, mirroring the way
        // GGUFLoaderVisitor streams the slice from a GGUF file.
        void visit_split(Parameter& parameter, const std::vector<std::string>& path, const Parameter::Split& split, ModulePath module_path) {
            auto fused_segments = parameter.split_fused_path(path);
            auto fused_path = module_path(fused_segments, prefix_);

            // The slice is named after its fused tensor and the parameter:
            // "...to_qkv_mlp_proj.weight-q-weight".
            auto slice_path = fused_path + "-" + path[path.size() - 2] + "-" + path[path.size() - 1];

            auto shape = parameter.shape();

            // Quantized fused tensor (see --quant): the file holds the
            // raw bytes of the fused tensor, like a tensor in a GGUF
            // file from utils/convert-model.
            if (auto type = quantization().resolve(dot_name(fused_segments), prefix_); type && *type != parameter.dtype()) {
                auto bytes = get_param_bytes(fused_path);

                std::vector<std::byte> sliced;

                if (split.axis == Parameter::Split::Axis::Rows) {
                    // Rows are contiguous in memory, so the split is a
                    // plain byte-range slice. The row length is the
                    // parameter's last dimension.
                    auto cols = shape[shape.rank() - 1];
                    auto row_bytes = static_cast<size_t>(ggml_row_size(*type, cols));

                    if (bytes.size() % row_bytes != 0 || split.end * row_bytes > bytes.size())
                        throw std::runtime_error(
                            "Split parameter '" + slice_path + "': the row split [" +
                            std::to_string(split.start) + ", " + std::to_string(split.end) +
                            ") of '" + fused_path + "' is out of bounds");

                    sliced = slice_rows_bytes(bytes, *type, cols, split.start, split.end);
                }
                else {
                    // The column range must fall on quantization block
                    // boundaries, so the slice is a per-row byte range.
                    // The row count is the parameter's first dimension
                    // (1 for a 1-D parameter); the full row length
                    // follows from the fused tensor's element count.
                    auto rows = shape.rank() > 1 ? shape[0] : 1;

                    auto blck = static_cast<size_t>(ggml_blck_size(*type));
                    auto block_bytes = static_cast<size_t>(ggml_type_size(*type));

                    if (bytes.size() % block_bytes != 0)
                        throw std::runtime_error(
                            "Split parameter '" + slice_path + "': the size of '" + fused_path +
                            "' is not a multiple of the quantization block size");

                    auto elements = bytes.size() / block_bytes * blck;

                    if (elements % rows != 0)
                        throw std::runtime_error(
                            "Split parameter '" + slice_path + "': the element count of '" +
                            fused_path + "' is not divisible by the row count");

                    auto cols = elements / rows;

                    if (split.end > cols)
                        throw std::runtime_error(
                            "Split parameter '" + slice_path + "': the column split [" +
                            std::to_string(split.start) + ", " + std::to_string(split.end) +
                            ") of '" + fused_path + "' is out of bounds");

                    sliced = slice_cols_bytes(bytes, *type, rows, cols, split.start, split.end);
                }

                // A permuted parameter (see Parameter::Permutation): the
                // raw bytes are reordered at load time, mirroring the
                // GGUF loader; the permutation applies to the slice.
                if (auto permutation = parameter.permutation())
                    sliced = permute_bytes(sliced, *type, shape, *permutation, slice_path);

                parameter.set(create_quantized(slice_path, shape, *type, std::move(sliced)));
                return;
            }

            // Plain (F32) fused tensor.
            auto fused_value = get_param(fused_path);
            ArgumentParser::parser<Tensor>::TensorParser parser(fused_path, fused_value);

            auto [fused_shape, data] = parser.parse();

            auto cols = fused_shape[fused_shape.rank() - 1];
            auto rows = 1;

            for (auto r = 0; r + 1 < fused_shape.rank(); ++r)
                rows *= fused_shape[r];

            std::vector<float> sliced;

            if (split.axis == Parameter::Split::Axis::Rows) {
                if (fused_shape.rank() != 2)
                    throw std::runtime_error(
                        "Split parameter '" + slice_path + "': a row split requires a 2-D fused tensor, got " + fused_shape.to_string());

                if (split.end > rows)
                    throw std::runtime_error(
                        "Split parameter '" + slice_path + "': the row split [" +
                        std::to_string(split.start) + ", " + std::to_string(split.end) +
                        ") of '" + fused_path + "' is out of bounds");

                sliced = slice_rows(data, cols, split.start, split.end);
            }
            else {
                if (fused_shape.rank() > 2)
                    throw std::runtime_error(
                        "Split parameter '" + slice_path + "': a column split requires a 1-D or 2-D fused tensor, got " + fused_shape.to_string());

                if (split.end > cols)
                    throw std::runtime_error(
                        "Split parameter '" + slice_path + "': the column split [" +
                        std::to_string(split.start) + ", " + std::to_string(split.end) +
                        ") of '" + fused_path + "' is out of bounds");

                sliced = slice_cols(data, rows, cols, split.start, split.end);
            }

            // A permuted parameter (see Parameter::Permutation): the
            // values are reordered at load time, mirroring the GGUF
            // loader; the permutation applies to the slice.
            if (auto permutation = parameter.permutation())
                sliced = permute_values(sliced, shape, *permutation, slice_path);

            auto elements = 1;

            for (auto r = 0; r < shape.rank(); ++r)
                elements *= shape[r];

            if (sliced.size() != elements)
                throw std::runtime_error(
                    "Split parameter '" + slice_path + "': the slice of '" + fused_path + "' has " +
                    std::to_string(sliced.size()) + " elements, expected " + std::to_string(elements));

            parameter.set(context().create<float>(shape, [sliced = std::move(sliced)](std::mt19937&) {
                return sliced;
            }).name(slice_path));
        }

        // PyTorch: x[start:end]
        template <typename T>
        static std::vector<T> slice_rows(
            const std::vector<T>& x,
            size_t cols,
            size_t start,
            size_t end)
        {
            return {
                x.begin() + start * cols,
                x.begin() + end * cols
            };
        }

        // PyTorch: x[:, start:end]
        template <typename T>
        static std::vector<T> slice_cols(
            const std::vector<T>& x,
            size_t rows,
            size_t cols,
            size_t start,
            size_t end)
        {
            std::vector<T> out;
            out.reserve(rows * (end - start));

            for (size_t r = 0; r < rows; ++r) {
                out.insert(
                    out.end(),
                    x.begin() + r * cols + start,
                    x.begin() + r * cols + end);
            }

            return out;
        }

        // The raw-bytes counterpart of slice_rows: rows of a 2-D quantized
        // tensor are contiguous, each row is ggml_row_size bytes long.
        static std::vector<std::byte> slice_rows_bytes(
            const std::vector<std::byte>& x,
            ggml_type type,
            size_t cols,
            size_t start,
            size_t end)
        {
            auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));

            return {
                x.begin() + start * row_bytes,
                x.begin() + end * row_bytes
            };
        }

        // The raw-bytes counterpart of slice_cols: the column range must
        // fall on quantization block boundaries, so the slice is a
        // per-row byte range.
        static std::vector<std::byte> slice_cols_bytes(
            const std::vector<std::byte>& x,
            ggml_type type,
            size_t rows,
            size_t cols,
            size_t start,
            size_t end)
        {
            auto blck = static_cast<size_t>(ggml_blck_size(type));

            if (start % blck != 0 || (end - start) % blck != 0)
                throw std::runtime_error(
                    "The quantized fused weight column range [" + std::to_string(start) + ", " +
                    std::to_string(end) + ") is not aligned to the quantization block size " +
                    std::to_string(blck));

            auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));
            auto block_bytes = static_cast<size_t>(ggml_type_size(type));
            auto slice_offset = start / blck * block_bytes;
            auto slice_bytes = (end - start) / blck * block_bytes;

            std::vector<std::byte> out;
            out.reserve(rows * slice_bytes);

            for (size_t r = 0; r < rows; ++r)
                out.insert(
                    out.end(),
                    x.begin() + r * row_bytes + slice_offset,
                    x.begin() + r * row_bytes + slice_offset + slice_bytes);

            return out;
        }

        // Applies a permutation configuration (see Parameter::Permutation)
        // to the parsed (float) values, mirroring the GGUF loader's
        // byte-level permutation: rows are reordered as whole rows,
        // elements are reordered within each row.
        static std::vector<float> permute_values(
            const std::vector<float>& x,
            const Tensor::Shape& shape,
            const Parameter::Permutation& permutation,
            const std::string& name)
        {
            auto cols = shape[shape.rank() - 1];
            auto rows = 1;

            for (auto r = 0; r + 1 < shape.rank(); ++r)
                rows *= shape[r];

            if (permutation.perm.size() != static_cast<size_t>(permutation.axis == Parameter::Permutation::Axis::Rows ? rows : cols))
                throw std::runtime_error(
                    "Parameter '" + name + "': permutation size mismatch");

            std::vector<float> out(x.size());

            if (permutation.axis == Parameter::Permutation::Axis::Rows) {
                for (size_t r = 0; r < rows; ++r) {
                    if (permutation.perm[r] >= rows)
                        throw std::runtime_error(
                            "Parameter '" + name + "': permutation out of range");

                    std::copy(
                        x.begin() + permutation.perm[r] * cols,
                        x.begin() + (permutation.perm[r] + 1) * cols,
                        out.begin() + r * cols);
                }
            } else {
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t j = 0; j < cols; ++j) {
                        if (permutation.perm[j] >= cols)
                            throw std::runtime_error(
                                "Parameter '" + name + "': permutation out of range");

                        out[r * cols + j] = x[r * cols + permutation.perm[j]];
                    }
                }
            }

            return out;
        }

        // The raw-bytes counterpart of permute_values, for quantized
        // types: rows are reordered as whole rows (exact for every
        // storage type), elements are reordered within each row (exact
        // for F32/F16/BF16 only).
        static std::vector<std::byte> permute_bytes(
            const std::vector<std::byte>& x,
            ggml_type type,
            const Tensor::Shape& shape,
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
                        "Parameter '" + name +
                        "': element permutation is not supported for type " +
                        std::string(ggml_type_name(type)));
                }
            }

            auto cols = shape[shape.rank() - 1];
            auto rows = 1;

            for (auto r = 0; r + 1 < shape.rank(); ++r)
                rows *= shape[r];

            auto row_bytes = static_cast<size_t>(ggml_row_size(type, cols));

            if (permutation.perm.size() != static_cast<size_t>(permutation.axis == Parameter::Permutation::Axis::Rows ? rows : cols))
                throw std::runtime_error(
                    "Parameter '" + name + "': permutation size mismatch");

            std::vector<std::byte> out(x.size());

            if (permutation.axis == Parameter::Permutation::Axis::Rows) {
                for (size_t r = 0; r < rows; ++r) {
                    if (permutation.perm[r] >= rows)
                        throw std::runtime_error(
                            "Parameter '" + name + "': permutation out of range");

                    std::copy(
                        x.begin() + permutation.perm[r] * row_bytes,
                        x.begin() + (permutation.perm[r] + 1) * row_bytes,
                        out.begin() + r * row_bytes);
                }
            } else {
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t j = 0; j < cols; ++j) {
                        if (permutation.perm[j] >= cols)
                            throw std::runtime_error(
                                "Parameter '" + name + "': permutation out of range");

                        std::copy(
                            x.begin() + r * row_bytes + permutation.perm[j] * element_bytes,
                            x.begin() + r * row_bytes + (permutation.perm[j] + 1) * element_bytes,
                            out.begin() + r * row_bytes + j * element_bytes);
                    }
                }
            }

            return out;
        }

        Context& context_;
        const ArgumentParser& args_;
        std::string prefix_;
        mutable std::optional<QuantSpec> quant_spec_;
    };
public:
    virtual int run(const std::vector<std::pair<Context*, Backend*>>& backends, Computation<std::vector<Tensor>> computation, ShardingRuntime* sharding = nullptr) {
        std::mt19937 rng;
        auto results = ExecutionRuntime::Default.run(backends, rng, computation, sharding);

        for (auto& tensor : results) {
            switch (tensor.dtype()) {
            case Tensor::DType<float>::value:
            {
                auto data = ExecutionRuntime::Default.read<float>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int8_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int8_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int16_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int16_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int32_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int32_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            default:
                throw std::runtime_error("Unsupported tensor type: " + std::string(ggml_type_name(tensor.dtype())));
            }
        }

        return EXIT_SUCCESS;
    }

    void print_escaped_string(const std::string& value, std::ostream& stream) const {
        stream << '"';

        for (char c : value) {
            switch (c) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                stream << c;
                break;
            }
        }

        stream << '"';
    }

    template <typename T>
    void print_tensor_recursively(const std::vector<T>& data, const Tensor::Shape& shape, size_t dim, size_t& index, std::ostream& stream) const {
        if (shape.rank() == 0 && dim == 0) {
            stream << data[index++];
            return;
        }

        stream << "[";

        if (dim == shape.rank() - 1)
        {
            // Last dimension: print elements
            for (auto i = 0; i < shape[dim]; ++i)
            {
                if constexpr (std::is_same_v<T, std::string>)
                    print_escaped_string(data[index++], stream);
                else
                    stream << data[index++];

                if (i + 1 != shape[dim])
                    stream << ", ";
            }
        }
        else
        {
            // Print nested arrays
            for (auto i = 0; i < shape[dim]; ++i)
            {
                print_tensor_recursively(data, shape, dim + 1, index, stream);
                if (i + 1 != shape[dim])
                    stream << ", ";
            }
        }

        stream << "]";
    }
};
