#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <cctype>
#include "./ArgumentParser.hpp"
#include "ggml/Tensor.hpp"
#include "ggml/Context.hpp"

template <>
struct ArgumentParser::parser<Tensor> {
    parser(Scope scope, const ggml_type& dtype = Tensor::DType<float>::value)
        : scope_(scope), dtype_(dtype)
    {
    }

    Tensor operator ()(const std::string& option, const std::string& value) const {
        TensorParser parser(option, value, dtype_);

        try {
            auto [shape, data] = parser.parse();

            auto tensor = create(scope_.context(), shape, std::move(data), dtype_);

            tensor.name(option.c_str());

            return tensor;
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("invalid argument " + option + ": " + error.what());
        }
    }

    /** @brief Creates a tensor of the given dtype from the parsed
     *  (float) values, the way the visitors bind parameter values.
     */
    static Tensor create(Context& context, const Tensor::Shape& shape, std::vector<float> data, const ggml_type& dtype) {
        Tensor tensor;

        switch (dtype) {
        case Tensor::DType<int64_t>::value:
            tensor = context.create<int64_t>(shape, [data = std::move(data)](std::mt19937&) {
                return std::vector<int64_t>(data.begin(), data.end());
            });
            break;

        case Tensor::DType<int32_t>::value:
            tensor = context.create<int32_t>(shape, [data = std::move(data)](std::mt19937&) {
                return std::vector<int32_t>(data.begin(), data.end());
            });
            break;

        case Tensor::DType<int16_t>::value:
            tensor = context.create<int16_t>(shape, [data = std::move(data)](std::mt19937&) {
                return std::vector<int16_t>(data.begin(), data.end());
            });
            break;

        case Tensor::DType<int8_t>::value:
            tensor = context.create<int8_t>(shape, [data = std::move(data)](std::mt19937&) {
                return std::vector<int8_t>(data.begin(), data.end());
            });
            break;

        default:
            tensor = context.create<float>(shape, [data = std::move(data)](std::mt19937&) {
                return data;
            });
            break;
        }

        return tensor;
    }

private:
    Scope scope_;
    ggml_type dtype_;

public:
    class TensorParser {
    public:
        explicit TensorParser(std::string_view option, std::string_view s, const ggml_type& dtype = GGML_TYPE_F32)
            : option_(option), s_(s), i_(0), dtype_(dtype) {}

        std::pair<Tensor::Shape, std::vector<float>> parse() {
            skip_ws();

            Node result;

            if (peek() == '[')
                result = parse_array();
            else
                result = parse_number();

            skip_ws();

            if (i_ != s_.size())
                throw std::runtime_error("Unexpected trailing characters in tensor literal");

            if (result.scalar) {
                result.shape = Tensor::Shape{};   // rank-0 tensor
                result.scalar = false;
            }

            //std::cerr << option_ << ": shape = " << result.shape.to_string() << ", data size = " << result.values.size() << ", dtype = " << ggml_type_name(dtype_) << std::endl;

            return {result.shape, std::move(result.values)};
        }

    private:
        struct Node {
            std::vector<float> values;
            Tensor::Shape shape;
            bool scalar = false; // true only for parse_number()
        };

        std::string_view option_;
        std::string_view s_;
        size_t i_;
        ggml_type dtype_;

        Node parse_array() {
            expect('[');

            std::vector<Node> elements;

            skip_ws();

            if (peek() == ']') {
                consume();

                Node out;
                out.shape = Tensor::Shape{0};
                return out;
            }

            while (true) {
                skip_ws();

                if (peek() == '[')
                    elements.push_back(parse_array());
                else
                    elements.push_back(parse_number());

                skip_ws();

                if (peek() == ',') {
                    consume();
                    continue;
                }

                if (peek() == ']') {
                    consume();
                    break;
                }

                throw std::runtime_error("Expected ',' or ']'");
            }

            return flatten(elements);
        }

        Node parse_number() {
            skip_ws();

            const char* begin = s_.data() + i_;
            char* end = nullptr;

            float value = std::strtof(begin, &end);

            if (begin == end)
                throw std::runtime_error("Invalid number in tensor literal");

            i_ += end - begin;

            Node out;
            out.values.push_back(value);
            out.shape = Tensor::Shape{1};
            out.scalar = true;

            return out;
        }

        Node flatten(const std::vector<Node>& nodes) {
            Node out;

            if (nodes.empty())
                return out;

            // Array of scalars: [1, 2, 3]
            if (nodes[0].scalar) {
                for (const auto& n : nodes) {
                    if (!n.scalar)
                        throw std::runtime_error("Mixed scalars and arrays are not supported");

                    out.values.push_back(n.values[0]);
                }

                out.shape = Tensor::Shape{static_cast<int64_t>(nodes.size())};
                out.scalar = false;
                return out;
            }

            // Array of arrays: [[...], [...], ...]
            const Tensor::Shape& firstShape = nodes[0].shape;

            for (const auto& n : nodes) {
                if (n.scalar)
                    throw std::runtime_error("Mixed scalars and arrays are not supported");

                if (n.shape.rank() != firstShape.rank())
                    throw std::runtime_error("Jagged tensor not supported");

                for (int i = 0; i < firstShape.rank(); ++i) {
                    if (n.shape[i] != firstShape[i])
                        throw std::runtime_error("Jagged tensor not supported");
                }

                out.values.insert(
                    out.values.end(),
                    n.values.begin(),
                    n.values.end());
            }

            std::vector<int64_t> dims;
            dims.push_back(static_cast<int64_t>(nodes.size()));

            for (int i = 0; i < firstShape.rank(); ++i)
                dims.push_back(firstShape[i]);

            Tensor::Shape shape(dims.size());

            for (size_t i = 0; i < dims.size(); ++i)
                shape[i] = dims[i];

            out.shape = std::move(shape);
            out.scalar = false;

            return out;
        }

        char peek() const {
            if (i_ >= s_.size())
                return '\0';

            return s_[i_];
        }

        char consume() {
            return s_[i_++];
        }

        void expect(char c) {
            if (peek() != c)
                throw std::runtime_error(std::string("Expected '") + c + "'");

            consume();
        }

        void skip_ws() {
            while (i_ < s_.size() &&
                std::isspace(static_cast<unsigned char>(s_[i_])))
                ++i_;
        }
    };
};
