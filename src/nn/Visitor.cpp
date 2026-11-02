#include "nn/Visitor.hpp"
#include "nn/PermutedParameter.hpp"

void Visitor::visit(PermutedParameter& parameter, std::vector<std::string> path) {
    // At runtime an MROPE parameter is a plain (pre-permuted) weight:
    // the load-time permutation is applied by the GGUF loader. Visitors
    // that do not override visit(PermutedParameter&) bind it like a plain
    // parameter.
    visit(static_cast<Parameter&>(parameter), std::move(path));
}
