#pragma once

#include <vector>
#include <string>
#include <cstddef>

class Parameter;
class Module;
class Flux2FusedQKVProjection;
class Flux2FusedAttentionOutput;
class Flux2FusedLinearIn;

class Visitor {
public:
    virtual ~Visitor() = default;

    virtual void visit(Parameter&, std::vector<std::string>) {}
    virtual void visit(Module&, std::vector<std::string>) {}
    virtual void visit(Flux2FusedQKVProjection&, std::vector<std::string>) {}
    virtual void visit(Flux2FusedAttentionOutput&, std::vector<std::string>) {}
    virtual void visit(Flux2FusedLinearIn&, std::vector<std::string>) {}
};
