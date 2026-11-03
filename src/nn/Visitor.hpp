#pragma once

#include <vector>
#include <string>
#include <cstddef>

class Parameter;
class Module;

class Visitor {
public:
    virtual ~Visitor() = default;

    virtual void visit(Parameter&, std::vector<std::string>) {}
    virtual void visit(Module&, std::vector<std::string>) {}
};
