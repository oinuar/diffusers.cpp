#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <sstream>

#include "nn/Visitor.hpp"

class Parameter;
class Module;
class Flux2FusedLinearIn;

class RethrowVisitor : public Visitor {
public:
    explicit RethrowVisitor(Visitor& parent) : parent_(parent) {}

    void visit(Parameter& parameter, std::vector<std::string> path) override {
        visitWithCatch([&] {
            parent_.visit(parameter, std::move(path));
        });
    }

    void visit(Module& module, std::vector<std::string> path) override {
        visitWithCatch([&] {
            parent_.visit(module, std::move(path));
        });
    }

    void visit(Flux2FusedQKVProjection& module, std::vector<std::string> path) override {
        visitWithCatch([&] {
            parent_.visit(module, std::move(path));
        });
    }

    void visit(Flux2FusedAttentionOutput& module, std::vector<std::string> path) override {
        visitWithCatch([&] {
            parent_.visit(module, std::move(path));
        });
    }

    void visit(Flux2FusedLinearIn& module, std::vector<std::string> path) override {
        visitWithCatch([&] {
            parent_.visit(module, std::move(path));
        });
    }

    void rethrow() const {
        if (errors_.empty())
            return;

        std::ostringstream message;
        message << "Visitor failed with " << errors_.size() << " error(s):\n";

        for (auto& error : errors_)
            message << " - " << error << std::endl;

        throw std::runtime_error(message.str());
    }

private:
    template <typename F>
    void visitWithCatch(F&& visitorCall) {
        try {
            std::forward<F>(visitorCall)();
        } catch (const std::exception& e) {
            errors_.push_back(e.what());
        }
    }

    Visitor& parent_;
    std::vector<std::string> errors_;
};
