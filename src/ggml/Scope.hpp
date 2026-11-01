#pragma once

#include <stdexcept>
#include <memory>

class Context;
class Runtime;

class Scope {
public:

    explicit Scope(Context& context, Runtime& runtime)
        : frame_(std::make_shared<Frame>(
              current_context_,
              current_runtime_))
    {
        current_context_ = &context;
        current_runtime_ = &runtime;
    }

    // Set Context in the scope, but does not overwrite Runtime.
    Scope(Context& context);

    // Set Runtime in the scope, but does not change Context.
    explicit Scope(Runtime& runtime)
        : frame_(std::make_shared<Frame>(
              current_context_,
              current_runtime_))
    {
        current_runtime_ = &runtime;
    }

    // Forks share the same scope frame.
    Scope(const Scope&) = default;

    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

    ~Scope() {
        if (frame_.unique()) {
            current_context_ = frame_->previous_context;
            current_runtime_ = frame_->previous_engine;
        }
    }

    static bool active() {
        return current_runtime_ != nullptr;
    }

    static Context& context() {
        if (!current_context_)
            throw std::runtime_error("context(): No active Scope");

        return *current_context_;
    }

    static Runtime& runtime() {
        if (!current_runtime_)
            throw std::runtime_error("engine(): No active Scope");

        return *current_runtime_;
    }

private:
    struct Frame {
        Context* previous_context;
        Runtime* previous_engine;

        Frame(Context* previous_context, Runtime* previous_engine)
            : previous_context(previous_context)
            , previous_engine(previous_engine)
        {}
    };

    std::shared_ptr<Frame> frame_;

    inline static thread_local Context* current_context_ = nullptr;
    inline static thread_local Runtime* current_runtime_ = nullptr;
};
