#include "ggml/Scope.hpp"
#include "ggml/ExecutionRuntime.hpp"

Scope::Scope(Context& context)
    : frame_(std::make_shared<Frame>(
            current_context_,
            current_runtime_))
{
    current_context_ = &context;

    // Set Runtime to default if it is not set.
    if (!current_runtime_)
        current_runtime_ = &ExecutionRuntime::Default;
}

bool Scope::sharded() {
    if (!current_runtime_)
        return false;

    return current_runtime_->sharded();
}
