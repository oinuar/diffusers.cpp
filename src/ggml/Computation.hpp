#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Runtime.hpp"
#include <deque>
#include <memory>
#include <set>
#include <optional>
#include <type_traits>
#include <utility>
#include <iostream>

// Computation is the build-phase value: a Tensor (a ggml tensor handle in
// one of the computation's contexts) plus a shared Description of the whole
// computation. The build allocates nothing and executes nothing; the result
// is a lazy description that the ExecutionRuntime executes in one pass.
//
// A scope's temporary context is disposable: once the scope has executed,
// its buffer is freed. The values that must survive the scopes are exactly
// the ones the computation dereferences (operator* / operator->): each such
// value is materialized into a persistent cell in the state context, and
// the producing scope's saves copy it there after every execution.

// One value crossing a scope boundary: the value (in the producing
// scope's context) is copied into the persistent destination (a
// state cell). The copy runs as part of the producing scope's
// execution -- as a node of its graph on a non-meta backend, as a
// host-side copy on the meta backend (which cannot copy into a
// static tensor from the graph).

struct ComputationScope {
    struct Repeat {
        size_t count; // number of iteration steps
        size_t iter; // the loop clock: the per-iteration providers read it
    };

    struct Save {
        Tensor src;
        Tensor dst;
        bool is_feedback;
    };

    std::optional<Context> context;     // the graph's scratch (disposable)
    std::vector<Tensor> outputs;        // the graph's outputs (in the temporary context)
    std::optional<Repeat> repeat;       // the graph should repeat
    std::vector<Save> saves; // the values the scope persists into the state
};

// Trait to extract tensors from T and reconstruct T from tensors.
// This allows the monad to process std::vector<Tensor> (or custom structs)
// identically to a single Tensor.
template <typename T>
struct ComputationValue {
    static T unwrap(const T& x) { return x; }
    static T wrap(const T& x) { return x; }
};

// Expansion for std::optional<T>
template <typename T>
struct ComputationValue<std::optional<T>> {
    static std::vector<Tensor> unwrap(const std::optional<T>& x) {
        if (x) return ComputationValue<T>::unwrap(*x);
        return {};
    }

    static std::optional<T> wrap(const std::vector<Tensor>& ts) {
        if (ts.empty()) return std::nullopt;
        return ComputationValue<T>::wrap(ts);
    }

    static std::optional<T> state(const std::optional<T>& x) {
        if (x) return ComputationValue<T>::state(*x);
        return std::nullopt;
    }
};

// Expansion for std::vector<T>
template <typename T>
struct ComputationValue<std::vector<T>> {
    static std::vector<Tensor> unwrap(const std::vector<T>& x) {
        std::vector<Tensor> res;
        for (const auto& elem : x) {
            auto sub = ComputationValue<T>::unwrap(elem);
            res.insert(res.end(), sub.begin(), sub.end());
        }
        return res;
    }

    static std::vector<T> wrap(const std::vector<Tensor>& ts) {
        std::vector<T> res;
        res.reserve(ts.size());
        // Reconstructs the vector by wrapping each individual tensor.
        // This perfectly supports std::vector<Tensor> and any T that maps to a single tensor.
        for (const auto& t : ts) {
            res.push_back(ComputationValue<T>::wrap({t}));
        }
        return res;
    }

    static std::vector<T> state(const std::vector<T>& x) {
        std::vector<T> res;
        res.reserve(x.size());
        for (const auto& elem : x) {
            res.push_back(ComputationValue<T>::state(elem));
        }
        return res;
    }
};

template <>
struct ComputationValue<Tensor> {
    static std::vector<Tensor> unwrap(const Tensor& t) {
        return {t};
    }

    static Tensor wrap(const std::vector<Tensor>& ts) {
        return ts.at(0);
    }

    static Tensor state(const Tensor& t) {
        return Tensor::empty(t.shape(), t.dtype());
    }

    static Tensor& output(Scope scope, Tensor& t) {
        // The meta backend skips a node whose view_src is a static
        // tensor (a GGML_OP_NONE in a host buffer) and asserts that the
        // graph's last node is not such a skip (ggml_backend_meta
        // _graph_compute: i_start == n_nodes). ggml flattens view chains
        // in ggml_set_view_op, so view_src is always the base tensor. A graph
        // that ends in a view of a static tensor has no computable last node,
        // so materialize it with a copy: the dup is a compute node the meta 
        // can place, and it keeps the output in the same (replicated) distribution
        // the plan committed.
        if ((*t)->view_src != nullptr && (*t)->view_src->op == GGML_OP_NONE)
            t = t.clone();

        // Materialize output tensor by making it contiguous if needed.
        else if (!t.is_contiguous())
            t = t.contiguous();
        
        // Mark output to active runtime.
        scope.runtime().set_output(*t);

        return t;
    }
};

struct ComputationDescription {
    std::set<Context*> pinned;        // provided from outside (pinned)
    std::optional<Context> state;     // created on demand
    // A deque, not a vector: push_back must not invalidate the references
    // to scope elements that Computation::scope() / fold() hold across the
    // user body (the body nests more scopes), nor the Context addresses
    // captured in the Scope frames.
    std::deque<ComputationScope> scopes;

    Context& context() {
        if (!state)
            state.emplace(6000 /*TODO: remove temporary fix to get around context exhaustion problem*/);
        return *state;
    }

    ComputationScope& add_scope() {
        scopes.push_back(ComputationScope());
        auto& r = scopes.back();
        r.context.emplace(6000 /*TODO: remove temporary fix to get around context exhaustion problem*/);
        return r;
    }

    // The scope whose scratch context owns the tensor, or nullptr when
    // the tensor lives outside the scope contexts (in the state or a
    // pinned context).
    ComputationScope* find_scope(const Tensor& t) {
        for (auto& scope : scopes) {
            if (!scope.context)
                continue;

            for (auto cur = ggml_get_first_tensor(**scope.context); cur != nullptr; cur = ggml_get_next_tensor(**scope.context, cur))
                if (cur == *t)
                    return &scope;
        }

        return nullptr;
    }
};

template <class T>
class Computation {
public:
    struct VoidRef {};
    typedef std::conditional_t<std::is_void_v<T>, VoidRef, T> Ref;

    template <class Body, bool Void = std::is_void_v<T>>
    struct BindResult;

    template <class Body>
    struct BindResult<Body, true> {
        using type = std::invoke_result_t<Body&>;
    };

    template <class Body>
    struct BindResult<Body, false> {
        using type = std::invoke_result_t<Body&, T&>;
    };

    template <class Body, bool Void = std::is_void_v<T>>
    struct ScopeResult;

    template <class Body>
    struct ScopeResult<Body, true> {
        using type = std::invoke_result_t<Body&, Scope&>;
    };

    template <class Body>
    struct ScopeResult<Body, false> {
        using type = std::invoke_result_t<Body&, Scope&, T&>;
    };

    template <class Body, bool Void = std::is_void_v<T>>
    struct FoldResult;

    template <class Body>
    struct FoldResult<Body, true> {
        using type = std::invoke_result_t<Body&, Scope&, size_t&>;
    };

    template <class Body>
    struct FoldResult<Body, false> {
        using type = std::invoke_result_t<Body&, Scope&, size_t&, T&>;
    };

    template <class... Args>
    static Computation<std::vector<T>> all(const Computation<T>& first, const Args&... rest) {
        std::vector<T> values;
        values.reserve(1 + sizeof...(rest));

        values.push_back(*first);

        (
            [&] {
                if (rest.desc() != first.desc_)
                    throw std::runtime_error(
                        "all(): computations have different descriptions");

                values.push_back(*rest);
            }(),
            ...
        );

        return {
            std::move(values),
            first.desc_
        };
    }

    explicit Computation(std::set<Context*> pinned) : Computation() {
        desc_->pinned = std::move(pinned);
    }

    Computation() : ref_(), desc_() {
        desc_ = std::make_shared<ComputationDescription>();
    }

    Computation(Ref ref, std::shared_ptr<ComputationDescription> desc) : ref_(ref), desc_(desc) {

    }

    // Dereferences the computation value, materializing it into the
    // state context when it still lives in a scope's scratch context:
    // the scratch buffer dies with the scope, so the dereference returns
    // the persistent cell instead (see state_cell). The Description
    // records the materialization, which is the only data the
    // ExecutionRuntime needs to know which values the state context
    // must carry.
    T operator *() const {
        static_assert(!std::is_void_v<T>);

        auto values = ComputationValue<T>::unwrap(ref_);

        for (auto& value : values)
            value = state(value);

        return ComputationValue<T>::wrap(values);
    }

    T* operator ->() {
        static_assert(!std::is_void_v<T>);

        auto values = ComputationValue<T>::unwrap(ref_);

        for (auto& value : values)
            value = state(value);

        ref_ = ComputationValue<T>::wrap(values);

        return &ref_;
    }

    std::shared_ptr<ComputationDescription> desc() const {
        return desc_;
    }

    // Sequences this computation with a function that produces the next 
    // value from the current value. 
    template <class Body>
    auto bind(Body&& body) -> Computation<typename BindResult<Body>::type> {
        using U = typename BindResult<Body>::type;

        U next;

        if constexpr (std::is_void_v<T>) {
            next = body();
        } else {
            next = body(**this);
        }

        return {next, desc_};
    }

    // Creates a single-execution graph.
    // The provided `body` builds a low-level tensor chain within a fresh,
    // scheduler-allocated temporary context. The returned Computation<U>
    // wraps the output(s) of the body. Because the temporary context is
    // disposable after execution, the outputs survive across scopes only
    // through a dereference (operator*), which materializes them into the
    // state context.
    template <class Body>
    auto scope(Body body) -> Computation<typename ScopeResult<Body>::type> {
        using U = typename ScopeResult<Body>::type;

        auto& r = desc_->add_scope();
        Scope scope(*r.context);

        U out;

        if constexpr (std::is_void_v<T>) {
            out = body(scope);
        } else {
            out = body(scope, **this);
        }

        auto values = ComputationValue<U>::unwrap(out);

        for (auto& value : values)
            value = ComputationValue<std::decay_t<decltype(value)>>::output(scope, value);

        r.outputs = values;

        return {ComputationValue<U>::wrap(values), desc_};
    }

    // The Computation loop.
    // The provided `body` builds a single graph that is re-executed `count` times.
    // The body receives the scope, the loop's iteration clock (`iter`), and the current
    // computational value. Per-iteration inputs are dynamically rewritten before each
    // execution, and the body's output is fed back into the loop's state cells after
    // every step (as a save) to persist the loop state.
    template <class Body>
    auto fold(size_t count, Body body) -> Computation<T> {
        using U = typename FoldResult<Body>::type;

        auto& r = desc_->add_scope();
        r.repeat = {count, 0};
        Scope scope(*r.context);

        U next;

        if constexpr (std::is_void_v<T>) {
            next = body(scope, r.repeat->iter);
        } else {
            next = body(scope, r.repeat->iter, **this);
        }
        
        auto next_values = ComputationValue<U>::unwrap(next);
        auto curr_values = ComputationValue<T>::unwrap(**this);

        for (auto& value : next_values)
            value = ComputationValue<std::decay_t<decltype(value)>>::output(scope, value);
        
        r.outputs = next_values;

        for (auto i = 0; i < next_values.size(); ++i)
            r.saves.push_back({
                next_values[i], // src
                curr_values[i], // dst
                true // is_feedback
            });

        return *this;
    }

    // Creates an initial state cell.
    // Allocates a persistent tensor in the state context that exists before 
    // any graph runs. The value is populated by the provided `provider` callback.
    template <typename F>
    Computation<Tensor> state(const Tensor::Shape& shape, F&& provider) {
        // U is deduced from the provider's return type (std::vector<U>).
        using U = typename std::invoke_result_t<F, std::mt19937&>::value_type;
        auto cell = desc_->context().create<U>(shape, Context::Provider<U>(provider));
        return {cell, desc_};
    }

private:
    // Materializes the value into the state context when it lives in a
    // scope's scratch context: the scratch buffer dies with the scope, so
    // a value consumed across a scope boundary must be copied into a
    // persistent cell. The cell is allocated in the state context (the
    // temporarily active context; the active runtime is kept, so a
    // sharding runtime still traces the cell), and a save is registered
    // in the producing scope so the copy runs after every execution of
    // it. Every materialization of the same value returns the same cell
    // (the Description deduplicates), and a value that already lives in
    // a persistent context (the state or a pinned context) is returned
    // as-is.
    Tensor state(const Tensor& value) const {
        auto* scope = desc_->find_scope(value);

        // No scope boundary: no need for state, use as-is.
        if (scope == nullptr)
            return value;

        // Use already saved destination tensor.
        for (auto& save : scope->saves)
            if (*save.src == *value)
                return save.dst;

        // Create new state cell.
        Scope state_scope(desc_->context());
        auto cell = ComputationValue<Tensor>::state(value).name(value.name() + std::string(" (state)"));

        scope->saves.push_back({
            value, // src
            cell, // dst
            false // is_feedback
        });

        return cell;
    }

    Ref ref_;
    std::shared_ptr<ComputationDescription> desc_;
};
