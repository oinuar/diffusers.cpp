#pragma once

#include "ggml/Runtime.hpp"
#include "ggml/MetaDevice.hpp"
#include <unordered_map>
#include <set>
#include <map>
#include <numeric>
#include <sstream>
#include <iomanip>

class ShardingRuntime : public Runtime {
public:
    static constexpr int kNoAxis = -1;

    struct Dist {
        enum Type { R, S, P };

        Type type = Type::R;
        int axis = kNoAxis;

        bool operator==(const Dist& o) const { return type == o.type && axis == o.axis; }
        bool operator!=(const Dist& o) const { return !(*this == o); }
        bool operator<(const Dist& o) const {
            if (type != o.type) return type < o.type;
            return axis < o.axis;
        }

        std::string to_string() const {
            switch (type) {
                case Type::R: return "R";
                case Type::S: return "S(" + std::to_string(axis) + ")";
                case Type::P: return axis == kNoAxis ? "P" : "P(" + std::to_string(axis) + ")";
            }
            return "?";
        }

        // The plan -> GGML mapping for the axis field of the split state.
        enum ggml_backend_meta_split_axis to_split_axis() const {
            if (type == Type::S)
                return static_cast<enum ggml_backend_meta_split_axis>(axis);
            return GGML_BACKEND_SPLIT_AXIS_MIRRORED;
       }

        // The preference rank of the state in the planner's objective:
        // sharded (any axis) beats partial, partial beats replicated. A
        // candidate is priced with the sum of its inputs' ranks (see
        // candidate_cost), so a fully sharded path -- every input in some
        // S(a) -- always wins whenever it is feasible.
        int rank() const {
            switch (type) {
                case Type::S: return 0;
                case Type::P: return 1;
                case Type::R: return 2;
            }
            return 0;
        }

        static Dist replicated() { return Dist{}; }
        static Dist shard(int axis) { return {Type::S, axis}; }
        static Dist partial(int axis = kNoAxis) { return {Type::P, axis}; }
    };

    // One way an op can compute: the distribution it produces and the
    // distributions its inputs must be in (one per trace input, in
    // order). The score is not stored: the DP prices a candidate with
    // the sum of its inputs' preference ranks (Dist::rank), so a
    // sharded path always wins whenever it is feasible.
    struct Candidate {
        Dist output;
        std::vector<Dist> inputs;
    };

    struct TraceNode {
        int id = 0;
        std::string op_name;
        int rank = 0;                            // 0..4, logical dims
        int64_t ne[4] = {1, 1, 1, 1};            // GGML order, padded with 1s
        std::vector<int> inputs;                 // trace node ids
        std::vector<Candidate> candidates;       // empty = the meta backend cannot run this op
        bool is_param = false;                   // set via set_param(): static param, R or S(a) storage decision
        bool is_fixed = false;                   // graph input / compute leaf: externally fixed to R
        bool is_output = false;                  // set via set_output(): a graph output, a DP goal root
    };

    // `parent` creates every ggml tensor (in its ggml_context); the
    // engine only borrows the pointers and traces the graph. `device` is
    // the shared resource the plan commits its parameter splits to (the
    // split table the meta backend queries at allocation time) and
    // supplies the device count that sizes the sharded candidates.
    ShardingRuntime(Runtime& parent, MetaDevice& device)
        : parent_(parent), device_(device), n_devices_(device.count()) {}

    virtual ~ShardingRuntime() = default;

    // The sharding runtime plans every tensor across the meta
    // device's backends, so the sharding-adjusted module graphs apply
    // for every scope traced through it.
    bool sharded() const override { return true; }

    const std::vector<TraceNode>& nodes() const { return nodes_; }
    const std::vector<ggml_tensor*>& raw_of() const { return raw_of_; }

    // The trace node of a tensor this engine traced (or lazy-traced via
    // set_param / set_input / set_output).
    int id_of(ggml_tensor* t) const { return raw_to_id_.at(t); }

    // The goal roots of the trace: the outputs marked with set_output()
    // -- the DP roots plan() solves the whole trace for.
    const std::set<int>& goal_roots() const { return goal_roots_; }

    // The traced contexts are going away: their tensors are destroyed
    // with them, so the trace (which references them) is invalid. The
    // next generation re-traces through the same engine (the persistent
    // weights are re-traced lazily by set_param()). The plan of the
    // outgoing trace is retired with it: its committed splits are
    // removed from the meta device's split table and the next plan()
    // re-solves the re-traced trace from scratch.
    void reset() {
        for (const auto& [t, st] : splits_)
            device_.splits().erase(t);
        splits_.clear();

        nodes_.clear();
        raw_of_.clear();
        raw_to_id_.clear();
        goal_roots_.clear();

        decisions_.clear();
        exact_memo_.clear();
        best_memo_.clear();
        sink_memo_.clear();
        sink_required_.clear();
        planned_sinks_.clear();
        last_plan_.reset();
    }

    // ---------------------------------------------------------------------
    // Planning: one DP over the WHOLE trace
    // ---------------------------------------------------------------------
    // One way the plan stores a tensor: the distribution it is produced
    // in, the distribution its first consumer needs, and the collective
    // (bridge) between them.
    struct PlanNode {
        int id = 0;
        std::string op_name;
        std::string tensor_name;        // non-empty for params (the callback key)
        Dist produced;
        Dist required;
        std::string bridge;                 // collective between produced and required
    };

    struct Plan {
        int total_cost = 0;
        size_t device_count = 0;   // for printing the per-device split sizes
        bool infeasible = false;
        std::string infeasible_reason;
        std::vector<PlanNode> nodes;        // DFS preorder; printed in reverse = execution order
        // The plan -> GGML tensor split mapping: for every statically allocated
        // tensor, the split state a ggml_backend_meta_get_split_state_t callback
        // must return, keyed by tensor name. Compute tensors need no entry: the
        // meta backend derives their splits from these and its per-op rules.
        std::map<std::string, ggml_backend_meta_split_state> callback_states;

        std::string to_string() const {
            std::ostringstream ss;
            if (infeasible) {
                ss << "=== plan INFEASIBLE ===\n";
                ss << "  " << infeasible_reason << "\n";
                ss << "=======================================\n";
                return ss.str();
            }
            ss << "=== plan (total cost " << total_cost << ") ===\n";
#if 1
            for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
                const PlanNode& pn = *it;

                ss << "  [" << pn.id << "] ";

                if (pn.op_name == "param")
                    ss << (pn.tensor_name.empty() ? pn.op_name : pn.tensor_name);
                else
                    ss << pn.op_name << (pn.tensor_name.empty() ? "" : " " + pn.tensor_name);

                ss << ": ";

                if (pn.produced == pn.required) {
                    ss << pn.produced.to_string();
                } else {
                    ss << pn.produced.to_string() << " --" << pn.bridge << "--> " << pn.required.to_string();
                }
                ss << "\n";
            }
#endif
            if (!callback_states.empty()) {
                ss << "meta device callback table (ggml_backend_meta_split_state per static tensor):\n";
                for (const auto& [name, st] : callback_states) {
                    ss << "  " << std::left << std::setw(18) << name << std::right;
                    switch (st.axis) {
                        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: ss << "  MIRRORED      ne=[]"; break;
                        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:  ss << "  PARTIAL       ne=[]"; break;
                        default:
                            ss << "  axis " << (int)st.axis << "    ne=[";
                            for (size_t j = 0; j < device_count; ++j)
                                ss << (j ? ", " : "") << st.ne[j];
                            ss << "]";
                    }
                    ss << "\n";
                }
            }

            return ss.str();
        }
    };

    // Solves the WHOLE trace in one go: a single DP over every traced
    // tensor. The goal roots are the outputs the trace marked with
    // set_output() (each required exactly in R -- the only state a graph
    // output can be read back in; a tensor shared by several roots is
    // planned exactly once, for all of them). The trace's sinks -- the
    // tensors no other traced tensor consumes -- are planned as well, in
    // any state they can be produced in: the allocator assigns EVERY
    // tensor of the context to the meta buffer (not only the ones
    // reachable from the goal roots), and the meta backend derives a
    // split state for every tensor it is assigned. A dead end left
    // unplanned keeps its weights at the default MIRRORED split and can
    // derive an illegal state (a dead-end flash_attn hard-requires its
    // q/k/v sharded S(2) -- MIRRORED aborts its assert). Nobody reads a
    // sink, so it is required in no particular state and needs no bridge.
    // Returns
    // the plan: the distribution every planned tensor is produced in,
    // the P -> R bridges, and the meta device's callback states for the
    // params -- whose splits are committed to the device's split table,
    // the shared resource GLOBAL across contexts and allocators. A
    // re-plan with an unchanged trace reuses the last
    // plan (the DP is not re-run); an infeasible trace returns the plan
    // marked infeasible (nothing is committed).
    const Plan& plan();
    // Zeros the storage of every PARTIAL (row-parallel) tensor that is
    // currently allocated (i.e. belongs to the scope context allocated
    // most recently; other scopes' tensors are either unallocated or
    // already released and are skipped).
    //
    // A PARTIAL tensor's per-device contribution is computed only on
    // the devices that have a non-empty slice of its sources: the meta
    // backend clears the compute of every zero-slice tensor, so on a
    // zero-factor device the partial is never written and its storage
    // keeps whatever stale bytes the allocator left there (possibly
    // NaN). The AllReduce's zeroing of such a disabled device is an
    // in-place scale by 0.0f, which cannot clear a stale NaN (0.0f *
    // NaN == NaN, the FIXME ggml carries at that spot), and the
    // butterfly sum would add the stale values into the real result.
    // Seeding the storage with zero right after the allocation makes
    // every disabled device contribute exactly zero -- the correct
    // partial sum for an empty slice -- on every (re-)execution: the
    // storage stays finite from then on (the zeroing scales it by 0
    // and the butterfly only adds finite values).
    void zero_partials() {
        std::vector<std::byte> zeros;

        for (const auto& node : plan().nodes) {
            if (node.produced.type != Dist::Type::P)
                continue;

            ggml_tensor* t = raw_of_[node.id];

            // Only the tensors of the just-allocated scope context are
            // assigned to a buffer: the plan spans the whole trace.
            if (t->buffer == nullptr || t->data == nullptr)
                continue;

            if (t->type != GGML_TYPE_F32)
                throw std::runtime_error("zero_partials(): a PARTIAL tensor is not F32");

            zeros.assign(ggml_nbytes(t), std::byte{0});
            ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
        }
    }

    // Debug dump of the trace: every node with its shape and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // The parameter splits of the last plan(): one entry per shared
    // parameter.
    const std::map<const ggml_tensor*, Dist>& decisions() const { return decisions_; }

    const MetaDevice& device() const { return device_; }

    // ---------------------------------------------------------------------
    // Runtime: tensor creation / initialization
    // ---------------------------------------------------------------------
    ggml_tensor* new_tensor(ggml_type type, int n_dims, const int64_t* ne) override {
        ggml_tensor* t = parent_.new_tensor(type, n_dims, ne);
        const int rank = std::clamp(n_dims, 0, 4);
        int64_t padded[4] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i)
            padded[i] = ne[i];

        // A bare tensor created through the engine is a compute leaf:
        // the meta backend stores it in the compute buffer as GGML_OP_NONE
        // and derives MIRRORED for it, so the only legal state is R. A
        // model param created through the engine (Context::create) is
        // pinned to R by the following set_input(), and set_param() (the
        // param's forward) refines this node to the param candidates when
        // the weight enters the graph.
        trace_op("new_tensor", {}, {{Dist::replicated(), {}}}, rank, padded, t);
        return t;
    }

    ggml_tensor* new_tensor_1d(ggml_type type, int64_t ne0) override {
        const int64_t ne[1] = {ne0};
        return new_tensor(type, 1, ne);
    }

    // A tensor whose state is externally fixed: a graph input or a compute
    // leaf (Tensor::empty). The meta backend stores compute leaves in the
    // compute buffer as GGML_OP_NONE, which is MIRRORED -- so the only
    // legal fixed state is R.
    void set_input(ggml_tensor* t) override {
        parent_.set_input(t);
        TraceNode& n = nodes_[ensure_node(t)];
        n.is_fixed = true;
        n.op_name = "input";
        n.is_param = false;
        n.candidates = {{Dist::replicated(), {}}};
    }

    // Marks a tensor as a model param (the project's Parameter::forward()
    // calls this when a weight enters the graph): a static tensor whose
    // storage split (R or S(a)) the plan materializes in the callback
    // table, keyed by the tensor name. The param state is re-established
    // here unconditionally: the tensor may have been created through this
    // engine earlier and pinned to R by set_input() (Context::create
    // makes every tensor an input leaf first) -- only set_param's state
    // decides its storage split.
    void set_param(ggml_tensor* t) override {
        parent_.set_param(t);
        TraceNode& n = nodes_[ensure_node(t)];
        n.is_param = true;
        n.is_fixed = false;
        n.op_name = "param";
        n.candidates = param_candidates(n.rank);
    }

    // Marks a tensor as a graph output (the Computation monad marks its
    // scope outputs through the active runtime): the trace node becomes a
    // DP goal root -- plan() solves the whole trace for the roots, each
    // required exactly replicated (R), since that is the only state a
    // graph output can be read back in. The node is only flagged
    // (is_output, goal_roots_): an output is a computed tensor, its
    // distribution the DP decides -- not fixed like an input (set_input).
    void set_output(ggml_tensor* t) override {
        parent_.set_output(t);
        const int id = ensure_node(t);
        nodes_[id].is_output = true;
        goal_roots_.insert(id);
    }

    ggml_tensor* fill(ggml_tensor* t, float value) override {
        // The meta runs FILL through handle_generic (state carries over from
        // the shape template); each device simply fills its own slice.
        ggml_tensor* out = parent_.fill(t, value);
        const int id = get_id(t);
        const int rank = rank_of(t);
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        for (int a = 0; a < rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}});
        return traced("fill", {id}, std::move(cands), rank, nodes_[id].ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: copy / cast
    // ---------------------------------------------------------------------
    ggml_tensor* cont(ggml_tensor* t) override {
        // GGML_OP_CONT and GGML_OP_RESHAPE share the meta's handle_reshape rule.
        return reinterpret_op("cont", parent_.cont(t), t, nodes_[get_id(t)].ne);
    }

    ggml_tensor* dup(ggml_tensor* t) override {
        // The meta runs DUP with scalar_only: a DUP of a sharded tensor
        // ABORTS. Only a replicated copy is planned.
        ggml_tensor* out = parent_.dup(t);
        const int id = get_id(t);
        return traced("dup", {id}, {{Dist::replicated(), {Dist::replicated()}}}, nodes_[id].rank, nodes_[id].ne, out);
    }

    ggml_tensor* cast(ggml_tensor* t, ggml_type type) override {
        // GGML_OP_CAST is not in the meta backend's switch -> it ABORTs on
        // any split state. The node gets no candidates: any graph that casts
        // is infeasible (keep the graph dtype-uniform instead).
        ggml_tensor* out = parent_.cast(t, type);
        const int id = get_id(t);
        return traced("cast", {id}, {}, nodes_[id].rank, nodes_[id].ne, out);
    }

    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override {
        // dst is a shape template; data flows from src. The meta routes a
        // sharded CPY through handle_reshape (the shard remaps to the dst
        // shape); a replicated CPY is a plain copy. handle_reshape requires
        // src_rank <= dst_rank, so sharded candidates only exist then.
        ggml_tensor* out = parent_.cpy(src, dst);
        const int si = get_id(src);
        const int di = get_id(dst);
        const int rank = nodes_[di].rank;
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}}};
        if (nodes_[si].rank <= rank) {
            for (int a = 0; a < nodes_[si].rank; ++a) {
                if (derive_reshape(nodes_[si].ne, nodes_[di].ne, Dist::shard(a)))
                    cands.push_back({Dist::shard(derive_reshape_axis(nodes_[si].ne, nodes_[di].ne, a)), {Dist::shard(a), Dist::replicated()}});
            }
        }
        return traced("cpy", {si, di}, std::move(cands), rank, nodes_[di].ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: unary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* sqrt(ggml_tensor* t) override { return carry_over_op("sqrt", parent_.sqrt(t), t); }
    ggml_tensor* exp(ggml_tensor* t) override { return carry_over_op("exp", parent_.exp(t), t); }
    ggml_tensor* log(ggml_tensor* t) override { return carry_over_op("log", parent_.log(t), t); }
    ggml_tensor* sin(ggml_tensor* t) override { return carry_over_op("sin", parent_.sin(t), t); }
    ggml_tensor* cos(ggml_tensor* t) override { return carry_over_op("cos", parent_.cos(t), t); }
    ggml_tensor* sigmoid(ggml_tensor* t) override { return carry_over_op("sigmoid", parent_.sigmoid(t), t); }

    // ---------------------------------------------------------------------
    // Runtime: binary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* add(ggml_tensor* l, ggml_tensor* r) override { return binary_op("add", parent_.add(l, r), l, r); }
    ggml_tensor* sub(ggml_tensor* l, ggml_tensor* r) override { return binary_op("sub", parent_.sub(l, r), l, r); }
    ggml_tensor* mul(ggml_tensor* l, ggml_tensor* r) override { return binary_op("mul", parent_.mul(l, r), l, r); }
    ggml_tensor* div(ggml_tensor* l, ggml_tensor* r) override { return binary_op("div", parent_.div(l, r), l, r); }

    // ---------------------------------------------------------------------
    // Runtime: scalar arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* scale(ggml_tensor* t, float value) override { return carry_over_op("scale", parent_.scale(t, value), t); }
    ggml_tensor* clamp(ggml_tensor* t, float min, float max) override { return carry_over_op("clamp", parent_.clamp(t, min, max), t); }

    // ---------------------------------------------------------------------
    // Runtime: matrix operations
    // ---------------------------------------------------------------------
    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        // Convention (see nn/Linear): lhs = weight [in, out], rhs = activation.
        // ggml_mul_mat: result ne = {lhs->ne[1], rhs->ne[1], rhs->ne[2], rhs->ne[3]},
        // so the batch dims (and the output rank) come from the rhs.
        ggml_tensor* out = parent_.mul_mat(l, r);
        const int li = get_id(l);
        const int ri = get_id(r);
        const TraceNode& w = nodes_[li];
        const TraceNode& a = nodes_[ri];
        const int64_t out_ne[4] = {w.ne[1], a.ne[1], a.ne[2], a.ne[3]};
        return traced("mul_mat", {li, ri}, mul_mat_candidates(w, a), a.rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: reshape / permute / views
    // ---------------------------------------------------------------------
    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t ne0) override {
        const int64_t out_ne[4] = {ne0, 1, 1, 1};
        return reinterpret_op("reshape", parent_.reshape_1d(t, ne0), t, out_ne);
    }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t ne0, int64_t ne1) override {
        const int64_t out_ne[4] = {ne0, ne1, 1, 1};
        return reinterpret_op("reshape", parent_.reshape_2d(t, ne0, ne1), t, out_ne);
    }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, 1};
        return reinterpret_op("reshape", parent_.reshape_3d(t, ne0, ne1, ne2), t, out_ne);
    }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return reinterpret_op("reshape", parent_.reshape_4d(t, ne0, ne1, ne2, ne3), t, out_ne);
    }

    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override {                                                                                                                                 
        ggml_tensor* out = parent_.permute(t, a0, a1, a2, a3);                                                                                                                                                      
        const int id = get_id(t);                                                                                                                                                                                   
        const TraceNode& src = nodes_[id];                                                                                                                                                                          
        const int ax[4] = {a0, a1, a2, a3};                                                                                                                                                                         
        int64_t out_ne[4] = {1, 1, 1, 1};                                                                                                                                                                           
        for (int i = 0; i < 4; ++i)                                                                                                                                                                                 
            out_ne[i] = src.ne[ax[i] >= 0 && ax[i] < 4 ? ax[i] : i];                                                                                                                                                
                                                            
        // GGML_OP_PERMUTE: a shard of the input along axis b reappears on
        // the output axis i with ax[i] == b (the meta's handle_permute).                                                                                                                                                        
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        for (int b = 0; b < src.rank; ++b) {          // b: the source's meaningful axes
            for (int i = 0; i < 4; ++i) {             // i: the output axis (the full 4D space)
                if (ax[i] == b)
                    cands.push_back({Dist::shard(i), {Dist::shard(b)}});
            }                                                                                                                                                                                                       
        }                                                                                                                                                                                                           
        return traced("permute", {id}, std::move(cands), out_rank_of(out_ne), out_ne, out);                                                                                                                         
    }   

    // view_1d/2d/3d/4d → GGML_OP_VIEW: modeled by view_op() (the meta's
    // handle_view: contiguous views follow the handle_reshape rule, views
    // with unchanged strides carry the source state over, the remaining
    // unpermuted case remaps the axis via the matching next-dim stride).
    ggml_tensor* view_1d(ggml_tensor* t, int64_t ne0, size_t offset) override {
        return view_op("view", parent_.view_1d(t, ne0, offset), t);
    }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) override {
        return view_op("view", parent_.view_2d(t, ne0, ne1, nb1, offset), t);
    }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) override {
        return view_op("view", parent_.view_3d(t, ne0, ne1, ne2, nb1, nb2, offset), t);
    }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) override {
        return view_op("view", parent_.view_4d(t, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset), t);
    }

    // ---------------------------------------------------------------------
    // Runtime: repeat / broadcast, concatenation
    // ---------------------------------------------------------------------
    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override {
        // The meta runs REPEAT through handle_generic: all srcs must be in
        // the SAME state. The target is a compute leaf (fixed R), so only a
        // replicated source can be broadcast.
        ggml_tensor* out = parent_.repeat(t, target);
        const int ti = get_id(t);
        const int ri = get_id(target);
        return traced("repeat", {ti, ri}, {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}}}, nodes_[ri].rank, nodes_[ri].ne, out);
    }

    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) override {
        // GGML_OP_CONCAT (the meta's handle_concat): one operand may be
        // sharded along any axis OTHER than the concat dim; both may be
        // sharded along the same axis. (dim is the GGML axis.)
        ggml_tensor* out = parent_.concat(a, b, dim);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int rank = std::max(nodes_[ai].rank, nodes_[bi].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = (i == dim) ? nodes_[ai].ne[i] + nodes_[bi].ne[i] : nodes_[ai].ne[i];

        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}}};
        for (int a = 0; a < rank; ++a) {
            if (a == dim)
                continue;
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::shard(a)}});
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::replicated()}});
            cands.push_back({Dist::shard(a), {Dist::replicated(), Dist::shard(a)}});
        }
        return traced("concat", {ai, bi}, std::move(cands), rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: reduction
    // ---------------------------------------------------------------------
    ggml_tensor* sum_rows(ggml_tensor* t) override {
        // GGML_OP_SUM_ROWS (the meta's handle_per_row): asserts the src is
        // not sharded along the reduced axis 0 and keeps the src state
        // UNCHANGED (axis preserved, ne preserved).
        ggml_tensor* out = parent_.sum_rows(t);
        const int id = get_id(t);
        const int in_rank = nodes_[id].rank;
        int64_t out_ne[4] = {1, nodes_[id].ne[1], nodes_[id].ne[2], nodes_[id].ne[3]};
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        for (int a = 1; a < in_rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}});
        return traced("sum_rows", {id}, std::move(cands), in_rank > 0 ? in_rank - 1 : 0, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: attention
    // ---------------------------------------------------------------------
    ggml_tensor* flash_attn_ext(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale, float max_bias, float logit_softcap) override {
        // GGML_OP_FLASH_ATTN_EXT (the meta's handle_flash_attn_ext): q, k
        // and v are HARD-asserted to be S(2) (the sequence axis of the
        // rank-4 {head_dim, heads, seq, batch} layout), a present mask must
        // be MIRRORED, and the output is S(1). There is no replicated
        // candidate. ggml shape: out ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}.
        ggml_tensor* out = parent_.flash_attn_ext(q, k, v, mask, scale, max_bias, logit_softcap);
        const int qi = get_id(q);
        const int ki = get_id(k);
        const int vi = get_id(v);
        const TraceNode& qt = nodes_[qi];
        const int64_t out_ne[4] = {nodes_[vi].ne[0], qt.ne[2], qt.ne[1], qt.ne[3]};

        std::vector<int> inputs = {qi, ki, vi};
        std::vector<Dist> in_dists = {Dist::shard(2), Dist::shard(2), Dist::shard(2)};
        if (mask) {
            inputs.push_back(get_id(mask));
            in_dists.push_back(Dist::replicated());
        }
        return traced("flash_attn", inputs, {{Dist::shard(1), std::move(in_dists)}}, qt.rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: convolution / pooling / resampling (vision)
    // ---------------------------------------------------------------------
    ggml_tensor* conv_2d_direct(ggml_tensor* a, ggml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1) override {
        // GGML_OP_CONV_2D goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned. a is the kernel [KW, KH, IC, OC], b
        // the input [W, H, C, N]; out = [OW, OH, OC, N].
        ggml_tensor* out = parent_.conv_2d_direct(a, b, s0, s1, p0, p1, d0, d1);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int64_t out_ne[4] = {
            conv_out_size(nodes_[bi].ne[0], nodes_[ai].ne[0], s0, p0, d0),
            conv_out_size(nodes_[bi].ne[1], nodes_[ai].ne[1], s1, p1, d1),
            nodes_[ai].ne[3],
            nodes_[bi].ne[3],
        };
        return traced("conv_2d", {ai, bi}, {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}}}, 4, out_ne, out);
    }

    ggml_tensor* pool_2d(ggml_tensor* a, ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1) override {
        // GGML_OP_POOL_2D goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned.
        ggml_tensor* out = parent_.pool_2d(a, op, k0, k1, s0, s1, p0, p1);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {
            pool_out_size(nodes_[ai].ne[0], k0, s0, p0),
            pool_out_size(nodes_[ai].ne[1], k1, s1, p1),
            nodes_[ai].ne[2],
            nodes_[ai].ne[3],
        };
        return traced("pool_2d", {ai}, {{Dist::replicated(), {Dist::replicated()}}}, 4, out_ne, out);
    }

    ggml_tensor* interpolate(ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode) override {
        // ggml_interpolate creates a GGML_OP_UPSCALE node, which goes
        // through the meta's handle_generic with scalar_only = true: a
        // sharded src would ABORT. Only the replicated form is planned.
        ggml_tensor* out = parent_.interpolate(a, ne0, ne1, ne2, ne3, mode);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return traced("interpolate", {ai}, {{Dist::replicated(), {Dist::replicated()}}}, out_rank_of(out_ne), out_ne, out);
    }

    ggml_tensor* upscale(ggml_tensor* a, int scale_factor, ggml_scale_mode mode) override {
        // GGML_OP_UPSCALE goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned. ne0/ne1 are multiplied by the scale
        // factor.
        ggml_tensor* out = parent_.upscale(a, scale_factor, mode);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {nodes_[ai].ne[0] * scale_factor, nodes_[ai].ne[1] * scale_factor, nodes_[ai].ne[2], nodes_[ai].ne[3]};
        return traced("upscale", {ai}, {{Dist::replicated(), {Dist::replicated()}}}, out_rank_of(out_ne), out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: embeddings / normalization / rotary embeddings
    // ---------------------------------------------------------------------
    ggml_tensor* get_rows(ggml_tensor* a, ggml_tensor* b) override {
        // GGML_OP_GET_ROWS (the meta's handle_get_rows): the data may be
        // sharded along axis 0 while the row indices stay replicated;
        // otherwise everything must be replicated (scalar_only fallback).
        // ggml asserts b = {n_rows, a->ne[2], a->ne[3], 1};
        // out ne = {a->ne[0], b->ne[0], b->ne[1], b->ne[2]}.
        ggml_tensor* out = parent_.get_rows(a, b);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int64_t out_ne[4] = {nodes_[ai].ne[0], nodes_[bi].ne[0], nodes_[bi].ne[1], nodes_[bi].ne[2]};
        std::vector<Candidate> cands = {
            {Dist::replicated(), {Dist::replicated(), Dist::replicated()}},
            {Dist::shard(0), {Dist::shard(0), Dist::replicated()}},
        };
        return traced("get_rows", {ai, bi}, std::move(cands), out_rank_of(out_ne), out_ne, out);
    }

    ggml_tensor* norm(ggml_tensor* a, float eps) override { return per_row_op("norm", parent_.norm(a, eps), a); }
    ggml_tensor* rms_norm(ggml_tensor* a, float eps) override { return per_row_op("rms_norm", parent_.rms_norm(a, eps), a); }

    ggml_tensor* rope_ext(ggml_tensor* a, ggml_tensor* b, ggml_tensor* c,
                          int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
                          float ext_factor, float attn_factor, float beta_fast, float beta_slow) override {
        // GGML_OP_ROPE (the meta's handle_rope): the position src b must be
        // MIRRORED (hard assert); a's state carries over unchanged, any
        // axis (even 0). c (cos/sin cache) is not state-checked by the
        // meta; a sharded one would fail its ratio asserts, so only a
        // replicated c is planned.
        ggml_tensor* out = parent_.rope_ext(a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        const int ai = get_id(a);
        const TraceNode& at = nodes_[ai];
        const bool has_c = c != nullptr;

        std::vector<int> inputs = {ai, get_id(b)};
        if (has_c)
            inputs.push_back(get_id(c));

        std::vector<Candidate> cands;
        {
            std::vector<Dist> ins = {Dist::replicated(), Dist::replicated()};
            if (has_c) ins.push_back(Dist::replicated());
            cands.push_back({Dist::replicated(), std::move(ins)});
        }
        for (int ax = 0; ax < at.rank; ++ax) {
            std::vector<Dist> ins = {Dist::shard(ax), Dist::replicated()};
            if (has_c) ins.push_back(Dist::replicated());
            cands.push_back({Dist::shard(ax), std::move(ins)});
        }
        return traced("rope", inputs, std::move(cands), at.rank, at.ne, out);
    }

private:
    // ---------------------------------------------------------------------
    // Candidate generation -- exactly the states the meta backend accepts
    // (see the per-op rules in the file header).
    // ---------------------------------------------------------------------
    std::vector<Candidate> param_candidates(int rank) const;

    // Elementwise unary: the meta carries the src state over unchanged.
    std::vector<Candidate> carry_over_candidates(int rank) const;

    // ggml binary op: lhs broadcasts rhs against itself (the project's
    // Tensor operators keep the broadcast superset on the left).
    std::vector<Candidate> binary_candidates(const TraceNode& lhs, const TraceNode& rhs, int out_rank) const;

    // mul_mat(lhs = weight [in, out], rhs = activation), result rank = rank(rhs).
    // The meta's handle_mul_mat accepts exactly these four tuples; the
    // row-parallel one additionally GGML_ASSERTs that the weight and
    // activation splits are equal, which holds for near-uniform splits iff
    // the contract dim sizes match.
    std::vector<Candidate> mul_mat_candidates(const TraceNode& w, const TraceNode& a) const;

    // ---------------------------------------------------------------------
    // The meta's handle_reshape, ported: given an input sharded along axis
    // `axis`, which output axis does a reshape/view/cont to shape
    // `out_ne` produce? Returns std::nullopt when the meta would
    // GGML_ABORT("shape mismatch ...").
    // ---------------------------------------------------------------------
    static int n_dims(const int64_t ne[4]) {
        int n = 4;
        while (n > 1 && ne[n - 1] == 1) --n;
        return n;
    }

    static bool derive_reshape(const int64_t src_ne[4], const int64_t out_ne[4], const Dist& in) {
        if (in.type == Dist::Type::R)
            return true;
        if (in.type != Dist::Type::S)
            return false;   // a P source is visible as R (handled by the R candidate)
        const int axis = in.axis;
        if (axis < 0 || axis > 3)
            return false;
        // The nr[0] == 1 fast path (the planner only produces single-segment
        // splits with nr == 1): a shard of the src's last meaningful dim
        // lands on the output's last meaningful dim.
        if (axis == n_dims(src_ne) - 1)
            return true;
        int64_t base_ne_in = 1;
        for (int dim = 0; dim <= axis; ++dim)
            base_ne_in *= src_ne[dim];
        int64_t base_ne_out = 1;
        for (int dim = 0; dim < 4; ++dim) {
            base_ne_out *= out_ne[dim];
            if (base_ne_out % base_ne_in == 0)
                return true;
            if (base_ne_out > base_ne_in)
                return true;   // the meta asserts n_segments == 1 && nr[0] == 1 (both true here)
        }
        return false;   // the meta GGML_ABORTs: "shape mismatch"
    }

    // reshape/cont/view: zero-cost memory reinterpretation.
    ggml_tensor* reinterpret_op(const char* name, ggml_tensor* out, ggml_tensor* t, const int64_t out_ne[4]) {
        const int id = get_id(t);
        const TraceNode& src = nodes_[id];
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        for (int a = 0; a < src.rank; ++a) {
            if (derive_reshape(src.ne, out_ne, Dist::shard(a)))
                cands.push_back({Dist::shard(derive_reshape_axis(src.ne, out_ne, a)), {Dist::shard(a)}});
        }
        return traced(name, {id}, std::move(cands), out_rank_of(out_ne), out_ne, out);
    }

    // Which output axis a shard of input axis `a` maps to (handle_reshape).
    static int derive_reshape_axis(const int64_t src_ne[4], const int64_t out_ne[4], int a) {
        if (a == n_dims(src_ne) - 1)
            return n_dims(out_ne) - 1;
        int64_t base_ne_in = 1;
        for (int dim = 0; dim <= a; ++dim)
            base_ne_in *= src_ne[dim];
        int64_t base_ne_out = 1;
        for (int dim = 0; dim < 4; ++dim) {
            base_ne_out *= out_ne[dim];
            if (base_ne_out % base_ne_in == 0)
                return dim;
            if (base_ne_out > base_ne_in)
                return dim;
        }
        return a;   // unreachable for candidate axes (derive_reshape checked)
    }


    // GGML_OP_VIEW (the meta's handle_view): a view that is contiguous (and
    // views a contiguous source) follows the handle_reshape rule above; a view
    // whose strides match the source's on every non-trivial dim (e.g. a dim-0
    // slice) carries the source's state over unchanged (axis preserved); the
    // remaining unpermuted case maps the source shard axis to the output axis
    // whose next-dim stride matches; R/P always carry over.
    ggml_tensor* view_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        const TraceNode& src = nodes_[id];
        int64_t out_ne[4] = {out->ne[0], out->ne[1], out->ne[2], out->ne[3]};
        const int out_rank = out_rank_of(out_ne);
        if (ggml_is_contiguous(out) && ggml_is_contiguous(t))
            return reinterpret_op(name, out, t, out_ne);
        bool all_strides_the_same = true;
        for (int i = 0; i < 4; ++i) {
            if (out_ne[i] == 1 && src.ne[i] == 1)
                continue;
            if (out->nb[i] != t->nb[i]) {
                all_strides_the_same = false;
                break;
            }
        }
        const size_t off = (out->view_src == t) ? out->view_offs : 0;

        if (all_strides_the_same) {
            std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
            // A shard-axis carry-over (the meta returns src_ss[0] as-is,
            // nr included) is sound only if the view covers the sharded
            // axis in full from its start: every device must own the same
            // uniform slice of the view. A windowed slice along the sharded
            // axis (a chunk at an offset) would leave some devices with the
            // wrong or no data while the derived state claims a uniform
            // split. Slices along the other axes are local to each device's
            // region and always fine.
            for (int a = 0; a < src.rank; ++a) {
                const int64_t s_a = (int64_t)((off / out->nb[a]) % src.ne[a]);
                if (s_a == 0 && out_ne[a] == src.ne[a])
                    cands.push_back({Dist::shard(a), {Dist::shard(a)}});
            }
            return traced(name, {id}, std::move(cands), out_rank, out_ne, out);
        }
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        if (!ggml_is_permuted(out) && !ggml_is_permuted(t)) {
            for (int axis = 0; axis < 3 && axis < src.rank; ++axis) {
                // Same uniformity requirement as the stride branch above:
                // the source shard axis must be covered in full from its
                // start, and the mapped output axis must keep its full size.
                const int64_t s_axis = (int64_t)((off / t->nb[axis]) % src.ne[axis]);
                if (s_axis != 0)
                    continue;
                for (int dim = 0; dim < 3; ++dim) {
                    if (out->nb[dim + 1] == t->nb[axis + 1]) {
                        if (out_ne[dim] == src.ne[axis] && out_ne[dim] > 1)
                            cands.push_back({Dist::shard(dim), {Dist::shard(axis)}});
                        break;
                    }
                }
            }
        }
        return traced(name, {id}, std::move(cands), out_rank, out_ne, out);
    }
    static int out_rank_of(const int64_t ne[4]) { return n_dims(ne); }

    // ggml's output-size formulas (ggml.c), for conv_2d_direct and pool_2d.
    static int64_t conv_out_size(int64_t in, int64_t k, int s, int p, int d) {
        return (in + 2 * p - d * (k - 1) - 1) / s + 1;
    }
    static int64_t pool_out_size(int64_t in, int k, int s, float p) {
        return (in + 2 * p - k) / s + 1;
    }

    ggml_tensor* carry_over_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        return traced(name, {id}, carry_over_candidates(nodes_[id].rank), nodes_[id].rank, nodes_[id].ne, out);
    }

    // GGML_OP_NORM / GGML_OP_RMS_NORM (the meta's handle_per_row): the src
    // must not be sharded along the reduced axis 0; the state carries over
    // unchanged (axis preserved, ne preserved).
    ggml_tensor* per_row_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        const int rank = nodes_[id].rank;
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}}};
        for (int a = 1; a < rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}});
        return traced(name, {id}, std::move(cands), rank, nodes_[id].ne, out);
    }

    // An op the meta backend does not handle: zero candidates, so the DP
    // can only report the graph infeasible (with this op named).
    ggml_tensor* unsupported_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        return traced(name, {id}, {}, nodes_[id].rank, nodes_[id].ne, out);
    }

    // ---------------------------------------------------------------------
    // Tracing helpers
    // ---------------------------------------------------------------------
    int rank_of(ggml_tensor* t) const { return nodes_[get_id(t)].rank; }

    // The trace node for a tensor the planner has not traced yet: it was
    // created outside the plan phase (e.g. by a loader under a different
    // engine), so its shape is read from the ggml tensor and it starts as
    // a bare static-tensor candidate (set_param / set_input refine it).
    int ensure_node(ggml_tensor* t) {
        const auto it = raw_to_id_.find(t);
        if (it != raw_to_id_.end())
            return it->second;
        const int rank = std::clamp(ggml_n_dims(t), 0, 4);
        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i)
            ne[i] = t->ne[i];
        return trace_op("new_tensor", {}, param_candidates(rank), rank, ne, t);
    }

    int trace_op(const char* name, const std::vector<int>& inputs,
                 std::vector<Candidate> candidates, int rank, const int64_t ne[4], ggml_tensor* out) {
        const int id = (int)nodes_.size();
        TraceNode n;
        n.id = id;
        n.op_name = name;
        n.rank = rank;
        for (int i = 0; i < 4; ++i)
            n.ne[i] = ne[i];
        n.inputs = inputs;
        n.candidates = std::move(candidates);
        nodes_.push_back(std::move(n));
        raw_of_.push_back(out);
        raw_to_id_[out] = id;
        return id;
    }

    // trace_op + return the parent-created tensor (the op methods above).
    ggml_tensor* traced(const char* name, const std::vector<int>& inputs,
                        std::vector<Candidate> candidates, int rank, const int64_t ne[4], ggml_tensor* out) {
        trace_op(name, inputs, std::move(candidates), rank, ne, out);
        return out;
    }

    int get_id(ggml_tensor* t) const { return raw_to_id_.at(t); }

    ggml_tensor* binary_op(const char* name, ggml_tensor* out, ggml_tensor* l, ggml_tensor* r) {
        const int li = get_id(l);
        const int ri = get_id(r);
        const int rank = std::max(nodes_[li].rank, nodes_[ri].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = std::max(nodes_[li].ne[i], nodes_[ri].ne[i]);
        return traced(name, {li, ri}, binary_candidates(nodes_[li], nodes_[ri], rank), rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Plan -> GGML split mapping
    // ---------------------------------------------------------------------
    // Materialize the split state a callback must return for a static
    // tensor with distribution `d`, GGML shape `ne` and dtype `type`:
    //   R  -> the canonical MIRRORED form (axis = MIRRORED, ne = 0,
    //         nr[0] = 1, n_segments = 1)
    //   S(a) -> one segment, nr = 1, with the per-device sizes either
    //         taken from `split` (a group-consistent split computed by
    //         compute_split_groups(); its invariants -- sum to ne[a],
    //         block-aligned for a == 0 -- are the solver's to uphold)
    //         or, when `split` is null, proportional to the tensor-split
    //         factors:
    //         boundary(j) = ne * (f_0 + ... + f_j) / (f_0 + ... + f_n-1),
    //         truncated and (for a == 0) rounded down to a multiple of
    //         ggml_blck_size (the meta GGML_ASSERTs it); the last device
    //         gets the remainder. An all-zero factor list is the default
    //         even split with even-split boundaries
    //         (boundary(j) = ne * (j + 1) / n).
    // P is never materialized: the callback is only called for static
    // tensors, and a static tensor is never PARTIAL.
    ggml_backend_meta_split_state materialize(
        const Dist& d, const int64_t ne[4], ggml_type type,
        const std::vector<int64_t>* split = nullptr) const {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = d.to_split_axis();
        st.nr[0] = 1;
        st.n_segments = 1;
        if (d.type == Dist::Type::S) {
            const int n = (int)device_.count();

            if (split != nullptr) {
                for (int j = 0; j < n; ++j)
                    st.ne[j] = (*split)[j];
            } else {
                const int64_t gran = d.axis == 0 ? ggml_blck_size(type) : 1;
                int64_t low = 0;

                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += device_.tensor_split(j);

                double prefix = 0.0;
                for (int j = 0; j < n; ++j) {
                    int64_t high;
                    if (j + 1 < n) {
                        prefix += device_.tensor_split(j);
                        high = (sum > 0.0)
                            ? (int64_t)((double)ne[d.axis] * prefix / sum)
                            : ne[d.axis] * (int64_t)(j + 1) / n;
                        high = (high / gran) * gran;
                    } else {
                        high = ne[d.axis];
                    }
                    st.ne[j] = high - low;
                    low = high;
                }
            }
        }
        return st;
    }

    // ---------------------------------------------------------------------
    // Ratio-coupled split groups (the solver lives in ShardingRuntime.cpp)
    // ---------------------------------------------------------------------
    // The per-device shard sizes the meta backend derives for a sharded
    // COMPUTE tensor come from the committed sizes of its sharded sources
    // (an exact ratio, GGML_ASSERTed integral -- the "take over ratio"
    // block of ggml-backend-meta.cpp), and every two sharded sources of
    // one op must agree on that ratio (a row-parallel mul_mat requires
    // the two states equal). So every set of tensors that derive one
    // another's splits -- a "ratio group" -- must share ONE proportional
    // split: member T owns N_T * s_j / N elements on device j (s sums to
    // N, the anchor member's size). For a non-uniform tensor-split
    // request the plain per-tensor proportional truncation breaks the
    // ratio integrality (a [in, 512] weight split [227, 285] reshaped
    // into 4 heads: 4 * 227 / 512 is not an integer), so each group is
    // solved for the split closest to the requested factors that the
    // ratio rules admit.
    struct SplitGroupResult {
        bool ok = true;
        std::string reason;                        // when !ok: why no split works
        std::map<int, std::vector<int64_t>> splits; // node id -> per-device sizes
    };
    SplitGroupResult compute_split_groups(const std::map<int, Dist>& produced_of);

    std::string param_name(int id) const {
        const char* n = raw_of_[id]->name;
        return n[0] ? n : ("node" + std::to_string(id));
    }

    // ---------------------------------------------------------------------
    // Dynamic program
    // ---------------------------------------------------------------------
    // Tree DP over the whole trace, solved once per plan() (one go): a
    // single memo shared by every goal root and sink, so F(node, d) is
    // computed once, for all consumers. The objective is the state
    // preference, not a cost: sharded (any S) beats partial (P), partial
    // beats replicated (R) (Dist::rank). A candidate is priced with the
    // sum of its inputs' ranks, so a fully sharded path -- score 0 --
    // always wins whenever it is feasible. The scores are small integers
    // independent of the graph size, so deep traces neither overflow nor
    // lose precision. Feasibility is tracked separately (the `feasible`
    // flags). A tensor shared by several roots (a param consumed by
    // several contexts' forwards) is ONE node here, so its split is
    // decided exactly once.
    //
    // F(node, d): node produces exactly d.
    struct ExactState {
        bool done = false;
        bool feasible = false;   // a candidate exists whose inputs are all feasible
        int cost = 0;            // sum of the chosen candidate's inputs' ranks
        int cand = -1;
    };

    // G(node, d): node satisfies d (produces some d' and bridges d' -> d).
    struct BestState {
        bool done = false;
        bool feasible = false;   // some producible d' is exact-feasible and bridges to d
        int cost = 0;            // score of the chosen production
        Dist produced;
    };

    // H(node): a trace sink (no in-trace consumer) is producible in some
    // state. Nobody reads a sink, so it needs no bridge and no particular
    // distribution: any exact-feasible production qualifies; the
    // lowest-scored one is the production emit() commits to.
    struct SinkState {
        bool done = false;
        bool feasible = false;
        int cost = 0;            // score of the chosen production
        Dist produced;
    };

    // The collective needed to turn a tensor in `from` into the
    // distribution `to`. The meta backend's only collective is the
    // AllReduce at a PARTIAL subgraph boundary -- there is no
    // AllGather/ReduceScatter/AllToAll, so everything except P -> R is
    // infeasible. A sharded tensor is consumed sharded through the
    // per-op rules; a full tensor is (re-)produced by a row-parallel
    // mul_mat + the implicit AllReduce. A bridge is pure feasibility: it
    // has no score in the preference model.
    static bool bridges(const Dist& from, const Dist& to) {
        return from == to || (from.type == Dist::Type::P && to.type == Dist::Type::R);
    }

    static std::string bridge_name(const Dist& from, const Dist& to) {
        if (from == to) return "None";
        if (from.type == Dist::Type::P && to.type == Dist::Type::R)
            return "AllReduce";
        return "Infeasible";
    }

    // A candidate's score: the sum of the preference ranks of its input
    // distributions (Dist::rank). 0 = every input sharded (a sharded
    // path), larger = more replicated inputs.
    static int candidate_cost(const Candidate& cand) {
        int cost = 0;
        for (const Dist& in : cand.inputs)
            cost += in.rank();
        return cost;
    }

    ExactState& exact(int node, const Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const TraceNode& n = nodes_[node];
        for (int c = 0; c < (int)n.candidates.size(); ++c) {
            const Candidate& cand = n.candidates[c];
            if (cand.output != d) continue;

            bool ok = true;
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                if (!best(n.inputs[i], cand.inputs[i]).feasible) { ok = false; break; }
            }
            if (!ok) continue;
            const int cost = candidate_cost(cand);
            if (!m.feasible || cost < m.cost) {
                m.feasible = true;
                m.cost = cost;
                m.cand = c;
            }
        }
        return m;
    }

    // G(node, d): node satisfies d -- produce some producible d', then
    // bridge. A goal root (an output marked with set_output(), see
    // goal_roots()) must be produced exactly in d: the only bridge
    // (P -> R, the AllReduce) is materialized by the meta backend only at
    // a subgraph boundary before a consumer, and its
    // ggml_backend_meta_buffer_get_tensor (assume_sync = false) has no
    // PARTIAL case, so a PARTIAL output can never be read.
    BestState& best(int node, const Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const bool exact_only = goal_roots_.count(node) != 0;

        // A root is producible only in d (exact-only); every other node in
        // every candidate output.
        std::set<Dist> producible;
        if (exact_only)
            producible.insert(d);
        else
            for (const Candidate& cand : nodes_[node].candidates)
                producible.insert(cand.output);

        for (const Dist& p : producible) {
            if (!bridges(p, d)) continue;
            const ExactState& e = exact(node, p);
            if (!e.feasible) continue;
            if (!m.feasible || e.cost < m.cost) {
                m.feasible = true;
                m.cost = e.cost;
                m.produced = p;
            }
        }
        return m;
    }

    // The trace's sinks: the traced tensors no other traced tensor
    // consumes (goal roots among them included -- plan() plans those as
    // roots, required exactly R).
    std::set<int> compute_sinks() const {
        std::vector<int> consumers(nodes_.size(), 0);
        for (const TraceNode& n : nodes_)
            for (const int in : n.inputs)
                if (in >= 0 && in < (int)consumers.size())
                    ++consumers[in];
        std::set<int> sinks;
        for (int i = 0; i < (int)nodes_.size(); ++i)
            if (consumers[i] == 0)
                sinks.insert(i);
        return sinks;
    }

    // H(node): the lowest-scored production of a trace sink (see SinkState).
    SinkState& sink_state(int node) {
        auto& m = sink_memo_[node];
        if (m.done) return m;
        m.done = true;

        for (const Candidate& cand : nodes_[node].candidates) {
            const ExactState& e = exact(node, cand.output);
            if (!e.feasible) continue;
            if (!m.feasible || e.cost < m.cost) {
                m.feasible = true;
                m.cost = e.cost;
                m.produced = cand.output;
            }
        }
        return m;
    }
    void emit(int node, const Dist& required, Plan& plan, std::set<std::pair<int, Dist>>& emitted) {
        // A tensor consumed several times (even by different roots) in the
        // same distribution is planned once. A tensor needed in two
        // different distributions would get two split states, and plan()
        // rejects that: the meta backend derives exactly one state per
        // tensor.
        if (!emitted.insert({node, required}).second) return;

        Dist produced;
        std::string bridge;
        if (sink_required_.count(node) != 0 && goal_roots_.count(node) == 0) {
            // A trace sink: produced exactly in the state sink_state()
            // chose (passed as `required`). Nobody reads it, so there is
            // no bridge (a bridge would be collective traffic the sink
            // never pays for).
            produced = required;
            bridge = "None";
        } else {
            const BestState& b = best(node, required);
            produced = b.produced;
            bridge = bridge_name(produced, required);
        }

        PlanNode pn;
        pn.id = node;
        pn.op_name = nodes_[node].op_name;
        pn.tensor_name = nodes_[node].is_param ? param_name(node) : "";
        pn.produced = produced;
        pn.required = required;
        pn.bridge = std::move(bridge);
        plan.nodes.push_back(std::move(pn));

        const ExactState& e = exact(node, produced);
        if (e.cand < 0) return;
        const Candidate& cand = nodes_[node].candidates[e.cand];
        const TraceNode& n = nodes_[node];
        for (size_t i = 0; i < cand.inputs.size(); ++i)
            emit(n.inputs[i], cand.inputs[i], plan, emitted);
    }

    std::string infeasibility_reason(const Dist& required) const {
        std::vector<std::string> r;
        for (const TraceNode& n : nodes_) {
            if (n.is_fixed)
                continue;

            // If there are no candidates for node, the plan is trivially infeasible.
            if (n.candidates.empty())
                r.push_back(n.op_name + " (node " + std::to_string(n.id) + ") is not supported by the meta backend (no split-state rule)");

            // flash_attn_ext is only splittable with q/k/v sharded along the
            // sequence axis and the output sharded along the heads axis (S(1)).
            // That S(1) split must be uniform across the devices, so the
            // heads dim (axis 1 of the flash_attn output) must divide by the
            // device count.
            else if (n.op_name == "flash_attn" && n.ne[1] % device_.count() != 0)
                r.push_back(n.op_name + " (node " + std::to_string(n.id) + ") " +
                    "must be sharded along the heads axis (S(1)), but its size " +
                    std::to_string(n.ne[1]) + " is not divisible by the device count " +
                    std::to_string(device_.count()));
        }

        // Infeasibility is because of unknown reason.
        if (r.empty())
            r.push_back("no feasible split plan satisfies the required output distribution " + required.to_string());

        return std::accumulate(std::begin(r), std::end(r), std::string(),
            [](const std::string& acc, const std::string& x) {
                if (acc.empty())
                    return x;

                return acc + "\n" + x;
            }
        );
    }

    // ---------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------
    Runtime& parent_;                     // creates the ggml tensors (context)
    MetaDevice& device_;                  // the shared split-state table + device count
    size_t n_devices_;                    // from the meta device; sizes the sharded candidates

    std::vector<TraceNode> nodes_;
    std::vector<ggml_tensor*> raw_of_;                 // index = trace node id (owned by the parent)
    std::unordered_map<ggml_tensor*, int> raw_to_id_;
    std::set<int> goal_roots_;                         // the outputs marked with set_output() (DP roots)

    // The parameter splits of the last plan(): one entry per shared
    // parameter (the public view via decisions()).
    std::map<const ggml_tensor*, Dist> decisions_;

    std::map<int, std::map<Dist, ExactState>> exact_memo_;
    std::map<int, std::map<Dist, BestState>> best_memo_;
    std::map<int, SinkState> sink_memo_;
    // The production a sink is emitted in (set by plan() before emit()).
    std::map<int, Dist> sink_required_;

    // The splits this runtime committed to the meta device's split table
    // (retired on re-plan and on reset()).
    MetaDevice::Splits splits_;

    std::optional<Plan> last_plan_;

    // The fingerprint of the last successful plan(): the trace's node
    // count and its goal roots. A re-plan with an unchanged fingerprint
    // reuses last_plan_ instead of re-running the DP (repeated allocations
    // with the same trace).
    size_t planned_trace_size_ = 0;
    std::set<int> planned_roots_;
    std::set<int> planned_sinks_;
};
