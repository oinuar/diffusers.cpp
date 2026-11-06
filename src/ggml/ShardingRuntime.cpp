#include "ggml/ShardingRuntime.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>

bool ShardingRuntime::use_sharding() const {
    return true;
}

// ShardingRuntime candidate generation: exactly the states the meta backend
// accepts (see the per-op rules in the file header). The DP prices a
// candidate with the sum of its inputs' preference ranks (Dist::rank()):
// sharded (any S) beats partial (P), partial beats replicated (R),
// whenever all of them are feasible.

std::vector<ShardingRuntime::Candidate> ShardingRuntime::param_candidates(int rank) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {}});   // full replica on every device
    for (int a = 0; a < rank; ++a)
        cands.push_back({Dist::shard(a), {}});   // the weight split across devices
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::carry_over_candidates(int rank) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated()}});
    for (int a = 0; a < rank; ++a)
        cands.push_back({Dist::shard(a), {Dist::shard(a)}});
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::binary_candidates(const TraceNode& lhs, const TraceNode& rhs, int out_rank) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated(), Dist::replicated()}});
    for (int a = 0; a < out_rank && a < lhs.rank; ++a) {
        if (rhs.ne[a] == 1) {
            // The 2nd operand's dim a is size 1: it is a broadcast (the
            // meta's handle_bin_bcast keeps the 1st operand's shard and
            // requires the size-1 operand to stay mirrored).
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::replicated()}});
        } else if (a < rhs.rank) {
            // The 2nd operand's dim a is real: the meta's handle_bin_bcast
            // carries the shard only when both operands are sharded along
            // the same axis.
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::shard(a)}});
        }
    }
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::mul_mat_candidates(const TraceNode& w, const TraceNode& a) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated(), Dist::replicated()}});
    if (w.rank >= 2)
        cands.push_back({Dist::shard(0), {Dist::shard(1), Dist::replicated()}});    // column-parallel
    if (a.rank >= 2)
        cands.push_back({Dist::shard(1), {Dist::replicated(), Dist::shard(1)}});    // token-parallel
    if (w.rank >= 1 && a.rank >= 1 && w.ne[0] == a.ne[0])
        cands.push_back({Dist::partial(), {Dist::shard(0), Dist::shard(0)}});       // row-parallel
    return cands;
}
// ============================================================================
// Planning: one DP over the WHOLE trace
// ============================================================================
const ShardingRuntime::Plan& ShardingRuntime::plan() {
    // Solves the WHOLE trace in one go: a single DP over every traced
    // tensor. The goal roots are the outputs the trace marked with
    // set_output() (goal_roots()), each required exactly replicated (R)
    // -- the only state a graph output can be read back in. The roots
    // share the one DP memo, so a tensor shared by several roots (a
    // parameter consumed by several contexts' forwards) is ONE node,
    // planned exactly once, for all of them. The DP is strictly acyclic
    // (topological order), so the memoized recursion terminates.
    //
    // The trace's sinks -- the tensors no other traced tensor consumes --
    // are planned as well: the allocator assigns every tensor of the
    // context to the meta buffer (not only the ones reachable from the
    // goal roots), and the meta backend derives a split state for every
    // tensor it is assigned. A dead end left unplanned keeps its weights
    // at the default MIRRORED split, and the meta's derivation can reach
    // a state the op's rules reject (a dead-end flash_attn hard-requires
    // its q/k/v sharded S(2) -- MIRRORED aborts its assert). Nobody reads
    // a sink, so it is required in no particular state: any state it can
    // be produced in is fine, and no bridge is needed.

    Plan new_plan;
    new_plan.device_count = device_.count();

    const std::set<int>& roots = goal_roots_;
    if (nodes_.empty() || roots.empty()) {
        new_plan.infeasible = true;
        new_plan.infeasible_reason = "the trace has no outputs marked with set_output() (nothing to plan for)";
        last_plan_ = new_plan;
        return *last_plan_;
    }

    // The trace's sinks (see the comment above): they join the goal
    // roots as planning targets, each producible in any state.
    const std::set<int> sinks = compute_sinks();

    // A re-plan with an unchanged trace reuses the last
    // plan (the DP is not re-run): repeated allocations with the same
    // trace do not re-solve it.
    if (last_plan_ && !last_plan_->infeasible && planned_trace_size_ == nodes_.size() &&
        planned_roots_ == roots && planned_sinks_ == sinks) {
            return *last_plan_;
    }

    const Dist root_dist = Dist::replicated();

    // One DP for the whole trace: the single memo is shared by every goal
    // root and sink (a shared tensor is planned once, for all of them).
    exact_memo_.clear();
    best_memo_.clear();
    sink_memo_.clear();
    sink_required_.clear();
    for (const int root : roots) {
        const BestState& root_state = best(root, root_dist);   // roots are exact-only (no bridge)
        if (!root_state.feasible) {
            fprintf(stderr, "DEBUG infeasible root id=%d op=%s ne={%ld,%ld,%ld,%ld} rank=%d\n",
                root, nodes_[root].op_name.c_str(), (long)nodes_[root].ne[0], (long)nodes_[root].ne[1], (long)nodes_[root].ne[2], (long)nodes_[root].ne[3], nodes_[root].rank);
            new_plan.infeasible = true;
            new_plan.infeasible_reason = infeasibility_reason(root_dist);
            last_plan_ = new_plan;
            return *last_plan_;
        }
    }

    // The sinks: producible in some state. The chosen production is
    // recorded so emit() produces it exactly (a sink is never read:
    // no bridge).
    for (const int sink : sinks) {
        if (roots.count(sink))
            continue;   // planned as a goal root (required exactly R)
        const SinkState& s = sink_state(sink);
        if (!s.feasible) {
            new_plan.infeasible = true;
            new_plan.infeasible_reason = "a dead-end subgraph (a tensor no output consumes) cannot be produced in any split state; "
                "the meta backend derives a split state for every tensor of the context, so the dead end must be plannable. " +
                infeasibility_reason(root_dist);
            last_plan_ = new_plan;
            return *last_plan_;
        }
        sink_required_[sink] = s.produced;
    }

    // The plan itself: DFS from every goal root and sink. A tensor is
    // planned (emitted) exactly once, in the distribution its first
    // consumer needs, and every later consumer is served from that same
    // state.
    std::set<std::pair<int, Dist>> emitted;
    for (const int root : roots)
        emit(root, root_dist, new_plan, emitted);
    for (const int sink : sinks) {
        if (roots.count(sink))
            continue;
        emit(sink, sink_required_[sink], new_plan, emitted);
    }

    // The meta backend derives exactly one split state per tensor (its
    // storage layout), so a tensor the plan emitted with two different
    // PRODUCED distributions cannot be planned. (A produced distribution
    // bridged to a different required one is fine: the bridge is a
    // collective at the consumer, not a second storage state.)
    std::map<int, Dist> produced_of;
    for (const PlanNode& pn : new_plan.nodes) {
        auto [it, inserted] = produced_of.insert({pn.id, pn.produced});
        if (!inserted && it->second != pn.produced) {
            new_plan.infeasible = true;
            new_plan.infeasible_reason = nodes_[pn.id].op_name + " (node " + std::to_string(pn.id) +
                ") would need two different storage states (" + it->second.to_string() + " and " +
                pn.produced.to_string() + "), but the meta backend derives a single state per tensor";
            last_plan_ = new_plan;
            return *last_plan_;
        }
    }

    // The ratio-coupled groups (see compute_split_groups): the meta
    // backend derives a sharded compute tensor's per-device sizes from
    // its sharded sources by an exact ratio (and asserts the ratios of
    // all sharded sources of one op agree), so every group of mutually
    // derived tensors must be committed with ONE proportional split.
    // For a non-uniform tensor-split request the per-tensor truncation
    // of the legacy materialize() broke that integrality; each group is
    // solved here for the closest split the ratio rules admit.
    const SplitGroupResult split_groups = compute_split_groups(produced_of);
    if (!split_groups.ok) {
        new_plan.infeasible = true;
        new_plan.infeasible_reason = split_groups.reason;
        last_plan_ = new_plan;
        return *last_plan_;
    }

    // The parameter splits: one entry per shared parameter (the meta
    // backend derives the compute tensors' states). Retires the splits of
    // the previous plan (its raw tensors may have been re-created by a
    // re-trace) and commits the new ones to the meta device's split table
    // -- the shared resource, global across contexts and allocators. The
    // plan's score: every planned tensor's production is paid exactly
    // once, plus its P -> R bridge.
    for (const auto& [t, st] : splits_)
        device_.splits().erase(t);
    splits_.clear();

    decisions_.clear();
    new_plan.callback_states.clear();

    int cost = 0;
    for (const PlanNode& pn : new_plan.nodes) {
        /*if (nodes_[pn.id].is_param)*/ {
            const ggml_tensor* raw = raw_of_[pn.id];

            decisions_[raw] = pn.produced;
            const auto git = split_groups.splits.find(pn.id);
            const ggml_backend_meta_split_state st = materialize(
                pn.produced, nodes_[pn.id].ne, raw->type,
                git != split_groups.splits.end() ? &git->second : nullptr);
            new_plan.callback_states[param_name(pn.id)] = st;
            device_.splits()[raw] = st;
            splits_[raw] = st;
        }

        const ExactState& e = exact(pn.id, pn.produced);
        if (e.cand >= 0)
            cost += candidate_cost(nodes_[pn.id].candidates[e.cand]);
        if (pn.bridge != "None")
            cost += 1;
    }
    new_plan.total_cost = cost;

    planned_trace_size_ = nodes_.size();
    planned_roots_ = roots;
    planned_sinks_ = sinks;
    last_plan_ = new_plan;

    return *last_plan_;
}

std::string ShardingRuntime::dump_trace() const {
    std::ostringstream ss;
    for (const TraceNode& n : nodes_) {
        ss << "  [" << n.id << "] " << n.op_name;
        if (n.is_fixed) ss << " (fixed R)";
        if (n.is_param) ss << " " << (raw_of_[n.id]->name[0] ? raw_of_[n.id]->name : "?");
        ss << " ne={" << n.ne[0];
        for (int i = 1; i < n.rank; ++i) ss << ", " << n.ne[i];
        ss << "} in={";
        for (size_t i = 0; i < n.inputs.size(); ++i)
            ss << (i ? ", " : "") << n.inputs[i];
        ss << "} candidates:";
        for (const Candidate& c : n.candidates)
            ss << " " << c.output.to_string();
        if (n.candidates.empty())
            ss << " (none -- unsupported by the meta backend)";
        ss << "\n";
    }
    return ss.str();
}

// ============================================================================
// Ratio-coupled split groups
// ============================================================================
// The meta backend derives a sharded compute tensor's per-device sizes
// from the committed sizes of its sharded sources by an EXACT ratio --
//   ne_j = out_ne[axis] * (src_ne_field_j * src_nr) / (src_ne[src_axis] * nr)
// (the "take over ratio" block of ggml-backend-meta.cpp, GGML_ASSERTed
// integral) -- and asserts that every two sharded sources of one op agree
// on that ratio (a row-parallel mul_mat requires the two states equal).
// By induction over the derivation chain, every tensor that derives (or
// is derived through a chain) from a committed param T owns exactly
//   share_j = N_T * s_j / N
// elements on device j (s = the param's per-device split, N = its sharded
// size, N_T = the tensor's own sharded size): the whole "ratio group" of
// mutually derived tensors therefore moves together with ONE proportional
// split.
//
// The integrality constraints a group split s (summing to N, the anchor's
// sharded size) must satisfy:
//   (a) N_T * s_j / (N * nr_T) integral for every member T and device j
//       (the meta's ratio assert; nr_T is the nr of T's derived state)
//   (b) N_T * s_j / N a multiple of the ggml block size for every
//       committed param T sharded on axis 0 (the meta's static-tensor
//       assert; nr_T = 1 for params)
//   (c) sum_j s_j == N (the meta's static-tensor assert)
//
// (a) and (b) hold for every j iff s_j is a multiple of
//   L = lcm over the members of
//   N * nr_T   / gcd(N_T, N * nr_T)          (a)
//   N * gran_T / gcd(N_T, N * gran_T)        (b, axis-0 params only)
// so every working split is s_j = L * m_j with sum m_j == M == N / L (the
// group is infeasible when L does not divide N). The requested factors f
// ask for s_j ~ N * f_j / sum f, i.e. m_j ~ c_j == M * f_j / sum f; the
// objective (the total L1 distance of every member's per-device size from
// the requested proportion) is separable in j with a weight common to all
// members, so its exact optimum is the standard floor-and-remainder
// rounding of c to the integer vector m summing to M:
//   m_j = floor(c_j), then hand out the M - sum floor(c_j) remaining
//   units one by one to the device with the largest fractional part
//   (ties: the later device, as the legacy even split does).
ShardingRuntime::SplitGroupResult ShardingRuntime::compute_split_groups(const std::map<int, Dist>& produced_of) {
    SplitGroupResult res;
    const int n_dev = (int)device_.count();

    double sum_f = 0.0;
    for (int j = 0; j < n_dev; ++j)
        sum_f += device_.tensor_split(j);

    // --- The ratio coupling (union-find over the traced nodes) ---------
    // Two traced tensors are ratio-coupled when the meta must derive one's
    // per-device shard sizes from the other's:
    //  - a sharded output O of an op and each of the op's sharded sources:
    //    O's sizes are taken over by ratio from the first sharded source;
    //  - every pair of sharded sources of the same op: the meta asserts
    //    their ratios agree (a row-parallel mul_mat requires them equal).
    const int n_nodes = (int)nodes_.size();
    std::vector<int> parent(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
        parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x)
            parent[x] = parent[parent[x]], x = parent[x];
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b)
            parent[a] = b;
    };

    for (const auto& [id, d] : produced_of) {
        if (d.type != Dist::Type::S && d.type != Dist::Type::P)
            continue;
        const ExactState& e = exact(id, d);
        if (e.cand < 0)
            continue;
        const TraceNode& node = nodes_[id];
        const Candidate& cand = node.candidates[e.cand];

        int first = -1;
        for (size_t i = 0; i < cand.inputs.size(); ++i) {
            if (cand.inputs[i].type != Dist::Type::S)
                continue;
            if (first < 0)
                first = node.inputs[i];
            else
                unite(first, node.inputs[i]);
        }
        if (d.type == Dist::Type::S && first >= 0)
            unite(id, first);
    }

    // The members of every group: the planned nodes with a sharded
    // production. (A PARTIAL production owns no split: its op's sharded
    // sources are coupled through the node above.)
    std::map<int, std::vector<int>> groups;
    for (const auto& [id, d] : produced_of) {
        if (d.type != Dist::Type::S)
            continue;
        groups[find(id)].push_back(id);
    }

    // The nr the meta derives for a member's state: 1 for a committed
    // param; preserved through the carry-over ops (norm, rope, permute,
    // concat, ...); set by a reshape/cont/cpy (and a contiguous view) to
    // base_out / base_in (or 1) exactly as the meta's handle_reshape
    // does; 1 for a column-parallel mul_mat output and a flash_attn
    // output, carried from the sharded activation by a token-parallel
    // mul_mat.
    std::map<int, int64_t> nr_memo;
    std::function<int64_t(int)> nr_of = [&](int t) -> int64_t {
        auto it = nr_memo.find(t);
        if (it != nr_memo.end())
            return it->second;

        const TraceNode& node = nodes_[t];
        int64_t nr = 1;
        if (!node.is_param) {
            const Dist& d = produced_of.at(t);
            const ExactState& e = exact(t, d);
            if (e.cand >= 0) {
                const Candidate& cand = node.candidates[e.cand];
                int si = -1;
                int ai = -1;
                for (size_t i = 0; i < cand.inputs.size(); ++i) {
                    if (cand.inputs[i].type == Dist::Type::S) {
                        si = node.inputs[i];
                        ai = cand.inputs[i].axis;
                        break;   // the first sharded source in src order
                    }
                }
                if (si >= 0) {
                    const TraceNode& src = nodes_[si];
                    if (node.op_name == "mul_mat") {
                        // Column-parallel (the sharded weight is src 0)
                        // starts a fresh nr = 1 state; token-parallel
                        // (the sharded activation is src 1) carries the
                        // activation's state, nr included.
                        nr = (si == node.inputs[0]) ? 1 : nr_of(si);
                    } else if (node.op_name == "flash_attn") {
                        nr = 1;
                    } else if (node.op_name == "reshape" || node.op_name == "cont" || node.op_name == "cpy" ||
                               node.op_name == "view") {
                        const ggml_tensor* out = raw_of_[t];
                        const ggml_tensor* src_t = raw_of_[si];
                        if (ggml_is_contiguous(out) && ggml_is_contiguous(src_t)) {
                            // The handle_reshape rule (also for a
                            // contiguous view).
                            if (ai == n_dims(src.ne) - 1 && nr_of(si) == 1) {
                                nr = 1;   // the last-dim fast path
                            } else {
                                int64_t base_in = 1;
                                for (int i = 0; i <= ai; ++i)
                                    base_in *= src.ne[i];
                                base_in /= nr_of(si);
                                int64_t base_out = 1;
                                for (int i = 0; i <= d.axis; ++i)
                                    base_out *= node.ne[i];
                                nr = (base_out % base_in == 0) ? base_out / base_in : 1;
                            }
                        } else {
                            // A stride-matching view (e.g. a dim-0 slice)
                            // carries the source state over, nr included;
                            // the remaining unpermuted stride-remap branch
                            // starts a fresh nr = 1 state.
                            bool same_strides = true;
                            for (int i = 0; i < 4; ++i) {
                                if (out->ne[i] == 1 && src_t->ne[i] == 1)
                                    continue;
                                if (out->nb[i] != src_t->nb[i]) {
                                    same_strides = false;
                                    break;
                                }
                            }
                            nr = same_strides ? nr_of(si) : 1;
                        }
                    } else {
                        nr = nr_of(si);   // the carry-over ops
                    }
                }
            }
        }
        return nr_memo[t] = nr;
    };

    // --- Solve every group ----------------------------------------------
    for (const auto& [root, members] : groups) {
        // The anchor: the member with the largest sharded size (the base
        // split s sums to its size).
        int anchor = members[0];
        for (int m : members)
            if (nodes_[m].ne[produced_of.at(m).axis] > nodes_[anchor].ne[produced_of.at(anchor).axis])
                anchor = m;
        const int64_t N = nodes_[anchor].ne[produced_of.at(anchor).axis];

        // L: the step every s_j must be a multiple of (see above).
        int64_t L = 1;
        for (int m : members) {
            const int64_t N_m = nodes_[m].ne[produced_of.at(m).axis];
            const int64_t nr_m = nr_of(m);
            if (N_m % nr_m != 0) {
                res.ok = false;
                res.reason = "tensor '" + param_name(m) + "' (sharded on axis " + std::to_string(produced_of.at(m).axis) +
                    ", size " + std::to_string(N_m) +
                    ") derives a strided split state (nr = " + std::to_string(nr_m) +
                    ") whose repeat count does not divide its sharded size; no tensor split keeps the "
                    "per-device shard sizes integral across this ratio-coupled group";
                return res;
            }
            const int64_t term = N * nr_m / std::gcd(N_m, N * nr_m);
            L = L / std::gcd(L, term) * term;
            if (nodes_[m].is_param && produced_of.at(m).axis == 0) {
                const int64_t gran = ggml_blck_size(raw_of_[m]->type);
                if (N_m % gran != 0) {
                    res.ok = false;
                    res.reason = "tensor '" + param_name(m) + "' has a sharded axis-0 size (" + std::to_string(N_m) +
                        ") that is not a multiple of its ggml block size (" + std::to_string(gran) +
                        "); it cannot be split into block-aligned shards across the " + std::to_string(n_dev) + " devices";
                    return res;
                }
                const int64_t gterm = N * gran / std::gcd(N_m, N * gran);
                L = L / std::gcd(L, gterm) * gterm;
            }
        }
        if (N % L != 0) {
            res.ok = false;
            res.reason = "the ratio-coupled group containing tensor '" + param_name(anchor) + "' (sharded size " +
                std::to_string(N) +
                ") has no tensor split that keeps every derived per-device shard size integral across the " +
                std::to_string(n_dev) + " devices";
            return res;
        }
        const int64_t M = N / L;

        // m: the closest integer split of M to the requested factors.
        std::vector<int64_t> m(n_dev, 0);
        std::vector<double> frac(n_dev, 0.0);
        int64_t sum_m = 0;
        for (int j = 0; j < n_dev; ++j) {
            const double c = (sum_f > 0.0)
                ? (double)M * device_.tensor_split(j) / sum_f
                : (double)M / n_dev;
            m[j] = (int64_t)c;   // c >= 0: floor
            frac[j] = c - (double)m[j];
            sum_m += m[j];
        }
        // Hand out the remaining units one by one to the device with the
        // largest fractional part (ties: the later device, so an even
        // request gives the last device the remainder, as before).
        while (sum_m < M) {
            int j = 0;
            for (int i = 1; i < n_dev; ++i)
                if (frac[i] > frac[j] || (frac[i] == frac[j] && i > j))
                    j = i;
            m[j]++;
            frac[j] -= 1.0;   // a unit already granted drops out of the priority
            sum_m++;
        }

        // Commit every member's per-device sizes (a param's committed
        // split; a compute tensor's derived one, recorded for the plan's
        // callback table).
        for (int m_id : members) {
            const int64_t N_m = nodes_[m_id].ne[produced_of.at(m_id).axis];
            std::vector<int64_t> split(n_dev);
            for (int j = 0; j < n_dev; ++j)
                split[j] = N_m * (L * m[j]) / N;
            res.splits[m_id] = std::move(split);
        }
    }
    return res;
}

