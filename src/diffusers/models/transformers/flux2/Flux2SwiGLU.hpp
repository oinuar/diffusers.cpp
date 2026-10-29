#pragma once

#include "nn/Module.hpp"
#include "nn/SiLU.hpp"

class Flux2SwiGLU : public Module {
public:
    Flux2SwiGLU() {
        modules["gate_fn"] = std::make_shared<SiLU>();
    }

    // Reference form: the gate and up projections are fused into one
    // (gate | up) output, which is chunked down the middle.
    Tensor forward(Scope scope, Tensor x) {
        auto chunks = x.chunk(2, -1);
        auto gate_fn = std::static_pointer_cast<SiLU>(modules["gate_fn"]);

        return gate_fn->forward(scope, chunks[0]) * chunks[1];
    }

    // Sharding patch: a windowed view of a column-parallel (S(0))
    // output cannot carry the shard on the meta device (each device
    // holds a different half of the features, so the product would be
    // cross-device), so the feedforward computes the two projections
    // separately and passes the halves in.
    Tensor forward(Scope scope, Tensor gate, Tensor up) {
        auto gate_fn = std::static_pointer_cast<SiLU>(modules["gate_fn"]);

        return gate_fn->forward(scope, gate) * up;
    }
};
