#pragma once
#include "moo_optimizer.h"

namespace optimsolution {

// NSGA-II (Deb et al., 2002) -- the standard baseline multi-objective
// evolutionary algorithm. Uses fast non-dominated sorting + crowding
// distance for environmental selection, SBX crossover, and polynomial
// mutation. Works with any number of objectives (>=2) exposed via
// Problem::evaluateMulti().
//
// Parameters (set via setParam(), all optional -- sensible defaults below):
//   "crossover_prob"  in [0,1], default 0.9
//   "mutation_prob"   in [0,1], default 1/dim (set at run() time if unset)
//   "eta_c" (SBX distribution index),          default 15.0
//   "eta_m" (polynomial mutation distribution), default 20.0
class NSGA2 final : public MOOOptimizer {
public:
    std::string name() const override { return "nsga2"; }
    MOOResult run() override;
};

} // namespace optimsolution
