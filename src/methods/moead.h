#pragma once
#include "moo_optimizer.h"

namespace optimsolution {

// MOEA/D-DE (Zhang & Li, 2007; DE-operator variant per Li & Zhang, 2009).
//
// Decomposes the multi-objective problem into `population` scalar
// Tchebycheff sub-problems, one per weight vector lambda_i (uniformly
// spread over the objective simplex), and optimizes them simultaneously,
// exploiting neighborhood relations between weight vectors: sub-problems
// with similar weight vectors should have similar optimal solutions, so
// each sub-problem's search reuses information from its T nearest
// neighbors (in weight-vector space) via DE-style variation + a limited
// number of neighbor replacements per generation.
//
// Currently supports exactly 2 objectives (weight vectors are the standard
// 1-D simplex sweep lambda_i = (i/(N-1), 1-i/(N-1))); extending to >2
// objectives would need a different weight-vector generator (e.g. the
// systematic simplex-lattice design) and is not implemented here.
//
// Parameters (set via setParam(), all optional -- defaults below):
//   "neighbor_size"   T, neighborhood size,               default 20
//   "de_F"            differential weight for the DE variation, default 0.5
//   "crossover_prob"  DE/SBX-style crossover rate,         default 1.0
//   "mutation_prob"   polynomial mutation probability,     default 1/dim
//   "eta_m"           polynomial mutation distribution index, default 20.0
//   "max_replace"     nr, max neighbor replacements per child, default 2
class MOEAD final : public MOOOptimizer {
public:
    std::string name() const override { return "moead"; }
    MOOResult run() override;
};

} // namespace optimsolution
