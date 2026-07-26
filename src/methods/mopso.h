#pragma once
#include "moo_optimizer.h"

namespace optimsolution {

// MOPSO (Coello Coello & Lechuga, 2002, "MOPSO: A Proposal for Multiple
// Objective Particle Swarm Optimization"), using the common crowding-
// distance-based leader-selection variant (as in later refinements such as
// OMOPSO) instead of the original paper's adaptive hypercube grid, since
// it is simpler and does not need a grid resolution parameter.
//
// A swarm of particles moves through the search space guided by its own
// personal best AND a "leader" drawn from a bounded external archive of
// non-dominated solutions found so far; leaders are chosen preferentially
// from LESS crowded regions of the archive (via crowding distance) to keep
// the front spread out rather than clumped.
//
// Parameters (set via setParam(), all optional -- defaults below):
//   "inertia_w"       velocity inertia weight w,           default 0.4
//   "c1"              personal-best attraction coefficient, default 1.5
//   "c2"              leader attraction coefficient,        default 1.5
//   "archive_size"    max external archive size,            default = population
//   "mutation_prob"   per-dimension "re-randomize" probability (light
//                     diversity-preserving perturbation), default 0.05
class MOPSO final : public MOOOptimizer {
public:
    std::string name() const override { return "mopso"; }
    MOOResult run() override;
};

} // namespace optimsolution
