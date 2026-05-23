#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>

namespace optimsolution {

/**
 * CLPSO: Comprehensive Learning Particle Swarm Optimization
 *
 * Implementation notes (framework-aligned):
 * - Population initialization uses Initializer + InitOptions (same pattern as PSO).
 * - The algorithm updates positions every iteration (no "greedy move" on positions).
 *   Personal-best updates remain greedy (standard PSO behavior).
 * - Comprehensive learning:
 *   For each particle i and dimension j, an exemplar index exemplar_[i][j] is selected.
 *   The exemplar provides the target coordinate pbest[exemplar][j].
 * - Refreshing:
 *   If particle i has not improved its pbest for refresh_gap_ iterations, exemplars are re-sampled.
 * - Optional in-run local search is executed only after a pbest improvement (same pattern as PSO/DE).
 * - Optional end polishing is driven only by [global] end_local_refine_ and end_local_method_.
 */
class CLPSO : public Optimizer {
public:
    CLPSO() = default;
    ~CLPSO() override = default;

    std::string methodShortName() const override { return "clpso"; }
    std::string methodFullName()  const override { return "Comprehensive Learning Particle Swarm Optimization"; }

    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using VecD = std::vector<double>;

    double evalAndUpdateBest(const VecD& x);

    void computePcSchedule();
    void selectExemplarsForParticle(int i, bool force_nonself = true);

    void clampVelocity(VecD& v) const;
    void ensureBounds(VecD& x) const;
    void handleBounds(VecD& x, VecD& v) const;

private:
    // State
    std::vector<VecD> X_;
    std::vector<VecD> V_;
    std::vector<double> FX_;

    std::vector<VecD> Pbest_;
    std::vector<double> Fpbest_;

    VecD Vmax_;

    // CLPSO-specific
    std::vector<double> Pc_;                 // per-particle learning probability
    std::vector<int>    no_improve_;          // stagnation counter per particle
    std::vector<std::vector<int>> exemplar_;  // exemplar_[i][j] = index

    // Parameters
    double w_max_{0.9};
    double w_min_{0.4};
    double c_{1.49445};     // learning coefficient
    int    refresh_gap_{7}; // exemplar refresh gap (m)

    double pc_min_{0.05};
    double pc_max_{0.50};
    double pc_exp_{10.0};   // exponent-like shaping used in the standard schedule

    double vmax_frac_{0.2}; // Vmax = (ub-lb) * vmax_frac
    bool   clamp_pos_{true};

    // In-run local (only after pbest improvement)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // End polishing (driven by [global])
    bool        end_local_refine_{false};
    std::string end_local_method_{""};
};

} // namespace optimsolution
