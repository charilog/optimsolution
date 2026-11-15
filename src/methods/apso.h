#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  APSO -- Adaptive Particle Swarm Optimization
//
//  Reference:
//    Z.-H. Zhan, J. Zhang, Y. Li, H. S.-H. Chung, "Adaptive Particle Swarm
//    Optimization", IEEE Transactions on Systems, Man, and Cybernetics,
//    Part B 39(6) (2009) 1362-1381 (doi:10.1109/TSMCB.2009.2015956).
//
//  Global-best PSO, v = w v + c1 r1 (p - x) + c2 r2 (g - x), whose w, c1, c2
//  are adapted every generation by Evolutionary State Estimation (ESE):
//
//   * Mean distance of particle i to all others (Eq. 7):
//        d_i = 1/(N-1) * sum_{j != i} ||x_i - x_j||
//     Evolutionary factor (Eq. 8): f = (d_g - d_min) / (d_max - d_min),
//     d_g = mean distance of the globally best particle.
//   * Fuzzy classification (Eq. 9) into S1 exploration, S2 exploitation,
//     S3 convergence, S4 jumping-out, with the membership functions of the
//     paper and singleton defuzzification. When f belongs to two states the
//     paper's rule base (sequence S1 => S2 => S3 => S4 => S1, previous state
//     taken into account) is applied as:
//        1. previous state is a candidate          -> keep it
//        2. successor of previous is a candidate   -> move forward
//        3. predecessor of previous is a candidate -> step back
//        4. otherwise                              -> largest membership
//     This reproduces every case stated in the paper (e.g. f in S1/S2:
//     prev S1 -> S1, prev S2 or S3 -> S2, prev S4 -> S1). Initial state: S1.
//   * w(f) = 1 / (1 + 1.5 exp(-2.6 f))  (in [0.4, 0.9]).
//   * c1/c2 (start 2.0/2.0) change by delta ~ U(0.05, 0.1) per generation:
//        S1: c1 += d,  c2 -= d          S2: c1 += d/2, c2 -= d/2
//        S3: c1 += d/2, c2 += d/2       S4: c1 -= d,  c2 += d
//     clamped to [1.5, 2.5]; if c1 + c2 > 4 both are rescaled to sum 4.
//   * Elitist learning (ELS), only in S3: one random dimension d of the
//     global best P is perturbed,
//        P_d += (ub_d - lb_d) * N(0, sigma^2),
//        sigma = sigma_max - (sigma_max - sigma_min) * progress,
//     sigma_max = 1.0, sigma_min = 0.1. If better than gbest it replaces it,
//     otherwise it replaces the worst particle.
//
//  Paper settings: N = 20, Vmax = 0.2*range.
//  Implementation choices (not fixed by the paper): box handling
//  bound_mode = skip (out-of-range particles are not evaluated, the
//  convention of the same authors' OLPSO paper; it was also the best of
//  skip/clamp/reflect on CEC 2017 in our tests); delta is drawn independently
//  for c1 and c2; d_g uses the current position of the particle that owns the
//  global best; the ELS candidate is clipped to the box; a worst particle
//  replaced by ELS also updates its pbest if the new point is better.
//  Note: as soon as the swarm gathers around gbest, d_g is the smallest mean
//  distance, so f ~ 0 and APSO spends most generations in S3 (w = 0.4, ELS
//  every generation) -- this is the behaviour implied by Eqs. 7-9.
// =============================================================================
class APSO : public Optimizer {
public:
    APSO() = default;
    ~APSO() override = default;

    std::string methodShortName() const override { return "apso"; }
    std::string methodFullName()  const override { return "Adaptive Particle Swarm Optimization"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    void configure(const MethodConfig& mc) override;

    // Exposed for testing/diagnostics: classify evolutionary factor f given
    // the previous state (1..4). Pure function of its arguments.
    static int classifyState(double f, int prev_state);

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    bool   tryEval(const Vec& x, double& f);
    double evolutionaryFactor() const;
    void   adaptParameters(int state, double f);
    void   elitistLearning();
    void   maybeLocalSearch(int i);

private:
    std::vector<Vec>    X_, V_, P_;
    std::vector<double> FX_, FP_;
    Vec                 Vmax_;
    int                 g_{0};

    int    state_{1};
    double w_{0.9};
    double c1_{2.0}, c2_{2.0};

    // Parameters
    int    pop_cfg_{0};        // 0 = paper default 20
    double c_init_{2.0};
    double c_min_{1.5}, c_max_{2.5}, c_sum_max_{4.0};
    double delta_min_{0.05}, delta_max_{0.1};
    double sigma_max_{1.0}, sigma_min_{0.1};
    double vmax_frac_{0.2};
    bool   use_els_{true};

    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Skip};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
