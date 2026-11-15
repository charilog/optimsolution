#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  HCLPSO -- Heterogeneous Comprehensive Learning Particle Swarm Optimization
//
//  Reference:
//    N. Lynn, P. N. Suganthan, "Heterogeneous comprehensive learning particle
//    swarm optimization with enhanced exploration and exploitation",
//    Swarm and Evolutionary Computation 24 (2015) 11-24
//    (doi:10.1016/j.swevo.2015.05.002).
//
//  The swarm (default 40 particles) is split into two subpopulations that both
//  use comprehensive-learning (CL) exemplars:
//
//   * Exploration group (first g1 = 15 particles): exemplars are drawn ONLY
//     from the pbests of the exploration group itself, and there is no
//     global-best term:
//         v = w v + K r (pbest_fi(d) - x)
//   * Exploitation group (remaining g2 = 25 particles): exemplars are drawn
//     from the pbests of the WHOLE swarm, plus a global-best term:
//         v = w v + c1 r1 (pbest_fi(d) - x) + c2 r2 (gbest - x)
//
//  Linear schedules over the run (progress t in [0,1], evaluation-based):
//         w  : 0.99 -> 0.2      K  : 3.0 -> 1.5
//         c1 : 2.5  -> 0.5      c2 : 0.5 -> 2.5
//
//  CL exemplar construction (authors' code): for each dimension, with
//  probability Pc_i the coordinate of the better of two randomly chosen pbests
//  of the allowed pool, otherwise the particle's own pbest coordinate; at
//  least one dimension is forced to learn from a friend. The exemplar is a
//  SNAPSHOT of those coordinates, rebuilt only once the particle's pbest has
//  failed to improve for MORE than refresh_gap (= 5) consecutive generations
//  (reference: "> 5"; a generation in which the particle was not evaluated
//  counts as not improved).
//
//  Learning probabilities (per particle index i over the whole swarm):
//      Pc_i = pc_min + (pc_max - pc_min) * (exp(pc_exp*t_i) - 1)/(exp(pc_exp)-1),
//      t_i = i/(N-1). Defaults pc_min = 0, pc_max = 0.25, pc_exp = 10 follow
//  the authors' released code (as ported e.g. in NEORL), not CLPSO's
//  0.05..0.5 range; both are configurable from [hclpso].
//
//  Velocity clamp Vmax = 0.2 * (ub - lb), initial velocities U(-Vmax, Vmax).
//  Synchronous generations as in the released code: all particles move with
//  the previous generation's gbest, then pbests, gbest and exemplars update.
//  Box handling (bound_mode, default skip = released code): particles that
//  leave the box are not evaluated and keep flying until they return.
//  clamp / reflect are available as alternatives (include/pso_family_common.h).
// =============================================================================
class HCLPSO : public Optimizer {
public:
    HCLPSO() = default;
    ~HCLPSO() override = default;

    std::string methodShortName() const override { return "hclpso"; }
    std::string methodFullName()  const override { return "Heterogeneous Comprehensive Learning Particle Swarm Optimization"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    bool tryEval(const Vec& x, double& f);
    void buildExemplar(int i);
    void maybeLocalSearch(int i);

private:
    std::vector<Vec>    X_, V_, P_;
    std::vector<double> FX_, FP_;
    std::vector<Vec>    E_;      // exemplar coordinates (snapshot)
    std::vector<int>    stay_;
    std::vector<double> Pc_;
    Vec                 Vmax_;
    int                 g_{0};   // index of best pbest
    int                 g1_{0};  // effective size of the exploration group

    // Parameters
    int    pop_cfg_{0};          // 0 = paper default 40
    int    g1_cfg_{0};           // 0 = round(N * expl_ratio)
    double expl_ratio_{15.0 / 40.0};
    double w_start_{0.99}, w_end_{0.2};
    double k_start_{3.0},  k_end_{1.5};
    double c1_start_{2.5}, c1_end_{0.5};
    double c2_start_{0.5}, c2_end_{2.5};
    double pc_min_{0.0}, pc_max_{0.25}, pc_exp_{10.0};
    int    refresh_gap_{5};
    double vmax_frac_{0.2};
    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Skip};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
