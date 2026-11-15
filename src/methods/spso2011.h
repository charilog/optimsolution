#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  SPSO-2011 -- Standard Particle Swarm Optimisation 2011
//
//  References:
//    M. Clerc, "Standard Particle Swarm Optimisation -- From 2006 to 2011",
//      technical report, hal-00764996 (2012).
//    M. Zambrano-Bigiarini, M. Clerc, R. Rojas, "Standard Particle Swarm
//      Optimisation 2011 at CEC-2013: A baseline for future PSO improvements",
//      IEEE CEC 2013, pp. 2337-2344 (doi:10.1109/CEC.2013.6557848).
//
//  This is the official "baseline" PSO used to benchmark PSO variants on the
//  CEC-2013 suite. Main differences from the classic (Kennedy-Eberhart/Shi)
//  PSO already available as "pso":
//
//   * Adaptive random topology: every particle informs itself and K (=3)
//     particles chosen at random (with replacement). The links are re-drawn
//     after every iteration in which the best-so-far fitness did NOT improve.
//   * Rotation-invariant move. With p = own best, l = best previous position
//     among the particle's informants, x = current position:
//         G  = x + c * (p + l - 2x) / 3          (if l != p)
//         G  = x + c * (p - x) / 2               (if the particle is its own
//                                                 best informant)
//         x' = random point in the hypersphere H(G, ||G - x||): uniform
//              direction and uniform radius r = U(0, ||G - x||) (Clerc's
//              default; denser near G, not uniform in volume)
//         v  = w v + x' - x ;   x = x + v
//     with w = 1/(2 ln 2) ~ 0.721 and c = 1/2 + ln 2 ~ 1.193.
//   * Velocity initialisation v_d = U(lb_d - x_d, ub_d - x_d).
//   * Confinement (bound_mode = reflect): a coordinate leaving the box is
//     set to the bound and its velocity component becomes -0.5 * v_d.
//   * Asynchronous update in a random particle order (the C reference code's
//     default), swarm size S = 40.
//
//  Framework integration (same conventions as pso/clpso/cso):
//   * [init] options are honoured for the initial positions.
//   * Every evaluation is checked against max_evals (no overshoot).
//   * Optional in-run local search after a pbest improvement
//     (local_rate / local_method), optional end-of-run refinement from
//     [global] end_local_refine / end_local_method.
// =============================================================================
class SPSO2011 : public Optimizer {
public:
    SPSO2011() = default;
    ~SPSO2011() override = default;

    std::string methodShortName() const override { return "spso2011"; }
    std::string methodFullName()  const override { return "Standard Particle Swarm Optimisation 2011"; }

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
    void buildTopology();
    int  bestInformant(int i) const;
    void randomInSphere(const Vec& center, double radius, Vec& out);
    void maybeLocalSearch(int i);

private:
    std::vector<Vec>    X_, V_, P_;
    std::vector<double> FX_, FP_;
    std::vector<std::vector<int>> informants_; // informants_[i] = particles that inform i (incl. i)
    int g_{0};                                  // index of the best pbest

    // Parameters (Clerc 2012)
    int    pop_cfg_{0};       // 0 = paper default S = 40
    int    K_{3};             // number of informed particles per particle
    double w_{0.0};           // <=0 -> 1/(2 ln 2)
    double c_{0.0};           // <=0 -> 0.5 + ln 2
    bool   rand_order_{true}; // random permutation of particle order each iteration
    bool   volume_uniform_{false}; // false = Clerc default (uniform radius), true = uniform in volume

    // In-run local search (only after a pbest improvement)
    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Reflect};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // End-of-run refinement ([global] or [spso2011])
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
