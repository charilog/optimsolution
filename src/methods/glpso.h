#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  GL-PSO -- Genetic Learning Particle Swarm Optimization
//
//  Reference:
//    Y.-J. Gong, J.-J. Li, Y. Zhou, Y. Li, H. S.-H. Chung, Y.-H. Shi, J. Zhang,
//    "Genetic Learning Particle Swarm Optimization", IEEE Transactions on
//    Cybernetics 46(10) (2016) 2277-2290 (doi:10.1109/TCYB.2015.2475174).
//
//  Two cascading layers per particle and generation (paper, Algorithm 1):
//
//   1. Exemplar breeding (genetic layer) from the historical information:
//        crossover (Eq. 5), per dimension d with a random particle k_d and
//        r_d ~ U(0,1):
//            o_d = r_d * p_{i,d} + (1 - r_d) * g_d   if f(P_i) < f(P_{k_d})
//            o_d = p_{k_d,d}                         otherwise
//        mutation (Eq. 6): with probability pm, o_d ~ U(lb_d, ub_d)
//        selection (Eq. 7): O_i is evaluated; E_i <- O_i if f(O_i) < f(E_i)
//        if E_i has not improved for sg generations, it is replaced by the
//        best of 20%*N exemplars picked at random (tournament).
//   2. PSO layer, learning from the exemplar only:
//            v_d = w v_d + c r_d (e_{i,d} - x_d) ,  x = x + v
//
//  Paper settings: N = 50, w = 0.7298, c = 1.49618, pm = 0.01, sg = 7,
//  tournament 20% of N. Two evaluations per particle and generation.
//
//  Points the paper leaves open follow the MetaBox implementation of GL-PSO
//  (based on the authors' official code):
//   * Initial exemplars: one bred offspring per particle, evaluated and
//     accepted unconditionally (init_exemplar_offspring = 1; 0 = pbests).
//   * The tournament fires when the exemplar's stagnation counter EXCEEDS sg.
//   * Vmax = 0.2*range; box handling bound_mode = reflect: a coordinate that
//     leaves the box is clipped and its velocity becomes -0.5*v.
//  Own choices: the tournament draws 20%*N distinct exemplars (the current
//  one may be among them), the winner is copied with its fitness, and the
//  stagnation counter is reset; g = best pbest.
// =============================================================================
class GLPSO : public Optimizer {
public:
    GLPSO() = default;
    ~GLPSO() override = default;

    std::string methodShortName() const override { return "glpso"; }
    std::string methodFullName()  const override { return "Genetic Learning Particle Swarm Optimization"; }

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
    void maybeLocalSearch(int i);
    void breed(int i, Vec& o);

private:
    std::vector<Vec>    X_, V_, P_, E_;
    std::vector<double> FX_, FP_, FE_;
    std::vector<int>    stag_;
    Vec                 Vmax_;
    int                 g_{0};

    // Parameters
    int    pop_cfg_{0};       // 0 = paper default 50
    double w_{0.7298};
    double c_{1.49618};
    double pm_{0.01};
    int    sg_{7};
    double tour_frac_{0.2};
    double vmax_frac_{0.2};
    bool   init_exemplar_offspring_{true};
    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Reflect};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
