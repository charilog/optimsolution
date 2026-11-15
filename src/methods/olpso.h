#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  OLPSO -- Orthogonal Learning Particle Swarm Optimization
//
//  Reference:
//    Z.-H. Zhan, J. Zhang, Y. Li, Y.-H. Shi, "Orthogonal Learning Particle
//    Swarm Optimization", IEEE Transactions on Evolutionary Computation
//    15(6) (2011) 832-847 (doi:10.1109/TEVC.2010.2052054).
//
//  Each particle learns from ONE guidance vector Po built by orthogonal
//  experimental design (OED) on two exemplars, its own pbest Pi and a
//  neighbourhood best Pn:
//         v_d = w v_d + c r_d (Po_d - x_d) ,   x = x + v
//  OED (paper, Sec. III and Appendix):
//   1. Orthogonal array L_M(2^(M-1)), M = 2^ceil(log2(D+1)); the first D
//      columns are used; level 1 -> dimension taken from Pi, level 2 -> Pn.
//   2. The M test vectors are evaluated; Xb = best of them.
//   3. Factor analysis: for every dimension the level with the smaller mean
//      fitness is chosen, giving the predicted vector Xp (evaluated).
//   4. Po = level combination of the better of Xb and Xp.
//  Po stores LEVELS (which of Pi / Pn supplies each dimension), so later
//  improvements of Pi or Pn are used immediately. Po is rebuilt only after the
//  particle's pbest has not improved for G (= 5) consecutive generations.
//  If Pi and Pn are the same particle, a random other particle Pr is used.
//
//  Variants ("topology" key):
//    local  (default) OLPSO-L: Pn = best pbest in the ring {i-1, i, i+1}
//    global           OLPSO-G: Pn = global best pbest
//  partner_mode: fixed (default) -- the partner particle of Po is fixed when
//  Po is built; dynamic -- Pn is re-resolved every generation as the current
//  neighbourhood best (a random partner Pr stays fixed). The paper does not
//  say which; both performed alike on CEC 2017 in our tests.
//
//  Paper settings: N = 40, w 0.9 -> 0.4 (linear), c = 2.0, Vmax = 0.2*range,
//  G = 5. Out-of-range particles are not evaluated and do not update their
//  pbest (paper, Sec. II; they are drawn back by Po): bound_mode = skip.
//  clamp / reflect are available (include/pso_family_common.h).
//
//  Implementation notes:
//   * The OA's first row is all level 1, i.e. Pi itself, whose fitness is
//     already known: it is not evaluated again (identical result, saves one
//     evaluation per OED). Each OED therefore costs at most M evaluations.
//   * Test vectors are not written back into pbest (the paper uses them only
//     to build Po); they do update the reported best-so-far solution.
//   * OED evaluations count towards max_evals; an OED interrupted by the
//     budget uses the tests evaluated so far.
// =============================================================================
class OLPSO : public Optimizer {
public:
    OLPSO() = default;
    ~OLPSO() override = default;

    std::string methodShortName() const override { return "olpso"; }
    std::string methodFullName()  const override { return "Orthogonal Learning Particle Swarm Optimization"; }

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
    void buildOrthogonalArray(int D);
    int  neighbourBest(int i) const;
    void constructPo(int i);
    void maybeLocalSearch(int i);

private:
    std::vector<Vec>    X_, V_, P_;
    std::vector<double> FX_, FP_;
    std::vector<std::vector<unsigned char>> level_; // level_[i][d]: 0 -> Pi, 1 -> partner
    std::vector<int>    partner_;                   // partner particle of Po_i
    std::vector<char>   random_partner_;            // 1 if partner_ is a random Pr (Pi == Pn)
    std::vector<int>    stay_;
    std::vector<std::vector<unsigned char>> OA_;    // M x D orthogonal array (0/1)
    Vec                 Vmax_;
    int                 g_{0};

    // Parameters
    int    pop_cfg_{0};          // 0 = paper default 40
    double w_start_{0.9}, w_end_{0.4};
    double c_{2.0};
    double vmax_frac_{0.2};
    int    G_{5};                // stagnation gap for rebuilding Po
    bool   global_topology_{false};
    bool   dynamic_partner_{false};

    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Skip};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
