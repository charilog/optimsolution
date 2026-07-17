#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig;

// CSO -- Competitive Swarm Optimizer.
// Reference: Cheng, R.; Jin, Y. "A Competitive Swarm Optimizer for Large
//   Scale Optimization." IEEE Transactions on Cybernetics, 45(2), 191-204,
//   2015.
//
// CSO departs deliberately from classic PSO: there is no global best and
// no personal best at all, which removes both the premature-convergence
// pressure those introduce and the O(D) memory each personal-best vector
// costs per particle -- exactly the two things that make plain PSO scale
// poorly to very high dimensions. Instead, each generation:
//
//   1. The swarm's mean position x_bar(t) is computed once.
//   2. The N particles are randomly shuffled and split into N/2 pairs.
//   3. Each pair holds a simple fitness "competition": the WINNER (better
//      fitness) carries over to the next generation completely UNCHANGED
//      (no evaluation spent on it at all this generation); the LOSER
//      updates its velocity and position by learning from BOTH the winner
//      and the swarm mean:
//        v_l(t+1) = r1*v_l(t) + r2*(x_w(t)-x_l(t)) + phi*r3*(x_bar(t)-x_l(t))
//        x_l(t+1) = x_l(t) + v_l(t+1)
//      with r1,r2,r3 independently uniform in [0,1] per dimension.
//
// Only losers are ever re-evaluated, so a full "generation" costs exactly
// N/2 objective evaluations regardless of D -- a major reason this method
// scales well to very large dimensions. The single social-learning
// coefficient phi controls how strongly the swarm mean pulls losers versus
// how strongly the winner does; the paper's recommended, dimension-aware
// default is phi = D/N (the "social factor" grows with problem
// dimensionality relative to population size, since in very high
// dimensions the swarm mean carries an increasingly useful, tempering
// signal that keeps the population diverse enough not to collapse
// prematurely along just a few directions).
class CSO : public Optimizer {
public:
    CSO() = default;
    ~CSO() override = default;

    std::string methodShortName() const override { return "cso"; }
    std::string methodFullName()  const override {
        return "Competitive Swarm Optimizer";
    }

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
    using Vec = std::vector<double>;

    double safeEval(const Vec& x);
    void   ensureBounds(Vec& x) const;

private:
    std::vector<Vec>    X_;   // positions
    std::vector<Vec>    V_;   // velocities
    std::vector<double> FX_;  // fitness

    // BUG FIX: the auto default used to be phi = D/pop_ (a formula
    // sometimes cited for CSO), but this becomes badly destabilizing
    // exactly in the large-D/small-N regime this method targets -- e.g.
    // D=1000, pop_=100 gives phi=10, which blew up the loser's velocity
    // over successive generations (r1 in [0,1] provides no guaranteed
    // damping; a run of high r1 draws lets velocity compound) and gave
    // catastrophically bad results even on trivial problems like Sphere
    // in testing (best_f ~3e6 instead of ~0). The paper's proof that the
    // swarm eventually converges for any phi>=0 does not help if it first
    // diverges within the available evaluation budget. Multiple
    // independent sources consistently report the empirically effective
    // range as phi in [0.1, 0.6] regardless of D (e.g. a CSO-derived
    // algorithm's own parameter sweep found phi=0.5 best over a
    // 0.1-0.6 sweep), so that is used as the safe default here instead.
    double phi_default_{0.2};

    // Social-learning coefficient. <=0 in config means "auto: phi_default_".
    double phi_{0.0};
    double phi_cfg_{0.0};

    double v_init_frac_{0.05}; // initial velocity magnitude, as a fraction of box range

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
