#pragma once
// =============================================================================
//  IMSPSO-DNI -- Improved Multiple Subgroups Particle Swarm Optimization with
//                Dynamic Neighborhood Information mechanism
//
//  Reference (implemented from the manuscript, equations (1)-(13) and
//  Algorithms 1-4; no source code of the authors is available):
//    X. Yao, D. Tian, "An Improved Multiple Subgroups Particle Swarm
//    Optimization with Dynamic Neighborhood Information Mechanism",
//    Algorithms (MDPI), 2026 (manuscript algorithms-4692278, peer review v1).
//
//  Components
//   (1) RL-driven dynamic subgroup adjustment (Sec. 3.1, Alg. 1, Eqs. 6-8):
//       Q-learning over the subgroup numbers S = {2,4,6,8,10}; states and
//       actions are both elements of S. Each iteration the action (= number
//       of subgroups K) is the argmax of Q(K_c, .) with probability
//       epsilon = 0.8, otherwise a random element of S. Reward r = 1 if the
//       gbest value improved in the iteration, else 0;
//       Q(i,j) += alpha*(r + gamma*max_k Q(j,k) - Q(i,j)), alpha 0.1,
//       gamma 0.9; next state = chosen action.
//   (2) Hierarchical elite cooperative updating mechanism, HECUM
//       (Sec. 3.2, Alg. 2, Eqs. 9-10): the swarm is sorted by pbest fitness
//       and divided into K subgroups.
//         subgroup 1  : not moved ("directly enter the next iteration");
//         subgroup 2  : V = w V + c1 r1 (g1_mean - X) + c2 r2 (gbest - X)
//                       (g1_mean = mean pbest of subgroup 1), X = X + V;
//         subgroup >=3: two candidates are drawn at random from the
//                       subgroups with smaller indices; a candidate whose
//                       fitness is worse than that of the current particle
//                       is replaced by a random particle of subgroup 1;
//                       V = r1 V + r2 (X_ex1 - X) + phi r3 (X_ex2 - X),
//                       phi = 0.4, X = X + V.
//   (3) gbest perturbation based on dynamic neighborhood information,
//       gPS-DNI (Sec. 3.3, Alg. 3, Eqs. 11-13): a counter g_c is increased
//       in every iteration without gbest improvement; when g_c reaches
//       g_l = 5 the diversity D (mean Euclidean distance of the particles to
//       the swarm centre, Eq. 11) is computed and
//         D >  T : gbest_new = gbest + D * U(-1,1)          (Eq. 12)
//         D <= T : gbest_new = gbest + R * D * U(-1,1)      (Eq. 13)
//       with T = 0.5, R = 0.4; g_c = 0; gbest_new is adopted if it is not
//       worse than gbest.
//   Parameters (Sec. 4.2, Table 2): N = 30; w 0.9 -> 0.4 (Eq. 5),
//   c1 2.5 -> 0.5, c2 0.5 -> 2.5 (linear); position and velocity bounded
//   (CEC: both within [-100,100], i.e. |V| <= (ub-lb)/2).
//
//  Interpretation choices where the manuscript is ambiguous (all
//  documented in imspsodni.cpp, the defaults follow the text):
//   * uneven division of N particles into K subgroups: sizes differ by at
//     most one, the larger subgroups first (better particles);
//   * r1, r2, r3 and U(-1,1) are drawn per dimension;
//   * candidates / exemplars of Eq. (10) use the particles' current
//     positions and fitness (the text: "positions of two particles ... both
//     of them possess higher fitness values than the current particle"),
//     taken from the start of the iteration (synchronous update);
//   * Alg. 3 line 23: on an improving iteration g_c is left unchanged (not
//     reset); option gc_reset_on_improve = 1 resets it;
//   * Eq. 13: "the parameter D fixed at 0.4" is read as R = 0.4 (D is the
//     measured diversity);
//   * the "evaluate every particle" step of Alg. 4 evaluates only particles
//     that moved (subgroup 1 is unchanged, its fitness is known), which
//     saves evaluations without changing the search;
//   * iteration-based schedules use the fraction of the evaluation budget
//     (the framework stops on max_evals);
//   * initial Q-learning state: a random element of S; ties in argmax are
//     broken at random; initial velocities U(-vmax, vmax).
// =============================================================================

#include "optimizer.h"

#include <vector>
#include <string>
#include <random>

namespace optimsolution {

class IMSPSODNI : public Optimizer {
public:
    IMSPSODNI() = default;
    ~IMSPSODNI() override = default;

    std::string methodShortName() const override { return "imspsodni"; }
    std::string methodFullName()  const override {
        return "IMSPSO-DNI (multiple subgroups PSO with RL subgroup control, HECUM and gbest perturbation)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x);
    double U01();
    int    randInt(int n);
    int    chooseAction(int state);                    // Alg. 1 / Eq. 7
    void   updateQ(int s, int a, double reward);       // Eq. 6
    double diversity() const;                          // Eq. 11
    void   clampVelocity(Vec& v) const;
    void   clipPosition(Vec& x) const;
    void   noteGbest(const Vec& x, double f);

    // ---- swarm ----
    std::vector<Vec>    X_, V_, P_;     // positions, velocities, pbest positions
    std::vector<double> F_, PF_;        // current fitness, pbest fitness
    Vec                 vmax_;          // per-dimension velocity bound

    // ---- RL ----
    std::vector<int>    S_ = {2, 4, 6, 8, 10};          // subgroup number candidates
    std::vector<std::vector<double>> Q_;
    int                 state_ = 0;                     // index into S_

    // ---- gPS-DNI ----
    int                 gc_ = 0;

    // ---- configuration ([imspsodni] section) ----
    int    pop_cfg_    = 30;
    double w_max_      = 0.9, w_min_ = 0.4;
    double c1_start_   = 2.5, c1_end_ = 0.5;
    double c2_start_   = 0.5, c2_end_ = 2.5;
    double phi_        = 0.4;
    double alpha_      = 0.1, gamma_ = 0.9, eps_ = 0.8;
    int    gl_         = 5;
    double T_          = 0.5;
    double R_          = 0.4;
    double vmax_frac_  = 0.5;           // |V_j| <= vmax_frac*(ub_j - lb_j)  (CEC: 100)
    bool   gc_reset_on_improve_ = false;
    bool   accept_equal_ = true;        // "no inferior" (text) vs strictly better (Alg. 3)
    bool   v_init_random_ = true;

    bool        end_local_refine_ = false;
    std::string end_local_method_;
    int         debug_ = 0;
};

} // namespace optimsolution
