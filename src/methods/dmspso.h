#pragma once
#include "optimizer.h"
#include "pso_family_common.h"
#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// =============================================================================
//  DMS-PSO -- Dynamic Multi-Swarm Particle Swarm Optimizer
//
//  References:
//    J. J. Liang, P. N. Suganthan, "Dynamic multi-swarm particle swarm
//      optimizer", Proc. IEEE Swarm Intelligence Symposium (SIS 2005),
//      pp. 124-129 (doi:10.1109/SIS.2005.1501611).
//    J. J. Liang, P. N. Suganthan, "Dynamic multi-swarm particle swarm
//      optimizer with local search", IEEE CEC 2005, pp. 522-528
//      (doi:10.1109/CEC.2005.1554727).
//    S. Z. Zhao, J. J. Liang, P. N. Suganthan, M. F. Tasgetiren, "Dynamic
//      multi-swarm particle swarm optimizer with local search for large scale
//      global optimization", IEEE CEC 2008, pp. 3845-3852.
//
//  The swarm is split into many SMALL sub-swarms (3 particles each). Each
//  particle follows its own pbest and the best pbest of its sub-swarm:
//         v = w v + c1 r1 (pbest - x) + c2 r2 (lbest - x)
//  Every R generations all particles are randomly regrouped into new
//  sub-swarms, so information spreads through the whole population while
//  diversity is preserved. For the last (1 - global_phase) fraction of the
//  budget (default: the last 10% of the evaluations), all particles form a
//  single swarm and run global-best PSO to converge.
//
//  Optional local search (DMS-L-PSO, CEC 2005/2008): every ls_period
//  generations the pbests of ls_count random particles are refined with the
//  framework's local_method; each refined point replaces the NEAREST pbest
//  (Euclidean) if it is better. ls_period = 0 (default) disables it, giving
//  the plain DMS-PSO.
//
//  Defaults (CEC 2008 DMS-L-PSO paper): sub-swarm size 3, R = 10,
//  w = 0.729, c1 = c2 = 1.49445, Vmax = 0.2*range, global phase after 90% of
//  the evaluations, L = 100, 5 particles refined. Number of sub-swarms:
//  max(10, round(0.3*D)), which gives the 30/150/300 sub-swarms the CEC 2008
//  paper used for D = 100/500/1000 and 10 sub-swarms (30 particles) for small
//  D. Box handling bound_mode = skip: as in the CEC 2008 paper, a particle's
//  pbest is only updated while it lies inside the range; out-of-range
//  particles are not evaluated (clamp / reflect are available).
//  w_end >= 0 turns the constant inertia into a linear w -> w_end schedule
//  (e.g. w = 0.9, w_end = 0.2, a setting found in later DMS-PSO comparisons;
//  not verified against the original papers, hence not the default).
// =============================================================================
class DMSPSO : public Optimizer {
public:
    DMSPSO() = default;
    ~DMSPSO() override = default;

    std::string methodShortName() const override { return "dmspso"; }
    std::string methodFullName()  const override { return "Dynamic Multi-Swarm Particle Swarm Optimizer"; }

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
    void regroup();
    void periodicLocalSearch();
    void maybeLocalSearch(int i);

private:
    std::vector<Vec>    X_, V_, P_;
    std::vector<double> FX_, FP_;
    std::vector<int>    group_;     // group_[i] = sub-swarm of particle i
    std::vector<int>    lbest_;     // lbest_[s] = best member of sub-swarm s
    Vec                 Vmax_;
    int                 g_{0};
    int                 n_swarms_{0};
    long long           gen_{0};

    // Parameters
    int    pop_cfg_{0};       // 0 = sub_size * n_swarms
    int    sub_size_{3};
    int    n_swarms_cfg_{0};  // 0 = max(10, round(0.3*D))
    int    R_{10};
    double w_{0.729};
    double w_end_{-1.0};      // < 0: constant w; otherwise linear w -> w_end over the run
    double c1_{1.49445}, c2_{1.49445};
    double vmax_frac_{0.2};
    double global_phase_{0.9};
    int    ls_period_{0};
    int    ls_count_{5};

    psofamily::BoundMode bound_mode_{psofamily::BoundMode::Skip};
    int    no_eval_gens_{0};

    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
