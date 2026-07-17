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

// CCDG2 -- Cooperative Co-evolution with DG2 (Differential Grouping v2)
// problem decomposition.
//
// References:
//   Omidvar, M.N.; Li, X.; Mei, Y.; Yao, X. "Cooperative Co-evolution with
//     Differential Grouping for Large Scale Optimization." IEEE Trans.
//     Evol. Comput. 18(3), 2014.
//   Omidvar, M.N.; Yang, M.; Mei, Y.; Li, X.; Yao, X. "DG2: A Faster and
//     More Accurate Differential Grouping for Large-Scale Black-Box
//     Optimization." IEEE Trans. Evol. Comput. 21(6), 929-942, 2017.
//
// Motivation: for D in the many hundreds/thousands, monolithic optimizers
// (even large-scale-aware ones like this framework's LM-CMA-ES/LRA-CMA-ES)
// still search the FULL D-dimensional space at once. Cooperative
// Co-evolution (CC) instead DECOMPOSES the problem into subgroups of
// variables that interact with each other but not (much) with variables
// outside the group, and optimizes each subgroup independently, cycling
// between them and sharing a single "context vector" (the best full-length
// solution found so far). This divide-and-conquer approach is what has
// actually won nearly every CEC Large-Scale Global Optimization (LSGO)
// competition since 2010 -- it is a fundamentally different strategy from
// this framework's existing large-scale methods (LM-CMA-ES, LRA-CMA-ES),
// which are monolithic.
//
// Two phases:
//
//  (1) DG2 grouping (init(), budget-capped by grouping_budget_frac_): for
//      each pair of variables (i,j), DG2 tests whether perturbing x_i by
//      the same amount produces a DIFFERENT change in f depending on x_j's
//      value -- if so, i and j interact and belong in the same group. DG2's
//      key improvement over the original 2014 DG is a PARAMETER-FREE,
//      adaptive interaction threshold: rather than a hand-tuned magic
//      number, the threshold is derived from an estimate of the
//      floating-point round-off error accumulated while evaluating f
//      (assuming a linear-in-D operation count, the standard conservative
//      assumption for a generic black-box objective), so the test scales
//      correctly with the objective's own numerical magnitude instead of
//      requiring per-problem tuning. Sample points are cached and reused
//      across tests against the same seed variable (DG2's efficiency
//      contribution), and interactions are propagated transitively within
//      a bounded number of passes to catch indirectly-interacting
//      variables without full O(D^2) recomputation.
//
//  (2) Cooperative co-evolution main loop (one_iteration(), round-robin
//      over groups): each call runs ONE generation of a compact
//      current-to-pbest/1/bin DE with a success-history parameter memory
//      (a SHADE-lite subcomponent optimizer) for the NEXT group in
//      rotation, evaluating candidates by plugging that group's variables
//      into the shared context vector (all other dimensions held fixed).
//      Any improvement updates the context vector and, if it is also the
//      best full solution seen so far, best_x_/best_f_.
//
// IMPORTANT correctness note (stale-fitness fix): a naive round-robin CC
// loop has a subtle but serious bug -- while group A is idle (other groups
// are being processed), those OTHER groups keep changing the shared
// context vector. When control returns to group A, its cached FX[] values
// were computed against whatever context existed the LAST time A ran, not
// the current one. Comparing freshly-generated trial candidates (evaluated
// under the CURRENT context) against those stale FX[] values is an
// apples-to-oranges comparison: it can accept genuinely worse candidates
// (because the stale baseline looked artificially good under an outdated
// context) or reject genuinely better ones, silently degrading
// convergence without ever crashing or looking obviously wrong. This is
// fixed via refreshStaleFitness(): every group tracks which
// context_version_ its FX[] values were last computed against, and the
// ENTIRE subpopulation is re-evaluated against the current context before
// a group's generation proceeds whenever the context has changed since its
// last visit (context_version_ is bumped on every context_ update).
class CCDG2 : public Optimizer {
public:
    CCDG2() = default;
    ~CCDG2() override = default;

    std::string methodShortName() const override { return "ccdg2"; }
    std::string methodFullName()  const override {
        return "Cooperative Co-evolution with DG2 decomposition (large-scale)";
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

    // Safe full-D evaluation; updates best_f_/best_x_ immediately.
    double safeEval(const Vec& xfull);

    // ---- DG2 grouping ----
    // Interaction test between dims i and j, given the two cached baseline
    // evaluations (f at p1, and f at p1 with x_i perturbed) that do not
    // depend on j -- the caller supplies these so they can be reused across
    // many j's for the same seed i (DG2's stated efficiency contribution).
    bool dg2Interacts(int i, int j, double f_p1, double f_p1_pert,
                       Vec& p1, Vec& p1_pert, double eps_D);
    void runGrouping();

    // ---- per-group SHADE-lite subcomponent optimizer ----
    struct Group {
        std::vector<int> dims;             // indices into the full D-vector
        std::vector<Vec> X;                // population, each sized dims.size()
        std::vector<double> FX;
        std::vector<Vec> archive;          // external archive (replaced parents)
        std::vector<double> M_F, M_CR;     // SHADE success-history memory
        int mem_pos{0};
        int pop{0};
        // Which context_version_ this group's FX[] values were last
        // evaluated against (see refreshStaleFitness() for why this
        // matters).
        long long fx_context_version{-1};
    };
    // Re-evaluates g.FX[] (and g.X[] fitness only, not positions) against
    // the CURRENT context_ if it has changed since this group's fitness
    // values were last computed -- see class-level comment for why this
    // is necessary for correct cooperative-coevolution comparisons.
    void refreshStaleFitness(Group& g);
    void initGroupPopulation(Group& g);
    void groupGeneration(Group& g); // one DE generation for this group

    // Build a full-D candidate from the context vector + a group's partial vector.
    Vec buildFull(const Group& g, const Vec& partial) const;

private:
    std::vector<Group> groups_;
    Vec    context_;
    double context_f_{std::numeric_limits<double>::infinity()};
    int    current_group_{0};
    // Incremented every time context_ changes. Compared against each
    // Group::fx_context_version to detect stale per-individual fitness
    // (see class-level comment / refreshStaleFitness()).
    long long context_version_{0};

    // --- DG2 / grouping configuration ---
    double grouping_budget_frac_{0.20}; // fraction of max_evals_ spent on grouping
    int    dg2_max_passes_{-1};         // -1 = auto: min(D, 200); bound on
                                         // transitive-closure passes per seed.
                                         // A small FIXED cap (an earlier
                                         // version used 8) artificially cuts
                                         // off propagation along long
                                         // interaction chains (e.g. standard
                                         // chained Rosenbrock, where x_i and
                                         // x_{i+1} interact for every i),
                                         // causing real interactions between
                                         // distant-but-connected dimensions
                                         // to be missed. Total grouping cost
                                         // is bounded regardless by the
                                         // per-evaluation grouping_budget_
                                         // check inside the grouping loop.

    // --- subcomponent DE configuration (SHADE-lite) ---
    int    group_pop_cfg_{-1};  // <=0 -> auto per group size
    int    shade_H_{5};
    double archive_rate_{1.0};
    double p_best_min_frac_{0.20}; // pbest fraction upper bound (paper: p in [pmin,0.2])

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
