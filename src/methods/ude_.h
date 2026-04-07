#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>
#include <array>

namespace optimsolution {

/**
 * UDE – Unified Adaptive Differential Evolution
 *
 * Four-strategy self-adaptive portfolio with weighted-voting strategy
 * selection and per-strategy stagnation recovery.
 *
 * Strategy portfolio
 * ──────────────────
 *   Exploration  [0] DE/rand/1/bin          – classic, unbiased
 *                [1] DE/rand/2/bin          – wider search step
 *   Exploitation [2] DE/best/1/bin          – best-guided, fast convergence
 *                [3] JSO / current-to-pbest/1/bin  – top-p% guided perturbation
 *
 *   Strategy [3] is the mutation core of jSO / L-SHADE:
 *     v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
 *   where x_pbest is drawn uniformly from the top (pbest_p * N) individuals.
 *   pbest_p is configurable (key "pbest_p", default 0.10 = top 10%).
 *   This yields softer exploitation than best/1 and avoids premature convergence
 *   while still steering the search toward high-quality regions.
 *
 * Adaptive selection (voting)
 * ───────────────────────────
 *   Each strategy s carries a selection probability strat_prob_[s].
 *   After every adapt_window_ iterations the probabilities are recomputed
 *   proportionally to each strategy's success rate (successes / trials) over
 *   the current window; a floor of min_prob_ prevents any strategy from
 *   being completely abandoned.  Window counters are reset after every update.
 *
 * Per-strategy stagnation recovery
 * ──────────────────────────────────
 *   stag_count_[s] tracks consecutive iterations in which strategy s was
 *   actually applied to at least one individual but produced zero improvements.
 *   When stag_count_[s] reaches stagnation_limit_ only that strategy is reset
 *   (probability returned to uniform share, counters cleared); all other
 *   strategies are left intact.
 *
 * Local search hooks
 * ──────────────────
 *   In-run  : applied with probability local_rate_ after every accepted trial.
 *   End-run : one final polish on best_x_ when end_local_refine_ = true,
 *             identical to the DE implementation.
 *
 * Configuration keys (MethodConfig / INI section [ude])
 * ──────────────────────────────────────────────────────
 *   F                 scale factor                      (default 0.6)
 *   CR                crossover rate                    (default 0.9)
 *   pbest_p           top fraction for JSO pbest pool   (default 0.10)
 *   population / pop  population size                   (default from [global])
 *   adapt_window      probability update interval       (default 50 iterations)
 *   stagnation_limit  iterations before per-strat reset (default 100 iterations)
 *   min_prob          probability floor per strategy    (default 0.05)
 *   local_method      in-run local method               (default lbfgs)
 *   local_rate        in-run local probability          (default 0.0 = off)
 *   end_local_refine  enable end-of-run local           (default false)
 *   end_local_method  end-of-run local method           (default "")
 *   debug_ude         verbosity (0/1)                   (default 0)
 */
class UDE : public Optimizer {
public:
    // Number of strategies in the portfolio
    static constexpr int NS = 4;

    enum Strategy : int {
        RAND1 = 0,   // DE/rand/1/bin
        RAND2 = 1,   // DE/rand/2/bin
        BEST1 = 2,   // DE/best/1/bin
        JSO   = 3    // jSO / DE/current-to-pbest/1/bin
    };

    UDE()  = default;
    ~UDE() override = default;

    std::string methodShortName() const override { return "UDE"; }
    std::string methodFullName()  const override {
        return "Unified Adaptive Differential Evolution (UDE)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init()          override;
    void one_iteration() override;
    void end()           override;

private:
    using Vec = std::vector<double>;

    // ── Mutation helpers ─────────────────────────────────────────────────────
    // Each function returns the donor vector (before crossover and bounds repair).
    Vec mutate_rand1(int i);   // v = x_r1 + F*(x_r2 - x_r3)
    Vec mutate_rand2(int i);   // v = x_r1 + F*(x_r2-x_r3) + F*(x_r4-x_r5)
    Vec mutate_best1(int i);   // v = x_best + F*(x_r1 - x_r2)
    Vec mutate_jso  (int i);   // v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)

    // Binomial crossover (uses CR_ and member rng_)
    Vec  crossover  (const Vec& target, const Vec& donor);

    // Bound-repair: clamp each component to [lb, ub]
    void ensureBounds(Vec& v);

    // Draw a random integer in [0, n) distinct from up to three exclusions
    int  pickDistinct(int n, int a = -1, int b = -1, int c = -1);

    inline double eval(const Vec& v) {
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    // ── Adaptation helpers ───────────────────────────────────────────────────
    // Draw a strategy index according to strat_prob_
    int  selectStrategy();

    // Recompute strat_prob_ from success_/trials_ and reset window counters
    void updateProbabilities();

    // Reset a single stagnated strategy to uniform share; re-normalise all
    void resetStrategy(int s);

    // Enforce min_prob_ floor and re-normalise strat_prob_ to sum to 1
    void normalizeProbabilities();

private:
    // Population
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // Per-strategy bookkeeping
    // (prefix strat_ avoids collision with base-class member prob_)
    std::array<double, NS> strat_prob_  {};  // selection probabilities (sum = 1)
    std::array<int,    NS> success_     {};  // successes in current window
    std::array<int,    NS> trials_      {};  // trials in current window
    std::array<int,    NS> stag_count_  {};  // consecutive iterations w/o improvement

    // DE control parameters
    double F_       = 0.6;
    double CR_      = 0.9;
    double pbest_p_ = 0.10;  // fraction of population forming the JSO pbest pool

    // Adaptation hyper-parameters
    double min_prob_         = 0.05;   // probability floor per strategy
    int    adapt_window_     = 50;     // iterations between probability updates
    int    stagnation_limit_ = 100;    // consecutive idle iterations → per-strategy reset
    int    iter_             = 0;      // iteration counter

    // In-run local search
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // End-of-run local search
    bool        end_local_refine_ = false;
    std::string end_local_method_;

    // Misc
    int debug_ude_    = 0;
    int pop_override_ = -1;
};

} // namespace optimsolution
