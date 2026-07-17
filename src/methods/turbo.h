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

// TuRBO -- Trust Region Bayesian Optimization (single trust region variant,
// "TuRBO-1").
// Reference: Eriksson, D.; Pearce, M.; Gardner, J.; Turner, R.D.; Poloczek, M.
//   "Scalable Global Optimization via Local Bayesian Optimization"
//   NeurIPS 2019, arXiv:1910.01739.
//
// Standard global BO with a single Gaussian Process over the WHOLE domain
// scales poorly and tends to over-explore in high dimensions ("BO gets
// stuck exploring the boundary"). TuRBO instead fits its GP surrogate
// locally, inside a trust region (TR) box centered on the best point found
// so far, whose per-dimension side length adapts online:
//   - a run of succtol_ consecutive improving batches DOUBLES the length
//     (successful region -> be bolder),
//   - a run of failtol_ consecutive non-improving batches HALVES it
//     (stuck -> be more local),
//   - if the length collapses below length_min_, the trust region has
//     converged/stalled and is RESTARTED from a fresh random design
//     (the global incumbent best_x_/best_f_ is of course never lost).
//
// Candidates are proposed by perturbing the TR center coordinate-wise
// (matching the paper's use of a random perturbation mask so that not
// every coordinate is perturbed at once in high dimensions -- perturbing
// all D coordinates simultaneously makes acceptance vanishingly unlikely
// as D grows), then a batch is selected from those candidates via Thompson
// sampling against the (locally-fit) GP posterior.
//
// The GP itself (Matern-5/2 kernel, exact inference via Cholesky) is a
// self-contained, minimal implementation -- this framework has no shared
// linear-algebra/GP utility module, so (consistent with how the other
// methods in this codebase are each fully self-contained, e.g. the CMA-ES
// family) all of the GP machinery below is private to this class.
class TuRBO : public Optimizer {
public:
    TuRBO() = default;
    ~TuRBO() override = default;

    std::string methodShortName() const override { return "turbo"; }
    std::string methodFullName()  const override {
        return "TuRBO (Trust Region Bayesian Optimization)";
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
    using Vec  = std::vector<double>;
    using Mat  = std::vector<Vec>;

    // ---- problem-space <-> unit-cube [0,1]^D normalization ----
    Vec toUnit(const Vec& x) const;
    Vec fromUnit(const Vec& u) const;
    void clampUnit(Vec& u) const;

    // Safe evaluation in ORIGINAL (non-normalized) coordinates; updates
    // best_f_/best_x_ immediately, NaN/Inf sanitized to +inf.
    double safeEvalUnit(const Vec& u);

    // ---- Gaussian Process (Matern-5/2, exact) ----
    double kernel(const Vec& a, const Vec& b) const;
    // (Re)fits lengthscale_/signal_var_ from the current dataset via a
    // simple median-heuristic + data-variance rule (cheap, robust, avoids
    // needing a full marginal-likelihood optimizer), then builds and
    // Cholesky-factorises K + noise*I. Returns false if the dataset is
    // empty or the factorisation fails even after jitter inflation.
    bool fitGP();
    // Posterior mean & (marginal, i.e. non-joint) variance at a query point.
    void predictGP(const Vec& xq, double& mean, double& var) const;

    // ---- Trust region candidate generation & Thompson sampling batch ----
    void generateCandidates(int nCand, std::vector<Vec>& cands);
    void thompsonSelect(const std::vector<Vec>& cands, int batch, std::vector<Vec>& chosen);

    void restartTrustRegion();

private:
    // --- dataset (normalized unit-cube inputs, raw objective outputs) ---
    Mat    X_;
    Vec    Y_;
    int    max_dataset_{200}; // cap for GP cubic-cost tractability (keep most recent/best)

    // --- GP hyperparameters (re-fit each iteration from the data) ---
    double lengthscale_{0.2};
    double signal_var_{1.0};
    double noise_var_{1e-6};
    double y_mean_{0.0};
    double y_std_{1.0};
    Mat    L_chol_;     // Cholesky factor of (K + noise*I), size n x n
    Vec    alpha_;      // (K+noise*I)^{-1} (y - y_mean_), for the posterior mean
    bool   gp_ready_{false};

    // --- trust region state (in normalized [0,1]^D units) ---
    Vec    tr_center_;
    double tr_length_{0.8};
    double length_min_{0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5}; // 0.5^7
    double length_max_{1.6};
    int    succ_count_{0};
    int    fail_count_{0};
    int    succ_tol_{3};
    int    fail_tol_{0};      // computed from D in init() if left at 0

    // --- batch / candidate sizing ---
    int    batch_size_{4};
    int    n_candidates_{0};  // computed from D in init() if left at 0
    double n_init_frac_{0.0}; // if >0, overrides n_init_ as a fraction of max_evals_
    int    n_init_{0};        // initial random design size (0 = auto)

    // --- BO evaluation budget cap ---
    // Bayesian Optimization (GP fit cost O(n^3), acquisition search cost
    // O(n_candidates_)) is designed for EXPENSIVE objectives with tiny
    // evaluation budgets (tens to low thousands), not the 150,000+ evals
    // this framework defaults to for population-based metaheuristics.
    // Naively running the full GP-based loop for the whole budget makes a
    // single run take minutes to hours -- completely impractical for
    // interactive/batch use. bo_budget_ caps how many evaluations the GP-
    // based search actually spends; once exhausted, cheapFallbackStep()
    // (O(D) per eval, no GP fit) takes over for any remaining budget so
    // the method still terminates in reasonable time and keeps refining
    // the incumbent rather than simply idling.
    long long bo_budget_{0};       // 0 = auto: min(max_evals_, 20*D + 200)
    double    fallback_sigma_{0.05}; // relative to per-dim box range
    bool      in_fallback_{false};

    void cheapFallbackStep();

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
