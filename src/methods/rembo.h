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

// REMBO -- Random EMbedding Bayesian Optimization.
// Reference: Wang, Z.; Hutter, F.; Zoghi, M.; Matheson, D.; de Freitas, N.
//   "Bayesian Optimization in a Billion Dimensions via Random Embeddings"
//   Journal of Artificial Intelligence Research (JAIR), 2016
//   (conference version: ICML 2013), arXiv:1301.1942.
//
// Motivation: many high-dimensional objectives have a much lower "effective
// dimensionality" -- only a handful of directions in x-space actually
// matter to f, the rest being near-constant/irrelevant. REMBO exploits this
// WITHOUT knowing which directions those are, by drawing a single random
// D x d_e Gaussian matrix A (d_e << D, the assumed effective dimension) and
// performing Bayesian Optimization entirely inside the LOW-dimensional
// space y in R^{d_e}, mapping each proposed y back up via x = A*y before
// evaluating the real objective. If the low effective dimensionality
// assumption holds, doing BO in d_e dimensions instead of D is dramatically
// cheaper and better-conditioned for a GP surrogate, while still (with high
// probability, by a Johnson-Lindenstrauss-style argument specific to the
// paper) reaching the region of x-space where the objective's important
// directions live.
//
// Following the paper, the low-dimensional search box is
//   y_i in [-sqrt(d_e), sqrt(d_e)]
// (large enough that the projection A*y, after being clipped back into the
// real box, covers the region where the true optimum is expected to lie
// with high probability), and points are mapped back with
//   x = clip( A*y, [lb,ub] )
// The GP itself (Matern-5/2, exact Cholesky inference) mirrors the
// self-contained implementation used in TuRBO -- this framework keeps each
// method file fully self-contained, so the small amount of duplication
// here is intentional and matches the codebase's existing convention
// (e.g. the CMA-ES family).
class REMBO : public Optimizer {
public:
    REMBO() = default;
    ~REMBO() override = default;

    std::string methodShortName() const override { return "rembo"; }
    std::string methodFullName()  const override {
        return "REMBO (Random Embedding Bayesian Optimization)";
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
    using Mat = std::vector<Vec>;

    // y (low-dim, d_e) -> x (original D-dim), clipped to the real box.
    Vec embedToX(const Vec& y) const;

    // Safe evaluation of a low-dim point: maps up to x, evaluates, updates
    // best_f_/best_x_ (in ORIGINAL space) immediately.
    double safeEvalY(const Vec& y);

    // ---- Gaussian Process over the LOW-dimensional (d_e) space ----
    double kernel(const Vec& a, const Vec& b) const;
    bool   fitGP();
    void   predictGP(const Vec& yq, double& mean, double& var) const;

    // Expected Improvement at a low-dim candidate (minimizing convention).
    double expectedImprovement(const Vec& yq) const;

    // Propose the next y via random search + local refinement of EI over
    // the low-dim box (a full inner gradient-based EI maximizer would need
    // its own optimizer; random-search-with-polish is the standard
    // practical stand-in used by many from-scratch BO implementations).
    Vec proposeNextY();

private:
    // --- random embedding ---
    int d_embed_{0};      // effective dimension (0 = auto: min(D, d_embed_max_))
    int d_embed_max_{8};
    Mat A_;               // D x d_embed_ embedding matrix
    double y_box_{0.0};   // low-dim box half-width = sqrt(d_embed_)

    // --- dataset (low-dim inputs, raw objective outputs) ---
    Mat    Y_pts_;   // low-dim query points, size n x d_embed_
    Vec    F_;       // objective values
    int    max_dataset_{200};

    // --- GP hyperparameters ---
    double lengthscale_{1.0};
    double signal_var_{1.0};
    double noise_var_{1e-6};
    double f_mean_{0.0};
    double f_std_{1.0};
    Mat    L_chol_;
    Vec    alpha_;
    bool   gp_ready_{false};
    double f_best_std_{0.0}; // best standardized value seen (for EI)

    // --- search / EI proposal sizing ---
    int n_init_{0};          // initial random design size (0 = auto)
    int ei_candidates_{500};
    int ei_polish_steps_{15};
    double ei_polish_sigma_{0.1}; // relative to y_box_

    // --- BO evaluation budget cap ---
    // Same rationale as TuRBO: naively fitting a GP every iteration for
    // this framework's default 150,000-eval budget makes a single run
    // take minutes to hours, which is impractical for interactive/batch
    // use. Once bo_budget_ evaluations are spent, cheapFallbackStep()
    // (O(D) per eval, no GP fit) takes over for the remainder.
    long long bo_budget_{0};        // 0 = auto: min(max_evals_, 20*d_embed_ + 200)
    double    fallback_sigma_{0.05};
    bool      in_fallback_{false};

    void cheapFallbackStep();

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
