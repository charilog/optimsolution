#include "turbo.h"
#include <numeric>
#include <limits>

namespace optimsolution {

namespace {
    // Small dense linear-algebra helpers, private to this translation unit
    // (this codebase has no shared linear-algebra module; every method here
    // is fully self-contained, matching e.g. how the CMA-ES family is
    // written).

    // In-place lower-Cholesky of a symmetric positive-(semi)definite matrix.
    // A is n x n; on success A's lower triangle holds L with L*L^T = A_in.
    // Returns false if a non-positive pivot is hit (caller should retry
    // with more jitter on the diagonal).
    bool choleskyInPlace(std::vector<std::vector<double>>& A, int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = A[i][j];
                for (int k = 0; k < j; ++k) sum -= A[i][k] * A[j][k];
                if (i == j) {
                    if (sum <= 0.0) return false;
                    A[i][j] = std::sqrt(sum);
                } else {
                    A[i][j] = sum / A[j][j];
                }
            }
        }
        return true;
    }

    // Solve L*L^T*x = b given the lower-Cholesky factor L (n x n).
    std::vector<double> choleskySolve(const std::vector<std::vector<double>>& L,
                                       const std::vector<double>& b, int n) {
        std::vector<double> y(n), x(n);
        for (int i = 0; i < n; ++i) {
            double s = b[i];
            for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
            y[i] = s / L[i][i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = y[i];
            for (int k = i + 1; k < n; ++k) s -= L[k][i] * x[k];
            x[i] = s / L[i][i];
        }
        return x;
    }

    // Forward-substitution only: solve L*v = b (used to get the predictive
    // variance term v^T v = k*^T K^-1 k*).
    std::vector<double> forwardSolve(const std::vector<std::vector<double>>& L,
                                      const std::vector<double>& b, int n) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) {
            double s = b[i];
            for (int k = 0; k < i; ++k) s -= L[i][k] * v[k];
            v[i] = s / L[i][i];
        }
        return v;
    }
}

void TuRBO::configure(const MethodConfig& mc) {
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    batch_size_    = mc.getInt("batch_size", batch_size_);
    n_init_        = mc.getInt("n_init", n_init_);
    n_candidates_  = mc.getInt("n_candidates", n_candidates_);
    succ_tol_      = mc.getInt("succ_tol", succ_tol_);
    fail_tol_      = mc.getInt("fail_tol", fail_tol_);
    tr_length_     = mc.getDbl("tr_length_init", tr_length_);
    length_min_    = mc.getDbl("length_min", length_min_);
    length_max_    = mc.getDbl("length_max", length_max_);
    max_dataset_   = mc.getInt("max_dataset", max_dataset_);
    noise_var_     = mc.getDbl("noise_var", noise_var_);
    bo_budget_     = mc.getInt("bo_budget", (int)bo_budget_);
    fallback_sigma_= mc.getDbl("fallback_sigma", fallback_sigma_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

TuRBO::Vec TuRBO::toUnit(const Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    Vec u(D);
    for (int j = 0; j < D; ++j) {
        const double lo = L[j], hi = U[j];
        u[j] = (hi > lo) ? (x[j] - lo) / (hi - lo) : 0.5;
    }
    return u;
}

TuRBO::Vec TuRBO::fromUnit(const Vec& u) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)u.size();
    Vec x(D);
    for (int j = 0; j < D; ++j) {
        const double lo = L[j], hi = U[j];
        x[j] = lo + u[j] * (hi - lo);
    }
    return x;
}

void TuRBO::clampUnit(Vec& u) const {
    for (double& v : u) {
        if (!std::isfinite(v)) v = 0.5;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
    }
}

double TuRBO::safeEvalUnit(const Vec& u) {
    Vec x = fromUnit(u);
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

double TuRBO::kernel(const Vec& a, const Vec& b) const {
    double r2 = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        const double d = a[k] - b[k];
        r2 += d * d;
    }
    const double r = std::sqrt(r2) / std::max(1e-12, lengthscale_);
    const double sqrt5 = 2.2360679774997896;
    const double poly = 1.0 + sqrt5 * r + (5.0 / 3.0) * r * r;
    return signal_var_ * poly * std::exp(-sqrt5 * r);
}

bool TuRBO::fitGP() {
    const int n = (int)X_.size();
    gp_ready_ = false;
    if (n < 2) return false;

    // --- median-distance lengthscale heuristic + data-driven signal var ---
    {
        std::vector<double> dists;
        dists.reserve((size_t)n * (n - 1) / 2);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double r2 = 0.0;
                for (size_t k = 0; k < X_[i].size(); ++k) {
                    const double d = X_[i][k] - X_[j][k];
                    r2 += d * d;
                }
                dists.push_back(std::sqrt(r2));
            }
        }
        std::sort(dists.begin(), dists.end());
        double med = dists[dists.size() / 2];
        if (!(med > 1e-8)) med = 0.1;
        lengthscale_ = std::max(1e-3, med);

        double ymean = 0.0;
        for (double y : Y_) ymean += y;
        ymean /= (double)n;
        double yvar = 0.0;
        for (double y : Y_) { const double d = y - ymean; yvar += d * d; }
        yvar /= std::max(1, n - 1);
        y_mean_ = ymean;
        y_std_  = std::sqrt(std::max(1e-12, yvar));
        signal_var_ = 1.0; // outputs are standardized below, so unit signal variance
    }

    Vec y_std(n);
    for (int i = 0; i < n; ++i) y_std[i] = (Y_[i] - y_mean_) / y_std_;

    // --- build K + jitter, retry with escalating jitter until PD ---
    double jitter = noise_var_;
    for (int attempt = 0; attempt < 6; ++attempt) {
        Mat K(n, Vec(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double kij = kernel(X_[i], X_[j]);
                if (i == j) kij += jitter;
                K[i][j] = kij;
                K[j][i] = kij;
            }
        }
        if (choleskyInPlace(K, n)) {
            L_chol_ = std::move(K);
            alpha_  = choleskySolve(L_chol_, y_std, n);
            gp_ready_ = true;
            return true;
        }
        jitter = (jitter > 0.0) ? jitter * 10.0 : 1e-8;
    }
    return false;
}

void TuRBO::predictGP(const Vec& xq, double& mean, double& var) const {
    const int n = (int)X_.size();
    if (!gp_ready_ || n == 0) {
        mean = y_mean_;
        var  = signal_var_ + noise_var_;
        return;
    }
    Vec kstar(n);
    for (int i = 0; i < n; ++i) kstar[i] = kernel(xq, X_[i]);

    double m = y_mean_;
    for (int i = 0; i < n; ++i) m += kstar[i] * alpha_[i] * y_std_;

    Vec v = forwardSolve(L_chol_, kstar, n);
    double vtv = 0.0;
    for (double vi : v) vtv += vi * vi;

    double kxx = kernel(xq, xq);
    double post_var = std::max(1e-12, kxx - vtv);
    var  = post_var * y_std_ * y_std_;
    mean = m;
}

void TuRBO::generateCandidates(int nCand, std::vector<Vec>& cands) {
    const int D = (int)tr_center_.size();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Probability of perturbing any given coordinate -- matches the paper's
    // motivation: perturbing all D coordinates of a high-D candidate at
    // once makes it vanishingly unlikely to land anywhere useful, so only
    // a (shrinking-with-D) subset of coordinates are perturbed per sample.
    const double prob_perturb = std::min(1.0, 20.0 / std::max(1, D));

    cands.assign(nCand, tr_center_);
    for (int c = 0; c < nCand; ++c) {
        bool any = false;
        for (int j = 0; j < D; ++j) {
            if (U01(rng_) < prob_perturb) {
                cands[c][j] += (tr_length_ * 0.5) * N01(rng_);
                any = true;
            }
        }
        if (!any) {
            // Guarantee at least one perturbed coordinate per candidate.
            const int j = (int)(U01(rng_) * D) % std::max(1, D);
            cands[c][j] += (tr_length_ * 0.5) * N01(rng_);
        }
        // Confine to the trust-region box (in addition to the global unit cube).
        for (int j = 0; j < D; ++j) {
            const double lo = std::max(0.0, tr_center_[j] - 0.5 * tr_length_);
            const double hi = std::min(1.0, tr_center_[j] + 0.5 * tr_length_);
            if (cands[c][j] < lo) cands[c][j] = lo;
            if (cands[c][j] > hi) cands[c][j] = hi;
        }
    }
}

void TuRBO::thompsonSelect(const std::vector<Vec>& cands, int batch, std::vector<Vec>& chosen) {
    // Approximate (marginal, not fully joint) Thompson sampling: draw one
    // independent posterior sample per candidate and take the best `batch`
    // of them. This is a standard, cheap simplification of true joint TS
    // (which needs the full candidate-candidate posterior covariance) used
    // in many practical TuRBO reproductions.
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<std::pair<double,int>> scored;
    scored.reserve(cands.size());

    for (size_t i = 0; i < cands.size(); ++i) {
        double mean, var;
        predictGP(cands[i], mean, var);
        const double z = N01(rng_);
        const double sample = mean + std::sqrt(std::max(0.0, var)) * z;
        scored.emplace_back(sample, (int)i);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    chosen.clear();
    const int take = std::min((int)scored.size(), batch);
    for (int i = 0; i < take; ++i) chosen.push_back(cands[scored[i].second]);
}

// Cheap (O(D) per eval, no GP fit) local refinement for any evaluation
// budget remaining once bo_budget_ is exhausted: a simple (1+1)-style
// isotropic Gaussian hill-climber around the incumbent, with the classic
// 1/5-success-rule step-size adaptation, evaluated directly in original
// problem coordinates.
void TuRBO::cheapFallbackStep() {
    const int D = prob_->dimension();
    std::normal_distribution<double> N01(0.0, 1.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    Vec cand = best_x_;
    for (int j = 0; j < D; ++j) {
        const double range = U[j] - L[j];
        cand[j] += fallback_sigma_ * range * N01(rng_);
        if (cand[j] < L[j]) cand[j] = L[j];
        if (cand[j] > U[j]) cand[j] = U[j];
    }
    const double f = prob_->evaluate(cand);
    const bool improved = std::isfinite(f) && f < best_f_;
    if (improved) {
        best_f_ = f;
        best_x_ = cand;
        fallback_sigma_ = std::min(0.5, fallback_sigma_ * 1.2);
    } else {
        fallback_sigma_ = std::max(1e-6, fallback_sigma_ * 0.95);
    }
}

void TuRBO::restartTrustRegion() {
    tr_length_  = 0.8;
    succ_count_ = 0;
    fail_count_ = 0;

    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const int n0 = std::max(2 * D + 1, 8);
    for (int i = 0; i < n0 && prob_->calls() < max_evals_; ++i) {
        Vec u(D);
        for (int j = 0; j < D; ++j) u[j] = U01(rng_);
        const double f = safeEvalUnit(u);
        X_.push_back(u);
        Y_.push_back(f);
    }

    // Re-center on the best point seen so far (global incumbent, mapped
    // back into normalized unit-cube coordinates).
    tr_center_ = toUnit(best_x_);
    clampUnit(tr_center_);

    // Keep the dataset from growing unbounded across repeated restarts.
    if ((int)X_.size() > max_dataset_) {
        const int drop = (int)X_.size() - max_dataset_;
        X_.erase(X_.begin(), X_.begin() + drop);
        Y_.erase(Y_.begin(), Y_.begin() + drop);
    }
}

void TuRBO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    X_.clear();
    Y_.clear();
    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    if (fail_tol_ <= 0)     fail_tol_ = std::max(4, D);
    if (n_candidates_ <= 0) n_candidates_ = std::max(100, 50 * D <= 5000 ? 50 * D : 5000);
    if (batch_size_ < 1)    batch_size_ = 1;
    if (bo_budget_ <= 0) {
        // Literature-consistent BO budget: a few hundred evaluations,
        // scaling modestly with D, capped at whatever max_evals_ actually
        // is (so a deliberately tiny budget configured by the user is
        // still respected rather than overridden upward).
        bo_budget_ = std::min<long long>(max_evals_, 20LL * D + 200);
    }
    in_fallback_ = false;

    tr_length_  = 0.8;
    succ_count_ = 0;
    fail_count_ = 0;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const int n0 = (n_init_ > 0) ? n_init_ : std::max(2 * D + 1, 8);

    for (int i = 0; i < n0 && prob_->calls() < max_evals_; ++i) {
        Vec u(D);
        for (int j = 0; j < D; ++j) u[j] = U01(rng_);
        const double f = safeEvalUnit(u);
        X_.push_back(u);
        Y_.push_back(f);
    }

    if (X_.empty()) {
        // Degenerate (essentially zero eval budget): fall back to the box center.
        Vec u(D, 0.5);
        const double f = safeEvalUnit(u);
        X_.push_back(u);
        Y_.push_back(f);
    }

    tr_center_ = toUnit(best_x_);
    clampUnit(tr_center_);

    Vec fx(Y_);
    updateStop(fx);
    printBest();
}

void TuRBO::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    if (prob_->calls() >= bo_budget_) {
        // GP-based search budget exhausted: keep making progress cheaply
        // (O(D) per eval, no GP fit) for the rest of the run instead of
        // continuing to pay O(n^3) GP fits forever.
        in_fallback_ = true;
        const long long remaining = max_evals_ - prob_->calls();
        const int steps = (int)std::max<long long>(1, std::min<long long>(50, remaining));
        for (int s = 0; s < steps && prob_->calls() < max_evals_; ++s) {
            cheapFallbackStep();
        }
        Vec fx{best_f_};
        printBest();
        updateStop(fx);
        return;
    }

    const double f_before = best_f_;

    if (!fitGP()) {
        // Not enough data / numerically degenerate: fall back to a plain
        // random exploration step within the trust region this round.
    }

    std::vector<Vec> cands;
    generateCandidates(n_candidates_, cands);

    std::vector<Vec> chosen;
    thompsonSelect(cands, batch_size_, chosen);
    if (chosen.empty() && !cands.empty()) chosen.push_back(cands.front());

    std::vector<double> fx;
    fx.reserve(chosen.size());
    for (const Vec& u : chosen) {
        if (prob_->calls() >= max_evals_) break;
        const double f = safeEvalUnit(u);
        X_.push_back(u);
        Y_.push_back(f);
        fx.push_back(f);
    }

    // Cap dataset size (keep the most recent points -- the trust region
    // has typically moved on from much older ones anyway).
    if ((int)X_.size() > max_dataset_) {
        const int drop = (int)X_.size() - max_dataset_;
        X_.erase(X_.begin(), X_.begin() + drop);
        Y_.erase(Y_.begin(), Y_.begin() + drop);
    }

    // --- trust-region length adaptation ---
    if (best_f_ < f_before - 1e-12) {
        ++succ_count_;
        fail_count_ = 0;
    } else {
        ++fail_count_;
        succ_count_ = 0;
    }

    if (succ_count_ >= succ_tol_) {
        tr_length_ = std::min(length_max_, tr_length_ * 2.0);
        succ_count_ = 0;
    } else if (fail_count_ >= fail_tol_) {
        tr_length_ = tr_length_ * 0.5;
        fail_count_ = 0;
    }

    // Re-center on the (possibly updated) global incumbent.
    tr_center_ = toUnit(best_x_);
    clampUnit(tr_center_);

    if (tr_length_ < length_min_ && prob_->calls() < max_evals_) {
        restartTrustRegion();
    }

    // Optional in-run local search after a successful global-best improvement.
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
            }
        }
    }

    if (fx.empty()) fx.push_back(best_f_);
    printBest();
    updateStop(fx);
}

void TuRBO::end() {
    if (!end_local_refine_)        return;
    if (!prob_)                    return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }
    printBest();
}

} // namespace optimsolution
