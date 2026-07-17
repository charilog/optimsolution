#include "rembo.h"
#include <numeric>
#include <limits>

namespace optimsolution {

namespace {
    // Same self-contained dense linear-algebra helpers as TuRBO (this
    // codebase has no shared linear-algebra module; every method file here
    // is fully self-contained).
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

    inline double normPdf(double z) {
        static const double inv_sqrt_2pi = 0.3989422804014327;
        return inv_sqrt_2pi * std::exp(-0.5 * z * z);
    }
    inline double normCdf(double z) {
        return 0.5 * (1.0 + std::erf(z / 1.4142135623730951));
    }
}

void REMBO::configure(const MethodConfig& mc) {
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    d_embed_        = mc.getInt("d_embed", d_embed_);
    d_embed_max_    = mc.getInt("d_embed_max", d_embed_max_);
    n_init_         = mc.getInt("n_init", n_init_);
    max_dataset_    = mc.getInt("max_dataset", max_dataset_);
    noise_var_      = mc.getDbl("noise_var", noise_var_);
    ei_candidates_  = mc.getInt("ei_candidates", ei_candidates_);
    ei_polish_steps_= mc.getInt("ei_polish_steps", ei_polish_steps_);
    ei_polish_sigma_= mc.getDbl("ei_polish_sigma", ei_polish_sigma_);
    bo_budget_      = mc.getInt("bo_budget", (int)bo_budget_);
    fallback_sigma_ = mc.getDbl("fallback_sigma", fallback_sigma_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

REMBO::Vec REMBO::embedToX(const Vec& y) const {
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    Vec x(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int k = 0; k < d_embed_; ++k) s += A_[i][k] * y[k];
        x[i] = s;
    }

    // Map from the embedding's natural (roughly [-1,1]-ish, since A has
    // i.i.d. N(0,1) entries and y is confined to [-sqrt(d_e),sqrt(d_e)])
    // scale into the real box, then clip -- the standard, practical
    // REMBO projection-back step.
    for (int i = 0; i < D; ++i) {
        const double lo = L[i], hi = U[i];
        const double mid = 0.5 * (lo + hi);
        const double half = 0.5 * (hi - lo);
        double v = mid + x[i] * half;
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        x[i] = v;
    }
    return x;
}

double REMBO::safeEvalY(const Vec& y) {
    Vec x = embedToX(y);
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

double REMBO::kernel(const Vec& a, const Vec& b) const {
    double r2 = 0.0;
    for (int k = 0; k < d_embed_; ++k) {
        const double d = a[k] - b[k];
        r2 += d * d;
    }
    const double r = std::sqrt(r2) / std::max(1e-12, lengthscale_);
    const double sqrt5 = 2.2360679774997896;
    const double poly = 1.0 + sqrt5 * r + (5.0 / 3.0) * r * r;
    return signal_var_ * poly * std::exp(-sqrt5 * r);
}

bool REMBO::fitGP() {
    const int n = (int)Y_pts_.size();
    gp_ready_ = false;
    if (n < 2) return false;

    {
        std::vector<double> dists;
        dists.reserve((size_t)n * (n - 1) / 2);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double r2 = 0.0;
                for (int k = 0; k < d_embed_; ++k) {
                    const double d = Y_pts_[i][k] - Y_pts_[j][k];
                    r2 += d * d;
                }
                dists.push_back(std::sqrt(r2));
            }
        }
        std::sort(dists.begin(), dists.end());
        double med = dists[dists.size() / 2];
        if (!(med > 1e-8)) med = 0.5;
        lengthscale_ = std::max(1e-3, med);

        double fmean = 0.0;
        for (double f : F_) fmean += f;
        fmean /= (double)n;
        double fvar = 0.0;
        for (double f : F_) { const double d = f - fmean; fvar += d * d; }
        fvar /= std::max(1, n - 1);
        f_mean_ = fmean;
        f_std_  = std::sqrt(std::max(1e-12, fvar));
        signal_var_ = 1.0;
    }

    Vec f_std(n);
    double best_std = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        f_std[i] = (F_[i] - f_mean_) / f_std_;
        if (f_std[i] < best_std) best_std = f_std[i];
    }
    f_best_std_ = best_std;

    double jitter = noise_var_;
    for (int attempt = 0; attempt < 6; ++attempt) {
        Mat K(n, Vec(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double kij = kernel(Y_pts_[i], Y_pts_[j]);
                if (i == j) kij += jitter;
                K[i][j] = kij;
                K[j][i] = kij;
            }
        }
        if (choleskyInPlace(K, n)) {
            L_chol_ = std::move(K);
            alpha_  = choleskySolve(L_chol_, f_std, n);
            gp_ready_ = true;
            return true;
        }
        jitter = (jitter > 0.0) ? jitter * 10.0 : 1e-8;
    }
    return false;
}

void REMBO::predictGP(const Vec& yq, double& mean, double& var) const {
    const int n = (int)Y_pts_.size();
    if (!gp_ready_ || n == 0) {
        mean = 0.0; // standardized-space prior mean
        var  = signal_var_ + noise_var_;
        return;
    }
    Vec kstar(n);
    for (int i = 0; i < n; ++i) kstar[i] = kernel(yq, Y_pts_[i]);

    double m = 0.0;
    for (int i = 0; i < n; ++i) m += kstar[i] * alpha_[i];

    Vec v = forwardSolve(L_chol_, kstar, n);
    double vtv = 0.0;
    for (double vi : v) vtv += vi * vi;

    double kxx = kernel(yq, yq);
    var  = std::max(1e-12, kxx - vtv);
    mean = m;
}

double REMBO::expectedImprovement(const Vec& yq) const {
    double mean, var;
    predictGP(yq, mean, var);
    const double sigma = std::sqrt(std::max(1e-12, var));
    // Standardized space: minimizing, improvement = f_best_std_ - mean.
    const double imp = f_best_std_ - mean;
    if (sigma < 1e-9) return std::max(0.0, imp);
    const double z = imp / sigma;
    return imp * normCdf(z) + sigma * normPdf(z);
}

REMBO::Vec REMBO::proposeNextY() {
    std::uniform_real_distribution<double> Uy(-y_box_, y_box_);
    std::normal_distribution<double> N01(0.0, 1.0);

    Vec best_y(d_embed_, 0.0);
    double best_ei = -std::numeric_limits<double>::infinity();

    // Random search phase.
    for (int c = 0; c < ei_candidates_; ++c) {
        Vec y(d_embed_);
        for (int k = 0; k < d_embed_; ++k) y[k] = Uy(rng_);
        const double ei = expectedImprovement(y);
        if (ei > best_ei) { best_ei = ei; best_y = y; }
    }

    // Local polish: perturb the best random candidate a few times, keep
    // improvements (a lightweight, gradient-free stand-in for a proper
    // inner EI maximizer).
    const double sigma = ei_polish_sigma_ * y_box_;
    for (int s = 0; s < ei_polish_steps_; ++s) {
        Vec cand = best_y;
        for (int k = 0; k < d_embed_; ++k) {
            cand[k] += sigma * N01(rng_);
            if (cand[k] < -y_box_) cand[k] = -y_box_;
            if (cand[k] >  y_box_) cand[k] =  y_box_;
        }
        const double ei = expectedImprovement(cand);
        if (ei > best_ei) { best_ei = ei; best_y = cand; }
    }
    return best_y;
}

// Cheap (O(D) per eval, no GP fit) local refinement for any evaluation
// budget remaining once bo_budget_ is exhausted -- same rationale and
// design as TuRBO's cheapFallbackStep(), operating directly in ORIGINAL
// problem coordinates (bypassing the random embedding entirely, since once
// the incumbent has been found there is no reason to keep restricting
// further local refinement to the low-dimensional subspace).
void REMBO::cheapFallbackStep() {
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

void REMBO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    d_embed_ = (d_embed_ > 0) ? std::min(d_embed_, D) : std::min(d_embed_max_, D);
    if (d_embed_ < 1) d_embed_ = 1;
    y_box_ = std::sqrt((double)d_embed_);

    if (bo_budget_ <= 0) {
        bo_budget_ = std::min<long long>(max_evals_, 20LL * d_embed_ + 200);
    }
    in_fallback_ = false;

    // Draw the random embedding matrix A (D x d_embed_), i.i.d. N(0,1).
    std::normal_distribution<double> N01(0.0, 1.0);
    A_.assign(D, Vec(d_embed_, 0.0));
    for (int i = 0; i < D; ++i)
        for (int k = 0; k < d_embed_; ++k)
            A_[i][k] = N01(rng_);

    Y_pts_.clear();
    F_.clear();
    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<double> Uy(-y_box_, y_box_);
    const int n0 = (n_init_ > 0) ? n_init_ : std::max(2 * d_embed_ + 1, 6);

    for (int i = 0; i < n0 && prob_->calls() < max_evals_; ++i) {
        Vec y(d_embed_);
        for (int k = 0; k < d_embed_; ++k) y[k] = Uy(rng_);
        const double f = safeEvalY(y);
        Y_pts_.push_back(y);
        F_.push_back(f);
    }

    if (Y_pts_.empty()) {
        Vec y(d_embed_, 0.0);
        const double f = safeEvalY(y);
        Y_pts_.push_back(y);
        F_.push_back(f);
    }

    Vec fx(F_);
    updateStop(fx);
    printBest();
}

void REMBO::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    if (prob_->calls() >= bo_budget_) {
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

    fitGP(); // if this fails (too little/degenerate data), proposeNextY()
             // falls back to its prior-based EI, which reduces to
             // undirected random search -- a safe, non-crashing degradation.

    Vec y_next = proposeNextY();
    const double f = safeEvalY(y_next);
    Y_pts_.push_back(y_next);
    F_.push_back(f);

    if ((int)Y_pts_.size() > max_dataset_) {
        const int drop = (int)Y_pts_.size() - max_dataset_;
        Y_pts_.erase(Y_pts_.begin(), Y_pts_.begin() + drop);
        F_.erase(F_.begin(), F_.begin() + drop);
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

    Vec fx{f};
    printBest();
    updateStop(fx);
}

void REMBO::end() {
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
