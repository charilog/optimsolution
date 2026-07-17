#include "mscso.h"
#include "init.h"
#include <cmath>
#include <limits>

namespace optimsolution {

namespace { constexpr double MSCSO_PI = 3.141592653589793238462643383279502884; }

void MSCSO::goodPointSet(int n, int z, std::vector<std::vector<double>>& F) const {
    F.assign(std::max(0, n), std::vector<double>(std::max(0, z), 0.0));
    if (z <= 0 || n <= 0) return;

    // Step 1 (Eq. 9): smallest prime p with z <= (p-3)/2  <=>  p >= 2z+3
    auto is_prime = [](int v) {
        if (v < 2) return false;
        for (int d = 2; (long long)d * d <= v; ++d) {
            if (v % d == 0) return false;
        }
        return true;
    };
    int p = 2 * z + 3;
    while (!is_prime(p)) ++p;

    // Step 2 (Eq. 10): r_j = 2*cos(2*pi*j/p), j = 1..z
    std::vector<double> r(z);
    for (int j = 1; j <= z; ++j) {
        r[j - 1] = 2.0 * std::cos(2.0 * MSCSO_PI * (double)j / (double)p);
    }

    // Step 3 (Eq. 11): good point set F_k = ({k*r_1}, ..., {k*r_z}), k = 1..n
    // ({.} denotes fractional part, giving a point in the unit cube [0,1]^z)
    for (int k = 1; k <= n; ++k) {
        for (int j = 0; j < z; ++j) {
            const double v = (double)k * r[j];
            F[k - 1][j] = v - std::floor(v);
        }
    }
}

void MSCSO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    X_.assign(pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    Xbest_.assign(D, 0.0);
    Fbest_ = std::numeric_limits<double>::infinity();

    // Good point set population initialization (Eq. 12: x = lb + F*(ub-lb)).
    std::vector<std::vector<double>> F;
    goodPointSet(pop_, D, F);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < pop_; ++i) {
        for (int j = 0; j < D; ++j) {
            double lo = L[j], hi = U[j];
            if (!std::isfinite(lo) || !std::isfinite(hi)) { lo = -10.0; hi = 10.0; }
            X_[i][j] = lo + F[i][j] * (hi - lo);
        }
        FX_[i] = eval(X_[i]);
        if (prob_->calls() >= max_evals_) break;
    }

    Xbest_ = X_[0];
    Fbest_ = FX_[0];
    for (int i = 1; i < pop_; ++i) {
        if (FX_[i] < Fbest_) { Fbest_ = FX_[i]; Xbest_ = X_[i]; }
    }

    best_x_ = Xbest_;
    best_f_ = Fbest_;
}

void MSCSO::ensureBounds(std::vector<double>& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void MSCSO::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int n = pop_;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Uang(1.0, 360.0);   // paper Table 2: P = [1,360]
    std::uniform_real_distribution<double> Uk(-1.0, 1.0);      // paper: K in [-1,1]
    std::normal_distribution<double>       Nstd(0.0, 1.0);     // beta ~ N(0,1) (SSA convention)

    const double t    = (double)iters_;
    const double Tmax = (double)std::max(1, max_iters_);

    // --- Nonlinear adjustment strategy (Eqs. 13-14) ---
    const double f_nl = std::sin((MSCSO_PI / 4.0) * (t / Tmax));
    const double rg   = SM_ - (SM_ * t / Tmax) * f_nl;

    // --- R, r for this iteration (Eqs. 4-5); shared by the whole population ---
    const double R      = 2.0 * rg * U01(rng_) - rg;
    const double r_sens = rg - U01(rng_);

    // --- Main SCSO position update (Algorithm 1 / Section 2.2-2.3) ---
    for (int i = 0; i < n; ++i) {
        const double theta_rad = Uang(rng_) * (MSCSO_PI / 180.0);
        const double c_theta   = std::cos(theta_rad);

        std::vector<double> xnew(D);
        if (std::fabs(R) <= 1.0) {
            // Attack / exploitation (Eqs. 7-8)
            for (int j = 0; j < D; ++j) {
                const double xrnd = std::fabs(U01(rng_) * (Xbest_[j] - X_[i][j]));
                xnew[j] = Xbest_[j] - r_sens * xrnd * c_theta;
            }
        } else {
            // Search / exploration (Eq. 6, corrected form -- see header note)
            for (int j = 0; j < D; ++j) {
                xnew[j] = r_sens * (Xbest_[j] - U01(rng_) * X_[i][j]);
            }
        }

        ensureBounds(xnew);
        const double fnew = eval(xnew);

        X_[i]  = std::move(xnew);
        FX_[i] = fnew;

        if (fnew < Fbest_) { Fbest_ = fnew; Xbest_ = X_[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    // Worst individual snapshot for this iteration (x_worst^t, f_w), used by
    // the warning mechanism below.
    int worst_idx = 0;
    double fworst = FX_[0];
    for (int i = 1; i < n; ++i) {
        if (FX_[i] > fworst) { fworst = FX_[i]; worst_idx = i; }
    }

    // --- Sparrow early-warning mechanism (Eq. 15), Section 3.3 ---
    const int nwarn = std::max(1, (int)std::lround(warning_frac_ * n));
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng_);

    for (int w = 0; w < nwarn; ++w) {
        const int i = idx[w];
        std::vector<double> xcand(D);

        if (FX_[i] > Fbest_) {
            // "Safe around x_b": random move around the current global best.
            const double beta = Nstd(rng_);
            for (int j = 0; j < D; ++j) {
                xcand[j] = Xbest_[j] + beta * std::fabs(X_[i][j] - Xbest_[j]);
            }
        } else {
            // At/near the global best: nudge away from the current worst.
            const double K = Uk(rng_);
            const double denom = std::fabs(FX_[i] - fworst) + eps_;
            for (int j = 0; j < D; ++j) {
                xcand[j] = X_[i][j] + K * ((X_[i][j] - X_[worst_idx][j]) / denom);
            }
        }

        ensureBounds(xcand);
        const double fcand = eval(xcand);

        // Step 21 of Algorithm 2: accept only if the candidate is better.
        if (fcand < FX_[i]) {
            X_[i]  = std::move(xcand);
            FX_[i] = fcand;
            if (fcand < Fbest_) { Fbest_ = fcand; Xbest_ = X_[i]; }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    best_x_ = Xbest_;
    best_f_ = Fbest_;

    // Optional in-run local search after a successful global-best improvement
    // (same convention as DE/PSO in this framework).
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
                Xbest_  = best_x_;
                Fbest_  = best_f_;

                if (!X_.empty()) {
                    int wi = 0; double wv = FX_[0];
                    for (int i = 1; i < n; ++i) if (FX_[i] > wv) { wv = FX_[i]; wi = i; }
                    X_[wi] = best_x_;
                    FX_[wi] = best_f_;
                }
            }
        }
    }

    printBest();
    updateStop(FX_);
}

void MSCSO::end() {
    // Executed at the end. Controlled ONLY by [global]. (same pattern as DE/PSO)
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

    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }
    printBest();
}

} // namespace optimsolution
