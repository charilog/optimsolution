#include "fa.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void FA::ensureBounds(Vec& v){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(v[j])) v[j] = 0.5*(lo + hi);
        if (v[j] < lo) v[j] = lo;
        if (v[j] > hi) v[j] = hi;
    }
}

void FA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));

    this->setPopulation(N);

    X_.clear();
    FX_.clear();
    it_ = 0;
    alpha_ = alpha0_;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_fa_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[fa] cfg -> N=%d (population() now=%d, override=%d), beta0=%.6f, gamma=%.6f, alpha0=%.6f, alpha_damp=%.6f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, beta0_, gamma_, alpha0_, alpha_damp_,
            lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void FA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // A synchronous update is used to avoid order bias within the iteration.
    const std::vector<Vec>    Xold  = X_;
    const std::vector<double> Fold  = FX_;

    for (int i=0; i<N; ++i){
        Vec xi = Xold[i];

        for (int j=0; j<N; ++j){
            if (Fold[j] >= Fold[i]) continue;

            // Euclidean distance squared
            double r2 = 0.0;
            for (int k=0; k<D; ++k){
                const double d = xi[k] - Xold[j][k];
                r2 += d*d;
            }

            const double beta = beta0_ * std::exp(-gamma_ * r2);

            for (int k=0; k<D; ++k){
                double lo = (k < (int)L.size() ? L[k] : -1.0);
                double hi = (k < (int)U.size() ? U[k] :  1.0);
                if (lo > hi) std::swap(lo, hi);
                const double span = (std::isfinite(lo) && std::isfinite(hi)) ? (hi - lo) : 1.0;
                const double eps  = (U01(rng_) - 0.5) * span;

                xi[k] = xi[k] + beta * (Xold[j][k] - xi[k]) + alpha_ * eps;
            }
        }

        ensureBounds(xi);

        double fi = eval(xi);
        if (local_rate_ > 0.0 && !local_method_.empty()) {
            if (U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, xi);
                if (std::isfinite(floc) && floc < fi) {
                    xi = std::move(xloc);
                    fi = floc;
                }
            }
        }

        X_[i]  = std::move(xi);
        FX_[i] = fi;

        if (fi < best_f_) {
            best_f_ = fi;
            best_x_ = X_[i];
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Dampen alpha after each iteration (classic FA uses a decreasing randomization factor).
    if (alpha_damp_ < 1.0) {
        alpha_ *= alpha_damp_;
        if (alpha_ < 0.0) alpha_ = 0.0;
    }
    ++it_;

    printBest();
    updateStop(FX_);
}

void FA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // The refined best is written to the worst position.
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
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
