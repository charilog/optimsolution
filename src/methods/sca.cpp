#include "sca.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void SCA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();

    // If an override exists in [sca], it takes precedence; otherwise [global] population() is used
    const int N = std::max(3, (pop_override_ >= 3 ? pop_override_ : population()));

    // Synchronization is performed here as well for full consistency with the reporter
    this->setPopulation(N);

    X_.clear(); FX_.clear();

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

    if (debug_sca_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[sca] cfg -> N=%d (population() now=%d, override=%d), a=%.6f, r3_max=%.6f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, a_, r3_max_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void SCA::ensureBounds(Vec& v){
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

void SCA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double TWO_PI = 2.0 * std::acos(-1.0);
    std::uniform_real_distribution<double> U_r2(0.0, TWO_PI);
    std::uniform_real_distribution<double> U_r3(0.0, r3_max_);

    // Progress proxy: uses evaluation budget, because SCA is typically iteration-based but optimsolution may stop by evals
    double progress = 0.0;
    if (max_evals_ > 0) {
        progress = (double)prob_->calls() / (double)max_evals_;
        if (progress < 0.0) progress = 0.0;
        if (progress > 1.0) progress = 1.0;
    }

    // r1 decreases linearly from a_ to 0 (as in the original SCA paper)
    const double r1 = a_ * (1.0 - progress);

    for (int i=0; i<N; ++i){
        Vec xnew = X_[i];

        for (int j=0; j<D; ++j){
            const double r2 = U_r2(rng_);
            const double r3 = U_r3(rng_);
            const double r4 = U01(rng_);

            const double diff = std::abs(r3 * best_x_[j] - X_[i][j]);
            if (r4 < 0.5) {
                xnew[j] = X_[i][j] + r1 * std::sin(r2) * diff;
            } else {
                xnew[j] = X_[i][j] + r1 * std::cos(r2) * diff;
            }
        }

        ensureBounds(xnew);

        double fnew = eval(xnew);

        // Optional in-run local refinement
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew){
                    xnew = std::move(xloc);
                    fnew = floc;
                }
            }
        }

        // Position update (standard SCA: always move)
        X_[i]  = std::move(xnew);
        FX_[i] = fnew;

        if (fnew < best_f_){
            best_f_ = fnew;
            best_x_ = X_[i];
        }

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void SCA::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // The refined best is written to the worst position (as in GA/BHO/DE)
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
