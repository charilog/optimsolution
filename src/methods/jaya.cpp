#include "jaya.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void JAYA::ensureBounds(Vec& v){
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

void JAYA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(3, (pop_override_ >= 3 ? pop_override_ : population()));

    // Keep the optimizer reporter consistent
    this->setPopulation(N);

    X_.clear();
    FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    // Evaluate initial population
    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_jaya_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[jaya] cfg -> N=%d (population() now=%d, override=%d), use_abs=%s, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, use_abs_ ? "on" : "off", lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void JAYA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 3) return;

    // Identify best and worst in the current population
    int best_i = 0;
    int worst_i = 0;
    for (int i=1; i<N; ++i){
        if (FX_[i] < FX_[best_i])  best_i  = i;
        if (FX_[i] > FX_[worst_i]) worst_i = i;
    }
    const Vec best = X_[best_i];
    const Vec worst = X_[worst_i];

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i=0; i<N; ++i){
        Vec u = X_[i];

        for (int j=0; j<D; ++j){
            const double r1 = U01(rng_);
            const double r2 = U01(rng_);
            const double xij = X_[i][j];
            const double ref = use_abs_ ? std::abs(xij) : xij;
            u[j] = xij + r1 * (best[j]  - ref) - r2 * (worst[j] - ref);
        }
        ensureBounds(u);

        double fu = eval(u);
        if (fu < FX_[i]) {
            // optional in-run local
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, u);
                    if (std::isfinite(floc) && floc < fu){
                        u  = std::move(xloc);
                        fu = floc;
                    }
                }
            }

            X_[i]  = std::move(u);
            FX_[i] = fu;

            if (FX_[i] < best_f_){
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void JAYA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best to the worst position
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
