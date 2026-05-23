#include "de.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static inline std::string to_lower(std::string s){
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void DE::init(){
    if (!prob_) return;

    const int D = prob_->dimension();

    // If an override exists in [de], it takes precedence; otherwise [global] population() is used
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));

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

    if (debug_de_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[de] cfg -> N=%d (population() now=%d, override=%d), F=%.6f, CR=%.6f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, F_, CR_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

int DE::pickDistinct(int n, int a, int b, int c){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a || r==b || r==c);
    return r;
}

void DE::ensureBounds(Vec& v){
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

void DE::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();

    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::uniform_int_distribution<int>     Jrand(0, std::max(0, D-1));

    // DE/rand/1/bin
    for (int i=0; i<N; ++i){
        int r1 = pickDistinct(N, i);
        int r2 = pickDistinct(N, i, r1);
        int r3 = pickDistinct(N, i, r1, r2);

        // mutation
        Vec v(D, 0.0);
        for (int j=0; j<D; ++j){
            v[j] = X_[r1][j] + F_ * (X_[r2][j] - X_[r3][j]);
        }

        // crossover
        Vec u = X_[i];
        const int jr = Jrand(rng_);
        for (int j=0; j<D; ++j){
            if (U01(rng_) < CR_ || j == jr){
                u[j] = v[j];
            }
        }
        ensureBounds(u);

        // selection (+ optional in-run local)
        double fu = eval(u);
        if (fu < FX_[i]) {
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

void DE::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // The refined best is written to the worst position (as in GA/BHO)
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
