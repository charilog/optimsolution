#include "tlbo.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void TLBO::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(3, (pop_override_ >= 3 ? pop_override_ : population()));
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

    if (debug_tlbo_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[tlbo] cfg -> N=%d (population() now=%d, override=%d), TF_fixed=%d, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, teaching_factor_fixed_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

int TLBO::pickDistinct(int n, int a){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a);
    return r;
}

void TLBO::ensureBounds(Vec& v){
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

TLBO::Vec TLBO::meanVector() const {
    const int N = (int)X_.size();
    if (N <= 0) return Vec{};
    const int D = (int)X_[0].size();
    Vec m(D, 0.0);
    for (int i=0; i<N; ++i){
        for (int j=0; j<D; ++j) m[j] += X_[i][j];
    }
    const double invN = 1.0 / (double)N;
    for (int j=0; j<D; ++j) m[j] *= invN;
    return m;
}

void TLBO::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::uniform_int_distribution<int>     TFdist(1,2);

    auto currentTeacherIndex = [&]()->int{
        int best_i = 0;
        double best_f = FX_[0];
        for (int i=1; i<N; ++i){
            if (FX_[i] < best_f) { best_f = FX_[i]; best_i = i; }
        }
        return best_i;
    };

    // Phase 1: Teacher phase
    {
        const int t = currentTeacherIndex();
        const Vec teacher = X_[t];
        const Vec mean = meanVector();

        for (int i=0; i<N; ++i){
            int TF = teaching_factor_fixed_;
            if (TF != 1 && TF != 2) TF = TFdist(rng_);

            Vec cand = X_[i];
            for (int j=0; j<D; ++j){
                const double r = U01(rng_);
                cand[j] = cand[j] + r * (teacher[j] - (double)TF * mean[j]);
            }
            ensureBounds(cand);

            double fc = eval(cand);
            if (fc < FX_[i]) {
                if (local_rate_ > 0.0 && !local_method_.empty()){
                    if (U01(rng_) < local_rate_){
                        auto [xloc, floc] = localSearch(local_method_, cand);
                        if (std::isfinite(floc) && floc < fc){
                            cand = std::move(xloc);
                            fc   = floc;
                        }
                    }
                }
                X_[i]  = std::move(cand);
                FX_[i] = fc;

                if (FX_[i] < best_f_){
                    best_f_ = FX_[i];
                    best_x_ = X_[i];
                }
            }

            if (prob_->calls() >= max_evals_) break;
        }
    }

    if (prob_->calls() < max_evals_) {
        // Phase 2: Learner phase
        for (int i=0; i<N; ++i){
            const int jidx = pickDistinct(N, i);

            Vec cand = X_[i];
            if (FX_[i] < FX_[jidx]) {
                for (int j=0; j<D; ++j){
                    const double r = U01(rng_);
                    cand[j] = cand[j] + r * (X_[i][j] - X_[jidx][j]);
                }
            } else {
                for (int j=0; j<D; ++j){
                    const double r = U01(rng_);
                    cand[j] = cand[j] + r * (X_[jidx][j] - X_[i][j]);
                }
            }
            ensureBounds(cand);

            double fc = eval(cand);
            if (fc < FX_[i]) {
                if (local_rate_ > 0.0 && !local_method_.empty()){
                    if (U01(rng_) < local_rate_){
                        auto [xloc, floc] = localSearch(local_method_, cand);
                        if (std::isfinite(floc) && floc < fc){
                            cand = std::move(xloc);
                            fc   = floc;
                        }
                    }
                }
                X_[i]  = std::move(cand);
                FX_[i] = fc;

                if (FX_[i] < best_f_){
                    best_f_ = FX_[i];
                    best_x_ = X_[i];
                }
            }

            if (prob_->calls() >= max_evals_) break;
        }
    }

    printBest();
    updateStop(FX_);
}

void TLBO::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

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
