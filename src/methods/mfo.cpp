#include "mfo.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static constexpr double kPI = 3.141592653589793238462643383279502884;

void MFO::ensureBounds(Vec& v){
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

void MFO::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
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
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    iter_ = 0;

    if (debug_mfo_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[mfo] cfg -> N=%d (population() now=%d, override=%d), b=%.4f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, b_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void MFO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    const int T_est = std::max(1, (int)(max_evals_ / std::max(1, N)));
    const double ratio = (T_est <= 1 ? 1.0 : std::min(1.0, (double)iter_ / (double)(T_est - 1)));

    // number of flames decreases linearly from N to 1
    int flame_no = (int)std::lround((double)N - ratio * (double)(N - 1));
    if (flame_no < 1) flame_no = 1;
    if (flame_no > N) flame_no = N;

    // Sort population -> flames
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    std::vector<Vec> flames;
    flames.reserve(N);
    for (int k=0; k<N; ++k) flames.push_back(X_[order[k]]);

    // Keep current global best
    if (FX_[order[0]] < best_f_) { best_f_ = FX_[order[0]]; best_x_ = X_[order[0]]; }

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i=0; i<N; ++i){
        if (prob_->calls() >= max_evals_) break;

        int jfl = i;
        if (jfl >= flame_no) jfl = flame_no - 1;
        if (jfl < 0) jfl = 0;

        const Vec& F = flames[jfl];

        Vec Xnew = X_[i];
        for (int j=0; j<D; ++j){
            double dist = std::fabs(F[j] - X_[i][j]);
            double l = 2.0 * U01(rng_) - 1.0; // [-1,1]
            Xnew[j] = dist * std::exp(b_ * l) * std::cos(2.0 * kPI * l) + F[j];
        }

        ensureBounds(Xnew);
        double fnew = eval(Xnew);

        // Greedy selection (+ optional local)
        if (fnew < FX_[i]) {
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, Xnew);
                    if (std::isfinite(floc) && floc < fnew){
                        Xnew = std::move(xloc);
                        fnew = floc;
                    }
                }
            }

            X_[i] = std::move(Xnew);
            FX_[i] = fnew;

            if (fnew < best_f_){
                best_f_ = fnew;
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // elitism: write best to worst
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    iter_++;

    printBest();
    updateStop(FX_);
}

void MFO::end(){
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
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    printBest();
}

} // namespace optimsolution
