#include "eo.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void EO::ensureBounds(Vec& v){
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

void EO::buildEquilibriumPool(std::vector<Vec>& pool) const{
    pool.clear();
    const int N = (int)X_.size();
    if (N <= 0) return;

    // Indices sorted by fitness (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    const int D = prob_->dimension();
    const int k = std::min(4, N);
    pool.reserve(5);

    Vec mean(D, 0.0);
    for (int i=0; i<k; ++i){
        pool.push_back(X_[idx[i]]);
        for (int d=0; d<D; ++d) mean[d] += X_[idx[i]][d];
    }
    if (k > 0){
        for (int d=0; d<D; ++d) mean[d] /= (double)k;
        pool.push_back(std::move(mean));
    }
}

void EO::init(){
    if (!prob_) return;

    iter_ = 0;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear();
    FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_eo_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[eo] cfg -> N=%d, a1=%.4f, a2=%.4f, gp=%.3f, in-run: %s @ %.3f, final@end: %s (%s)\n",
            N, a1_, a2_, gp_, lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void EO::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Iteration-based progress (stable for any population size)
    const double T_est = std::max(1.0, std::floor((double)max_evals_ / (double)std::max(1, N)));
    const double progress = std::min(1.0, std::max(0.0, (double)iter_ / T_est));
    const double t = std::pow(std::max(0.0, 1.0 - progress), a2_ * progress + 1e-12);

    // Build equilibrium pool: 4 best + mean => size 5
    std::vector<Vec> pool;
    buildEquilibriumPool(pool);
    if (pool.empty()) return;
    const int poolSz = (int)pool.size();
    std::uniform_int_distribution<int> pickPool(0, std::max(0, poolSz-1));

    std::vector<Vec>    newX = X_;
    std::vector<double> newF = FX_;

    for (int i=0; i<N; ++i){
        const Vec& Xi = X_[i];
        const Vec& Xeq = pool[pickPool(rng_)];

        Vec xnew(D, 0.0);
        for (int d=0; d<D; ++d){
            double lambda = U01(rng_);
            if (lambda < 1e-12) lambda = 1e-12;

            const double r = U01(rng_);
            const double sgn = (r >= 0.5 ? 1.0 : -1.0);
            const double F = a1_ * sgn * (std::exp(-lambda * t) - 1.0);

            // Generation rate control probability (canonical)
            const double r1 = U01(rng_);
            const double r2 = U01(rng_);
            const double GCP = (r2 < gp_) ? (0.5 * r1) : 0.0;

            const double G0 = GCP * (Xeq[d] - lambda * Xi[d]);
            const double G  = G0 * F;

            xnew[d] = Xeq[d] + (Xi[d] - Xeq[d]) * F + (G / lambda) * (1.0 - F);
        }

        ensureBounds(xnew);
        double fnew = eval(xnew);

        // Optional in-run local refinement on improvements
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (fnew < FX_[i] && U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew){
                    xnew = std::move(xloc);
                    fnew = floc;
                }
            }
        }

        newX[i] = std::move(xnew);
        newF[i] = fnew;

        if (newF[i] < best_f_){
            best_f_ = newF[i];
            best_x_ = newX[i];
        }

        if (prob_->calls() >= max_evals_) break;
    }

    X_.swap(newX);
    FX_.swap(newF);

    // Elitism: write the best into the worst slot
    if (!FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val){ worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    ++iter_;

    printBest();
    updateStop(FX_);
}

void EO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_){
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Ensure refined best is present in the population
    if (!X_.empty() && !FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val){ worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    printBest();
}

} // namespace optimsolution
