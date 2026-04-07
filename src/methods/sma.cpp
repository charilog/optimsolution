#include "sma.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void SMA::init(){
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
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_sma_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[sma] cfg -> N=%d (population() now=%d, override=%d), z=%.4f, a_cap=%.3f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, z_, a_cap_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

int SMA::pickDistinct(int n, int a){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a);
    return r;
}

void SMA::ensureBounds(Vec& v){
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

void SMA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Iteration-based progress estimate
    const double T_est = std::max(1.0, std::floor((double)max_evals_ / (double)std::max(1, N)));
    const double progress = std::min(1.0, std::max(0.0, (double)iter_ / T_est));

    // Sort indices by fitness (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    const int best_i = idx.front();
    const int worst_i = idx.back();
    const double fbest = FX_[best_i];
    const double fworst = FX_[worst_i];

    // Update global best
    if (fbest < best_f_) {
        best_f_ = fbest;
        best_x_ = X_[best_i];
    }

    // Compute weights W_i (rank-based)
    std::vector<double> W(N, 1.0);
    const double denom = (fworst - fbest);
    for (int r=0; r<N; ++r){
        const int i = idx[r];
        double s = 1.0;
        if (std::isfinite(denom) && denom > 0.0 && std::isfinite(FX_[i])) {
            const double ratio = (fworst - FX_[i]) / denom; // best->1, worst->0
            s = std::log10(ratio + 1.0);
            if (!std::isfinite(s)) s = 1.0;
        }
        const double rr = U01(rng_);
        if (r < N/2) W[i] = 1.0 + rr * s;
        else         W[i] = 1.0 - rr * s;
    }

    // a and b schedules
    double x = 1.0 - progress;
    if (x >= 1.0) x = 0.999999;
    if (x < 0.0)  x = 0.0;
    double a = std::atanh(x);
    if (!std::isfinite(a)) a = a_cap_;
    if (a > a_cap_) a = a_cap_;
    const double b = 1.0 - progress;

    // Candidate generation (greedy replacement for stability)
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int i=0; i<N; ++i){
        Vec cand = X_[i];

        // Probability p_i based on fitness distance from best
        double p = 0.0;
        if (std::isfinite(FX_[i]) && std::isfinite(best_f_)) {
            p = std::tanh(std::fabs(FX_[i] - best_f_));
            if (!std::isfinite(p)) p = 0.0;
            if (p < 0.0) p = 0.0;
            if (p > 1.0) p = 1.0;
        }

        const int r1 = pickDistinct(N, i);
        const int r2 = pickDistinct(N, i);

        for (int j=0; j<D; ++j){
            const double lo = (j < (int)L.size() ? L[j] : -1.0);
            const double hi = (j < (int)U.size() ? U[j] :  1.0);
            const double range = (hi - lo);

            if (U01(rng_) < z_) {
                cand[j] = lo + U01(rng_) * range;
                continue;
            }

            std::uniform_real_distribution<double> Uvb(-a, a);
            std::uniform_real_distribution<double> Uvc(-b, b);
            const double vb = Uvb(rng_);
            const double vc = Uvc(rng_);

            if (U01(rng_) < p) {
                cand[j] = best_x_[j] + vb * (W[i] * X_[r1][j] - X_[r2][j]);
            } else {
                // Mild attraction/repulsion around the best (stable exploitation)
                cand[j] = X_[i][j] + vc * (best_x_[j] - X_[i][j]);
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

            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Elitism
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    ++iter_;
    printBest();
    updateStop(FX_);
}

void SMA::end(){
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
