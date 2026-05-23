#include "mvo.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void MVO::init(){
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

    if (debug_mvo_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[mvo] cfg -> N=%d (population() now=%d, override=%d), WEP=[%.3f..%.3f], TDR_power=%.3f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, wep_min_, wep_max_, tdr_power_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void MVO::ensureBounds(Vec& v){
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

int MVO::rouletteSelect(const std::vector<double>& w){
    if (w.empty()) return 0;
    double sum = 0.0;
    for (double x : w) sum += (x > 0.0 ? x : 0.0);
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        std::uniform_int_distribution<int> I(0, (int)w.size()-1);
        return I(rng_);
    }
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_) * sum;
    double acc = 0.0;
    for (int i=0; i<(int)w.size(); ++i){
        double wi = (w[i] > 0.0 ? w[i] : 0.0);
        acc += wi;
        if (acc >= r) return i;
    }
    return (int)w.size()-1;
}

void MVO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Iteration-based progress (avoids premature decay when N is large)
    const double T_est = std::max(1.0, std::floor((double)max_evals_ / (double)std::max(1, N)));
    const double progress = std::min(1.0, std::max(0.0, (double)iter_ / T_est));
    const double WEP = wep_min_ + progress * (wep_max_ - wep_min_);
    const double TDR = 1.0 - std::pow(progress, 1.0 / tdr_power_);

    // Best/Worst for normalization (minimization)
    double fbest = FX_[0], fworst = FX_[0];
    int best_i = 0;
    for (int i=1; i<N; ++i){
        if (FX_[i] < fbest) { fbest = FX_[i]; best_i = i; }
        if (FX_[i] > fworst) fworst = FX_[i];
    }

    // Normalized inflation rate: best -> 1, worst -> 0
    std::vector<double> NI(N, 1.0);
    if (std::isfinite(fworst) && std::isfinite(fbest) && (fworst - fbest) > 0.0) {
        const double denom = (fworst - fbest);
        for (int i=0; i<N; ++i){
            double v = (fworst - FX_[i]) / denom;
            if (!std::isfinite(v)) v = 0.0;
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            NI[i] = v;
        }
    }

    // Create new population
    std::vector<Vec>    newX = X_;
    std::vector<double> newF = FX_;

    const Vec& bestX = X_[best_i];
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int i=0; i<N; ++i){
        if (i == best_i) continue; // keep best as-is

        Vec cand = X_[i];

        // White-hole operator (copy dimensions from better universes)
        for (int j=0; j<D; ++j){
            if (U01(rng_) < NI[i]) {
                const int donor = rouletteSelect(NI);
                cand[j] = X_[donor][j];
            }
        }

        // Wormhole operator (exploit around the best)
        for (int j=0; j<D; ++j){
            if (U01(rng_) < WEP) {
                const double lo = (j < (int)L.size() ? L[j] : -1.0);
                const double hi = (j < (int)U.size() ? U[j] :  1.0);
                const double range = std::fabs(hi - lo);
                const double r = U01(rng_);
                if (U01(rng_) < 0.5) cand[j] = bestX[j] + TDR * range * r;
                else                 cand[j] = bestX[j] - TDR * range * r;
            }
        }

        ensureBounds(cand);

        double fc = eval(cand);
        if (fc < newF[i]) {
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, cand);
                    if (std::isfinite(floc) && floc < fc){
                        cand = std::move(xloc);
                        fc   = floc;
                    }
                }
            }
            newX[i] = std::move(cand);
            newF[i] = fc;
        }

        if (prob_->calls() >= max_evals_) break;
    }

    X_  = std::move(newX);
    FX_ = std::move(newF);

    // Update global best
    for (int i=0; i<N; ++i){
        if (FX_[i] < best_f_){
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
    }

    // Elitism: keep best inside the population for consistent reporting
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

void MVO::end(){
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
