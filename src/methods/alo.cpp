#include "alo.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void ALO::ensureBounds(Vec& v){
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

ALO::Vec ALO::randomWalkNormalized(const Vec& c, const Vec& d){
    const int D = (int)c.size();
    const int T = std::max(5, rw_len_);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Cumulative random walk positions (T steps)
    Vec pos(D, 0.0);
    Vec mn(D, 0.0), mx(D, 0.0);

    // initialize mn/mx with first point (0)
    for (int j=0; j<D; ++j){ mn[j]=0.0; mx[j]=0.0; }

    // perform walk
    for (int t=0; t<T; ++t){
        for (int j=0; j<D; ++j){
            double step = (U01(rng_) < 0.5 ? -1.0 : 1.0);
            pos[j] += step;
            if (pos[j] < mn[j]) mn[j] = pos[j];
            if (pos[j] > mx[j]) mx[j] = pos[j];
        }
    }

    // normalize the LAST position to [c,d]
    Vec out(D, 0.0);
    for (int j=0; j<D; ++j){
        double lo = c[j], hi = d[j];
        if (lo > hi) std::swap(lo, hi);
        const double denom = (mx[j] - mn[j]);
        double u = 0.5;
        if (denom > 1e-12) {
            u = (pos[j] - mn[j]) / denom;
            if (u < 0.0) u = 0.0;
            if (u > 1.0) u = 1.0;
        }
        out[j] = lo + u * (hi - lo);
    }
    return out;
}

void ALO::init(){
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

    if (debug_alo_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[alo] cfg -> N=%d (population() now=%d, override=%d), rw_len=%d, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, rw_len_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void ALO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    // iteration-based progress (robust across different populations)
    const int T_est = std::max(1, (int)(max_evals_ / std::max(1, N)));
    const double ratio = (T_est <= 1 ? 1.0 : std::min(1.0, (double)iter_ / (double)(T_est - 1)));

    // ALO shrinking factor (common piecewise strategy)
    double I = 1.0;
    if (ratio > 0.95)      I = 1e6;
    else if (ratio > 0.90) I = 1e5;
    else if (ratio > 0.75) I = 1e4;
    else if (ratio > 0.50) I = 1e3;
    else if (ratio > 0.10) I = 1e2;
    else                   I = 1.0;

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    // Sort antlions by fitness (ascending)
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    // Elite (best antlion)
    Vec elite = X_[order[0]];
    double elite_f = FX_[order[0]];
    if (elite_f < best_f_) { best_f_ = elite_f; best_x_ = elite; }
    else { elite = best_x_; } // keep globally best

    // Roulette weights: 1/(rank+1) assigned to original indices
    std::vector<double> w(N, 0.0);
    for (int k=0; k<N; ++k){
        w[order[k]] = 1.0 / (double)(k + 1);
    }
    std::discrete_distribution<int> pickAntlion(w.begin(), w.end());
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Generate ants
    std::vector<Vec> ants;
    std::vector<double> antsF;
    ants.reserve(N);
    antsF.reserve(N);

    for (int i=0; i<N; ++i){
        int aidx = pickAntlion(rng_);
        const Vec& al = X_[aidx];

        Vec c1(D), d1(D), c2(D), d2(D);
        for (int j=0; j<D; ++j){
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            const double range = (hi - lo);
            const double rad = (I <= 1.0 ? range : range / I);

            c1[j] = al[j]    - rad;
            d1[j] = al[j]    + rad;
            c2[j] = elite[j] - rad;
            d2[j] = elite[j] + rad;

            // clamp
            if (c1[j] < lo) c1[j] = lo;
            if (d1[j] > hi) d1[j] = hi;
            if (c2[j] < lo) c2[j] = lo;
            if (d2[j] > hi) d2[j] = hi;
        }

        Vec rw1 = randomWalkNormalized(c1, d1);
        Vec rw2 = randomWalkNormalized(c2, d2);

        Vec x(D, 0.0);
        for (int j=0; j<D; ++j) x[j] = 0.5 * (rw1[j] + rw2[j]);
        ensureBounds(x);

        double fx = eval(x);

        // Optional in-run local
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, x);
                if (std::isfinite(floc) && floc < fx){
                    x  = std::move(xloc);
                    fx = floc;
                }
            }
        }

        ants.push_back(std::move(x));
        antsF.push_back(fx);

        if (fx < best_f_) { best_f_ = fx; best_x_ = ants.back(); }

        if (prob_->calls() >= max_evals_) break;
    }

    // Combine antlions + ants and select best N
    std::vector<Vec> combX;
    std::vector<double> combF;
    combX.reserve((size_t)N + ants.size());
    combF.reserve((size_t)N + ants.size());

    for (int i=0; i<N; ++i){ combX.push_back(X_[i]); combF.push_back(FX_[i]); }
    for (size_t k=0; k<ants.size(); ++k){ combX.push_back(std::move(ants[k])); combF.push_back(antsF[k]); }

    std::vector<int> cord((int)combF.size());
    std::iota(cord.begin(), cord.end(), 0);
    std::sort(cord.begin(), cord.end(), [&](int a, int b){ return combF[a] < combF[b]; });

    const int Nkeep = std::min(N, (int)cord.size());
    std::vector<Vec> newX;
    std::vector<double> newF;
    newX.reserve(Nkeep);
    newF.reserve(Nkeep);

    for (int k=0; k<Nkeep; ++k){
        int idx = cord[k];
        newX.push_back(std::move(combX[idx]));
        newF.push_back(combF[idx]);
    }

    X_.swap(newX);
    FX_.swap(newF);

    // elitism: write best into worst position
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

void ALO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // write best to worst position
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
