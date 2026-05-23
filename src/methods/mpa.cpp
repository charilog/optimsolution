#include "mpa.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void MPA::init(){
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

    if (debug_mpa_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[mpa] cfg -> N=%d (population() now=%d, override=%d), FADs=%.3f, P=%.3f, levy_beta=%.3f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, FADs_, P_, levy_beta_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

int MPA::pickDistinct(int n, int a){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a);
    return r;
}

void MPA::ensureBounds(Vec& v){
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

// Mantegna's algorithm (1D) for Levy flights
double MPA::levyStep(){
    const double beta = levy_beta_;
    const double pi = 3.14159265358979323846;
    const double sigma_u = std::pow(
        (std::tgamma(1.0 + beta) * std::sin(pi * beta / 2.0)) /
        (std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0)),
        1.0 / beta);

    static thread_local std::normal_distribution<double> N01(0.0, 1.0);
    const double u = N01(rng_) * sigma_u;
    const double v = N01(rng_);
    const double denom = std::pow(std::fabs(v) + 1e-12, 1.0 / beta);
    return u / denom;
}

void MPA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Iteration-based progress estimate
    const double T_est = std::max(1.0, std::floor((double)max_evals_ / (double)std::max(1, N)));
    const double progress = std::min(1.0, std::max(0.0, (double)iter_ / T_est));

    // Elite (best) index in current population
    int elite_i = 0;
    double elite_f = FX_[0];
    for (int i=1; i<N; ++i){
        if (FX_[i] < elite_f) { elite_f = FX_[i]; elite_i = i; }
    }
    const Vec elite = X_[elite_i];
    if (elite_f < best_f_) {
        best_f_ = elite_f;
        best_x_ = elite;
    }

    // Control factor from the original MPA formulation
    const double CF = std::pow(1.0 - progress, 2.0 * progress);

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int i=0; i<N; ++i){
        Vec cand = X_[i];

        // 3 phases
        if (progress < (1.0/3.0)) {
            // High velocity ratio: Brownian motion dominated
            for (int j=0; j<D; ++j){
                const double rb = N01(rng_);
                const double R  = U01(rng_);
                const double step = rb * (elite[j] - rb * X_[i][j]);
                cand[j] = X_[i][j] + P_ * R * step;
            }
        } else if (progress < (2.0/3.0)) {
            // Transition
            if (i < N/2) {
                for (int j=0; j<D; ++j){
                    const double rl = levyStep();
                    const double R  = U01(rng_);
                    const double step = rl * (elite[j] - rl * X_[i][j]);
                    cand[j] = X_[i][j] + P_ * R * step;
                }
            } else {
                for (int j=0; j<D; ++j){
                    const double rb = N01(rng_);
                    const double R  = U01(rng_);
                    const double step = rb * (rb * elite[j] - X_[i][j]);
                    cand[j] = elite[j] + P_ * CF * step * R;
                }
            }
        } else {
            // Low velocity ratio: Levy motion dominated around elite
            for (int j=0; j<D; ++j){
                const double rl = levyStep();
                const double R  = U01(rng_);
                const double step = rl * (rl * elite[j] - X_[i][j]);
                cand[j] = elite[j] + P_ * CF * step * R;
            }
        }

        // FADs effect (diversification)
        if (U01(rng_) < FADs_) {
            for (int j=0; j<D; ++j){
                if (U01(rng_) < FADs_) {
                    const double lo = (j < (int)L.size() ? L[j] : -1.0);
                    const double hi = (j < (int)U.size() ? U[j] :  1.0);
                    cand[j] = cand[j] + CF * (lo + U01(rng_) * (hi - lo));
                }
            }
        } else {
            const int r1 = pickDistinct(N, i);
            const int r2 = pickDistinct(N, i);
            const double r = U01(rng_);
            for (int j=0; j<D; ++j){
                cand[j] = cand[j] + r * (X_[r1][j] - X_[r2][j]);
            }
        }

        ensureBounds(cand);
        double fc = eval(cand);

        // Memory saving (greedy replacement)
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

void MPA::end(){
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
