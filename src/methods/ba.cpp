#include "ba.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void BA::ensureBounds(Vec& v){
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

void BA::clampVelocity(Vec& v){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double span = (std::isfinite(lo) && std::isfinite(hi)) ? (hi - lo) : 1.0;
        if (!std::isfinite(span) || span <= 0.0) span = 1.0;
        const double vmax = vmax_scale_ * span;
        if (!std::isfinite(v[j])) v[j] = 0.0;
        if (v[j] >  vmax) v[j] =  vmax;
        if (v[j] < -vmax) v[j] = -vmax;
    }
}

void BA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); V_.clear(); FX_.clear(); A_.clear(); R_.clear();
    it_ = 0;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    V_.assign(N, Vec(D, 0.0));
    FX_.assign(N, std::numeric_limits<double>::infinity());
    A_.assign(N, A0_);
    // In the canonical BA formulation, pulse rate increases from ~0 toward r0.
    // Starting low encourages more local random walks early.
    R_.assign(N, 0.0);

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

    if (debug_ba_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[ba] cfg -> N=%d (population() now=%d, override=%d), f=[%.6f,%.6f], A0=%.6f, r0=%.6f, alpha=%.6f, gamma=%.6f, vmax_scale=%.6f, walk_scale=%.6f, accept_impr_always=%s, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, fmin_, fmax_, A0_, r0_, alpha_, gamma_, vmax_scale_, walk_scale_,
            accept_improvement_always_ ? "on" : "off",
            lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void BA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Synchronous update to avoid order bias.
    const std::vector<Vec>    Xold = X_;
    const std::vector<Vec>    Vold = V_;
    const std::vector<double> Fold = FX_;
    const std::vector<double> Aold = A_;
    // R_ is updated deterministically (per-iteration) below.

    // Pulse rate increases deterministically with iteration count in the canonical BA formulation.
    // It must not depend on acceptance; otherwise some bats may keep an almost-zero pulse rate and
    // perform excessive random walks indefinitely.
    const double t_iter = (double)(it_ + 1);
    auto pulseAt = [&](double t)->double{
        double ri = r0_ * (1.0 - std::exp(-gamma_ * t));
        if (!std::isfinite(ri)) ri = r0_;
        if (ri < 0.0) ri = 0.0;
        if (ri > 1.0) ri = 1.0;
        return ri;
    };

    for (int i=0; i<N; ++i){
        // Frequency
        const double freq = fmin_ + (fmax_ - fmin_) * U01(rng_);

        // Velocity update
        Vec v = Vold[i];
        if ((int)v.size() != D) v.assign(D, 0.0);
        for (int k=0; k<D; ++k){
            // Move toward the current global best.
            v[k] = v[k] + (best_x_[k] - Xold[i][k]) * freq;
        }
        clampVelocity(v);

        // Position update
        Vec x = Xold[i];
        for (int k=0; k<D; ++k){
            x[k] = x[k] + v[k];
        }
        ensureBounds(x);

        // Update pulse rate (always) and perform local random walk around the current best.
        // In the original BA, the local walk is triggered when rand > r_i.
        const double rcur = pulseAt(t_iter);
        R_[i] = rcur;

        if (U01(rng_) > rcur) {
            const Vec& L = prob_->lb();
            const Vec& U = prob_->ub();
            for (int k=0; k<D; ++k){
                double lo = (k < (int)L.size() ? L[k] : -1.0);
                double hi = (k < (int)U.size() ? U[k] :  1.0);
                if (lo > hi) std::swap(lo, hi);
                double span = (std::isfinite(lo) && std::isfinite(hi)) ? (hi - lo) : 1.0;
                if (!std::isfinite(span) || span <= 0.0) span = 1.0;
                // Canonical BA: x = best + epsilon * A, with epsilon in [-1,1].
                // Here epsilon is scaled by the variable range so that it remains meaningful across problems.
                const double eps  = (U01(rng_) * 2.0 - 1.0);
                const double step = eps * walk_scale_ * Aold[i] * span;
                x[k] = best_x_[k] + step;
            }
            ensureBounds(x);
        }

        double fx = eval(x);

        // Optional in-run local refinement (only when the candidate improves the current individual)
        if (fx < Fold[i]) {
            if (local_rate_ > 0.0 && !local_method_.empty()) {
                if (U01(rng_) < local_rate_) {
                    auto [xloc, floc] = localSearch(local_method_, x);
                    if (std::isfinite(floc) && floc < fx) {
                        x  = std::move(xloc);
                        fx = floc;
                    }
                }
            }
        }

        // Always track the best solution encountered (independent of acceptance).
        if (fx < best_f_) {
            best_f_ = fx;
            best_x_ = x;
        }

        const bool improved = (fx < Fold[i]);
        bool accept = false;

        if (improved) {
            if (accept_improvement_always_) {
                accept = true;
            } else {
                accept = (U01(rng_) < Aold[i]);
            }
        }

        if (accept) {
            X_[i]  = std::move(x);
            V_[i]  = std::move(v);
            FX_[i] = fx;

            // Loudness decreases, pulse rate increases
            A_[i] = Aold[i] * alpha_;
            if (!std::isfinite(A_[i]) || A_[i] < 0.0) A_[i] = 0.0;

            // Pulse rate is handled above (deterministic per-iteration).
        }

        if (prob_->calls() >= max_evals_) break;
    }

    ++it_;

    // Elitism: ensure the best solution is present in the population.
    // This prevents inconsistencies when the framework derives the reported minimum from FX_.
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (best_f_ < worst_val && worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    printBest();
    updateStop(FX_);
}

void BA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best to worst position.
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
