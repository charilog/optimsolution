#include "sa.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>

namespace optimsolution {

void SA::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// FIX: not const, so RNG distributions can use non-const rng_
void SA::propose(int i, std::vector<double>& cand) {
    // Gaussian proposal around X_[i], per dimension
    const int D = (int)cand.size();
    std::normal_distribution<double> N01(0.0, 1.0);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int j=0; j<D; ++j){
        double span = (std::isfinite(L[j]) && std::isfinite(U[j])) ? (U[j]-L[j]) : 1.0;
        double sigma = (step_sigma_ > 0.0) ? step_sigma_ : (step_frac_ * span);
        cand[j] = X_[i][j] + sigma * N01(rng_);
    }
}

void SA::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    // Initial sampling via Initializer
    Initializer initSampler;
    initSampler.configure(initopt_);

    X_  = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 1));
    FX_.assign((size_t)pop_, std::numeric_limits<double>::infinity());
    T_.assign((size_t)pop_, std::max(T0_, Tmin_));

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    // Evaluate initial agents
    for (int i=0; i<pop_; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) return;
    }

    printBest();
    updateStop(FX_);
}

void SA::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // For each agent, perform "moves_" Metropolis steps
    for (int i=0; i<pop_; ++i){
        double Ti = T_[i];

        for (int m=0; m<moves_; ++m){
            std::vector<double> cand = X_[i];
            propose(i, cand);
            ensureBounds(cand);

            double fnew = eval(cand);

            // Metropolis criterion
            double dE = fnew - FX_[i];
            bool accept = false;
            if (dE <= 0.0) {
                accept = true;
            } else {
                double Tuse = std::max(Ti, 1e-300);
                double p = std::exp(-dE / Tuse);
                if (U01(rng_) < p) accept = true;
            }

            if (accept) {
                X_[i] = std::move(cand);
                FX_[i] = fnew;

                // Optional in-run local immediately after acceptance
                if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
                    auto [xl, fl] = localSearch(local_method_, X_[i]);
                    if (fl < FX_[i]) { X_[i] = std::move(xl); FX_[i] = fl; }
                }

                if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
            }

            if (prob_->calls() >= max_evals_) break;
        }

        // Geometric cooling
        Ti *= alpha_;
        if (Ti < Tmin_) Ti = Tmin_;
        T_[i] = Ti;

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void SA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    printBest();
}

} // namespace optimsolution
