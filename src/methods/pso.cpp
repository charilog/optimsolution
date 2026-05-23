#include "pso.h"
#include "init.h"
#include <cmath>
#include <limits>

namespace optimsolution {

void PSO::init(){
    if (!prob_) return;
    const int D = prob_->dimension();

    X_.clear();
    V_.clear();
    FX_.clear();
    Pbest_.clear();
    Fpbest_.clear();
    Vmax_.clear();
    Gbest_.assign(D, 0.0);
    Fgbest_ = std::numeric_limits<double>::infinity();

    // Initialize population based on init options from cfg
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);

    V_.assign(pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    Pbest_.assign(pop_, std::vector<double>(D, 0.0));
    Fpbest_.assign(pop_, std::numeric_limits<double>::infinity());
    Vmax_.assign(D, 0.0);

    // Compute Vmax per dimension
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int j=0; j<D; ++j){
        if (std::isfinite(L[j]) && std::isfinite(U[j])){
            Vmax_[j] = vmax_frac_ * (U[j] - L[j]);
        } else {
            Vmax_[j] = std::numeric_limits<double>::infinity();
        }
    }

    // Initial velocities and evaluations
    for (int i=0; i<pop_; ++i){
        for (int j=0; j<D; ++j){
            if (std::isfinite(Vmax_[j])){
                std::uniform_real_distribution<double> Uv(-Vmax_[j], Vmax_[j]);
                V_[i][j] = Uv(rng_);
            } else {
                V_[i][j] = 0.0;
            }
        }
        FX_[i] = eval(X_[i]);
        Pbest_[i]  = X_[i];
        Fpbest_[i] = FX_[i];
        if (FX_[i] < Fgbest_) { Fgbest_ = FX_[i]; Gbest_ = X_[i]; }
        if (prob_->calls() >= max_evals_) return;
    }

    best_x_ = Gbest_;
    best_f_ = Fgbest_;
}

int PSO::neighborhoodBestIndex(int i) const {
    if (topology_ != "ring" || pop_ <= 1) return i;
    int best = i;
    double fb = Fpbest_[i];
    const int n = pop_;
    const int k = std::max(1, ring_k_);
    for (int d = -k; d <= k; ++d){
        int idx = (i + d) % n; if (idx < 0) idx += n;
        if (Fpbest_[idx] < fb){ fb = Fpbest_[idx]; best = idx; }
    }
    return best;
}

void PSO::clampVelocity(std::vector<double>& v){
    const int D = (int)v.size();
    for (int j=0; j<D; ++j){
        if (std::isfinite(Vmax_[j])){
            const double vmax = Vmax_[j];
            if (v[j] >  vmax) v[j] =  vmax;
            if (v[j] < -vmax) v[j] = -vmax;
        }
    }
}

void PSO::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j=0; j<D; ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void PSO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // --- DE TEMPLATE: "trial" state and GREEDY SELECTION ---
    for (int i=0; i<pop_; ++i){
        const std::vector<double>& gb =
            (topology_ == "ring") ? Pbest_[neighborhoodBestIndex(i)] : Gbest_;

        // Compute trial velocity/position
        std::vector<double> vnew = V_[i];
        std::vector<double> xnew = X_[i];

        for (int j=0; j<D; ++j){
            double r1 = U01(rng_), r2 = U01(rng_);
            vnew[j] = w_ * V_[i][j]
                    + c1_ * r1 * (Pbest_[i][j] - X_[i][j])
                    + c2_ * r2 * (gb[j]        - X_[i][j]);
        }
        clampVelocity(vnew);
        for (int j=0; j<D; ++j) xnew[j] = X_[i][j] + vnew[j];
        ensureBounds(xnew);

        // Trial evaluation
        double fnew = eval(xnew);

        // --- GREEDY SELECTION (as in DE): Accept ONLY if it improves ---
        bool accepted = false;
        if (fnew < FX_[i]) {
            X_[i]  = std::move(xnew);
            V_[i]  = std::move(vnew);
            FX_[i] = fnew;
            accepted = true;

            // In-run local: ONLY after improvement (as in DE)
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, X_[i]);
                    if (floc < FX_[i]){
                        X_[i]  = std::move(xloc);
                        FX_[i] = floc;
                    }
                }
            }

            // Update personal/global best
            if (FX_[i] < Fpbest_[i]) {
                Fpbest_[i] = FX_[i];
                Pbest_[i]  = X_[i];
            }
            if (FX_[i] < Fgbest_) {
                Fgbest_ = FX_[i];
                Gbest_  = X_[i];
                best_f_ = Fgbest_;
                best_x_ = Gbest_;
            }
        } else {
            // Reject: keep X/FX as-is and "zero" the velocity (stability)
            for (int j=0;j<D;++j) V_[i][j] *= 0.5;
        }

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_); // Same as DE — the termination rule now works correctly
}

void PSO::end(){
    // Executed at the end. Controlled ONLY by [global]. (same pattern as DE)
    if (!end_local_refine_)     return;
    if (!prob_)                 return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Optional: write the refined best into the worst position (as in DE)
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
