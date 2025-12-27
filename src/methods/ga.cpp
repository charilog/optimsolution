#include "ga.h"
#include "init.h"
#include <random>
#include <limits>
#include <algorithm>

namespace optimsolution {

void GA::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);

    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i=0; i<pop_; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) return;
    }
}

void GA::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// Non-const to call Ui(rng_) with a non-const engine
int GA::tournamentSelect() {
    if (pop_ <= 1) return 0;
    std::uniform_int_distribution<int> Ui(0, pop_ - 1);
    int best = Ui(rng_);
    double fb = FX_[best];
    int k = std::max(2, tournament_k_);
    for (int t=1; t<k; ++t){
        int idx = Ui(rng_);
        if (FX_[idx] < fb) { fb = FX_[idx]; best = idx; }
    }
    return best;
}

void GA::crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                   std::vector<double>& child)
{
    const int D = (int)p1.size();
    child = p1; // default: without crossover

    std::uniform_real_distribution<double> U01(0.0,1.0);
    if (U01(rng_) > pc_) return;

    if (cx_type_ == "sbx") {
        child.resize(D);
        for (int j=0; j<D; ++j){
            double u = U01(rng_);
            double beta;
            if (u <= 0.5) beta = std::pow(2.0*u, 1.0/(eta_c_+1.0));
            else          beta = std::pow(2.0*(1.0-u), -1.0/(eta_c_+1.0));
            child[j] = 0.5*((1.0+beta)*p1[j] + (1.0-beta)*p2[j]);
        }
    } else if (cx_type_ == "blx") {
        child.resize(D);
        for (int j=0; j<D; ++j){
            double lo = std::min(p1[j], p2[j]);
            double hi = std::max(p1[j], p2[j]);
            double range = hi - lo;
            double L = lo - blx_a_*range;
            double U = hi + blx_a_*range;
            std::uniform_real_distribution<double> Useg(L, U);
            child[j] = Useg(rng_);
        }
    } else { // "uniform"
        child.resize(D);
        for (int j=0; j<D; ++j){
            child[j] = (U01(rng_) < uox_p_) ? p1[j] : p2[j];
        }
    }
}

void GA::mutate(std::vector<double>& child){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    for (size_t j=0; j<child.size(); ++j){
        if (U01(rng_) < pm_){
            double span = (std::isfinite(L[j]) && std::isfinite(U[j])) ? (U[j]-L[j]) : 1.0;
            double sigma = mut_sigma_ * span;
            child[j] += sigma * N01(rng_);
        }
    }
    ensureBounds(child);
}

void GA::one_iteration(){
    if (!prob_) return;
    std::uniform_real_distribution<double> U01(0.0,1.0);

    for (int i=0; i<pop_; ++i){
        int p1 = tournamentSelect();
        int p2 = tournamentSelect();
        if (p1 == p2) { p2 = (p2+1) % pop_; }

        std::vector<double> child;
        crossover(X_[p1], X_[p2], child);
        mutate(child);
        double fchild = eval(child);

        if (fchild < FX_[i]) {
            X_[i]  = std::move(child);
            FX_[i] = fchild;

            if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, X_[i]);
                if (floc < FX_[i]) {
                    X_[i]  = std::move(xloc);
                    FX_[i] = floc;
                    fchild = floc;
                }
            }

            if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void GA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    if (!X_.empty() && !FX_.empty()){
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > fw){ fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
