#include "aco.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>

namespace optimsolution {

void ACO::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double ACO::valueAtLevel(int j, int l) const {
    // linearly spaced grid on [lb, ub]
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double lb = L[j], ub = U[j];
    if (!std::isfinite(lb)) lb = -1.0;
    if (!std::isfinite(ub)) ub =  1.0;
    if (levels_ <= 1) return 0.5*(lb+ub);
    double t = (double)l / (double)(levels_-1);
    return lb + t*(ub-lb);
}

int ACO::sampleLevel(int j, double gbest_j){
    // heuristic eta_jl = 1 / (1 + |v - gbest_j|)
    std::vector<double> w(levels_, 0.0);
    double sumw = 0.0;
    for (int l=0; l<levels_; ++l){
        double v   = valueAtLevel(j,l);
        double eta = 1.0 / (1.0 + std::fabs(v - gbest_j));
        double ww  = std::pow(std::max(tau_[j][l], 0.0), alpha_) * std::pow(eta, beta_);
        w[l] = ww;
        sumw += ww;
    }
    if (sumw <= 0.0){
        // fallback: uniform
        std::uniform_int_distribution<int> Ui(0, levels_-1);
        return Ui(rng_);
    }
    std::uniform_real_distribution<double> U01(0.0, sumw);
    double r = U01(rng_);
    double acc = 0.0;
    for (int l=0; l<levels_; ++l){
        acc += w[l];
        if (r <= acc) return l;
    }
    return levels_-1;
}

void ACO::evaporate(){
    for (auto& row : tau_){
        for (double& t : row){
            t = (1.0 - rho_) * t;
            if (t < tau_min_) t = tau_min_;
        }
    }
}

void ACO::deposit(const std::vector<int>& order){
    // order: indices of ants sorted by ascending f (best first)
    int k = std::min(deposit_top_, (int)order.size());
    const double eps = 1e-32;

    for (int r=0; r<k; ++r){
        int idx = order[r];
        const auto& x = X_[idx];
        double f = FX_[idx];
        // amount = Q / (f - fmin + eps)  (shift to keep positive; fmin is best over these k)
        double fmin = FX_[order[0]];
        double amount = Q_ / ( (f - fmin) + eps );

        // deposit on chosen levels for this solution
        int D = (int)x.size();
        for (int j=0; j<D; ++j){
            // find nearest level to x[j]
            int best_l = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int l=0; l<levels_; ++l){
                double v = valueAtLevel(j,l);
                double d = std::fabs(v - x[j]);
                if (d < best_d){ best_d = d; best_l = l; }
            }
            tau_[j][best_l] += amount;
            if (tau_[j][best_l] > tau_max_) tau_[j][best_l] = tau_max_;
        }
    }
}

void ACO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    // init pheromones
    tau_.assign(D, std::vector<double>(std::max(1,levels_), tau0_));

    // do an initial sampling (pop_ ants) to seed best
    X_.assign(pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());

    // initial best via uniform initializer (better seeding)
    {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto X0 = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 8));
        best_f_ = std::numeric_limits<double>::infinity();
        best_x_.assign(D, 0.0);
        for (auto& x : X0){
            ensureBounds(x);
            double f = eval(x);
            if (f < best_f_) { best_f_ = f; best_x_ = x; }
            if (prob_->calls() >= max_evals_) return;
        }
    }

    // one constructive pass to fill X_, FX_ (respecting max_evals_)
    for (int i=0; i<pop_; ++i){
        std::vector<double> x(D,0.0);
        for (int j=0; j<D; ++j){
            int l = sampleLevel(j, best_x_[j]);
            x[j] = valueAtLevel(j,l);
        }
        ensureBounds(x);
        double f = eval(x);
        X_[i] = std::move(x);
        FX_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) return;
    }
    printBest();
    updateStop(FX_);
}

void ACO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0,1.0);

    // construct new ants
    std::vector<std::vector<double>> Xnew(pop_, std::vector<double>(D, 0.0));
    std::vector<double> Fnew(pop_, std::numeric_limits<double>::infinity());

    for (int i=0; i<pop_; ++i){
        // build solution with roulette sampling per dimension
        for (int j=0; j<D; ++j){
            int l = sampleLevel(j, best_x_[j]); // heuristic uses current gbest
            Xnew[i][j] = valueAtLevel(j,l);
        }
        ensureBounds(Xnew[i]);
        double f = eval(Xnew[i]);

        // optional in-run local
        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, Xnew[i]);
            if (fl < f){ Xnew[i] = std::move(xl); f = fl; }
        }

        Fnew[i] = f;
        if (prob_->calls() >= max_evals_) break;
    }

    // select survivors: elitist replacement per slot (keep improving ant vs current)
    for (int i=0; i<pop_; ++i){
        if (Fnew[i] < FX_[i]) {
            X_[i]  = std::move(Xnew[i]);
            FX_[i] = Fnew[i];
        } else if (std::isfinite(Fnew[i]) && !std::isfinite(FX_[i])) {
            X_[i]  = std::move(Xnew[i]);
            FX_[i] = Fnew[i];
        }
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
    }

    // pheromone update
    evaporate();
    // order indices by ascending f
    std::vector<int> order(pop_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });
    deposit(order);

    printBest();
    updateStop(FX_);
}

void ACO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Replace worst with best (consistency with DE/GA)
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
