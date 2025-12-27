#include "gao.h"
#include "init.h"
#include <algorithm>
#include <limits>
#include <random>
#include <cmath>
#include <numeric>   // std::iota
#include <cfloat>

namespace optimsolution {

void GAO::ensureBounds(Vec& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double GAO::progress01() const {
    if (max_evals_ <= 0) return 0.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
    return p;
}

void GAO::elitismInject(){
    if (X_.empty() || FX_.empty()) return;
    size_t worst = 0; double fw = FX_[0];
    for (size_t i=1; i<FX_.size(); ++i){
        if (FX_[i] > fw){ fw = FX_[i]; worst = i; }
    }
    X_[worst]  = best_x_;
    FX_[worst] = best_f_;
}

void GAO::leadersABC(int& a, int& b, int& c) const {
    a = b = c = -1;
    double fa = DBL_MAX, fb = DBL_MAX, fc = DBL_MAX;
    for (int i=0; i<pop_; ++i){
        const double f = FX_[i];
        if (f < fa){ c=b; fc=fb; b=a; fb=fa; a=i; fa=f; }
        else if (f < fb){ c=b; fc=fb; b=i; fb=f; }
        else if (f < fc){ c=i; fc=f; }
    }
    if (a < 0) a = 0;
    if (b < 0) b = std::min(1, pop_-1);
    if (c < 0) c = std::min(2, pop_-1);
}

GAO::Vec GAO::topCentroid(int k) const {
    const int D = prob_->dimension();
    k = std::max(1, std::min(k, pop_));
    std::vector<int> idx(pop_);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
                      [&](int i, int j){ return FX_[i] < FX_[j]; });
    Vec c(D, 0.0);
    for (int r=0; r<k; ++r){
        const auto& x = X_[idx[r]];
        for (int j=0; j<D; ++j) c[j] += x[j];
    }
    for (int j=0; j<D; ++j) c[j] /= (double)k;
    return c;
}

void GAO::init(){
    if (!prob_) return;
    const int D = prob_->dimension();

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_  = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 2));
    FX_.assign((size_t)pop_, std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<pop_; ++i){
        ensureBounds(X_[i]);
        double f = eval(X_[i]);
        FX_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    elitismInject();
    updateStop(FX_);
    printBest();
}

void GAO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();

    const double prog = progress01();

    int ia, ib, ic;
    leadersABC(ia, ib, ic);
    const Vec Xa = X_[ia], Xb = X_[ib], Xc = X_[ic];
    const int kcent = std::max(3, pop_/5);
    const Vec Ck = topCentroid(kcent);

    int n_burrow = (int)std::round(burrow_frac_ * pop_);
    int n_roll   = (int)std::round(roll_frac_   * pop_);
    int n_forage = std::max(0, pop_ - n_burrow - n_roll);
    if (n_burrow + n_roll + n_forage < pop_) n_forage += pop_ - (n_burrow+n_roll+n_forage);

    std::vector<int> idx(pop_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j){ return FX_[i] < FX_[j]; });

    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    std::vector<Vec>    Xnew(pop_, Vec(D, 0.0));
    std::vector<double> Fnew(pop_, std::numeric_limits<double>::infinity());

    auto heavyTail = [&](double scale){
        // ~Cauchy-like: scale * N / |N|
        double n = N01(rng_);
        double d = std::fabs(N01(rng_));
        if (d < 1e-12) d = 1e-12;
        return scale * (n / d);
    };

    // 1) BURROW: exploitation near the best solution
    for (int r=0; r<n_burrow; ++r){
        int i = idx[r];
        Vec x = X_[i];
        for (int j=0; j<D; ++j){
            double s = burrow_step_ * (1.0 - 0.6*prog);
            x[j] += s * (best_x_[j] - x[j]) + 0.1 * s * N01(rng_);
            if (jitter_sigma_ > 0.0) x[j] += jitter_sigma_ * N01(rng_);
        }
        ensureBounds(x);
        double fx = eval(x);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, x);
            if (fl < fx){ x = std::move(xl); fx = fl; }
        }

        Xnew[i] = std::move(x);
        Fnew[i] = fx;
        if (fx < best_f_) { best_f_ = fx; best_x_ = Xnew[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // 2) ROLL: heavy-tailed exploration around the anchor (best/centroid)
    for (int r=n_burrow; r<n_burrow+n_roll; ++r){
        int i = idx[r];
        Vec x = X_[i];
        for (int j=0; j<D; ++j){
            double s = roll_step_ * (1.0 - 0.3*prog);
            double anchor = 0.6*best_x_[j] + 0.4*Ck[j];
            x[j] = anchor + heavyTail(s);
            if (jitter_sigma_ > 0.0) x[j] += jitter_sigma_ * N01(rng_);
        }
        ensureBounds(x);
        double fx = eval(x);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, x);
            if (fl < fx){ x = std::move(xl); fx = fl; }
        }

        Xnew[i] = std::move(x);
        Fnew[i] = fx;
        if (fx < best_f_) { best_f_ = fx; best_x_ = Xnew[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // 3) FORAGE: attraction toward centroid/top-k with contraction
    for (int r=n_burrow+n_roll; r<pop_; ++r){
        int i = idx[r];
        Vec x = X_[i];
        for (int j=0; j<D; ++j){
            double pull = forage_beta_ * (Ck[j] - x[j]);
            x[j] += pull;
            x[j] = best_x_[j] + forage_gamma_*(x[j] - best_x_[j])*(1.0 - 0.5*prog);
            if (jitter_sigma_ > 0.0) x[j] += jitter_sigma_ * N01(rng_);
        }
        ensureBounds(x);
        double fx = eval(x);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, x);
            if (fl < fx){ x = std::move(xl); fx = fl; }
        }

        Xnew[i] = std::move(x);
        Fnew[i] = fx;
        if (fx < best_f_) { best_f_ = fx; best_x_ = Xnew[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // replacement
    for (int i=0; i<pop_; ++i){
        if (std::isfinite(Fnew[i])){
            X_[i]  = std::move(Xnew[i]);
            FX_[i] = Fnew[i];
        }
    }

    // elitism + BSS update + best reporting
    elitismInject();
    updateStop(FX_);
    printBest();
}

void GAO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) { best_f_ = floc; best_x_ = xloc; }

    elitismInject();
    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
