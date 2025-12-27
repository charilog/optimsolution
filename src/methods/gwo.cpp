#include "gwo.h"
#include "init.h"
#include <algorithm>
#include <limits>
#include <random>
#include <cmath>

namespace optimsolution {

void GWO::ensureBounds(Vec& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double GWO::progress01() const {
    if (max_evals_ <= 0) return 0.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
    return p;
}

void GWO::elitismInject(){
    if (X_.empty() || FX_.empty()) return;
    // Finds the worst individual.
    size_t worst = 0; double fw = FX_[0];
    for (size_t i=1; i<FX_.size(); ++i){
        if (FX_[i] > fw){ fw = FX_[i]; worst = i; }
    }
    X_[worst]  = best_x_;
    FX_[worst] = best_f_;
}

void GWO::computeLeaders(int& alpha, int& beta, int& delta) const {
    // Selects the three best individuals (smallest FX).
    alpha = beta = delta = -1;
    double fa = std::numeric_limits<double>::infinity();
    double fb = std::numeric_limits<double>::infinity();
    double fd = std::numeric_limits<double>::infinity();

    for (int i=0; i<pop_; ++i){
        const double f = FX_[i];
        if (f < fa){
            // shift
            delta = beta; fd = fb;
            beta  = alpha; fb = fa;
            alpha = i;     fa = f;
        } else if (f < fb){
            delta = beta; fd = fb;
            beta  = i;    fb = f;
        } else if (f < fd){
            delta = i;    fd = f;
        }
    }

    // Safeguard for small population sizes.
    if (alpha < 0) alpha = 0;
    if (beta  < 0) beta  = std::min(1, pop_-1);
    if (delta < 0) delta = std::min(2, pop_-1);
}

void GWO::init(){
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

    // Applies elitism so that BSS 'sees' the best.
    elitismInject();

    updateStop(FX_);
    printBest();
}

void GWO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();

    // Computes alpha, beta, delta leaders.
    int aIdx, bIdx, dIdx;
    computeLeaders(aIdx, bIdx, dIdx);

    const Vec Xa = X_[aIdx];
    const Vec Xb = X_[bIdx];
    const Vec Xd = X_[dIdx];

    // Progressive parameter a.
    double prog = progress01();
    double a = a_start_ + (a_end_ - a_start_) * prog;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double>       N01(0.0, 1.0);

    std::vector<Vec> Xnew(pop_, Vec(D, 0.0));
    std::vector<double> Fnew(pop_, std::numeric_limits<double>::infinity());

    for (int i=0; i<pop_; ++i){
        if (prob_->calls() >= max_evals_) break;

        Vec x(D, 0.0);

        for (int j=0; j<D; ++j){
            // Triple encircling.
            double r1 = U01(rng_), r2 = U01(rng_);
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;

            r1 = U01(rng_); r2 = U01(rng_);
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;

            r1 = U01(rng_); r2 = U01(rng_);
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;

            double D_alpha = std::fabs(C1 * Xa[j] - X_[i][j]);
            double D_beta  = std::fabs(C2 * Xb[j] - X_[i][j]);
            double D_delta = std::fabs(C3 * Xd[j] - X_[i][j]);

            double X1 = Xa[j] - A1 * D_alpha;
            double X2 = Xb[j] - A2 * D_beta;
            double X3 = Xd[j] - A3 * D_delta;

            double val = (X1 + X2 + X3) / 3.0;

            if (jitter_sigma_ > 0.0) {
                val += jitter_sigma_ * N01(rng_);
            }

            x[j] = val;
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
    }

    // Generation replacement.
    for (int i=0; i<pop_; ++i){
        if (std::isfinite(Fnew[i])){
            X_[i]  = std::move(Xnew[i]);
            FX_[i] = Fnew[i];
        }
    }

    // Elitism at each iteration.
    elitismInject();

    updateStop(FX_);
    printBest();
}

void GWO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Preserves the best in the population.
    elitismInject();

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
