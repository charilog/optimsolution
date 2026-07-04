#include "woa.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>
#include <cmath> // exp, cos, fabs

namespace optimsolution {

static constexpr double kPi = 3.1415926535897932384626433832795;

void WOA::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double WOA::progress01() const {
    if (max_evals_ <= 0) return 0.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
    return p;
}

void WOA::init() {
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

    // FIX (logic): updateStop() must receive the fitness values of the WHOLE
    // population, exactly as in every other method (DE/PSO/GWO/MEWOA...).
    // Previously only a sorted "elite" subset (top 20%, min 12 values) was
    // passed, which biased the population statistics used by the stopping
    // rules (e.g. variance/BSS based) and caused premature termination.
    updateStop(FX_);
    printBest();
}


void WOA::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Unegpos(-1.0, 1.0);

    double prog = progress01();
    double a = a_start_ + (a_end_ - a_start_) * prog;

    auto pick_rand_index = [&](int exclude)->int{
        if (pop_ <= 1) return 0;
        std::uniform_int_distribution<int> Ui(0, pop_-1);
        int r = Ui(rng_);
        if (r == exclude) r = (r+1) % pop_;
        return r;
    };

    const double b = b_spiral_;

    std::vector<std::vector<double>> Xnew(pop_, std::vector<double>(D, 0.0));
    std::vector<double> Fnew(pop_, std::numeric_limits<double>::infinity());

    for (int i=0; i<pop_; ++i){
        double p = U01(rng_);
        // FIX (logic): the original WOA draws A and C from INDEPENDENT random
        // numbers (A = 2*a*r1 - a, C = 2*r2). Using the same r for both fully
        // correlates the shrinking coefficient with the encircling coefficient
        // (e.g. |A| small always implies C small), which distorts the
        // exploration/exploitation balance of the algorithm.
        double r1 = U01(rng_);
        double r2 = U01(rng_);
        double A = 2.0 * a * r1 - a;
        double C = 2.0 * r2;

        std::vector<double> x(D, 0.0);

        if (p < 0.5) {
            if (std::fabs(A) < 1.0) {
                for (int j=0; j<D; ++j){
                    double Dj = std::fabs(C * best_x_[j] - X_[i][j]);
                    x[j] = best_x_[j] - A * Dj;
                }
            } else {
                int r_idx = pick_rand_index(i);
                for (int j=0; j<D; ++j){
                    double Dj = std::fabs(C * X_[r_idx][j] - X_[i][j]);
                    x[j] = X_[r_idx][j] - A * Dj;
                }
            }
        } else {
            double l = Unegpos(rng_);
            for (int j=0; j<D; ++j){
                double Dj = std::fabs(best_x_[j] - X_[i][j]);
                x[j] = Dj * std::exp(b * l) * std::cos(2.0 * kPi * l) + best_x_[j];
            }
        }

        ensureBounds(x);
        double f = eval(x);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, x);
            if (fl < f){ x = std::move(xl); f = fl; }
        }

        Xnew[i] = std::move(x);
        Fnew[i] = f;

        if (f < best_f_) { best_f_ = f; best_x_ = Xnew[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    for (int i=0; i<pop_; ++i){
        if (std::isfinite(Fnew[i])) {
            X_[i]  = std::move(Xnew[i]);
            FX_[i] = Fnew[i];
        }
    }

    // FIX (logic): passing a single value {best_f_} to updateStop() makes any
    // population-statistics-based stopping rule (variance, doublebox, BSS...)
    // see zero spread and fire immediately, so WOA stopped after a handful of
    // iterations and appeared to "not converge". The full FX_ must be passed,
    // as in all other methods.
    updateStop(FX_);
    printBest();
}


void WOA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // for consistency with other methods
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
