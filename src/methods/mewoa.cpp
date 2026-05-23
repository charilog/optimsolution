#include "mewoa.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>
#include <cmath>

namespace optimsolution {

static constexpr double kPi = 3.1415926535897932384626433832795;

void MEWOA::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double MEWOA::progress01() const {
    if (max_evals_ <= 0) return 0.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
    return p;
}

void MEWOA::buildElite(){
    elite_idx_.resize(pop_);
    std::iota(elite_idx_.begin(), elite_idx_.end(), 0);
    std::sort(elite_idx_.begin(), elite_idx_.end(), [&](int a, int b){
        return FX_[a] < FX_[b];
    });

    int k = (int)std::ceil(elite_frac_ * pop_);
    if (k < 1) k = 1;
    if (k > pop_) k = pop_;
    elite_idx_.resize(k);

    elite_w_.assign(k, 1.0 / (double)k);
    elite_cdf_.assign(k, 0.0);
    if (use_ranked_elite_) {
        double q = 0.5;
        double denom = 2.0 * std::pow(q * k, 2.0);
        if (denom <= 0.0) denom = 1e-9;
        double sumw = 0.0;
        for (int i=0; i<k; ++i){
            double a = (double)i;
            elite_w_[i] = std::exp(-(a*a)/denom);
            sumw += elite_w_[i];
        }
        if (sumw > 0.0) for (double& w: elite_w_) w /= sumw;
        else for (double& w: elite_w_) w = 1.0 / (double)k;
    }
    double acc=0.0;
    for (int i=0; i<k; ++i){
        acc += elite_w_[i];
        elite_cdf_[i] = (i==k-1) ? 1.0 : acc;
    }
}


int MEWOA::sampleEliteIndex() {
    int k = (int)elite_idx_.size();
    if (k <= 0) return 0;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    for (int i=0; i<k; ++i){
        if (r <= elite_cdf_[i]) return elite_idx_[i];
    }
    return elite_idx_.back();
}

void MEWOA::init() {
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
        if (prob_->calls() >= max_evals_) return;
    }

    
    {
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > fw){ fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    buildElite();

    
    updateStop(FX_);
    printBest();
}

void MEWOA::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Unegpos(-1.0, 1.0);
    std::normal_distribution<double>       N01(0.0, 1.0);

    double prog = progress01();
    double a = a_start_ + (a_end_ - a_start_) * prog;

    auto pick_rand_index = [&](int exclude)->int{
        if (pop_ <= 1) return 0;
        std::uniform_int_distribution<int> Ui(0, pop_-1);
        int r = Ui(rng_);
        if (r == exclude) r = (r+1) % pop_;
        return r;
    };

    std::vector<std::vector<double>> Xnew(pop_, std::vector<double>(D, 0.0));
    std::vector<double> Fnew(pop_, std::numeric_limits<double>::infinity());

    for (int i=0; i<pop_; ++i){
        double p = U01(rng_);
        double r = U01(rng_);
        double A = 2.0 * a * r - a;   // A in [-a, a]
        double C = 2.0 * r;           // C in [0,2]

        int eidx = sampleEliteIndex();
        const std::vector<double>& elite = X_[eidx];

        std::vector<double> x(D, 0.0);

        if (p < 0.5) {
            if (std::fabs(A) < 1.0) {
                for (int j=0; j<D; ++j){
                    double Dj = std::fabs(C * elite[j] - X_[i][j]);
                    double jitter = beta_ * N01(rng_);
                    x[j] = elite[j] - A * Dj + jitter;
                }
            } else {
                int r_idx = pick_rand_index(i);
                for (int j=0; j<D; ++j){
                    double Dj = std::fabs(C * X_[r_idx][j] - X_[i][j]);
                    double jitter = beta_ * N01(rng_);
                    x[j] = X_[r_idx][j] - A * Dj + jitter;
                }
            }
        } else {
            double l = Unegpos(rng_); // in [-1,1]
            for (int j=0; j<D; ++j){
                double Dj = std::fabs(elite[j] - X_[i][j]);
                double jitter = beta_ * N01(rng_);
                x[j] = Dj * std::exp(b_spiral_ * l) * std::cos(2.0 * kPi * l) + elite[j] + jitter;
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

    
    {
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > fw){ fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    buildElite();

    
    updateStop(FX_);
    printBest();
}

void MEWOA::end(){
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

    
    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
