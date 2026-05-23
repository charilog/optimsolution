#include "egco.h"
#include "init.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>

namespace optimsolution {

// ----------------- small helpers -----------------

static inline bool isFinite(double x){ return std::isfinite(x); }

static inline double big_penalty(){
    return 1e100; // Large finite value to prevent infinity from persisting
}

// Clamp to bounds
void EGCO::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int j=0; j<(int)x.size(); ++j){
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo,hi);
        if (!std::isfinite(x[j])) x[j] = 0.5*(lo+hi);
        if (x[j] < lo) x[j] = lo;
        if (x[j] > hi) x[j] = hi;
    }
}

// Progress in [0,1] based on iterations
double EGCO::progress01() const
{
    if (max_iters_ <= 0) return 0.0;
    double t = std::min<double>(1.0, std::max<double>(0.0, double(iters_) / double(max_iters_)));
    return t;
}

// Finds the three best indices (alpha, beta, delta)
void EGCO::leadersABC(int& a, int& b, int& c) const
{
    a = 0; double fa = FX_[0];
    b = 0; double fb = std::numeric_limits<double>::infinity();
    c = 0; double fc = std::numeric_limits<double>::infinity();

    // First pass: find the best
    for (int i=1;i<(int)FX_.size();++i){
        if (FX_[i] < fa){ fa = FX_[i]; a = i; }
    }
    // Second/third pass: find the next two
    for (int i=0;i<(int)FX_.size();++i){
        if (i==a) continue;
        if (FX_[i] < fb){ fc = fb; c = b; fb = FX_[i]; b = i; }
        else if (FX_[i] < fc){ fc = FX_[i]; c = i; }
    }
}

// Centroid of the top-k (k>=1)
EGCO::Vec EGCO::topCentroid(int k) const
{
    k = std::max(1, std::min(k, (int)X_.size()));
    std::vector<int> idx(X_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
        [&](int i, int j){ return FX_[i] < FX_[j]; });

    const int D = (X_.empty() ? 0 : (int)X_[0].size());
    Vec c(D, 0.0);
    for (int t=0; t<k; ++t){
        const Vec& v = X_[ idx[t] ];
        for (int j=0;j<D;++j) c[j] += v[j];
    }
    for (int j=0;j<D;++j) c[j] /= double(k);
    return c;
}

// Injects the elite into the worst slot (optional)
void EGCO::elitismInject()
{
    if (X_.empty()) return;
    int worst = 0; double wf = FX_[0];
    for (int i=1;i<(int)FX_.size();++i){
        if (FX_[i] > wf){ wf = FX_[i]; worst = i; }
    }
    X_[worst]  = best_x_;
    FX_[worst] = best_f_;
}

// ----------------- lifecycle -----------------

void EGCO::init()
{
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, population());

    // Initial population via Initializer (as in GA/BHO)
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    V_.assign(N, Vec(D, 0.0));
    FX_.assign(N, std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i=0;i<N;++i){
        double f = eval(X_[i]);
        if (!isFinite(f)) f = big_penalty();
        FX_[i] = f;
        if (f < best_f_){ best_f_ = f; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    iters_ = 0;
    printBest();
}

void EGCO::one_iteration()
{
    if (!prob_) return;
    ++iters_;

    const int N = (int)X_.size();
    const int D = (N>0 ? (int)X_[0].size() : 0);
    if (N == 0 || D == 0) return;

    // leaders + centroid
    int ia, ib, ic; leadersABC(ia, ib, ic);
    const Vec centroidTop = topCentroid(std::max(3, (int)std::round(0.15 * N)));

    // eel vs grouper partition
    int eel_n = std::max(1, (int)std::floor(eel_frac_ * N));
    // A fixed permutation is created to avoid random_shuffle on every iteration
    std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
    // A lightweight selection can be used to pick the best as "groupers"; here the existing order is retained
    // If the worst are intended as eels, FX_ can be sorted in descending order

    // EEL step: the first eel_n individuals
    for (int t=0; t<eel_n; ++t){
        int i = order[t];
        if (i == ia) continue; // Leaves the absolute elite unchanged
        Vec y = X_[i];

        // Updates the "velocity"
        for (int j=0; j<D; ++j){
            double dir = (1.0 - coop_bias_) * (best_x_[j] - X_[i][j])
                       + (coop_bias_)      * (centroidTop[j] - X_[i][j]);
            // Small random jitter (controlled by jitter_sigma_)
            double rnd = (jitter_sigma_>0.0 ? jitter_sigma_ * std::normal_distribution<double>(0.0,1.0)(rng_) : 0.0);
            V_[i][j] = eel_inertia_ * V_[i][j] + eel_step_ * dir + rnd;
            y[j] = X_[i][j] + V_[i][j];
        }

        ensureBounds(y);
        double fy = eval(y);
        if (!isFinite(fy)) fy = big_penalty();

        if (fy < FX_[i]){
            // In-run local search with probability local_rate_
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (std::uniform_real_distribution<double>(0.0,1.0)(rng_) < local_rate_){
                    auto [xl, fl] = localSearch(local_method_, y);
                    if (fl < fy){ y = std::move(xl); fy = fl; }
                }
            }
            X_[i] = std::move(y);
            FX_[i] = fy;
            if (fy < best_f_){ best_f_ = fy; best_x_ = X_[i]; }
        }

        if (prob_->calls() >= max_evals_) { printBest(); updateStop(FX_); return; }
    }

    // GROUPER step: the remaining individuals
    for (int t=eel_n; t<N; ++t){
        int i = order[t];
        if (i == ia) continue;

        Vec y = X_[i];
        for (int j=0; j<D; ++j){
            // Attraction towards best plus shrinkage towards centroidTop
            double range = 1.0;
            const auto& L = prob_->lb();
            const auto& U = prob_->ub();
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo,hi);
            range = (hi - lo);

            double attract = grp_beta_  * (best_x_[j]     - X_[i][j]);
            double shrink  = grp_gamma_ * (centroidTop[j] - X_[i][j]);
            double jitter  = grp_jitter_ * range * std::normal_distribution<double>(0.0,1.0)(rng_);
            y[j] = X_[i][j] + attract + shrink + jitter;
        }

        ensureBounds(y);
        double fy = eval(y);
        if (!isFinite(fy)) fy = big_penalty();

        if (fy < FX_[i]){
            // In-run local search with probability local_rate_
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (std::uniform_real_distribution<double>(0.0,1.0)(rng_) < local_rate_){
                    auto [xl, fl] = localSearch(local_method_, y);
                    if (fl < fy){ y = std::move(xl); fy = fl; }
                }
            }
            X_[i] = std::move(y);
            FX_[i] = fy;
            if (fy < best_f_){ best_f_ = fy; best_x_ = X_[i]; }
        }

        if (prob_->calls() >= max_evals_) { printBest(); updateStop(FX_); return; }
    }

    // Prints and updates stop-rules
    printBest();
    updateStop(FX_);
}

void EGCO::end()
{
    // Final polishing controlled by [global]
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_){
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Optional elite injection into the population
    if (!X_.empty() && !FX_.empty()){
        elitismInject();
    }
    printBest();
}

} // namespace optimsolution
