#include "hba.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static inline double clamp01(double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

double HBA::meanRange() const {
    if (!prob_) return 1.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = prob_->dimension();
    double s = 0.0;
    for (int j=0; j<D; ++j){
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double r = hi - lo;
        if (!std::isfinite(r) || r <= 0.0) r = 1.0;
        s += r;
    }
    s /= std::max(1, D);
    if (!std::isfinite(s) || s <= 0.0) s = 1.0;
    return s;
}

void HBA::ensureBounds(Vec& v){
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

void HBA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    iter_ = 0;

    if (debug_hba_) {
        std::fprintf(stdout,
            "[hba] cfg -> N=%d (population() now=%d, override=%d), beta=%.6g, C=%.6g, p_dig=%.3f, eps=%.3g, i_cap=%.3g, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, beta_, C_, pdig_, eps_, i_cap_,
            (local_method_.empty() ? "none" : local_method_.c_str()), local_rate_,
            end_local_refine_ ? "on" : "off", (end_local_method_.empty() ? "none" : end_local_method_.c_str()));
        std::fflush(stdout);
    }

    printBest();
}

void HBA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    // iteration horizon estimate based on max_evals and population
    const long long T = std::max(1LL, (long long)std::max(1, (int)(max_evals_ / std::max(1, N))));
    const double progress = clamp01((double)iter_ / (double)T);

    // density factor alpha = C * exp(-t/tmax)
    const double alpha = C_ * std::exp(-progress);

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Iidx(0, N-1);

    constexpr double PI = 3.141592653589793238462643383279502884;

    for (int i=0; i<N; ++i){
        const double r2 = U01(rng_);
        const double r3 = U01(rng_);
        const double r4 = U01(rng_);
        const double r5 = U01(rng_);
        const double r6 = U01(rng_);
        const double r7 = U01(rng_);

        const double F = (r6 <= 0.5 ? 1.0 : -1.0);

        // distance vector to prey (global best)
        Vec dvec(D, 0.0);
        double dist2 = 0.0;
        for (int j=0; j<D; ++j){
            dvec[j] = best_x_[j] - X_[i][j];
            dist2 += dvec[j] * dvec[j];
        }

        // concentration strength S (neighbor signal) and smell intensity I
        int k = Iidx(rng_);
        if (k == i) k = (k + 1) % N;
        double S = 0.0;
        for (int j=0; j<D; ++j){
            const double t = X_[i][j] - X_[k][j];
            S += t * t;
        }
        S = std::sqrt(std::max(0.0, S)) + eps_;

        const double I = std::min(i_cap_, r2 * S / (4.0 * PI * (dist2 + eps_)));

        // cardioid term for digging
        const double cardioid = std::fabs(std::cos(2.0 * PI * r4) * (1.0 - std::cos(2.0 * PI * r5)));

        Vec xnew = X_[i];

        if (U01(rng_) < pdig_) {
            // Digging phase:
            // x_new = x_prey + F*beta*I*x_prey + F*r3*alpha*d_i*|cos(2π r4) * (1 - cos(2π r5))|
            for (int j=0; j<D; ++j){
                const double term1 = best_x_[j];
                const double term2 = F * beta_ * I * best_x_[j];
                const double term3 = F * r3 * alpha * dvec[j] * cardioid;
                xnew[j] = term1 + term2 + term3;
            }
        } else {
            // Honey phase:
            // x_new = x_prey + F*r7*alpha*d_i
            for (int j=0; j<D; ++j){
                xnew[j] = best_x_[j] + F * r7 * alpha * dvec[j];
            }
        }

        ensureBounds(xnew);
        double fnew = eval(xnew);

        // Greedy update (more stable in continuous benchmarks)
        bool accepted = false;
        if (fnew < FX_[i]) {
            if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew) { xnew = std::move(xloc); fnew = floc; }
            }
            X_[i]  = std::move(xnew);
            FX_[i] = fnew;
            accepted = true;
        }

        if (accepted && FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    // Elitism: write best to the worst slot for consistent reporting
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

    ++iter_;
    printBest();
    updateStop(FX_);
}

void HBA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best to worst position
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
