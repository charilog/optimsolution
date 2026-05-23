#include "gsa.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void GSA::ensureBounds(Vec& v){
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

void GSA::clampVelocity(Vec& vel){
    if (vmax_scale_ <= 0.0) return;
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<vel.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        const double vmax = std::fabs(vmax_scale_ * (hi - lo));
        if (vmax <= 0.0) continue;
        if (!std::isfinite(vel[j])) vel[j] = 0.0;
        if (vel[j] >  vmax) vel[j] =  vmax;
        if (vel[j] < -vmax) vel[j] = -vmax;
    }
}

int GSA::currentKbest(int N) const{
    if (N <= 2) return N;
    if (fixed_kbest_ >= 2) return std::min(N, std::max(2, fixed_kbest_));

    // ratio-based (possibly adaptive)
    int k = std::max(2, (int)std::lround(kbest_ratio_ * (double)N));
    if (!adaptive_kbest_) return std::min(N, k);

    // linearly decrease from N to 2 with iteration progress
    double progress = 0.0;
    if (max_iters_est_ > 0) progress = (double)iter_ / (double)max_iters_est_;
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;
    const double kk = (double)N - (double)(N - 2) * progress;
    k = (int)std::floor(kk + 1e-12);
    k = std::max(2, std::min(N, k));
    return k;
}

void GSA::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); V_.clear(); FX_.clear();
    X_.reserve(N);
    V_.reserve(N);

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    V_.assign(N, Vec(D, 0.0));
    FX_.assign(N, std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    iter_ = 0;
    // Estimate maximum iterations from the evaluation budget.
    // Using calls/max_evals as "progress" makes G(t) decay too fast when N is large.
    // Classic GSA uses iteration-based decay, so approximate T ~= max_evals / N.
    max_iters_est_ = 1;
    if (max_evals_ > 0) {
        const long long T = (long long)max_evals_ / (long long)std::max(1, N);
        max_iters_est_ = std::max(1LL, T);
    }

    if (debug_gsa_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[gsa] cfg -> N=%d, est_T=%lld, G0=%.6g, alpha=%.6g, w=%.4f, kbest_ratio=%.3f, fixed_kbest=%d, adaptive=%s, vmax_scale=%.4f, in-run: %s @ %.3f, final@end: %s (%s)\n",
            N, (long long)max_iters_est_, G0_, alpha_, w_, kbest_ratio_, fixed_kbest_, adaptive_kbest_ ? "on" : "off", vmax_scale_,
            lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void GSA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Fitness extremes
    double fbest = FX_[0], fworst = FX_[0];
    int ibest = 0, iworst = 0;
    for (int i=1; i<N; ++i){
        if (FX_[i] < fbest){ fbest = FX_[i]; ibest = i; }
        if (FX_[i] > fworst){ fworst = FX_[i]; iworst = i; }
    }

    // Update global best (robust)
    if (fbest < best_f_){
        best_f_ = fbest;
        best_x_ = X_[ibest];
    }

    // Compute masses (minimization): larger mass for better (smaller) fitness
    std::vector<double> m(N, 1.0);
    const double denom = (fworst - fbest);
    if (std::isfinite(denom) && std::fabs(denom) > 0.0){
        for (int i=0; i<N; ++i){
            double mi = (fworst - FX_[i]) / denom;
            if (!std::isfinite(mi) || mi < 0.0) mi = 0.0;
            m[i] = mi;
        }
    } else {
        std::fill(m.begin(), m.end(), 1.0);
    }

    double sum_m = std::accumulate(m.begin(), m.end(), 0.0);
    if (!std::isfinite(sum_m) || sum_m <= 0.0) sum_m = 1.0;
    for (int i=0; i<N; ++i) m[i] /= sum_m;

    // Rank indices by fitness (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    const int K = currentKbest(N);

    // progress and gravitational constant decay (iteration-based)
    double progress = 0.0;
    if (max_iters_est_ > 0) progress = (double)iter_ / (double)max_iters_est_;
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;

    const double G = G0_ * std::exp(-alpha_ * progress);

    // Compute accelerations and update
    std::vector<Vec> A(N, Vec(D, 0.0));

    for (int ii=0; ii<N; ++ii){
        const int i = ii;

        // force accumulation over the K best agents
        for (int kk=0; kk<K; ++kk){
            const int j = idx[kk];
            if (j == i) continue;

            // Euclidean distance
            double R2 = 0.0;
            for (int d=0; d<D; ++d){
                const double diff = (X_[j][d] - X_[i][d]);
                R2 += diff*diff;
            }
            double R = std::sqrt(std::max(0.0, R2));
            R += eps_;

            const double coeff = U01(rng_) * G * m[j] / R; // a_i uses m[j] only

            for (int d=0; d<D; ++d){
                A[i][d] += coeff * (X_[j][d] - X_[i][d]);
            }
        }
    }

    // Move agents (standard GSA updates positions regardless of improvement)
    int best_idx_this_iter = ibest;
    double best_val_this_iter = fbest;

    for (int i=0; i<N; ++i){
        Vec xnew = X_[i];
        Vec vnew = V_[i];

        for (int d=0; d<D; ++d){
            const double r = U01(rng_);
            vnew[d] = w_ * vnew[d] + r * A[i][d];
            if (!std::isfinite(vnew[d])) vnew[d] = 0.0;
            xnew[d] += vnew[d];
        }

        clampVelocity(vnew);
        ensureBounds(xnew);

        double fnew = eval(xnew);

        // optional in-run local refinement (probabilistic)
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew){
                    xnew = std::move(xloc);
                    fnew = floc;
                }
            }
        }

        X_[i]  = std::move(xnew);
        V_[i]  = std::move(vnew);
        FX_[i] = fnew;

        if (FX_[i] < best_val_this_iter){
            best_val_this_iter = FX_[i];
            best_idx_this_iter = i;
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // update global best
    if (best_val_this_iter < best_f_){
        best_f_ = best_val_this_iter;
        best_x_ = X_[best_idx_this_iter];
    }

    // elitism: write best into worst slot so reporters that look at min(FX_) are always correct/stable
    if (!X_.empty() && !FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()){
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    ++iter_;
    printBest();
    updateStop(FX_);
}

void GSA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // place refined best into worst slot
    if (!X_.empty() && !FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()){
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    printBest();
}

} // namespace optimsolution
