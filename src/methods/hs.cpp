#include "hs.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static inline std::string to_lower(std::string s){
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void HS::init(){
    if (!prob_) return;

    const int D = prob_->dimension();

    // If an override exists in [hs], it takes precedence; otherwise [global] population() is used
    const int N = std::max(3, (pop_override_ >= 3 ? pop_override_ : population()));

    // Synchronization for consistency with the reporter
    this->setPopulation(N);

    X_.clear();
    FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_hs_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[hs] cfg -> HMS=%d (population() now=%d, override=%d), HMCR=%.6f, PAR=%.6f, bw_scale=%.6f, bw_abs=%s, adaptive_bw=%s, improvisations=%d, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, HMCR_, PAR_, bw_scale_,
            std::isfinite(bw_abs_) ? "on" : "off", adaptive_bw_ ? "on" : "off", improvisations_,
            lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

size_t HS::worstIndex() const {
    if (FX_.empty()) return 0;
    size_t wi = 0;
    double wv = FX_[0];
    for (size_t i=1; i<FX_.size(); ++i){
        if (FX_[i] > wv) { wv = FX_[i]; wi = i; }
    }
    return wi;
}

void HS::ensureBounds(Vec& v){
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

double HS::bandwidthForDim(int j) const {
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
    double hi = (j < (int)U.size() ? U[(size_t)j] :  1.0);
    if (lo > hi) std::swap(lo, hi);
    double range = hi - lo;
    if (!std::isfinite(range) || range <= 0.0) range = 1.0;

    double bw0 = std::isfinite(bw_abs_) ? bw_abs_ : bw_scale_ * range;
    double bw1 = std::isfinite(bw_min_abs_) ? bw_min_abs_ : bw_min_scale_ * range;
    if (bw0 < 0.0) bw0 = 0.0;
    if (bw1 < 0.0) bw1 = 0.0;

    if (!adaptive_bw_) {
        return (bw0 > 0.0 ? bw0 : 1e-12 * range);
    }

    double t = 0.0;
    if (max_evals_ > 0) {
        t = (double)prob_->calls() / (double)max_evals_;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
    }

    double bw = bw0 * (1.0 - t) + bw1 * t;
    if (!std::isfinite(bw) || bw <= 0.0) bw = 1e-12 * range;
    return bw;
}

void HS::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     I(0, std::max(0, N-1));

    const int K = std::max(1, (improvisations_ > 0 ? improvisations_ : N));

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int it=0; it<K; ++it){
        Vec u(D, 0.0);

        for (int j=0; j<D; ++j){
            double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
            double hi = (j < (int)U.size() ? U[(size_t)j] :  1.0);
            if (lo > hi) std::swap(lo, hi);

            if (U01(rng_) < HMCR_ && N > 0) {
                int k = I(rng_);
                u[j] = X_[(size_t)k][(size_t)j];
                if (U01(rng_) < PAR_) {
                    double bw = bandwidthForDim(j);
                    u[j] += (U01(rng_) * 2.0 - 1.0) * bw;
                }
            } else {
                double r = U01(rng_);
                u[j] = lo + r * (hi - lo);
            }
        }

        ensureBounds(u);

        double fu = eval(u);

        // optional in-run local improvement (applied to the improvised harmony)
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, u);
                if (std::isfinite(floc) && floc < fu){
                    u  = std::move(xloc);
                    fu = floc;
                }
            }
        }

        // replace the worst harmony if improved
        size_t w = worstIndex();
        if (w < FX_.size() && fu < FX_[w]){
            X_[w]  = u;
            FX_[w] = fu;

            if (fu < best_f_){
                best_f_ = fu;
                best_x_ = u;
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Ensure elitism: write the current best into the worst slot if it is missing
    if (!X_.empty() && !FX_.empty()) {
        double minv = FX_[0];
        for (size_t i=1; i<FX_.size(); ++i) minv = std::min(minv, FX_[i]);
        if (std::isfinite(best_f_) && best_f_ < minv) {
            size_t w = worstIndex();
            if (w < X_.size()) {
                X_[w]  = best_x_;
                FX_[w] = best_f_;
            }
        }
    }

    printBest();
    updateStop(FX_);
}

void HS::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best into the worst position
    if (!X_.empty() && !FX_.empty()) {
        size_t w = worstIndex();
        if (w < X_.size()) {
            X_[w]  = best_x_;
            FX_[w] = best_f_;
        }
    }

    printBest();
}

} // namespace optimsolution
