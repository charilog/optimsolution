#include "wca.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void WCA::ensureBounds(Vec& v){
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

WCA::Vec WCA::randomVec(){
    const int D = prob_->dimension();
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    Vec x(D, 0.0);
    for (int d=0; d<D; ++d){
        double lo = (d < (int)L.size() ? L[d] : -1.0);
        double hi = (d < (int)U.size() ? U[d] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        x[d] = lo + U01(rng_) * (hi - lo);
    }
    return x;
}

double WCA::dist(const Vec& a, const Vec& b) const{
    const int D = (int)std::min(a.size(), b.size());
    double s = 0.0;
    for (int d=0; d<D; ++d){
        double diff = a[d] - b[d];
        s += diff*diff;
    }
    return std::sqrt(std::max(0.0, s));
}

double WCA::currentDmaxAbs() const{
    // Iteration-based progress (stable for large N)
    const int N = (int)X_.size();
    const double T_est = std::max(1.0, std::floor((double)max_evals_ / (double)std::max(1, N)));
    const double progress = std::min(1.0, std::max(0.0, (double)iter_ / T_est));

    auto to_abs = [&](double d)->double{
        if (d <= 1.0) return d * mean_range_;
        return d;
    };
    const double d0 = to_abs(dmax0_);
    const double d1 = to_abs(dmax_min_);
    const double d  = d0 + progress * (d1 - d0);
    // Compare against Euclidean distance: scale by sqrt(D)
    const int D = prob_->dimension();
    return d * std::sqrt(std::max(1, D));
}

void WCA::init(){
    if (!prob_) return;

    iter_ = 0;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    // mean range for scale interpretation
    {
        const Vec& L = prob_->lb();
        const Vec& U = prob_->ub();
        double sum = 0.0;
        int cnt = 0;
        for (int d=0; d<D; ++d){
            double lo = (d < (int)L.size() ? L[d] : -1.0);
            double hi = (d < (int)U.size() ? U[d] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            const double r = std::fabs(hi - lo);
            if (std::isfinite(r) && r > 0.0){ sum += r; ++cnt; }
        }
        mean_range_ = (cnt > 0 ? sum / (double)cnt : 1.0);
        if (!std::isfinite(mean_range_) || mean_range_ <= 0.0) mean_range_ = 1.0;
    }

    X_.clear();
    FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_){ best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_wca_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[wca] cfg -> N=%d, nsr=%d, C=%.4f, dmax0=%.6g, dmax_min=%.6g, rain_prob=%.3f, greedy=%s, in-run: %s @ %.3f, final@end: %s (%s)\n",
            N, nsr_, C_, dmax0_, dmax_min_, rain_prob_, greedy_ ? "on" : "off",
            lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void WCA::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Sort indices by fitness (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    // Update global best
    if (FX_[idx[0]] < best_f_){ best_f_ = FX_[idx[0]]; best_x_ = X_[idx[0]]; }

    const int NSR = std::min(std::max(2, nsr_), N);
    const int nstreams = std::max(0, N - NSR);
    const int sea = idx[0];

    std::vector<int> rivers;
    rivers.reserve(std::max(0, NSR-1));
    for (int k=1; k<NSR; ++k) rivers.push_back(idx[k]);

    std::vector<int> streams;
    streams.reserve(nstreams);
    for (int k=NSR; k<N; ++k) streams.push_back(idx[k]);

    // Weights for assignment (sea + rivers): better => larger weight
    double fworst = FX_[idx.back()];
    if (!std::isfinite(fworst)) fworst = FX_[sea];
    std::vector<double> w(NSR, 1.0);
    w[0] = std::max(0.0, fworst - FX_[sea]);
    for (int r=0; r<(int)rivers.size(); ++r){
        w[r+1] = std::max(0.0, fworst - FX_[rivers[r]]);
    }
    double sumw = std::accumulate(w.begin(), w.end(), 0.0);
    if (!(sumw > 0.0) || !std::isfinite(sumw)){
        std::fill(w.begin(), w.end(), 1.0);
        sumw = (double)w.size();
    }

    // Determine number of streams per target
    std::vector<int> cnt(NSR, 0);
    int assigned = 0;
    for (int t=0; t<NSR; ++t){
        cnt[t] = (nstreams > 0) ? (int)std::floor((w[t] / sumw) * (double)nstreams) : 0;
        if (cnt[t] < 0) cnt[t] = 0;
        assigned += cnt[t];
    }
    // Distribute the remainder to the best targets
    int rem = nstreams - assigned;
    int tbest = 0;
    for (int t=1; t<NSR; ++t) if (w[t] > w[tbest]) tbest = t;
    while (rem-- > 0) cnt[tbest]++;

    // Map each stream to a target: 0 -> sea, 1.. -> river index in 'rivers'
    std::vector<int> targetOf(streams.size(), 0);
    int spos = 0;
    for (int t=0; t<NSR; ++t){
        for (int c=0; c<cnt[t] && spos<(int)streams.size(); ++c){
            targetOf[spos++] = t;
        }
    }
    for (; spos<(int)streams.size(); ++spos) targetOf[spos] = tbest;

    auto try_improve = [&](int i, const Vec& cand, double fcand){
        if (!std::isfinite(fcand)) return;
        if (greedy_){
            if (fcand < FX_[i]) {
                X_[i] = cand;
                FX_[i] = fcand;
            }
        } else {
            X_[i] = cand;
            FX_[i] = fcand;
        }
        if (FX_[i] < best_f_){ best_f_ = FX_[i]; best_x_ = X_[i]; }
    };

    // Move rivers toward sea
    for (int rid : rivers){
        Vec xnew = X_[rid];
        for (int d=0; d<D; ++d){
            xnew[d] = xnew[d] + U01(rng_) * C_ * (X_[sea][d] - xnew[d]);
        }
        ensureBounds(xnew);
        double fnew = eval(xnew);
        if (local_rate_ > 0.0 && !local_method_.empty()){
            if (fnew < FX_[rid] && U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew){ xnew = std::move(xloc); fnew = floc; }
            }
        }
        try_improve(rid, xnew, fnew);
        if (prob_->calls() >= max_evals_) break;
    }

    if (prob_->calls() < max_evals_){
        // Move streams toward their assigned target
        for (size_t si=0; si<streams.size(); ++si){
            const int sid = streams[si];
            const int t = targetOf[si];
            const int tid = (t == 0 ? sea : rivers[t-1]);

            Vec xnew = X_[sid];
            for (int d=0; d<D; ++d){
                xnew[d] = xnew[d] + U01(rng_) * C_ * (X_[tid][d] - xnew[d]);
            }
            ensureBounds(xnew);
            double fnew = eval(xnew);
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (fnew < FX_[sid] && U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, xnew);
                    if (std::isfinite(floc) && floc < fnew){ xnew = std::move(xloc); fnew = floc; }
                }
            }
            try_improve(sid, xnew, fnew);
            if (prob_->calls() >= max_evals_) break;
        }
    }

    // Swap: if any stream beats its river, swap them
    for (size_t si=0; si<streams.size(); ++si){
        const int sid = streams[si];
        const int t = targetOf[si];
        if (t == 0) continue;
        const int rid = rivers[t-1];
        if (FX_[sid] < FX_[rid]){
            std::swap(X_[sid], X_[rid]);
            std::swap(FX_[sid], FX_[rid]);
        }
    }

    // Swap: if any river beats sea, swap
    for (int rid : rivers){
        if (FX_[rid] < FX_[sea]){
            std::swap(X_[rid], X_[sea]);
            std::swap(FX_[rid], FX_[sea]);
        }
    }

    // Evaporation / raining
    const double dthr = currentDmaxAbs();
    if (dthr > 0.0 && prob_->calls() < max_evals_){
        // For each riveriver close to the sea: reinitialize its assigned streams
        for (int rpos=0; rpos<(int)rivers.size(); ++rpos){
            const int rid = rivers[rpos];
            if (dist(X_[rid], X_[sea]) < dthr){
                for (size_t si=0; si<streams.size(); ++si){
                    if (targetOf[si] == rpos+1){
                        Vec xr = randomVec();
                        ensureBounds(xr);
                        double fr = eval(xr);
                        X_[streams[si]] = std::move(xr);
                        FX_[streams[si]] = fr;
                        if (fr < best_f_){ best_f_ = fr; best_x_ = X_[streams[si]]; }
                        if (prob_->calls() >= max_evals_) break;
                    }
                }
            }
            if (prob_->calls() >= max_evals_) break;
        }

        // Streams close to sea (assigned to sea) may be reinitialized with rain probability
        if (prob_->calls() < max_evals_){
            for (size_t si=0; si<streams.size(); ++si){
                if (targetOf[si] != 0) continue;
                const int sid = streams[si];
                if (dist(X_[sid], X_[sea]) < dthr && U01(rng_) < rain_prob_){
                    Vec xr = randomVec();
                    ensureBounds(xr);
                    double fr = eval(xr);
                    X_[sid] = std::move(xr);
                    FX_[sid] = fr;
                    if (fr < best_f_){ best_f_ = fr; best_x_ = X_[sid]; }
                    if (prob_->calls() >= max_evals_) break;
                }
            }
        }
    }

    // Elitism: keep best inside population
    if (!FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val){ worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx] = best_x_;
        FX_[worst_idx] = best_f_;
    }

    ++iter_;

    printBest();
    updateStop(FX_);
}

void WCA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_){
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    if (!X_.empty() && !FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val){ worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx] = best_x_;
        FX_[worst_idx] = best_f_;
    }

    printBest();
}

} // namespace optimsolution
