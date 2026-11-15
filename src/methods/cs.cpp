#include "cs.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

void CS::init(){
    if (!prob_) return;

    const int D = prob_->dimension();

    // population=0 (or unset): literature default. Yang & Deb's own
    // "Cuckoo Search via Levy Flights" paper (2009/arXiv:1003.1594) states:
    // "we found that n=15 ... are sufficient for most optimization
    // problems. Therefore, we will use fixed n=15 ... in the rest of the
    // simulations." Previously population=0 silently fell back to
    // [global] population (default 100) instead of CS's own default -- fixed here.
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : 15));

    // Synchronize for reporters
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
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_cs_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[cs] cfg -> N=%d (population() now=%d, override=%d), pa=%.4f, alpha=%.6f, beta=%.4f, dir=%s, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, pa_, alpha_, beta_,
            use_best_direction_ ? "best" : "range",
            lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

int CS::pickIndex(int n){
    std::uniform_int_distribution<int> I(0, n-1);
    return I(rng_);
}

void CS::ensureBounds(Vec& v){
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

CS::Vec CS::randomVector(int D){
    Vec x((size_t)D, 0.0);
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int j=0; j<D; ++j){
        double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
        double hi = (j < (int)U.size() ? U[(size_t)j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        std::uniform_real_distribution<double> Uj(lo, hi);
        x[(size_t)j] = Uj(rng_);
    }
    return x;
}

CS::Vec CS::levyFlightStep(int D){
    // Mantegna algorithm for Levy stable distribution
    // beta_ in (1,2]
    const double beta = beta_;
    const double pi = 3.14159265358979323846264338327950288;

    double num = std::tgamma(1.0 + beta) * std::sin(pi * beta / 2.0);
    double den = std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0);
    double sigma_u = std::pow(num / den, 1.0 / beta);

    std::normal_distribution<double> N_u(0.0, sigma_u);
    std::normal_distribution<double> N_v(0.0, 1.0);

    Vec step((size_t)D, 0.0);
    for (int j=0; j<D; ++j){
        double u = N_u(rng_);
        double v = N_v(rng_);
        double av = std::fabs(v);
        if (av < 1e-12) av = 1e-12;
        step[(size_t)j] = u / std::pow(av, 1.0 / beta);
        if (!std::isfinite(step[(size_t)j])) step[(size_t)j] = 0.0;
    }
    return step;
}

void CS::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    auto rangeAt = [&](int j)->double{
        double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
        double hi = (j < (int)U.size() ? U[(size_t)j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double r = hi - lo;
        return (std::isfinite(r) && r > 0.0) ? r : 1.0;
    };

    auto worstIndex = [&]()->size_t{
        size_t wi = 0;
        double wv = FX_.empty() ? std::numeric_limits<double>::infinity() : FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > wv) { wv = FX_[k]; wi = k; }
        }
        return wi;
    };

    auto maybeLocalImprove = [&](Vec& x, double& fx){
        if (local_rate_ <= 0.0 || local_method_.empty()) return;
        if (prob_->calls() >= max_evals_) return;
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, x);
            if (std::isfinite(floc) && floc < fx) {
                x  = std::move(xloc);
                fx = floc;
            }
        }
    };

    // ------------------------------------------------------------------
    // step_mode = best_diff (default): CANONICAL Cuckoo Search, a line-by-line
    // port of the authors' reference code (X.-S. Yang, cuckoo_search.m,
    // Yang & Deb 2009/2010):
    //   get_cuckoos : s' = s + alpha * Levy(beta) .* (s - best) .* N(0,1)
    //   get_best_nest: nest i is replaced by its OWN new solution if better
    //   empty_nests : K = (rand > pa) per component,
    //                 s' = s + rand * (nest[perm1] - nest[perm2]) .* K, greedy
    // FIX: the previous move used a FIXED step alpha*(ub-lb)*Levy, which can
    // not refine below that scale (sphere-10D stalled at f ~ 1), compared the
    // new solution with a RANDOM nest (so near-copies of good nests overwrote
    // the others and the population collapsed), and replaced the worst nests
    // in the discovery phase without a greedy test. The old operators are
    // kept as step_mode = range.
    // ------------------------------------------------------------------
    if (step_mode_best_diff_) {
        std::normal_distribution<double> N01(0.0, 1.0);
        auto offer = [&](size_t i, Vec& cand){
            ensureBounds(cand);
            double fc = eval(cand);
            maybeLocalImprove(cand, fc);
            if (fc < FX_[i]) { X_[i] = cand; FX_[i] = fc; }
            if (fc < best_f_) { best_f_ = fc; best_x_ = cand; }
        };
        // get_cuckoos + get_best_nest
        for (int i=0; i<N; ++i){
            if (prob_->calls() >= max_evals_) break;
            Vec step = levyFlightStep(D);
            Vec u = X_[(size_t)i];
            for (int j=0; j<D; ++j){
                const double diff = X_[(size_t)i][(size_t)j] - best_x_[(size_t)j];
                u[(size_t)j] += alpha_ * step[(size_t)j] * diff * N01(rng_);
            }
            offer((size_t)i, u);
        }
        // empty_nests + get_best_nest
        if (pa_ > 0.0 && N >= 2) {
            std::vector<size_t> p1((size_t)N), p2((size_t)N);
            std::iota(p1.begin(), p1.end(), 0);
            std::iota(p2.begin(), p2.end(), 0);
            std::shuffle(p1.begin(), p1.end(), rng_);
            std::shuffle(p2.begin(), p2.end(), rng_);
            const double r = U01(rng_);
            const std::vector<Vec> nest = X_;     // snapshot, as in the reference
            for (int i=0; i<N; ++i){
                if (prob_->calls() >= max_evals_) break;
                Vec xnew = nest[(size_t)i];
                bool changed = false;
                for (int j=0; j<D; ++j){
                    if (U01(rng_) > pa_) {
                        xnew[(size_t)j] += r * (nest[p1[(size_t)i]][(size_t)j] - nest[p2[(size_t)i]][(size_t)j]);
                        changed = true;
                    }
                }
                if (!changed) continue;           // identical point: no evaluation
                offer((size_t)i, xnew);
            }
        }
        printBest();
        updateStop(FX_);
        return;
    }

    // Phase 1: Levy flights
    // Canonical CS move: x_new = x + alpha * Levy(beta), where alpha is typically a small fraction of the domain range.
    // If use_best_direction_ is enabled, add a mild drift term towards the best solution to stabilize convergence.
    for (int i=0; i<N; ++i){
        if (prob_->calls() >= max_evals_) break;

        Vec step = levyFlightStep(D);
        Vec u = X_[(size_t)i];

        for (int j=0; j<D; ++j){
            const double r = rangeAt(j);

            // Interpret alpha_ as relative scale when alpha_ <= 1, otherwise as absolute step scale.
            const double a = (alpha_ <= 1.0 ? alpha_ * r : alpha_);

            u[(size_t)j] += a * step[(size_t)j];

            if (use_best_direction_) {
                // Mild drift towards current best (keeps search stable on easy convex problems).
                const double drift = U01(rng_);
                u[(size_t)j] += drift * (best_x_[(size_t)j] - X_[(size_t)i][(size_t)j]);
            }
        }

        ensureBounds(u);
        double fu = eval(u);

        maybeLocalImprove(u, fu);

        // Replace a random nest if improved (greedy replacement)
        const int k = pickIndex(N);
        if (fu < FX_[(size_t)k]) {
            X_[(size_t)k]  = u;
            FX_[(size_t)k] = fu;
        }

        // Track global best over all evaluated solutions (not only accepted replacements)
        if (fu < best_f_) {
            best_f_ = fu;
            best_x_ = u;

            // Ensure best exists inside population (elitism)
            if (!X_.empty() && !FX_.empty()) {
                size_t wi = worstIndex();
                if (best_f_ < FX_[wi]) {
                    X_[wi]  = best_x_;
                    FX_[wi] = best_f_;
                }
            }
        }
    }

    // Phase 2: Discovery/abandonment
    // Replace a fraction of the worst nests using a differential random walk (standard CS discovery operator).
    if (prob_->calls() < max_evals_ && pa_ > 0.0 && N >= 4) {
        int m = (int)std::floor(pa_ * (double)N);
        if (m < 1) m = 1;
        if (m > N) m = N;

        std::vector<size_t> idx((size_t)N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            return FX_[a] > FX_[b]; // worst first
        });

        for (int t=0; t<m; ++t){
            if (prob_->calls() >= max_evals_) break;

            const size_t wi = idx[(size_t)t];

            int r1 = pickIndex(N);
            int r2 = pickIndex(N);
            while (r2 == r1) r2 = pickIndex(N);

            Vec xnew = X_[wi];
            for (int j=0; j<D; ++j){
                const double eps = U01(rng_);
                xnew[(size_t)j] += eps * (X_[(size_t)r1][(size_t)j] - X_[(size_t)r2][(size_t)j]);
            }

            ensureBounds(xnew);
            double fnew = eval(xnew);

            maybeLocalImprove(xnew, fnew);

            X_[wi]  = std::move(xnew);
            FX_[wi] = fnew;

            if (fnew < best_f_) {
                best_f_ = fnew;
                best_x_ = X_[wi];
            }
        }
    }

    // Final elitism (ensure best exists in population)
    if (!X_.empty() && !FX_.empty()) {
        size_t wi = 0;
        double wv = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > wv) { wv = FX_[k]; wi = k; }
        }
        if (best_f_ < FX_[wi]) {
            X_[wi]  = best_x_;
            FX_[wi] = best_f_;
        }
    }

    printBest();
    updateStop(FX_);
}

void CS::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best to worst slot
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
