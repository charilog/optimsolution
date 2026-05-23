#include "polyde.h"

#include <chrono>
#include <cctype>
#include <cstdio>

namespace optimsolution {

// --- small helper for seeding ---

static inline uint64_t splitmix64_polyde(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// --- bounds / distances ---

void PolyphaseDE::ensureBounds(Vec& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double PolyphaseDE::bnDistance_(const Vec& a, const Vec& b) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double sum = 0.0;
    const double eps = 1e-12;
    for (size_t j = 0; j < a.size(); ++j) {
        double r = U[j] - L[j];
        double d = (a[j] - b[j]) / (r + eps);
        sum += d * d;
    }
    return std::sqrt(sum);
}

// --- progress 0..1 ---

double PolyphaseDE::progress01_() const {
    if (!prob_) return 0.0;
    if (max_evals_ <= 0) return 0.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

// --- diversity around best (normalized) ---

double PolyphaseDE::diversityBestNorm_() const {
    const int N = (int)X_.size();
    if (N <= 1 || !prob_) return 0.0;

    int    best_i = 0;
    double bf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
        double f = FX_[i];
        if (std::isfinite(f) && f < bf) {
            bf     = f;
            best_i = i;
        }
    }

    double sum = 0.0;
    int    cnt = 0;
    for (int i = 0; i < N; ++i) {
        if (i == best_i) continue;
        if (!std::isfinite(FX_[i])) continue;
        double d = bnDistance_(X_[best_i], X_[i]);
        sum += d;
        cnt++;
    }
    if (cnt == 0) return 0.0;

    double mean = sum / (double)cnt;
    double D    = (double)prob_->dimension();
    double norm = mean / (std::sqrt(D) + 1e-12);
    return norm;
}

// --- archive helpers ---

void PolyphaseDE::archivePush_(const Vec& x) {
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    int N = population();
    if (N <= 0) return;

    size_t cap = (size_t)std::max(1.0, std::round(archive_rate_ * (double)N));
    if (archive_.size() > cap) {
        std::uniform_int_distribution<size_t> A(0, archive_.size() - 1);
        size_t idx = A(rng_);
        archive_.erase(archive_.begin() + (long)idx);
    }
}

int PolyphaseDE::archivePickIndex_() {
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> A(0, (int)archive_.size() - 1);
    return A(rng_);
}

// --- strategy selection & adaptation ---

int PolyphaseDE::sampleStrategy_() {
    // roulette wheel on p_strat_
    double s = 0.0;
    for (double p : p_strat_) s += p;
    if (s <= 0.0) {
        std::uniform_int_distribution<int> U(0, NST_ - 1);
        return U(rng_);
    }
    double r = U01_(rng_) * s;
    double c = 0.0;
    for (int k = 0; k < NST_; ++k) {
        c += p_strat_[k];
        if (r <= c) return k;
    }
    return NST_ - 1;
}

void PolyphaseDE::sampleFCR_(int strat, double& F, double& CR) {
    // Cauchy sampling for F, Normal sampling for CR around muF/muCR.
    static const double PI = 3.14159265358979323846;
    double u = U01_(rng_);
    double t = std::tan(PI * (u - 0.5));
    F = muF_[strat] + 0.1 * t;
    if (!std::isfinite(F) || F <= 0.0) F = muF_[strat];
    F = clamp_(F, F_lo_[strat], F_hi_[strat]);

    std::normal_distribution<double> N0(0.0, 1.0);
    CR = muCR_[strat] + 0.1 * N0(rng_);
    if (!std::isfinite(CR)) CR = muCR_[strat];
    CR = clamp_(CR, CR_lo_[strat], CR_hi_[strat]);
}

void PolyphaseDE::updateStrategyStats_(int strat, double parent_f, double child_f) {
    if (strat < 0 || strat >= NST_) return;
    trial_strat_[strat] += 1.0;
    if (child_f < parent_f - 1e-12) {
        succ_strat_[strat] += (parent_f - child_f);
    }
}

// --- pick helpers ---

int PolyphaseDE::pickDistinct_(int n, int a, int b, int c, int d, int e) {
    std::uniform_int_distribution<int> I(0, n - 1);
    int r;
    do {
        r = I(rng_);
    } while (r == a || r == b || r == c || r == d || r == e);
    return r;
}

int PolyphaseDE::pickPbestIndex_(double pfrac) {
    int N = population();
    if (N <= 0) return 0;

    int P = std::max(1, (int)std::round(pfrac * (double)N));
    if (P > N) P = N;

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + P, idx.end(),
        [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> U(0, P - 1);
    return idx[U(rng_)];
}

// --- Strategy 0: rand/1/bin (exploration) ---

void PolyphaseDE::strat0_rand1_bin_(int i, Vec& trial, double F, double CR) {
    const int D = dim_;
    const int N = population();

    int r1 = pickDistinct_(N, i);
    int r2 = pickDistinct_(N, i, r1);
    int r3 = pickDistinct_(N, i, r1, r2);

    const Vec& x1 = X_[r1];
    const Vec& x2 = X_[r2];
    const Vec& x3 = X_[r3];

    trial = X_[i];
    std::uniform_int_distribution<int> J(0, D - 1);
    int jrand = J(rng_);
    for (int j = 0; j < D; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            trial[j] = x1[j] + F * (x2[j] - x3[j]);
        }
    }
    ensureBounds(trial);
}

// --- Strategy 1: current-to-pbest/1/bin (balanced) ---

void PolyphaseDE::strat1_current_to_pbest1_bin_(int i, Vec& trial, double F, double CR) {
    const int D = dim_;
    const int N = population();

    int pbest = pickPbestIndex_(0.20); // top 20%
    int r1    = pickDistinct_(N, i, pbest);
    int r2    = pickDistinct_(N, i, pbest, r1);

    const Vec& xi  = X_[i];
    const Vec& xp  = X_[pbest];
    const Vec& xr1 = X_[r1];
    const Vec& xr2 = X_[r2];

    trial = xi;
    std::uniform_int_distribution<int> J(0, D - 1);
    int jrand = J(rng_);
    for (int j = 0; j < D; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = xi[j]
                     + F * (xp[j]  - xi[j])
                     + F * (xr1[j] - xr2[j]);
            trial[j] = v;
        }
    }
    ensureBounds(trial);
}

// --- Strategy 2: pbest/2/bin (aggressive exploitation) ---

void PolyphaseDE::strat2_pbest2_bin_(int i, Vec& trial, double F, double CR) {
    const int D = dim_;
    const int N = population();

    int pbest = pickPbestIndex_(0.10); // top 10%
    int r1    = pickDistinct_(N, i, pbest);
    int r2    = pickDistinct_(N, i, pbest, r1);
    int r3    = pickDistinct_(N, i, pbest, r1, r2);
    int r4    = pickDistinct_(N, i, pbest, r1, r2, r3);

    const Vec& xp  = X_[pbest];
    const Vec& xr1 = X_[r1];
    const Vec& xr2 = X_[r2];
    const Vec& xr3 = X_[r3];
    const Vec& xr4 = X_[r4];

    trial = X_[i];
    std::uniform_int_distribution<int> J(0, D - 1);
    int jrand = J(rng_);
    for (int j = 0; j < D; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = xp[j]
                     + F * (xr1[j] - xr2[j])
                     + F * (xr3[j] - xr4[j]);
            trial[j] = v;
        }
    }
    ensureBounds(trial);
}

// --- micro-restart around best ---

void PolyphaseDE::microRestart_(bool strong) {
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N <= 2) return;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int   D = dim_;

    // find best
    int    best_i = 0;
    double bf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
        double f = FX_[i];
        if (std::isfinite(f) && f < bf) {
            bf     = f;
            best_i = i;
        }
    }
    if (!std::isfinite(bf)) return;
    Vec best = X_[best_i];

    // sort by fitness to restart worst
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    double frac = strong ? restart_frac_strong_ : restart_frac_soft_;
    double sig  = strong ? restart_sigma_ * 1.8 : restart_sigma_;
    if (frac <= 0.0) return;

    int nreset = std::max(1, (int)std::round(frac * (double)N));
    if (nreset >= N) nreset = N - 1;

    std::normal_distribution<double> N0(0.0, 1.0);
    for (int k = 0; k < nreset; ++k) {
        int i = idx[N - 1 - k];   // from worst up
        if (i == best_i) continue;

        Vec cand(D);
        for (int j = 0; j < D; ++j) {
            double step = sig * (U[j] - L[j]) * N0(rng_);
            double v    = best[j] + step;
            cand[j]     = clamp_(v, L[j], U[j]);
        }
        ensureBounds(cand);
        double f = eval(cand);

        X_[i]  = std::move(cand);
        FX_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }
}

// --- final local search ---

void PolyphaseDE::finalLocalSearch_() {
    if (!end_local_enable_) return;
    if (end_local_method_.empty()) return;
    if (!prob_) return;

    int N = (int)X_.size();
    if (N <= 0) return;

    // sort by fitness
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int top_k = std::min(4, N);
    for (int t = 0; t < top_k; ++t) {
        int i = idx[t];
        if (!std::isfinite(FX_[i])) continue;

        auto [xloc, floc] = localSearch(end_local_method_, X_[i]);
        if (floc < FX_[i]) {
            X_[i]  = xloc;
            FX_[i] = floc;
        }
        if (floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }
        if (prob_->calls() >= max_evals_) break;
    }
}

// --- configure ---

void PolyphaseDE::configure(const MethodConfig& mc) {
    // base population
    int p = mc.getInt("population",
            mc.getInt("pop",
            mc.getInt("Pop", pop_)));
    if (p < 4) p = pop_;
    pop_ = p;
    this->setPopulation(pop_);

    // probabilities init
    p_strat_.assign(NST_, 1.0 / (double)NST_);
    succ_strat_.assign(NST_, 0.0);
    trial_strat_.assign(NST_, 0.0);

    // adaptation speed
    c_adapt_ = mc.getDbl("c_adapt", c_adapt_);

    // phase boundaries
    phase_explore_end_ = mc.getDbl("phase_explore_end", phase_explore_end_);
    phase_balance_end_ = mc.getDbl("phase_balance_end", phase_balance_end_);

    // micro-restart & diversity thresholds
    stagn_trigger_soft_   = mc.getInt("stagn_soft",   stagn_trigger_soft_);
    stagn_trigger_strong_ = mc.getInt("stagn_strong", stagn_trigger_strong_);
    restart_frac_soft_    = mc.getDbl("restart_frac_soft",  restart_frac_soft_);
    restart_frac_strong_  = mc.getDbl("restart_frac_strong", restart_frac_strong_);
    restart_sigma_        = mc.getDbl("restart_sigma",       restart_sigma_);

    div_low_  = mc.getDbl("div_low",  div_low_);
    div_high_ = mc.getDbl("div_high", div_high_);

    // archive
    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);

    // end-game local search
    end_local_enable_ = mc.getBool("end_local_enable", end_local_enable_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);

    // in-run local base probability
    inrun_local_base_ = mc.getDbl("inrun_local_base", inrun_local_base_);

    // RNG seed
    uint64_t seed = (uint64_t)mc.getInt("seed", 0);
    if (seed == 0) {
        uint64_t now  = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed = splitmix64_polyde(now ^ 0x1234abcd9876ULL);
    }
    rng_.seed(seed);
}

// --- init ---

void PolyphaseDE::init() {
    if (!prob_) return;

    dim_ = prob_->dimension();
    int N = std::max(4, pop_);
    this->setPopulation(N);

    X_.clear();
    FX_.clear();
    archive_.clear();

    X_.resize(N, Vec(dim_, 0.0));
    FX_.assign(N, std::numeric_limits<double>::infinity());

    best_x_.assign(dim_, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    // Samples the initial population via Initializer.
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    for (int i = 0; i < N; ++i) {
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    stagn_iters_ = 0;
    iter_        = 0;

    printBest();
    updateStop(FX_);
}

// --- main iteration ---

void PolyphaseDE::one_iteration() {
    if (!prob_) return;
    int N = (int)X_.size();
    if (N <= 0) {
        updateStop(FX_);
        return;
    }
    iter_++;

    // sort indices (ascending fitness)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int best_i = idx[0];
    if (std::isfinite(FX_[best_i]) && FX_[best_i] < best_f_) {
        best_f_ = FX_[best_i];
        best_x_ = X_[best_i];
    }
    double prev_best = best_f_;

    // reset window stats
    std::fill(succ_strat_.begin(),  succ_strat_.end(),  0.0);
    std::fill(trial_strat_.begin(), trial_strat_.end(), 0.0);

    double phase = progress01_();

    // In-run local rate scales with the phase (stronger toward the end).
    double inrun_rate = inrun_local_base_ * (0.2 + 0.8 * phase);
    if (inrun_rate > 1.0) inrun_rate = 1.0;

    // main loop on all individuals
    for (int t = 0; t < N; ++t) {
        int i = idx[t];

        Vec xi = X_[i];
        ensureBounds(xi);
        double fi = FX_[i];
        if (!std::isfinite(fi)) {
            fi = eval(xi);
            FX_[i] = fi;
        }

        // Sample strategy
        int strat = sampleStrategy_();

        // Phase-aware bias: early -> strat0, late -> strat2
        if (phase < phase_explore_end_) {
            if (U01_(rng_) < 0.30) strat = 0;
        } else if (phase > phase_balance_end_) {
            if (U01_(rng_) < 0.35) strat = 2;
        }

        double F, CR;
        sampleFCR_(strat, F, CR);

        Vec trial(dim_, 0.0);
        switch (strat) {
            case 0: strat0_rand1_bin_(i, trial, F, CR); break;
            case 1: strat1_current_to_pbest1_bin_(i, trial, F, CR); break;
            case 2: strat2_pbest2_bin_(i, trial, F, CR); break;
            default:strat1_current_to_pbest1_bin_(i, trial, F, CR); break;
        }

        double ft = eval(trial);

        // selection
        if (ft < fi) {
            archivePush_(xi);
            X_[i]  = trial;
            FX_[i] = ft;

            // in-run local search (late phase, optional)
            if (inrun_rate > 0.0 && !end_local_method_.empty() && phase > 0.6) {
                if (U01_(rng_) < inrun_rate) {
                    auto [xloc, floc] = localSearch(end_local_method_, X_[i]);
                    if (floc < FX_[i]) {
                        X_[i]  = xloc;
                        FX_[i] = floc;
                        ft     = floc;
                    }
                }
            }

            if (ft < best_f_) {
                best_f_ = ft;
                best_x_ = X_[i];
            }
        }

        updateStrategyStats_(strat, fi, FX_[i]);

        if (prob_->calls() >= max_evals_) break;
    }

    // update success-history of strategies
    double total_succ   = 0.0;
    double total_trials = 0.0;
    for (int k = 0; k < NST_; ++k) {
        total_succ   += succ_strat_[k];
        total_trials += trial_strat_[k];
    }

    if (total_trials > 0.0) {
        // Updates muF/muCR and p_strat_.
        for (int k = 0; k < NST_; ++k) {
            if (trial_strat_[k] <= 0.0) continue;
            double wk = succ_strat_[k] / (total_succ + 1e-12);
            double new_muF  = muF_[k];
            double new_muCR = muCR_[k];

            if (succ_strat_[k] > 0.0) {
                new_muF  = clamp_(muF_[k]  + c_adapt_ * wk * (0.8 - muF_[k]),  F_lo_[k],  F_hi_[k]);
                new_muCR = clamp_(muCR_[k] + c_adapt_ * wk * (0.9 - muCR_[k]), CR_lo_[k], CR_hi_[k]);
            }

            muF_[k]  = new_muF;
            muCR_[k] = new_muCR;
        }

        std::vector<double> q(NST_, 0.0);
        double sumq = 0.0;
        for (int k = 0; k < NST_; ++k) {
            double eff = succ_strat_[k] / (trial_strat_[k] + 1e-12);
            q[k] = eff;
            sumq += eff;
        }
        if (sumq <= 0.0) {
            for (int k = 0; k < NST_; ++k) q[k] = 1.0;
            sumq = (double)NST_;
        }

        for (int k = 0; k < NST_; ++k) {
            double pk_new = q[k] / sumq;
            p_strat_[k] = (1.0 - c_adapt_) * p_strat_[k] + c_adapt_ * pk_new;
        }
    }

    // stagnation detection
    if (best_f_ < prev_best - 1e-12) {
        stagn_iters_ = 0;
    } else {
        stagn_iters_++;
    }

    // micro-restarts
    if (stagn_iters_ >= stagn_trigger_soft_ && stagn_iters_ < stagn_trigger_strong_) {
        microRestart_(false);
    } else if (stagn_iters_ >= stagn_trigger_strong_) {
        microRestart_(true);
        stagn_iters_ = 0;
    }

    // If diversity is very low (except at very late phase), performs a mild restart.
    double div = diversityBestNorm_();
    if (div < div_low_ && phase < 0.85) {
        microRestart_(false);
    }

    printBest();
    updateStop(FX_);
}

// --- end() ---

void PolyphaseDE::end() {
    finalLocalSearch_();

    // Ensures best_x_/best_f_ are in the population for consistency.
    if (!X_.empty() && std::isfinite(best_f_)) {
        size_t worst = 0;
        double fw    = FX_[0];
        for (size_t i = 1; i < FX_.size(); ++i) {
            if (FX_[i] > fw) {
                fw    = FX_[i];
                worst = i;
            }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
