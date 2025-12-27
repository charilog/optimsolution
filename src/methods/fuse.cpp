#include "fuse.h"
#include "init.h"
#include "options.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <random>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>

namespace optimsolution {

// -------------------------------------------------------------
// configure
// -------------------------------------------------------------
void FUSE::configure(const MethodConfig &mc)
{
    int p = -1;
    p = mc.getInt("population",
        mc.getInt("Population",
        mc.getInt("pop",
        mc.getInt("Pop", -1))));
    if (p < 0) p = mc.getInt("NP", -1);
    if (p >= 5) {
        pop_override_ = p;
        setPopulation(pop_override_);
    }

    muF_   = mc.getDbl("muF_init",  muF_);
    muCR_  = mc.getDbl("muCR_init", muCR_);
    sh_c_  = mc.getDbl("sh_c",      sh_c_);

    F_lo_  = mc.getDbl("F_lo",  F_lo_);
    F_hi_  = mc.getDbl("F_hi",  F_hi_);
    CR_lo_ = mc.getDbl("CR_lo", CR_lo_);
    CR_hi_ = mc.getDbl("CR_hi", CR_hi_);

    pbest_frac_      = clamp_(mc.getDbl("pbest_frac", pbest_frac_), 0.05, 0.9);
    archive_rate_    = mc.getDbl("archive_rate",      archive_rate_);

    stagnation_trigger_ = mc.getInt("stagnation_trigger", stagnation_trigger_);
    restart_frac_       = mc.getDbl("restart_frac",       restart_frac_);
    restart_sigma_      = mc.getDbl("restart_sigma",      restart_sigma_);
}

// -------------------------------------------------------------
// helpers
// -------------------------------------------------------------
void FUSE::ensureBounds(Vec &v)
{
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j) {
        if (!std::isfinite(v[j])) v[j] = 0.5 * (L[j] + U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

// RNG helpers (not const)
int FUSE::pickDistinct_(int n, int a, int b, int c, int d)
{
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> U(0, n-1);
    int r;
    do {
        r = U(rng_);
    } while ((a >= 0 && r == a) ||
             (b >= 0 && r == b) ||
             (c >= 0 && r == c) ||
             (d >= 0 && r == d));
    return r;
}

int FUSE::pickPbestIndex_(const std::vector<int> &sorted_idx)
{
    const int N = (int)sorted_idx.size();
    if (N <= 0) return 0;

    int pmax = std::max(1, (int)std::round(pbest_frac_ * N));
    if (pmax > N) pmax = N;

    std::uniform_int_distribution<int> U(0, pmax-1);
    int pos = U(rng_);
    return sorted_idx[pos];
}

void FUSE::pushArchive_(const Vec &x)
{
    if (archive_rate_ <= 0.0) return;
    const int N = (int)X_.size();
    int maxArch = (int)std::round(archive_rate_ * (double)N);
    if (maxArch <= 0) return;

    if ((int)archive_.size() < maxArch) {
        archive_.push_back(x);
    } else {
        std::uniform_int_distribution<int> U(0, maxArch-1);
        int pos = U(rng_);
        archive_[pos] = x;
    }
}

void FUSE::microRestart_()
{
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N <= 1) return;

    const int D = prob_->dimension();
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int elite = idx[0];
    std::normal_distribution<double> N0(0.0, 1.0);

    double eff_frac = restart_frac_;
    if (eff_frac > 0.9) eff_frac = 0.9;
    int nreset = std::max(1, (int)std::round(eff_frac * (double)N));
    if (nreset > N-1) nreset = N-1;

    for (int k = 0; k < nreset; ++k) {
        int i = idx[N-1-k];
        if (i == elite) continue;
        Vec cand(D);
        for (int j = 0; j < D; ++j) {
            double step = restart_sigma_ * (U[j] - L[j]) * N0(rng_);
            cand[j] = clamp_(best_x_[j] + step, L[j], U[j]);
        }
        ensureBounds(cand);
        double f = eval(cand);
        X_[i] = std::move(cand);
        FX_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }
}

// Operator selection: 0 = LSHADE, 1 = BEST2
int FUSE::selectOperator_(double eval_ratio, double rank_q)
{
    double p_best2;

    if (eval_ratio < 0.3) {
        p_best2 = 0.10;
    } else if (eval_ratio < 0.7) {
        p_best2 = 0.25;
    } else {
        p_best2 = 0.50;
    }

    if (rank_q < 0.2) {      // elite
        p_best2 += 0.20;
    } else if (rank_q > 0.7) { // tail
        p_best2 -= 0.10;
    }

    if (p_best2 < 0.0) p_best2 = 0.0;
    if (p_best2 > 0.80) p_best2 = 0.80;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    if (r < p_best2) return 1;
    return 0;
}

// -------------------------------------------------------------
// Operators
// -------------------------------------------------------------

// LSHADE – current-to-pbest/1/bin + archive
void FUSE::opLSHADE_(int i,
                     const std::vector<int> &sorted_idx,
                     double F, double CR,
                     Vec &trial)
{
    const int N = (int)X_.size();
    const int D = prob_->dimension();

    const Vec &xi = X_[i];

    int pbest = pickPbestIndex_(sorted_idx);
    int r1    = pickDistinct_(N, i, pbest);

    std::uniform_real_distribution<double> U01(0.0,1.0);
    bool useArch = (!archive_.empty() && U01(rng_) < 0.5);

    Vec x_r2;
    if (useArch) {
        std::uniform_int_distribution<int> Ua(0, (int)archive_.size()-1);
        int ia = Ua(rng_);
        x_r2   = archive_[ia];
    } else {
        int r2 = pickDistinct_(N, i, pbest, r1);
        x_r2   = X_[r2];
    }

    Vec donor(D);
    for (int j = 0; j < D; ++j) {
        donor[j] = xi[j]
                 + F * (X_[pbest][j] - xi[j])
                 + F * (X_[r1][j]     - x_r2[j]);
    }

    std::uniform_int_distribution<int> J(0, D-1);
    int jrand = J(rng_);

    trial = xi;
    for (int j = 0; j < D; ++j) {
        if (U01(rng_) < CR || j == jrand) {
            trial[j] = donor[j];
        }
    }
    ensureBounds(trial);
}

// BEST2 – DE/best/2/bin
void FUSE::opBEST2_(int i, double F, double CR, Vec &trial)
{
    const int N = (int)X_.size();
    const int D = prob_->dimension();

    const Vec &xi = X_[i];
    const Vec &gb = best_x_;

    int r1 = pickDistinct_(N, i);
    int r2 = pickDistinct_(N, i, r1);
    int r3 = pickDistinct_(N, i, r1, r2);
    int r4 = pickDistinct_(N, i, r1, r2, r3);

    const Vec &x1 = X_[r1];
    const Vec &x2 = X_[r2];
    const Vec &x3 = X_[r3];
    const Vec &x4 = X_[r4];

    Vec donor(D);
    for (int j = 0; j < D; ++j) {
        donor[j] = gb[j]
                 + F * (x1[j] - x2[j])
                 + F * (x3[j] - x4[j]);
    }

    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::uniform_int_distribution<int> J(0, D-1);
    int jrand = J(rng_);

    trial = xi;
    for (int j = 0; j < D; ++j) {
        if (U01(rng_) < CR || j == jrand) {
            trial[j] = donor[j];
        }
    }
    ensureBounds(trial);
}

// -------------------------------------------------------------
// init
// -------------------------------------------------------------
void FUSE::init()
{
    if (!prob_) return;

    int N = std::max(5, (pop_override_ >= 5 ? pop_override_ : population()));
    setPopulation(N);

    stagn_iters_ = 0;

    const int D = prob_->dimension();
    X_.clear();
    FX_.clear();
    archive_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < N; ++i) {
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

// -------------------------------------------------------------
// one_iteration
// -------------------------------------------------------------
void FUSE::one_iteration()
{
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (D <= 0 || N <= 0) {
        updateStop(FX_);
        return;
    }

    double eval_ratio = 0.0;
    if (max_evals_ > 0) {
        eval_ratio = (double)prob_->calls() / (double)max_evals_;
        eval_ratio = clamp_(eval_ratio, 0.0, 1.0);
    }

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int elite = idx[0];
    double prevBest = best_f_;
    if (FX_[elite] < best_f_) {
        best_f_ = FX_[elite];
        best_x_ = X_[elite];
    }

    // LSHADE accumulators
    std::vector<double> sF;
    std::vector<double> sCR;
    std::vector<double> wGain;
    sF.reserve(N);
    sCR.reserve(N);
    wGain.reserve(N);

    std::normal_distribution<double> N0(0.0, 1.0);
    const double PI = 3.14159265358979323846;

    for (int pos = 0; pos < N; ++pos) {
        int i  = idx[pos];
        Vec &xi = X_[i];
        double fx = FX_[i];

        // F (Cauchy-like)
        double F;
        {
            std::uniform_real_distribution<double> U01(0.0,1.0);
            do {
                double r = U01(rng_);
                F = muF_ + 0.1 * std::tan(PI*(r - 0.5));
            } while (F <= 0.0);
        }
        if (F > 1.5) F = 1.5;
        F = clamp_(F, F_lo_, F_hi_);

        // CR (normal around muCR)
        double CR = muCR_ + 0.1 * N0(rng_);
        if (!std::isfinite(CR)) CR = muCR_;
        CR = clamp_(CR, CR_lo_, CR_hi_);

        double rank_q = (N > 1) ? (double)pos / (double)(N-1) : 0.0;
        int op_id = selectOperator_(eval_ratio, rank_q);

        Vec trial;
        if (op_id == 1) {
            opBEST2_(i, F, CR, trial);
        } else {
            opLSHADE_(i, idx, F, CR, trial);
        }

        double fT = eval(trial);

        bool replaced = false;
        double parent_f = fx;

        if (fT <= fx) {
            pushArchive_(xi);
            xi       = trial;
            FX_[i]   = fT;
            replaced = true;
        }

        if (replaced) {
            double new_f = FX_[i];
            if (new_f < best_f_) {
                best_f_ = new_f;
                best_x_ = xi;
            }

            double denom  = std::fabs(parent_f) + 1e-12;
            double reward = (parent_f - new_f) / denom;
            if (reward < 0.0) reward = 0.0;

            sF.push_back(F);
            sCR.push_back(CR);
            wGain.push_back(reward);
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // LSHADE-style update for muF_, muCR_
    if (!sF.empty()) {
        double sumw = 0.0;
        for (double w : wGain) sumw += w;
        if (sumw > 0.0) {
            double wmCR = 0.0, wmF = 0.0, wmF2 = 0.0;
            for (size_t k = 0; k < sF.size(); ++k) {
                double wk = wGain[k] / sumw;
                wmCR += wk * sCR[k];
                wmF  += wk * sF[k];
                wmF2 += wk * sF[k] * sF[k];
            }
            double LmeanF = (wmF > 1e-12) ? (wmF2 / wmF) : wmF;
            muCR_ = (1.0 - sh_c_) * muCR_ + sh_c_ * clamp_(wmCR, CR_lo_, CR_hi_);
            muF_  = (1.0 - sh_c_) * muF_  + sh_c_ * clamp_(LmeanF, F_lo_, F_hi_);
        }
    }

    // stagnation
    if (best_f_ < prevBest - 1e-12) {
        stagn_iters_ = 0;
    } else {
        stagn_iters_++;
    }

    double eval_ratio_now = 0.0;
    if (max_evals_ > 0) {
        eval_ratio_now = (double)prob_->calls() / (double)max_evals_;
        eval_ratio_now = clamp_(eval_ratio_now, 0.0, 1.0);
    }

    if (stagn_iters_ >= stagnation_trigger_ && eval_ratio_now < 0.90) {
        microRestart_();
        stagn_iters_ = 0;
    }

    printBest();
    updateStop(FX_);
}

// -------------------------------------------------------------
// end - without local search
// -------------------------------------------------------------
void FUSE::end()
{
    if (!prob_) return;

    if (!X_.empty() && !FX_.empty()) {
        int ei = 0;
        double bf = std::numeric_limits<double>::infinity();
        for (int i = 0; i < (int)FX_.size(); ++i) {
            double f = FX_[i];
            if (std::isfinite(f) && f < bf) {
                bf = f;
                ei = i;
            }
        }
        if (bf < best_f_) {
            best_f_ = bf;
            best_x_ = X_[ei];
        }

        // copy the best into the worst position for consistency
        size_t worst = 0;
        double fw = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > fw) {
                fw    = FX_[k];
                worst = k;
            }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
