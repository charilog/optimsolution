// garq.cpp
#include "garq.h"
#include "options.h"
#include "localsearch.h"

#include <chrono>
#include <cctype>
#include <iostream>
#include <cstdint>
#include <random>
#include <cmath>

namespace optimsolution {

// A small mixer is used as in ARQ.
static inline uint64_t splitmix64(uint64_t z){
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return z;
}

// ---------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------

double GARQ::eval(const Vec& x)
{
    if (!prob_) return std::numeric_limits<double>::infinity();
    return prob_->evaluate(x);
}

void GARQ::ensureBounds(Vec& x)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) {
            x[j] = 0.5 * (L[j] + U[j]);
        }
        if (x[j] < L[j])      x[j] = L[j];
        else if (x[j] > U[j]) x[j] = U[j];
    }
}

double GARQ::bnDistance(const Vec& a, const Vec& b) const
{
    if (!prob_) return 0.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const double eps = 1e-12;
    double s = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        double r = U[j] - L[j];
        double d = (a[j] - b[j]) / (r + eps);
        s += d * d;
    }
    return std::sqrt(s);
}

double GARQ::computeDiversity() const
{
    if (!prob_ || X_.empty()) return 0.0;

    // If the best is well-defined, the mean distance from the best is measured.
    if (std::isfinite(best_f_) && !best_x_.empty()) {
        double sum = 0.0;
        int    cnt = 0;
        for (std::size_t i = 0; i < X_.size(); ++i) {
            double fi = FX_[i];
            if (!std::isfinite(fi)) continue;
            sum += bnDistance(X_[i], best_x_);
            cnt++;
        }
        if (cnt == 0) return 0.0;
        return sum / (double)cnt;
    }

    // Fallback: mean pairwise distance.
    double sum = 0.0;
    int    cnt = 0;
    for (std::size_t i = 0; i < X_.size(); ++i) {
        for (std::size_t j = i + 1; j < X_.size(); ++j) {
            double fi = FX_[i], fj = FX_[j];
            if (!std::isfinite(fi) || !std::isfinite(fj)) continue;
            sum += bnDistance(X_[i], X_[j]);
            cnt++;
        }
    }
    if (cnt == 0) return 0.0;
    return sum / (double)cnt;
}

// ---------------------------------------------------------------
// configure
// ---------------------------------------------------------------

void GARQ::configure(const MethodConfig& mc)
{
    // population
    pop_init_ = mc.getInt("population", pop_init_);
    if (pop_init_ <= 0) {
        pop_init_ = population();
    }
    if (pop_init_ < 4) pop_init_ = 50;
    setPopulation(pop_init_);

    pop_min_ = mc.getInt("pop_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    // LSHADE memory
    H_     = mc.getInt("H", H_);
    if (H_ <= 0) H_ = 10;
    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    pmin_  = mc.getDbl("pmin", pmin_);
    pmax_  = mc.getDbl("pmax", pmax_);
    if (pmin_ <= 0.0 || pmin_ > 1.0) pmin_ = 0.05;
    if (pmax_ <= 0.0 || pmax_ > 1.0) pmax_ = 0.30;
    if (pmin_ > pmax_) std::swap(pmin_, pmax_);

    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);
    if (archive_rate_ < 0.0) archive_rate_ = 0.0;

    // dual-zone
    zone_radius_ = mc.getDbl("zone_radius", zone_radius_);
    if (zone_radius_ <= 0.0) zone_radius_ = 0.25;
    if (zone_radius_ > 1.0)  zone_radius_ = 1.0;

    rl_alpha_ = mc.getDbl("rl_alpha", rl_alpha_);
    if (rl_alpha_ <= 0.0 || rl_alpha_ >= 1.0) rl_alpha_ = 0.35;

    min_weight_ = mc.getDbl("min_weight", min_weight_);
    if (min_weight_ < 0.0) min_weight_ = 0.05;

    // diversity thresholds
    div_low_  = mc.getDbl("div_low",  div_low_);
    div_high_ = mc.getDbl("div_high", div_high_);
    if (div_low_ <= 0.0) div_low_ = 0.08;
    if (div_high_ <= div_low_) div_high_ = std::max(0.2, 3.0 * div_low_);

    // restart / stagnation
    restart_frac_ = mc.getDbl("restart_frac", restart_frac_);
    if (restart_frac_ < 0.0) restart_frac_ = 0.0;
    if (restart_frac_ > 1.0 && restart_frac_ <= 100.0) {
        restart_frac_ /= 100.0;
    }
    if (restart_frac_ > 0.5) restart_frac_ = 0.5;

    restart_sigma_ = mc.getDbl("restart_sigma", restart_sigma_);
    if (restart_sigma_ <= 0.0) restart_sigma_ = 0.35;

    stagnation_window_ = mc.getInt("stagnation_window", stagnation_window_);
    if (stagnation_window_ <= 0) stagnation_window_ = 25;

    // quarantine (ARQ-style)
    quarantine_rate_ = mc.getDbl("quarantine_rate", quarantine_rate_);
    if (quarantine_rate_ < 0.0) quarantine_rate_ = 0.0;
    if (quarantine_rate_ > 1.0) quarantine_rate_ = 1.0;

    quarantine_sigma_ = mc.getDbl("quarantine_sigma", quarantine_sigma_);
    if (quarantine_sigma_ <= 0.0) quarantine_sigma_ = 0.20;

    outlier_alpha_ = mc.getDbl("outlier_alpha", outlier_alpha_);
    if (outlier_alpha_ <= 0.0) outlier_alpha_ = 1.7;

    relocate_rate_ = mc.getDbl("relocate_rate", relocate_rate_);
    if (relocate_rate_ < 0.0) relocate_rate_ = 0.25;

    // local search
    {
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local", local_method_)));
        for (auto& c : lm) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lm == "none" || lm == "off" || lm == "0") {
            local_method_.clear();
        } else {
            local_method_ = lm;
        }

        double lr = mc.getDbl("local_rate",
                    mc.getDbl("local.rate",
                    mc.getDbl("inrun_rate", local_rate_)));
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

    end_local_refine_ = mc.getBool("end_local_refine", false);
    end_local_method_ = mc.getStr("end_local_method",
                          mc.getStr("final_local_method", "lbfgs"));

    // RL weights initialization
    for (int z = 0; z < NUM_ZONES_; ++z) {
        for (int s = 0; s < NUM_STRAT_; ++s) {
            strat_weight_[z][s] = 1.0;
            strat_reward_[z][s] = 0.0;
            strat_uses_[z][s]   = 0;
        }
    }
    // Zone 0: far from the best -> very aggressive exploration
    strat_weight_[0][0] = 2.0; // rand/1
    strat_weight_[0][1] = 0.8; // pbest
    strat_weight_[0][2] = 0.4; // best/1
    strat_weight_[0][3] = 0.6; // gaussian
    strat_weight_[0][4] = 2.2; // longjump

    // Zone 1: near the best -> full exploitation
    strat_weight_[1][0] = 0.5;
    strat_weight_[1][1] = 2.0;
    strat_weight_[1][2] = 2.0;
    strat_weight_[1][3] = 2.0;
    strat_weight_[1][4] = 0.4;

    diversity_ema_          = 0.0;
    diversity_initialized_  = false;
    stagnation_counter_     = 0;
    last_best_              = std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------
// init
// ---------------------------------------------------------------

void GARQ::init()
{
    if (!prob_) return;

    D_ = prob_->dimension();
    if (D_ <= 0) return;

    if (pop_init_ <= 0) {
        pop_init_ = population();
        if (pop_init_ < 4) pop_init_ = 50;
        setPopulation(pop_init_);
    }
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    X_.assign(pop_init_, Vec(D_, 0.0));
    FX_.assign(pop_init_, std::numeric_limits<double>::infinity());
    archive_.clear();

    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    // RNG seed (stable and high-entropy)
    uint64_t now = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t mix = splitmix64(now ^ (uint64_t)(uintptr_t)this);
    rng_.seed(mix);

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D_, 0.0);
    last_best_ = best_f_;
    stagnation_counter_ = 0;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < pop_init_; ++i) {
        for (int j = 0; j < D_; ++j) {
            double r = U01_(rng_);
            X_[i][j] = L[j] + r * (U[j] - L[j]);
        }
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    diversity_initialized_ = false;
    diversity_ema_         = 0.0;

    updateStop(FX_);
    printBest();
}

// ---------------------------------------------------------------
// LSHADE F/CR memory
// ---------------------------------------------------------------

void GARQ::sample_F_CR(double& F, double& CR)
{
    if (H_ <= 0 || MF_.empty() || MCR_.empty()) {
        F  = 0.5;
        CR = 0.8;
        return;
    }

    std::uniform_int_distribution<int> Ui_mem(0, std::max(0, H_ - 1));
    int    rMem = Ui_mem(rng_);
    double mF   = MF_[rMem];
    double mCR  = MCR_[rMem];

    std::cauchy_distribution<double> cauchy(mF, 0.1);
    do {
        F = cauchy(rng_);
    } while (F <= 0.0);
    if (F > 1.0) F = 1.0;
    const double F_MIN = 0.20; // Aggressive
    if (F < F_MIN) F = F_MIN;

    std::normal_distribution<double> normal(mCR, 0.1);
    CR = normal(rng_);
    if (CR < 0.0) CR = 0.0;
    if (CR > 1.0) CR = 1.0;
}

void GARQ::updateMemory(const std::vector<double>& succF,
                        const std::vector<double>& succCR)
{
    if (succF.empty()) return;
    if ((int)MF_.size() != H_) {
        MF_.assign(H_, 0.5);
        MCR_.assign(H_, 0.8);
        mem_idx_ = 0;
    }

    double numF = 0.0, denF = 0.0;
    double numCR = 0.0, denCR = 0.0;

    for (std::size_t i = 0; i < succF.size(); ++i) {
        double w = succF[i] * succF[i];
        numF  += w * succF[i];
        denF  += w;

        numCR += w * succCR[i];
        denCR += w;
    }

    if (denF > 0.0) {
        MF_[mem_idx_] = (1.0 - c_mem_) * MF_[mem_idx_] + c_mem_ * (numF / denF);
    }
    if (denCR > 0.0) {
        MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * (numCR / denCR);
    }

    mem_idx_++;
    if (mem_idx_ >= H_) mem_idx_ = 0;
}

// ---------------------------------------------------------------
// RL controller
// ---------------------------------------------------------------

int GARQ::determineZone(const Vec& x) const
{
    if (!prob_ || !std::isfinite(best_f_)) return 0;
    double d = bnDistance(x, best_x_);
    return (d <= zone_radius_) ? 1 : 0;
}

int GARQ::selectStrategy(int zone, double progress, double diversity)
{
    if (zone < 0 || zone >= NUM_ZONES_) zone = 0;
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;

    // Computation of the "exploration level"
    // - At the beginning (progress~0) -> large exploration
    // - If diversity < div_low_ -> very large exploration
    // - If diversity > div_high_ -> small exploration
    double explore_early = 1.0 - progress;  // 1 at the beginning, 0 at the end
    double explore_div   = 0.0;
    if (diversity <= 0.0) {
        explore_div = 1.0;
    } else if (diversity < div_low_) {
        explore_div = 1.0;
    } else if (diversity >= div_high_) {
        explore_div = 0.0;
    } else {
        double t = (div_high_ - diversity) / (div_high_ - div_low_);
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        explore_div = t;
    }

    double explore_bias = 0.5 * explore_early + 0.5 * explore_div;

    // [Comment translated to English]
    // [Comment translated to English]
    if (explore_bias < 0.05) explore_bias = 0.05;
    if (explore_bias > 1.0)  explore_bias = 1.0;

    // Strategy groups
    // Exploratory: rand/1, longjump
    // Exploit: pbest, best/1, gaussian
    const int explore_group[2] = {0, 4};
    const int exploit_group[3] = {1, 2, 3};

    auto sampleFromGroup = [&](const int* group, int gsize)->int{
        double wsum = 0.0;
        double wloc[NUM_STRAT_] = {0.0,0.0,0.0,0.0,0.0};

        for (int k = 0; k < gsize; ++k) {
            int s = group[k];
            double base = strat_weight_[zone][s];
            if (base < min_weight_) base = min_weight_;
            wloc[s] = base;
            wsum += base;
        }
        if (wsum <= 0.0) {
            // Fallback: the first of the group
            int s = group[0];
            strat_uses_[zone][s] += 1;
            return s;
        }
        double r = U01_(rng_) * wsum;
        double cum = 0.0;
        for (int k = 0; k < gsize; ++k) {
            int s = group[k];
            cum += wloc[s];
            if (r <= cum) {
                strat_uses_[zone][s] += 1;
                return s;
            }
        }
        int s = group[gsize - 1];
        strat_uses_[zone][s] += 1;
        return s;
    };

    double rChoice = U01_(rng_);

    // First, explore vs exploit is decided,
    // then RL is allowed to select within the group.
    if (rChoice < explore_bias) {
        return sampleFromGroup(explore_group, 2);
    }
    return sampleFromGroup(exploit_group, 3);
}

void GARQ::recordOutcome(int zone, int strat, double gain)
{
    if (zone < 0 || zone >= NUM_ZONES_) return;
    if (strat < 0 || strat >= NUM_STRAT_) return;
    if (gain <= 0.0) return;
    strat_reward_[zone][strat] += gain;
}

void GARQ::updateRL()
{
    for (int z = 0; z < NUM_ZONES_; ++z) {
        double maxGain = 0.0;
        for (int s = 0; s < NUM_STRAT_; ++s) {
            if (strat_uses_[z][s] > 0) {
                double avg = strat_reward_[z][s] / (double)strat_uses_[z][s];
                if (avg > maxGain) maxGain = avg;
            }
        }

        if (maxGain <= 0.0) {
            // No strategy stood out -> gentle equalization
            for (int s = 0; s < NUM_STRAT_; ++s) {
                double target = min_weight_ + 1.0;
                strat_weight_[z][s] = (1.0 - rl_alpha_) * strat_weight_[z][s]
                                     + rl_alpha_     * target;
                strat_reward_[z][s] = 0.0;
                strat_uses_[z][s]   = 0;
            }
            continue;
        }

        for (int s = 0; s < NUM_STRAT_; ++s) {
            double avgNorm = 0.0;
            if (strat_uses_[z][s] > 0) {
                double avg = strat_reward_[z][s] / (double)strat_uses_[z][s];
                avgNorm = avg / maxGain;  // 0..1
            }
            double target = min_weight_ + avgNorm;
            if (target <= 0.0) target = min_weight_;

            strat_weight_[z][s] = (1.0 - rl_alpha_) * strat_weight_[z][s]
                                 + rl_alpha_     * target;

            strat_reward_[z][s] = 0.0;
            strat_uses_[z][s]   = 0;
        }
    }
}

// ---------------------------------------------------------------
// DE operators
// ---------------------------------------------------------------

int GARQ::randomIndexExcept(int n, int forbid)
{
    std::uniform_int_distribution<int> Ui(0, n - 1);
    int r;
    do { r = Ui(rng_); } while (r == forbid);
    return r;
}

int GARQ::pickPBestIndex(double pbest_fraction)
{
    int N = (int)FX_.size();
    if (N <= 0) return 0;
    int P = std::max(1, (int)std::round(pbest_fraction * (double)N));

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + P, idx.end(),
        [&](int a, int b){ return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> Ui(0, P - 1);
    return idx[Ui(rng_)];
}

void GARQ::pushArchive(const Vec& x)
{
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    const std::size_t cap =
        (std::size_t)std::max(0.0, std::round(archive_rate_ * (double)population()));
    if (cap > 0 && archive_.size() > cap) {
        std::shuffle(archive_.begin(), archive_.end(), rng_);
        archive_.resize(cap);
    }
}

int GARQ::pickArchiveIndex()
{
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> Ui(0, (int)archive_.size() - 1);
    return Ui(rng_);
}

void GARQ::make_trial_rand1(int i, Vec& trial, double F, double CR)
{
    const int N = (int)X_.size();
    const Vec& xi = X_[i];

    int r1 = randomIndexExcept(N, i);
    int r2;
    do { r2 = randomIndexExcept(N, i); } while (r2 == r1);
    int r3;
    do { r3 = randomIndexExcept(N, i); } while (r3 == r1 || r3 == r2);

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    trial.resize(D_);
    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            trial[j] = X_[r1][j] + F * (X_[r2][j] - X_[r3][j]);
        } else {
            trial[j] = xi[j];
        }
    }
}

void GARQ::make_trial_pbest(int i, Vec& trial, double F, double CR, double pbest_fraction)
{
    const int N = (int)X_.size();
    const Vec& xi = X_[i];

    int pbest = pickPBestIndex(pbest_fraction);

    int r1;
    do { r1 = randomIndexExcept(N, i); } while (r1 == pbest);
    int r2;
    do { r2 = randomIndexExcept(N, i); } while (r2 == pbest || r2 == r1);

    Vec base_r2 = X_[r2];
    int aIdx = pickArchiveIndex();
    if (aIdx >= 0) base_r2 = archive_[aIdx];

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    trial.resize(D_);
    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = xi[j]
                     + F * (X_[pbest][j] - xi[j])
                     + F * (X_[r1][j]   - base_r2[j]);
            trial[j] = v;
        } else {
            trial[j] = xi[j];
        }
    }
}

void GARQ::make_trial_best1(int i, Vec& trial, double F, double CR)
{
    const Vec& xi = X_[i];

    int r1 = randomIndexExcept((int)X_.size(), i);
    int r2;
    do { r2 = randomIndexExcept((int)X_.size(), i); } while (r2 == r1);

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    trial.resize(D_);
    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            trial[j] = best_x_[j] + F * (X_[r1][j] - X_[r2][j]);
        } else {
            trial[j] = xi[j];
        }
    }
}

void GARQ::make_trial_gaussian(Vec& trial, double sigma)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    if (!std::isfinite(best_f_) || best_x_.empty()) return;

    std::normal_distribution<double> N0(0.0, 1.0);

    trial.resize(D_);
    for (int j = 0; j < D_; ++j) {
        double step = sigma * (U[j] - L[j]) * N0(rng_);
        trial[j] = best_x_[j] + step;
    }
}

void GARQ::make_trial_longjump(Vec& trial)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::normal_distribution<double> N0(0.0, 1.0);

    trial.resize(D_);
    bool use_best = std::isfinite(best_f_) && !best_x_.empty();
    if (use_best) {
        for (int j = 0; j < D_; ++j) {
            double rad  = (U[j] - L[j]);
            double step = 0.7 * rad * N0(rng_);
            trial[j] = best_x_[j] + step;
        }
    } else {
        for (int j = 0; j < D_; ++j) {
            trial[j] = L[j] + U01_(rng_) * (U[j] - L[j]);
        }
    }
}

// ---------------------------------------------------------------
// ARQ-style quarantine outliers
// ---------------------------------------------------------------

void GARQ::quarantineOutliers()
{
    if (quarantine_rate_ <= 0.0) return;
    if ((int)FX_.size() < 8) return;

    std::vector<double> f;
    f.reserve(FX_.size());
    for (double v : FX_) if (std::isfinite(v)) f.push_back(v);
    if ((int)f.size() < 8) return;

    std::sort(f.begin(), f.end());
    auto qpos = [&](double q){
        double pos = (f.size() - 1) * q;
        std::size_t lo = (std::size_t)std::floor(pos);
        std::size_t hi = (std::size_t)std::ceil(pos);
        double a = f[lo], b = f[hi];
        return a + (b - a) * (pos - lo);
    };
    double q1  = qpos(0.25);
    double q3  = qpos(0.75);
    double iqr = std::max(0.0, q3 - q1);
    double thr = q3 + outlier_alpha_ * iqr;

    const int N = (int)FX_.size();
    const int D = D_;
    if (D <= 0) return;

    // Robust center: mean of the best 50%
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    int cut = std::max(1, (int)std::round(0.5 * (double)N));
    std::partial_sort(idx.begin(), idx.begin() + cut, idx.end(),
        [&](int a, int b){ return FX_[a] < FX_[b]; });

    Vec center(D, 0.0);
    int counted = 0;
    for (int t = 0; t < cut; ++t) {
        int i = idx[t];
        double fi = FX_[i];
        if (!std::isfinite(fi)) continue;
        for (int j = 0; j < D; ++j) center[j] += X_[i][j];
        counted++;
    }
    if (counted <= 0) return;
    for (int j = 0; j < D; ++j) center[j] /= (double)counted;

    std::normal_distribution<double> N0(0.0, 1.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    int maxFix = std::max(1, (int)std::round(quarantine_rate_ * (double)N));
    int fixed  = 0;

    for (int i = 0; i < N && fixed < maxFix; ++i) {
        double fi = FX_[i];
        if (!std::isfinite(fi) || fi <= thr) continue;

        Vec cand = X_[i];
        for (int j = 0; j < D; ++j) {
            double v = cand[j]
                     + quarantine_sigma_ * (center[j] - cand[j])
                     + 0.01 * (U[j] - L[j]) * N0(rng_);
            if (v < L[j])      v = L[j];
            else if (v > U[j]) v = U[j];
            cand[j] = v;
        }
        ensureBounds(cand);
        double fnew = eval(cand);
        if (fnew < FX_[i]) {
            pushArchive(X_[i]);
            X_[i]  = std::move(cand);
            FX_[i] = fnew;
            if (fnew < best_f_) {
                best_f_ = fnew;
                best_x_ = X_[i];
            }
        }
        fixed++;
        if (prob_->calls() >= max_evals_) break;
    }
}

// ---------------------------------------------------------------
// Micro-restart: restart of the worst individuals (half near the best, half global)
// ---------------------------------------------------------------

void GARQ::restartIndividuals()
{
    if (!prob_) return;
    int N = (int)X_.size();
    if (N <= 1) return;
    if (restart_frac_ <= 0.0) return;

    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

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

    int nrestart = std::max(1, (int)std::round(restart_frac_ * (double)N));
    for (int k = 0; k < nrestart; ++k) {
        int i = idx[N - 1 - k];
        if (i == elite) continue;

        Vec cand(D);
        bool around_best = std::isfinite(best_f_) && !best_x_.empty();
        bool global_jump  = (k % 2 == 1); // Half global, half near the best

        if (around_best && !global_jump) {
            for (int j = 0; j < D; ++j) {
                double step = restart_sigma_ * (U[j] - L[j]) * N0(rng_);
                cand[j] = best_x_[j] + step;
            }
        } else {
            for (int j = 0; j < D; ++j) {
                cand[j] = L[j] + U01_(rng_) * (U[j] - L[j]);
            }
        }
        ensureBounds(cand);
        double f = eval(cand);
        X_[i]  = cand;
        FX_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = cand;
        }
        if (prob_->calls() >= max_evals_) break;
    }
}

// ---------------------------------------------------------------
// one_iteration
// ---------------------------------------------------------------

void GARQ::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    int N = (int)X_.size();
    if (N < 4) return;

    std::vector<double> succF;
    std::vector<double> succCR;
    succF.reserve(N);
    succCR.reserve(N);

    double maxFES = (double)std::max<long long>(max_evals_, 1);

    // Diversity is computed once per iteration
    double div_now = computeDiversity();
    if (!diversity_initialized_) {
        diversity_ema_         = div_now;
        diversity_initialized_ = true;
    } else {
        diversity_ema_ = 0.8 * diversity_ema_ + 0.2 * div_now;
    }

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        double fi = FX_[i];
        if (!std::isfinite(fi)) {
            ensureBounds(X_[i]);
            fi     = eval(X_[i]);
            FX_[i] = fi;
        }

        int zone = determineZone(X_[i]);

        double nfes = (double)prob_->calls();
        double prog = (maxFES > 0.0 ? nfes / maxFES : 0.0);
        if (prog < 0.0) prog = 0.0;
        if (prog > 1.0) prog = 1.0;

        int strat = selectStrategy(zone, prog, diversity_ema_);

        double F, CR;
        sample_F_CR(F, CR);

        Vec trial;
        switch (strat) {
        case 0: // rand/1
            make_trial_rand1(i, trial, F, CR);
            break;
        case 1: { // current-to-pbest/1 + archive
            double p = pmax_ - (pmax_ - pmin_) * prog;
            if (p < pmin_) p = pmin_;
            make_trial_pbest(i, trial, F, CR, p);
            break;
        }
        case 2: // best/1
            make_trial_best1(i, trial, F, CR);
            break;
        case 3: { // Gaussian around the best
            double sigma = 0.30 * (1.0 - prog) + 0.05;
            make_trial_gaussian(trial, sigma);
            break;
        }
        case 4:
        default: // long-jump
            make_trial_longjump(trial);
            break;
        }

        ensureBounds(trial);
        double ftrial = eval(trial);

        if (ftrial < fi) {
            double gain = fi - ftrial;
            recordOutcome(zone, strat, gain);
            succF.push_back(F);
            succCR.push_back(CR);

            pushArchive(X_[i]);
            X_[i]  = std::move(trial);
            FX_[i] = ftrial;

            if (ftrial < best_f_) {
                best_f_ = ftrial;
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Update of LSHADE memory + RL
    updateMemory(succF, succCR);
    updateRL();

    // ARQ-style quarantine for very poor individuals
    quarantineOutliers();

    // Stagnation + micro-restart
    const double eps_impr = 1e-8;
    if (best_f_ < last_best_ - eps_impr) {
        last_best_ = best_f_;
        stagnation_counter_ = 0;
    } else {
        stagnation_counter_++;
        if (stagnation_counter_ >= stagnation_window_) {
            double nfes    = (double)prob_->calls();
            double maxFES2 = (double)std::max<long long>(max_evals_, 1);
            double ratio   = (maxFES2 > 0.0 ? nfes / maxFES2 : 1.0);

            // If the run is not at the very end, strong restarts are performed
            if (restart_frac_ > 0.0 && ratio < 0.98) {
                restartIndividuals();
            }

            // Reset and a small boost to exploration weights to avoid getting stuck again
            for (int z = 0; z < NUM_ZONES_; ++z) {
                strat_weight_[z][0] = std::max(strat_weight_[z][0], 1.8); // rand/1
                strat_weight_[z][4] = std::max(strat_weight_[z][4], 2.0); // longjump
            }

            stagnation_counter_ = 0;
            last_best_          = best_f_;
        }
    }

    // In-run local search (if requested)
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        int Nloc = (int)X_.size();
        int nsel = (int)std::round(local_rate_ * (double)Nloc);
        if (nsel > 0) {
            std::vector<int> idx(Nloc);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng_);
            nsel = std::min(nsel, Nloc);
            for (int k = 0; k < nsel; ++k) {
                int i = idx[k];
                auto res = localSearch(local_method_, X_[i]);
                if (!res.first.empty() && std::isfinite(res.second) && res.second < FX_[i]) {
                    X_[i]  = res.first;
                    FX_[i] = res.second;
                    if (res.second < best_f_) {
                        best_f_ = res.second;
                        best_x_ = res.first;
                    }
                }
                if (prob_->calls() >= max_evals_) break;
            }
        }
    }

    updateStop(FX_);
    printBest();
}

// ---------------------------------------------------------------
// end
// ---------------------------------------------------------------

void GARQ::end()
{
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        if (!res.first.empty() && std::isfinite(res.second) && res.second < best_f_) {
            best_x_ = res.first;
            best_f_ = res.second;
        }

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) {
                    fw = FX_[i];
                    worst = i;
                }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
