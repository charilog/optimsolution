// gderl.cpp
#include "gderl.h"
#include "options.h"
#include "localsearch.h"

#include <numeric>
#include <cctype>
#include <iostream>

namespace optimsolution {

double GDERL::eval(const Vec& x)
{
    if (!prob_) return std::numeric_limits<double>::infinity();
    return prob_->evaluate(x);
}

void GDERL::ensureBounds(Vec& x)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        // If NaN/Inf appears for any reason, move to middle of the domain
        if (!std::isfinite(x[j])) {
            x[j] = 0.5 * (L[j] + U[j]);
        }
        if (x[j] < L[j]) x[j] = L[j];
        else if (x[j] > U[j]) x[j] = U[j];
    }
}

double GDERL::bnDistance(const Vec& a, const Vec& b) const
{
    if (!prob_) return 0.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double s = 0.0;
    const double eps = 1e-12;
    for (size_t j = 0; j < a.size(); ++j) {
        double r = U[j] - L[j];
        double d = (a[j] - b[j]) / (r + eps);
        s += d * d;
    }
    return std::sqrt(s);
}

void GDERL::configure(const MethodConfig& mc)
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
    if (pmax_ <= 0.0 || pmax_ > 1.0) pmax_ = 0.25;
    if (pmin_ > pmax_) std::swap(pmin_, pmax_);

    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);
    if (archive_rate_ < 0.0) archive_rate_ = 0.0;

    // dual-zone parameters
    zone_radius_ = mc.getDbl("zone_radius", zone_radius_);
    if (zone_radius_ <= 0.0) zone_radius_ = 0.2;
    if (zone_radius_ > 1.0)  zone_radius_ = 1.0;

    rl_alpha_ = mc.getDbl("rl_alpha", rl_alpha_);
    if (rl_alpha_ <= 0.0 || rl_alpha_ >= 1.0) rl_alpha_ = 0.3;

    min_weight_ = mc.getDbl("min_weight", min_weight_);
    if (min_weight_ < 0.0) min_weight_ = 0.0;

    restart_frac_ = mc.getDbl("restart_frac", restart_frac_);
    if (restart_frac_ < 0.0) restart_frac_ = 0.0;
    // If user writes e.g. 018 -> 18, interpret as 0.18
    if (restart_frac_ > 1.0 && restart_frac_ <= 100.0) {
        restart_frac_ /= 100.0;
    }
    if (restart_frac_ > 0.5) restart_frac_ = 0.5;

    restart_sigma_ = mc.getDbl("restart_sigma", restart_sigma_);
    if (restart_sigma_ <= 0.0) restart_sigma_ = 0.3;

    stagnation_window_ = mc.getInt("stagnation_window", stagnation_window_);
    if (stagnation_window_ <= 0) stagnation_window_ = 30;

    outlier_alpha_ = mc.getDbl("outlier_alpha", outlier_alpha_);
    if (outlier_alpha_ <= 0.0) outlier_alpha_ = 1.5;

    relocate_rate_ = mc.getDbl("relocate_rate", relocate_rate_);
    if (relocate_rate_ < 0.0) relocate_rate_ = 0.2;

    // local search
    {
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local", local_method_)));
        for (auto &c : lm) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        local_method_ = lm;

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
}

void GDERL::init()
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

    // LSHADE memory initialisation
    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    // Dual-zone RL initialisation
    for (int z = 0; z < NUM_ZONES_; ++z) {
        for (int s = 0; s < NUM_STRAT_; ++s) {
            strat_weight_[z][s] = 1.0;
            strat_reward_[z][s] = 0.0;
            strat_uses_[z][s]   = 0;
        }
    }
    // more exploration initially in zone 0
    strat_weight_[0][0] = 1.5; // rand/1
    strat_weight_[0][1] = 0.7; // pbest
    strat_weight_[0][2] = 0.4; // best/1
    strat_weight_[0][3] = 0.4; // gaussian
    strat_weight_[0][4] = 2.0; // longjump

    // more exploitation initially in zone 1
    strat_weight_[1][0] = 0.4; // rand/1
    strat_weight_[1][1] = 2.0; // pbest
    strat_weight_[1][2] = 2.0; // best/1
    strat_weight_[1][3] = 2.0; // gaussian
    strat_weight_[1][4] = 0.3; // longjump

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
    }
}

void GDERL::sample_F_CR(double& F, double& CR)
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
    const double F_MIN = 0.20; // aggressive as in the earlier strong Potential 38 configuration
    if (F < F_MIN) F = F_MIN;

    std::normal_distribution<double> normal(mCR, 0.1);
    CR = normal(rng_);
    if (CR < 0.0) CR = 0.0;
    if (CR > 1.0) CR = 1.0;
}

void GDERL::updateMemory(const std::vector<double>& succF,
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

    for (size_t i = 0; i < succF.size(); ++i) {
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

int GDERL::determineZone(const Vec& x) const
{
    if (!prob_ || !std::isfinite(best_f_)) return 0;
    double d = bnDistance(x, best_x_);
    if (d <= zone_radius_) return 1; // exploit near best
    return 0;                        // explore far away
}

// ======================
// Only the strategy selection changes
// Criterion: only strat_weight_ per zone
// progress is no longer used.
// ======================
int GDERL::selectStrategy(int zone, double progress)
{
    (void)progress; // progress is intentionally unused to avoid warnings

    if (zone < 0 || zone >= NUM_ZONES_) zone = 0;

    // Weights are taken only from strat_weight_[zone][s] (with a min_weight_ threshold)
    double w[NUM_STRAT_];
    for (int s = 0; s < NUM_STRAT_; ++s) {
        double base = strat_weight_[zone][s];
        if (base < min_weight_) base = min_weight_;
        w[s] = base;
    }

    double sumw = 0.0;
    for (int s = 0; s < NUM_STRAT_; ++s) {
        sumw += w[s];
    }

    if (sumw <= 0.0) {
        // Fallback: if something goes very wrong, select rand/1
        return 0;
    }

    std::uniform_real_distribution<double> U(0.0, sumw);
    double r = U(rng_);

    for (int s = 0; s < NUM_STRAT_; ++s) {
        if (r <= w[s]) return s;
        r -= w[s];
    }
    return NUM_STRAT_ - 1;
}

void GDERL::recordOutcome(int zone, int strat, double gain)
{
    if (zone < 0 || zone >= NUM_ZONES_) return;
    if (strat < 0 || strat >= NUM_STRAT_) return;
    if (gain <= 0.0) return;
    strat_reward_[zone][strat] += gain;
}

void GDERL::updateRL()
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
            // No clear preference -> soft equalization
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
                avgNorm = avg / maxGain;
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

int GDERL::randomIndexExcept(int n, int forbid)
{
    std::uniform_int_distribution<int> Ui(0, n - 1);
    int r;
    do { r = Ui(rng_); } while (r == forbid);
    return r;
}

int GDERL::pickPBestIndex(double pbest_fraction)
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

void GDERL::pushArchive(const Vec& x)
{
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    const size_t cap = (size_t)std::max(0.0, std::round(archive_rate_ * (double)population()));
    if (cap > 0 && archive_.size() > cap) {
        std::shuffle(archive_.begin(), archive_.end(), rng_);
        archive_.resize(cap);
    }
}

int GDERL::pickArchiveIndex()
{
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> Ui(0, (int)archive_.size() - 1);
    return Ui(rng_);
}

void GDERL::make_trial_rand1(int i, Vec& trial, double F, double CR)
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
            double v = X_[r1][j] + F * (X_[r2][j] - X_[r3][j]);
            trial[j] = v;
        } else {
            trial[j] = xi[j];
        }
    }
}

void GDERL::make_trial_pbest(int i, Vec& trial, double F, double CR, double pbest_fraction)
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

void GDERL::make_trial_best1(int i, Vec& trial, double F, double CR)
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
            double v = best_x_[j] + F * (X_[r1][j] - X_[r2][j]);
            trial[j] = v;
        } else {
            trial[j] = xi[j];
        }
    }
}

void GDERL::make_trial_gaussian(Vec& trial, double sigma)
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

void GDERL::make_trial_longjump(Vec& trial)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::normal_distribution<double> N0(0.0, 1.0);

    trial.resize(D_);

    bool use_best = std::isfinite(best_f_) && !best_x_.empty();
    if (use_best) {
        for (int j = 0; j < D_; ++j) {
            double rad = (U[j] - L[j]);
            double step = 0.7 * rad * N0(rng_);
            trial[j] = best_x_[j] + step;
        }
    } else {
        for (int j = 0; j < D_; ++j) {
            trial[j] = L[j] + U01_(rng_) * (U[j] - L[j]);
        }
    }
}

void GDERL::restartIndividuals()
{
    if (!prob_) return;
    int N = (int)X_.size();
    if (N <= 0) return;
    if (restart_frac_ <= 0.0) return;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::normal_distribution<double> N0(0.0, 1.0);

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });

    int best_idx = idx[0];
    int nreset   = std::max(1, (int)std::round(restart_frac_ * (double)N));

    for (int k = 0; k < nreset; ++k) {
        int i = idx[N - 1 - k];
        if (i == best_idx) continue;

        Vec cand(D_);
        bool around_best = (k % 2 == 0);
        if (around_best && std::isfinite(best_f_)) {
            for (int j = 0; j < D_; ++j) {
                double step = restart_sigma_ * (U[j] - L[j]) * N0(rng_);
                cand[j] = best_x_[j] + step;
            }
        } else {
            for (int j = 0; j < D_; ++j) {
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

void GDERL::one_iteration()
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

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        double nfes  = (double)prob_->calls();
        double ratio = nfes / maxFES;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;

        double pbest_fraction = pmax_ - (pmax_ - pmin_) * ratio;

        double F = 0.0, CR = 0.0;
        sample_F_CR(F, CR);

        int zone  = determineZone(X_[i]);
        int strat = selectStrategy(zone, ratio);
        if (zone  < 0 || zone  >= NUM_ZONES_) zone  = 0;
        if (strat < 0 || strat >= NUM_STRAT_) strat = 0;
        strat_uses_[zone][strat]++;

        Vec trial;
        switch (strat) {
        default:
        case 0:
            make_trial_rand1(i, trial, F, CR);
            break;
        case 1:
            make_trial_pbest(i, trial, F, CR, pbest_fraction);
            break;
        case 2:
            make_trial_best1(i, trial, F, CR);
            break;
        case 3:
            make_trial_gaussian(trial, 0.10);
            break;
        case 4:
            make_trial_longjump(trial);
            break;
        }

        ensureBounds(trial);
        double ftrial = eval(trial);

        if (ftrial < FX_[i]) {
            double gain = FX_[i] - ftrial;
            recordOutcome(zone, strat, gain);

            FX_[i] = ftrial;
            X_[i]  = trial;

            succF.push_back(F);
            succCR.push_back(CR);

            if (ftrial < best_f_) {
                best_f_ = ftrial;
                best_x_ = trial;
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    if (!succF.empty()) {
        updateMemory(succF, succCR);
    }

    updateRL();

    // Stagnation & restart
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
            if (restart_frac_ > 0.0 && ratio < 0.98) {
                restartIndividuals();
            }
            stagnation_counter_ = 0;
            last_best_ = best_f_;
        }
    }

    // In-run local search (if enabled in settings)
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

void GDERL::end()
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
