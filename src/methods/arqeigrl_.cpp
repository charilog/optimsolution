#include "arqeigrl.h"
#include "options.h"
#include "localsearch.h"

#include <numeric>
#include <cctype>
#include <random>
#include <iostream>

namespace optimsolution {

void ARQEigRL::configure(const MethodConfig& mc)
{
    // population
    pop_cfg_  = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
        pop_init_ = pop_cfg_;
    }

    // min population (LPSR)
    pop_min_  = mc.getInt("pop_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;

    // LSHADE memory parameters
    H_       = mc.getInt("H", H_);
    if (H_ <= 0) H_ = 10;
    c_mem_   = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    pmin_    = mc.getDbl("pmin", pmin_);
    pmax_    = mc.getDbl("pmax", pmax_);
    if (pmin_ <= 0.0 || pmin_ > 1.0) pmin_ = 0.05;
    if (pmax_ <= 0.0 || pmax_ > 1.0) pmax_ = 0.25;
    if (pmin_ > pmax_) std::swap(pmin_, pmax_);

    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);
    if (archive_rate_ < 0.0) archive_rate_ = 0.0;

    // ARQ-style parameters
    rtr_frac_        = mc.getDbl("rtr_frac", rtr_frac_);
    if (rtr_frac_ <= 0.0 || rtr_frac_ > 1.0) rtr_frac_ = 0.15;

    rtr_pool_        = mc.getInt("rtr_pool", rtr_pool_);
    if (rtr_pool_ < 2) rtr_pool_ = 5;

    outlier_alpha_   = mc.getDbl("outlier_alpha", outlier_alpha_);
    if (outlier_alpha_ <= 0.0) outlier_alpha_ = 1.5;

    quarantine_rate_ = mc.getDbl("quarantine_rate", quarantine_rate_);
    if (quarantine_rate_ < 0.0)  quarantine_rate_ = 0.0;
    if (quarantine_rate_ > 1.0)  quarantine_rate_ = 1.0;

    restart_frac_    = mc.getDbl("restart_frac", restart_frac_);
    if (restart_frac_ < 0.0) restart_frac_ = 0.0;
    if (restart_frac_ > 1.0) restart_frac_ = 0.15;

    restart_sigma_   = mc.getDbl("restart_sigma", restart_sigma_);
    if (restart_sigma_ <= 0.0) restart_sigma_ = 0.25;

    stagnation_window_ = mc.getInt("stagnation_window", stagnation_window_);
    if (stagnation_window_ <= 0) stagnation_window_ = 30;

    quarantine_period_ = mc.getInt("quarantine_period", quarantine_period_);
    if (quarantine_period_ <= 0) quarantine_period_ = 5;

    // in-run local search
    {
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local", local_method_)));
        for (char& c : lm) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        local_method_ = lm;

        double lr = mc.getDbl("local_rate",
                    mc.getDbl("local.rate",
                    mc.getDbl("inrun_rate", local_rate_)));
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

    // final local search (from [global])
    end_local_refine_ = mc.getBool("end_local_refine", false);
    end_local_method_ = mc.getStr("end_local_method",
                         mc.getStr("final_local_method", "lbfgs"));
}

void ARQEigRL::init()
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

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    X_.assign(pop_init_, Vec(D_, 0.0));
    FX_.assign(pop_init_, std::numeric_limits<double>::infinity());
    archive_.clear();

    // MF/MCR
    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D_, 0.0);
    last_best_ = best_f_;
    stagnation_counter_ = 0;

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

    updateStop(FX_);
    printBest();
}

// LSHADE sampling: random memory index per individual
void ARQEigRL::sample_F_CR(double& F, double& CR)
{
    if (H_ <= 0 || MF_.empty() || MCR_.empty()) {
        F  = 0.5;
        CR = 0.8;
        return;
    }

    std::uniform_int_distribution<int> Ui_mem(0, std::max(0, H_ - 1));
    int rMem = Ui_mem(rng_);
    double mF  = MF_[rMem];
    double mCR = MCR_[rMem];

    std::cauchy_distribution<double> cauchy(mF, 0.1);
    do {
        F = cauchy(rng_);
    } while (F <= 0.0);
    if (F > 1.0) F = 1.0;

    // minimum F so it does not collapse entirely
    const double F_MIN = 0.15;
    if (F < F_MIN) F = F_MIN;

    std::normal_distribution<double> normal(mCR, 0.1);
    CR = normal(rng_);
    if (CR < 0.0) CR = 0.0;
    if (CR > 1.0) CR = 1.0;
}

// stage-based strategy selection
int ARQEigRL::selectStrategy(double fes_ratio) const
{
    double w[4] = {1.0, 1.0, 1.0, 1.0};

    // initial stage: strong exploration (rand/1)
    if (fes_ratio < 0.30) {
        w[0] = 0.7;  // pbest
        w[1] = 2.5;  // rand/1
        w[2] = 0.5;  // best/1
        w[3] = 0.3;  // gaussian
    }
    // middle stage: balance
    else if (fes_ratio < 0.70) {
        w[0] = 2.0;  // pbest
        w[1] = 1.0;  // rand/1
        w[2] = 1.0;  // best/1
        w[3] = 0.7;  // gaussian
    }
    // late stage: exploitation
    else if (fes_ratio < 0.90) {
        w[0] = 1.0;
        w[1] = 0.5;
        w[2] = 2.0;
        w[3] = 1.5;
    }
    // final stage: very strong intensification around best
    else {
        w[0] = 0.5;
        w[1] = 0.2;
        w[2] = 3.0;
        w[3] = 2.5;
    }

    double sumw = 0.0;
    for (int k = 0; k < 4; ++k) {
        if (w[k] < 0.0) w[k] = 0.0;
        sumw += w[k];
    }
    if (sumw <= 0.0) {
        std::uniform_int_distribution<int> Ui(0, 3);
        return Ui(const_cast<ARQEigRL*>(this)->rng_);
    }

    std::uniform_real_distribution<double> U(0.0, sumw);
    double r = U(const_cast<ARQEigRL*>(this)->rng_);
    double acc = 0.0;
    for (int k = 0; k < 4; ++k) {
        acc += w[k];
        if (r <= acc) return k;
    }
    return 3;
}

double ARQEigRL::bnDistance(const Vec& a, const Vec& b) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double s = 0.0, eps = 1e-12;
    for (size_t j = 0; j < a.size(); ++j) {
        double r = U[j] - L[j];
        double d = (a[j] - b[j]) / (r + eps);
        s += d * d;
    }
    return std::sqrt(s);
}

int ARQEigRL::pickRTRNeighbor(const Vec& trial, const std::vector<int>& pool) const
{
    int best = pool[0];
    double bd = bnDistance(trial, X_[best]);
    for (size_t t = 1; t < pool.size(); ++t) {
        int k = pool[t];
        double d = bnDistance(trial, X_[k]);
        if (d < bd) {
            bd = d;
            best = k;
        }
    }
    return best;
}

void ARQEigRL::pushArchive(const Vec& x)
{
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    const size_t cap = (size_t)std::max(0.0, std::round(archive_rate_ * (double)population()));
    if (cap > 0 && archive_.size() > cap) {
        std::shuffle(archive_.begin(), archive_.end(), rng_);
        archive_.resize(cap);
    }
}

int ARQEigRL::pickArchiveIndex()
{
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> Ui(0, (int)archive_.size() - 1);
    return Ui(rng_);
}

int ARQEigRL::pickPBestIndex(double pbest_frac)
{
    int N = (int)FX_.size();
    if (N <= 0) return 0;
    int P = std::max(1, (int)std::round(pbest_frac * (double)N));

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + P, idx.end(),
        [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> J(0, P - 1);
    return idx[J(rng_)];
}

int ARQEigRL::distinctIndex(int n, int i)
{
    std::uniform_int_distribution<int> Ui(0, n - 1);
    int r;
    do { r = Ui(rng_); } while (r == i);
    return r;
}

// S0: current-to-pbest/1 + archive
void ARQEigRL::make_trial_S0(int i, Vec& tr, double F, double CR, double pbest_frac)
{
    const int N = (int)X_.size();
    const int pbest = pickPBestIndex(pbest_frac);

    std::uniform_int_distribution<int> Ui(0, N - 1);
    int r1, r2;
    do { r1 = Ui(rng_); } while (r1 == i || r1 == pbest);
    do { r2 = Ui(rng_); } while (r2 == i || r2 == pbest || r2 == r1);

    Vec base_r2 = X_[r2];
    int aIdx = pickArchiveIndex();
    if (aIdx >= 0) base_r2 = archive_[aIdx];

    const Vec& xi = X_[i];
    const Vec& xp = X_[pbest];

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = xi[j] + F * (xp[j] - xi[j]) + F * (X_[r1][j] - base_r2[j]);
            tr[j] = v;
        } else {
            tr[j] = xi[j];
        }
    }
}

// S1: rand/1/bin
void ARQEigRL::make_trial_S1(int i, Vec& tr, double F, double CR)
{
    const int N = (int)X_.size();
    const Vec& xi = X_[i];

    std::uniform_int_distribution<int> Ui(0, N - 1);
    int r1, r2, r3;
    do { r1 = Ui(rng_); } while (r1 == i);
    do { r2 = Ui(rng_); } while (r2 == i || r2 == r1);
    do { r3 = Ui(rng_); } while (r3 == i || r3 == r1 || r3 == r2);

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = X_[r1][j] + F * (X_[r2][j] - X_[r3][j]);
            tr[j] = v;
        } else {
            tr[j] = xi[j];
        }
    }
}

// S2: best/1/bin
void ARQEigRL::make_trial_S2(int i, Vec& tr, double F, double CR)
{
    const Vec& xi = X_[i];

    int best_idx = 0;
    double fb = FX_[0];
    for (int k = 1; k < (int)FX_.size(); ++k) {
        if (FX_[k] < fb) {
            fb = FX_[k];
            best_idx = k;
        }
    }

    std::uniform_int_distribution<int> Ui(0, (int)X_.size() - 1);
    int r1, r2;
    do { r1 = Ui(rng_); } while (r1 == i || r1 == best_idx);
    do { r2 = Ui(rng_); } while (r2 == i || r2 == best_idx || r2 == r1);

    std::uniform_int_distribution<int> Jdim(0, D_ - 1);
    int jrand = Jdim(rng_);

    for (int j = 0; j < D_; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = best_x_[j] + F * (X_[r1][j] - X_[r2][j]);
            tr[j] = v;
        } else {
            tr[j] = xi[j];
        }
    }
}

// S3: Gaussian around best_x_
void ARQEigRL::make_trial_S3_gaussian(Vec& tr, double sigma)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::normal_distribution<double> N0(0.0, 1.0);
    tr.resize(D_);
    for (int j = 0; j < D_; ++j) {
        double step = sigma * (U[j] - L[j]) * N0(rng_);
        tr[j] = clamp(best_x_[j] + step, L[j], U[j]);
    }
}

void ARQEigRL::quarantineOutliers()
{
    if (quarantine_rate_ <= 0.0) return;
    if ((int)FX_.size() < 8) return;

    std::vector<double> f;
    f.reserve(FX_.size());
    for (double v : FX_) if (std::isfinite(v)) f.push_back(v);
    if ((int)f.size() < 8) return;

    std::sort(f.begin(), f.end());
    auto qpos = [&](double q) {
        double pos = (f.size() - 1) * q;
        size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
        double a = f[lo], b = f[hi];
        return a + (b - a) * (pos - lo);
    };
    double q1  = qpos(0.25), q3 = qpos(0.75);
    double iqr = std::max(0.0, q3 - q1);
    double thr = q3 + outlier_alpha_ * iqr;

    const int N = (int)FX_.size();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    int cut = std::max(1, (int)std::round(0.5 * (double)N));
    std::partial_sort(idx.begin(), idx.begin() + cut, idx.end(),
        [&](int a, int b) { return FX_[a] < FX_[b]; });

    Vec center(D_, 0.0);
    int counted = 0;
    for (int t = 0; t < cut; ++t) {
        int i = idx[t];
        double fi = FX_[i];
        if (!std::isfinite(fi)) continue;
        for (int j = 0; j < D_; ++j) center[j] += X_[i][j];
        counted++;
    }
    if (counted == 0) return;
    for (int j = 0; j < D_; ++j) center[j] /= (double)counted;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::normal_distribution<double> N0(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        double fi = FX_[i];
        if (!std::isfinite(fi) || fi <= thr) continue;

        Vec cand(D_);
        for (int j = 0; j < D_; ++j) {
            double step = quarantine_rate_ * (U[j] - L[j]) * N0(rng_);
            cand[j] = clamp(center[j] + step, L[j], U[j]);
        }
        ensureInBounds(cand);
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
        if (prob_->calls() >= max_evals_) break;
    }
}

void ARQEigRL::microRestart()
{
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N <= 1) return;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

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

    int elite = idx[0];
    std::normal_distribution<double> N0(0.0, 1.0);

    int nreset = std::max(1, (int)std::round(restart_frac_ * (double)N));
    for (int k = 0; k < nreset; ++k) {
        int i = idx[N - 1 - k];
        if (i == elite) continue;

        Vec cand(D_);
        bool globalReset = (k % 2 == 0);  // half global, half around best

        if (globalReset) {
            for (int j = 0; j < D_; ++j) {
                double r = U01_(rng_);
                cand[j] = L[j] + r * (U[j] - L[j]);
            }
        } else {
            for (int j = 0; j < D_; ++j) {
                double step = restart_sigma_ * (U[j] - L[j]) * N0(rng_);
                cand[j] = clamp(best_x_[j] + step, L[j], U[j]);
            }
        }

        ensureInBounds(cand);
        double f = eval(cand);

        pushArchive(X_[i]);
        X_[i]  = std::move(cand);
        FX_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }
}

void ARQEigRL::shrinkPopulationLPSR()
{
    int N = (int)X_.size();
    if (N <= pop_min_) return;

    double maxFES = (double)std::max<long long>(max_evals_, 1);
    double nfes   = (double)prob_->calls();
    double ratio  = nfes / maxFES;

    // slower LPSR for small dimensions
    if (D_ <= 20) {
        ratio *= 0.5;
        if (ratio > 1.0) ratio = 1.0;
    }

    int targetN = (int)std::round(pop_init_ - (pop_init_ - pop_min_) * ratio);
    if (targetN < pop_min_)  targetN = pop_min_;
    if (targetN > pop_init_) targetN = pop_init_;
    if (targetN >= N) return;

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + targetN, idx.end(),
        [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    newX(targetN);
    std::vector<double> newF(targetN);
    for (int i = 0; i < targetN; ++i) {
        newX[i] = X_[idx[i]];
        newF[i] = FX_[idx[i]];
    }
    X_.swap(newX);
    FX_.swap(newF);
}

void ARQEigRL::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    int N = (int)X_.size();
    if (N < 4) return;

    shrinkPopulationLPSR();
    N = (int)X_.size();
    if (N < 4) return;

    std::vector<Vec>    newX = X_;
    std::vector<double> newF = FX_;

    std::vector<double> succ_F;
    std::vector<double> succ_CR;
    succ_F.reserve(N);
    succ_CR.reserve(N);

    double maxFES = (double)std::max<long long>(max_evals_, 1);

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        double nfes   = (double)prob_->calls();
        double ratio  = nfes / maxFES;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;

        int strat = selectStrategy(ratio);

        double F = 0.0, CR = 0.0;
        sample_F_CR(F, CR);

        double pbest_frac = pmax_ - (pmax_ - pmin_) * ratio;

        Vec trial(D_);
        switch (strat) {
        default:
        case 0:
            make_trial_S0(i, trial, F, CR, pbest_frac);
            break;
        case 1:
            make_trial_S1(i, trial, F, CR);
            break;
        case 2:
            make_trial_S2(i, trial, F, CR);
            break;
        case 3:
            make_trial_S3_gaussian(trial, 0.10);
            break;
        }

        ensureInBounds(trial);
        double ftrial = eval(trial);

        std::vector<int> pool;
        int maxPoolByFrac = (int)std::round(rtr_frac_ * (double)N);
        if (maxPoolByFrac < 2) maxPoolByFrac = 2;
        if (maxPoolByFrac > N) maxPoolByFrac = N;
        int poolSize = std::max(2, std::min(rtr_pool_, maxPoolByFrac));
        pool.reserve(poolSize);
        std::uniform_int_distribution<int> Ui(0, N - 1);
        for (int k = 0; k < poolSize; ++k) {
            pool.push_back(Ui(rng_));
        }
        int repl_idx = pickRTRNeighbor(trial, pool);

        double parent_f = newF[repl_idx];

        if (ftrial < parent_f) {
            // successful step
            pushArchive(newX[repl_idx]);
            newX[repl_idx] = std::move(trial);
            newF[repl_idx] = ftrial;

            succ_F.push_back(F);
            succ_CR.push_back(CR);

            if (ftrial < best_f_) {
                best_f_ = ftrial;
                best_x_ = newX[repl_idx];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    X_.swap(newX);
    FX_.swap(newF);

    // LSHADE memory update
    if (!succ_F.empty()) {
        double numF = 0.0, denF = 0.0;
        double meanCR = 0.0;
        int    cntCR  = 0;
        for (size_t i = 0; i < succ_F.size(); ++i) {
            double Fi = succ_F[i];
            numF += Fi * Fi;
            denF += Fi;
        }
        for (double cr : succ_CR) {
            meanCR += cr;
            cntCR++;
        }
        if (denF > 0.0) {
            double mF = numF / denF;
            MF_[mem_idx_] = (1.0 - c_mem_) * MF_[mem_idx_] + c_mem_ * mF;
        }
        if (cntCR > 0) {
            meanCR /= (double)cntCR;
            MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * meanCR;
        }
        mem_idx_++;
        if (mem_idx_ >= H_) mem_idx_ = 0;
    }

    // quarantine not at every iteration, but periodically
    if (quarantine_period_ > 0 && (iters_ % quarantine_period_) == 0) {
        quarantineOutliers();
    }

    // stagnation tracking for microRestart
    double maxFES2 = (double)std::max<long long>(max_evals_, 1);
    double ratio2  = (double)prob_->calls() / maxFES2;
    if (ratio2 < 0.0) ratio2 = 0.0;
    if (ratio2 > 1.0) ratio2 = 1.0;

    if (best_f_ < last_best_ - 1e-12) {
        last_best_ = best_f_;
        stagnation_counter_ = 0;
    } else {
        stagnation_counter_++;
        if (stagnation_counter_ >= stagnation_window_) {
            if (ratio2 < 0.90) {
                microRestart();
            }
            stagnation_counter_ = 0;
            last_best_ = best_f_;
        }
    }

    // in-run local search (if enabled)
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        int local_n = (int)std::round(local_rate_ * (double)X_.size());
        if (local_n > 0) {
            std::vector<int> idx(X_.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng_);
            local_n = std::min(local_n, (int)idx.size());
            for (int t = 0; t < local_n; ++t) {
                int i = idx[t];
                auto res = localSearch(local_method_, X_[i]);
                if (!res.first.empty() && std::isfinite(res.second) && res.second < FX_[i]) {
                    X_[i]  = std::move(res.first);
                    FX_[i] = res.second;
                    if (res.second < best_f_) {
                        best_f_ = res.second;
                        best_x_ = X_[i];
                    }
                }
                if (prob_->calls() >= max_evals_) break;
            }
        }
    }

    updateStop(FX_);
    printBest();
}

void ARQEigRL::end()
{
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        if (!res.first.empty() && std::isfinite(res.second) && res.second < best_f_) {
            best_x_ = std::move(res.first);
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

        printBest();
    }

    updateStop(FX_);
}

} // namespace optimsolution
