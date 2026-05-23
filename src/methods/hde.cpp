#include "hde.h"
#include "init.h"
#include "options.h"

#include <numeric>
#include <cctype>
#include <cmath>

namespace optimsolution  {

void HDE::configure(const MethodConfig& mc)
{
    // Population
    int p = mc.getInt("population", pop_init_);
    if (p > 3) {
        pop_init_ = p;
        Optimizer::setPopulation(pop_init_);
    }
    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;

    // CoBiDE / eigen-like controls
    CBps_ = mc.getDbl("cb_ps", CBps_);
    if (CBps_ <= 0.0 || CBps_ >= 1.0) CBps_ = 0.5;

    peig_ = mc.getDbl("peig", peig_);
    if (peig_ < 0.0) peig_ = 0.0;
    if (peig_ > 1.0) peig_ = 0.4;

    // In-run local search
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // Final local refinement
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);

    // ARQ mechanisms (optional overrides)
    arq_pbest_ = mc.getDbl("arq_pbest", mc.getDbl("arq_p", arq_pbest_));
    if (arq_pbest_ < 0.01) arq_pbest_ = 0.01;
    if (arq_pbest_ > 0.50) arq_pbest_ = 0.50;

    arq_agent_fraction_ = mc.getDbl("arq_agent_fraction", mc.getDbl("arq_agentfraction", arq_agent_fraction_));
    if (arq_agent_fraction_ <= 0.0) arq_agent_fraction_ = 0.60;
    if (arq_agent_fraction_ > 1.0)  arq_agent_fraction_ = 1.0;

    arq_muF_ = mc.getDbl("arq_muF", arq_muF_);
    arq_muCR_ = mc.getDbl("arq_muCR", arq_muCR_);

    arq_Flo_ = mc.getDbl("arq_Flo", arq_Flo_);
    arq_Fhi_ = mc.getDbl("arq_Fhi", arq_Fhi_);
    if (arq_Flo_ <= 0.0) arq_Flo_ = 0.01;
    if (arq_Fhi_ < arq_Flo_) std::swap(arq_Fhi_, arq_Flo_);

    arq_rtr_pool_ = mc.getInt("arq_rtr_pool", mc.getInt("arq_rtrpool", arq_rtr_pool_));
    if (arq_rtr_pool_ < 2) arq_rtr_pool_ = 2;

    arq_archive_rate_ = mc.getDbl("arq_archive_rate", mc.getDbl("arq_archiverate", arq_archive_rate_));
    if (arq_archive_rate_ <= 0.1) arq_archive_rate_ = 1.0;

    arq_shc_ = mc.getDbl("arq_shc", arq_shc_);
    if (arq_shc_ < 0.0) arq_shc_ = 0.0;
    if (arq_shc_ > 1.0) arq_shc_ = 1.0;

    // BHO mechanisms (optional overrides)
    bho_heal_prob_ = mc.getDbl("bho_heal_prob", mc.getDbl("heal_prob", bho_heal_prob_));
    if (bho_heal_prob_ < 0.0) bho_heal_prob_ = 0.0;
    if (bho_heal_prob_ > 1.0) bho_heal_prob_ = 1.0;

    bho_heal_rate_ = mc.getDbl("bho_heal_rate", mc.getDbl("heal_rate", bho_heal_rate_));
    if (bho_heal_rate_ < 0.0) bho_heal_rate_ = 0.0;

    bho_wound_strength_init_ = mc.getDbl("bho_wound_strength_init", mc.getDbl("wound_strength_init", bho_wound_strength_init_));
    if (bho_wound_strength_init_ < 0.0) bho_wound_strength_init_ = 0.0;

    bho_stagnation_kick_ = mc.getInt("bho_stagnation_kick", mc.getInt("stagnation_kick", bho_stagnation_kick_));
    if (bho_stagnation_kick_ < 1) bho_stagnation_kick_ = 1;

    bho_stagnation_restart_ = mc.getInt("bho_stagnation_restart", mc.getInt("stagnation_restart", bho_stagnation_restart_));
    if (bho_stagnation_restart_ < 1) bho_stagnation_restart_ = 1;

    bho_elite_kick_sigma_ = mc.getDbl("bho_elite_kick_sigma", mc.getDbl("elite_kick_sigma", bho_elite_kick_sigma_));
    if (bho_elite_kick_sigma_ < 0.0) bho_elite_kick_sigma_ = 0.0;

    bho_restart_frac_ = mc.getDbl("bho_restart_frac", mc.getDbl("restart_frac", bho_restart_frac_));
    if (bho_restart_frac_ < 0.0) bho_restart_frac_ = 0.0;
    if (bho_restart_frac_ > 1.0) bho_restart_frac_ = 1.0;
}

void HDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    Optimizer::setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    A_.clear();
    A_arq_.clear();

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);

    N_ = static_cast<int>(X_.size());
    if (N_ < 4) return;

    FX_.assign(N_, std::numeric_limits<double>::infinity());
    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N_; ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    N_init_run_ = N_;

    if (pop_min_ <= 0) pop_min_ = 10;
    pop_min_ = std::max(pop_min_, 10);

    // Roulette (6 operators)
    h_      = 6;
    n0_     = 2;
    delta_  = 1.0 / (5.0 * h_);
    ni_.assign(h_, static_cast<double>(n0_));
    cni_.assign(h_, 0.0);
    success_.assign(h_, 0);
    nrst_ = 0;

    // IDE schedule
    g_    = 0;
    gmax_ = std::max(1, (int)std::round((double)max_evals_ / std::max(N_, 1)));
    T_    = gmax_ / 10.0;
    GT_   = gmax_ / 2;
    gt_   = GT_;
    Tcurr_= 0;
    SRT_  = 0.0;

    // CoBiDE parameters per individual
    CBF_.assign(N_, 0.0);
    CBCR_.assign(N_, 0.0);
    for (int i = 0; i < N_; ++i) {
        double F;
        if (randU() < 0.5) F = cauchy(0.65, 0.1);
        else               F = cauchy(1.0, 0.1);
        while (F < 0.0) {
            if (randU() < 0.5) F = cauchy(0.65, 0.1);
            else               F = cauchy(1.0, 0.1);
        }
        if (F > 1.0) F = 1.0;
        CBF_[i] = F;

        double CR;
        if (randU() < 0.5) CR = cauchy(0.1, 0.1);
        else               CR = cauchy(0.95, 0.1);
        if (CR > 1.0) CR = 1.0;
        if (CR < 0.0) CR = 0.0;
        CBCR_[i] = CR;
    }

    // CMA-ES strategy (simplified, C=I)
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double range0 = (D > 0) ? (U[0] - L[0]) : 1.0;
    if (!std::isfinite(range0) || range0 <= 0.0) range0 = 1.0;
    sigma_ = range0 / 2.0;
    myeps_ = 1e-6;

    mu_ = N_ / 2;
    if (mu_ < 1) mu_ = 1;
    weights_.assign(mu_, 0.0);
    for (int i = 0; i < mu_; ++i) {
        double wi = std::log(mu_ + 0.5) - std::log((double)(i + 1));
        weights_[i] = wi;
    }
    double sw = 0.0;
    for (double w : weights_) sw += w;
    if (sw <= 0.0) sw = 1.0;
    for (double& w : weights_) w /= sw;
    double sw2 = 0.0;
    for (double w : weights_) sw2 += w * w;
    mueff_ = sw * sw / sw2;

    const double Dd = (double)D;
    cc_ = (4.0 + mueff_ / Dd) / (Dd + 4.0 + 2.0 * mueff_ / Dd);
    cs_ = (mueff_ + 2.0) / (Dd + mueff_ + 5.0);
    c1_ = 2.0 / (std::pow(Dd + 1.3, 2.0) + mueff_);
    cmu_= std::min(1.0 - c1_,
                   2.0 * (mueff_ - 2.0 + 1.0 / mueff_) /
                       (std::pow(Dd + 2.0, 2.0) + mueff_));
    damps_ = 1.0 + 2.0 *
        std::max(0.0, std::sqrt((mueff_ - 1.0) / (Dd + 1.0)) - 1.0) + cs_;

    pc_.assign(D, 0.0);
    ps_.assign(D, 0.0);
    B_.assign(D * D, 0.0);
    diagD_.assign(D, 1.0);
    C_.assign(D * D, 0.0);
    invsqrtC_.assign(D * D, 0.0);
    for (int i = 0; i < D; ++i) {
        B_[i * D + i]        = 1.0;
        C_[i * D + i]        = 1.0;
        invsqrtC_[i * D + i] = 1.0;
    }
    eigeneval_ = 0;
    chiN_ = std::sqrt(Dd) *
            (1.0 - 1.0 / (4.0 * Dd) + 1.0 / (21.0 * Dd * Dd));

    // oldPop_ is the previous CMA sample batch (index-aligned to CMA sampling)
    oldPop_.resize(N_);
    for (int i = 0; i < N_; ++i)
        oldPop_[i] = X_[i];

    // jSO memory
    Asize_max_ = (int)std::round(N_ * 2.6);
    Asize_     = 0;
    H_jso_ = 5;
    MF_.assign(H_jso_, 0.3);
    MCR_.assign(H_jso_, 0.8);
    MF_.back()  = 0.9;
    MCR_.back() = 0.9;
    k_mem_ = 0;
    pmax_  = 0.25;
    pmin_  = pmax_ / 2.0;

    // BHO counters
    bho_iters_ = 0;
    bho_sinceBest_ = 0;

    updateStop(FX_);
    printBest();
}

void HDE::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D   = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j]))
            x[j] = 0.5 * (L[j] + U[j]);
        // mirror repair
        while (x[j] < L[j] || x[j] > U[j]) {
            if (x[j] > U[j])
                x[j] = 2.0 * U[j] - x[j];
            else if (x[j] < L[j])
                x[j] = 2.0 * L[j] - x[j];
        }
    }
}

int HDE::randInt(int lo, int hi)
{
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double HDE::randU()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double HDE::randN01()
{
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double HDE::cauchy(double loc, double scale)
{
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

void HDE::sampleDistinct(int N, int k, std::vector<int>& out)
{
    out.clear();
    out.reserve(k);
    if (N <= 0 || k <= 0) return;
    if (k >= N) {
        out.resize(N);
        std::iota(out.begin(), out.end(), 0);
        return;
    }
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, N - 1);
        int r = dist(rng_);
        std::swap(idx[i], idx[r]);
        out.push_back(idx[i]);
    }
}

void HDE::sampleDistinctExcluding(int N, int k,
                                  const std::vector<int>& exclude,
                                  std::vector<int>& out)
{
    out.clear();
    out.reserve(k);
    if (N <= 0 || k <= 0) return;
    std::vector<int> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::find(exclude.begin(), exclude.end(), i) == exclude.end())
            candidates.push_back(i);
    }

    if (candidates.empty()) {
        out.clear();
        return;
    }

    if ((int)candidates.size() <= k) {
        out = candidates;
        return;
    }
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)candidates.size() - 1);
        int r = dist(rng_);
        std::swap(candidates[i], candidates[r]);
        out.push_back(candidates[i]);
    }
}

std::pair<int,double> HDE::rouletteSelect() const
{
    const int h = h_;
    if (h <= 0) return {0, 0.0};

    double sumni = 0.0;
    for (int i = 0; i < h; ++i) sumni += ni_[i];

    auto* self = const_cast<HDE*>(this);

    if (sumni <= 0.0) {
        std::uniform_int_distribution<int> dist(0, h - 1);
        int idx = dist(self->rng_);
        double p = 1.0 / h;
        return {idx, p};
    }

    std::vector<double> p(h);
    double pmin = 1.0;
    for (int i = 0; i < h; ++i) {
        p[i] = ni_[i] / sumni;
        if (p[i] < pmin) pmin = p[i];
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(self->rng_);
    double acc = 0.0;
    for (int i = 0; i < h; ++i) {
        acc += p[i];
        if (r <= acc) return {i, pmin};
    }
    return {h - 1, pmin};
}

void HDE::sortByFitness()
{
    const int N = N_;
    if (N <= 1) return;

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec> newX(N);
    std::vector<double> newF(N);
    std::vector<double> newCBF(N);
    std::vector<double> newCBCR(N);

    for (int i = 0; i < N; ++i) {
        int s = idx[i];
        newX[i] = std::move(X_[s]);
        newF[i] = FX_[s];
        newCBF[i]  = (s < (int)CBF_.size() ? CBF_[s] : 0.5);
        newCBCR[i] = (s < (int)CBCR_.size() ? CBCR_[s] : 0.9);
    }

    X_.swap(newX);
    FX_.swap(newF);
    CBF_.swap(newCBF);
    CBCR_.swap(newCBCR);
}

void HDE::shrinkPopulation(int newN)
{
    if (newN >= N_) return;
    if (newN < pop_min_) newN = pop_min_;
    if (newN >= N_) return;

    sortByFitness();

    X_.resize(newN);
    FX_.resize(newN);
    CBF_.resize(newN);
    CBCR_.resize(newN);
    N_ = newN;

    // Resize CMA batch buffer to match current N (index-aligned).
    if ((int)oldPop_.size() > N_) oldPop_.resize(N_);
    if ((int)oldPop_.size() < N_) oldPop_.resize(N_, best_x_);

    // Keep jSO archive bounded
    Asize_max_ = (int)std::round(N_ * 2.6);
    if (Asize_max_ < 0) Asize_max_ = 0;
    if ((int)A_.size() > Asize_max_) {
        std::shuffle(A_.begin(), A_.end(), rng_);
        A_.resize(Asize_max_);
    }
    Asize_ = (int)A_.size();
}

void HDE::addToArchive(const Vec& x)
{
    if (Asize_max_ <= 0) return;
    if (Asize_ < Asize_max_) {
        A_.push_back(x);
        ++Asize_;
    } else if (!A_.empty()) {
        int idx = randInt(0, Asize_ - 1);
        A_[idx] = x;
    }
}

// ===================== ARQ mechanisms =====================








// ===================== BHO mechanisms =====================




// ===================== Operators (0..5) =====================

// ---------------- CoBiDE ----------------
void HDE::stepCobide()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    for (int i = 0; i < N; ++i) {
        std::vector<int> idx;
        sampleDistinctExcluding(N, 3, {i}, idx);
        if ((int)idx.size() < 3) continue;

        int r1 = idx[0], r2 = idx[1], r3 = idx[2];

        const Vec& x1 = X_[r1];
        const Vec& x2 = X_[r2];
        const Vec& x3 = X_[r3];

        double F = CBF_[i];
        Vec v(D);
        for (int j = 0; j < D; ++j)
            v[j] = x1[j] + F * (x2[j] - x3[j]);
        ensureBounds(v);

        Vec y = X_[i];
        double CR = CBCR_[i];
        int jrand = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jrand)
                y[j] = v[j];
        }
        ensureBounds(y);
        double fy = eval(y);

        Q[i]  = std::move(y);
        QF[i] = fy;
        if (prob_->calls() >= max_evals_) break;
    }

    int suc = 0;
    for (int i = 0; i < N; ++i) {
        if (QF[i] <= FX_[i]) {
            X_[i]  = Q[i];
            FX_[i] = QF[i];
            ++suc;
            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        } else {
            double F;
            if (randU() < 0.5) F = cauchy(0.65, 0.1);
            else               F = cauchy(1.0, 0.1);
            while (F < 0.0) {
                if (randU() < 0.5) F = cauchy(0.65, 0.1);
                else               F = cauchy(1.0, 0.1);
            }
            if (F > 1.0) F = 1.0;
            CBF_[i] = F;

            double CR;
            if (randU() < 0.5) CR = cauchy(0.1, 0.1);
            else               CR = cauchy(0.95, 0.1);
            if (CR > 1.0) CR = 1.0;
            if (CR < 0.0) CR = 0.0;
            CBCR_[i] = CR;
        }
    }

    success_[0] += suc;
    ni_[0]      += (double)suc;
    sortByFitness();
}

// ---------------- IDE ----------------
void HDE::stepIDE()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    ++g_;
    double SRT = SRT_;
    double IDEps = CBps_;

    if (g_ == 1) {
        SRT = 0.0;
        SRT_ = 0.0;
    }
    if (g_ > 10 && g_ < 200) {
        SRT = 0.1;
    } else if (g_ >= 200 && g_ < 500) {
        SRT = 0.2;
    } else if (g_ >= 500 && g_ < 800) {
        SRT = 0.3;
    } else if (g_ >= 800) {
        SRT = 0.4;
    }
    SRT_ = SRT;

    sortByFitness();
    Vec xo = X_[0];

    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    for (int i = 0; i < N; ++i) {
        std::vector<int> idx;
        sampleDistinctExcluding(N, 3, {i}, idx);
        if ((int)idx.size() < 3) continue;
        int r1 = idx[0], r2 = idx[1], r3 = idx[2];

        const Vec* xr1ptr = nullptr;
        if (g_ <= gt_) {
            int high_ind_S = std::max(2, (int)std::round(IDEps * N));
            if (high_ind_S > N) high_ind_S = N;
            if (randU() < 0.5) {
                std::uniform_int_distribution<int> dist(0, high_ind_S - 1);
                int pick = dist(rng_);
                xr1ptr = &X_[pick];
            } else {
                xr1ptr = &X_[r1];
            }
        } else {
            int high_ind_S = std::max(2, (int)std::round(IDEps * N));
            if (high_ind_S > N) high_ind_S = N;
            if (randU() < 0.5) {
                std::uniform_int_distribution<int> dist(0, high_ind_S - 1);
                int pick = dist(rng_);
                xr1ptr = &X_[pick];
            } else {
                xr1ptr = &X_[r1];
            }
        }

        const Vec& xr1 = *xr1ptr;
        const Vec& xr2 = X_[r2];
        const Vec& xr3 = X_[r3];

        double Fo = CBF_[i];
        Vec v(D);
        if (g_ > gt_ && randU() < 0.5) {
            for (int j = 0; j < D; ++j)
                v[j] = X_[i][j] + Fo * (xr1[j] - xo[j]) + Fo * (xr2[j] - xr3[j]);
        } else {
            for (int j = 0; j < D; ++j)
                v[j] = xo[j] + Fo * (xr1[j] - xo[j]) + Fo * (xr2[j] - xr3[j]);
        }
        ensureBounds(v);
        Q[i] = v;
    }

    for (int i = 0; i < N; ++i) {
        Vec y = X_[i];
        const Vec& v = Q[i];
        double CR = CBCR_[i];
        int jrand = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jrand)
                y[j] = v[j];
        }
        ensureBounds(y);
        Q[i] = y;
    }

    for (int i = 0; i < N; ++i) {
        double fy = eval(Q[i]);
        QF[i] = fy;
        if (prob_->calls() >= max_evals_) break;
    }

    std::vector<int> indsucc;
    indsucc.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (QF[i] <= FX_[i]) indsucc.push_back(i);
    }

    int suc = (int)indsucc.size();
    success_[1] += suc;
    ni_[1]      += (double)suc;

    double SR = (N > 0) ? ((double)suc / (double)N) : 0.0;
    if (g_ < gt_) {
        if (SR <= SRT) ++Tcurr_;
        else Tcurr_ = 0;
        if (Tcurr_ >= T_) {
            gt_ = g_;
        }
    }

    for (int idx : indsucc) {
        X_[idx]  = Q[idx];
        FX_[idx] = QF[idx];
        if (FX_[idx] < best_f_) {
            best_f_ = FX_[idx];
            best_x_ = X_[idx];
        }
    }
    sortByFitness();
}

// ---------------- CMA-ES ----------------
void HDE::stepCMAES()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    int N = N_;
    if (N < 4) return;

    if (mu_ > N) {
        mu_ = std::max(1, N / 2);
        weights_.assign(mu_, 0.0);
        for (int i = 0; i < mu_; ++i) {
            double wi = std::log(mu_ + 0.5) - std::log((double)(i + 1));
            weights_[i] = wi;
        }
        double sw = 0.0;
        for (double w : weights_) sw += w;
        if (sw <= 0.0) sw = 1.0;
        for (double& w : weights_) w /= sw;
        double sw2 = 0.0;
        for (double w : weights_) sw2 += w * w;
        mueff_ = sw * sw / sw2;

        const double Dd = (double)D;
        cc_ = (4.0 + mueff_ / Dd) / (Dd + 4.0 + 2.0 * mueff_ / Dd);
        cs_ = (mueff_ + 2.0) / (Dd + mueff_ + 5.0);
        c1_ = 2.0 / (std::pow(Dd + 1.3, 2.0) + mueff_);
        cmu_= std::min(1.0 - c1_,
                       2.0 * (mueff_ - 2.0 + 1.0 / mueff_) /
                           (std::pow(Dd + 2.0, 2.0) + mueff_));
        damps_ = 1.0 + 2.0 *
            std::max(0.0, std::sqrt((mueff_ - 1.0) / (Dd + 1.0)) - 1.0) + cs_;
        chiN_ = std::sqrt(Dd) *
                (1.0 - 1.0 / (4.0 * Dd) + 1.0 / (21.0 * Dd * Dd));
    }

    sortByFitness();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    Vec xold(D, 0.0);
    for (int j = 0; j < D; ++j) {
        for (int i = 0; i < mu_ && i < N; ++i)
            xold[j] += X_[i][j] * weights_[i];
    }

    std::vector<Vec> Pop(N, Vec(D));
    std::vector<double> PopFit(N, std::numeric_limits<double>::infinity());
    for (int k = 0; k < N; ++k) {
        Vec z(D);
        for (int j = 0; j < D; ++j)
            z[j] = randN01();
        Vec y(D);
        for (int j = 0; j < D; ++j)
            y[j] = xold[j] + sigma_ * z[j];

        for (int j = 0; j < D; ++j) {
            if (y[j] < L[j]) y[j] = 0.5 * (oldPop_[k][j] + L[j]);
            if (y[j] > U[j]) y[j] = 0.5 * (oldPop_[k][j] + U[j]);
        }
        ensureBounds(y);
        double fy = eval(y);
        Pop[k] = std::move(y);
        PopFit[k] = fy;
        if (prob_->calls() >= max_evals_) break;
    }

    int bestIdx = 0;
    double bestFit = PopFit[0];
    for (int k = 1; k < N; ++k) {
        if (PopFit[k] < bestFit) {
            bestFit = PopFit[k];
            bestIdx = k;
        }
    }

    int worstIdx = 0;
    double worstFit = FX_[0];
    for (int k = 1; k < N; ++k) {
        if (FX_[k] > worstFit) {
            worstFit = FX_[k];
            worstIdx = k;
        }
    }

    if (bestFit < worstFit) {
        X_[worstIdx]  = Pop[bestIdx];
        FX_[worstIdx] = bestFit;
        success_[2] += 1;
        ni_[2]      += 1.0;
        if (bestFit < best_f_) {
            best_f_ = bestFit;
            best_x_ = X_[worstIdx];
        }
    }

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return PopFit[a] < PopFit[b]; });

    Vec xnew(D, 0.0);
    for (int j = 0; j < D; ++j) {
        for (int i = 0; i < mu_ && i < N; ++i)
            xnew[j] += Pop[idx[i]][j] * weights_[i];
    }

    Vec diff(D);
    for (int j = 0; j < D; ++j)
        diff[j] = (xnew[j] - xold[j]) / std::max(sigma_, 1e-12);

    for (int j = 0; j < D; ++j)
        ps_[j] = (1.0 - cs_) * ps_[j] +
                 std::sqrt(cs_ * (2.0 - cs_) * mueff_) * diff[j];

    double ps_norm_sq = 0.0;
    for (double val : ps_) ps_norm_sq += val * val;

    for (int j = 0; j < D; ++j)
        pc_[j] = (1.0 - cc_) * pc_[j] +
                 std::sqrt(cc_ * (2.0 - cc_) * mueff_) * diff[j];

    double ps_norm = std::sqrt(ps_norm_sq);
    sigma_ *= std::exp((cs_ / damps_) * (ps_norm / chiN_ - 1.0));
    if (!std::isfinite(sigma_) || sigma_ < 1e-12 || sigma_ > 1e6) {
        sigma_ = (U[0] - L[0]) / 2.0;
    }

    oldPop_ = Pop;
}

// ---------------- jSO ----------------
void HDE::stepJSO()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    double FES  = (double)prob_->calls();
    double maxF = (double)std::max<long long>(max_evals_, 1);

    std::vector<double> Fpole(N, -1.0);
    std::vector<double> CRpole(N, -1.0);
    std::vector<double> SCR;
    std::vector<double> SF;
    std::vector<double> deltaF(N, -1.0);
    int suc = 0;

    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    double pp = pmax_ - ((pmax_ - pmin_) * (FES / maxF));
    if (pp < pmin_) pp = pmin_;
    if (pp > pmax_) pp = pmax_;

    const auto& L = prob_->lb();

    std::vector<int> rank(N);
    std::iota(rank.begin(), rank.end(), 0);
    std::sort(rank.begin(), rank.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    int pNP = std::max(2, (int)std::round(pp * N));
    if (pNP > N) pNP = N;

    for (int i = 0; i < N; ++i) {
        int r = randInt(0, H_jso_ - 1);
        double MF_r = MF_[r];
        double MCR_r = MCR_[r];

        double F = cauchy(MF_r, 0.1);
        while (F <= 0.0) F = cauchy(MF_r, 0.1);
        if (F > 1.0) F = 1.0;

        double CR = randN01() * 0.1 + MCR_r;
        if (CR > 1.0) CR = 1.0;
        if (CR < 0.0) CR = 0.0;

        int pbestIndex = rank[randInt(0, pNP - 1)];
        const Vec& xpbest = X_[pbestIndex];

        std::vector<int> idx;
        sampleDistinctExcluding(N, 2, {i, pbestIndex}, idx);
        if ((int)idx.size() < 2) continue;
        int r1 = idx[0], r2 = idx[1];

        bool useA = (Asize_ > 0 && randU() < 0.5);
        Vec xr2 = X_[r2];
        if (useA && !A_.empty()) {
            int ai = randInt(0, Asize_ - 1);
            xr2 = A_[ai];
        }

        Vec v(D);
        for (int j = 0; j < D; ++j)
            v[j] = X_[i][j] + F * (xpbest[j] - X_[i][j]) +
                   F * (X_[r1][j] - xr2[j]);

        Vec y = X_[i];
        int jrand = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jrand)
                y[j] = v[j];
        }

        // Optional eigen crossover
        if (randU() < peig_ && D > 1) {
            std::vector<double> mean(D, 0.0);
            for (int j = 0; j < D; ++j) mean[j] = 0.5 * (X_[i][j] + y[j]);

            std::vector<double> z(D);
            for (int j = 0; j < D; ++j) z[j] = randN01();

            for (int j = 0; j < D; ++j)
                y[j] = mean[j] + 0.5 * z[j] * (y[j] - X_[i][j]);
        }

        for (int j = 0; j < D; ++j) {
            if (y[j] < L[j]) y[j] = 0.5 * (X_[i][j] + L[j]);
        }
        ensureBounds(y);

        double fy = eval(y);
        Q[i]  = y;
        QF[i] = fy;
        Fpole[i]  = F;
        CRpole[i] = CR;

        if (prob_->calls() >= max_evals_) break;
    }

    for (int i = 0; i < N; ++i) {
        if (QF[i] <= FX_[i]) {
            addToArchive(X_[i]);
            deltaF[i] = std::abs(FX_[i] - QF[i]);
            X_[i]  = Q[i];
            FX_[i] = QF[i];
            ++suc;
            if (QF[i] < best_f_) {
                best_f_ = QF[i];
                best_x_ = Q[i];
            }
        }
    }

    if (suc > 0) {
        for (int i = 0; i < N; ++i) {
            if (deltaF[i] > 0.0) {
                SF.push_back(Fpole[i]);
                SCR.push_back(CRpole[i]);
            }
        }

        if (!SF.empty()) {
            double sum_dF = 0.0;
            for (int i = 0; i < N; ++i) if (deltaF[i] > 0.0) sum_dF += deltaF[i];
            if (sum_dF <= 0.0) sum_dF = 1.0;

            double num = 0.0, den = 0.0;
            double newMCR = 0.0;

            int k = 0;
            for (int i = 0; i < N; ++i) {
                if (deltaF[i] > 0.0) {
                    double w = deltaF[i] / sum_dF;
                    double Fv = SF[k];
                    double CRv= SCR[k];
                    num += w * Fv * Fv;
                    den += w * Fv;
                    newMCR += w * CRv;
                    ++k;
                }
            }

            double newMF = (den > 0.0) ? (num / den) : MF_[k_mem_];
            if (newMF > 1.0) newMF = 1.0;
            if (newMF < 0.0) newMF = 0.0;

            MF_[k_mem_]  = newMF;
            MCR_[k_mem_] = newMCR;
            k_mem_++;
            if (k_mem_ >= H_jso_) k_mem_ = 0;
        }
    }

    success_[3] += suc;
    ni_[3]      += (double)suc;
    sortByFitness();
}

// ===================== main loop =====================

void HDE::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    int N = N_;

    // Linear population size reduction
    double evals_now = (double)prob_->calls();
    double maxE      = (double)std::max<long long>(max_evals_, 1);
    int targetN = (int)std::round(pop_init_ -
                                  (pop_init_ - pop_min_) * (evals_now / maxE));
    if (targetN < pop_min_) targetN = pop_min_;
    if (targetN < N) {
        shrinkPopulation(targetN);
        N = N_;
    }

    // Roulette-based heuristic selection
    auto sel = rouletteSelect();
    int    hh   = sel.first;
    double pmin = sel.second;

    if (pmin < delta_) {
        for (int i = 0; i < h_; ++i)
            cni_[i] += ni_[i] - (double)n0_;
        ni_.assign(h_, (double)n0_);
        ++nrst_;
    }

    // IMPORTANT:
    // - The default operator must be jSO.
    // - jSO must not appear as an explicit case; when hh == 3 (the jSO arm), it must fall through to default.
    // - ARQ and BHO are not implemented as separate step methods. Their mechanisms are executed directly in the
    //   corresponding cases so the hybrid operator remains single-dispatch.
    switch (hh) {
        case 0: stepCobide(); break;
        case 1: stepIDE();    break;
        case 2: stepCMAES();  break;

        // Case 4: mechanisms (selection + RTR replacement + adaptive (muF, muCR) update)
        case 4: {
            const int D = prob_->dimension();
            const int Ncur = N_;
            if (Ncur < 4) break;

            // Local helpers (inlined; no separate ARQ/BHO step methods)
            auto archivePush = [&](const Vec& x) {
                A_arq_.push_back(x);
            };
            auto archiveTrim = [&](int N) {
                const int cap = std::max(1, (int)std::floor(arq_archive_rate_ * (double)N));
                if ((int)A_arq_.size() <= cap) return;
                std::shuffle(A_arq_.begin(), A_arq_.end(), rng_);
                A_arq_.resize(cap);
            };
            auto distBN = [&](const Vec& a, const Vec& b) -> double {
                const auto& L = prob_->lb();
                const auto& U = prob_->ub();
                double s = 0.0;
                for (size_t j = 0; j < a.size(); ++j) {
                    double lo = (j < L.size() ? L[j] : -1.0);
                    double hi = (j < U.size() ? U[j] :  1.0);
                    if (lo > hi) std::swap(lo, hi);
                    double denom = (hi - lo);
                    double z = (denom > 0.0) ? ((a[j] - b[j]) / denom) : (a[j] - b[j]);
                    s += z * z;
                }
                return std::sqrt(s);
            };
            auto sampleFCR = [&](double& F, double& CR) {
                std::cauchy_distribution<double> cF(arq_muF_, 0.1);
                std::normal_distribution<double> nCR(arq_muCR_, 0.1);

                F = -1.0;
                for (int tries = 0; tries < 50; ++tries) {
                    F = cF(rng_);
                    if (F > 0.0) break;
                }
                if (F <= 0.0) F = arq_muF_;
                if (F < arq_Flo_) F = arq_Flo_;
                if (F > arq_Fhi_) F = arq_Fhi_;

                CR = nCR(rng_);
                if (CR < 0.0) CR = 0.0;
                if (CR > 1.0) CR = 1.0;
            };
            auto updateMuFromSuccess = [&](const std::vector<double>& SF,
                                         const std::vector<double>& SCR,
                                         const std::vector<double>& SG) {
                if (SG.empty()) return;
                double sumG = std::accumulate(SG.begin(), SG.end(), 0.0);
                if (!(sumG > 0.0)) return;

                std::vector<double> w(SG.size());
                for (size_t i = 0; i < w.size(); ++i) w[i] = SG[i] / sumG;

                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < SF.size(); ++i) {
                    num += w[i] * SF[i] * SF[i];
                    den += w[i] * SF[i];
                }
                double newMuF  = (den > 0.0) ? (num / den) : arq_muF_;

                double newMuCR = 0.0;
                for (size_t i = 0; i < SCR.size(); ++i) newMuCR += w[i] * SCR[i];

                arq_muF_  = (1.0 - arq_shc_) * arq_muF_  + arq_shc_ * newMuF;
                arq_muCR_ = (1.0 - arq_shc_) * arq_muCR_ + arq_shc_ * newMuCR;

                if (arq_muF_ < arq_Flo_) arq_muF_ = arq_Flo_;
                if (arq_muF_ > arq_Fhi_) arq_muF_ = arq_Fhi_;
                if (arq_muCR_ < 0.0) arq_muCR_ = 0.0;
                if (arq_muCR_ > 1.0) arq_muCR_ = 1.0;
            };
            auto makeTrial = [&](int i, const std::vector<int>& ord, double F, double CR, Vec& u) {
                const int N = N_;

                int pcount = std::max(2, (int)std::ceil(arq_pbest_ * (double)N));
                if (pcount > N) pcount = N;
                std::uniform_int_distribution<int> Ip(0, pcount - 1);
                int ipbest = ord[Ip(rng_)];
                const Vec& xpbest = X_[ipbest];

                int r1 = randInt(0, N - 1);
                while (r1 == i) r1 = randInt(0, N - 1);

                bool useA = (!A_arq_.empty()) && (randU() < 0.5);

                Vec r2v(D, 0.0);
                if (useA) {
                    std::uniform_int_distribution<int> Ia(0, (int)A_arq_.size() - 1);
                    r2v = A_arq_[Ia(rng_)];
                } else {
                    int r2 = randInt(0, N - 1);
                    while (r2 == i || r2 == r1) r2 = randInt(0, N - 1);
                    r2v = X_[r2];
                }

                Vec v(D, 0.0);
                for (int j = 0; j < D; ++j)
                    v[j] = X_[i][j] + F * (xpbest[j] - X_[i][j]) + F * (X_[r1][j] - r2v[j]);

                u = X_[i];
                int jr = randInt(0, std::max(0, D - 1));
                for (int j = 0; j < D; ++j) {
                    if (randU() < CR || j == jr) u[j] = v[j];
                }
                ensureBounds(u);
            };
            auto selectionRTR = [&](int parentIndex, const Vec& u, double fu,
                                  double F, double CR,
                                  std::vector<double>& SF,
                                  std::vector<double>& SCR,
                                  std::vector<double>& SG) -> bool {
                if (fu < FX_[parentIndex]) {
                    double gain = FX_[parentIndex] - fu;
                    archivePush(X_[parentIndex]);
                    X_[parentIndex]  = u;
                    FX_[parentIndex] = fu;

                    SF.push_back(F);
                    SCR.push_back(CR);
                    SG.push_back(gain);
                    return true;
                }

                int qstar = -1;
                double bestD = std::numeric_limits<double>::infinity();
                std::uniform_int_distribution<int> I(0, N_ - 1);

                for (int k = 0; k < arq_rtr_pool_; ++k) {
                    int q = I(rng_);
                    double d = distBN(u, X_[q]);
                    if (d < bestD) { bestD = d; qstar = q; }
                }
                if (qstar < 0) return false;

                if (fu < FX_[qstar]) {
                    double gain = FX_[qstar] - fu;
                    archivePush(X_[qstar]);
                    X_[qstar]  = u;
                    FX_[qstar] = fu;

                    SF.push_back(F);
                    SCR.push_back(CR);
                    SG.push_back(gain);
                    return true;
                }

                return false;
            };

            archiveTrim(Ncur);

            std::vector<int> ord(Ncur);
            std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

            int m = std::max(1, (int)std::ceil(arq_agent_fraction_ * (double)Ncur));
            if (m > Ncur) m = Ncur;

            std::vector<int> idx = ord;
            std::shuffle(idx.begin(), idx.end(), rng_);
            idx.resize(m);

            std::vector<double> SF, SCR, SG;
            SF.reserve(m);
            SCR.reserve(m);
            SG.reserve(m);

            for (int t = 0; t < m && prob_->calls() < max_evals_; ++t) {
                int i = idx[t];

                double F, CR;
                sampleFCR(F, CR);

                Vec u(D, 0.0);
                makeTrial(i, ord, F, CR, u);

                double fu = eval(u);
                selectionRTR(i, u, fu, F, CR, SF, SCR, SG);

                if (prob_->calls() >= max_evals_) break;
            }

            updateMuFromSuccess(SF, SCR, SG);
            archiveTrim(Ncur);

            const int suc = (int)SG.size();
            success_[4] += suc;
            ni_[4]      += (double)suc;

            for (int i = 0; i < N_; ++i) {
                if (FX_[i] < best_f_) {
                    best_f_ = FX_[i];
                    best_x_ = X_[i];
                }
            }

            sortByFitness();
            break;
        }

        // Case 5: mechanisms (healing, wound exploration, and stagnation handling)
        case 5: {
            const int D = prob_->dimension();
            const int Ncur = N_;
            if (Ncur < 2) break;

            ++bho_iters_;

            int elite = 0;
            double fbest = FX_[0];
            for (int i = 1; i < Ncur; ++i)
                if (FX_[i] < fbest) { fbest = FX_[i]; elite = i; }

            if (fbest < best_f_) {
                best_f_ = fbest;
                best_x_ = X_[elite];
                bho_sinceBest_ = 0;
            } else {
                ++bho_sinceBest_;
            }

            double evalFrac = (max_evals_ > 0)
                ? std::min(1.0, (double)prob_->calls() / (double)max_evals_)
                : 0.0;
            double wound = std::max(0.05 * bho_wound_strength_init_,
                                    bho_wound_strength_init_ * (1.0 - evalFrac));

            int suc = 0;
            for (int i = 0; i < Ncur && prob_->calls() < max_evals_; ++i) {
                if (i == elite) continue;

                Vec y = X_[i];
                if (randU() < bho_heal_prob_) {
                    for (int j = 0; j < D; ++j) {
                        double step = bho_heal_rate_ * (best_x_[j] - X_[i][j]) +
                                      bho_elite_kick_sigma_ * randN01();
                        y[j] = X_[i][j] + step;
                    }
                } else {
                    for (int j = 0; j < D; ++j)
                        y[j] = best_x_[j] + wound * randN01();
                }

                ensureBounds(y);
                double fy = eval(y);

                if (fy < FX_[i]) {
                    X_[i]  = y;
                    FX_[i] = fy;
                    ++suc;
                    if (fy < best_f_) {
                        best_f_ = fy;
                        best_x_ = y;
                        bho_sinceBest_ = 0;
                    }
                }
            }

            if (suc > 0) {
                success_[5] += suc;
                ni_[5]      += (double)suc;
            }

            auto eliteGaussianKick = [&]() {
                Vec y = X_[elite];
                for (int j = 0; j < D; ++j) y[j] += bho_elite_kick_sigma_ * randN01();
                ensureBounds(y);
                double fy = eval(y);
                if (fy < FX_[elite]) {
                    X_[elite]  = y;
                    FX_[elite] = fy;
                    if (fy < best_f_) {
                        best_f_ = fy;
                        best_x_ = y;
                        bho_sinceBest_ = 0;
                    }
                }
            };
            auto softKickPopulation = [&]() {
                int ksuc = 0;
                for (int i = 0; i < N_ && prob_->calls() < max_evals_; ++i) {
                    if (i == elite) continue;
                    Vec y = X_[i];
                    for (int j = 0; j < D; ++j) y[j] += bho_elite_kick_sigma_ * randN01();
                    ensureBounds(y);
                    double fy = eval(y);
                    if (fy < FX_[i]) {
                        X_[i]  = y;
                        FX_[i] = fy;
                        ++ksuc;
                        if (fy < best_f_) {
                            best_f_ = fy;
                            best_x_ = y;
                            bho_sinceBest_ = 0;
                        }
                    }
                }
                if (ksuc > 0) {
                    success_[5] += ksuc;
                    ni_[5]      += (double)ksuc;
                }
                bho_sinceBest_ = 0;
            };
            auto restartPartial = [&]() {
                int n_resample = std::max(1, (int)std::round(bho_restart_frac_ * (double)N_));
                const auto& L = prob_->lb();
                const auto& U = prob_->ub();
                int rsuc = 0;

                for (int k = 0; k < n_resample && prob_->calls() < max_evals_; ++k) {
                    int i = randInt(0, N_ - 1);
                    if (i == elite) i = (i + 1) % N_;

                    for (int j = 0; j < (int)X_[i].size(); ++j) {
                        double lo = (j < (int)L.size() ? L[j] : -1.0);
                        double hi = (j < (int)U.size() ? U[j] :  1.0);
                        if (lo > hi) std::swap(lo, hi);
                        X_[i][j] = lo + randU() * (hi - lo);
                    }

                    double fy = eval(X_[i]);
                    FX_[i] = fy;
                    ++rsuc;

                    if (fy < best_f_) {
                        best_f_ = fy;
                        best_x_ = X_[i];
                        bho_sinceBest_ = 0;
                    }
                }

                if (rsuc > 0) {
                    success_[5] += rsuc;
                    ni_[5]      += (double)rsuc;
                }
            };

            if (bho_sinceBest_ > bho_stagnation_kick_) {
                eliteGaussianKick();
                softKickPopulation();
            }
            if (bho_sinceBest_ > bho_stagnation_restart_) {
                restartPartial();
                bho_sinceBest_ = 0;
            }

            sortByFitness();
            break;
        }

        default: stepJSO(); break;
    }

    // Optional in-run local search (shared)
    if (!local_method_.empty() && local_rate_ > 0.0) {
        for (int i = 0; i < N_ && prob_->calls() < max_evals_; ++i) {
            if (randU() < local_rate_) {
                auto res = localSearch(local_method_, X_[i]);
                if (!res.first.empty() && std::isfinite(res.second) &&
                    res.second < FX_[i]) {
                    X_[i]  = std::move(res.first);
                    FX_[i] = res.second;
                    if (res.second < best_f_) {
                        best_f_ = res.second;
                        best_x_ = X_[i];
                    }
                }
            }
        }
    }

    updateStop(FX_);
    printBest();
}

void HDE::end()
{
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        if (!res.first.empty() && std::isfinite(res.second) &&
            res.second < best_f_) {
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