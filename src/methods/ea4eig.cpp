#include "ea4eig.h"
#include "init.h"
#include "options.h"

#include <numeric>
#include <cctype>
#include <cmath>

namespace optimsolution {

static constexpr double EA4EIG_PI = 3.14159265358979323846;

void EA4Eig::configure(const MethodConfig& mc)
{
    // population
    int p = mc.getInt("population", pop_init_);
    if (p > 3) {
        pop_init_ = p;
        Optimizer::setPopulation(pop_init_);
    }
    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;

    // cobide / jSO eigen controls
    CBps_ = mc.getDbl("cb_ps", CBps_);
    if (CBps_ <= 0.0 || CBps_ >= 1.0) CBps_ = 0.5;

    peig_ = mc.getDbl("peig", peig_);
    if (peig_ < 0.0) peig_ = 0.0;
    if (peig_ > 1.0) peig_ = 0.4;

    // in–run local search
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // Final local refinement (if not present in the block, the global value is kept)
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

void EA4Eig::init()
{
    if (!prob_) return ;
    const int D = prob_->dimension();

    Optimizer::setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
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

    // roulette
    h_      = 4;
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

    // cobide parameters per individual
    CBF_.assign(N_, 0.0);
    CBCR_.assign(N_, 0.0);
    for (int i = 0; i < N_; ++i) {
        double F;
        if (randU() < 0.5)
            F = cauchy(0.65, 0.1);
        else
            F = cauchy(1.0, 0.1);
        while (F < 0.0) {
            if (randU() < 0.5)
                F = cauchy(0.65, 0.1);
            else
                F = cauchy(1.0, 0.1);
        }
        if (F > 1.0) F = 1.0;
        CBF_[i] = F;

        double CR;
        if (randU() < 0.5)
            CR = cauchy(0.1, 0.1);
        else
            CR = cauchy(0.95, 0.1);
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

    oldPop_.clear();
    oldPop_.resize(N_);
    for (int i = 0; i < N_; ++i)
        oldPop_[i] = X_[i];

    // jSO memory / archive
    Asize_max_ = (int)std::round(N_ * 1.4);
    if (Asize_max_ < 0) Asize_max_ = 0;
    Asize_     = 0;
    A_.clear();
    H_jso_ = 5;
    MF_.assign(H_jso_, 0.3);
    MCR_.assign(H_jso_, 0.8);
    MF_.back()  = 0.9;
    MCR_.back() = 0.9;
    k_mem_ = 0;
    pmax_  = 0.25;
    pmin_  = 0.05;

    updateStop(FX_);
    printBest();
}

void EA4Eig::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D   = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j]))
            x[j] = 0.5 * (L[j] + U[j]);
        // mirror (zrcad)
        while (x[j] < L[j] || x[j] > U[j]) {
            if (x[j] > U[j])
                x[j] = 2.0 * U[j] - x[j];
            else if (x[j] < L[j])
                x[j] = 2.0 * L[j] - x[j];
        }
    }
}

int EA4Eig::randInt(int lo, int hi)
{
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double EA4Eig::randU()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double EA4Eig::randN01()
{
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double EA4Eig::cauchy(double loc, double scale)
{
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

void EA4Eig::sampleDistinct(int N, int k, std::vector<int>& out)
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

void EA4Eig::sampleDistinctExcluding(int N, int k,
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

std::pair<int,double> EA4Eig::rouletteSelect() const
{
    const int h = h_;
    if (h <= 0) return {0, 0.0};

    double sumni = 0.0;
    for (int i = 0; i < h; ++i) sumni += ni_[i];

    auto* self = const_cast<EA4Eig*>(this);

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

void EA4Eig::sortByFitness()
{
    const int N = N_;
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });
    std::vector<Vec> newX(N);
    std::vector<double> newF(N);
    for (int i = 0; i < N; ++i) {
        newX[i] = std::move(X_[idx[i]]);
        newF[i] = FX_[idx[i]];
    }
    X_.swap(newX);
    FX_.swap(newF);
}

void EA4Eig::shrinkPopulation(int newN)
{
    if (newN >= N_) return;
    if (newN < pop_min_) newN = pop_min_;
    if (newN >= N_) return;
    sortByFitness();
    X_.resize(newN);
    FX_.resize(newN);
    N_ = newN;
}

void EA4Eig::addToArchive(const Vec& x)
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

// ---------------- cobide ----------------
void EA4Eig::stepCobide()
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

    for (int i = 0; i < N; ++i) {
        if (QF[i] <= FX_[i]) {
            X_[i]  = Q[i];
            FX_[i] = QF[i];
            success_[0] += 1;
            ni_[0] += 1.0;
            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        } else {
            double F;
            if (randU() < 0.5)
                F = cauchy(0.65, 0.1);
            else
                F = cauchy(1.0, 0.1);
            while (F < 0.0) {
                if (randU() < 0.5)
                    F = cauchy(0.65, 0.1);
                else
                    F = cauchy(1.0, 0.1);
            }
            if (F > 1.0) F = 1.0;
            CBF_[i] = F;

            double CR;
            if (randU() < 0.5)
                CR = cauchy(0.1, 0.1);
            else
                CR = cauchy(0.95, 0.1);
            if (CR > 1.0) CR = 1.0;
            if (CR < 0.0) CR = 0.0;
            CBCR_[i] = CR;
        }
    }
}

// --------------- IDE -----------------
void EA4Eig::stepIDE()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    ++g_;
    if (g_ > gmax_) g_ = gmax_;

    sortByFitness();
    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    double IDEps = 0.1 + 0.9 * std::pow(10.0, 5.0 * ((double)g_ / (double)gmax_ - 1.0));
    double SRT = (g_ < gt_) ? 0.0 : 0.1;

    for (int i = 0; i < N; ++i) {
        std::vector<int> idx;
        sampleDistinctExcluding(N, 4, {i}, idx);
        if ((int)idx.size() < 4) continue;
        int o  = idx[0];
        int r1 = idx[1];
        int r2 = idx[2];
        int r3 = idx[3];

        const Vec& xo = X_[o];

        const Vec* xr1ptr = nullptr;

        if (g_ <= gt_) {
            double probSup = 0.9 * IDEps;
            if (randU() < probSup) {
                int high_ind_S = std::max(2, (int)std::round(IDEps * N));
                if (high_ind_S > N) high_ind_S = N;
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
    for (int i = 0; i < N; ++i) {
        if (QF[i] <= FX_[i]) {
            indsucc.push_back(i);
        }
    }
    success_[1] += (int)indsucc.size();
    ni_[1]      += (double)indsucc.size();

    double SR = (N > 0) ? ((double)indsucc.size() / (double)N) : 0.0;
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

// --------------- CMA-ES (simple, with dynamic mu_) ---------------
void EA4Eig::stepCMAES()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    int N = N_;
    if (N < 4) return;

    // If the population has been reduced, mu_ and weights_ are adjusted
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

// --------------- jSO -----------------
void EA4Eig::stepJSO()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));

    auto trimArchive = [&](int max_size) {
        if (max_size < 0) max_size = 0;
        while ((int)A_.size() > max_size) {
            int idx = randInt(0, (int)A_.size() - 1);
            A_.erase(A_.begin() + idx);
        }
        Asize_ = (int)A_.size();
        Asize_max_ = max_size;
    };

    auto isInBounds = [&](const Vec& x) -> bool {
        if ((int)x.size() != D) return false;
        for (int j = 0; j < D; ++j) {
            if (!std::isfinite(x[j])) return false;
            if (x[j] < L[j] || x[j] > U[j]) return false;
        }
        return true;
    };

    auto clampInBounds = [&](Vec& x) {
        for (int j = 0; j < D; ++j) {
            if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
            if (x[j] < L[j]) x[j] = L[j];
            if (x[j] > U[j]) x[j] = U[j];
        }
    };

    trimArchive((int)std::round(1.4 * N));

    double nfes = static_cast<double>(prob_->calls());
    double fes_ratio = nfes / maxFES;
    if (fes_ratio < 0.0) fes_ratio = 0.0;
    if (fes_ratio > 1.0) fes_ratio = 1.0;

    double p = pmax_ - (pmax_ - pmin_) * fes_ratio;
    if (p < pmin_) p = pmin_;
    if (p > pmax_) p = pmax_;

    int p_best_size = (int)std::round(p * N);
    if (p_best_size < 2) p_best_size = 2;
    if (p_best_size > N) p_best_size = N;

    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> dF;
    SF.reserve(N);
    SCR.reserve(N);
    dF.reserve(N);

    int strict_successes = 0;

    std::uniform_int_distribution<int> Ui_dim(0, D - 1);
    std::uniform_int_distribution<int> Ui_mem(0, H_jso_ - 1);

    auto samplePopIndex = [&](int avoid1, int avoid2, int avoid3) {
        int idx = 0;
        do {
            idx = randInt(0, N - 1);
        } while (idx == avoid1 || idx == avoid2 || idx == avoid3);
        return idx;
    };

    auto sampleR2 = [&](int avoid1, int avoid2, int avoid3, Vec& xr2) -> bool {
        const int total = N + (int)A_.size();
        if (total <= 0) return false;

        for (int tries = 0; tries < 128; ++tries) {
            int pick = randInt(0, total - 1);
            if (pick < N) {
                if (pick == avoid1 || pick == avoid2 || pick == avoid3) continue;
                xr2 = X_[pick];
                return true;
            }
            xr2 = A_[pick - N];
            return true;
        }

        int r2 = samplePopIndex(avoid1, avoid2, avoid3);
        xr2 = X_[r2];
        return true;
    };

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        const int r_mem = Ui_mem(rng_);
        const double muF  = (H_jso_ > 0 && r_mem == H_jso_ - 1) ? 0.9 : MF_[r_mem];
        const double muCR = (H_jso_ > 0 && r_mem == H_jso_ - 1) ? 0.9 : MCR_[r_mem];

        double Fi = -1.0;
        do {
            Fi = cauchy(muF, 0.1);
        } while (Fi <= 0.0);
        if (Fi > 1.0) Fi = 1.0;

        double CRi = muCR + 0.1 * randN01();
        if (CRi < 0.0) CRi = 0.0;
        if (CRi > 1.0) CRi = 1.0;

        const double g_ratio = static_cast<double>(prob_->calls()) / maxFES;
        if (g_ratio < 0.25) {
            if (CRi < 0.7) CRi = 0.7;
        } else if (g_ratio < 0.5) {
            if (CRi < 0.6) CRi = 0.6;
        }
        if (g_ratio < 0.6 && Fi > 0.7) {
            Fi = 0.7;
        }

        double Fw;
        if (g_ratio < 0.2) {
            Fw = 0.7 * Fi;
        } else if (g_ratio < 0.4) {
            Fw = 0.8 * Fi;
        } else {
            Fw = 1.2 * Fi;
        }

        const Vec& xi = X_[i];

        Vec ui(D);
        bool feasible = false;
        for (int repeat = 0; repeat < 64 && !feasible; ++repeat) {
            int p_pick = randInt(0, p_best_size - 1);
            int p_idx = idx_sorted[p_pick];
            const Vec& xp = X_[p_idx];

            int r1 = samplePopIndex(i, p_idx, -1);
            const Vec& xr1 = X_[r1];

            Vec xr2;
            if (!sampleR2(i, p_idx, r1, xr2)) {
                break;
            }

            int jrand = Ui_dim(rng_);
            for (int j = 0; j < D; ++j) {
                if (randU() < CRi || j == jrand) {
                    ui[j] = xi[j]
                          + Fw * (xp[j] - xi[j])
                          + Fi * (xr1[j] - xr2[j]);
                } else {
                    ui[j] = xi[j];
                }
            }

            feasible = isInBounds(ui);
        }

        if (!feasible) {
            clampInBounds(ui);
        }

        const double f_old = FX_[i];
        const double f_new = eval(ui);
        const bool accepted = (f_new <= f_old);

        if (accepted) {
            newPop[i] = ui;
            newFit[i] = f_new;

            if (f_new < f_old) {
                ++strict_successes;
                A_.push_back(xi);
                SF.push_back(Fi);
                SCR.push_back(CRi);
                dF.push_back(f_old - f_new);
            }

            if (f_new < best_f_) {
                best_f_ = f_new;
                best_x_ = newPop[i];
            }
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);
    trimArchive((int)std::round(1.4 * (int)X_.size()));

    if (!SF.empty()) {
        double sum_dF = 0.0;
        for (double v : dF) sum_dF += v;
        if (sum_dF <= 0.0) sum_dF = 1.0;

        double sum_wF = 0.0;
        double sum_wF2 = 0.0;
        double sum_wCR = 0.0;
        double sum_w = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
            const double wk = dF[k] / sum_dF;
            const double Fk = SF[k];
            const double CRk = SCR[k];
            sum_w += wk;
            sum_wF += wk * Fk;
            sum_wF2 += wk * Fk * Fk;
            sum_wCR += wk * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0 && H_jso_ > 1) {
            const double meanF_Lehmer = sum_wF2 / sum_wF;
            const double meanCR = sum_wCR / sum_w;

            MF_[k_mem_]  = 0.5 * (MF_[k_mem_]  + meanF_Lehmer);
            MCR_[k_mem_] = 0.5 * (MCR_[k_mem_] + meanCR);

            ++k_mem_;
            if (k_mem_ >= H_jso_ - 1) k_mem_ = 0;
        }
    }

    if (H_jso_ > 0) {
        MF_[H_jso_ - 1]  = 0.9;
        MCR_[H_jso_ - 1] = 0.9;
    }

    success_[3] += strict_successes;
    ni_[3]      += (double)strict_successes;
}

// --------------- main iteration ---------------
void EA4Eig::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    int N = N_;

    // linear population size reduction
    double evals_now = (double)prob_->calls();
    double maxE      = (double)std::max<long long>(max_evals_, 1);
    int targetN = (int)std::round(pop_init_ -
                                  (pop_init_ - pop_min_) * (evals_now / maxE));
    if (targetN < pop_min_) targetN = pop_min_;
    if (targetN < N) {
        shrinkPopulation(targetN);
        N = N_;
    }

    // roulette-based heuristic selection
    auto sel = rouletteSelect();
    int    hh   = sel.first;
    double pmin = sel.second;

    if (pmin < delta_) {
        for (int i = 0; i < h_; ++i)
            cni_[i] += ni_[i] - (double)n0_;
        ni_.assign(h_, (double)n0_);
        ++nrst_;
    }

    switch (hh) {
        case 0: stepCobide(); break;
        case 1: stepIDE();    break;
        case 2: stepCMAES();  break;
        case 3:
        default: stepJSO();   break;
    }

    // optional in–run local search
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

void EA4Eig::end()
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
