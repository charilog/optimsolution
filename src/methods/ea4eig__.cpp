#include "ea4eig.h"
#include "init.h"
#include "options.h"

#include <cfloat>
#include <cmath>
#include <limits>

namespace optimsolution {

using Eigen::MatrixXd;
using Eigen::VectorXd;

static constexpr double EA4EIG_PI = 3.14159265358979323846;

// ========================= small helpers =========================

double EA4Eig::randU() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double EA4Eig::randN01() {
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

int EA4Eig::randInt(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double EA4Eig::cauchyRnd(double x0, double gamma) {
    std::cauchy_distribution<double> dist(x0, gamma);
    return dist(rng_);
}

// mirror (zrcad) into [lb, ub]
void EA4Eig::ensureBounds(Vec& y) {
    const int D = (int)y.size();
    for (int j = 0; j < D; ++j) {
        double a = lb_[j];
        double b = ub_[j];
        if (!std::isfinite(y[j]))
            y[j] = 0.5 * (a + b);
        while (y[j] < a || y[j] > b) {
            if (y[j] > b)
                y[j] = 2.0 * b - y[j];
            else if (y[j] < a)
                y[j] = 2.0 * a - y[j];
        }
    }
}

// random sample k of N without repetition
void EA4Eig::sampleDistinct(int N, int k, std::vector<int>& out) {
    out.clear();
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

// random sample k of N without repetition, excluding expt
void EA4Eig::sampleDistinctExcluding(int N, int k,
                                     const std::vector<int>& exclude,
                                     std::vector<int>& out) {
    out.clear();
    if (N <= 0 || k <= 0) return;
    std::vector<int> cand;
    cand.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::find(exclude.begin(), exclude.end(), i) == exclude.end())
            cand.push_back(i);
    }
    if ((int)cand.size() <= k) {
        out = cand;
        return;
    }
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)cand.size() - 1);
        int r = dist(rng_);
        std::swap(cand[i], cand[r]);
        out.push_back(cand[i]);
    }
}

// roulette selection
std::pair<int,double> EA4Eig::rouletteSelect() const {
    int h = H_;
    if (h <= 0) return {0, 0.0};

    auto self = const_cast<EA4Eig*>(this);

    double ss = 0.0;
    for (int i = 0; i < h; ++i) ss += ni_[i];
    if (ss <= 0.0) {
        int idx = self->randInt(0, h - 1);
        return {idx, 1.0 / h};
    }

    double p_min = std::numeric_limits<double>::infinity();
    std::vector<double> cp(h);
    double acc = 0.0;
    for (int i = 0; i < h; ++i) {
        double p = ni_[i] / ss;
        if (p < p_min) p_min = p;
        acc += p;
        cp[i] = acc;
    }
    for (int i = 0; i < h; ++i)
        cp[i] /= ss;

    double r = self->randU();
    for (int i = 0; i < h; ++i) {
        if (r <= cp[i]) return {i, p_min};
    }
    return {h - 1, p_min};
}

void EA4Eig::sortByFitness() {
    int N = (int)X_.size();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });
    std::vector<Vec> Xnew(N);
    std::vector<double> Fnew(N);
    for (int i = 0; i < N; ++i) {
        Xnew[i] = std::move(X_[idx[i]]);
        Fnew[i] = FX_[idx[i]];
    }
    X_.swap(Xnew);
    FX_.swap(Fnew);

    if ((int)CBF_.size() == N) {
        std::vector<double> nCBF(N), nCBCR(N);
        for (int i = 0; i < N; ++i) {
            nCBF[i]  = CBF_[idx[i]];
            nCBCR[i] = CBCR_[idx[i]];
        }
        CBF_.swap(nCBF);
        CBCR_.swap(nCBCR);
    }
}

// linear population size reduction
void EA4Eig::shrinkPopulationTo(int newN) {
    if (newN >= N_) return;
    if (newN < N_min_) newN = N_min_;
    if (newN >= N_) return;

    sortByFitness();
    X_.resize(newN);
    FX_.resize(newN);
    CBF_.resize(newN);
    CBCR_.resize(newN);
    N_ = newN;

    Asize_max_ = (int)std::round(N_ * 2.6);
    while (Asize_ > Asize_max_ && !A_.empty()) {
        int idx = randInt(0, Asize_ - 1);
        A_.erase(A_.begin() + idx);
        --Asize_;
    }

    if (mu_ > N_) {
        mu_ = std::max(1, N_ / 2);
        weights_.assign(mu_, 0.0);
        for (int i = 0; i < mu_; ++i) {
            double wi = std::log(mu_ + 0.5) - std::log((double)(i + 1));
            weights_[i] = wi;
        }
        double sw = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        if (sw <= 0.0) sw = 1.0;
        for (double& w : weights_) w /= sw;
        double sw2 = 0.0;
        for (double w : weights_) sw2 += w * w;
        mueff_ = (sw * sw) / std::max(sw2, 1e-16);
    }
}

void EA4Eig::addToArchive(const Vec& x) {
    if (Asize_max_ <= 0) return;
    if (Asize_ < Asize_max_) {
        A_.push_back(x);
        ++Asize_;
    } else {
        int idx = randInt(0, Asize_max_ - 1);
        A_[idx] = x;
    }
}

// eig(cov(elite)) για Eigen-crossover
bool EA4Eig::computeEigenVectorsFromElite(const std::vector<int>& eliteIdx,
                                          MatrixXd& eigVecs) {
    int m = (int)eliteIdx.size();
    if (m < 2) return false;
    int D = (int)lb_.size();
    MatrixXd M(m, D);
    for (int i = 0; i < m; ++i) {
        const Vec& x = X_[eliteIdx[i]];
        for (int j = 0; j < D; ++j)
            M(i, j) = x[j];
    }
    MatrixXd centered = M.rowwise() - M.colwise().mean();
    MatrixXd C = (centered.transpose() * centered) / double(m - 1);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(C);
    if (es.info() != Eigen::Success)
        return false;
    eigVecs = es.eigenvectors();
    return true;
}

// ========================= configure =========================

void EA4Eig::configure(const MethodConfig& mc) {
    int pop = mc.getInt("population", N_init_);
    if (pop > 3) {
        N_init_ = pop;
        Optimizer::setPopulation(N_init_);
    }
    N_min_ = mc.getInt("np_min", N_min_);
    if (N_min_ < 4) N_min_ = 4;

    CBps_ = mc.getDbl("cb_ps", CBps_);
    if (CBps_ <= 0.0 || CBps_ >= 1.0) CBps_ = 0.5;

    peig_ = mc.getDbl("peig", peig_);
    if (peig_ < 0.0) peig_ = 0.0;
    if (peig_ > 1.0) peig_ = 0.4;

    local_method_ = mc.getStr("local_method", local_method_);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

// ========================= init =========================

void EA4Eig::init() {
    if (!prob_) return;

    const int D = prob_->dimension();
    lb_ = prob_->lb();
    ub_ = prob_->ub();

    Initializer sampler;
    sampler.configure(initopt_);
    X_ = sampler.samplePopulation(*prob_, rng_, N_init_);
    N_ = (int)X_.size();
    if (N_ < 4) return;

    FX_.assign(N_, std::numeric_limits<double>::infinity());
    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N_; ++i) {
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    ni_.assign(H_, (double)n0_);
    cni_.assign(H_, 0.0);
    success_.assign(H_, 0);
    delta_ = 1.0 / (5.0 * H_);
    nrst_ = 0;

    g_     = 0;
    gmax_  = std::max(1, (int)std::round((double)max_evals_ /
                                         std::max(N_, 1)));
    T_     = gmax_ / 10.0;
    GT_    = gmax_ / 2;
    gt_    = GT_;
    Tcurr_ = 0;

    CBF_.assign(N_, 0.0);
    CBCR_.assign(N_, 0.0);
    for (int i = 0; i < N_; ++i) {
        double F;
        if (randU() < 0.5)
            F = cauchyRnd(0.65, 0.1);
        else
            F = cauchyRnd(1.0, 0.1);
        while (F < 0.0) {
            if (randU() < 0.5)
                F = cauchyRnd(0.65, 0.1);
            else
                F = cauchyRnd(1.0, 0.1);
        }
        if (F > 1.0) F = 1.0;
        CBF_[i] = F;

        double CR;
        if (randU() < 0.5)
            CR = cauchyRnd(0.1, 0.1);
        else
            CR = cauchyRnd(0.95, 0.1);
        if (CR > 1.0) CR = 1.0;
        if (CR < 0.0) CR = 0.0;
        CBCR_[i] = CR;
    }

    double range = 0.0;
    for (int j = 0; j < D; ++j)
        range = std::max(range, std::fabs(ub_[j] - lb_[j]));
    if (range <= 0.0) range = 1.0;
    sigma_ = range / 2.0;

    mu_ = N_ / 2;
    if (mu_ < 1) mu_ = 1;
    weights_.assign(mu_, 0.0);
    for (int i = 0; i < mu_; ++i) {
        double wi = std::log(mu_ + 0.5) - std::log((double)(i + 1));
        weights_[i] = wi;
    }
    double sw = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sw <= 0.0) sw = 1.0;
    for (double& w : weights_) w /= sw;
    double sw2 = 0.0;
    for (double w : weights_) sw2 += w * w;
    mueff_ = (sw * sw) / std::max(sw2, 1e-16);

    double DIMd = (double)D;
    cc_ = (4.0 + mueff_ / DIMd) / (DIMd + 4.0 + 2.0 * mueff_ / DIMd);
    cs_ = (mueff_ + 2.0) / (DIMd + mueff_ + 5.0);
    c1_ = 2.0 / (std::pow(DIMd + 1.3, 2.0) + mueff_);
    cmu_= std::min(1.0 - c1_,
                   2.0 * (mueff_ - 2.0 + 1.0 / mueff_) /
                   (std::pow(DIMd + 2.0, 2.0) + mueff_));
    damps_ = 1.0 + 2.0 *
        std::max(0.0, std::sqrt((mueff_ - 1.0) / (DIMd + 1.0)) - 1.0) + cs_;

    pc_ = VectorXd::Zero(D);
    ps_ = VectorXd::Zero(D);
    B_  = MatrixXd::Identity(D, D);
    D_  = VectorXd::Ones(D);
    C_  = MatrixXd::Identity(D, D);
    invsqrtC_ = MatrixXd::Identity(D, D);
    eigeneval_ = 0;
    chiN_ = std::sqrt(DIMd) *
            (1.0 - 1.0 / (4.0 * DIMd) + 1.0 / (21.0 * DIMd * DIMd));

    oldPop_ = MatrixXd(D, N_);
    for (int i = 0; i < N_; ++i)
        for (int j = 0; j < D; ++j)
            oldPop_(j, i) = X_[i][j];

    Asize_max_ = (int)std::round(N_ * 2.6);
    Asize_     = 0;
    A_.clear();
    Hjso_ = 5;
    MF_.assign(Hjso_, 0.3);
    MCR_.assign(Hjso_, 0.8);
    MF_.back()  = 0.9;
    MCR_.back() = 0.9;
    k_mem_ = 0;
    pmax_  = 0.25;
    pmin_  = pmax_ / 2.0;

    updateStop(FX_);
    printBest();
}

// ========================= CoBiDE (case 1) =========================

void EA4Eig::stepCoBiDE() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = N_;
    if (N < 4) return;

    const double INF = std::numeric_limits<double>::infinity();

    std::vector<Vec> mutants(N, Vec(D));
    std::vector<Vec> trial(N,   Vec(D));
    std::vector<double> trialF(N, INF);

    // Mutation DE/rand/1
    for (int i = 0; i < N; ++i) {
        std::vector<int> idx;
        sampleDistinctExcluding(N, 3, {i}, idx);
        if ((int)idx.size() < 3) {
            mutants[i] = X_[i];
            continue;
        }

        int r1 = idx[0], r2 = idx[1], r3 = idx[2];
        const Vec& x1 = X_[r1];
        const Vec& x2 = X_[r2];
        const Vec& x3 = X_[r3];

        double F = CBF_[i];
        Vec v(D);
        for (int j = 0; j < D; ++j)
            v[j] = x1[j] + F * (x2[j] - x3[j]);

        ensureBounds(v);
        mutants[i] = std::move(v);
    }

    // eigen–crossover με πιθανότητα peig_
    bool useEigen = false;
    MatrixXd EigVecs;

    if (randU() < peig_) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });

        int keep = (int)std::round(N * CBps_);
        if (keep < 2) keep = 2;
        if (keep > N) keep = N;

        MatrixXd Popeig(keep, D);
        for (int k = 0; k < keep; ++k) {
            const Vec& x = X_[order[k]];
            for (int j = 0; j < D; ++j)
                Popeig(k, j) = x[j];
        }

        if (keep > 1) {
            MatrixXd centered = Popeig.rowwise() - Popeig.colwise().mean();
            MatrixXd C = (centered.transpose() * centered) / double(keep - 1);

            Eigen::SelfAdjointEigenSolver<MatrixXd> es(C);
            if (es.info() == Eigen::Success) {
                EigVecs = es.eigenvectors();
                useEigen = true;
            }
        }
    }

    if (useEigen) {
        for (int i = 0; i < N; ++i) {
            VectorXd y(D), v(D);
            for (int j = 0; j < D; ++j) {
                y[j] = X_[i][j];
                v[j] = mutants[i][j];
            }
            VectorXd yeig = EigVecs.transpose() * y;
            VectorXd veig = EigVecs.transpose() * v;

            double CR = CBCR_[i];
            std::vector<int> changed;
            changed.reserve(D);
            for (int j = 0; j < D; ++j)
                if (randU() < CR) changed.push_back(j);
            if (changed.empty())
                changed.push_back(randInt(0, D - 1));

            for (int j : changed)
                yeig[j] = veig[j];

            VectorXd ynew = EigVecs * yeig;

            Vec t(D);
            for (int j = 0; j < D; ++j)
                t[j] = ynew[j];

            ensureBounds(t);
            trial[i] = std::move(t);
        }
    } else {
        for (int i = 0; i < N; ++i) {
            const Vec& xi = X_[i];
            const Vec& v  = mutants[i];

            double CR = CBCR_[i];
            int jrand = randInt(0, D - 1);

            Vec t(D);
            for (int j = 0; j < D; ++j) {
                if (randU() < CR || j == jrand)
                    t[j] = v[j];
                else
                    t[j] = xi[j];
            }
            ensureBounds(t);
            trial[i] = std::move(t);
        }
    }

    for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
        trialF[i] = eval(trial[i]);
    }

    for (int i = 0; i < N; ++i) {
        if (trialF[i] <= FX_[i]) {
            X_[i]  = trial[i];
            FX_[i] = trialF[i];
            success_[0] += 1;
            ni_[0]      += 1.0;

            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        } else {
            double F;
            if (randU() < 0.5)
                F = cauchyRnd(0.65, 0.1);
            else
                F = cauchyRnd(1.0, 0.1);
            while (F < 0.0) {
                if (randU() < 0.5)
                    F = cauchyRnd(0.65, 0.1);
                else
                    F = cauchyRnd(1.0, 0.1);
            }
            if (F > 1.0) F = 1.0;
            CBF_[i] = F;

            double CR;
            if (randU() < 0.5)
                CR = cauchyRnd(0.1, 0.1);
            else
                CR = cauchyRnd(0.95, 0.1);
            if (CR > 1.0) CR = 1.0;
            if (CR < 0.0) CR = 0.0;
            CBCR_[i] = CR;
        }
    }
}

// ========================= IDE (case 2) =========================

void EA4Eig::stepIDE() {
    int N = N_;
    if (N < 4) return;
    const int D = (int)lb_.size();

    ++g_;
    if (g_ > gmax_) g_ = gmax_;

    sortByFitness();

    std::vector<Vec> Qx(N, Vec(D));
    std::vector<double> Qf(N, std::numeric_limits<double>::infinity());

    double IDEps = 0.1 + 0.9 * std::pow(10.0, 5.0 * ((double)g_ / gmax_ - 1.0));
    double pd    = 0.1 * IDEps;
    double SRT   = (g_ < gt_) ? 0.0 : 0.1;

    bool useEigen = false;
    MatrixXd EigVect;
    if (randU() < peig_) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });
        int keep = (int)std::round(N * CBps_);
        if (keep < 2) keep = std::min(2, N);
        idx.resize(keep);
        useEigen = computeEigenVectorsFromElite(idx, EigVect);
    }

    for (int i = 0; i < N; ++i) {
        std::vector<int> expt = {i};
        std::vector<int> vyb;
        sampleDistinctExcluding(N, 4, expt, vyb);
        if ((int)vyb.size() < 4) continue;
        int o  = vyb[0];
        int r1 = vyb[1];
        int r2 = vyb[2];
        int r3 = vyb[3];
        if (g_ <= gt_) o = i;

        Vec xo  = X_[o];
        Vec xr1 = X_[r1];
        Vec xr2 = X_[r2];
        Vec xr3 = X_[r3];

        for (int j = 0; j < D; ++j) {
            if (randU() < pd)
                xr3[j] = lb_[j] + randU() * (ub_[j] - lb_[j]);
        }

        double Fo = (double)(o + 1) / (double)N + 0.1 * randN01();
        while (Fo <= 0.0 || Fo > 1.0)
            Fo = (double)(o + 1) / (double)N + 0.1 * randN01();

        int high_ind_S = (int)std::floor(IDEps * N);
        if (high_ind_S < 1) high_ind_S = 1;
        if (high_ind_S > N) high_ind_S = N;

        if (o >= high_ind_S) {
            if (r1 >= high_ind_S) {
                std::vector<int> cand;
                cand.reserve(high_ind_S);
                for (int id = 0; id < high_ind_S; ++id)
                    if (id != r1 && id != r2 && id != r3 && id != i)
                        cand.push_back(id);
                if (!cand.empty()) {
                    int pick = cand[randInt(0, (int)cand.size() - 1)];
                    xr1 = X_[pick];
                }
            }
        }

        Vec v(D);
        if ((g_ > gt_) && (randU() < 0.5)) {
            for (int j = 0; j < D; ++j)
                v[j] = X_[i][j] + Fo * (xr1[j] - xo[j]) +
                       Fo * (xr2[j] - xr3[j]);
        } else {
            for (int j = 0; j < D; ++j)
                v[j] = xo[j] + Fo * (xr1[j] - xo[j]) +
                       Fo * (xr2[j] - xr3[j]);
        }
        ensureBounds(v);
        Qx[i] = v;
    }

    for (int i = 0; i < N; ++i) {
        Vec y = X_[i];
        Vec v = Qx[i];
        if (useEigen) {
            VectorXd yv(D), vv(D);
            for (int j = 0; j < D; ++j) {
                yv[j] = y[j];
                vv[j] = v[j];
            }
            VectorXd yeig = EigVect.transpose() * yv;
            VectorXd veig = EigVect.transpose() * vv;
            double CR = CBCR_[i];
            std::vector<int> change;
            for (int j = 0; j < D; ++j)
                if (randU() < CR) change.push_back(j);
            if (change.empty())
                change.push_back(randInt(0, D - 1));
            for (int j : change)
                yeig[j] = veig[j];
            VectorXd ynew = EigVect * yeig;
            for (int j = 0; j < D; ++j)
                y[j] = ynew[j];
            ensureBounds(y);
        } else {
            double CR = (double)(i + 1) / (double)N + 0.1 * randN01();
            while (CR < 0.0 || CR > 1.0)
                CR = (double)(i + 1) / (double)N + 0.1 * randN01();

            int jrand = randInt(0, D - 1);
            for (int j = 0; j < D; ++j) {
                if (!(randU() <= CR || j == jrand)) {
                    v[j] = X_[i][j];
                }
                if (v[j] < lb_[j] || v[j] > ub_[j]) {
                    v[j] = lb_[j] + randU() * (ub_[j] - lb_[j]);
                }
            }
            y = v;
        }
        Qx[i] = y;
    }

    int suc_cnt = 0;
    for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
        double fy = eval(Qx[i]);
        Qf[i] = fy;
    }

    std::vector<int> indsucc;
    for (int i = 0; i < N; ++i) {
        if (Qf[i] <= FX_[i]) {
            indsucc.push_back(i);
        }
    }
    suc_cnt = (int)indsucc.size();
    success_[1] += suc_cnt;
    ni_[1]      += (double)suc_cnt;
    double SR = (N > 0) ? (double)suc_cnt / (double)N : 0.0;

    if (g_ < gt_) {
        if (SR <= SRT) ++Tcurr_;
        else           Tcurr_ = 0;
        if (Tcurr_ >= T_) gt_ = g_;
    }

    for (int idx : indsucc) {
        X_[idx]  = Qx[idx];
        FX_[idx] = Qf[idx];
        if (FX_[idx] < best_f_) {
            best_f_ = FX_[idx];
            best_x_ = X_[idx];
        }
    }
    sortByFitness();
}

// ========================= CMA-ES (case 3) =========================

void EA4Eig::stepCMAES() {
    int N = N_;
    if (N < 4) return;
    const int D = (int)lb_.size();

    sortByFitness();

    if (mu_ > N) {
        mu_ = std::max(1, N / 2);
        weights_.assign(mu_, 0.0);
        for (int i = 0; i < mu_; ++i) {
            double wi = std::log(mu_ + 0.5) - std::log((double)(i + 1));
            weights_[i] = wi;
        }
        double sw = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        if (sw <= 0.0) sw = 1.0;
        for (double& w : weights_) w /= sw;
        double sw2 = 0.0;
        for (double w : weights_) sw2 += w * w;
        mueff_ = (sw * sw) / std::max(sw2, 1e-16);
    }

    VectorXd xmean(D);
    xmean.setZero();
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < mu_ && i < N; ++i)
            s += X_[i][j] * weights_[i];
        xmean[j] = s;
    }

    MatrixXd Pop(D, N);
    std::vector<double> PopFit(N, std::numeric_limits<double>::infinity());

    for (int kk = 0; kk < N && prob_->calls() < max_evals_; ++kk) {
        VectorXd z(D);
        for (int j = 0; j < D; ++j) z[j] = randN01();
        VectorXd Bd = B_ * (D_.asDiagonal() * z);
        VectorXd cand = xmean + sigma_ * Bd;

        for (int j = 0; j < D; ++j) {
            if (cand[j] < lb_[j])
                cand[j] = 0.5 * (oldPop_(j, kk) + lb_[j]);
            if (cand[j] > ub_[j])
                cand[j] = 0.5 * (oldPop_(j, kk) + ub_[j]);
        }

        for (int j = 0; j < D; ++j)
            Pop(j, kk) = cand[j];

        Vec tmp(D);
        for (int j = 0; j < D; ++j) tmp[j] = cand[j];
        double f = eval(tmp);
        PopFit[kk] = f;

        int worst = 0;
        double wf = FX_[0];
        for (int i = 1; i < N; ++i) {
            if (FX_[i] > wf) {
                wf = FX_[i];
                worst = i;
            }
        }
        if (f < wf) {
            X_[worst]  = tmp;
            FX_[worst] = f;
            success_[2] += 1;
            ni_[2]      += 1.0;
            if (f < best_f_) {
                best_f_ = f;
                best_x_ = tmp;
            }
        }
    }

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return PopFit[a] < PopFit[b]; });

    VectorXd xold = xmean;
    xmean.setZero();
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < mu_ && i < N; ++i)
            s += Pop(j, idx[i]) * weights_[i];
        xmean[j] = s;
    }

    oldPop_ = Pop;

    double DIMd = (double)D;
    VectorXd delta = (xmean - xold) / std::max(sigma_, 1e-30);
    ps_ = (1.0 - cs_) * ps_ + std::sqrt(cs_ * (2.0 - cs_) * mueff_) *
          (invsqrtC_ * delta);

    double ps_norm = ps_.norm();
    double hsig = (ps_norm / std::sqrt(1.0 - std::pow(1.0 - cs_,
                               2.0 * (double)prob_->calls() / (double)N)) /
                   chiN_) < (2.0 + 4.0 / (DIMd + 1.0));
    pc_ = (1.0 - cc_) * pc_ +
          (hsig ? 1.0 : 0.0) *
          std::sqrt(cc_ * (2.0 - cc_) * mueff_) * delta;

    MatrixXd artmp(D, mu_);
    for (int i = 0; i < mu_; ++i) {
        int col = idx[i];
        artmp.col(i) = (Pop.col(col) - xold) / std::max(sigma_, 1e-30);
    }

    C_ = (1.0 - c1_ - cmu_) * C_
         + c1_ * (pc_ * pc_.transpose()
                  + (1.0 - hsig) * cc_ * (2.0 - cc_) * C_)
         + cmu_ * artmp *
           Eigen::Map<VectorXd>(weights_.data(), mu_).asDiagonal() *
           artmp.transpose();

    sigma_ *= std::exp((cs_ / damps_) * (ps_norm / chiN_ - 1.0));

    if (!std::isfinite(sigma_) || sigma_ < 1e-300 || sigma_ > 1e300) {
        double range = 0.0;
        for (int j = 0; j < D; ++j)
            range = std::max(range, std::fabs(ub_[j] - lb_[j]));
        if (range <= 0.0) range = 1.0;
        sigma_ = range / 2.0;
    }

    long long FES = prob_->calls();
    if (FES - eigeneval_ > N / (c1_ + cmu_) / D / 10) {
        eigeneval_ = FES;
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(C_);
        if (es.info() == Eigen::Success) {
            B_ = es.eigenvectors();
            D_ = es.eigenvalues().cwiseMax(1e-30).cwiseSqrt();
            invsqrtC_ = B_ * D_.cwiseInverse().asDiagonal() * B_.transpose();
        }
    }
}

// ========================= jSO (case 4) =========================

void EA4Eig::stepJSO() {
    int N = N_;
    if (N < 4) return;
    const int D = (int)lb_.size();

    double FES  = (double)prob_->calls();
    double maxF = (double)std::max<long long>(max_evals_, 1);

    std::vector<double> Fpole(N, -1.0);
    std::vector<double> CRpole(N, -1.0);
    std::vector<double> SCR;
    std::vector<double> SF;
    std::vector<double> deltaF(N, -1.0);
    int suc = 0;

    std::vector<Vec> Qx(N, Vec(D));
    std::vector<double> Qf(N, std::numeric_limits<double>::infinity());

    double pp = pmax_ - ((pmax_ - pmin_) * (FES / maxF));
    if (pp < pmin_) pp = pmin_;
    if (pp > pmax_) pp = pmax_;

    bool useEigen = false;
    MatrixXd EigVect;
    if (randU() < peig_) {
        std::vector<int> idxN(N);
        std::iota(idxN.begin(), idxN.end(), 0);
        std::sort(idxN.begin(), idxN.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });
        int keep = (int)std::round(N * CBps_);
        if (keep < 2) keep = std::min(2, N);
        idxN.resize(keep);
        useEigen = computeEigenVectorsFromElite(idxN, EigVect);
    }

    for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
        int rr = randInt(0, Hjso_ - 1);
        double CR = MCR_[rr] + std::sqrt(0.1) * randN01();
        if (CR > 1.0) CR = 1.0;
        if (CR < 0.0) CR = 0.0;
        if (FES < 0.25 * maxF)      CR = std::max(CR, 0.7);
        else if (FES < 0.5 * maxF)  CR = std::max(CR, 0.6);

        double F = -1.0;
        int guard = 0;
        while (F <= 0.0 && guard < 20) {
            double u = randU() * EA4EIG_PI - EA4EIG_PI / 2.0;
            F = 0.1 * std::tan(u) + MF_[rr];
            guard++;
        }
        if (F <= 0.0) F = 0.5;
        if (F > 1.0)  F = 1.0;
        if (FES < 0.6 * maxF && F > 0.7) F = 0.7;

        Fpole[i]  = F;
        CRpole[i] = CR;

        int p = std::max(2, (int)std::ceil(pp * N));
        if (p > N - 1) p = N - 1;

        std::vector<int> idxN(N);
        std::iota(idxN.begin(), idxN.end(), 0);
        std::sort(idxN.begin(), idxN.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });
        std::vector<int> pbestIdx(idxN.begin(), idxN.begin() + p);
        int ktery = randInt(0, p - 1);
        const Vec& xpbest = X_[pbestIdx[ktery]];

        int ex = i;
        const Vec& xi = X_[ex];

        std::vector<int> expt = {ex};
        std::vector<int> tmp;
        sampleDistinctExcluding(N, 1, expt, tmp);
        if (tmp.empty()) continue;
        const Vec& r1 = X_[tmp[0]];
        expt.push_back(tmp[0]);

        int total = N + Asize_;
        std::vector<int> exptAll = expt;
        sampleDistinctExcluding(total, 1, exptAll, tmp);
        const Vec& r2 = (tmp[0] < N) ? X_[tmp[0]] : A_[tmp[0] - N];

        double Fw;
        if (FES < 0.2 * maxF)      Fw = 0.7 * F;
        else if (FES < 0.4 * maxF) Fw = 0.8 * F;
        else                       Fw = 1.2 * F;

        Vec v(D);
        for (int j = 0; j < D; ++j)
            v[j] = xi[j] + Fw * (xpbest[j] - xi[j]) +
                   F * (r1[j] - r2[j]);

        Vec y = xi;
        if (useEigen) {
            VectorXd yv(D), vv(D);
            for (int j = 0; j < D; ++j) {
                yv[j] = y[j];
                vv[j] = v[j];
            }
            VectorXd yeig = EigVect.transpose() * yv;
            VectorXd veig = EigVect.transpose() * vv;
            double CRloc = CR;
            std::vector<int> change;
            for (int j = 0; j < D; ++j)
                if (randU() < CRloc) change.push_back(j);
            if (change.empty())
                change.push_back(randInt(0, D - 1));
            for (int j : change)
                yeig[j] = veig[j];
            VectorXd ynew = EigVect * yeig;
            for (int j = 0; j < D; ++j)
                y[j] = ynew[j];
        } else {
            std::vector<int> change;
            for (int j = 0; j < D; ++j)
                if (randU() < CR) change.push_back(j);
            if (change.empty())
                change.push_back(randInt(0, D - 1));
            for (int j : change)
                y[j] = v[j];
        }

        ensureBounds(y);
        Qx[i] = y;
    }

    for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
        double fy = eval(Qx[i]);
        Qf[i] = fy;
    }

    for (int i = 0; i < N; ++i) {
        if (Qf[i] < FX_[i]) {
            double gain = FX_[i] - Qf[i];
            deltaF[i]   = gain;
            ++suc;
            if (Asize_ < Asize_max_) {
                addToArchive(X_[i]);
            } else {
                int idx = randInt(0, Asize_ - 1);
                A_[idx] = X_[i];
            }
            SCR.push_back(CRpole[i]);
            SF.push_back(Fpole[i]);
        }
        if (Qf[i] <= FX_[i]) {
            X_[i]  = Qx[i];
            FX_[i] = Qf[i];
            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        }
    }

    if (suc > 0 && !SF.empty()) {
        double sum_delta = 0.0;
        for (double d : deltaF)
            if (d > 0.0) sum_delta += d;
        if (sum_delta > 0.0) {
            std::vector<double> w(deltaF.size(), 0.0);
            for (int i = 0; i < N; ++i) {
                if (deltaF[i] > 0.0)
                    w[i] = deltaF[i] / sum_delta;
            }
            double numF = 0.0, denF = 0.0;
            double numCR = 0.0, denCR = 0.0;
            int idxSF = 0;
            for (int i = 0; i < N && idxSF < (int)SF.size(); ++i) {
                if (deltaF[i] > 0.0) {
                    double wi = w[i];
                    numF  += wi * SF[idxSF] * SF[idxSF];
                    denF  += wi * SF[idxSF];
                    numCR += wi * SCR[idxSF];
                    denCR += wi;
                    ++idxSF;
                }
            }
            int k = k_mem_;
            double MF_old  = MF_[k];
            double MCR_old = MCR_[k];
            if (denF > 0.0) {
                double meanF = numF / denF;
                MF_[k] = 0.5 * (MF_old + meanF);
            }
            if (denCR > 0.0) {
                double meanCR = numCR / denCR;
                MCR_[k] = 0.5 * (MCR_old + meanCR);
            }
            k_mem_ = (k + 1) % Hjso_;
        }
    }

    success_[3] += suc;
    ni_[3]      += (double)suc;
}

// ========================= one_iteration =========================

void EA4Eig::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (N_ < 4) return;

    auto sel = rouletteSelect();
    int    hh   = sel.first;
    double pmin = sel.second;

    if (pmin < delta_) {
        for (int i = 0; i < H_; ++i)
            cni_[i] += ni_[i] - (double)n0_;
        ni_.assign(H_, (double)n0_);
        ++nrst_;
    }

    switch (hh) {
        case 0: stepCoBiDE(); break;
        case 1: stepIDE();    break;
        case 2: stepCMAES();  break;
        case 3:
        default: stepJSO();   break;
    }

    if (!local_method_.empty() && local_rate_ > 0.0) {
        for (int i = 0; i < N_ && prob_->calls() < max_evals_; ++i) {
            if (randU() < local_rate_) {
                auto res = localSearch(local_method_, X_[i]);
                if (!res.first.empty() && std::isfinite(res.second) &&
                    res.second < FX_[i]) {
                    X_[i]  = res.first;
                    FX_[i] = res.second;
                    if (res.second < best_f_) {
                        best_f_ = res.second;
                        best_x_ = X_[i];
                    }
                }
            }
        }
    }

    long long FES = prob_->calls();
    double maxF = (double)std::max<long long>(max_evals_, 1);
    double Ninit = (double)N_init_;
    double Nmin  = (double)N_min_;
    int optN = (int)std::round(((Nmin - Ninit) / maxF) * (double)FES + Ninit);
    if (optN < N_min_) optN = N_min_;
    if (N_ > optN) {
        int diffPop = N_ - optN;
        if (N_ - diffPop < N_min_)
            diffPop = N_ - N_min_;
        if (diffPop > 0)
            shrinkPopulationTo(N_ - diffPop);
    }

    updateStop(FX_);
    printBest();
}

// ========================= end =========================

void EA4Eig::end() {
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        if (!res.first.empty() && std::isfinite(res.second) &&
            res.second < best_f_) {
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
