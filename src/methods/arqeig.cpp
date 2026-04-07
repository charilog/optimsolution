#include "arqeig.h"
#include "init.h"
#include "options.h"

#include <utility>

namespace optimsolution {

static inline std::string to_lower(std::string s) {
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void ARQEig::configure(const MethodConfig& mc)
{
    // population (initial)
    int p = mc.getInt("population", pop_init_);
    if (p > 3) {
        pop_init_ = p;
        Optimizer::setPopulation(pop_init_);
    }

    // ---------------- Adaptive Population Leaps (APL) ----------------
    adaptive_population_ = mc.getBool("adaptive_population", adaptive_population_);
    pop_min_             = mc.getInt("pop_min", pop_min_);
    pop_max_             = mc.getInt("pop_max", std::max(pop_max_, pop_init_));
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_max_ < pop_min_) pop_max_ = pop_min_;

    pop_warmup_iters_    = mc.getInt("pop_warmup_iters", pop_warmup_iters_);
    if (pop_warmup_iters_ < 0) pop_warmup_iters_ = 0;

    pop_check_interval_  = mc.getInt("pop_check_interval", pop_check_interval_);
    if (pop_check_interval_ < 1) pop_check_interval_ = 1;

    pop_window_          = mc.getInt("pop_window", pop_window_);
    if (pop_window_ < 5) pop_window_ = 5;

    pop_success_thr_     = mc.getDbl("pop_success_thr", pop_success_thr_);
    if (pop_success_thr_ < 0.0) pop_success_thr_ = 0.0;
    if (pop_success_thr_ > 1.0) pop_success_thr_ = 1.0;

    pop_impr_thr_        = mc.getDbl("pop_impr_thr", pop_impr_thr_);
    if (pop_impr_thr_ < 0.0) pop_impr_thr_ = 0.0;

    pop_div_low_         = mc.getDbl("pop_div_low", pop_div_low_);
    pop_div_high_        = mc.getDbl("pop_div_high", pop_div_high_);
    if (pop_div_low_ < 0.0) pop_div_low_ = 0.0;
    if (pop_div_high_ < pop_div_low_) pop_div_high_ = pop_div_low_;

    pop_shrink_factor_   = mc.getDbl("pop_shrink_factor", pop_shrink_factor_);
    if (pop_shrink_factor_ <= 0.0) pop_shrink_factor_ = 0.25;
    if (pop_shrink_factor_ > 0.95) pop_shrink_factor_ = 0.95;

    pop_expand_factor_   = mc.getDbl("pop_expand_factor", pop_expand_factor_);
    if (pop_expand_factor_ < 1.05) pop_expand_factor_ = 2.0;
    if (pop_expand_factor_ > 10.0) pop_expand_factor_ = 10.0;

    pop_elite_frac_      = mc.getDbl("pop_elite_frac", pop_elite_frac_);
    if (pop_elite_frac_ < 0.05) pop_elite_frac_ = 0.05;
    if (pop_elite_frac_ > 0.80) pop_elite_frac_ = 0.80;

    pop_cooldown_        = mc.getInt("pop_cooldown", pop_cooldown_);
    if (pop_cooldown_ < 0) pop_cooldown_ = 0;

    // ---------------- ARQ / JADE-like ----------------
    H_          = mc.getInt("H", H_);
    if (H_ < 2) H_ = 2;

    pbest_      = mc.getDbl("pbest", pbest_);
    if (pbest_ < 0.01) pbest_ = 0.01;
    if (pbest_ > 0.50) pbest_ = 0.50;

    Fmin_       = mc.getDbl("Fmin", Fmin_);
    Fmax_       = mc.getDbl("Fmax", Fmax_);
    if (Fmin_ <= 0.0) Fmin_ = 0.01;
    if (Fmax_ < Fmin_) std::swap(Fmax_, Fmin_);
    if (Fmax_ > 2.0) Fmax_ = 2.0;

    archiverate_ = mc.getDbl("archiverate", archiverate_);
    if (archiverate_ <= 0.1) archiverate_ = 1.0;

    // RTR / restart / quarantine
    rtr_k_            = mc.getInt("rtr_k", rtr_k_);
    if (rtr_k_ < 2) rtr_k_ = 2;

    outlier_alpha_    = mc.getDbl("outlier_alpha", outlier_alpha_);
    if (outlier_alpha_ <= 0.0) outlier_alpha_ = 1.5;

    outlier_rho_      = mc.getDbl("outlier_rho", outlier_rho_);
    if (outlier_rho_ < 0.0) outlier_rho_ = 0.0;
    if (outlier_rho_ > 1.0) outlier_rho_ = 1.0;

    qsigma_           = mc.getDbl("qsigma", qsigma_);
    if (qsigma_ < 0.0) qsigma_ = 0.0;
    if (qsigma_ > 2.0) qsigma_ = 2.0;

    worst_frac_       = mc.getDbl("worst_frac", mc.getDbl("w", worst_frac_));
    if (worst_frac_ < 0.0) worst_frac_ = 0.0;
    if (worst_frac_ > 0.5) worst_frac_ = 0.5;

    rsigma_           = mc.getDbl("rsigma", rsigma_);
    if (rsigma_ < 0.0) rsigma_ = 0.0;
    if (rsigma_ > 2.0) rsigma_ = 2.0;

    stagnationtrigger_ = mc.getInt("stagnationtrigger", stagnationtrigger_);
    if (stagnationtrigger_ < 5) stagnationtrigger_ = 5;

    // Eig controls
    eiginterval_ = mc.getInt("eiginterval", eiginterval_);
    if (eiginterval_ < 1) eiginterval_ = 1;

    peig_ = mc.getDbl("peig", peig_);
    if (peig_ < 0.0) peig_ = 0.0;
    if (peig_ > 1.0) peig_ = 1.0;

    eig_eps_ = mc.getDbl("eig_eps", eig_eps_);
    if (eig_eps_ <= 0.0) eig_eps_ = 1e-12;

    // in-run local search (optional)
    local_method_ = mc.getStr("local_method", local_method_);
    local_method_ = to_lower(local_method_);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // final local refinement (method-level; falls back to global if not present)
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
    end_local_method_ = to_lower(end_local_method_);
}

void ARQEig::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    // apply initial N
    Optimizer::setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);

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
        if (terminated()) break;
    }

    // init memories
    MF_.assign(H_, 0.6);
    MCR_.assign(H_, 0.8);
    k_mem_ = 0;

    // archive
    A_.clear();

    // eig state
    mean_bn_.assign(D, 0.0);
    B_.assign(D * D, 0.0);
    for (int i = 0; i < D; ++i) B_[i * D + i] = 1.0;
    eig_ready_ = false;
    eig_age_ = 0;

    best_prev_ = best_f_;
    no_improve_ = 0;

    // APL state
    iter_ = 0;
    pop_cooldown_left_ = 0;
    best_hist_.clear();
    if (std::isfinite(best_f_)) best_hist_.push_back(best_f_);
    last_resize_iter_ = -999999;

    // initial basis
    std::vector<int> idx;
    sortByFitness(idx);
    recomputeEigenBasis(idx);

    updateStop(FX_);
    printBest();
}

void ARQEig::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D   = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j]))
            x[j] = 0.5 * (L[j] + U[j]);
        // mirror (as in EA4Eig)
        while (x[j] < L[j] || x[j] > U[j]) {
            if (x[j] > U[j])
                x[j] = 2.0 * U[j] - x[j];
            else if (x[j] < L[j])
                x[j] = 2.0 * L[j] - x[j];
        }
    }
}

int ARQEig::randInt(int lo, int hi)
{
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double ARQEig::randU()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQEig::randN01()
{
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQEig::cauchy(double loc, double scale)
{
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

void ARQEig::sampleDistinctExcluding(int N, int k,
                                     const std::vector<int>& exclude,
                                     std::vector<int>& out)
{
    out.clear();
    out.reserve(k);
    if (N <= 0 || k <= 0) return;

    std::vector<int> cand;
    cand.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::find(exclude.begin(), exclude.end(), i) == exclude.end())
            cand.push_back(i);
    }
    if ((int)cand.size() <= k) { out = cand; return; }

    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)cand.size() - 1);
        int r = dist(rng_);
        std::swap(cand[i], cand[r]);
        out.push_back(cand[i]);
    }
}

void ARQEig::addToArchive(const Vec& x)
{
    A_.push_back(x);
}

void ARQEig::trimArchive()
{
    if (N_ <= 0) return;
    int cap = (int)std::round(archiverate_ * (double)N_);
    if (cap < 0) cap = 0;
    if ((int)A_.size() <= cap) return;
    std::shuffle(A_.begin(), A_.end(), rng_);
    A_.resize(cap);
}

void ARQEig::sortByFitness(std::vector<int>& idx) const
{
    idx.resize(N_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });
}

double ARQEig::quantile(std::vector<double> v, double q01)
{
    if (v.empty()) return std::numeric_limits<double>::infinity();
    if (q01 < 0.0) q01 = 0.0;
    if (q01 > 1.0) q01 = 1.0;

    const double pos  = q01 * (double)(v.size() - 1);
    const size_t k    = (size_t)std::floor(pos);
    const double frac = pos - (double)k;

    std::nth_element(v.begin(), v.begin() + k, v.end());
    double a = v[k];
    if (k + 1 >= v.size()) return a;
    std::nth_element(v.begin(), v.begin() + (k + 1), v.end());
    double b = v[k + 1];
    return a + frac * (b - a);
}

void ARQEig::toBN(const Vec& x, Vec& y) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D   = (int)x.size();
    y.assign(D, 0.0);
    for (int j = 0; j < D; ++j) {
        double denom = (U[j] - L[j]);
        if (denom <= 0.0) denom = 1.0;
        y[j] = (x[j] - L[j]) / denom; // [0,1]
    }
}

void ARQEig::fromBN(const Vec& y, Vec& x) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D   = (int)y.size();
    x.assign(D, 0.0);
    for (int j = 0; j < D; ++j) {
        double denom = (U[j] - L[j]);
        if (denom <= 0.0) denom = 1.0;
        x[j] = L[j] + y[j] * denom;
    }
}

void ARQEig::matT_vec(const std::vector<double>& M, int D, const Vec& x, Vec& y) const
{
    y.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) {
            // M^T(i,j) = M(j,i)
            s += M[j * D + i] * x[j];
        }
        y[i] = s;
    }
}

void ARQEig::mat_vec(const std::vector<double>& M, int D, const Vec& x, Vec& y) const
{
    y.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) {
            s += M[i * D + j] * x[j];
        }
        y[i] = s;
    }
}

// Jacobi eigen decomposition for symmetric matrix A (row-major)
// Returns V (eigenvectors, columns) and evals.
void ARQEig::jacobiEigenSymmetric(std::vector<double>& A, int D,
                                  std::vector<double>& V, std::vector<double>& evals,
                                  int max_sweeps, double tol)
{
    V.assign(D * D, 0.0);
    for (int i = 0; i < D; ++i) V[i * D + i] = 1.0;

    auto idx = [D](int r, int c){ return r * D + c; };

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // find largest off-diagonal
        int p = 0, q = 1;
        double maxa = 0.0;
        for (int i = 0; i < D; ++i) {
            for (int j = i + 1; j < D; ++j) {
                double aij = std::fabs(A[idx(i,j)]);
                if (aij > maxa) { maxa = aij; p = i; q = j; }
            }
        }
        if (maxa < tol) break;

        double app = A[idx(p,p)];
        double aqq = A[idx(q,q)];
        double apq = A[idx(p,q)];

        double phi = 0.5 * std::atan2(2.0 * apq, (aqq - app));
        double c = std::cos(phi);
        double s = std::sin(phi);

        // rotate A: A = G^T A G
        for (int k = 0; k < D; ++k) {
            if (k == p || k == q) continue;
            double aik = A[idx(p,k)];
            double aqk = A[idx(q,k)];
            double new_p = c * aik - s * aqk;
            double new_q = s * aik + c * aqk;

            A[idx(p,k)] = new_p;
            A[idx(k,p)] = new_p;
            A[idx(q,k)] = new_q;
            A[idx(k,q)] = new_q;
        }

        double new_app = c*c*app - 2.0*s*c*apq + s*s*aqq;
        double new_aqq = s*s*app + 2.0*s*c*apq + c*c*aqq;

        A[idx(p,p)] = new_app;
        A[idx(q,q)] = new_aqq;
        A[idx(p,q)] = 0.0;
        A[idx(q,p)] = 0.0;

        // update V: V = V G
        for (int k = 0; k < D; ++k) {
            double vip = V[idx(k,p)];
            double viq = V[idx(k,q)];
            V[idx(k,p)] = c * vip - s * viq;
            V[idx(k,q)] = s * vip + c * viq;
        }
    }

    evals.assign(D, 0.0);
    for (int i = 0; i < D; ++i) evals[i] = A[i * D + i];
}

void ARQEig::recomputeEigenBasis(const std::vector<int>& sorted_idx)
{
    const int D = prob_->dimension();
    if (N_ < 4 || D < 2) {
        eig_ready_ = false;
        return;
    }

    // take top M individuals
    int M = std::max(4, N_ / 2);
    if (M > N_) M = N_;

    // mean in BN
    mean_bn_.assign(D, 0.0);
    Vec bn(D, 0.0);

    for (int k = 0; k < M; ++k) {
        toBN(X_[sorted_idx[k]], bn);
        for (int j = 0; j < D; ++j) mean_bn_[j] += bn[j];
    }
    for (int j = 0; j < D; ++j) mean_bn_[j] /= (double)M;

    // covariance in BN: C = sum (y-mean)(y-mean)^T / (M-1)
    std::vector<double> C(D * D, 0.0);
    for (int k = 0; k < M; ++k) {
        toBN(X_[sorted_idx[k]], bn);
        for (int i = 0; i < D; ++i) {
            double di = bn[i] - mean_bn_[i];
            for (int j = 0; j < D; ++j) {
                C[i * D + j] += di * (bn[j] - mean_bn_[j]);
            }
        }
    }
    double denom = std::max(1, M - 1);
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) C[i * D + j] /= (double)denom;
        C[i * D + i] += eig_eps_;
    }

    // Jacobi eig
    std::vector<double> V, evals;
    jacobiEigenSymmetric(C, D, V, evals, /*max_sweeps*/ 50, /*tol*/ 1e-10);

    // sort eigenvectors by descending eigenvalue
    std::vector<int> order(D);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return evals[a] > evals[b]; });

    // Build B_ as row-major, columns are eigenvectors
    B_.assign(D * D, 0.0);
    for (int col = 0; col < D; ++col) {
        int src = order[col];
        for (int row = 0; row < D; ++row) {
            B_[row * D + col] = V[row * D + src];
        }
    }

    eig_ready_ = true;
    eig_age_ = 0;
}

double ARQEig::distBN(const Vec& a, const Vec& b) const
{
    Vec ba, bb;
    toBN(a, ba);
    toBN(b, bb);
    double s = 0.0;
    for (int i = 0; i < (int)ba.size(); ++i) {
        double d = ba[i] - bb[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void ARQEig::sample_F_CR(double& F, double& CR, double muF, double muCR)
{
    // Cauchy for F, Normal for CR (classic SHADE/JADE style)
    F = cauchy(muF, 0.1);
    int tries = 0;
    while (F <= 0.0 && tries < 25) { F = cauchy(muF, 0.1); ++tries; }
    if (F <= 0.0) F = muF;
    if (F > Fmax_) F = Fmax_;
    if (F < Fmin_) F = Fmin_;

    // CR
    CR = muCR + 0.1 * randN01();
    if (CR > 1.0) CR = 1.0;
    if (CR < 0.0) CR = 0.0;
}

void ARQEig::makeTrialBase(int i, const std::vector<int>& sorted_idx, double F, double CR, Vec& u)
{
    const int D = prob_->dimension();
    const int N = N_;

    int pcount = (int)std::ceil(pbest_ * (double)N);
    if (pcount < 2) pcount = 2;
    if (pcount > N) pcount = N;

    int pbest_idx = sorted_idx[randInt(0, pcount - 1)];

    std::vector<int> pick;
    sampleDistinctExcluding(N, 2, {i, pbest_idx}, pick);
    if ((int)pick.size() < 2) { u = X_[i]; return; }

    int r1 = pick[0];
    int r2 = pick[1];

    // use archive with prob 0.5
    Vec xr2 = X_[r2];
    if (!A_.empty() && randU() < 0.5) {
        int aidx = randInt(0, (int)A_.size() - 1);
        xr2 = A_[aidx];
    }

    Vec v(D);
    for (int j = 0; j < D; ++j) {
        v[j] = X_[i][j]
             + F * (X_[pbest_idx][j] - X_[i][j])
             + F * (X_[r1][j]       - xr2[j]);
    }
    ensureBounds(v);

    u = X_[i];
    int jrand = randInt(0, D - 1);
    for (int j = 0; j < D; ++j) {
        if (randU() < CR || j == jrand)
            u[j] = v[j];
    }
    ensureBounds(u);
}

void ARQEig::makeTrialEig(int i, const std::vector<int>& sorted_idx, double F, double CR, Vec& u)
{
    const int D = prob_->dimension();
    const int N = N_;

    int pcount = (int)std::ceil(pbest_ * (double)N);
    if (pcount < 2) pcount = 2;
    if (pcount > N) pcount = N;

    int pbest_idx = sorted_idx[randInt(0, pcount - 1)];

    std::vector<int> pick;
    sampleDistinctExcluding(N, 2, {i, pbest_idx}, pick);
    if ((int)pick.size() < 2) { u = X_[i]; return; }

    int r1 = pick[0];
    int r2 = pick[1];

    Vec xr2 = X_[r2];
    if (!A_.empty() && randU() < 0.5) {
        int aidx = randInt(0, (int)A_.size() - 1);
        xr2 = A_[aidx];
    }

    // BN transform
    Vec xi, xpb, xr1, xr2bn;
    toBN(X_[i], xi);
    toBN(X_[pbest_idx], xpb);
    toBN(X_[r1], xr1);
    toBN(xr2, xr2bn);

    // z = B^T (x - mean)
    Vec di(D), dp(D), dr1(D), dr2(D);
    for (int j = 0; j < D; ++j) {
        di[j]  = xi[j]  - mean_bn_[j];
        dp[j]  = xpb[j] - mean_bn_[j];
        dr1[j] = xr1[j] - mean_bn_[j];
        dr2[j] = xr2bn[j]- mean_bn_[j];
    }

    Vec zi, zpb, zr1, zr2;
    matT_vec(B_, D, di,  zi);
    matT_vec(B_, D, dp,  zpb);
    matT_vec(B_, D, dr1, zr1);
    matT_vec(B_, D, dr2, zr2);

    // mutation in eig-coordinates
    Vec zv(D);
    for (int j = 0; j < D; ++j) {
        zv[j] = zi[j]
              + F * (zpb[j] - zi[j])
              + F * (zr1[j] - zr2[j]);
    }

    // crossover in eig-coordinates
    Vec zu = zi;
    int jrand = randInt(0, D - 1);
    for (int j = 0; j < D; ++j) {
        if (randU() < CR || j == jrand)
            zu[j] = zv[j];
    }

    // back: x = mean + B z
    Vec bn_u_shift;
    mat_vec(B_, D, zu, bn_u_shift);

    Vec bn_u(D);
    for (int j = 0; j < D; ++j) bn_u[j] = mean_bn_[j] + bn_u_shift[j];

    fromBN(bn_u, u);
    ensureBounds(u);
}

bool ARQEig::selectionRTR(int parent, const Vec& u, double fu,
                          double F, double CR,
                          std::vector<double>& SF, std::vector<double>& SCR,
                          std::vector<double>& gains)
{
    // direct replacement
    if (fu <= FX_[parent]) {
        double g = FX_[parent] - fu;
        addToArchive(X_[parent]);
        X_[parent]  = u;
        FX_[parent] = fu;
        SF.push_back(F);
        SCR.push_back(CR);
        gains.push_back(std::max(0.0, g));
        return true;
    }

    // restricted tournament: pick rtr_k_ random opponents, replace closest if better
    int qstar = -1;
    double bestd = std::numeric_limits<double>::infinity();
    for (int t = 0; t < rtr_k_; ++t) {
        int q = randInt(0, N_ - 1);
        double d = distBN(u, X_[q]);
        if (d < bestd) { bestd = d; qstar = q; }
    }
    if (qstar < 0) return false;

    if (fu < FX_[qstar]) {
        double g = FX_[qstar] - fu;
        addToArchive(X_[qstar]);
        X_[qstar]  = u;
        FX_[qstar] = fu;
        SF.push_back(F);
        SCR.push_back(CR);
        gains.push_back(std::max(0.0, g));
        return true;
    }
    return false;
}

void ARQEig::updateMemories(const std::vector<double>& SF,
                            const std::vector<double>& SCR,
                            const std::vector<double>& gains)
{
    if (SF.empty() || SCR.empty() || gains.empty()) return;
    if (SF.size() != SCR.size() || SF.size() != gains.size()) return;

    double sumg = 0.0;
    for (double g : gains) sumg += std::max(0.0, g);
    if (sumg <= 0.0) return;

    double meanF_num = 0.0;
    double meanF_den = 0.0;
    double meanCR    = 0.0;

    for (size_t i = 0; i < SF.size(); ++i) {
        double w = std::max(0.0, gains[i]) / sumg;
        meanF_num += w * SF[i] * SF[i];
        meanF_den += w * SF[i];
        meanCR    += w * SCR[i];
    }

    double meanF = (meanF_den > 0.0) ? (meanF_num / meanF_den) : MF_[k_mem_];

    MF_[k_mem_]  = 0.5 * (MF_[k_mem_]  + meanF);
    MCR_[k_mem_] = 0.5 * (MCR_[k_mem_] + meanCR);

    if (MF_[k_mem_] < Fmin_) MF_[k_mem_] = Fmin_;
    if (MF_[k_mem_] > Fmax_) MF_[k_mem_] = Fmax_;
    if (MCR_[k_mem_] < 0.0)  MCR_[k_mem_] = 0.0;
    if (MCR_[k_mem_] > 1.0)  MCR_[k_mem_] = 1.0;

    k_mem_ = (k_mem_ + 1) % H_;
}

void ARQEig::quarantineAndRestart()
{
    if (N_ < 4) return;

    // outlier fence via IQR
    std::vector<double> f = FX_;
    double Q1 = quantile(f, 0.25);
    f = FX_;
    double Q3 = quantile(f, 0.75);
    double IQR = Q3 - Q1;
    double fence = Q3 + outlier_alpha_ * IQR;

    // center as mean of best half in BN
    std::vector<int> idx;
    sortByFitness(idx);

    int half = std::max(1, N_ / 2);
    Vec center_bn(prob_->dimension(), 0.0), bn;
    for (int k = 0; k < half; ++k) {
        toBN(X_[idx[k]], bn);
        for (int j = 0; j < (int)bn.size(); ++j) center_bn[j] += bn[j];
    }
    for (int j = 0; j < (int)center_bn.size(); ++j) center_bn[j] /= (double)half;

    // quarantine a fraction of outliers
    std::vector<int> out;
    for (int i = 0; i < N_; ++i) if (FX_[i] >= fence) out.push_back(i);

    int kfix = (int)std::floor(outlier_rho_ * (double)out.size());
    if (kfix > 0) {
        std::shuffle(out.begin(), out.end(), rng_);
        out.resize(kfix);

        for (int id : out) {
            if (terminated()) break;

            Vec cand_bn = center_bn;
            for (int j = 0; j < (int)cand_bn.size(); ++j)
                cand_bn[j] += qsigma_ * randN01();

            Vec cand;
            fromBN(cand_bn, cand);
            ensureBounds(cand);

            double fc = eval(cand);
            if (fc < FX_[id]) {
                addToArchive(X_[id]);
                X_[id] = std::move(cand);
                FX_[id] = fc;
            }
        }
    }

    if (no_improve_ < stagnationtrigger_) return;

    // micro-restart: perturb worst_frac_ around best in BN
    int wcount = (int)std::floor(worst_frac_ * (double)N_);
    if (wcount <= 0) { no_improve_ = 0; return; }

    Vec best_bn;
    toBN(best_x_, best_bn);

    for (int t = 0; t < wcount; ++t) {
        if (terminated()) break;
        int id = idx[N_ - 1 - t];

        Vec cand_bn = best_bn;
        for (int j = 0; j < (int)cand_bn.size(); ++j)
            cand_bn[j] += rsigma_ * randN01();

        Vec cand;
        fromBN(cand_bn, cand);
        ensureBounds(cand);

        double fc = eval(cand);
        // forced replace to inject diversity during stagnation
        addToArchive(X_[id]);
        X_[id] = std::move(cand);
        FX_[id] = fc;
    }

    // eigen basis should be rebuilt after large changes
    eig_ready_ = false;
    eig_age_ = 0;

    no_improve_ = 0;
}

// ---------------- Adaptive Population Leaps (APL) ----------------

double ARQEig::estimateDiversityBN(int sampleCount)
{
    if (!prob_ || N_ <= 1) return 0.0;
    const int D = prob_->dimension();
    if (D <= 0) return 0.0;

    int m = sampleCount;
    if (m < 2) m = 2;
    if (m > N_) m = N_;

    // sample indices (Fisher-Yates partial shuffle)
    std::vector<int> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);
    for (int i = 0; i < m; ++i) {
        std::uniform_int_distribution<int> dist(i, N_ - 1);
        int r = dist(rng_);
        std::swap(ids[i], ids[r]);
    }
    ids.resize(m);

    Vec mean(D, 0.0), bn(D, 0.0);
    for (int id : ids) {
        toBN(X_[id], bn);
        for (int j = 0; j < D; ++j) mean[j] += bn[j];
    }
    for (int j = 0; j < D; ++j) mean[j] /= (double)m;

    double ss = 0.0;
    for (int id : ids) {
        toBN(X_[id], bn);
        double d2 = 0.0;
        for (int j = 0; j < D; ++j) {
            double d = bn[j] - mean[j];
            d2 += d * d;
        }
        ss += d2;
    }
    ss /= (double)m;

    // RMS distance from mean in BN space
    return std::sqrt(ss);
}

double ARQEig::relImprovementFromHistory() const
{
    if (best_hist_.size() < 2) return 0.0;
    double first = best_hist_.front();
    double last  = best_hist_.back();
    if (!std::isfinite(first) || !std::isfinite(last)) return 0.0;
    double denom = std::fabs(first) + 1e-12;
    return (first - last) / denom;
}

void ARQEig::maybeAdaptivePopulationLeap(const std::vector<int>& sorted_idx,
                                         double success_rate,
                                         double diversity_bn,
                                         double rel_impr)
{
    if (!adaptive_population_) return;
    if (N_ < 4) return;
    if (pop_cooldown_left_ > 0) return;

    // only check every pop_check_interval_ iterations
    if (pop_check_interval_ > 1 && (iter_ % pop_check_interval_) != 0) return;

    const int D = prob_ ? prob_->dimension() : 0;

    // clamp bounds (in case of runtime changes)
    int hardMin = std::max(4, pop_min_);
    int hardMax = std::max(hardMin, pop_max_);

    int newN = N_;

    // 1) Early probe (helps on problems like potential38 where large N wastes evals,
    //    while smaller N increases generations and exploitation cadence).
    if (iter_ <= pop_warmup_iters_ && D >= 60 && N_ > hardMin) {
        bool ineffective = (success_rate < pop_success_thr_) && (rel_impr < pop_impr_thr_);
        if (ineffective) {
            // direct big shrink (aggressive)
            int cand = (int)std::round((double)N_ * pop_shrink_factor_);
            newN = std::max(hardMin, cand);
        }

        // extra aggressive rule for very high D: go straight to pop_min_
        if (D >= 100 && N_ >= 80 && (success_rate < 0.15)) {
            newN = hardMin;
        }
    }

    // 2) Stagnation-driven shrink: if we stall, reduce N sharply to increase
    //    iteration count and make parameter memories react faster.
    if (newN == N_ && no_improve_ >= stagnationtrigger_ && N_ > hardMin) {
        int cand = (int)std::round((double)N_ * pop_shrink_factor_);
        newN = std::max(hardMin, cand);
    }

    // 3) Diversity collapse: if population is too tight and success is low, expand
    //    to restore exploration capacity (large jump).
    if (newN == N_ && diversity_bn < pop_div_low_ && success_rate < 0.5 * pop_success_thr_ && N_ < hardMax) {
        int cand = (int)std::round((double)N_ * pop_expand_factor_);
        newN = std::min(hardMax, cand);
    }

    // 4) Optional: if diversity is extremely high but progress is tiny, shrink to focus.
    if (newN == N_ && diversity_bn > pop_div_high_ && rel_impr < pop_impr_thr_ && success_rate < pop_success_thr_ && N_ > hardMin) {
        int cand = std::max(hardMin, N_ / 2);
        newN = cand;
    }

    newN = std::max(hardMin, std::min(hardMax, newN));
    if (newN == N_) return;

    resizePopulation(newN, sorted_idx);
    pop_cooldown_left_ = pop_cooldown_;
    last_resize_iter_ = iter_;
}

void ARQEig::resizePopulation(int newN, const std::vector<int>& sorted_idx)
{
    if (newN < 4) newN = 4;
    if (newN == N_) return;

    // shrinking: keep elites + sample remaining from best half
    if (newN < N_) {
        int elite = (int)std::round(pop_elite_frac_ * (double)newN);
        elite = std::max(2, elite);
        elite = std::min(newN, elite);

        int half = std::max(elite, N_ / 2);

        std::vector<int> pool;
        pool.reserve(half);
        for (int i = 0; i < half; ++i) pool.push_back(sorted_idx[i]);
        std::shuffle(pool.begin() + std::min(elite, (int)pool.size()), pool.end(), rng_);

        std::vector<int> keep;
        keep.reserve(newN);

        for (int i = 0; i < elite; ++i) keep.push_back(sorted_idx[i]);
        for (int i = elite; i < newN; ++i) {
            int pick = pool[(i - elite) % (int)pool.size()];
            keep.push_back(pick);
        }

        std::vector<Vec> Xnew;
        std::vector<double> FXnew;
        Xnew.reserve(newN);
        FXnew.reserve(newN);

        for (int id : keep) {
            Xnew.push_back(X_[id]);
            FXnew.push_back(FX_[id]);
        }

        X_.swap(Xnew);
        FX_.swap(FXnew);
        N_ = newN;

        // archive cap depends on N
        trimArchive();

        // eigen basis invalidated
        eig_ready_ = false;
        eig_age_ = 0;

        // refresh best
        for (int i = 0; i < N_; ++i) {
            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        }

        Optimizer::setPopulation(N_);
        return;
    }

    // expanding: inject new individuals (mix uniform + best-centered BN Gaussian)
    int add = newN - N_;
    if (add <= 0) return;

    injectNewIndividuals(add);
    N_ = (int)X_.size();

    // archive cap depends on N
    trimArchive();

    // eigen basis invalidated
    eig_ready_ = false;
    eig_age_ = 0;

    // refresh best
    for (int i = 0; i < N_; ++i) {
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
    }

    Optimizer::setPopulation(N_);
}

void ARQEig::injectNewIndividuals(int addCount)
{
    if (!prob_ || addCount <= 0) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    X_.reserve(N_ + addCount);
    FX_.reserve(N_ + addCount);

    Vec best_bn;
    toBN(best_x_, best_bn);

    for (int k = 0; k < addCount; ++k) {
        if (terminated()) break;

        Vec cand(D, 0.0);

        // 50%: local BN-Gaussian around current best (exploit)
        // 50%: uniform global sample (explore)
        if (randU() < 0.5 && !best_bn.empty()) {
            Vec bn = best_bn;
            for (int j = 0; j < D; ++j) {
                bn[j] += rsigma_ * randN01();
            }
            fromBN(bn, cand);
        } else {
            for (int j = 0; j < D; ++j) {
                double denom = (U[j] - L[j]);
                if (denom <= 0.0) denom = 1.0;
                cand[j] = L[j] + randU() * denom;
            }
        }

        ensureBounds(cand);
        double fc = eval(cand);

        X_.push_back(std::move(cand));
        FX_.push_back(fc);

        if (fc < best_f_) {
            best_f_ = fc;
            best_x_ = X_.back();
        }
    }
}

void ARQEig::one_iteration()
{
    if (!prob_) return;
    if (terminated()) return;
    if (X_.empty()) return;

    trimArchive();

    std::vector<int> idx;
    sortByFitness(idx);

    // recompute eigen basis periodically
    eig_age_++;
    if (!eig_ready_ || eig_age_ >= eiginterval_) {
        recomputeEigenBasis(idx);
    }

    std::vector<double> SF, SCR, gains;
    const int D = prob_->dimension();

    for (int i = 0; i < N_; ++i) {
        if (terminated()) break;

        // choose memory slot
        int r = randInt(0, H_ - 1);
        double muF  = MF_[r];
        double muCR = MCR_[r];

        double F, CR;
        sample_F_CR(F, CR, muF, muCR);

        Vec u(D);

        bool useEig = eig_ready_ && (randU() < peig_);
        if (useEig) makeTrialEig(i, idx, F, CR, u);
        else        makeTrialBase(i, idx, F, CR, u);

        double fu = eval(u);

        selectionRTR(i, u, fu, F, CR, SF, SCR, gains);

        // global best update
        if (fu < best_f_) {
            best_f_ = fu;
            best_x_ = u;
        }
    }

    updateMemories(SF, SCR, gains);
    trimArchive();

    quarantineAndRestart();

    // optional in-run local search (as in EA4Eig)
    if (!local_method_.empty() && local_rate_ > 0.0) {
        for (int i = 0; i < N_; ++i) {
            if (terminated()) break;
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

    // refresh best from population
    for (int i = 0; i < N_; ++i) {
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
    }

    if (best_f_ < best_prev_) { best_prev_ = best_f_; no_improve_ = 0; }
    else { no_improve_++; }

    // ---- APL update & decision ----
    iter_++;

    if (pop_cooldown_left_ > 0) pop_cooldown_left_--;

    if (std::isfinite(best_f_)) {
        best_hist_.push_back(best_f_);
        while ((int)best_hist_.size() > pop_window_) best_hist_.pop_front();
    }

    double success_rate = (N_ > 0) ? ((double)SF.size() / (double)N_) : 0.0;
    int sampleN = std::min(N_, 24);
    double diversity_bn = estimateDiversityBN(sampleN);
    double rel_impr = relImprovementFromHistory();

    // Re-sort before potential resize
    sortByFitness(idx);
    maybeAdaptivePopulationLeap(idx, success_rate, diversity_bn, rel_impr);

    updateStop(FX_);
    printBest();
}

void ARQEig::end()
{
    if (!prob_) return;

    // final local refinement (as in EA4Eig)
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
