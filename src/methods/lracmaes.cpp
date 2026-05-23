#include "lracmaes.h"
#include "init.h"
#include "options.h"

#include <numeric>   // iota
#include <cassert>

namespace optimsolution {

// =========================================================================
//  configure
// =========================================================================
void LRACMAES::configure(const MethodConfig& mc)
{
    lambda_cfg_ = mc.getInt("lambda", lambda_cfg_);
    pop_cfg_    = mc.getInt("population", pop_cfg_);

    sigma0_     = mc.getDbl("sigma0", sigma0_);
    if (sigma0_ <= 0.0) sigma0_ = 0.3;

    // LRA hyper-parameters (sensible defaults from Nomura et al.)
    alpha_lra_  = mc.getDbl("lra_alpha",      alpha_lra_);
    beta_mean_  = mc.getDbl("lra_beta_mean",  beta_mean_);
    beta_Sigma_ = mc.getDbl("lra_beta_sigma", beta_Sigma_);
    gamma_lra_  = mc.getDbl("lra_gamma",      gamma_lra_);

    // In-run local search
    local_method_ = mc.getStr("local_method", local_method_);
    local_rate_   = mc.getDbl("local_rate",   local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    // Final local
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method",  end_local_method_);
}

// =========================================================================
//  init
// =========================================================================
void LRACMAES::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    iter_ = 0;

    // ---- population size (Hansen default: 4 + floor(3 ln D)) ----
    int lambda_auto = 4 + static_cast<int>(3.0 * std::log(static_cast<double>(D)));
    if (lambda_auto < 4) lambda_auto = 4;

    if      (lambda_cfg_ > 0)  lambda_ = lambda_cfg_;
    else if (pop_cfg_    > 0)  lambda_ = pop_cfg_;
    else                       lambda_ = lambda_auto;
    if (lambda_ < 4) lambda_ = 4;

    mu_ = lambda_ / 2;
    if (mu_ < 1) mu_ = 1;

    // ---- active weights (positive for best mu, negative for worst) ----
    // Reference: Hansen tutorial eq. 49-53; cmaes Python library _cma.py
    std::vector<double> w_prime(lambda_);
    for (int i = 0; i < lambda_; ++i)
        w_prime[i] = std::log((lambda_ + 1.0) / 2.0) - std::log(i + 1.0);

    // mu_eff for positive weights
    double pos_sum = 0.0, pos_sq = 0.0;
    for (int i = 0; i < mu_; ++i) {
        pos_sum += w_prime[i];
        pos_sq  += w_prime[i] * w_prime[i];
    }
    mu_eff_ = (pos_sum * pos_sum) / pos_sq;

    // mu_eff for negative weights
    double neg_sum = 0.0, neg_sq = 0.0;
    for (int i = mu_; i < lambda_; ++i) {
        neg_sum += w_prime[i];            // these are negative
        neg_sq  += w_prime[i] * w_prime[i];
    }
    double mu_eff_minus = (neg_sum != 0.0) ? (neg_sum * neg_sum) / neg_sq : 0.0;

    // ---- strategy parameters (Hansen defaults) ----
    double alpha_cov = 2.0;
    c1_      = alpha_cov / ((D + 1.3) * (D + 1.3) + mu_eff_);
    c_mu_    = std::min(1.0 - c1_ - 1e-8,
                   alpha_cov * (mu_eff_ - 2.0 + 1.0 / mu_eff_) /
                   ((D + 2.0) * (D + 2.0) + alpha_cov * mu_eff_ / 2.0));

    c_sigma_ = (mu_eff_ + 2.0) / (D + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0,
                   std::sqrt((mu_eff_ - 1.0) / (D + 1.0)) - 1.0) + c_sigma_;

    c_c_     = (4.0 + mu_eff_ / D) / (D + 4.0 + 2.0 * mu_eff_ / D);
    chiN_    = std::sqrt((double)D) * (1.0 - 1.0/(4.0*D) + 1.0/(21.0*D*D));

    // min_alpha for negative-weight scaling (eq. 50-52)
    double min_alpha = 1.0;
    if (c_mu_ > 0.0)
        min_alpha = std::min(min_alpha, (1.0 - c1_ - c_mu_) / (D * c_mu_));
    min_alpha = std::min(min_alpha, 1.0 + c1_ / c_mu_);
    min_alpha = std::min(min_alpha,
                    1.0 + (2.0 * mu_eff_minus) / (mu_eff_ + 2.0));

    // normalize weights (eq. 53)
    double abs_neg_sum = 0.0;
    for (int i = mu_; i < lambda_; ++i) abs_neg_sum += std::fabs(w_prime[i]);

    w_.resize(lambda_);
    for (int i = 0; i < mu_; ++i)
        w_[i] = w_prime[i] / pos_sum;                       // positive, sum = 1
    for (int i = mu_; i < lambda_; ++i)
        w_[i] = (abs_neg_sum > 0.0)
                ? min_alpha / abs_neg_sum * w_prime[i]       // negative
                : 0.0;

    // ---- eigen schedule ----
    eigen_period_  = std::max(1, static_cast<int>(1.0 / ((c1_ + c_mu_) * D * 10.0)));
    eigen_counter_ = 0;

    setPopulation(lambda_);

    // ---- mean initialization ----
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    m_.assign(D, 0.0);
    bool have_bounds = (static_cast<int>(L.size()) == D &&
                        static_cast<int>(U.size()) == D);
    if (have_bounds) {
        for (int j = 0; j < D; ++j) m_[j] = 0.5 * (L[j] + U[j]);
    } else {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto X0 = initSampler.samplePopulation(*prob_, rng_, 1);
        if (!X0.empty() && static_cast<int>(X0[0].size()) == D)
            m_ = X0[0];
    }

    // initial sigma
    if (have_bounds) {
        double avg_range = 0.0;
        for (int j = 0; j < D; ++j) avg_range += (U[j] - L[j]);
        avg_range /= (double)D;
        if (avg_range <= 0.0) avg_range = 1.0;
        sigma_ = sigma0_ * avg_range;
    } else {
        sigma_ = sigma0_;
    }

    // ---- covariance, eigenvectors, paths ----
    C_.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) C_[i][i] = 1.0;

    B_.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) B_[i][i] = 1.0;

    diagD_.assign(D, 1.0);
    p_sigma_.assign(D, 0.0);
    p_c_.assign(D, 0.0);

    // ---- LRA state ----
    E_mean_.assign(D, 0.0);
    E_Sigma_.assign(D * D, 0.0);
    V_mean_    = 0.0;
    V_Sigma_   = 0.0;
    eta_mean_  = 1.0;
    eta_Sigma_ = 1.0;

    // ---- evaluate initial mean ----
    best_x_ = m_;
    best_f_ = eval(best_x_);
    FX_.assign(1, best_f_);

    updateStop(FX_);
    printBest();
}

// =========================================================================
//  linear-algebra helpers (identical to cmaes.cpp)
// =========================================================================
LRACMAES::Vec LRACMAES::matVecMul(const Mat& A, const Vec& x) const
{
    const int D = (int)x.size();
    Vec y(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}

LRACMAES::Vec LRACMAES::matTVecMul(const Mat& A, const Vec& x) const
{
    const int D = (int)x.size();
    Vec y(D, 0.0);
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < D; ++i) s += A[i][j] * x[i];
        y[j] = s;
    }
    return y;
}

void LRACMAES::symOuterAdd(Mat& A, const Vec& x, double w)
{
    const int D = (int)x.size();
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) {
            double d = w * x[i] * x[j];
            A[i][j] += d;
            if (j != i) A[j][i] += d;
        }
    }
}

void LRACMAES::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = prob_->dimension();
    const int m = std::min(D, (int)x.size());
    bool have = ((int)L.size() == D && (int)U.size() == D);

    if (!have) {
        for (int j = 0; j < (int)x.size(); ++j)
            if (!std::isfinite(x[j])) x[j] = 0.0;
        return;
    }
    for (int j = 0; j < m; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// =========================================================================
//  Jacobi eigen-decomposition  (symmetric A → evecs, evals = sqrt(eigenvalues))
// =========================================================================
void LRACMAES::eigenDecomposition(const Mat& Cin, Mat& evecs, Vec& evals)
{
    const int D = (int)Cin.size();
    if (D <= 0) return;

    Mat A = Cin;
    // enforce symmetry
    for (int i = 0; i < D; ++i)
        for (int j = i+1; j < D; ++j) {
            double v = 0.5 * (A[i][j] + A[j][i]);
            A[i][j] = A[j][i] = v;
        }

    evecs.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) evecs[i][i] = 1.0;

    const int maxIter = std::max(50 * D * D, 10);
    const double eps  = 1e-12;

    for (int it = 0; it < maxIter; ++it) {
        int p = 0, q = 1;
        double mx = std::fabs(A[0][1]);
        for (int i = 0; i < D; ++i)
            for (int j = i+1; j < D; ++j) {
                double v = std::fabs(A[i][j]);
                if (v > mx) { mx = v; p = i; q = j; }
            }
        if (mx < eps) break;

        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(phi), s = std::sin(phi);

        for (int k = 0; k < D; ++k) {
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < D; ++k) {
            double apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk;
            A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < D; ++k) {
            double bkp = evecs[k][p], bkq = evecs[k][q];
            evecs[k][p] = c * bkp - s * bkq;
            evecs[k][q] = s * bkp + c * bkq;
        }
    }

    evals.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double val = A[i][i];
        if (val < 0.0) val = 0.0;
        double d = std::sqrt(val);
        if (d < 1e-16) d = 1e-16;
        evals[i] = d;
    }
}

// =========================================================================
//  LRA: Learning Rate Adaptation  (Nomura et al., 2023)
// =========================================================================
void LRACMAES::lrAdaptation(const Vec& old_mean, double old_sigma,
                              const Mat& old_C, const Mat& old_invsqrtC)
{
    const int D = (int)m_.size();
    const int D2 = D * D;

    // --- Δm = new_mean - old_mean ---
    Vec delta_m(D);
    for (int j = 0; j < D; ++j) delta_m[j] = m_[j] - old_mean[j];

    // --- old Σ = old_sigma² · old_C,  new Σ = sigma² · C ---
    // --- ΔΣ = new Σ - old Σ  (stored flat, row-major) ---
    Vec delta_Sigma_flat(D2);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            delta_Sigma_flat[i * D + j] =
                sigma_ * sigma_ * C_[i][j] - old_sigma * old_sigma * old_C[i][j];

    // --- local coordinates: Σ_old^{-1/2} = old_invsqrtC / old_sigma ---
    // locΔm = Σ_old^{-1/2} · Δm
    Vec loc_dm(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int j = 0; j < D; ++j)
            s += old_invsqrtC[i][j] * delta_m[j];
        loc_dm[i] = s / old_sigma;
    }

    // locΔΣ = vec( Σ_old^{-1/2} · ΔΣ · Σ_old^{-1/2} ) / sqrt(2)
    // First compute tmp = old_invsqrtC/old_sigma · ΔΣ (as DxD)
    // Then  result = tmp · old_invsqrtC^T/old_sigma   (symmetric, so same)
    Vec loc_dS(D2, 0.0);
    {
        // temp = (old_invsqrtC / old_sigma) * delta_Sigma_mat * (old_invsqrtC / old_sigma)^T
        // Since invsqrtC is symmetric: temp = inv * dS * inv
        // Do in two steps: A = inv * dS,  then result = A * inv^T = A * inv
        Mat tmp(D, Vec(D, 0.0));
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (int k = 0; k < D; ++k)
                    s += old_invsqrtC[i][k] * delta_Sigma_flat[k * D + j];
                tmp[i][j] = s / old_sigma;
            }
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (int k = 0; k < D; ++k)
                    s += tmp[i][k] * old_invsqrtC[j][k];  // inv^T = inv (symmetric)
                loc_dS[i * D + j] = s / (old_sigma * std::sqrt(2.0));
            }
    }

    // --- moving averages E, V ---
    double norm_dm2 = 0.0;
    for (int j = 0; j < D; ++j) {
        E_mean_[j] = (1.0 - beta_mean_) * E_mean_[j] + beta_mean_ * loc_dm[j];
        norm_dm2 += loc_dm[j] * loc_dm[j];
    }
    V_mean_ = (1.0 - beta_mean_) * V_mean_ + beta_mean_ * norm_dm2;

    double norm_dS2 = 0.0;
    for (int j = 0; j < D2; ++j) {
        E_Sigma_[j] = (1.0 - beta_Sigma_) * E_Sigma_[j] + beta_Sigma_ * loc_dS[j];
        norm_dS2 += loc_dS[j] * loc_dS[j];
    }
    V_Sigma_ = (1.0 - beta_Sigma_) * V_Sigma_ + beta_Sigma_ * norm_dS2;

    // --- estimate SNR ---
    double sqnorm_Em = 0.0;
    for (int j = 0; j < D; ++j) sqnorm_Em += E_mean_[j] * E_mean_[j];

    double sqnorm_ES = 0.0;
    for (int j = 0; j < D2; ++j) sqnorm_ES += E_Sigma_[j] * E_Sigma_[j];

    double denom_m = V_mean_ - sqnorm_Em;
    double snr_mean = (std::fabs(denom_m) > 1e-30)
        ? (sqnorm_Em - (beta_mean_ / (2.0 - beta_mean_)) * V_mean_) / denom_m
        : 0.0;

    double denom_S = V_Sigma_ - sqnorm_ES;
    double snr_Sigma = (std::fabs(denom_S) > 1e-30)
        ? (sqnorm_ES - (beta_Sigma_ / (2.0 - beta_Sigma_)) * V_Sigma_) / denom_S
        : 0.0;

    // --- update η ---
    double before_eta_mean = eta_mean_;

    // η_mean
    double rel_m = snr_mean / (alpha_lra_ * eta_mean_) - 1.0;
    rel_m = std::max(-1.0, std::min(1.0, rel_m));
    eta_mean_ *= std::exp(
        std::min(gamma_lra_ * eta_mean_, beta_mean_) * rel_m);
    eta_mean_ = std::min(eta_mean_, 1.0);

    // η_Σ
    double rel_S = snr_Sigma / (alpha_lra_ * eta_Sigma_) - 1.0;
    rel_S = std::max(-1.0, std::min(1.0, rel_S));
    eta_Sigma_ *= std::exp(
        std::min(gamma_lra_ * eta_Sigma_, beta_Sigma_) * rel_S);
    eta_Sigma_ = std::min(eta_Sigma_, 1.0);

    // --- apply scaled updates ---
    // m = old_m + η_mean · Δm
    for (int j = 0; j < D; ++j)
        m_[j] = old_mean[j] + eta_mean_ * delta_m[j];

    // Σ_new = old_Σ + η_Σ · ΔΣ   then decompose back to sigma, C
    Mat Sigma(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            Sigma[i][j] = old_sigma * old_sigma * old_C[i][j]
                        + eta_Sigma_ * delta_Sigma_flat[i * D + j];

    // decompose Σ into σ² C:  σ = exp( mean(log(eigenvalues)) / 2 )
    Mat Sevecs;
    Vec Sevals;
    eigenDecomposition(Sigma, Sevecs, Sevals);   // Sevals = sqrt(eigenvalues)

    double log_sum = 0.0;
    for (int i = 0; i < D; ++i) {
        double ev = Sevals[i] * Sevals[i];       // actual eigenvalue
        if (ev < 1e-30) ev = 1e-30;
        log_sum += std::log(ev);
    }
    sigma_ = std::exp(log_sum / (2.0 * D));
    if (sigma_ < 1e-30) sigma_ = 1e-30;
    if (sigma_ > 1e32)  sigma_ = 1e32;

    // C = Σ / σ²
    double inv_s2 = 1.0 / (sigma_ * sigma_);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            C_[i][j] = Sigma[i][j] * inv_s2;

    // re-compute B_, diagD_ from the new C_
    eigenDecomposition(C_, B_, diagD_);

    // step-size correction (Nomura et al.):  σ *= η_mean_before / η_mean_after
    if (eta_mean_ > 1e-30)
        sigma_ *= before_eta_mean / eta_mean_;
}

// =========================================================================
//  one_iteration
// =========================================================================
void LRACMAES::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    // B_ and diagD_ are kept current by lrAdaptation() at the end of
    // every iteration.  On the very first call they are identity/ones
    // (set in init()), which is correct for C_ = I.

    // ---- save old state for LRA ----
    Vec old_mean = m_;
    double old_sigma = sigma_;
    Mat old_C = C_;

    // Build C^{-1/2} = B · D^{-1} · B^T  (used for p_sigma and LRA)
    Mat invsqrtC(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += B_[i][k] * (1.0 / diagD_[k]) * B_[j][k];
            invsqrtC[i][j] = s;
        }

    // ---- sample offspring ----
    X_.assign(lambda_, Vec(D, 0.0));
    FX_.assign(lambda_, std::numeric_limits<double>::infinity());

    std::normal_distribution<double> N01(0.0, 1.0);

    for (int k = 0; k < lambda_ && prob_->calls() < max_evals_; ++k) {
        Vec z(D);
        for (int j = 0; j < D; ++j) z[j] = N01(rng_);

        Vec Dz(D);
        for (int j = 0; j < D; ++j) Dz[j] = diagD_[j] * z[j];
        Vec y = matVecMul(B_, Dz);

        Vec x(D);
        for (int j = 0; j < D; ++j) x[j] = m_[j] + sigma_ * y[j];
        ensureBounds(x);

        X_[k]  = x;
        FX_[k] = eval(X_[k]);

        if (FX_[k] < best_f_) {
            best_f_ = FX_[k];
            best_x_ = X_[k];
        }
    }

    for (int k = 0; k < lambda_; ++k)
        if (!std::isfinite(FX_[k]))
            FX_[k] = std::numeric_limits<double>::infinity();

    // ---- sort by fitness ----
    std::vector<int> idx(lambda_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return FX_[a] < FX_[b];
    });

    // ---- old mean ----
    Vec m_old = m_;

    // ---- new mean: weighted recombination (positive weights only) ----
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < mu_; ++i)
            s += w_[i] * X_[idx[i]][j];
        m_[j] = s;
    }

    // ---- y_w = (m_new - m_old) / sigma ----
    Vec y_step(D);
    for (int j = 0; j < D; ++j)
        y_step[j] = (m_[j] - m_old[j]) / sigma_;

    // ---- p_sigma (CSA) ----
    Vec z_step = matVecMul(invsqrtC, y_step);   // C^{-1/2} · y_w

    double cs_fac = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);
    for (int j = 0; j < D; ++j)
        p_sigma_[j] = (1.0 - c_sigma_) * p_sigma_[j] + cs_fac * z_step[j];

    double norm_ps = 0.0;
    for (int j = 0; j < D; ++j) norm_ps += p_sigma_[j] * p_sigma_[j];
    norm_ps = std::sqrt(norm_ps);

    // ---- h_sigma ----
    double ps_corr = norm_ps /
        std::sqrt(1.0 - std::pow(1.0 - c_sigma_, 2.0 * (iter_ + 1)));
    double h_sigma = (ps_corr < (1.4 + 2.0 / (D + 1.0)) * chiN_) ? 1.0 : 0.0;

    // ---- p_c ----
    double cc_fac = std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);
    for (int j = 0; j < D; ++j)
        p_c_[j] = (1.0 - c_c_) * p_c_[j] + h_sigma * cc_fac * y_step[j];

    // ---- covariance update (active CMA: uses all lambda weights) ----
    Mat C_new = C_;

    // scaling factor for old C
    double w_sum_all = 0.0;
    for (int i = 0; i < lambda_; ++i) w_sum_all += w_[i];

    double delta_hs = (1.0 - h_sigma) * c_c_ * (2.0 - c_c_);
    double factor_C = 1.0 + c1_ * delta_hs - c1_ - c_mu_ * w_sum_all;
    if (factor_C < 0.0) factor_C = 0.0;

    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            C_new[i][j] *= factor_C;

    // rank-one: c1 · p_c · p_c^T
    symOuterAdd(C_new, p_c_, c1_);

    // rank-mu (active): uses all lambda weights (positive + negative)
    // For negative weights, apply the correction ||C^{-1/2} y_i||^{-2} · n
    for (int i = 0; i < lambda_; ++i) {
        int k = idx[i];
        Vec y_i(D);
        for (int j = 0; j < D; ++j)
            y_i[j] = (X_[k][j] - m_old[j]) / sigma_;

        double wi = w_[i];
        if (wi < 0.0) {
            // active weight correction: w_io = w * n / ||C^{-1/2} y_i||^2
            Vec Cinv_y = matVecMul(invsqrtC, y_i);
            double nrm2 = 0.0;
            for (int j = 0; j < D; ++j) nrm2 += Cinv_y[j] * Cinv_y[j];
            if (nrm2 > 1e-30)
                wi = wi * D / nrm2;
        }
        symOuterAdd(C_new, y_i, c_mu_ * wi);
    }

    C_.swap(C_new);

    // ---- step-size update (must be LAST before LRA, per Hansen) ----
    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (norm_ps / chiN_ - 1.0));
    if (sigma_ > 1e32) sigma_ = 1e32;

    // ---- LRA: adapt learning rates and scale back updates ----
    lrAdaptation(old_mean, old_sigma, old_C, invsqrtC);

    // ---- optional in-run local search ----
    if (!local_method_.empty() && local_rate_ > 0.0 && prob_->calls() < max_evals_) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto res = localSearch(local_method_, best_x_);
            const Vec& xloc = res.first;
            double floc = res.second;
            if (!xloc.empty() && (int)xloc.size() == D &&
                std::isfinite(floc) && floc < best_f_) {
                best_x_ = xloc;
                best_f_ = floc;
            }
        }
    }

    updateStop(FX_);
    printBest();
    ++iter_;
}

// =========================================================================
//  end
// =========================================================================
void LRACMAES::end()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        const Vec& xloc = res.first;
        double     floc = res.second;
        if (!xloc.empty() && (int)xloc.size() == D &&
            std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_.empty()
                ? std::numeric_limits<double>::infinity() : FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i)
                if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
            if ((int)best_x_.size() == D) {
                X_[worst] = best_x_;
                if ((int)FX_.size() == (int)X_.size())
                    FX_[worst] = best_f_;
            }
        }
        printBest();
    }

    if (!FX_.empty()) updateStop(FX_);
}

} // namespace optimsolution
