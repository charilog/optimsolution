#include "cmaes.h"
#include "init.h"
#include "options.h"

#include <numeric>   // iota

namespace optimsolution {

void CMAES::configure(const MethodConfig& mc)
{
    // Implementation note.
    lambda_cfg_ = mc.getInt("lambda", lambda_cfg_);
    pop_cfg_    = mc.getInt("population", pop_cfg_);

    // initial sigma (relative to bounds)
    sigma0_     = mc.getDbl("sigma0", sigma0_);
    if (sigma0_ <= 0.0) sigma0_ = 0.3;

    // In-run local search (optional, typically small rate)
    local_method_ = mc.getStr("local_method", local_method_);
    local_rate_   = mc.getDbl("local_rate", local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    // Final local
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

void CMAES::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    iter_ = 0;

    // Implementation note.
    int lambda_auto = 4 + static_cast<int>(3.0 * std::log(static_cast<double>(D)));
    if (lambda_auto < 4) lambda_auto = 4;

    if (lambda_cfg_ > 0)      lambda_ = lambda_cfg_;
    else if (pop_cfg_ > 0)    lambda_ = pop_cfg_;
    else                      lambda_ = lambda_auto;

    if (lambda_ < 4) lambda_ = 4;

    // Implementation note.
    mu_ = lambda_ / 2;
    if (mu_ < 1) mu_ = 1;

    w_.resize(mu_);
    for (int i = 0; i < mu_; ++i) {
        w_[i] = std::log(mu_ + 0.5) - std::log(i + 1.0);
    }
    double w_sum = 0.0;
    for (double wi : w_) w_sum += wi;
    for (double &wi : w_) wi /= w_sum;

    double w_sq_sum = 0.0;
    for (double wi : w_) w_sq_sum += wi * wi;
    mu_eff_ = 1.0 / w_sq_sum;

    // --- strategy parameters (Hansen's defaults) ---
    c_sigma_ = (mu_eff_ + 2.0) / (D + mu_eff_ + 5.0);
    d_sigma_ = 1.0 + 2.0 * std::max(0.0,
                  std::sqrt((mu_eff_ - 1.0) / (D + 1.0)) - 1.0) + c_sigma_;
    c_c_     = (4.0 + mu_eff_ / D) / (D + 4.0 + 2.0 * mu_eff_ / D);
    c1_      = 2.0 / ( (D + 1.3) * (D + 1.3) + mu_eff_ );
    c_mu_    = std::min(1.0 - c1_,
                 2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) /
                 ((D + 2.0) * (D + 2.0) + mu_eff_));

    // E[||N(0, I)||]
    chiN_ = std::sqrt(static_cast<double>(D)) * (1.0 - 1.0/(4.0*D) + 1.0/(21.0*D*D));

    // Implementation note.
    eigen_period_  = static_cast<int>( 1.0 / ( (c1_ + c_mu_) * D * 10.0 ) );
    if (eigen_period_ < 1) eigen_period_ = 1;
    eigen_counter_ = 0; // force recompute at first iteration

    // For Optimizer logging
    setPopulation(lambda_);

    // --- initialize mean m from bounds or initializer ---
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    m_.assign(D, 0.0);
    bool have_bounds = (static_cast<int>(L.size()) == D &&
                        static_cast<int>(U.size()) == D);

    if (have_bounds) {
        for (int j = 0; j < D; ++j) {
            m_[j] = 0.5 * (L[j] + U[j]);
        }
    } else {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto X0 = initSampler.samplePopulation(*prob_, rng_, 1);
        if (!X0.empty() && static_cast<int>(X0[0].size()) == D) {
            m_ = X0[0];
        } else {
            for (int j = 0; j < D; ++j) m_[j] = 0.0;
        }
    }

    // initial sigma relative to bounds range
    if (have_bounds) {
        double avg_range = 0.0;
        for (int j = 0; j < D; ++j)
            avg_range += (U[j] - L[j]);
        avg_range /= static_cast<double>(D);
        if (avg_range <= 0.0) avg_range = 1.0;
        sigma_ = sigma0_ * avg_range;
    } else {
        sigma_ = sigma0_;
    }

    // --- covariance, eigenvectors, paths ---
    C_.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) C_[i][i] = 1.0;

    B_.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) B_[i][i] = 1.0;

    diagD_.assign(D, 1.0);
    p_sigma_.assign(D, 0.0);
    p_c_.assign(D, 0.0);

    // Evaluate initial mean as starting best
    best_x_ = m_;
    best_f_ = eval(best_x_);

    // For updateStop / printBest
    FX_.assign(1, best_f_);

    updateStop(FX_);
    printBest();
}

CMAES::Vec CMAES::matVecMul(const Mat& A, const Vec& x) const
{
    const int D = static_cast<int>(x.size());
    Vec y(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) {
            s += A[i][j] * x[j];
        }
        y[i] = s;
    }
    return y;
}

CMAES::Vec CMAES::matTVecMul(const Mat& A, const Vec& x) const
{
    const int D = static_cast<int>(x.size());
    Vec y(D, 0.0);
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < D; ++i) {
            s += A[i][j] * x[i]; // A^T * x
        }
        y[j] = s;
    }
    return y;
}

void CMAES::symOuterAdd(Mat& A, const Vec& x, double w)
{
    const int D = static_cast<int>(x.size());
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) {
            double delta = w * x[i] * x[j];
            A[i][j] += delta;
            if (j != i) A[j][i] += delta;
        }
    }
}

void CMAES::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int  D = prob_->dimension();
    const int  m = static_cast<int>(std::min<std::size_t>(D, x.size()));

    bool have_bounds = (static_cast<int>(L.size()) == D &&
                        static_cast<int>(U.size()) == D);

    if (!have_bounds) {
        for (int j = 0; j < m; ++j) {
            if (!std::isfinite(x[j])) x[j] = 0.0;
        }
        for (std::size_t j = m; j < x.size(); ++j) {
            if (!std::isfinite(x[j])) x[j] = 0.0;
        }
        return;
    }

    for (int j = 0; j < m; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
    for (std::size_t j = m; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.0;
    }
}

// Jacobi eigen-decomposition for symmetric C_, yields B_ (eigenvectors) and diagD_ = sqrt(eigenvalues)
void CMAES::updateEigenSystem()
{
    const int D = prob_->dimension();
    if (D <= 0) return;

    Mat A = C_;   // copy covariance

    // enforce symmetry
    for (int i = 0; i < D; ++i) {
        for (int j = i+1; j < D; ++j) {
            double v = 0.5 * (A[i][j] + A[j][i]);
            A[i][j] = A[j][i] = v;
        }
    }

    // initialize B_ as I
    B_.assign(D, Vec(D, 0.0));
    for (int i = 0; i < D; ++i) B_[i][i] = 1.0;

    const int    maxIter = std::max(50 * D * D, 10);
    const double eps     = 1e-12;

    for (int iter = 0; iter < maxIter; ++iter) {
        int p = 0, q = 1;
        double max_off = std::fabs(A[0][1]);
        for (int i = 0; i < D; ++i) {
            for (int j = i+1; j < D; ++j) {
                double v = std::fabs(A[i][j]);
                if (v > max_off) {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < eps) break;

        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];

        double phi  = 0.5 * std::atan2(2.0 * apq, (aqq - app));
        double c    = std::cos(phi);
        double s    = std::sin(phi);

        // rotate rows & cols p,q in A
        for (int k = 0; k < D; ++k) {
            double akp = A[k][p];
            double akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < D; ++k) {
            double apk = A[p][k];
            double aqk = A[q][k];
            A[p][k] = c * apk - s * aqk;
            A[q][k] = s * apk + c * aqk;
        }

        // update eigenvectors B_
        for (int k = 0; k < D; ++k) {
            double bkp = B_[k][p];
            double bkq = B_[k][q];
            B_[k][p] = c * bkp - s * bkq;
            B_[k][q] = s * bkp + c * bkq;
        }
    }

    diagD_.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double val = A[i][i];
        if (val < 0.0) val = 0.0;
        double d = std::sqrt(val);
        if (d < 1e-16) d = 1e-16;
        diagD_[i] = d;
    }

    eigen_counter_ = eigen_period_;
}

void CMAES::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    // update eigen system when needed
    if (eigen_counter_ <= 0) {
        updateEigenSystem();
    }
    eigen_counter_--;

    // Implementation note.
    X_.assign(lambda_, Vec(D, 0.0));
    FX_.assign(lambda_, std::numeric_limits<double>::infinity());

    std::normal_distribution<double> N01(0.0, 1.0);

    for (int k = 0; k < lambda_ && prob_->calls() < max_evals_; ++k) {
        // z ~ N(0, I)
        Vec z(D, 0.0);
        for (int j = 0; j < D; ++j) z[j] = N01(rng_);

        // y = B * (D * z)
        Vec Dz(D, 0.0);
        for (int j = 0; j < D; ++j) Dz[j] = diagD_[j] * z[j];
        Vec y = matVecMul(B_, Dz);

        // x = m + sigma * y
        Vec x(D, 0.0);
        for (int j = 0; j < D; ++j) x[j] = m_[j] + sigma_ * y[j];

        ensureBounds(x);

        X_[k]  = x;
        FX_[k] = eval(X_[k]);

        if (FX_[k] < best_f_) {
            best_f_ = FX_[k];
            best_x_ = X_[k];
        }
    }

    // make sure FX_ is finite
    for (int k = 0; k < lambda_; ++k) {
        if (!std::isfinite(FX_[k])) {
            FX_[k] = std::numeric_limits<double>::infinity();
        }
    }

    // Implementation note.
    std::vector<int> idx(lambda_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return FX_[a] < FX_[b];
    });

    // old mean
    Vec m_old = m_;

    // new mean
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < mu_; ++i) {
            int k = idx[i];
            s += w_[i] * X_[k][j];
        }
        m_[j] = s;
    }

    // --- evolution path for sigma & sigma update ---
    // y_w = (m_new - m_old)/sigma
    Vec y_step(D, 0.0);
    for (int j = 0; j < D; ++j) {
        y_step[j] = (m_[j] - m_old[j]) / sigma_;
    }

    // C^{-1/2} y_step = B * D^{-1} * B^T * y_step
    Vec u = matTVecMul(B_, y_step);  // B^T y
    for (int j = 0; j < D; ++j) {
        u[j] /= diagD_[j];
    }
    Vec z_step = matVecMul(B_, u);   // B D^{-1} B^T y

    double cs_factor = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);
    for (int j = 0; j < D; ++j) {
        p_sigma_[j] = (1.0 - c_sigma_) * p_sigma_[j] + cs_factor * z_step[j];
    }

    double norm_p_sigma = 0.0;
    for (int j = 0; j < D; ++j) norm_p_sigma += p_sigma_[j] * p_sigma_[j];
    norm_p_sigma = std::sqrt(norm_p_sigma);

    // NOTE: sigma update is deferred to AFTER the covariance matrix update,
    // per Hansen's CMA-ES tutorial (arXiv:1604.00772).  The rank-mu term
    // in the C update must divide by the OLD sigma (the one used for sampling).

    // --- evolution path for C ---
    double ps_norm_corr = norm_p_sigma /
        std::sqrt(1.0 - std::pow(1.0 - c_sigma_, 2.0 * (iter_ + 1)));
    double h_sigma = (ps_norm_corr
                      < (1.4 + 2.0 / (D + 1.0)) * chiN_) ? 1.0 : 0.0;

    double cc_factor = std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);
    for (int j = 0; j < D; ++j) {
        p_c_[j] = (1.0 - c_c_) * p_c_[j] + h_sigma * cc_factor * y_step[j];
    }

    // --- covariance update (uses OLD sigma_, before step-size adaptation) ---
    Mat C_new = C_;

    // Implementation note.
    double factor_C = 1.0 - c1_ - c_mu_ + c1_ * (1.0 - h_sigma) * c_c_ * (2.0 - c_c_);
    if (factor_C < 0.0) factor_C = 0.0;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            C_new[i][j] *= factor_C;
        }
    }

    // rank-one term: + c1 p_c p_c^T
    symOuterAdd(C_new, p_c_, c1_);

    // rank-mu term: + cmu * sum_i( w_i * y_i * y_i^T )
    // y_i = (x_i:lambda - m_old) / sigma   (OLD sigma, before adaptation)
    for (int i = 0; i < mu_; ++i) {
        int k = idx[i];
        Vec y_i(D, 0.0);
        for (int j = 0; j < D; ++j) {
            y_i[j] = (X_[k][j] - m_old[j]) / sigma_;
        }
        symOuterAdd(C_new, y_i, c_mu_ * w_[i]);
    }

    C_.swap(C_new);

    // --- step-size update (must be LAST, after C update) ---
    // Reference: Hansen tutorial, Section 'The CMA-ES', update order.
    sigma_ *= std::exp( (c_sigma_ / d_sigma_) * (norm_p_sigma / chiN_ - 1.0) );

    // In-run local search: optionally apply to current best
    if (!local_method_.empty() && local_rate_ > 0.0 && prob_->calls() < max_evals_) {
        std::uniform_real_distribution<double> Uloc(0.0, 1.0);
        if (Uloc(rng_) < local_rate_) {
            auto res = localSearch(local_method_, best_x_);
            const Vec& xloc = res.first;
            double     floc = res.second;
            if (!xloc.empty() &&
                static_cast<int>(xloc.size()) == D &&
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

void CMAES::end()
{
    if (!prob_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        const Vec& xloc = res.first;
        double     floc = res.second;
        if (!xloc.empty() &&
            static_cast<int>(xloc.size()) == D &&
            std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }

        if (!X_.empty()) {
            int    worst = 0;
            double fw    = (FX_.empty()
                            ? std::numeric_limits<double>::infinity()
                            : FX_[0]);
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) {
                    fw = FX_[i];
                    worst = i;
                }
            }
            if ((int)best_x_.size() == D) {
                X_[worst]  = best_x_;
                if ((int)FX_.size() == (int)X_.size())
                    FX_[worst] = best_f_;
            }
        }

        printBest();
    }

    if (!FX_.empty())
        updateStop(FX_);
}

} // namespace optimsolution
