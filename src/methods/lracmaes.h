#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig; // forward declaration

// LRA-CMA-ES: CMA-ES with Learning Rate Adaptation
// Reference: Nomura, Akimoto, Ono (2023)
//   "CMA-ES with Learning Rate Adaptation"
//   GECCO '23 / ACM Trans. Evol. Learn. (2024)
//   arXiv:2304.03473 / arXiv:2401.15876
//
// Implements the active (μ/μ_w, λ)-CMA-ES with an adaptive learning rate
// mechanism that estimates the signal-to-noise ratio (SNR) of parameter
// updates and scales them accordingly.  This enables the algorithm to solve
// multimodal and noisy problems with the default population size.
class LRACMAES : public Optimizer {
public:
    LRACMAES() = default;
    ~LRACMAES() override = default;

    std::string methodShortName() const override { return "lracmaes"; }
    std::string methodFullName()  const override {
        return "CMA-ES with Learning Rate Adaptation";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    using Mat = std::vector<Vec>;  // row-major: mat[i][j]

    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);

    // linear algebra helpers
    void   eigenDecomposition(const Mat& A, Mat& evecs, Vec& evals);
    Vec    matVecMul(const Mat& A, const Vec& x) const;
    Vec    matTVecMul(const Mat& A, const Vec& x) const;
    void   symOuterAdd(Mat& A, const Vec& x, double w);

    // LRA sub-routine: adapt eta_mean_ and eta_Sigma_ after a full CMA step
    void   lrAdaptation(const Vec& old_mean, double old_sigma,
                        const Mat& old_C, const Mat& old_invsqrtC);

private:
    // --- Config / population ---
    int    lambda_cfg_{-1};
    int    pop_cfg_{-1};
    int    lambda_{0};
    int    mu_{0};

    // step-size
    double sigma0_{0.3};
    double sigma_{0.3};

    // CMA strategy parameters (Hansen defaults)
    double c_sigma_{0.0};
    double d_sigma_{0.0};
    double c_c_{0.0};
    double c1_{0.0};
    double c_mu_{0.0};
    double chiN_{0.0};

    // eigen decomposition schedule
    int    eigen_period_{0};
    int    eigen_counter_{0};

    // In-run / final local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // --- CMA-ES State ---
    int         iter_{0};

    Vec         m_;            // mean
    Mat         C_;            // covariance matrix
    Mat         B_;            // eigenvectors  (columns)
    Vec         diagD_;        // sqrt(eigenvalues)
    Vec         p_sigma_;      // evolution path for σ
    Vec         p_c_;          // evolution path for C

    // active weights:  w_[0..lambda-1], first mu are positive, rest negative
    std::vector<double> w_;
    double              mu_eff_{0.0};

    // --- LRA State ---
    double alpha_lra_{1.4};      // target SNR
    double beta_mean_{0.1};      // EMA smoothing for mean
    double beta_Sigma_{0.03};    // EMA smoothing for Σ
    double gamma_lra_{0.1};      // meta-learning-rate for η

    Vec    E_mean_;              // EMA of local Δm          (D×1)
    Vec    E_Sigma_;             // EMA of local vec(ΔΣ)     (D²×1)
    double V_mean_{0.0};         // EMA of ||local Δm||²
    double V_Sigma_{0.0};        // EMA of ||local vec(ΔΣ)||²
    double eta_mean_{1.0};       // adaptive learning rate for m
    double eta_Sigma_{1.0};      // adaptive learning rate for Σ

    // current offspring
    std::vector<Vec>    X_;
    std::vector<double> FX_;
};

} // namespace optimsolution
