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

// Implementation note.
class CMAES : public Optimizer {
public:
    CMAES() = default;
    ~CMAES() override = default;
	std::string methodShortName() const override { return "cmaes"; }
	std::string methodFullName()  const override { return "Covariance Matrix Adaptation Evolution Strategy"; }

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

    void   updateEigenSystem();       // Jacobi eigen-decomposition for C
    Vec    matVecMul(const Mat& A, const Vec& x) const;  // A * x
    Vec    matTVecMul(const Mat& A, const Vec& x) const; // A^T * x
    void   symOuterAdd(Mat& A, const Vec& x, double w);  // A += w * (x x^T)

private:
    // --- Config / population ---
    int    lambda_cfg_{-1};    // "lambda" from cfg, if given
    int    pop_cfg_{-1};       // "population" from cfg (fallback)
    int    lambda_{0};         // offspring population size
    int    mu_{0};             // number of selected parents

    // step-size
    double sigma0_{0.3};       // Implementation note.
    double sigma_{0.3};

    // Implementation note.
    double c_sigma_{0.0};
    double d_sigma_{0.0};
    double c_c_{0.0};
    double c1_{0.0};
    double c_mu_{0.0};
    double chiN_{0.0};         // E||N(0,I)||

    // eigen decomposition schedule
    int    eigen_period_{0};
    int    eigen_counter_{0};

    // In-run / final local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // --- State ---
    int         iter_{0};

    Vec         m_;            // mean
    Mat         C_;            // covariance matrix
    Mat         B_;            // eigenvectors
    Vec         diagD_;        // sqrt eigenvalues (D on diagonal)
    Vec         p_sigma_;      // evolution path for sigma
    Vec         p_c_;          // evolution path for covariance

    // selection weights
    std::vector<double> w_;    // Implementation note.
    double              mu_eff_{0.0};

    // current offspring (for logging / updateStop)
    std::vector<Vec>    X_;
    std::vector<double> FX_;
};

} // namespace optimsolution
