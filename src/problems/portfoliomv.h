#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * Markowitz Mean–Variance Portfolio (long-only, sum-to-one via soft penalty).
 *
 * Decision: w[0..N-1] in [0,1], N = dim at init().
 * Objective:
 *   f(w) = w_risk * (w^T Σ w) - w_ret * (μ^T w) + w_sum * (sum(w)-1)^2
 *
 * Σ: Toeplitz covariance with corr ρ^{|i-j|} and per-asset σ_i (mild heteroskedasticity).
 * μ: deterministic increasing profile from 2% to 8%.
 */
class PortfolioMV : public Problem {
public:
    PortfolioMV();

    void init(int dim) override;                   // set dimension and bounds

protected:
    double evaluate_core(const Vec& w) override;   // objective value
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    // Weights
    double w_risk_ = 1.0;
    double w_ret_  = 0.5;
    double w_sum_  = 50.0;

    // Covariance model parameters
    double sigma_  = 0.20;     // base volatility
    double rho_    = 0.3;      // correlation base
    double ridge_  = 1e-6;     // SPD guard

    // Built after init(n)
    int N_ = 0;
    std::vector<double> mu_;        // size N_
    std::vector<double> cov_;       // row-major N_ x N_

    inline double cov(int i, int j) const { return cov_[N_*i + j]; }
    void buildProfiles();           // fills mu_ and cov_ deterministically
};

} // namespace optimsolution
