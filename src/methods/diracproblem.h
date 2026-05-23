#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Dirac-like Gaussian spike test function.
 *
 * f(x) = 1 - exp( -||x - μ||^2 / (2 σ^2) )
 *
 * - Domain: by default [-1, 1]^D
 * - Default center μ = 0 (all dimensions)
 * - Default width σ = 0.01 (very sharp spike)
 * - Global minimum: f* = 0 at x = μ
 */
class DiracProblem : public Problem {
public:
    DiracProblem();

    // Sets dimension and default bounds [-1, 1]^D, and μ = 0
    void init(int dim) override;

    // Optional configuration before running optimizers
    void setCenter(const Vec& mu);  // set μ (length adjusted to D)
    void setSigma(double s);        // set σ > 0

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec    mu_;     // center
    double sigma_;  // width of Gaussian spike
};

} // namespace optimsolution
