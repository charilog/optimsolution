#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Whitley function.
 *
 * Let y_ij = 100*(x_i^2 - x_j)^2 + (1 - x_j)^2.
 * f(x) = Sum_{i=1}^n Sum_{j=1}^n [ y_ij^2/4000 - cos(y_ij) + 1 ]
 *
 * Domain: [-10.24, 10.24]^n
 * Global minimum: f(1,...,1) = 0
 */
class Whitley : public Problem {
public:
    Whitley();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
