#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Salomon function.
 *
 * Let r = sqrt( Sum_i x_i^2 ).
 * f(x) = 1 - cos(2*pi*r) + 0.1*r
 *
 * Domain: [-100, 100]^n
 * Global minimum: f(0,...,0) = 0
 */
class Salomon : public Problem {
public:
    Salomon();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
