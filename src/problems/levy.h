#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Levy N.13 benchmark function.
 *
 *  w_i = 1 + (x_i - 1)/4
 *  f(x) = sin^2(π w_1)
 *       + Σ_{i=1}^{D-1} (w_i - 1)^2 [1 + 10 sin^2(π w_i + 1)]
 *       + (w_D - 1)^2 [1 + sin^2(2π w_D)]
 *
 * Domain: [-10, 10]^D
 * Global minimum: x* = (1,...,1), f(x*) = 0.
 */
class Levy : public Problem {
public:
    Levy();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
