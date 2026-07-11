#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Dixon-Price function.
 *
 * f(x) = (x_1 - 1)^2 + Sum_{i=2}^{n} i * (2*x_i^2 - x_{i-1})^2
 *
 * Domain: [-10, 10]^n
 * Global minimum: f(x*) = 0 at x_i = 2^(-(2^i - 2)/2^i)  (1-indexed i)
 */
class DixonPrice : public Problem {
public:
    DixonPrice();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
