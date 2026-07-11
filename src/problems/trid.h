#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Trid function.
 *
 * f(x) = Sum_{i=1}^{n} (x_i - 1)^2  -  Sum_{i=2}^{n} x_i * x_{i-1}
 *
 * Domain: [-n^2, n^2]^n
 * Global minimum: x*_i = i*(n+1-i) (1-indexed), f* = -n*(n+4)*(n-1)/6
 */
class Trid : public Problem {
public:
    Trid();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
