#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Powell singular function.
 *
 * For each group of 4 consecutive variables (a,b,c,d) = (x_{4k},x_{4k+1},x_{4k+2},x_{4k+3}):
 *   term = (a+10b)^2 + 5*(c-d)^2 + (b-2c)^4 + 10*(a-d)^4
 * f(x) = Sum over all groups of "term"
 *
 * D must be a multiple of 4 (rounded up if not, in init()).
 * Domain: [-4, 5]^D
 * Global minimum: f(0,...,0) = 0
 */
class Powell : public Problem {
public:
    Powell();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
