#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * McCormick function.
 *
 * f(x,y) = sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1
 *
 * Domain: x in [-1.5, 4], y in [-3, 4]
 * Global minimum: f(-0.54719, -1.54719) ~= -1.9133
 */
class McCormick : public Problem {
public:
    McCormick();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
