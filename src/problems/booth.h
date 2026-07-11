#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Booth function.
 *
 * f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
 *
 * Domain: [-10, 10]^2
 * Global minimum: f(1,3) = 0
 */
class Booth : public Problem {
public:
    Booth();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
