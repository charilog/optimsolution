#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Beale function.
 *
 * f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
 *
 * Domain: [-4.5, 4.5]^2
 * Global minimum: f(3, 0.5) = 0
 */
class Beale : public Problem {
public:
    Beale();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
