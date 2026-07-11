#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Matyas function.
 *
 * f(x,y) = 0.26*(x^2 + y^2) - 0.48*x*y
 *
 * Domain: [-10, 10]^2
 * Global minimum: f(0,0) = 0
 */
class Matyas : public Problem {
public:
    Matyas();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
