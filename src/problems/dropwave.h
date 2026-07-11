#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Drop-Wave function.
 *
 * f(x,y) = -(1 + cos(12*sqrt(x^2+y^2))) / (0.5*(x^2+y^2) + 2)
 *
 * Domain: [-5.12, 5.12]^2
 * Global minimum: f(0,0) = -1
 */
class DropWave : public Problem {
public:
    DropWave();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
