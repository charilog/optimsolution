#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Colville function (4D).
 *
 * f(x1,x2,x3,x4) = 100*(x1^2-x2)^2 + (x1-1)^2 + (x3-1)^2
 *                  + 90*(x3^2-x4)^2
 *                  + 10.1*[(x2-1)^2 + (x4-1)^2]
 *                  + 19.8*(x2-1)*(x4-1)
 *
 * Domain: [-10, 10]^4
 * Global minimum: f(1,1,1,1) = 0
 */
class Colville : public Problem {
public:
    Colville();
    void init(int dim) override;                  // force D=4, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
