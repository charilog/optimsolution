#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Cross-in-Tray function.
 *
 * f(x,y) = -0.0001 * ( |sin(x)*sin(y)*exp(|100 - sqrt(x^2+y^2)/pi|)| + 1 )^0.1
 *
 * Domain: [-10, 10]^2
 * Global minimum: f ~= -2.06261 at four points (+-1.3491, +-1.3491).
 */
class CrossInTray : public Problem {
public:
    CrossInTray();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
