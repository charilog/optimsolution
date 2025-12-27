#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Schaffer N.2 (a.k.a. F6)
 *
 * f(x1,x2) = 0.5 + (sin^2(x1^2 - x2^2) - 0.5) /
 *            (1 + 0.001(x1^2 + x2^2))^2
 *
 * Domain: [-100, 100]^2
 * Global minimum: f(0,0) = 0
 */
class Schaffer : public Problem {
public:
    Schaffer();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;  // objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
