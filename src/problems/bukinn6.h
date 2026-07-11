#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Bukin function N.6.
 *
 * f(x,y) = 100*sqrt(|y - 0.01*x^2|) + 0.01*|x + 10|
 *
 * Domain: x in [-15,-5], y in [-3,3]
 * Global minimum: f(-10, 1) = 0
 */
class BukinN6 : public Problem {
public:
    BukinN6();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
