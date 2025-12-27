#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Shekel function (m = 5, D = 4).
 *
 * Classic definition:
 *   f(x) = - sum_{i=1}^5 1 / ( ||x - a_i||^2 + c_i ),
 * με x ∈ [0,10]^4.
 *
 * Global minimum (περίπου):
 *   x* = (4,4,4,4),  f* ≈ -10.1532
 */
class Shekel5 : public Problem {
public:
    Shekel5();
    void init(int dim) override;  // forces D = 4

protected:
    double evaluate_core(const Vec& x) override;  // f(x) = - Σ 1/(||x-a_i||^2 + c_i)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
