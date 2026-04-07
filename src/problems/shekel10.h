#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Shekel function (m = 10, D = 4).
 *
 * Classic definition:
 *   f(x) = - sum_{i=1}^{10} 1 / ( ||x - a_i||^2 + c_i ),
 * with x ∈ [0,10]^4.
 *
 * Global minimum (approximately):
 *   x* ≈ (4,4,4,4),  f* ≈ -10.5364
 */
class Shekel10 : public Problem {
public:
    Shekel10();
    void init(int dim) override;  // forces D = 4

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
