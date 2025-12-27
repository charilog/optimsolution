#pragma once
#include "problem.h"
#include <cmath>

namespace optimsolution {

/**
 * Sinusoidal test function (multidimensional).
 *
 * f(x) = -2.5 * Π_i sin(x_i - π/6) - Π_i sin(5 (x_i - π/6)),
 * with x_i ∈ [0, π].
 *
 * Dimension:
 *   - If dim <= 0 in init(dim), we default to D = 1.
 *   - Otherwise D = dim.
 */
class Sinusoidal : public Problem {
public:
    Sinusoidal();

    void init(int dim) override;                  

protected:
    double evaluate_core(const Vec& x) override;  // objective value
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
