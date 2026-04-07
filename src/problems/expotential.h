#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Expotential function:
 *
 *   f(x) = 1 - exp( -0.5 * ||x||^2 )
 *
 * - Continuous, smooth, strictly convex.
 * - Global minimum: f* = 0 at x* = 0.
 * - Domain used: [-1, 1]^D (as in original file).
 */
class Expotential : public Problem {
public:
    Expotential();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
