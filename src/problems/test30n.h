#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Test30n – non-separable oscillatory benchmark.
 *
 * Definition:
 *
 *   Let D = dimension, x ∈ ℝ^D.
 *
 *   f(x) = 0.1 * sin^2(3π x_0) * Σ_{i=1}^{D-2} (x_i - 1)^2 * (1 + sin^2(3π x_{i+1}))
 *          + (x_{D-1} - 1)^2 * (1 + sin^2(2π x_{D-1}))
 *
 * Properties:
 *   - Dimension: D chosen at init(dim)
 *   - Domain: [-10, 10]^D
 *   - Non-separable, highly multimodal
 *   - Global minimum: x* = (1,…,1), f(x*) = 0
 */
class Test30n : public Problem {
public:
    Test30n();
    void init(int dim) override;                  // sets D and bounds [-10,10]^D

protected:
    double evaluate_core(const Vec& x) override;  // f(x)
    void   gradient_core(const Vec& x, Vec& g) override; // ∇f(x)
};

} // namespace optimsolution
