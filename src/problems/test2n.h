#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Test2n – separable quartic polynomial benchmark.
 *
 * Definition:
 *   f(x) = Σ_{i=1}^D 0.5 * (x_i^4 - 16 x_i^2 + 5 x_i)
 *
 * Properties:
 *   - Dimension: D chosen at init(dim)
 *   - Domain: [-5, 5]^D
 *   - Separable, multimodal in 1D
 */
class Test2n : public Problem {
public:
    Test2n();
    void init(int dim) override;            // sets D and bounds [-5,5]^D

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
