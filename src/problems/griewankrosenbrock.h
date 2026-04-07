#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Griewank–Rosenbrock composition function
 *
 * For i = 1..D-1:
 *     r_i = x_i^2 - x_{i+1}
 *     z_i = 100 r_i^2 + (x_i - 1)^2
 *     f   = Σ_i [ z_i/4000 - cos(z_i) + 1 ]
 *
 * Domain: [-5,5]^D
 * Global minimum: f = 0 at x = (1, 1, ..., 1).
 */
class GriewankRosenbrock : public Problem {
public:
    GriewankRosenbrock();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
