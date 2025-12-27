#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Griewank benchmark function.
 *
 *  f(x) = 1 + sum_{i=1}^D x_i^2 / 4000 - prod_{i=1}^D cos( x_i / sqrt(i) )
 *
 * - Domain: typically [-600, 600]^D
 * - Global minimum: f(0,...,0) = 0
 * - Non-separable, multimodal.
 */
class Griewank : public Problem {
public:
    Griewank();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
