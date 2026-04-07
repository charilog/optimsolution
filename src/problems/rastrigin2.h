#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Rastrigin2 – 2D Rastrigin function with shift
 *
 * Definition:
 *   f(x, y) = [x^2 - 10 cos(2πx) + 10] + [y^2 - 10 cos(2πy) + 10] - 2
 *
 * Properties:
 *   - Dimension: 2
 *   - Domain: [-5.12, 5.12]^2
 *   - Global minimum: at (0, 0), f* = -2
 *   - Highly multimodal, separable
 */
class Rastrigin2 : public Problem {
public:
    Rastrigin2();
    void init(int dim) override;  // forces D = 2

protected:
    double evaluate_core(const Vec& x) override;  // f(x,y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
