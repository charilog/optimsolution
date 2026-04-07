#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Shubert function (2D)
 *
 * Definition:
 *   S(t) = sum_{i=1}^5 i * cos( (i+1)*t + i )
 *   f(x, y) = S(x) * S(y)
 *
 * Properties:
 *   - Dimension: 2
 *   - Domain: [-10, 10]^2
 *   - Highly multimodal with many global minima
 *   - Global minimum value: f* ≈ -186.7309
 *
 * Multiple (x*, y*) pairs achieve this minimum; we only store f*.
 */
class Shubert : public Problem {
public:
    Shubert();
    void init(int dim) override;    // forces D=2, sets bounds

protected:
    double evaluate_core(const Vec& x) override;  // f(x,y) = S(x)*S(y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
