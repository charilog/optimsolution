#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Schwefel 2.26
 *
 *   f(x) = 418.9829 * n - sum_{i=1}^n x_i * sin(sqrt(|x_i|))
 *
 * Domain: [-500, 500]^n
 * Global optimum: x_i ≈ 420.968746..., f* ≈ 0
 */
class Schwefel : public Problem {
public:
    Schwefel();
    void init(int dim) override;                  // set dimension and bounds [-500,500]^n

protected:
    double evaluate_core(const Vec& x) override;  // Schwefel value
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
