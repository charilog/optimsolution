#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Alpine N.1 function.
 *
 * f(x) = Sum_i | x_i * sin(x_i) + 0.1 * x_i |
 *
 * Domain: [-10, 10]^n
 * Global minimum: f(0,...,0) = 0
 *
 * Non-smooth (absolute value): gradient_core uses numeric forward differences.
 */
class Alpine1 : public Problem {
public:
    Alpine1();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
