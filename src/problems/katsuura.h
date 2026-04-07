#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Katsuura benchmark test function (continuous, multimodal, non-separable).
 *
 * f(x) = ∏_{i=1}^n ( 1 + i * Σ_{k=1}^{32} |2^k x_i - round(2^k x_i)| / 2^k )^{10 / n^{1.2}} - 1
 *
 * Domain: [-100, 100]^n
 * Global minimum: f(x*) = 0 at x* = (0, ..., 0).
 */
class Katsuura : public Problem {
public:
    Katsuura();
    void init(int dim) override;  // sets dimension, bounds and known optimum

protected:
    double evaluate_core(const Vec& x) override;       // computes Katsuura(x)
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward differences
};

} // namespace optimsolution
