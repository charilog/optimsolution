#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Perm function (Perm d, beta).
 *
 * f(x) = Sum_{i=1}^n [ Sum_{j=1}^n (j^i + beta) * ( (x_j/j)^i - 1 ) ]^2
 *
 * beta = 0.5 (standard choice).
 * Domain: [-n, n]^n
 * Global minimum: x*_i = i (1-indexed), f* = 0
 */
class Perm : public Problem {
public:
    Perm();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient

private:
    double beta_ = 0.5;
};

} // namespace optimsolution
