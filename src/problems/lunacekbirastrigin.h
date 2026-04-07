#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Lunacek bi-Rastrigin function (continuous, multimodal, non-separable).
 *
 * f(x) = min( sum_i (x_i - mu1)^2, d*n + s*sum_i (x_i - mu2)^2 )
 *        + 10 * ( n - sum_i cos(2*pi*(x_i - mu1)) )
 *
 * Domain: [-5, 5]^n
 * Global optimum: f ≈ 0 near x = (mu1, ..., mu1)
 */
class LunacekBiRastrigin : public Problem {
public:
    LunacekBiRastrigin();

    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int    n_   = 0;
    double mu1_ = 2.5;
    double d_   = 1.0;
    double s_   = 1.0;
    double mu2_ = 0.0;
};

} // namespace optimsolution
