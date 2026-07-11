#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Langermann function (classic 2D, m=5 form).
 *
 * f(x) = -Sum_{i=1}^{5} c_i * exp( -(1/pi) * Sum_j (x_j - A_ij)^2 )
 *                          * cos( pi * Sum_j (x_j - A_ij)^2 )
 *
 * with A = [[3,5],[5,2],[2,1],[1,4],[7,9]], c = [1,2,5,2,3].
 *
 * Domain: [0, 10]^2
 * Global minimum: f ~= -5.1621 at x ~= (2.00299, 1.006).
 */
class Langermann : public Problem {
public:
    Langermann();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient

private:
    std::vector<std::vector<double>> A_;
    std::vector<double> c_;
};

} // namespace optimsolution
