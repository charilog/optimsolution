#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Zakharov benchmark function
 *
 * f(x) = sum_{i=1}^D x_i^2
 *      + ( sum_{i=1}^D (i/2) x_i )^2
 *      + ( sum_{i=1}^D (i/2) x_i )^4
 *
 * Domain (typical): x_i in [-5, 10], global min f(0)=0.
 *
 * Properties:
 *   - Dimension: D chosen at init(dim), default D=10 if dim<=0
 *   - Domain: [-5, 10]^D
 *   - Non-separable, unimodal
 */
class Zakharov : public Problem {
public:
    Zakharov();

    void   init(int dim) override;                 // set dimension & bounds

protected:
    double evaluate_core(const Vec& x) override;   // objective
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient

private:
    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
