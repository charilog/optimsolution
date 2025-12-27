#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Michalewicz benchmark (literature-faithful).
 *
 * f(x) = - sum_{i=1..n} sin(x_i) * [ sin(i * x_i^2 / pi) ]^(2m)
 * Domain: [0, pi]^n, typical m = 10.
 */
class Michalewicz : public Problem {
public:
    Michalewicz();
    void init(int dim) override;                  // sets n and [0,pi]^n bounds

protected:
    double evaluate_core(const Vec& x) override;  // computes f(x)
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    int    n_ = 0;
    double m_ = 10.0; // steepness parameter (typical)

    static inline double sqr(double v) { return v*v; }
};

} // namespace optimsolution
