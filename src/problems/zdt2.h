#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * ZDT2 (Zitzler, Deb & Thiele, 2000) - the standard companion to ZDT1, with
 * a known analytic NON-CONVEX (concave) Pareto front, useful for checking
 * that a multi-objective method doesn't just handle the convex case.
 *
 *   f1(x) = x_1
 *   g(x)  = 1 + 9/(D-1) * sum_{i=2}^{D} x_i
 *   f2(x) = g(x) * ( 1 - (f1(x) / g(x))^2 )
 *
 * Domain: [0, 1]^D. Global (Pareto-optimal) set: x_1 in [0,1], x_i = 0 for
 * i = 2..D, giving g(x*) = 1 and the analytic front f2 = 1 - f1^2,
 * f1 in [0,1] -- a concave curve (unlike ZDT1's convex f2 = 1 - sqrt(f1)).
 * Methods that only work on convex fronts (e.g. naive weighted-sum
 * scalarization) visibly fail to cover the middle of this front, which is
 * exactly why it is a standard companion benchmark to ZDT1.
 *
 * Single-objective fallback: evaluate_core() returns f1(x) + f2(x) (an
 * arbitrary but bounded scalarization) purely so this problem does not
 * crash if ever selected outside Multi-objective mode; it carries no
 * particular meaning of its own -- evaluateMultiCore() is the real
 * interface for this problem.
 */
class ZDT2 : public Problem {
public:
    ZDT2();
    void init(int dim) override;

    int numObjectives() const override { return 2; }
    Vec evaluateMultiCore(const Vec& x) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
