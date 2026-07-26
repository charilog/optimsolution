#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * ZDT3 (Zitzler, Deb & Thiele, 2000) - a companion to ZDT1/ZDT2 whose
 * Pareto-optimal front is DISCONNECTED (several separate convex segments),
 * specifically to test whether a multi-objective method can maintain
 * diversity across disjoint regions of the front rather than collapsing
 * onto just one of them.
 *
 *   f1(x) = x_1
 *   g(x)  = 1 + 9/(D-1) * sum_{i=2}^{D} x_i
 *   f2(x) = g(x) * ( 1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1) )
 *
 * Domain: [0, 1]^D. Global (Pareto-optimal) set: x_1 in [0,1], x_i = 0 for
 * i = 2..D, giving g(x*) = 1 and the analytic front
 * f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1) -- the oscillating sine term makes
 * several bands of f1 dominated by others, splitting the front into
 * roughly 5 disconnected convex arcs.
 *
 * Single-objective fallback: evaluate_core() returns f1(x) + f2(x) (an
 * arbitrary but bounded scalarization) purely so this problem does not
 * crash if ever selected outside Multi-objective mode; it carries no
 * particular meaning of its own -- evaluateMultiCore() is the real
 * interface for this problem.
 */
class ZDT3 : public Problem {
public:
    ZDT3();
    void init(int dim) override;

    int numObjectives() const override { return 2; }
    Vec evaluateMultiCore(const Vec& x) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
