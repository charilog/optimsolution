#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * ZDT4 (Zitzler, Deb & Thiele, 2000) - a companion to ZDT1/ZDT2/ZDT3 with
 * roughly 21^(D-1) local Pareto-optimal fronts surrounding the single
 * global one, specifically to test a multi-objective method's ability to
 * converge globally rather than getting stuck on one of the many deceptive
 * local fronts.
 *
 *   f1(x) = x_1
 *   g(x)  = 1 + 10*(D-1) + sum_{i=2}^{D} [ x_i^2 - 10*cos(4*pi*x_i) ]
 *   f2(x) = g(x) * ( 1 - sqrt(f1/g) )
 *
 * *** IMPORTANT: non-uniform bounds, unlike ZDT1/2/3 ***
 *   x_1 in [0, 1],  x_i in [-5, 5] for i = 2..D
 *
 * Global (Pareto-optimal) set: x_1 in [0,1], x_i = 0 for i = 2..D, giving
 * g(x*) = 1 (each Rastrigin-like term vanishes at x_i=0) and the analytic
 * front f2 = 1 - sqrt(f1), f1 in [0,1] -- the SAME convex curve as ZDT1,
 * but now every other integer-ish setting of x_i (i>=2) creates a nearby
 * local front that a weak method can get trapped on.
 *
 * Single-objective fallback: evaluate_core() returns f1(x) + f2(x) (an
 * arbitrary but bounded scalarization) purely so this problem does not
 * crash if ever selected outside Multi-objective mode; it carries no
 * particular meaning of its own -- evaluateMultiCore() is the real
 * interface for this problem.
 */
class ZDT4 : public Problem {
public:
    ZDT4();
    void init(int dim) override;

    int numObjectives() const override { return 2; }
    Vec evaluateMultiCore(const Vec& x) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
