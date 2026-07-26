#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * ZDT1 (Zitzler, Deb & Thiele, 2000) - the most widely used bi-objective
 * benchmark for validating multi-objective optimizers, with a known
 * analytic (CONVEX) Pareto front.
 *
 *   f1(x) = x_1
 *   g(x)  = 1 + 9/(D-1) * sum_{i=2}^{D} x_i
 *   f2(x) = g(x) * ( 1 - sqrt(f1(x) / g(x)) )
 *
 * Domain: [0, 1]^D. Global (Pareto-optimal) set: x_1 in [0,1], x_i = 0 for
 * i = 2..D, giving g(x*) = 1 and the analytic front f2 = 1 - sqrt(f1),
 * f1 in [0,1] -- a smooth, convex curve. This is the standard first test
 * any new multi-objective method is run against; NSGA-II/MOEA-D/MOPSO
 * should all converge to a front visibly matching f2 = 1 - sqrt(f1) when
 * run on this problem.
 *
 * Single-objective fallback: evaluate_core() returns f1(x) + f2(x) (an
 * arbitrary but bounded scalarization) purely so this problem does not
 * crash if ever selected outside Multi-objective mode; it carries no
 * particular meaning of its own -- evaluateMultiCore() is the real
 * interface for this problem.
 */
class ZDT1 : public Problem {
public:
    ZDT1();
    void init(int dim) override;

    int numObjectives() const override { return 2; }
    Vec evaluateMultiCore(const Vec& x) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
