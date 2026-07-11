#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Five-Uneven-Peak Trap (CEC 2013 niching benchmark F1, inverted to a
 * minimization problem).
 *
 * The CEC 2013 niching benchmark defines this as a MAXIMIZATION problem:
 *   F1(x) = 80*(2.5-x)   for 0   <= x < 2.5
 *           64*(x-2.5)   for 2.5 <= x < 5.0
 *           64*(7.5-x)   for 5.0 <= x < 7.5
 *           28*(x-7.5)   for 7.5 <= x < 12.5
 *           28*(17.5-x)  for 12.5<= x < 17.5
 *           32*(x-17.5)  for 17.5<= x < 22.5
 *           32*(27.5-x)  for 22.5<= x < 27.5
 *           80*(x-27.5)  for 27.5<= x <= 30
 * with 2 global optima (x=0 and x=30, both F1=200) and 3 local optima.
 *
 * This framework minimizes, so evaluate_core() returns -F1(x); the global
 * minimum is therefore f* = -200.0, attained at x=0 and x=30.
 *
 * Reference: Li, X.; Engelbrecht, A.; Epitropakis, M.G. "Benchmark
 * Functions for CEC'2013 Special Session and Competition on Niching
 * Methods for Multimodal Function Optimization" (2013), Section II-A.
 *
 * Domain: [0, 30] (D=1).
 */
class FiveUnevenPeakTrap : public Problem {
public:
    FiveUnevenPeakTrap();
    void init(int dim) override;                  // force D=1, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic (piecewise-linear) gradient
};

} // namespace optimsolution
