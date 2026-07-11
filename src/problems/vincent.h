#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Vincent function (CEC 2013 niching benchmark F7, inverted to a
 * minimization problem).
 *
 * The CEC 2013 niching benchmark defines Vincent as a MAXIMIZATION problem:
 *   F7(x) = (1/D) * Sum_i sin(10*log(x_i)),   x_i in [0.25, 10]
 * with 6^D global optima, each with F7 = 1.0, and no local optima.
 *
 * This framework minimizes, so evaluate_core() returns -F7(x); the global
 * minimum is therefore f* = -1.0, still attained at 6^D points.
 *
 * Reference: Li, X.; Engelbrecht, A.; Epitropakis, M.G. "Benchmark
 * Functions for CEC'2013 Special Session and Competition on Niching
 * Methods for Multimodal Function Optimization" (2013), Section II-G.
 *
 * Domain: [0.25, 10]^D.
 */
class Vincent : public Problem {
public:
    Vincent();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient
};

} // namespace optimsolution
