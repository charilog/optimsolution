#pragma once
#include "problem.h"
#include <cmath>

namespace optimsolution {

/**
 * GasCycle – simple Brayton-like gas cycle efficiency model.
 *
 * Decision variables (D = 4):
 *   x[0] = T1  ∈ [300, 1500]   (K)
 *   x[1] = T3  ∈ [1200, 2000]  (K)
 *   x[2] = P1  ∈ [1, 20]       (bar)
 *   x[3] = P3  ∈ [1, 20]       (bar)
 *
 * Objective:
 *   η = 1 - (1 / r^{(γ-1)/γ}) * (T1/T3),  where r = P3/P1, γ = 1.4
 *   We minimize f = -η (so maximizing efficiency).
 *
 * Outside the box constraints we return a large penalty (1e20).
 */
class GasCycle : public Problem {
public:
    GasCycle();

    void init(int dim) override;                  // forces D = 4 and sets bounds

protected:
    double evaluate_core(const Vec& x) override;  // f(x) = -η (with box penalty)
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
