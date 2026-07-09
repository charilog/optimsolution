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
 *   eta = 1 - (1 / r^{(gamma-1)/gamma}),  r = P3/P1, gamma = 1.4
 *   (standard ideal simple-Brayton-cycle thermal efficiency, a function of
 *   the pressure ratio alone). We minimize f = -eta (so maximizing
 *   efficiency).
 *
 * NOTE: an earlier version of this file multiplied the bracketed term by
 * an extra (T1/T3) factor. That is not the standard Brayton efficiency
 * formula, and it is demonstrably unphysical: at the box bounds
 * (T1=300, T3=2000, P3/P1=20) it evaluates to eta=0.936, which EXCEEDS the
 * Carnot limit 1-T1/T3=0.85 for those same temperatures — impossible for
 * any heat engine (second law of thermodynamics). The corrected formula
 * above stays at eta=0.575 there, safely below Carnot, as it must for any
 * real or ideal-but-irreversible cycle operating between T1 and T3.
 *
 * Consequence worth knowing: with the (T1/T3) factor removed, T1 and T3 no
 * longer affect this objective at all (ideal simple-Brayton efficiency
 * truly depends only on pressure ratio) — ONLY P1, P3 drive eta now, and
 * the optimizer will simply push P3/P1 to its upper bound. T1, T3 remain
 * present in the decision vector but are inert for this objective. If a
 * genuinely 4-D coupled objective was the intent (all four variables
 * mattering), that needs a different, explicitly-chosen model — e.g. net
 * specific work (which does depend on both the pressure ratio AND the
 * temperature ratio T3/T1) instead of, or alongside, thermal efficiency —
 * which is a design decision beyond the scope of this fix.
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
