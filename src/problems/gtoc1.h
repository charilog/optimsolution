#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * GTOC1 - surrogate model.
 *
 * Models the ESA/GTOP "GTOC1" benchmark: a multi-gravity-assist mission
 * Earth -> Venus -> Earth -> Jupiter -> Saturn -> asteroid TW229, with
 * no deep-space maneuvers. Unlike the other trajectory problems in this
 * framework, the REAL GTOC1 objective is to MAXIMIZE the change in the
 * asteroid's orbital semi-major axis achieved by an impact at the end of
 * the mission (not minimize ΔV). To keep this a minimization problem
 * (consistent with the rest of the framework), evaluate_core() returns
 * the NEGATIVE of that surrogate benefit.
 *
 * Surrogate model (algebraic approximation), not a Lambert-arc / real
 * ephemeris simulation — same disclaimer and pattern as messenger.cpp,
 * tandem.cpp, cassini1.cpp, sagas.cpp.
 *
 * Decision vector x (D=8):
 *  x[0]     = t0     (MJD2000, launch epoch)
 *  x[1..7]  = T1..T7 (days), leg durations for the 7 legs/events of the
 *             Earth->Venus->Earth->Jupiter->Saturn->asteroid sequence
 *
 * Objective: -(surrogate semi-major-axis benefit) (minimize). Known best
 * (real benchmark, maximization form): f(x) = 1 581 950.0.
 */
class GTOC1 : public Problem {
public:
    GTOC1();
    void init(int dim) override;                  // force D=8 and set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD

private:
    double t0_min_, t0_max_;
    Vec    T_min_, T_max_; // size 7

    double launch_base_;
    double ga_gain_;
    double leg_pen_scale_;
    double impact_gain_;
    double tof_ref_days_;
    double soft_total_span_;
    double hard_pen_;

    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
