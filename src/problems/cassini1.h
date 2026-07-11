#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Cassini1 (MGA, no DSM) - surrogate ΔV model.
 *
 * Models the classic ESA/GTOP "Cassini1" interplanetary trajectory
 * benchmark: an Earth -> Venus -> Venus -> Earth -> Jupiter -> Saturn
 * gravity-assist sequence ending in orbit insertion at Saturn, with NO
 * deep-space maneuvers (pure multi-gravity-assist, MGA).
 *
 * This is a surrogate ΔV model (algebraic approximation), not a full
 * Lambert-arc / real-ephemeris simulation. Reproducing the exact GTOP
 * physics would require a Lambert solver and planetary ephemerides; see
 * messenger.cpp / tandem.cpp for the same disclaimer and pattern.
 *
 * Decision vector x (D=6):
 *  x[0]      = t0  (MJD2000, launch epoch)
 *  x[1..5]   = T1..T5 (days), leg durations for the 5 legs
 *              Earth->Venus1->Venus2->Earth->Jupiter->Saturn
 *
 * Objective: surrogate total ΔV [km/s] (minimize). Known best (real
 * benchmark): f* ~= 4.93 km/s.
 */
class Cassini1 : public Problem {
public:
    Cassini1();
    void init(int dim) override;                  // force D=6 and set bounds

protected:
    double evaluate_core(const Vec& x) override;  // surrogate ΔV
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD

private:
    double t0_min_, t0_max_;
    double T1_min_, T1_max_;
    double T2_min_, T2_max_;
    double T3_min_, T3_max_;
    double T4_min_, T4_max_;
    double T5_min_, T5_max_;

    double dv_launch_base_;
    double dv_ga_gain_;
    double dv_leg_scale_;
    double dv_saturn_pen_;
    double tof_ref_days_;
    double soft_total_span_;
    double hard_pen_;

    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
