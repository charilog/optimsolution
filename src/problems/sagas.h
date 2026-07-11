#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Sagas (ΔV-EGA maneuver) - surrogate ΔV model.
 *
 * Models the ESA/GTOP "Sagas" benchmark: an Earth -> Earth -> Jupiter
 * delta-V Earth Gravity Assist (EGA) maneuver intended to reach 50 AU,
 * with one deep-space maneuver (DSM) on each of the two legs.
 *
 * Surrogate ΔV model (algebraic approximation), not a Lambert-arc / real
 * ephemeris simulation — same disclaimer and pattern as messenger.cpp,
 * tandem.cpp, cassini1.cpp.
 *
 * Decision vector x (D=12):
 *  x[0]  = t0    (MJD2000, launch epoch)
 *  x[1]  = vinf  (km/s, launch hyperbolic excess speed)
 *  x[2]  = alpha (rad, hyperbolic excess velocity angle 1)
 *  x[3]  = beta  (rad, hyperbolic excess velocity angle 2)
 *  x[4]  = T1    (days) Earth -> Earth
 *  x[5]  = T2    (days) Earth -> Jupiter
 *  x[6]  = s1    (0..1, DSM fraction on leg 1)
 *  x[7]  = s2    (0..1, DSM fraction on leg 2)
 *  x[8]  = rp1   (0..1, flyby radius shaping, leg 1)
 *  x[9]  = rp2   (0..1, flyby radius shaping, leg 2)
 *  x[10] = b1    (0..1, B-plane angle shaping, leg 1)
 *  x[11] = b2    (0..1, B-plane angle shaping, leg 2)
 *
 * Objective: surrogate total ΔV [km/s] (minimize). Known best (real
 * benchmark): f* ~= 18.19 km/s.
 */
class Sagas : public Problem {
public:
    Sagas();
    void init(int dim) override;                  // force D=12 and set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD

private:
    double t0_min_, t0_max_;
    double vinf_min_, vinf_max_;
    double angle_min_, angle_max_;
    double T1_min_, T1_max_;
    double T2_min_, T2_max_;

    double dv_launch_base_;
    double dv_ega_gain_;
    double dv_leg_scale_;
    double dsm_scale_;
    double tof_ref_days_;
    double hard_pen_;

    static inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
