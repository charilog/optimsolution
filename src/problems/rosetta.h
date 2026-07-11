#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Rosetta (MGA-1DSM) - surrogate ΔV model.
 *
 * Models the ESA/GTOP "Rosetta" benchmark: an Earth -> Earth -> Mars ->
 * Earth -> Earth -> 67P/Churyumov-Gerasimenko rendezvous mission (the
 * trajectory actually flown by the Rosetta spacecraft), with one deep
 * space maneuver (DSM) per leg.
 *
 * Surrogate ΔV model (algebraic approximation), not a Lambert-arc / real
 * ephemeris simulation — same disclaimer and pattern as messenger.cpp,
 * tandem.cpp, cassini1.cpp, sagas.cpp, gtoc1.cpp.
 *
 * Decision vector x (D=22):
 *  x[0]      = t0    (MJD2000, launch epoch)
 *  x[1]      = vinf  (km/s, launch hyperbolic excess speed)
 *  x[2]      = alpha (rad, hyperbolic excess velocity angle 1)
 *  x[3]      = beta  (rad, hyperbolic excess velocity angle 2)
 *  x[4..8]   = T1..T5   (days, 5 leg durations)
 *  x[9..13]  = s1..s5   (0..1, DSM fraction on each leg)
 *  x[14..17] = rp1..rp4 (0..1, flyby radius shaping, 4 intermediate flybys)
 *  x[18..21] = b1..b4   (0..1, B-plane angle shaping, 4 intermediate flybys)
 *
 * Objective: surrogate total ΔV [km/s] (minimize). Known best (real
 * benchmark): f* ~= 1.34 km/s (rendezvous variant).
 */
class Rosetta : public Problem {
public:
    Rosetta();
    void init(int dim) override;                  // force D=22 and set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD

private:
    double t0_min_, t0_max_;
    double vinf_min_, vinf_max_;
    double angle_min_, angle_max_;
    Vec    T_min_, T_max_; // size 5

    double dv_ega_gain_;
    double dv_leg_scale_;
    double dsm_scale_;
    double dv_comet_pen_;
    double tof_ref_days_;
    double hard_pen_;

    static inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
