#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Messenger (MGA-1DSM) – surrogate ΔV model (faithful to provided reference).
 *
 * Decision vector x (D=14):
 *  x[0]  = t0   (MJD2000, launch epoch)
 *  x[1]  = T1   (days) Earth->Venus1
 *  x[2]  = T2   (days) Venus1->Venus2
 *  x[3]  = T3   (days) Venus2->Mercury1
 *  x[4]  = T4   (days) Mercury1->Mercury2
 *  x[5]  = T5   (days) Mercury2->Mercury (rendezvous/arrival)
 *  x[6]..x[10]  = s1..s5 (DSM fractions, 0..1)
 *  x[11]        = rp (0..1 shaping factor)
 *  x[12], x[13] = k1, k2 (0..1 shaping scalars)
 *
 * Objective: surrogate total ΔV [km/s] (minimize).
 * Bounds: same as reference file.
 */
class Messenger : public Problem {
public:
    Messenger();
    void init(int dim) override;                  // force D=14 and set bounds

protected:
    double evaluate_core(const Vec& x) override;  // surrogate ΔV
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD

private:
    // ----- bounds (defaults from reference) -----
    double t0_min_, t0_max_;
    double T1_min_, T1_max_;
    double T2_min_, T2_max_;
    double T3_min_, T3_max_;
    double T4_min_, T4_max_;
    double T5_min_, T5_max_;

    // ----- surrogate knobs (same semantics as reference) -----
    double dv_launch_base_;
    double dv_venus_gain_;
    double dv_mercury_pen_;
    double dv_leg_scale_;
    double tof_ref_inner_;
    double dsm_scale_;
    double soft_total_span_;
    double hard_pen_;

    // helpers
    static inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
