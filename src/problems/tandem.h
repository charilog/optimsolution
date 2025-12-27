#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * Space Trajectory Optimization – Tandem (MGA-1DSM surrogate)
 *
 * Decision vector x (D=18):
 *  x[0]  : t0   [MJD2000] (launch epoch; only used for penalties/offsets)
 *  x[1]  : T1   [days]  E -> V1
 *  x[2]  : T2   [days]  V1 -> E1
 *  x[3]  : T3   [days]  E1 -> E2
 *  x[4]  : T4   [days]  E2 -> J
 *  x[5]  : T5A  [days]  J  -> Saturn (branch A)
 *  x[6]  : T5B  [days]  J  -> Saturn (branch B)
 *  x[7]..x[10] : s1..s4   (0..1) DSM fractions on common legs
 *  x[11]       : s5A      (0..1) DSM fraction on branch A
 *  x[12]       : s5B      (0..1) DSM fraction on branch B
 *  x[13]       : rp       (0..1) periapsis/shaping factor (surrogate)
 *  x[14],x[15] : kA1,kA2  (0..1) branch A shaping
 *  x[16],x[17] : kB1,kB2  (0..1) branch B shaping
 *
 * Objective: Minimize surrogate total ΔV [km/s].
 */
class Tandem : public Problem {
public:
    Tandem();
    void init(int dim) override;                  // force D=18, set bounds

protected:
    double evaluate_core(const Vec& x) override;  // surrogate ΔV
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    // ----- bounds (from reference) -----
    double t0_min_  = 7000.0,  t0_max_  = 10000.0;

    double T1_min_  =  30.0,   T1_max_  =  500.0;
    double T2_min_  =  30.0,   T2_max_  =  600.0;
    double T3_min_  =  30.0,   T3_max_  = 1200.0;
    double T4_min_  =  30.0,   T4_max_  = 1600.0;
    double T5A_min_ =  30.0,   T5A_max_ = 2000.0;
    double T5B_min_ =  30.0,   T5B_max_ = 2000.0;

    // ----- surrogate knobs (as in attachment) -----
    double dv_launch_base_ = 12.0;   // km/s
    double dv_ga_gain_     = 2.5;    // km/s per GA (benefit)
    double dv_jupiter_aid_ = 3.0;    // km/s benefit at J
    double dv_leg_scale_   = 18.0;   // km/s scaling per leg
    double tof_ref_days_   = 500.0;  // [days]
    double dsm_scale_      = 1.2;    // km/s
    double p_pen_base_     = 1e6;    // hard-violation penalty unit
    double soft_tof_span_  = 3500.0; // [days]

    static inline double clamp01(double v){
        return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    static inline double clamp(double v,double lo,double hi){
        return v<lo ? lo : (v>hi ? hi : v);
    }
};

} // namespace optimsolution
