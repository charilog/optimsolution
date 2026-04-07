#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * Hydrothermal Scheduling (HTS) – literature-faithful smooth model
 *
 * Decision vector (size D = NG*T + NH*T):
 *   x[ t*NG + i ]          = P_{i,t}  (thermal MW)
 *   x[ NG*T + t*NH + j ]   = Q_{j,t}  (hydro discharge, abstract units/hour)
 *
 * States (implicit, not in x): V_{j,t}, evolved by reservoir balance:
 *   V_{j,t+1} = V_{j,t} + Inflow_{j,t} - Q_{j,t}
 *
 * Hydro power (head-dependent):
 *   H_{j,t} = kappa_j * (h0_j + beta_j * Vbar_{j,t}) * Q_{j,t},
 *   with Vbar the mid-step volume, kappa in MW/(head*flow).
 *
 * Objective:
 *   Min  sum_{t,i} (a_i P_{i,t}^2 + b_i P_{i,t} + c_i)
 *        + w_bal * sum_t ( sum_i P_{i,t} + sum_j H_{j,t} - D_t )^2
 *        + w_V   * sum_{j,t} violation(V_{j,t} in [Vmin,Vmax])^2
 *        + w_Q/P * soft-bound penalties for Q,P
 *        + w_ramp * thermal ramp penalties (optional; default off)
 *        + w_final * sum_j (V_{j,T} - Vfinal_j)^2
 *
 * Bounds:
 *   P in [Pmin,Pmax], Q in [Qmin,Qmax]  (as variable bounds as well).
 */
class Hydrothermal : public Problem {
public:
    Hydrothermal();

    void init(int dim) override;                 // sets D=NG*T + NH*T and bounds

protected:
    double evaluate_core(const Vec& x) override; // smooth penalties (no 1e20)
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // ---- structure ----
    int T_, NG_, NH_;

    // ---- thermal data ----
    std::vector<double> a_, b_, c_;        // fuel cost coeffs
    std::vector<double> Pmin_, Pmax_;      // MW
    std::vector<double> UR_, DR_;          // ramp up/down (MW/h), optional

    // ---- hydro data ----
    std::vector<double> Qmin_, Qmax_;      // discharge bounds (abstract units)
    std::vector<double> Vmin_, Vmax_;      // storage bounds (same abstract units)
    std::vector<double> Vinit_, Vfinal_;   // initial/final storage

    // Head/power mapping: H = kappa * (h0 + beta * Vbar) * Q
    std::vector<double> kappa_, h0_, beta_;

    // ---- time series ----
    std::vector<double> demand_;           // MW for each t
    std::vector<double> inflow_;           // per hydro j,t in same units as Q

    // ---- penalties ----
    double w_balance_, w_V_, w_QP_, w_ramp_, w_final_;

    // ---- helpers ----
    inline int idxP(int t, int i) const { return t*NG_ + i; }
    inline int idxQ(int t, int j) const { return NG_*T_ + t*NH_ + j; }

    static inline double sqr(double v) { return v*v; }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
