#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * HeatExchanger – faithful port from the provided OPTIMUS version
 * Decision vector (core 6 dims):
 *  x[0]=Ds (shell dia) [0.30,1.50] m
 *  x[1]=Dt (tube OD)   [0.010,0.050] m
 *  x[2]=L  (length)    [1.00,6.00] m
 *  x[3]=pr (pitch/Dt)  [1.25,2.00]
 *  x[4]=B  (baffle sp) [0.10,0.60] m
 *  x[5]=cb (baffle cut fraction) [0.15,0.45]
 *
 * Objective = cost_area + w_pump*Ppump*sec_per_year + heavy UA shortfall penalty + soft penalties,
 * exactly as the reference implementation.
 */
class HeatExchanger : public Problem {
public:
    HeatExchanger();
    void init(int dim) override;                 // dim<6→6; dim>6→extra dims in [0,1]

protected:
    double evaluate_core(const Vec& x) override; // faithful objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diff

private:
    // ---- Design dimension ----
    int Dv_;     // total dimension (>=6)
    static constexpr int Dcore_ = 6;

    // ---- Process specs (constants; no static) ----
    double m_hot_,  m_cold_;     // kg/s
    double cp_hot_, cp_cold_;    // J/(kg·K)
    double Th_in_, Th_out_;      // °C
    double Tc_in_, Tc_out_;      // °C
    double rho_hot_, rho_cold_;  // kg/m^3
    double mu_hot_,  mu_cold_;   // Pa·s (kept for completeness)
    double k_hot_,   k_cold_;    // W/(m·K)

    // Required thermal duty
    double Q_req_;   // W
    double LMTD_;    // K
    double UA_req_;  // W/K
    double Rf_;      // m^2 K/W

    // Pump / energy scaling
    double pump_eta_;
    double sec_per_year_;
    double elec_cost_; // not monetized; kept for scale parity

    // Costs & weights
    double C_area_;
    double w_pump_;
    double pen_hard_;
    double pen_soft_;

    // Heat transfer surrogates
    double a_tube_, a_shell_;
    double k_tube_wall_;
    double di_factor_;

    // Pressure drop surrogates
    double kt_dp_, ks_dp_;

    // ---- Bounds (core 6 vars) ----
    double Ds_min_, Ds_max_;
    double Dt_min_, Dt_max_;
    double L_min_,  L_max_;
    double pr_min_, pr_max_;
    double B_min_,  B_max_;
    double cb_min_, cb_max_;

    // helpers
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    void set_core_bounds(Vec& lo, Vec& hi) const;
};

} // namespace optimsolution
