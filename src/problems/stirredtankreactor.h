#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Optimal control of a non-linear (continuous) stirred tank reactor (CSTR).
 *
 * Dimensionless CSTR model (Flores-Tlacuahuac & Grossmann, 2006), widely
 * used as a nonlinear-MPC / optimal-control benchmark:
 *
 *   dc/dt = (1 - c) * rho / V  -  c * k * exp(-N_ / T)
 *   dT/dt = (Tf - T) * rho / V +  c * k * exp(-N_ / T)  -  F * alpha_c * (T - Tc)
 *
 * States: c (product concentration), T (reactor temperature).
 * Controls: rho (production/dilution rate), F (coolant flow rate).
 * Steady state used as the tracking target: c_ss = 0.1367, T_ss = 0.7293,
 * rho_ss = 1.0, F_ss = 390.0 (matches the reference model).
 *
 * The decision vector holds the two controls as piecewise-constant values
 * over Nstages_ time stages:
 *   x[2*k]   = rho_k  (production rate on stage k), k = 0..Nstages_-1
 *   x[2*k+1] = F_k    (coolant flow rate on stage k)
 *
 * The CSTR ODEs are forward-Euler integrated over the stages starting from
 * a perturbed initial state, and the objective is a standard quadratic
 * tracking + control-effort cost:
 *   J = sum_k [ w_c*(c_k-c_ss)^2 + w_T*(T_k-T_ss)^2
 *             + w_rho*(rho_k-rho_ss)^2 + w_F*(F_k-F_ss)^2 ]
 * (minimize J). D = 2*Nstages_ (default Nstages_ = 10 => D = 20).
 */
class StirredTankReactor : public Problem {
public:
    StirredTankReactor();
    void init(int dim) override;                  // sets D = 2*Nstages_, bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    int Nstages_;   // default 10
    double dt_;     // stage duration (dimensionless time)

    // Model constants
    double V_, k_, N_, Tf_, Tc_, alpha_c_;

    // Steady state (tracking target)
    double c_ss_, T_ss_, rho_ss_, F_ss_;

    // Initial (perturbed) state
    double c0_, T0_;

    // Bounds
    double rho_min_, rho_max_;
    double F_min_,   F_max_;

    // Cost weights
    double w_c_, w_T_, w_rho_, w_F_;

    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // Simulate the CSTR forward over all stages and accumulate the cost.
    double simulate_cost(const Vec& x) const;
};

} // namespace optimsolution
