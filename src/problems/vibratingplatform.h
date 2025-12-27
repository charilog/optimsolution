#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * VibratingPlatform — SDOF base-excited isolator design
 *
 * Decision vector x (D=2):
 *   x[0] = k  [N/m]    spring stiffness
 *   x[1] = c  [N*s/m]  viscous damping
 *
 * Objective:
 *   Minimize average squared displacement transmissibility over [fmin, fmax]
 *   + smooth penalties (static sag, X at fwork, zeta bounds, fn bounds).
 */
class VibratingPlatform : public Problem {
public:
    VibratingPlatform();

    void   init(int dim) override;                  // force D=2, set bounds
protected:
    double evaluate_core(const Vec& x) override;    // objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    // Design constants (configurable)
    double m_;            // payload mass [kg]
    double fmin_, fmax_;  // band [Hz]
    double fwork_;        // representative disturbance [Hz]
    double delta_max_;    // max static deflection [m]
    double Y0_work_;      // base displacement amplitude at fwork [m]
    double zeta_min_, zeta_max_;
    double fn_min_, fn_max_;   // desirable natural frequency window [Hz]
    int    Nfreq_;        // # of log-spaced samples

    // Variable bounds
    double k_min_, k_max_;   // [N/m]
    double c_min_, c_max_;   // [N*s/m]

    // Penalty weights
    double w_delta_, w_disp_, w_zeta_, w_fn_;

    // helpers
    void build_freq_grid(std::vector<double>& f) const;
    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
