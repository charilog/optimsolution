#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * AntennaULA — Uniform Linear Array (half-wavelength spacing)
 *
 * Decision vars:
 *   w[0..N-1] ∈ [0,1]  (real amplitude taper per element)
 *
 * Geometry:
 *   - ULA along x-axis with half-wavelength spacing:
 *       d = 0.5 λ ⇒ kd = π
 *   - Element index n = 0..N-1, phase_n(θ) = π n cos(θ)
 *
 * Pattern:
 *   AF(θ) = Σ_n w_n e^{j π n cos(θ)}
 *   In this model we normalize by Σ w_n when forming the pattern,
 *   so that the objective focuses on pattern shape (not absolute gain).
 *
 * Objective:
 *   f(w) = w_main_ * (1 - AF_norm(θ0))^2
 *         + w_sll_  * mean_{θ ∉ mainlobe} |AF_norm(θ)|^{p_sll_}
 *         + w_smooth_ * Σ (w_{n+1} - w_n)^2
 *
 * όπου AF_norm(θ) = |AF(θ)| / Σ w_n
 */
class AntennaULA : public Problem {
public:
    AntennaULA();
    void init(int dim) override;                  // N = max(2, dim); bounds [0,1]^N

protected:
    double evaluate_core(const Vec& x) override;  // objective (SLL + mainlobe + smooth)
    void   gradient_core(const Vec& x, Vec& g) override; // central-diff gradient

private:
    
    double theta0_deg_   = 0.0;    // mainlobe pointing (broadside)
    double main_win_deg_ = 10.0;   // ± mainlobe window
    int    ntheta_       = 721;    // samples on [-90,90] (≈0.25°)
    double w_main_       = 5.0;    // weight for mainlobe unit-gain constraint
    double w_sll_        = 1.0;    // weight for sidelobe level
    double w_smooth_     = 0.01;   // weight for taper smoothness
    double p_sll_        = 4.0;    // L^p emphasis on sidelobes
    double eps_norm_     = 1e-12;  // guard for normalization

    // ---- Cached angle grid ----
    std::vector<double> thetas_rad_; // [-90°,90°] in radians

    // ---- Helpers ----
    void buildAngleGrid();
    static inline double deg2rad(double d) { return d * 3.1415926535897932384626433832795 / 180.0; }

    // Array-factor magnitude (NO normalization by sum(w))
    double arrayFactorMag(const Vec& w, double theta_rad) const;

    // Keep local bounds (Problem API in optimsolution δεν εκθέτει getters)
    Vec lo_, hi_;
};

} // namespace optimsolution
