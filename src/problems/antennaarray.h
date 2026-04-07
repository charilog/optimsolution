#pragma once
#include "problem.h"
#include <vector>
#include <complex>

namespace optimsolution {

/**
 * AntennaArray – 6-element circular antenna array optimization
 *
 * Decision vector x (D = 12):
 *   x[0..5]   : element radii r_k in [0.2, 1.0] (λ units, e.g. normalized)
 *   x[6..11]  : element phases φ_k in degrees in [-180, 180]
 *
 * Geometry:
 *   - 6 elements on a circle, equally spaced in angle:
 *       θ_k = 2π k / 6, k = 0..5
 *
 * Array factor at scan angle θ:
 *   AF(θ) = Σ_{k=0}^{5} exp{ j [ 2π r_k cos(θ - θ_k) + φ_k(rad) ] }
 *   Normalized pattern: |AF(θ)| / 6
 *
 * Objective:
 *   f(x) = max_{θ ∈ [0,2π] \ main-lobe-exclusion} |AF(θ)| / 6
 *
 *  - The scan is sampled on a dense uniform grid.
 *  - A small exclusion band around the main lobe (θ≈0) is removed.
 */
class AntennaArray : public Problem {
public:
    AntennaArray();
    void init(int dim) override;                 // forces D=12, sets bounds

    // Optional: override scan resolution & exclusion region
    // samples >= 181 enforced internally
    void setScan(int samples, double exclude_deg);

protected:
    double evaluate_core(const Vec& x) override; // returns max sidelobe level
    void   gradient_core(const Vec& x, Vec& g) override; // analytic subgradient at argmax

private:
    int N;                                // number of elements (6)
    // geometry
    std::vector<double> theta_k;          // element angular positions
    std::vector<double> cos_tk, sin_tk;   // precomputed cos/sin(theta_k)

    // scan grid
    int samples_;                         // #angles in [0,2π]
    double exclude_rad_;                  // exclusion around main lobe (radians)
    std::vector<double> thetas, cos_th, sin_th;

    // store bounds locally
    Vec lo_, hi_;

    // cache argmax index from last evaluate_core (used in gradient)
    int last_imax_;

    void build_geometry();
    void build_scan();
};

} // namespace optimsolution
