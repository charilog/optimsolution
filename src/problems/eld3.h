#pragma once
#include "problem.h"
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * ELD3 – CEC2011, Static Economic Load Dispatch (15 units)
 *
 * - Quadratic fuel (+ optional smooth valve-point)
 * - Power balance penalty (quadratic), optional losses via B-coefficients
 *   (disabled by default: use_losses = false)
 * - Prohibited Operating Zones (POZ) with smooth penalties
 * - Single-period (no ramp constraints)
 *
 * Decision vector x: size D = NG = 15 (one P_i per unit).
 */
class ELD3 : public Problem {
public:
    ELD3();
    void init(int dim) override;                 // forces NG=15, sets bounds

protected:
    double evaluate_core(const Vec& x) override; // fuel + penalties
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD (forward)

private:
    // ----- size & demand -----
    int    NG;        // 15
    double PD;        // 2630 MW

    // ----- unit data -----
    // fuel: f_i(P) = a_i P^2 + b_i P + c_i
    std::vector<double> a, b, c;
    // valve-point: e_i * |sin(f_i*(Pmin_i - P_i))|
    bool                use_valve;
    std::vector<double> e_vp, f_vp;
    // bounds
    std::vector<double> Pmin, Pmax;

    // POZ: per-unit list of [L,U]
    std::vector<std::vector<std::pair<double,double>>> poz;

    // Optional losses (default OFF)
    bool                                    use_losses;
    std::vector<std::vector<double>>        B;   // NG x NG
    std::vector<double>                     B0;  // NG
    double                                  B00;

    // penalties (faithful to reference)
    double w_balance;   // 0.15
    double w_bounds;    // 40.0
    double w_poz;       // 1500.0

    // helpers
    void set_default_data();
    void build_bounds();

    static inline double smooth_abs(double z) {
        const double eps = 1e-12; // smooth |z|
        return std::sqrt(z*z + eps*eps);
    }
};

} // namespace optimsolution
