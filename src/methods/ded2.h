#pragma once
#include "problem.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * DED2 – Dynamic Economic Dispatch (9-unit system, multi-hour)
 *
 * - Quadratic fuel cost + optional smooth valve-point terms
 * - Hourly power balance penalty with optional transmission losses
 * - Ramp-rate constraints
 * - Soft bound penalties if P < Pmin or P > Pmax
 *
 * Decision vector size: D = U * T, with U = 9.
 * If init(dim) is called with dim divisible by U, sets T = dim/U; else uses T=24.
 */
class DED2 : public Problem {
public:
    DED2();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // Size
    int U;
    int T;
    int Dv; // = U * T

    // Cost coefficients a,b,c
    std::vector<double> a, b, c;

    // Limits
    std::vector<double> Pmin, Pmax;
    std::vector<double> UR, DR;

    // Valve-point optional
    bool use_valve;
    std::vector<double> e_vp, f_vp;

    // Demand profile
    std::vector<double> Dload;

    // Loss coefficients (optional)
    bool use_losses;
    std::vector<std::vector<double>> B;
    std::vector<double> B0;
    double B00;

    // Penalties
    double w_balance;
    double w_ramp;
    double w_bounds;

    // Helpers
    inline int idx(int i, int t) const { return t * U + i; }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static inline double smooth_abs(double z) {
        const double eps = 1e-12;
        return std::sqrt(z*z + eps*eps);
    }

    void set_default_data();
    void synthesize_demand();
    void build_bounds();
};

} // namespace optimsolution
