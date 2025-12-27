#pragma once
#include "problem.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * ELD5 – CEC2011, Static Economic Load Dispatch (140 units)
 *
 * Faithful port of the provided reference:
 *  - Quadratic fuel (+ optional valve-point |d_i sin(e_i*(Pmin_i - P_i))|, default OFF)
 *  - Power balance penalty (ΣP - (PD + PL))^2, losses optional (default OFF)
 *  - Soft bounds penalties if x goes outside [Pmin_i, Pmax_i]
 *  - Gradient: numeric forward differences (as in the reference)
 *
 * Decision vector: x.size() = NG = 140 (one P_i per unit).
 */
class ELD5 : public Problem {
public:
    ELD5();
    void init(int dim) override;                  // forces NG=140, sets bounds

protected:
    double evaluate_core(const Vec& x) override;  // fuel + penalties
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diff

private:
    // ---- size & demand ----
    int NG;             // 140
    double PD;          // 49342 MW (reference)

    // ---- cost coefficients ----
    // fuel: a_i P^2 + b_i P + c_i
    std::vector<double> a, b, c;
    // valve-point: | d_i * sin( e_i * (Pmin_i - P_i) ) |
    std::vector<double> d, e;
    bool use_valve;

    // ---- limits ----
    std::vector<double> Pmin, Pmax;

    // ---- optional losses (default OFF) ----
    bool use_losses;
    std::vector<std::vector<double>> B;   // NGxNG
    std::vector<double> B0;               // NG
    double B00;                           

    // ---- penalties ----
    double w_balance;     // (sp - (PD+PL))^2
    double w_bounds;      // soft bounds

    // ---- helpers ----
    static inline double smooth_abs(double z) {
        const double eps = 1e-12;
        return std::sqrt(z*z + eps*eps);
    }

    void set_default_limits_from_cec(); // hard-coded Pmin/Pmax from reference (140u)
    void set_placeholder_costs();       // runnable defaults (same idea as old code)
    void build_bounds();                // setBounds from Pmin/Pmax
};

} // namespace optimsolution
