#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * ELD2 – 13-unit Economic Load Dispatch (single-period)
 *
 * Faithful port from the reference:
 *  - NG = 13 units
 *  - Pd = 1800 MW
 *  - Cost_i(P_i) = a_i P_i^2 + b_i P_i + c_i + | e_i sin(f_i (Pmin_i - P_i)) |
 *  - Penalty terms:
 *      * Power balance: (sum_i P_i - Pd)^2
 *      * Soft bounds outside [Pmin_i, Pmax_i]
 *  - Gradient: numeric forward differences.
 *
 * Global optimum not analytically known (no setKnownGlobalOptimum with x*).
 */
class ELD2 : public Problem {
public:
    ELD2();
    void init(int dim) override;                 // forces NG=13, sets bounds

protected:
    double evaluate_core(const Vec& x) override; // fuel + penalties
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // ----- problem size -----
    int NG;   // 13

    // ----- demand -----
    double Pd;  // 1800 MW

    // ----- unit data -----
    // quadratic fuel
    std::vector<double> a, b, c;
    // valve-point
    std::vector<double> e, f;
    // bounds
    std::vector<double> Pmin, Pmax;

    // ----- penalty weights -----
    double w_balance;  // for (sumP - Pd)^2
    double w_bounds;   // soft bounds penalty

    void set_default_data(); // fill coefficients and limits
    void build_bounds();     // setBounds from Pmin/Pmax
};

} // namespace optimsolution
