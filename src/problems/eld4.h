#pragma once
#include "problem.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * ELD4 — 40-unit Economic Load Dispatch (CEC 40-unit)
 *
 * - Quadratic fuel + valve-point effects
 * - Power balance penalty
 * - Optional transmission losses (default OFF)
 * - No POZ here (the 40-unit dataset usually has none)
 *
 * Dimensionality: fixed NG = 40
 */

class ELD4 : public Problem {
public:
    ELD4();
    void init(int dim) override;                   // forces NG=40

protected:
    double evaluate_core(const Vec& x) override;   // total cost + penalties
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int NG;              // number of units (40)
    double PD;           // total demand (10500 MW)

    bool use_losses;     // optional B-matrix losses
    std::vector<std::vector<double>> B;   // NGxNG
    std::vector<double> B0;
    double B00;

    // Quadratic cost: a_i P^2 + b_i P + c_i
    std::vector<double> a, b, c;

    // Valve-point term: | d_i * sin( e_i * (Pmin_i - P_i) ) |
    std::vector<double> d, e;

    // Bounds
    std::vector<double> Pmin, Pmax;

    // Penalty weights
    double w_balance;
    double w_bounds;

    // Helpers
    void set_default_data();
    void build_bounds();

    static inline double smooth_abs(double z) {
        const double eps = 1e-12;      // smooth | · |
        return std::sqrt(z*z + eps*eps);
    }
};

} // namespace optimsolution
