#pragma once
#include "problem.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * DED-1 (Dynamic Economic Dispatch – Case 1)
 *
 * Quadratic fuel cost, hourly power balance, ramp constraints.
 * - No transmission losses
 * - No valve-point effects
 *
 * Decision vector x has size D = U * T (U units, T hours).
 * Indexing: idx(i,t) = t*U + i, with i in [0..U-1], t in [0..T-1].
 *
 * init(dim):
 *  - If dim > 0 and divisible by U(=5), set T = dim/U; else T = 24.
 *  - Builds demand profile Dload of size T.
 *  - Sets bounds from Pmin/Pmax replicated per hour.
 */
class DED1 : public Problem {
public:
    DED1();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;  // total cost + penalties
    void   gradient_core(const Vec& x, Vec& g) override; // numeric FD (forward)

private:
    // Units & horizon
    int U;   // number of generating units (fixed: 5)
    int T;   // number of hours (default: 24)
    int Dv;  // = U * T

    // Quadratic cost: f_i(P) = a_i P^2 + b_i P + c_i
    std::vector<double> a, b, c;

    // Unit limits and ramps
    std::vector<double> Pmin, Pmax;   // MW
    std::vector<double> UR, DR;       // MW/hour (up/down ramp limits)

    // Hourly demand (MW), size T
    std::vector<double> Dload;

    // Penalty weights
    double w_balance;   // for power balance per hour (quadratic)
    double w_ramp;      // for ramp violations
    double w_bounds;    // soft bound penalty if outside margins

    // Helpers
    inline int idx(int i, int t) const { return t * U + i; }
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    void set_default_data();      // fills a,b,c,Pmin,Pmax,UR,DR and default 24h demand
    void build_bounds();          // sets bounds from Pmin/Pmax for current T
    void make_demand_profile();   // (re)builds Dload size T (repeat/sinus if T!=24)
};

} // namespace optimsolution
