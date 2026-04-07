#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * VibratingPlatform — vibrating platform design benchmark
 *
 * IMPORTANT:
 * This benchmark is not the simplified 2-variable SDOF (k,c) isolator model.
 * The standard formulation uses 5 decision variables:
 *   x[0] = d1   first thickness segment
 *   x[1] = d2   second thickness segment
 *   x[2] = d3   third thickness segment
 *   x[3] = b    platform width
 *   x[4] = L    platform length
 *
 * Bounds:
 *   0.05 <= d1 <= 0.50
 *   0.20 <= d2 <= 0.50
 *   0.20 <= d3 <= 0.60
 *   0.35 <= b  <= 0.50
 *   3.00 <= L  <= 6.00
 *
 * The original benchmark is bi-objective:
 *   f1 = -(pi / (2 L^2)) * sqrt(EI / mu)
 *   f2 = 2 b L [ c1 d1 + c2 (d2-d1) + c3 (d3-d2) ]
 * subject to 5 inequality constraints.
 *
 * Since optimsolution::Problem is single-objective, this implementation uses a
 * fixed scalarization of the two benchmark objectives plus smooth penalties for
 * constraint violations.
 */
class VibratingPlatform : public Problem {
public:
    VibratingPlatform();

    void   init(int dim) override;                       // force D=5, set bounds
protected:
    double evaluate_core(const Vec& x) override;         // scalarized objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    // Material / cost constants from the standard benchmark formulation
    double rho1_, rho2_, rho3_;
    double E1_,   E2_,   E3_;
    double c1_,   c2_,   c3_;

    // Bounds
    double d1_min_, d1_max_;
    double d2_min_, d2_max_;
    double d3_min_, d3_max_;
    double b_min_,  b_max_;
    double L_min_,  L_max_;

    // Scalarization / penalties (for single-objective framework usage)
    double w_f1_;    // weight on normalized -f1 (frequency term)
    double w_f2_;    // weight on normalized  f2 (cost term)
    double w_con_;   // weight on aggregated constraint violations

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
