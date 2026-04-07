#pragma once
#include "problem.h"
#include <array>
#include <vector>

namespace optimsolution {

/**
 * Bifunctional catalyst optimization problem.
 *
 * Single decision variable u \in [0.6, 0.9].
 * For each u, a 7-state ODE system is integrated on [0, 0.78] with
 * Euler steps (N = 1000), using reaction rate coefficients k_i(u)
 * defined by cubic polynomials of u.
 *
 * Objective:
 *   J(u) = 1000 * y_6(tf)
 * and we minimize
 *   f(u) = -J(u)
 * so that larger J corresponds to smaller f.
 *
 * - Dimension: D = 1 (forced)
 * - Bounds: [0.6, 0.9]
 * - Real-world dynamic process model (no known closed-form global optimum)
 */
class BifunctionalCatalyst : public Problem {
public:
    BifunctionalCatalyst();

    // Forces dimension = 1, sets bounds [umin_, umax_]
    void init(int dim) override;

protected:
    // Returns objective f(u) = -J(u), where J(u) = 1000 * y6(tf)
    double evaluate_core(const Vec& x) override;

    // Forward-difference gradient in u (consistent with other problems)
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // Coefficients c for k_i(u) = c0 + c1*u + c2*u^2 + c3*u^3  (10x4)
    std::array<std::array<double,4>,10> c_;

    double umin_;
    double umax_;

    // One-step RHS (faithful to the reference implementation)
    inline void rhs(double u,
                    const std::array<double,10>& k,
                    const std::array<double,7>&  y,
                    std::array<double,7>&        dy) const;

    // Euler integration (N=1000, tf=0.78), returns J = 1e3 * y6(tf)
    double simulate_and_objective(double u) const;
};

} // namespace optimsolution
