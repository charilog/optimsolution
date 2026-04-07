#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * TersoffC (Si–C) potential minimization
 *
 * - Decision vector x has D = 3N - 6 (default N = 10 => D = 24)
 *
 * - First 3 vars are internal coordinates for atoms 1–2:
 *     x0 = r12 length       in [0, 4]
 *     x1 = r13 length       in [0, 4]
 *     x2 = angle(1–0–2)     in [0, π]
 *
 * - The remaining variables are Cartesian coordinates for atoms 3..N-1,
 *   as in the CEC2011 reference.
 *
 * - Bounds (CEC2011 style):
 *     x0 ∈ [0, 4], x1 ∈ [0, 4], x2 ∈ [0, π], x[i ≥ 3] ∈ [-4.25, 4.25]
 *
 * - Objective: total Tersoff energy with cosine cutoff:
 *     E(x) = Σ_{i<j} f_c(r_ij) [ V_R(r_ij) - B_ij V_A(r_ij) ].
 */
class TersoffC : public Problem {
public:
    TersoffC();

    // D is inferred from N via D = 3N - 6.
    // If user passes a dim matching 3N - 6, we accept it and infer N;
    // otherwise default N = 10 (D = 24).
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // atoms
    int num_atoms_;   // N (default 10)
    int D_;           // 3N - 6

    // Tersoff parameters (Si–C, per reference)
    double A_, B_, lambda1_, lambda2_, beta_, n_, c_, d_, h_, R_, Dcut_, lambda3_;
    int    m_;

    // geometry helpers
    static double distance(const std::vector<double>& a,
                           const std::vector<double>& b);
    static double angle_abc(const std::vector<double>& a,
                            const std::vector<double>& b,
                            const std::vector<double>& c);

    // reconstruct 3D positions from x (N x 3)
    std::vector<std::vector<double>> reconstruct_positions(const Vec& x) const;

    // cutoff and pair terms
    double fc(double r) const;
    double VR(double r) const;
    double VA(double r) const;
    double Bij(int i, int j, const std::vector<std::vector<double>>& pos) const;
};

} // namespace optimsolution
