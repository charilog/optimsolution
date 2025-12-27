#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * TersoffB (Si(B)) potential minimization, faithful to CEC2011 form.
 *
 * - Decision vector size D = 3N - 6 (default N=10 -> D=24).
 *
 * - Geometry encoding:
 *     Atom 0: (0,0,0)
 *     Atom 1: (x0, 0, 0)                with x0 in [0,4]
 *     Atom 2: (x1 cos x2, x1 sin x2, 0) with x1 in [0,4], x2 in [0,π]
 *     Atoms 3..N-1: Cartesian triples from x[3..]
 *
 * - Bounds for i>=3: [-4.25, 4.25]
 *
 * - Objective:
 *     E(x) = sum_{i<j} f_c(r_ij)[ V_R(r_ij) - B_ij V_A(r_ij) ]
 *
 *   with the usual cosine cutoff f_c, Tersoff pair terms V_R, V_A,
 *   and angular environment term g(θ) inside B_ij.
 */
class TersoffB : public Problem {
public:
    TersoffB();

    // If dim has the form 3N-6 with N>=3, infer N from dim; otherwise N=10 (D=24).
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;   // total Tersoff energy
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward differences

private:
    // Atoms
    int num_atoms_;  // N
    int D_;          // 3N - 6

    // Tersoff parameters (Si(B)) from the provided reference
    double A_, B_, lambda1_, lambda2_, beta_, n_, c_, d_, h_, R_, Dcut_, lambda3_;
    int    m_;

    // ----- geometry helpers -----
    static double distance(const std::vector<double>& a, const std::vector<double>& b);
    static double angle_abc(const std::vector<double>& a,
                            const std::vector<double>& b,
                            const std::vector<double>& c);

    // reconstruct positions R^{N x 3} from x
    std::vector<std::vector<double>> reconstruct_positions(const Vec& x) const;

    // cutoff and pair terms
    double fc(double r) const;     // cosine cutoff
    double VR(double r) const;     // repulsive
    double VA(double r) const;     // attractive
    double Bij(int i, int j, const std::vector<std::vector<double>>& pos) const; // environment term

    static inline double sqr(double v) { return v*v; }
};

} // namespace optimsolution
