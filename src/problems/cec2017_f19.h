#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F19 - Hybrid Function 9 (N = 5).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   Five EQUAL fifths of D (all Gp = 0.2, last group takes the remainder):
 *   G1 -> Bent Cigar
 *   G2 -> Rastrigin        (scale 5.12/100)
 *   G3 -> Griewank-Rosenbrock (scale 5/100, circular)
 *   G4 -> Weierstrass      (scale 0.5/100)
 *   G5 -> Expanded Scaffer's F6 (circular; the non-quirky variant, see F16)
 *   f(x) = BentCigar(G1) + Rastrigin(G2) + GrieRosen(G3) + Weierstrass(G4)
 *        + ScafferF6(G5) + f_bias
 *
 * No reference-code buffer-aliasing quirks were found in this hybrid (all
 * five sub-functions correctly read only their own group's slice, including
 * group 5's Expanded Scaffer's F6 -- the non-quirky `escaffer6_func`
 * variant, not the standalone F6's `schaffer_F7_func`); this was confirmed
 * against a locally recompiled copy of the reference.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1900 (F19* = 1900)
 *   - Global minimum: f(x*) = 1900 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_19.txt,
 * input_data/M_19_D{10,30,50}.txt, input_data/shuffle_data_19_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F19 : public Problem {
public:
    CEC2017F19();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;
    std::vector<double> rotation_;
    std::vector<int> shuffle_;
};

} // namespace optimsolution
