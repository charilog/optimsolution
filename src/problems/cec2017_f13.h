#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F13 - Hybrid Function 3 (N = 3).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.3 D)-1]             -> Bent Cigar
 *   G2 = y[|G1| .. |G1|+ceil(0.3 D)-1]     -> Rosenbrock (scale 2.048/100)
 *   G3 = y[|G1|+|G2| .. D-1]               -> Lunacek Bi_Rastrigin
 *   f(x) = BentCigar(G1) + Rosenbrock(G2) + BiRastrigin(G3) + f_bias
 *
 * *** IMPORTANT / faithfulness note ***
 * The official reference implementation calls the Bi_Rastrigin sub-function
 * with the group-3 slice as its data, but passes the ORIGINAL, UNSLICED
 * shift vector (not offset to the group's own position) for the
 * sign-flip decision inside that sub-function (`if (Os[i] < 0) ... `). It
 * therefore ends up testing the sign of the FIRST |G3| elements of the
 * whole-problem shift vector, not the shift values that actually correspond
 * to group 3's shuffled positions. This was confirmed here by compiling the
 * official reference code and comparing this literal (unsliced) behavior
 * against a "corrected" (properly offset) one: only the unsliced version
 * reproduces the official output. This implementation intentionally
 * replicates the literal (unsliced) behavior, to remain numerically
 * identical to every published CEC2017 comparison table.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1300 (F13* = 1300)
 *   - Global minimum: f(x*) = 1300 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_13.txt,
 * input_data/M_13_D{10,30,50}.txt, input_data/shuffle_data_13_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F13 : public Problem {
public:
    CEC2017F13();
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
