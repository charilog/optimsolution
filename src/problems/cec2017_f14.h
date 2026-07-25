#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F14 - Hybrid Function 4 (N = 4).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.2 D)-1]             -> High-Conditioned Elliptic
 *   G2 = y[|G1| .. |G1|+ceil(0.2 D)-1]     -> Ackley
 *   G3 = y[|G1|+|G2| .. |G1|+|G2|+ceil(0.2 D)-1] -> Expanded Scaffer's F6
 *   G4 = y[|G1|+|G2|+|G3| .. D-1]          -> Rastrigin (scale 5.12/100)
 *   f(x) = Ellip(G1) + Ackley(G2) + ScafferF6(G3) + Rastrigin(G4) + f_bias
 *
 * *** IMPORTANT / faithfulness note ***
 * The official reference implementation's Expanded Scaffer's F6 sub-call
 * (group 3) has the SAME buffer-aliasing quirk documented for the
 * standalone F6 (see cec2017_f6.h): rather than operating on group 3's own
 * slice, it ends up reading the FIRST |G3| elements of the WHOLE shuffled
 * vector (i.e. part of group 1's data), because the sub-function internally
 * reads from a shared scratch buffer that a prior call already overwrote in
 * shift-only form, rather than from the value it was actually just handed.
 * This was confirmed here by compiling the official reference code and
 * comparing this literal (misaligned) behavior against a "corrected" one
 * that reads its own group's slice: only the literal behavior reproduces
 * the official output. This implementation intentionally replicates it, to
 * remain numerically identical to every published CEC2017 comparison table.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1400 (F14* = 1400)
 *   - Global minimum: f(x*) = 1400 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_14.txt,
 * input_data/M_14_D{10,30,50}.txt, input_data/shuffle_data_14_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F14 : public Problem {
public:
    CEC2017F14();
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
