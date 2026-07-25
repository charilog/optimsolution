#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F20 - Hybrid Function 10 (N = 6). This is the LAST hybrid
 * function in the suite (F21 onward are composition functions).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.1 D)-1]                          -> HGBat     (scale 5/100)
 *   G2 = y[|G1| .. |G1|+ceil(0.1 D)-1]                  -> Katsuura  (scale 5/100)
 *   G3 = y[|G1..2| .. |G1..2|+ceil(0.2 D)-1]             -> Ackley
 *   G4 = y[|G1..3| .. |G1..3|+ceil(0.2 D)-1]             -> Rastrigin (scale 5.12/100)
 *   G5 = y[|G1..4| .. |G1..4|+ceil(0.2 D)-1]             -> Schwefel
 *   G6 = y[|G1..5| .. D-1]                               -> Expanded Scaffer's F6 (quirky)
 *   f(x) = HGBat(G1) + Katsuura(G2) + Ackley(G3) + Rastrigin(G4) + Schwefel(G5)
 *        + ScafferF6_quirky(G6) + f_bias
 *
 * *** IMPORTANT / faithfulness note ***
 * Group 6's sub-function call is the reference code's `schaffer_F7_func`
 * (the SAME function/quirk documented for the standalone F6, NOT the
 * non-quirky `escaffer6_func` used elsewhere in this suite -- e.g. F16,
 * F19). Even though this is the LAST group (not the first, as in the
 * standalone F6), the same buffer-aliasing bug applies: it reads the FIRST
 * |G6| elements of the WHOLE shuffled vector (i.e. part of group 1's data),
 * not group 6's own slice. This was confirmed here by compiling the
 * official reference code directly and comparing this literal behavior
 * against a "corrected" one that reads group 6's own slice: only the
 * literal (misaligned) behavior reproduces the official output. This
 * implementation intentionally replicates it, to remain numerically
 * identical to every published CEC2017 comparison table.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2000 (F20* = 2000)
 *   - Global minimum: f(x*) = 2000 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_20.txt,
 * input_data/M_20_D{10,30,50}.txt, input_data/shuffle_data_20_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F20 : public Problem {
public:
    CEC2017F20();
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
