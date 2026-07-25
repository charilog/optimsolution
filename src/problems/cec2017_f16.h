#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F16 - Hybrid Function 6 (N = 4).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.2 D)-1]             -> Expanded Scaffer's F6
 *   G2 = y[|G1| .. |G1|+ceil(0.2 D)-1]     -> HGBat      (scale 5/100)
 *   G3 = y[|G1|+|G2| .. |G1|+|G2|+ceil(0.3 D)-1] -> Rosenbrock (scale 2.048/100)
 *   G4 = y[|G1|+|G2|+|G3| .. D-1]          -> Schwefel
 *   f(x) = ScafferF6(G1) + HGBat(G2) + Rosenbrock(G3) + Schwefel(G4) + f_bias
 *
 * Note: the "Expanded Scaffer's F6" used here (the reference code's
 * `escaffer6_func`, a circular/wrap-around sum over ALL D pairs including
 * (last,first)) is a genuinely DIFFERENT function from the one used in the
 * standalone F6 (`schaffer_F7_func`, a non-wrapping sum over adjacent pairs
 * only, and the one with the buffer-aliasing quirk documented in
 * cec2017_f6.h). `escaffer6_func` has no such quirk -- it correctly
 * operates on its own group's slice throughout; this was confirmed against
 * a locally recompiled copy of the reference.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1600 (F16* = 1600)
 *   - Global minimum: f(x*) = 1600 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_16.txt,
 * input_data/M_16_D{10,30,50}.txt, input_data/shuffle_data_16_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F16 : public Problem {
public:
    CEC2017F16();
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
