#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F17 - Hybrid Function 7 (N = 5).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.1 D)-1]                    -> Katsuura    (scale 5/100)
 *   G2 = y[|G1| .. |G1|+ceil(0.2 D)-1]             -> Ackley
 *   G3 = y[|G1|+|G2| .. |G1|+|G2|+ceil(0.2 D)-1]   -> Griewank-Rosenbrock (scale 5/100)
 *   G4 = y[|G1|+|G2|+|G3| .. |G1|+|G2|+|G3|+ceil(0.2 D)-1] -> Schwefel
 *   G5 = y[|G1|+|G2|+|G3|+|G4| .. D-1]             -> Rastrigin (scale 5.12/100)
 *   f(x) = Katsuura(G1) + Ackley(G2) + GrieRosen(G3) + Schwefel(G4)
 *        + Rastrigin(G5) + f_bias
 *
 * No reference-code buffer-aliasing quirks were found in this hybrid (all
 * five sub-functions correctly read only their own group's slice); this
 * was confirmed against a locally recompiled copy of the reference. Note
 * the Katsuura sub-function's scale factor is 5/100 here (as in the
 * standalone Katsuura-family functions elsewhere in the suite), which is
 * easy to mis-transcribe as "no scaling" -- double-checked against the
 * reference source directly.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1700 (F17* = 1700)
 *   - Global minimum: f(x*) = 1700 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_17.txt,
 * input_data/M_17_D{10,30,50}.txt, input_data/shuffle_data_17_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F17 : public Problem {
public:
    CEC2017F17();
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
