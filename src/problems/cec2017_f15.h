#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F15 - Hybrid Function 5 (N = 4).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.2 D)-1]             -> Bent Cigar
 *   G2 = y[|G1| .. |G1|+ceil(0.2 D)-1]     -> HGBat      (scale 5/100)
 *   G3 = y[|G1|+|G2| .. |G1|+|G2|+ceil(0.3 D)-1] -> Rastrigin (scale 5.12/100)
 *   G4 = y[|G1|+|G2|+|G3| .. D-1]          -> Rosenbrock (scale 2.048/100)
 *   f(x) = BentCigar(G1) + HGBat(G2) + Rastrigin(G3) + Rosenbrock(G4) + f_bias
 *
 * No reference-code buffer-aliasing quirks were found in this hybrid (all
 * four sub-functions correctly read only their own group's slice); this
 * was confirmed against a locally recompiled copy of the reference.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1500 (F15* = 1500)
 *   - Global minimum: f(x*) = 1500 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_15.txt,
 * input_data/M_15_D{10,30,50}.txt, input_data/shuffle_data_15_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F15 : public Problem {
public:
    CEC2017F15();
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
