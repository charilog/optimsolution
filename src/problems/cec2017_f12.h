#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F12 - Hybrid Function 2 (N = 3).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.3 D)-1]             -> High-Conditioned Elliptic
 *   G2 = y[|G1| .. |G1|+ceil(0.3 D)-1]     -> Schwefel     (scale 1000/100)
 *   G3 = y[|G1|+|G2| .. D-1]               -> Bent Cigar
 *   f(x) = Ellip(G1) + Schwefel(G2) + BentCigar(G3) + f_bias
 *
 * Unlike F13/F14, none of this hybrid's three sub-functions have the
 * reference-code buffer-aliasing quirks documented there (see cec2017_f13.h
 * and cec2017_f14.h) -- each one here correctly operates on its own group's
 * slice, verified against a locally recompiled copy of the reference.
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1200 (F12* = 1200)
 *   - Global minimum: f(x*) = 1200 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND shuffle permutation are the
 * EXACT OFFICIAL data (input_data/shift_data_12.txt,
 * input_data/M_12_D{10,30,50}.txt, input_data/shuffle_data_12_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F12 : public Problem {
public:
    CEC2017F12();
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
