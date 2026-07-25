#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F11 - Hybrid Function 1 (N = 3).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * The first hybrid function: the decision vector is shifted, rotated, and
 * then SHUFFLED via a fixed random permutation, before being split into
 * three contiguous, differently-sized groups (proportions 0.2 / 0.4 / 0.4
 * of D, official symbol Gp), each evaluated by a DIFFERENT classical
 * function (each with its own internal scale factor, exactly as in F3/F4/F5
 * respectively, but with no further shift/rotate of its own -- that
 * already happened once, at the top). The three partial results are simply
 * summed:
 *
 *   z = M(x - o)                          (shift then rotate)
 *   y_i = z_{S(i)}                        (S = fixed shuffle permutation)
 *   G1 = y[0 .. ceil(0.2 D)-1]             -> Zakharov      (sum1+sum2^2+sum2^4)
 *   G2 = y[|G1| .. |G1|+ceil(0.4 D)-1]     -> Rosenbrock     (scale 2.048/100)
 *   G3 = y[|G1|+|G2| .. D-1]               -> Rastrigin      (scale 5.12/100)
 *   f(x) = Zakharov(G1) + Rosenbrock(G2) + Rastrigin(G3) + f_bias
 *
 *   - Hybrid, non-separable, discontinuous "seams" between groups
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1100 (F11* = 1100)
 *   - Global minimum: f(x*) = 1100 at x* = o (the shift vector)
 *
 * Data note: shift vector, rotation matrix AND the shuffle permutation are
 * the EXACT OFFICIAL data (input_data/shift_data_11.txt,
 * input_data/M_11_D{10,30,50}.txt, input_data/shuffle_data_11_D{10,30,50}.txt),
 * verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F11 : public Problem {
public:
    CEC2017F11();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // D shift values (o)
    std::vector<double> rotation_; // D*D row-major rotation matrix (M)
    std::vector<int> shuffle_;     // D-length permutation, 1-indexed as stored (converted on use)
};

} // namespace optimsolution
