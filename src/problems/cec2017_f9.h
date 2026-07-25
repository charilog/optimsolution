#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F9 - Shifted and Rotated Levy Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = M(x - o)                          (shift then rotate, sh_rate = 1.0)
 *   w_i = 1 + (z_i - 1)/4
 *   f(x) = sin^2(pi*w_1)
 *        + sum_{i=1}^{D-1} (w_i-1)^2 * (1 + 10*sin^2(pi*w_i + 1))
 *        + (w_D-1)^2 * (1 + sin^2(2*pi*w_D))
 *        + f_bias
 *
 * *** faithfulness note *** The middle term's argument is literally
 * `pi*w_i + 1` in the official reference code (add 1 to the angle in
 * radians) -- NOT `pi*(w_i+1)` as the classical Levy-Montalvo formula is
 * sometimes transcribed in papers. This implementation intentionally
 * matches the official CODE (verified against a locally recompiled copy),
 * since that is what every published CEC2017 comparison table was actually
 * computed with.
 *
 *   - Simple multimodal, non-separable (via rotation)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 900 (F9* = 900)
 *   - Known optimal value: f* = 900. Unlike every other function in this
 *     batch, x* = o (the shift vector) is NOT exactly where that value is
 *     attained: the w_i = 1+(z_i-1)/4 transform means the true minimum sits
 *     where z_i = 1 for all i (x = o + M^{-1}*ones), not at z = 0 (x = o).
 *     The embedded "rotation" matrix is not always exactly orthogonal for
 *     this benchmark (confirmed numerically), so M^{-1} != M^T and was not
 *     computed here -- only the optimal f* is reported, not x*.
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_9.txt, input_data/M_9_D{10,30,50}.txt), verified
 * against a locally recompiled copy of the official cec17_test_func.cpp
 * reference (see cec2017_f1.h for the fuller verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F9 : public Problem {
public:
    CEC2017F9();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // D shift values (o)
    std::vector<double> rotation_; // D*D row-major rotation matrix (M)
};

} // namespace optimsolution
