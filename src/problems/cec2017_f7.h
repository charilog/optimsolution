#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F7 - Shifted and Rotated Lunacek Bi_Rastrigin Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * A double-funnel landscape (Lunacek et al.) built from two quadratic
 * "bowls" (means mu0 and mu1) blended with a cosine-based Rastrigin-style
 * ripple, so a naive optimizer that finds one bowl easily can still miss
 * the true global one:
 *
 *   mu0 = 2.5, d = 1
 *   s   = 1 - 1 / (2*sqrt(D+20) - 8.2)
 *   mu1 = -sqrt((mu0^2 - d) / s)
 *   y   = (x - o) * 0.1                 (shift, then scale by 10/100)
 *   z_i = 2*y_i * sign(o_i)              (sign flip mirrors the two bowls
 *                                         to align with the shift vector)
 *   tmp1 = sum(z_i^2)                            [distance^2 to bowl at mu0]
 *   tmp2 = s * sum((z_i + mu0 - mu1)^2) + d*D     [distance^2 to bowl at mu1]
 *   w    = M * z                                  (rotate)
 *   f(x) = min(tmp1, tmp2) + 10*(D - sum(cos(2*pi*w_i))) + f_bias
 *
 *   - Simple multimodal, non-separable (via rotation), double-funnel /
 *     deceptive: two widely-separated quadratic basins, only one of which
 *     contains the true optimum
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 700 (F7* = 700)
 *   - Global minimum: f(x*) = 700 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_7.txt, input_data/M_7_D{10,30,50}.txt), verified
 * against a locally recompiled copy of the official cec17_test_func.cpp
 * reference (see cec2017_f1.h for the fuller verification methodology).
 * Unlike F1/F3/F4/F5, this function's official code does not go through
 * the shared sr_func() helper (it has its own bespoke shift/scale/sign-flip
 * pipeline above), so this implementation mirrors that bespoke pipeline
 * directly rather than reusing a generic sr_func().
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F7 : public Problem {
public:
    CEC2017F7();
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
