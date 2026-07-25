#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F8 - Shifted and Rotated Non-Continuous Rastrigin's Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * f(x) = sum_{i=1}^{D} [z_i^2 - 10*cos(2*pi*z_i) + 10] + f_bias,
 *   z = 5.12/100 * M(x - o)
 *
 * *** IMPORTANT / faithfulness note ***
 * As NAMED, this function is supposed to first snap any shifted coordinate
 * more than 0.5 away from the shift into a coarser grid (the classic
 * "non-continuous"/step Rastrigin construction), before shifting/rotating.
 * In the official reference code, that snapping step operates on a scratch
 * buffer that is then immediately overwritten from scratch by the
 * subsequent shift-and-rotate call, before the snapped values are ever
 * used -- so the snapping has NO effect on the actual, published result.
 * F8 is therefore numerically IDENTICAL to a plain Shifted-and-Rotated
 * Rastrigin Function (same shape as F5), just with its own shift
 * vector/rotation matrix and a different bias (800 instead of 500). This
 * was confirmed here by compiling the official reference code directly:
 * a "properly stepped" reimplementation does NOT reproduce the official
 * output, while the plain (unstepped) rotated Rastrigin does, exactly.
 * Other independent reimplementations note the same kind of reference-code
 * quirk elsewhere in the suite and deliberately replicate it rather than
 * "fix" it, to stay compatible with every published comparison table; this
 * implementation does the same.
 *
 *   - Simple multimodal, non-separable (via rotation)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 800 (F8* = 800)
 *   - Global minimum: f(x*) = 800 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_8.txt, input_data/M_8_D{10,30,50}.txt), verified
 * against a locally recompiled copy of the official cec17_test_func.cpp
 * reference (see cec2017_f1.h for the fuller verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F8 : public Problem {
public:
    CEC2017F8();
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
