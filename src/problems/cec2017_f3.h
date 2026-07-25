#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F3 - Shifted and Rotated Zakharov Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * f(x) = sum1 + sum2^2 + sum2^4 + f_bias,  z = M(x - o)
 *   sum1 = sum_{i=1}^{D} z_i^2
 *   sum2 = sum_{i=1}^{D} 0.5*i*z_i
 *
 *   - Unimodal, non-separable (via rotation)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 300 (F3* = 300)
 *   - Global minimum: f(x*) = 300 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_3.txt, input_data/M_3_D{10,30,50}.txt from the
 * official release), verified against a locally recompiled copy of the
 * official cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification note, which applies identically here).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F3 : public Problem {
public:
    CEC2017F3();
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
