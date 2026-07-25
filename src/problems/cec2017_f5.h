#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F5 - Shifted and Rotated Rastrigin's Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * f(x) = sum_{i=1}^{D} [z_i^2 - 10*cos(2*pi*z_i) + 10] + f_bias
 *   z = 5.12/100 * M(x - o)
 *
 *   - Simple multimodal, highly multimodal with a regular grid of local
 *     optima, non-separable (via rotation)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 500 (F5* = 500)
 *   - Global minimum: f(x*) = 500 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_5.txt, input_data/M_5_D{10,30,50}.txt from the
 * official release), verified against a locally recompiled copy of the
 * official cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification note, which applies identically here).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F5 : public Problem {
public:
    CEC2017F5();
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
