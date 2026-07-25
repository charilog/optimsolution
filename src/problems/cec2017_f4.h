#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F4 - Shifted and Rotated Rosenbrock's Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * f(x) = sum_{i=1}^{D-1} [100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2] + f_bias
 *   z = 2.048/100 * M(x - o), then z_1 += 1 and each z_{i+1} += 1 in turn
 *   (shifting the classical Rosenbrock optimum from (1,...,1) to the origin
 *   of the transformed space; see gradient_core()/evaluate_core() for the
 *   exact sequential update, which matters -- it is NOT a one-shot +1 on a
 *   fixed vector).
 *
 *   - Simple multimodal (narrow curved valley), non-separable (via rotation)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 400 (F4* = 400)
 *   - Global minimum: f(x*) = 400 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_4.txt, input_data/M_4_D{10,30,50}.txt from the
 * official release), verified against a locally recompiled copy of the
 * official cec17_test_func.cpp reference (see cec2017_f1.h for the fuller
 * verification note, which applies identically here).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F4 : public Problem {
public:
    CEC2017F4();
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
