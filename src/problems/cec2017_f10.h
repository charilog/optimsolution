#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F10 - Shifted and Rotated Schwefel's Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 *   z = 10 * M(x - o)                     (shift, scale by 1000/100, rotate)
 *   for each z_i: z_i += 420.9687462275036
 *     if z_i >  500: f -= (500-fmod(z_i,500))*sin(sqrt(500-fmod(z_i,500)));
 *                    f += ((z_i-500)/100)^2 / D
 *     if z_i < -500: f -= (-500+fmod(|z_i|,500))*sin(sqrt(500-fmod(|z_i|,500)));
 *                    f += ((z_i+500)/100)^2 / D
 *     else:          f -= z_i*sin(sqrt(|z_i|))
 *   f += 418.9828872724338 * D
 *   f(x) = f + f_bias
 *
 * The additive constants (420.9687462275036, 418.9828872724338) recenter
 * the classical Schwefel function so f(x*) = f_bias exactly at x* = o (the
 * unshifted Schwefel optimum sits near 420.9687 per coordinate, where each
 * term contributes about -418.9829).
 *
 *   - Simple multimodal, non-separable (via rotation), many local optima
 *     spread far from the global one
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 1000 (F10* = 1000)
 *   - Global minimum: f(x*) = 1000 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_10.txt, input_data/M_10_D{10,30,50}.txt), verified
 * against a locally recompiled copy of the official cec17_test_func.cpp
 * reference (see cec2017_f1.h for the fuller verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F10 : public Problem {
public:
    CEC2017F10();
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
