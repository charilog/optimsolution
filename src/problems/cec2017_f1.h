#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F1 - Shifted and Rotated Bent Cigar Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * f(x) = z_1^2 + 1e6 * sum_{i=2}^{D} z_i^2 + f_bias,  z = M(x - o)
 *
 *   - Unimodal, non-separable (via rotation), optimum in a smooth but very
 *     narrow valley (the classic "bent cigar" shape).
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 100 (per the official numbering, F1* = 100)
 *   - Global minimum: f(x*) = 100 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data,
 * extracted verbatim from the official release
 * (P-N-Suganthan/CEC2017-BoundContrained, "Shifting and Rotation for CEC
 * 2017" archive: input_data/shift_data_1.txt and input_data/M_1_D{10,30,50}.txt).
 * Verified against a locally recompiled copy of the official
 * cec17_test_func.cpp reference (fixing its well-known %Lf/%lf scanf format
 * bug, which otherwise corrupts data loading when compiled under
 * Linux/GCC): this implementation reproduces the reference binary's output
 * exactly (to full double precision) for D=10, 30 and 50 at x=0, at
 * x=shift (f=100.0 exactly, as expected from the formula), and at several
 * random points.
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F1 : public Problem {
public:
    CEC2017F1();
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
