#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F6 - Shifted and Rotated Expanded Scaffer's F6 Function.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * g(u,v) = 0.5 + (sin^2(sqrt(u^2+v^2)) - 0.5) / (1 + 0.001*(u^2+v^2))^2
 * f(x)   = [ sum_{i=1}^{D-1} sqrt(w_i) + sqrt(w_i)*sin^2(50*w_i^0.2) ]^2 / (D-1)^2 + f_bias
 *   where w_i = sqrt(y_i^2 + y_{i+1}^2), y = x - o  (see IMPORTANT note below)
 *
 * *** IMPORTANT / faithfulness note ***
 * The official reference implementation calls its generic shift-and-rotate
 * helper (requesting rotation), but then a downstream buffer-reuse bug
 * makes the actual per-coordinate term use the SHIFTED-ONLY vector (the
 * intermediate value the helper leaves behind in its scratch buffer),
 * never the rotated one -- so despite being named/classified as a rotated
 * function and having an official rotation matrix bundled for it, F6's
 * published/competition results are NUMERICALLY IDENTICAL to using only
 * the shift, with the bundled rotation matrix having no effect at all.
 * This was independently confirmed here by compiling the official
 * reference code and comparing both interpretations against it: only the
 * "shift-only" version reproduces the official output; a "correctly"
 * rotated version does not. This is also documented by independent
 * reimplementations (e.g. the tilleyd/cec2017-py package's README notes
 * several such reference-code quirks that must be replicated, not
 * "fixed", to stay compatible with published results). This
 * implementation therefore intentionally does NOT apply the rotation
 * matrix to the term used in evaluate_core(), to remain numerically
 * identical to every published CEC2017 comparison table. The rotation
 * matrix is still loaded/stored (for provenance/possible future use) but
 * is not applied.
 *
 *   - Simple multimodal, effectively separable-after-shift in this
 *     implementation (see note above)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 600 (F6* = 600)
 *   - Global minimum: f(x*) = 600 at x* = o (the shift vector)
 *
 * Data note: shift vector and rotation matrix are the EXACT OFFICIAL data
 * (input_data/shift_data_6.txt, input_data/M_6_D{10,30,50}.txt), verified
 * against a locally recompiled copy of the official cec17_test_func.cpp
 * reference (see cec2017_f1.h for the fuller verification methodology).
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F6 : public Problem {
public:
    CEC2017F6();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // D shift values (o)
    std::vector<double> rotation_; // D*D row-major rotation matrix (M); loaded but unused, see note above
};

} // namespace optimsolution
