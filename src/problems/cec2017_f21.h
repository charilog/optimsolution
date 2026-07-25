#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F21 - Composition Function 1 (N = 3). This is the FIRST
 * composition function in the suite (F21-F30 are all composition
 * functions).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * A composition function blends N basic functions, each with its OWN
 * independent shift vector and rotation matrix, using a distance-based
 * weighting so the global optimum coincides with whichever component's own
 * optimum is closest to the query point (each component "owns" a region of
 * the search space):
 *
 *   for each component i = 1..N:
 *     z_i = M_i * (x - o_i)                (component's own shift+rotate)
 *     fit_i = (that component's classical formula, applied to z_i,
 *              scaled to a comparable magnitude) + bias_i
 *     w_i = ||x - o_i||^{-1} * exp(-||x-o_i||^2 / (2*D*delta_i^2))
 *           (or +inf if x == o_i exactly)
 *   f(x) = sum_i (w_i / sum(w)) * fit_i + f_bias
 *
 * Components here (delta = [10,20,30], bias = [0,100,200]):
 *   1: Rosenbrock                              (scale 2.048/100)
 *   2: High-Conditioned Elliptic, rescaled by 10000/1e10
 *   3: Rastrigin                               (scale 5.12/100)
 *
 *   - Composition, non-separable, multiple funnels (each component defines
 *     its own basin; the "active" one depends on which region of the
 *     search space x falls into)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2100 (F21* = 2100)
 *   - Global minimum: f(x*) = 2100 at x* = o_1 (component 1's shift vector,
 *     since bias_1 = 0 is the smallest of the per-component biases)
 *
 * Data note: each component's shift vector and rotation matrix are the
 * EXACT OFFICIAL data. IMPORTANT data-layout subtlety (verified against a
 * locally recompiled copy of the official cec17_test_func.cpp reference):
 * the official shift_data_21.txt file stores each component's shift vector
 * on its OWN LINE (10 generic slots, padded to 100 columns, of which only
 * the first D are used for a given dimension) -- it is NOT a flat
 * concatenation of N*D consecutive values as in some other multi-component
 * CEC files (e.g. the CEC2022 composition functions elsewhere in this
 * codebase). The rotation matrix file, by contrast, IS a flat sequential
 * concatenation of component blocks. This implementation's embedded arrays
 * already reflect the correctly de-interleaved values (row i truncated to
 * the first D columns, for i=0..N-1), so evaluate_core() itself can treat
 * `shift_` as a simple flat N*D array exactly like the CEC2022 examples do
 * -- the row/column subtlety only mattered during data extraction.
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F21 : public Problem {
public:
    CEC2017F21();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // concatenated 3*D shift data (component-major)
    std::vector<double> rotation_; // concatenated 3*(D*D) row-major matrices
};

} // namespace optimsolution
