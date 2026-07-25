#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F30 - Composition Function 10 (N = 3). This is the LAST function
 * in the CEC 2017 suite. Like F29, its components are ENTIRE hybrid
 * functions (each with its own shift, rotation, AND shuffle permutation).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * See cec2017_f21.h for the general composition-function blending (cf_cal)
 * and the shift-file row-layout note, and cec2017_f29.h for the
 * composition-of-hybrids data-layout note (rotation/shuffle files are flat
 * per-component concatenations, unlike the row-per-component shift file).
 * Each component here is itself an entire hybrid function -- see
 * cec2017_f15.h/f18.h/f19.h for the exact per-hybrid group structure and
 * formulas (Hybrid Functions 5, 8, and 9 respectively), reproduced
 * verbatim below as self-contained sub-routines.
 *
 * Components (delta = [10,30,50], bias = [0,100,200]):
 *   1: Hybrid Function 5 (N=4: BentCigar, HGBat, Rastrigin, Rosenbrock)
 *   2: Hybrid Function 8 (N=5: Ellip, Ackley, Rastrigin, HGBat, Discus)
 *   3: Hybrid Function 9 (N=5: BentCigar, Rastrigin, GrieRosen, Weierstrass, ScafferF6)
 *
 *   - Composition-of-hybrids, non-separable, multiple funnels each with
 *     internal discontinuous "seams" (from the hybrid grouping/shuffle)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 3000 (F30* = 3000)
 *   - Global minimum: f(x*) = 3000 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F30 : public Problem {
public:
    CEC2017F30();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;
    std::vector<double> rotation_;
    std::vector<int> shuffle_;
};

} // namespace optimsolution
