#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F29 - Composition Function 9 (N = 3). Unlike F21-F28, this
 * composition's components are ENTIRE hybrid functions (each with its own
 * shift, rotation, AND shuffle permutation), not simple basic functions.
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * See cec2017_f21.h for the general composition-function blending (cf_cal)
 * and the shift-file row-layout note. Each component here is itself an
 * entire hybrid function -- see cec2017_f15.h/f16.h/f17.h for the exact
 * per-hybrid group structure and formulas (Hybrid Functions 5, 6, and 7
 * respectively), reproduced verbatim below as self-contained sub-routines.
 *
 * Components (delta = [10,30,50], bias = [0,100,200]):
 *   1: Hybrid Function 5 (N=4: BentCigar, HGBat, Rastrigin, Rosenbrock)
 *   2: Hybrid Function 6 (N=4: ScafferF6, HGBat, Rosenbrock, Schwefel)
 *   3: Hybrid Function 7 (N=5: Katsuura, Ackley, GrieRosen, Schwefel, Rastrigin)
 *
 *   - Composition-of-hybrids, non-separable, multiple funnels each with
 *     internal discontinuous "seams" (from the hybrid grouping/shuffle)
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2900 (F29* = 2900)
 *   - Global minimum: f(x*) = 2900 at x* = component 1's shift vector
 *
 * Data note: each component needs its OWN shift vector, rotation matrix,
 * AND shuffle permutation. The shift file follows the same row-per-
 * component layout described in cec2017_f21.h; the rotation AND shuffle
 * files, unlike the shift file, are each sized exactly for the requested
 * dimension (no padding) and are read as a flat sequential concatenation
 * of per-component blocks -- this implementation's embedded arrays already
 * reflect the correctly-extracted values either way. Verified against a
 * locally recompiled copy of the official reference.
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F29 : public Problem {
public:
    CEC2017F29();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // concatenated 3*D shift data (component-major)
    std::vector<double> rotation_; // concatenated 3*(D*D) row-major matrices
    std::vector<int> shuffle_;     // concatenated 3*D shuffle permutations (1-indexed as stored)
};

} // namespace optimsolution
