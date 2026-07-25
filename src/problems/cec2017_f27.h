#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F27 - Composition Function 7 (N = 6).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * See cec2017_f21.h for the general composition-function structure and the
 * data-layout note.
 *
 * Components (delta = [10,20,30,40,50,60], bias = [0,100,200,300,400,500]):
 *   1: HGBat, rescaled by 10000/1000              (scale 5/100)
 *   2: Rastrigin, rescaled by 10000/1e3            (scale 5.12/100)
 *   3: Schwefel, rescaled by 10000/4e3
 *   4: Bent Cigar, rescaled by 10000/1e30
 *   5: High-Conditioned Elliptic, rescaled by 10000/1e10
 *   6: Expanded Scaffer's F6, rescaled by 10000/2e7
 *
 *   - Composition, non-separable, multiple funnels
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2700 (F27* = 2700)
 *   - Global minimum: f(x*) = 2700 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50 (per the reference, this and F28 are
 * NOT defined for D=2 -- not a concern here since D=2 is not among this
 * codebase's supported dimensions anyway).
 */
class CEC2017F27 : public Problem {
public:
    CEC2017F27();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;
    std::vector<double> rotation_;
};

} // namespace optimsolution
