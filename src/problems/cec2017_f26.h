#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F26 - Composition Function 6 (N = 5).
 *
 * Reference: N.H. Awad, M.Z. Ali, P.N. Suganthan, J.J. Liang, B.Y. Qu,
 * "Problem Definitions and Evaluation Criteria for the CEC 2017 Special
 * Session and Competition on Single Objective Bound Constrained
 * Real-Parameter Numerical Optimization", Nanyang Technological University,
 * Tech. Rep., 2016. Official code/data: github.com/P-N-Suganthan/CEC2017-BoundContrained.
 *
 * See cec2017_f21.h for the general composition-function structure and the
 * data-layout note (shift file is row-per-component; already correctly
 * de-interleaved into this file's embedded arrays).
 *
 * Components (delta = [10,20,20,30,40], bias = [0,100,200,300,400]):
 *   1: Expanded Scaffer's F6, rescaled by 10000/2e7
 *   2: Schwefel
 *   3: Griewank, rescaled by 1000/100
 *   4: Rosenbrock
 *   5: Rastrigin, rescaled by 10000/1e3
 *
 *   - Composition, non-separable, multiple funnels
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2600 (F26* = 2600)
 *   - Global minimum: f(x*) = 2600 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F26 : public Problem {
public:
    CEC2017F26();
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
