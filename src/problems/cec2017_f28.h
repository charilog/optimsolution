#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F28 - Composition Function 8 (N = 6). This is the LAST
 * composition function built purely from basic functions -- F29 and F30
 * compose entire hybrid functions instead.
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
 *   1: Ackley, rescaled by 1000/100
 *   2: Griewank, rescaled by 1000/100
 *   3: Discus, rescaled by 10000/1e10
 *   4: Rosenbrock
 *   5: HappyCat, rescaled by 1000/1e3
 *   6: Expanded Scaffer's F6, rescaled by 10000/2e7
 *
 *   - Composition, non-separable, multiple funnels
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2800 (F28* = 2800)
 *   - Global minimum: f(x*) = 2800 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F28 : public Problem {
public:
    CEC2017F28();
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
