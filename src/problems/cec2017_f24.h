#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F24 - Composition Function 4 (N = 4).
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
 * Components here (delta = [10,20,30,40], bias = [0,100,200,300]):
 *   1: Ackley, rescaled by 1000/100
 *   2: High-Conditioned Elliptic, rescaled by 10000/1e10
 *   3: Griewank, rescaled by 1000/100         (scale 600/100)
 *   4: Rastrigin                              (scale 5.12/100)
 *
 *   - Composition, non-separable, multiple funnels
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2400 (F24* = 2400)
 *   - Global minimum: f(x*) = 2400 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F24 : public Problem {
public:
    CEC2017F24();
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
