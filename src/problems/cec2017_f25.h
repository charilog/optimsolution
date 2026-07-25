#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2017 F25 - Composition Function 5 (N = 5).
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
 * Components here (delta = [10,20,30,40,50], bias = [0,100,200,300,400]):
 *   1: Rastrigin, rescaled by 10000/1e3          (scale 5.12/100)
 *   2: HappyCat, rescaled by 1000/1e3             (scale 5/100)
 *   3: Ackley, rescaled by 1000/100
 *   4: Discus, rescaled by 10000/1e10
 *   5: Rosenbrock                                 (scale 2.048/100)
 *
 *   - Composition, non-separable, multiple funnels
 *   - Search domain: [-100, 100]^D
 *   - f_bias = 2500 (F25* = 2500)
 *   - Global minimum: f(x*) = 2500 at x* = component 1's shift vector
 *
 * Supported dimensions: D = 10, 30, 50.
 */
class CEC2017F25 : public Problem {
public:
    CEC2017F25();
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
