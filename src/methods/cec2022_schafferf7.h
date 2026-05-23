#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F3 - Shifted and Rotated Expanded Schaffer F7 Function.
 *
 * Definition:
 *   F(x) = f(M (x - o)) + 600,
 *
 * where
 *   f(z) = [ (1 / (D - 1)) * sum_{i=1}^{D-1} ( sqrt(s_i) + sqrt(s_i) * sin^2(50 s_i^0.2) ) ]^2,
 *   s_i = sqrt(z_i^2 + z_{i+1}^2).
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift vector and rotation matrices are used for D = 2, 10, 20.
 */
class CEC2022SchafferF7 : public Problem {
public:
    CEC2022SchafferF7();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;
    std::vector<double> rotation_; // row-major D x D rotation matrix
};

} // namespace optimsolution
