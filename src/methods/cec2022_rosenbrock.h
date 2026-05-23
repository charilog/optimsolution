#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F2 - Shifted and Rotated Rosenbrock Function.
 *
 * Definition:
 *   F(x) = f(M (2.048 * (x - o) / 100.0) + 1) + 400,
 *
 * where f is the classical Rosenbrock chain:
 *   f(z) = sum_{i=1}^{D-1} [100 (z_i^2 - z_{i+1})^2 + (z_i - 1)^2].
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift vector and rotation matrices are used for D = 2, 10, 20.
 */
class CEC2022Rosenbrock : public Problem {
public:
    CEC2022Rosenbrock();
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
