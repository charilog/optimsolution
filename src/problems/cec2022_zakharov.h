#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F1 - Shifted and Rotated Zakharov Function.
 *
 * Definition:
 *   F(x) = f(M (x - o)) + 300,
 *
 * where
 *   f(z) = sum(z_i^2) + (sum(0.5 * i * z_i))^2 + (sum(0.5 * i * z_i))^4,
 *   using 1-based indexing in the inner summation.
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift vector and rotation matrices are used for D = 2, 10, 20.
 */
class CEC2022Zakharov : public Problem {
public:
    CEC2022Zakharov();
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
