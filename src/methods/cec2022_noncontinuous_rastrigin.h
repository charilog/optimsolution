#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F4 - Shifted and Rotated Noncontinuous Rastrigin Function.
 *
 * Definition:
 *   F(x) = f(M (5.12 * (x_nc - o) / 100.0)) + 800,
 *
 * where x_nc is the noncontinuous version of x:
 *   if |x_i - o_i| > 0.5 then
 *       x_nc_i = o_i + floor(2 (x_i - o_i) + 0.5) / 2
 *   else
 *       x_nc_i = x_i,
 *
 * and f is the classical Rastrigin function:
 *   f(z) = sum_i [ z_i^2 - 10 cos(2 pi z_i) + 10 ].
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift vector and rotation matrices are used for D = 2, 10, 20.
 */
class CEC2022NoncontinuousRastrigin : public Problem {
public:
    CEC2022NoncontinuousRastrigin();
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
