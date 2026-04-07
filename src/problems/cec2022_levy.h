#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F5 - Shifted and Rotated Levy Function.
 *
 * Definition used in the official CEC 2022 reference code:
 *   F(x) = levy(M (x - o)) + 900.
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift vector and rotation matrices are used for D = 2, 10, 20.
 */
class CEC2022Levy : public Problem {
public:
    CEC2022Levy();
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
