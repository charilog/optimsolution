#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F6 (reference helper hf02) - Hybrid Function 2.
 *
 * Definition used in the official CEC 2022 reference C code:
 *   1) z = M (x - o)
 *   2) y = z shuffled by the official permutation S
 *   3) F(x) = BentCigar(y_1) + HGBat(y_2) + Rastrigin(y_3) + 1800
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 10, 20
 *   - Embedded official shift vector, rotation matrix, and shuffle permutation are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Hybrid2 : public Problem {
public:
    CEC2022Hybrid2();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;
    std::vector<double> rotation_; // row-major D x D matrix
    std::vector<int> shuffle_;     // 1-based indices, as in the official code
};

} // namespace optimsolution
