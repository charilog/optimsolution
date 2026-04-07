#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F8 (reference helper hf06) - Hybrid Function 6.
 *
 * Official CEC 2022 mapping:
 *   - overall function number: 8
 *   - helper name in the released C code: hf06
 *
 * Hybrid composition (after global shift, rotation, and official shuffle):
 *   g1: Katsuura
 *   g2: HappyCat
 *   g3: Expanded Griewank plus Rosenbrock
 *   g4: Modified Schwefel
 *   g5: Ackley
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 10, 20
 *   - Embedded official shift vector, rotation matrix, and shuffle permutation are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Hybrid6 : public Problem {
public:
    CEC2022Hybrid6();
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
