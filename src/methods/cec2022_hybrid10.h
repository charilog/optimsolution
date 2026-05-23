#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F7 (reference helper hf10) - Hybrid Function 10.
 *
 * Official CEC 2022 mapping:
 *   - overall function number: 7
 *   - helper name in the released C code: hf10
 *
 * Hybrid composition (after global shift, rotation, and official shuffle):
 *   g1: HGBat
 *   g2: Katsuura
 *   g3: Ackley
 *   g4: Rastrigin
 *   g5: Modified Schwefel
 *   g6: Schaffer F7
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 10, 20
 *   - Embedded official shift vector, rotation matrix, and shuffle permutation are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Hybrid10 : public Problem {
public:
    CEC2022Hybrid10();
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
