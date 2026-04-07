#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F11 (reference helper cf06) - Composition Function 6.
 *
 * Official CEC 2022 mapping:
 *   - overall function number: 11
 *   - helper name in the released C code: cf06
 *
 * Composition structure:
 *   g1: Expanded Schaffer F6
 *   g2: Modified Schwefel
 *   g3: Griewank
 *   g4: Rotated Rosenbrock
 *   g5: Rotated Rastrigin
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift data and rotation matrices are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Composition6 : public Problem {
public:
    CEC2022Composition6();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // concatenated 5*D shift data
    std::vector<double> rotation_; // concatenated 5*(D*D) row-major matrices
};

} // namespace optimsolution
