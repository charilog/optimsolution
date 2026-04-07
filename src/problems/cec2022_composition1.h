#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F9 (reference helper cf01) - Composition Function 1.
 *
 * Official CEC 2022 mapping:
 *   - overall function number: 9
 *   - helper name in the released C code: cf01
 *
 * Composition structure:
 *   g1: Rotated Rosenbrock
 *   g2: High-Conditioned Elliptic
 *   g3: Rotated Bent Cigar
 *   g4: Rotated Discus
 *   g5: High-Conditioned Elliptic (without rotation in the official cf01 code path)
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift data and rotation matrices are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Composition1 : public Problem {
public:
    CEC2022Composition1();
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
