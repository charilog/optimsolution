#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F10 (reference helper cf02) - Composition Function 2.
 *
 * Official CEC 2022 mapping:
 *   - overall function number: 10
 *   - helper name in the released reference C code: cf02
 *
 * Composition structure used by the official code:
 *   g1: Schwefel
 *   g2: Rotated Rastrigin
 *   g3: HGBat
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift data and rotation matrices are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Composition2 : public Problem {
public:
    CEC2022Composition2();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // concatenated 3*D shift data
    std::vector<double> rotation_; // concatenated 3*(D*D) row-major matrices
};

} // namespace optimsolution
