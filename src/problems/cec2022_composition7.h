#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * CEC 2022 F12 (reference helper cf07) - Composition Function 7.
 *
 * Important note:
 *   - In the released reference C code, the helper name is `cf07`.
 *   - In the technical report, overall function 12 is the fourth composition
 *     problem (often described as Composition Function 4).
 *   - This file uses the user-requested label "Composition Function 7" to stay
 *     aligned with the reference helper naming.
 *
 * Composition structure used by the official code:
 *   g1: HGBat
 *   g2: Rotated Rastrigin
 *   g3: Rotated Schwefel
 *   g4: Rotated Bent Cigar
 *   g5: Rotated High Conditioned Elliptic
 *   g6: Rotated Expanded Schaffer F6
 *
 * Notes:
 *   - Search domain: [-100, 100]^D
 *   - Supported dimensions: D = 2, 10, 20
 *   - Embedded official shift data and rotation matrices are used.
 *   - gradient_core() uses central finite differences for API compatibility.
 */
class CEC2022Composition7 : public Problem {
public:
    CEC2022Composition7();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void gradient_core(const Vec& x, Vec& g) override;

private:
    void load_embedded_data(int dim);

    Vec shift_;                    // concatenated 6*D shift data
    std::vector<double> rotation_; // concatenated 6*(D*D) row-major matrices
};

} // namespace optimsolution
