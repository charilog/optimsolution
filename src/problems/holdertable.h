#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Holder Table function.
 *
 * f(x,y) = -| sin(x)*cos(y)*exp(|1 - sqrt(x^2+y^2)/pi|) |
 *
 * Domain: [-10, 10]^2
 * Global minimum: f ~= -19.2085 at four points (+-8.05502, +-9.66459).
 */
class HolderTable : public Problem {
public:
    HolderTable();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
