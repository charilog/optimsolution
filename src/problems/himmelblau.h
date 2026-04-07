#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Himmelblau (faithful to provided OPTIMUS version):
 *
 *   f(x1,x2) = 200 - (x1^2 + x2 - 11)^2 - (x1 + x2^2 - 7)^2
 *
 * Bounds: [-6, 6]^2
 *
 * Note: This is a "maximize"-style objective in the original;
 * here we keep the SAME formula (no sign change) for fidelity, so
 * from the optimizer perspective it is a weird landscape with a
 * global maximum ≈ 200 at the classical Himmelblau minima.
 */
class Himmelblau : public Problem {
public:
    Himmelblau();
    void init(int dim) override;                 // forces D=2 and sets bounds [-6,6]^2

protected:
    double evaluate_core(const Vec& x) override; // faithful objective
    void   gradient_core(const Vec& x, Vec& g) override; // central differences

private:
    static inline double dmax(double a, double b) { return (a > b) ? a : b; }
};

} // namespace optimsolution
