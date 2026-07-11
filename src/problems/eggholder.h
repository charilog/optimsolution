#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Eggholder function.
 *
 * f(x,y) = -(y+47)*sin(sqrt(|x/2+(y+47)|)) - x*sin(sqrt(|x-(y+47)|))
 *
 * Domain: [-512, 512]^2
 * Global minimum: f(512, 404.2319) ~= -959.6407
 */
class Eggholder : public Problem {
public:
    Eggholder();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs
};

} // namespace optimsolution
