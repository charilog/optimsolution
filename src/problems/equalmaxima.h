#pragma once
#include "problem.h"

namespace optimsolution {

class EqualMaxima : public Problem {
public:
    EqualMaxima();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;  // f(x) = sin^6(5πx)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
