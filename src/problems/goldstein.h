#pragma once
#include "problem.h"

namespace optimsolution {

class Goldstein : public Problem {
public:
    Goldstein();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;   // f(x,y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
