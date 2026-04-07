#pragma once
#include "problem.h"

namespace optimsolution {

class Rosenbrock : public Problem {
public:
    Rosenbrock();          
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
