#pragma once
#include "problem.h"

namespace optimsolution {

class Hansen : public Problem {
public:
    Hansen();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;   // f(x,y) = A(x) * B(y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
