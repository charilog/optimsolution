#pragma once
#include "problem.h"

namespace optimsolution {

class Bohachevsky1 : public Problem {
public:
    Bohachevsky1();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;  // f1(x,y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
