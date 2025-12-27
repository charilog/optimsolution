#pragma once
#include "problem.h"

namespace optimsolution {

class Bohachevsky2 : public Problem {
public:
    Bohachevsky2();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;  // f2(x,y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
