#pragma once
#include "problem.h"

namespace optimsolution {

class Bohachevsky3 : public Problem {
public:
    Bohachevsky3();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;  // f3(x,y)
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
