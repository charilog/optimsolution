#pragma once
#include "problem.h"
#include <cmath>

namespace optimsolution {

class Ackley : public Problem {
public:
    Ackley();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
