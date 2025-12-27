#pragma once
#include "problem.h"

namespace optimsolution {

class Gkls350 : public Problem {
public:
    Gkls350();
    void init(int dim) override;  // fixed D=3

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
