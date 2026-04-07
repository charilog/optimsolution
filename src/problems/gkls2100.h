#pragma once
#include "problem.h"

namespace optimsolution {

class Gkls2100 : public Problem {
public:
    Gkls2100();
    void init(int dim) override;  // fixed D=2

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
