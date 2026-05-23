#pragma once
#include "problem.h"

namespace optimsolution {

class DifferentPowers : public Problem {
public:
    DifferentPowers();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec exp_;    // <--- NECESSARY exponent vector for dimension-dependent powers
};

} // namespace optimsolution
