#pragma once
#include "problem.h"

namespace optimsolution {

class BucheRastrigin : public Problem {
public:
    BucheRastrigin();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec scale_;  // scaling vector s_i
};

} // namespace optimsolution
