#pragma once
#include "problem.h"

namespace optimsolution {

class StepEllipsoidal : public Problem {
public:
    StepEllipsoidal();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec w_;    // <---- NECESSARY WEIGHTS VECTOR
};

} // namespace optimsolution
