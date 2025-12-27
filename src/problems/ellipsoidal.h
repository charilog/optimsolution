#pragma once
#include "problem.h"

namespace optimsolution {

class Ellipsoidal : public Problem {
public:
    Ellipsoidal();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec w_;   // <---- NECESSARY WEIGHTS VECTOR
};

} // namespace optimsolution
