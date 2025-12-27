#pragma once
#include "problem.h"

namespace optimsolution {

class Hartmann6 : public Problem {
public:
    Hartmann6();
    void init(int dim) override;  

protected:
    double evaluate_core(const Vec& x) override;  // f(x) = -Σ c_i exp( -Σ a_ij (x_j - p_ij)^2 )
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
