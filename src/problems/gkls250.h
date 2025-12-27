#pragma once
#include "problem.h"

namespace optimsolution {

class Gkls250 : public Problem {
public:
    Gkls250();               
    void init(int dim) override;  

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
