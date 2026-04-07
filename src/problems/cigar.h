#pragma once
#include "problem.h"

namespace optimsolution {

class Cigar : public Problem {
public:
    Cigar();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;
};

} // namespace optimsolution
