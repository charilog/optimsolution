#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Rotated Rosenbrock function:
 *
 *  y = R x   (R: orthogonal rotation matrix, default = identity)
 *
 *  f(y) = Σ_{i=1}^{D-1} [ 100 (y_{i+1} - y_i^2)^2 + (1 - y_i)^2 ]
 *
 * Domain: [-5, 10]^D
 */
class RotatedRosenbrock : public Problem {
public:
    RotatedRosenbrock();
    void init(int dim) override;


    void setRotation(const std::vector<Vec>& R);

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:

    std::vector<Vec> R_;


    void applyRotation(const Vec& x, Vec& y) const;


    void applyRotationT(const Vec& gy, Vec& gx) const;
};

} // namespace optimsolution
