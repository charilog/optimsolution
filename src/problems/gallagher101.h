#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Gallagher’s Gaussian 101-peaks function (BBOB f21), faithful non-rotated COCO form.
 *
 * Continuous, multimodal, non-separable, highly irregular.
 * Domain: [-5, 5]^D.
 * Global optimum: f = 0 at the best peak centre y₁.
 */
class Gallagher101 : public Problem {
public:
    Gallagher101();

    void init(int dim) override;

    void setSeed(unsigned s) { seed_ = s; }

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int      D_ = 0;
    int      K_ = 101;
    unsigned seed_ = 42;

    std::vector<Vec>    Y_;       // peak centres
    std::vector<double> w_;       // weights
    std::vector<Vec>    diagCi_;  // diagonal of each Ci
    std::vector<double> R_;       // rotation matrix (row-major)

private:
    void build_instance();
    void make_rotation(std::vector<double>& M);

    static double f_pen(const Vec& x);
    static double tosz_scalar(double x);
};

} // namespace optimsolution
