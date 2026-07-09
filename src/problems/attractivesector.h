#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Attractive Sector function (BBOB F6), faithful implementation.
 *
 * f(x) = T_osz( sum_{i=1}^{D} (s_i * z_i)^2 )^0.9
 *
 * where z = Lambda^10 * R * x  (x_opt = 0, so the optimum sits at the
 * origin exactly as documented), Lambda^10 is a diagonal matrix with
 * condition number 10 (Lambda_ii = 10^(0.5*i/(D-1))), R is a random
 * orthogonal rotation matrix (fixed per instance via seed_), and
 * s_i = 100 if z_i > 0, else 1 — the asymmetric "attractive sector"
 * that gives the function its name: the landscape is far steeper in the
 * sector where every z_i > 0 than elsewhere, which is what a search
 * algorithm needs to detect and avoid overshooting.
 *
 * Domain: [-100, 100]^D. Global optimum: f = 0 at x = 0.
 *
 * NOTE: the previous version of this file computed
 *   f(x) = sum_i |x_i|^(2 + sgn(x_i))
 * a purely separable, unrotated, unconditioned placeholder formula that
 * does not match the BBOB F6 definition at all (its own file comments
 * admitted this was a guess) and directly contradicted the class's own
 * setSeparability("non-separable") metadata. This version replaces it
 * with the real, documented function.
 */
class AttractiveSector : public Problem {
public:
    AttractiveSector();
    void init(int dim) override;

    void setSeed(unsigned s) { seed_ = s; }

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int      D_ = 0;
    unsigned seed_ = 42;

    std::vector<double> R_;     // D x D rotation matrix, row-major
    std::vector<double> diag_;  // D, condition-10 diagonal scaling

    void build_instance();
    void make_rotation(std::vector<double>& M);

    // z = Lambda^10 * R * x, returned by value
    Vec rotatedScaled(const Vec& x) const;

    static double tosz_scalar(double x);
};

} // namespace optimsolution
