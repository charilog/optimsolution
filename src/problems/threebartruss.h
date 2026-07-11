#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Three-Bar Truss Design (classic constrained engineering benchmark).
 *
 * A symmetric truss with bars 1 and 3 sharing cross-sectional area x0 and
 * bar 2 having area x1.
 *
 * f(x) = l*(2*sqrt(2)*x0 + x1)
 *
 * Constraints g_i(x) <= 0 (with l=100 cm, P=2 kN/cm^2, sigma=2 kN/cm^2):
 *   g1 = (sqrt(2)*x0+x1)/(sqrt(2)*x0^2+2*x0*x1) * P - sigma
 *   g2 = x1/(sqrt(2)*x0^2+2*x0*x1) * P - sigma
 *   g3 = 1/(x0+sqrt(2)*x1) * P - sigma
 *
 * Domain: 1e-4<=x0,x1<=1.
 * Known best (literature): f* ~= 263.8958 at x* ~= (0.78867, 0.40902).
 */
class ThreeBarTruss : public Problem {
public:
    ThreeBarTruss();
    void init(int dim) override;                  // force D=2, set bounds

protected:
    double evaluate_core(const Vec& x) override;  // penalized objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    double l_, P_, sigma_;
    double w_pen_;

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
