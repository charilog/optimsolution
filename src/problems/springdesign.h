#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Tension/Compression Spring Design (classic constrained engineering
 * benchmark).
 *
 * Decision vector x (D=3): x0=d (wire diameter), x1=D (mean coil diameter),
 * x2=N (number of active coils).
 *
 * f(x) = (x2+2)*x1*x0^2
 *
 * Constraints g_i(x) <= 0:
 *   g1 = 1 - x1^3*x2 / (71785*x0^4)
 *   g2 = (4*x1^2 - x0*x1)/(12566*(x1*x0^3 - x0^4)) + 1/(5108*x0^2) - 1
 *   g3 = 1 - 140.45*x0/(x1^2*x2)
 *   g4 = (x0+x1)/1.5 - 1
 *
 * Domain: 0.05<=x0<=2.0, 0.25<=x1<=1.3, 2.0<=x2<=15.0.
 * Known best (literature): f* ~= 0.012665 at
 *   x* ~= (0.05169, 0.35673, 11.28896).
 */
class SpringDesign : public Problem {
public:
    SpringDesign();
    void init(int dim) override;                  // force D=3, set bounds

protected:
    double evaluate_core(const Vec& x) override;  // penalized objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    double w_pen_;

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
