#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Pressure Vessel Design (classic constrained engineering benchmark).
 *
 * Decision vector x (D=4): x0=Ts (shell thickness), x1=Th (head thickness),
 * x2=R (inner radius), x3=L (length of cylindrical section).
 * (Continuous relaxation: the classic version restricts x0,x1 to multiples
 * of 0.0625 in.; that discreteness is not enforced here.)
 *
 * f(x) = 0.6224*x0*x2*x3 + 1.7781*x1*x2^2 + 3.1661*x0^2*x3 + 19.84*x0^2*x2
 *
 * Constraints g_i(x) <= 0:
 *   g1 = -x0 + 0.0193*x2
 *   g2 = -x1 + 0.00954*x2
 *   g3 = -pi*x2^2*x3 - (4/3)*pi*x2^3 + 1296000
 *   g4 = x3 - 240
 *
 * Domain: 0<=x0<=99, 0<=x1<=99, 10<=x2<=200, 10<=x3<=200.
 * Known best (literature): f* ~= 6059.71 at
 *   x* ~= (0.8125, 0.4375, 42.0984, 176.6366).
 */
class PressureVessel : public Problem {
public:
    PressureVessel();
    void init(int dim) override;                  // force D=4, set bounds

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
