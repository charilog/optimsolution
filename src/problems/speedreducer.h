#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Speed Reducer (gearbox) Design (classic constrained engineering benchmark,
 * a.k.a. "Reducer Design").
 *
 * Decision vector x (D=7): x0=face width, x1=module (tooth die), x2=number
 * of teeth on pinion, x3=length of first shaft between bearings, x4=length
 * of second shaft between bearings, x5=diameter of first shaft, x6=diameter
 * of second shaft.
 *
 * f(x) = 0.7854*x0*x1^2*(3.3333*x2^2+14.9334*x2-43.0934)
 *        - 1.508*x0*(x5^2+x6^2) + 7.4777*(x5^3+x6^3)
 *        + 0.7854*(x3*x5^2+x4*x6^2)
 *
 * Subject to 11 inequality constraints g_i(x) <= 0 (see .cpp for the exact
 * formulas -- Peng et al., Mathematics 2024, 12, 2153, Eq. (16)).
 *
 * Domain: 2.6<=x0<=3.6, 0.7<=x1<=0.8, 17<=x2<=28, 7.3<=x3<=8.3,
 *         7.3<=x4<=8.3, 2.9<=x5<=3.9, 5.0<=x6<=5.5.
 * Known best (literature): f* ~= 2994.47 at
 *   x* ~= (3.5, 0.7, 17, 7.3, 7.7153, 3.3505, 5.2867).
 *
 * NOTE: see weldedbeam.h for a discussion of an apparent objective-value
 * swap between this problem and Welded Beam Design in the source paper's
 * result tables; the formula implemented here matches Eq. (16) exactly and
 * reproduces the long-established literature optimum above.
 */
class SpeedReducer : public Problem {
public:
    SpeedReducer();
    void init(int dim) override;                  // force D=7, set bounds

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
