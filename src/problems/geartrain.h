#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Gear Train Design (classic engineering benchmark).
 *
 * Decision vector x (D=4): number of teeth of 4 gears. The classic
 * benchmark restricts these to integers in [12,60]; that discreteness is
 * not enforced here (a continuous relaxation is used, consistent with how
 * this problem is commonly treated in continuous-optimizer benchmarking).
 *
 * f(x) = (1/6.931 - x0*x1/(x2*x3))^2
 *
 * No inequality constraints beyond the box bounds.
 *
 * Domain: 12<=xi<=60 for all i.
 * Known best (literature): f* ~= 2.7e-12 at x* = (19, 16, 43, 49)
 *   (or any permutation/scaling giving the same gear ratio).
 */
class GearTrain : public Problem {
public:
    GearTrain();
    void init(int dim) override;                  // force D=4, set bounds

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override; // analytic gradient

private:
    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
