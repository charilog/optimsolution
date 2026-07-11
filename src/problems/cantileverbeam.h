#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Cantilever Beam Design (classic constrained engineering benchmark).
 *
 * A cantilever beam built from 5 hollow-square segments of heights
 * x0..x4 (decreasing from the fixed support toward the free tip),
 * carrying a fixed tip load.
 *
 * f(x) = 0.0624*(x0+x1+x2+x3+x4)
 *
 * Constraint g(x) <= 0:
 *   g1 = 61/x0^3 + 37/x1^3 + 19/x2^3 + 7/x3^3 + 1/x4^3 - 1
 *
 * Domain: 0.01<=xi<=100 for all i.
 * Known best (literature): f* ~= 1.33996 at
 *   x* ~= (6.0160, 5.3092, 4.4950, 3.4967, 2.1526).
 */
class CantileverBeam : public Problem {
public:
    CantileverBeam();
    void init(int dim) override;                  // force D=5, set bounds

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
