#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Weierstrass benchmark function (continuous, multimodal, non-differentiable).
 *
 * f(x) = sum_{i=1}^D [ sum_{k=0}^{kmax} a^k cos( 2π b^k (x_i + 0.5) ) ]
 *        - D * sum_{k=0}^{kmax} a^k cos( 2π b^k * 0.5 )
 *
 * Typical parameters: a=0.5, b=3, kmax=20,
 * Domain: x_i in [-0.5, 0.5].
 * Global minimum: f(x*) = 0 at x* = (0, ..., 0).
 */
class Weierstrass : public Problem {
public:
    Weierstrass();

    void init(int dim) override;                   // set dimension & bounds

protected:
    double evaluate_core(const Vec& x) override;   // objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric gradient

private:
    int    kmax_;      // default 20
    double a_;         // default 0.5  (0<a<1)
    double b_;         // default 3.0

    double constant_term() const;  // C = sum_{k=0}^{kmax} a^k cos(2π b^k * 0.5)

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
