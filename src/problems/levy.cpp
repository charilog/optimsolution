#include "levy.h"
#include <cmath>

namespace optimsolution {

Levy::Levy()
{
    setName("levy");
    setFullName("Levy N.13 function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Levy::init(int dim)
{
    if (dim < 1) dim = 1;
    Problem::init(dim);

    // clasic bounds for Levy N.13
    Vec lo(dim, -10.0), hi(dim, 10.0);
    setBounds(lo, hi);

    // Global optimum x* = (1,...,1), f* = 0
    Vec xopt(dim, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

// Levy N.13 (D dimensions)
// w_i = 1 + (x_i - 1)/4
// f(x) = sin^2(π w_1)
//      + Σ_{i=1}^{D-1} (w_i - 1)^2 * [1 + 10 sin^2(π w_i + 1)]
//      + (w_D - 1)^2 * [1 + sin^2(2π w_D)]
// global minimum: x* = (1,...,1), f(x*) = 0
double Levy::evaluate_core(const Vec& x)
{
    const int D = dimension();
    const double pi = 3.1415926535897932384626433832795;

    double f = 0.0;

    //  sin^2(pi * w1)
    const double w1 = 1.0 + (x[0] - 1.0) * 0.25;
    const double term0 = std::sin(pi * w1);
    f += term0 * term0;

    // mid sum
    for (int i = 0; i < D - 1; ++i) {
        const double wi   = 1.0 + (x[i] - 1.0) * 0.25;
        const double sarg = pi * wi + 1.0;
        const double si   = std::sin(sarg);
        const double wi_1 = wi - 1.0;
        f += (wi_1 * wi_1) * (1.0 + 10.0 * (si * si));
    }

    // last
    const double wD   = 1.0 + (x[D - 1] - 1.0) * 0.25;
    const double s2   = std::sin(2.0 * pi * wD);
    const double wD_1 = wD - 1.0;
    f += (wD_1 * wD_1) * (1.0 + s2 * s2);

    return f;
}

// Gradient:
// w = 1 + (x-1)/4,  dw/dx = 1/4
// d/dx1 [sin^2(pi w1)] = (pi/4) * sin(2 pi w1)
//
// for 1 <= i <= D-1:
//  T_i = (w_i-1)^2 * [1 + 10 sin^2(u)]
//  with u = pi w_i + 1
//
// for last:
//  T_D = (w_D-1)^2 * [1 + sin^2(v)], v = 2 pi w_D
void Levy::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    const double pi = 3.1415926535897932384626433832795;
    const double dw = 0.25; // dw/dx

    for (int j = 0; j < D; ++j) {
        const double wj = 1.0 + (x[j] - 1.0) * dw;
        double gj = 0.0;

        // for j=0
        if (j == 0) {
            // d/dx [sin^2(pi w1)] = sin(2 pi w1) * (pi * dw)
            gj += (pi * dw) * std::sin(2.0 * pi * wj);
        }

        // mid sum: for j = 0..D-2
        if (j <= D - 2) {
            const double A    = (wj - 1.0);
            const double A2   = A * A;
            const double u    = pi * wj + 1.0;
            const double sinu = std::sin(u);
            const double B    = 1.0 + 10.0 * (sinu * sinu);

            // dA/dx [of A^2] = 2*A * dw/dx = 2*A*dw
            const double dA   = 2.0 * A * dw;

            // dB/dx = 10 * 2 sin(u)cos(u) * du/dx
            //       = 20 sin(u)cos(u) * (pi*dw)
            //       = 10*pi*dw * sin(2u) = (5*pi/2) * sin(2u)  (dw=1/4 already folded in)
            const double dB   = (5.0 * pi / 2.0) * std::sin(2.0 * u);

            gj += dA * B + A2 * dB;
        }

        // last: only for j = D-1
        if (j == D - 1) {
            const double A   = (wj - 1.0);
            const double A2  = A * A;
            const double v   = 2.0 * pi * wj;
            const double sv  = std::sin(v);
            const double B   = 1.0 + sv * sv;

            const double dA  = 2.0 * A * dw;
            // d/dx [sin^2(v)] = sin(2v) * dv/dx, dv/dx = 2 pi * dw = pi/2 (dw=1/4 already folded in)
            const double dB  = (pi / 2.0) * std::sin(2.0 * v);

            gj += dA * B + A2 * dB;
        }

        g[j] = gj;
    }
}

} // namespace optimsolution
