#include "beale.h"

namespace optimsolution {

Beale::Beale()
{
    setName("beale");
    setFullName("Beale function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Beale::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -4.5), hi(2, 4.5);
    setBounds(lo, hi);

    Vec xopt = {3.0, 0.5};
    setKnownGlobalOptimum(0.0, xopt);
}

double Beale::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];

    const double y2 = y * y;
    const double y3 = y2 * y;

    const double A = 1.5   - xv + xv * y;
    const double B = 2.25  - xv + xv * y2;
    const double C = 2.625 - xv + xv * y3;

    return A * A + B * B + C * C;
}

void Beale::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double xv = x[0];
    const double y  = x[1];

    const double y2 = y * y;
    const double y3 = y2 * y;

    const double A = 1.5   - xv + xv * y;
    const double B = 2.25  - xv + xv * y2;
    const double C = 2.625 - xv + xv * y3;

    // dA/dx = -1 + y,     dA/dy = x
    // dB/dx = -1 + y^2,   dB/dy = 2*x*y
    // dC/dx = -1 + y^3,   dC/dy = 3*x*y^2
    const double dAdx = -1.0 + y;
    const double dAdy = xv;
    const double dBdx = -1.0 + y2;
    const double dBdy = 2.0 * xv * y;
    const double dCdx = -1.0 + y3;
    const double dCdy = 3.0 * xv * y2;

    g[0] = 2.0 * A * dAdx + 2.0 * B * dBdx + 2.0 * C * dCdx;
    g[1] = 2.0 * A * dAdy + 2.0 * B * dBdy + 2.0 * C * dCdy;
}

} // namespace optimsolution
