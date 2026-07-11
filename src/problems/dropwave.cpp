#include "dropwave.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

DropWave::DropWave()
{
    setName("dropwave");
    setFullName("Drop-Wave function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-1.0);
}

void DropWave::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -5.12), hi(2, 5.12);
    setBounds(lo, hi);

    Vec xopt = {0.0, 0.0};
    setKnownGlobalOptimum(-1.0, xopt);
}

double DropWave::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];
    const double r  = std::sqrt(xv * xv + y * y);

    const double num = 1.0 + std::cos(12.0 * r);
    const double den = 0.5 * (xv * xv + y * y) + 2.0;

    double f = -num / den;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

// f = -u/v, u = 1+cos(12r), v = 0.5*(x^2+y^2)+2, r = sqrt(x^2+y^2)
// df/dx = x*[u + 12*v*sin(12r)/r] / v^2   (and symmetric for y)
void DropWave::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double xv = x[0];
    const double y  = x[1];
    const double r2 = xv * xv + y * y;
    const double r  = std::sqrt(r2);

    const double u = 1.0 + std::cos(12.0 * r);
    const double v = 0.5 * r2 + 2.0;

    // sin(12r)/r has a well-defined limit of 12 as r -> 0.
    const double sinc = (r < 1e-9) ? 12.0 : (std::sin(12.0 * r) / r);

    const double bracket = u + 12.0 * v * sinc;
    const double scale = bracket / (v * v);

    g[0] = xv * scale;
    g[1] = y  * scale;
}

} // namespace optimsolution
