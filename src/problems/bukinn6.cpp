#include "bukinn6.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

BukinN6::BukinN6()
{
    setName("bukinn6");
    setFullName("Bukin function N.6");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void BukinN6::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo = {-15.0, -3.0};
    Vec hi = {-5.0,   3.0};
    setBounds(lo, hi);

    Vec xopt = {-10.0, 1.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double BukinN6::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];

    double f = 100.0 * std::sqrt(std::fabs(y - 0.01 * xv * xv)) + 0.01 * std::fabs(xv + 10.0);
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void BukinN6::gradient_core(const Vec& x, Vec& g)
{
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
