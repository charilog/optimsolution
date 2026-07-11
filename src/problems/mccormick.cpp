#include "mccormick.h"
#include <cmath>

namespace optimsolution {

McCormick::McCormick()
{
    setName("mccormick");
    setFullName("McCormick function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-1.913222954981037);
}

void McCormick::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo = {-1.5, -3.0};
    Vec hi = { 4.0,  4.0};
    setBounds(lo, hi);

    Vec xopt = {-0.54719, -1.54719};
    setKnownGlobalOptimum(-1.913222954981037, xopt);
}

double McCormick::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];
    const double d  = x1 - x2;

    return std::sin(x1 + x2) + d * d - 1.5 * x1 + 2.5 * x2 + 1.0;
}

void McCormick::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double x1 = x[0];
    const double x2 = x[1];
    const double c  = std::cos(x1 + x2);
    const double d  = x1 - x2;

    g[0] = c + 2.0 * d - 1.5;
    g[1] = c - 2.0 * d + 2.5;
}

} // namespace optimsolution
