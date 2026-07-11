#include "matyas.h"

namespace optimsolution {

Matyas::Matyas()
{
    setName("matyas");
    setFullName("Matyas function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Matyas::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -10.0), hi(2, 10.0);
    setBounds(lo, hi);

    Vec xopt = {0.0, 0.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double Matyas::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    return 0.26 * (x1 * x1 + x2 * x2) - 0.48 * x1 * x2;
}

void Matyas::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double x1 = x[0];
    const double x2 = x[1];

    g[0] = 0.52 * x1 - 0.48 * x2;
    g[1] = 0.52 * x2 - 0.48 * x1;
}

} // namespace optimsolution
