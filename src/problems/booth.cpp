#include "booth.h"

namespace optimsolution {

Booth::Booth()
{
    setName("booth");
    setFullName("Booth function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Booth::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -10.0), hi(2, 10.0);
    setBounds(lo, hi);

    Vec xopt = {1.0, 3.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double Booth::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    const double t1 = x1 + 2.0 * x2 - 7.0;
    const double t2 = 2.0 * x1 + x2 - 5.0;

    return t1 * t1 + t2 * t2;
}

void Booth::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double x1 = x[0];
    const double x2 = x[1];

    const double t1 = x1 + 2.0 * x2 - 7.0;
    const double t2 = 2.0 * x1 + x2 - 5.0;

    // d/dx1 [t1^2 + t2^2] = 2*t1*1 + 2*t2*2
    g[0] = 2.0 * t1 + 4.0 * t2;
    // d/dx2 [t1^2 + t2^2] = 2*t1*2 + 2*t2*1
    g[1] = 4.0 * t1 + 2.0 * t2;
}

} // namespace optimsolution
