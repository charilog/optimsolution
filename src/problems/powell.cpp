#include "powell.h"
#include <cmath>

namespace optimsolution {

Powell::Powell()
{
    setName("powell");
    setFullName("Powell singular function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Powell::init(int dim)
{
    if (dim < 4) dim = 4;
    dim = ((dim + 3) / 4) * 4; // round up to nearest multiple of 4
    Problem::init(dim);

    Vec lo(dim, -4.0), hi(dim, 5.0);
    setBounds(lo, hi);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Powell::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double f = 0.0;

    for (int base = 0; base + 3 < D; base += 4) {
        const double a = x[base + 0];
        const double b = x[base + 1];
        const double c = x[base + 2];
        const double d = x[base + 3];

        const double t1 = a + 10.0 * b;
        const double t2 = c - d;
        const double t3 = b - 2.0 * c;
        const double t4 = a - d;

        f += t1 * t1 + 5.0 * t2 * t2
           + t3 * t3 * t3 * t3
           + 10.0 * t4 * t4 * t4 * t4;
    }
    return f;
}

void Powell::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int base = 0; base + 3 < D; base += 4) {
        const double a = x[base + 0];
        const double b = x[base + 1];
        const double c = x[base + 2];
        const double d = x[base + 3];

        const double t1 = a + 10.0 * b;
        const double t2 = c - d;
        const double t3 = b - 2.0 * c;
        const double t3cube = t3 * t3 * t3;
        const double t4 = a - d;
        const double t4cube = t4 * t4 * t4;

        // d/da = 2*t1 + 40*t4^3
        g[base + 0] = 2.0 * t1 + 40.0 * t4cube;
        // d/db = 20*t1 + 4*t3^3
        g[base + 1] = 20.0 * t1 + 4.0 * t3cube;
        // d/dc = 10*t2 - 8*t3^3
        g[base + 2] = 10.0 * t2 - 8.0 * t3cube;
        // d/dd = -10*t2 - 40*t4^3
        g[base + 3] = -10.0 * t2 - 40.0 * t4cube;
    }
}

} // namespace optimsolution
