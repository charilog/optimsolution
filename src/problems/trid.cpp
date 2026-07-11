#include "trid.h"

namespace optimsolution {

Trid::Trid()
{
    setName("trid");
    setFullName("Trid function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Trid::init(int dim)
{
    if (dim < 1) dim = 2;
    Problem::init(dim);

    const double b = static_cast<double>(dim) * static_cast<double>(dim);
    Vec lo(dim, -b), hi(dim, b);
    setBounds(lo, hi);

    // x*_i = i*(n+1-i)   (1-indexed i)
    Vec xopt(dim, 0.0);
    for (int i = 1; i <= dim; ++i) {
        xopt[i - 1] = static_cast<double>(i) * static_cast<double>(dim + 1 - i);
    }
    const double n = static_cast<double>(dim);
    const double fopt = -n * (n + 4.0) * (n - 1.0) / 6.0;
    setKnownGlobalOptimum(fopt, xopt);
}

double Trid::evaluate_core(const Vec& x)
{
    const int D = dimension();

    double sum1 = 0.0;
    for (int i = 0; i < D; ++i) {
        const double t = x[i] - 1.0;
        sum1 += t * t;
    }

    double sum2 = 0.0;
    for (int i = 1; i < D; ++i) {
        sum2 += x[i] * x[i - 1];
    }

    return sum1 - sum2;
}

void Trid::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int j = 0; j < D; ++j) {
        double gj = 2.0 * (x[j] - 1.0);
        if (j >= 1)     gj -= x[j - 1];
        if (j <= D - 2) gj -= x[j + 1];
        g[j] = gj;
    }
}

} // namespace optimsolution
