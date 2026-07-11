#include "holdertable.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double HT_PI = 3.141592653589793238462643383279502884; }

HolderTable::HolderTable()
{
    setName("holdertable");
    setFullName("Holder Table function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-19.2085);
}

void HolderTable::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -10.0), hi(2, 10.0);
    setBounds(lo, hi);

    Vec xopt = {8.05502, 9.66459};
    setKnownGlobalOptimum(-19.2085, xopt);
}

double HolderTable::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];
    const double r  = std::sqrt(xv * xv + y * y);

    const double inner = std::fabs(1.0 - r / HT_PI);
    double f = -std::fabs(std::sin(xv) * std::cos(y) * std::exp(inner));
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void HolderTable::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < 2; ++i) {
        const double h = std::max(1e-6, std::fabs(x[i]) * 1e-6);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
