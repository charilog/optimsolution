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
    // Non-smooth (sqrt of |.|, plus |.|); forward differences.
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
