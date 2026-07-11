#include "crossintray.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double CIT_PI = 3.141592653589793238462643383279502884; }

CrossInTray::CrossInTray()
{
    setName("crossintray");
    setFullName("Cross-in-Tray function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-2.06261);
}

void CrossInTray::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -10.0), hi(2, 10.0);
    setBounds(lo, hi);

    Vec xopt = {1.3491, 1.3491};
    setKnownGlobalOptimum(-2.06261, xopt);
}

double CrossInTray::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];
    const double r  = std::sqrt(xv * xv + y * y);

    const double inner = std::fabs(100.0 - r / CIT_PI);
    const double term  = std::fabs(std::sin(xv) * std::sin(y) * std::exp(inner)) + 1.0;

    double f = -0.0001 * std::pow(term, 0.1);
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void CrossInTray::gradient_core(const Vec& x, Vec& g)
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
