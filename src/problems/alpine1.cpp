#include "alpine1.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Alpine1::Alpine1()
{
    setName("alpine1");
    setFullName("Alpine N.1 function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");
}

void Alpine1::init(int dim)
{
    if (dim < 1) dim = 1;
    Problem::init(dim);

    Vec lo(dim, -10.0), hi(dim, 10.0);
    setBounds(lo, hi);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Alpine1::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double f = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x[i];
        f += std::fabs(xi * std::sin(xi) + 0.1 * xi);
    }
    return f;
}

void Alpine1::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences: f contains |.|, non-differentiable at
    // isolated points (including x_i=0), so a closed-form subgradient would
    // need special-casing there; forward differences give a usable direction
    // everywhere else, consistent with other non-smooth problems in this
    // framework (e.g. Katsuura, Polyphase).
    const int D = dimension();
    g.assign(D, 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < D; ++i) {
        const double h = std::max(1e-6, std::fabs(x[i]) * 1e-6);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
