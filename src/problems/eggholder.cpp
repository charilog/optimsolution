#include "eggholder.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Eggholder::Eggholder()
{
    setName("eggholder");
    setFullName("Eggholder function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-959.6407);
}

void Eggholder::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, -512.0), hi(2, 512.0);
    setBounds(lo, hi);

    Vec xopt = {512.0, 404.2319};
    setKnownGlobalOptimum(-959.6407, xopt);
}

double Eggholder::evaluate_core(const Vec& x)
{
    const double xv = x[0];
    const double y  = x[1];

    const double t1 = std::fabs(xv / 2.0 + (y + 47.0));
    const double t2 = std::fabs(xv - (y + 47.0));

    double f = -(y + 47.0) * std::sin(std::sqrt(t1)) - xv * std::sin(std::sqrt(t2));
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Eggholder::gradient_core(const Vec& x, Vec& g)
{
    // Non-smooth (|.| inside sqrt); forward differences, consistent with
    // other non-smooth problems in this framework (e.g. Alpine1, Katsuura).
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
