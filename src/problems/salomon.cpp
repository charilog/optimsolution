#include "salomon.h"
#include <cmath>

namespace optimsolution {

namespace { constexpr double PI = 3.141592653589793238462643383279502884; }

Salomon::Salomon()
{
    setName("salomon");
    setFullName("Salomon function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Salomon::init(int dim)
{
    if (dim < 1) dim = 1;
    Problem::init(dim);

    Vec lo(dim, -100.0), hi(dim, 100.0);
    setBounds(lo, hi);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Salomon::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double s = 0.0;
    for (int i = 0; i < D; ++i) s += x[i] * x[i];
    const double r = std::sqrt(s);

    return 1.0 - std::cos(2.0 * PI * r) + 0.1 * r;
}

void Salomon::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    double s = 0.0;
    for (int i = 0; i < D; ++i) s += x[i] * x[i];
    const double r = std::sqrt(s);

    if (r < 1e-12) {
        // f is smooth-ish (via r=|.|) but has a cusp at r=0; the sub-gradient
        // there is 0 by symmetry (all directions from the origin are
        // equivalent for this radially symmetric function).
        return;
    }

    // df/dr = 2*pi*sin(2*pi*r) + 0.1 ; dr/dx_i = x_i / r
    const double dfdr = 2.0 * PI * std::sin(2.0 * PI * r) + 0.1;
    const double scale = dfdr / r;

    for (int i = 0; i < D; ++i) {
        g[i] = scale * x[i];
    }
}

} // namespace optimsolution
