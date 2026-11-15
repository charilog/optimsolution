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
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
