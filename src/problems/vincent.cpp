#include "vincent.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Vincent::Vincent()
{
    setName("vincent");
    setFullName("Vincent function (CEC 2013 niching F7, inverted)");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("multimodal niching benchmark");
}

void Vincent::init(int dim)
{
    if (dim < 1) dim = 2;
    Problem::init(dim);

    Vec lo(dim, 0.25), hi(dim, 10.0);
    setBounds(lo, hi);

    // One of the 6^D global optima (others exist at other combinations):
    // sin(10*ln(x))=1  <=>  x = exp((pi/2 + 2*k*pi)/10); k=0 gives x~=1.1701.
    Vec xopt(dim, std::exp((3.141592653589793238462643383279502884 / 2.0) / 10.0));
    setKnownGlobalOptimum(-1.0, xopt);
}

double Vincent::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = std::max(1e-6, x[i]);
        sum += std::sin(10.0 * std::log(xi));
    }
    double f = -sum / static_cast<double>(D);
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

// f = -(1/D)*Sum_i sin(10*ln(x_i))
// df/dx_k = -(10/D) * cos(10*ln(x_k)) / x_k
void Vincent::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i) {
        const double xi = std::max(1e-6, x[i]);
        g[i] = -(10.0 / static_cast<double>(D)) * std::cos(10.0 * std::log(xi)) / xi;
    }
}

} // namespace optimsolution
