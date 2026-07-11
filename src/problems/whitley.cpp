#include "whitley.h"
#include <cmath>

namespace optimsolution {

Whitley::Whitley()
{
    setName("whitley");
    setFullName("Whitley function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Whitley::init(int dim)
{
    if (dim < 1) dim = 2;
    Problem::init(dim);

    Vec lo(dim, -10.24), hi(dim, 10.24);
    setBounds(lo, hi);

    Vec xopt(dim, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

// y_ij = 100*(x_i^2 - x_j)^2 + (1 - x_j)^2
// f = Sum_i Sum_j [ y_ij^2/4000 - cos(y_ij) + 1 ]
double Whitley::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double f = 0.0;

    for (int i = 0; i < D; ++i) {
        const double xi2 = x[i] * x[i];
        for (int j = 0; j < D; ++j) {
            const double a = xi2 - x[j];
            const double b = 1.0 - x[j];
            const double y = 100.0 * a * a + b * b;
            f += (y * y) / 4000.0 - std::cos(y) + 1.0;
        }
    }
    return f;
}

// d(term_ij)/dy = y/2000 + sin(y)
// dy_ij/dx_i = 400 * x_i * (x_i^2 - x_j)
// dy_ij/dx_j = -200*(x_i^2 - x_j) - 2*(1 - x_j)
//
// x_k contributes to the sum both as the "i" role (fixed i=k, all j) and the
// "j" role (fixed j=k, all i); both contributions are accumulated below.
void Whitley::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i) {
        const double xi  = x[i];
        const double xi2 = xi * xi;
        for (int j = 0; j < D; ++j) {
            const double xj = x[j];
            const double a  = xi2 - xj;      // (x_i^2 - x_j)
            const double b  = 1.0 - xj;
            const double y  = 100.0 * a * a + b * b;
            const double dterm_dy = y / 2000.0 + std::sin(y);

            const double dy_dxi = 400.0 * xi * a;
            const double dy_dxj = -200.0 * a - 2.0 * b;

            g[i] += dterm_dy * dy_dxi; // role: x_i
            g[j] += dterm_dy * dy_dxj; // role: x_j
        }
    }
}

} // namespace optimsolution
