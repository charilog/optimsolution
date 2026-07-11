#include "dixonprice.h"
#include <cmath>

namespace optimsolution {

DixonPrice::DixonPrice()
{
    setName("dixonprice");
    setFullName("Dixon-Price function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void DixonPrice::init(int dim)
{
    if (dim < 1) dim = 2;
    Problem::init(dim);

    Vec lo(dim, -10.0), hi(dim, 10.0);
    setBounds(lo, hi);

    // x*_i = 2^( -(2^i - 2) / 2^i ),  i = 1..D (1-indexed)
    Vec xopt(dim, 0.0);
    for (int i = 1; i <= dim; ++i) {
        const double p2i = std::pow(2.0, i);
        xopt[i - 1] = std::pow(2.0, -(p2i - 2.0) / p2i);
    }
    setKnownGlobalOptimum(0.0, xopt);
}

// f(x) = (x0-1)^2 + Sum_{k=1}^{D-1} (k+1)*(2*x_k^2 - x_{k-1})^2   [0-indexed]
double DixonPrice::evaluate_core(const Vec& x)
{
    const int D = dimension();
    const double t0 = x[0] - 1.0;
    double f = t0 * t0;

    for (int k = 1; k < D; ++k) {
        const double term = 2.0 * x[k] * x[k] - x[k - 1];
        f += static_cast<double>(k + 1) * term * term;
    }
    return f;
}

void DixonPrice::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    // df/dx0 = 2*(x0-1) - 4*(2*x1^2 - x0)     [only if D >= 2]
    g[0] = 2.0 * (x[0] - 1.0);
    if (D >= 2) {
        const double term1 = 2.0 * x[1] * x[1] - x[0];
        g[0] += -4.0 * term1;
    }

    // for 1 <= k <= D-2 (middle indices):
    // df/dxk = 8*(k+1)*xk*(2*xk^2 - x_{k-1}) - 2*(k+2)*(2*x_{k+1}^2 - xk)
    for (int k = 1; k <= D - 2; ++k) {
        const double termk  = 2.0 * x[k] * x[k] - x[k - 1];
        const double termk1 = 2.0 * x[k + 1] * x[k + 1] - x[k];
        g[k] = 8.0 * static_cast<double>(k + 1) * x[k] * termk
             - 2.0 * static_cast<double>(k + 2) * termk1;
    }

    // last index k = D-1 (if D >= 2): only its own term contributes
    if (D >= 2) {
        const int k = D - 1;
        const double termk = 2.0 * x[k] * x[k] - x[k - 1];
        g[k] = 8.0 * static_cast<double>(k + 1) * x[k] * termk;
    }
}

} // namespace optimsolution
