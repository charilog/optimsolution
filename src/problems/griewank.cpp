#include "griewank.h"
#include <cmath>
#include <vector>

namespace optimsolution {

Griewank::Griewank()
{
    setName("griewank");
    setFullName("Griewank function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");


    setKnownGlobalOptimum(0.0);
}

void Griewank::init(int dim)
{
    if (dim < 1) dim = 1;
    Problem::init(dim);

    // bounds Griewank
    Vec lo(dim, -600.0), hi(dim, 600.0);
    setBounds(lo, hi);

    // Global optimum: x* = 0, f* = 0
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

// f(x) = 1 + Σ_{i=1..D} x_i^2/4000 - Π_{i=1..D} cos(x_i / sqrt(i))
// global min: f(0,...,0)=0
double Griewank::evaluate_core(const Vec& x)
{
    const int D = dimension();

    double sum  = 0.0;
    double prod = 1.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x[i];
        sum  += (xi * xi) / 4000.0;
        prod *= std::cos(xi / std::sqrt(static_cast<double>(i + 1)));
    }

    double f = 1.0 + sum - prod;
    if (!std::isfinite(f))
        f = 1e12;
    return f;
}

// ∂f/∂x_j = x_j/2000 + (Π_{i≠j} cos(x_i/√i)) * (sin(x_j/√j)/√j)

void Griewank::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    if (D == 0)
        return;


    std::vector<double> u(D), c(D), s(D), pref(D), suff(D);

    for (int i = 0; i < D; ++i) {
        const double idx = static_cast<double>(i + 1);
        u[i] = x[i] / std::sqrt(idx);
        c[i] = std::cos(u[i]);
        s[i] = std::sin(u[i]);
    }

    // prefix products of cos
    double acc = 1.0;
    for (int i = 0; i < D; ++i) {
        pref[i] = acc;
        acc *= c[i];
    }

    // suffix products of cos
    acc = 1.0;
    for (int i = D - 1; i >= 0; --i) {
        suff[i] = acc;
        acc *= c[i];
    }

    for (int j = 0; j < D; ++j) {
        const double prod_except_j = pref[j] * suff[j]; // Π_{i≠j} cos(u_i)
        const double jroot         = std::sqrt(static_cast<double>(j + 1));

        const double dSum  = x[j] / 2000.0;
        const double dProd = prod_except_j * (s[j] / jroot); 

        g[j] = dSum + dProd;
    }
}

} // namespace optimsolution
