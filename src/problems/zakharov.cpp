#include "zakharov.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Zakharov::Zakharov()
{
    setName("zakharov");
    setFullName("Zakharov function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

}

void Zakharov::init(int dim) {
    const int D = (dim >= 1 ? dim : 10);  // default D=10 if caller passes non-positive
    Problem::init(D);

    Vec lo(D, -5.0), hi(D, 10.0);
    setBounds(lo, hi);

    // Global minimum at x* = 0, f* = 0
    Vec xopt(D, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Zakharov::evaluate_core(const Vec& x) {
    const int D = (int)x.size();

    double s1 = 0.0;   // sum x_i^2
    double sA = 0.0;   // sum (i/2) x_i
    for (int i = 0; i < D; ++i) {
        const double xi = clampd(x[i], -5.0, 10.0);     // keep within standard box
        s1 += xi * xi;
        sA += 0.5 * (i + 1) * xi;
    }

    const double f = s1 + sA * sA + std::pow(sA, 4.0);
    if (!std::isfinite(f)) return 1e12;
    return f;
}

void Zakharov::gradient_core(const Vec& x, Vec& g) {
    const int D = (int)x.size();
    g.assign(D, 0.0);

    // Precompute sA = sum (i/2) x_i (with clamped xi to be consistent with evaluate_core)
    double sA = 0.0;
    std::vector<double> xc(D);
    for (int i = 0; i < D; ++i) {
        xc[i] = clampd(x[i], -5.0, 10.0);
        sA += 0.5 * (i + 1) * xc[i];
    }

    // df/dx_j = 2 x_j + (2 sA + 4 sA^3) * (j/2)
    const double coeff = (2.0 * sA) + (4.0 * std::pow(sA, 3.0));
    for (int j = 0; j < D; ++j) {
        const double wj = 0.5 * (j + 1);
        g[j] = 2.0 * xc[j] + coeff * wj;
    }
}

} // namespace optimsolution
