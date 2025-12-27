#include "schwefel.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Schwefel::Schwefel()
{
    setName("schwefel");
    setFullName("Schwefel 2.26 function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

}

void Schwefel::init(int dim) {
    if (dim < 1) dim = 1;
    Problem::init(dim);

    Vec lo(dim, -500.0), hi(dim, 500.0);
    setBounds(lo, hi);

    // Global optimum x_i ≈ 420.968746..., f* ≈ 0
    Vec xopt(dim, 420.9687462275036);
    setKnownGlobalOptimum(0.0, xopt);
}

double Schwefel::evaluate_core(const Vec& x) {
    const int n = static_cast<int>(x.size());
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        sum += xi * std::sin(std::sqrt(std::fabs(xi)));
    }
    double f = 418.9829 * n - sum;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Schwefel::gradient_core(const Vec& x, Vec& g) {
    // Forward finite differences 
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
