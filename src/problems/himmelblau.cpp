#include "himmelblau.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Himmelblau::Himmelblau()
{
    setName("himmelblau");
    setFullName("Himmelblau (maximize-style variant)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

  
}

void Himmelblau::init(int /*dim*/)
{
    // Faithful: treat it as 2-D (the original uses x[0], x[1])
    Problem::init(2);

    Vec lo(2, -6.0), hi(2, 6.0);
    setBounds(lo, hi);  // same bounds as reference
}

double Himmelblau::evaluate_core(const Vec& x)
{
    // f(x) = 200 - (x1^2 + x2 - 11)^2 - (x1 + x2^2 - 7)^2
    const double x1 = x[0];
    const double x2 = x[1];

    const double t1 = (x1 * x1 + x2 - 11.0);
    const double t2 = (x1 + x2 * x2 - 7.0);
    const double f  = 200.0 - t1 * t1 - t2 * t2;

    // keep behavior stable if NaN/Inf
    if (!(f >= -1e300) || std::isnan(f) || std::isinf(f))
        return 1e30;

    return f;
}

void Himmelblau::gradient_core(const Vec& x, Vec& g)
{
    // Central differences with eps = (1e-18)^(1/3) * max(1, |x_i|)
    const int n = static_cast<int>(x.size());
    g.assign(n, 0.0);

    Vec xt = x;  // local copy

    for (int i = 0; i < n; ++i) {
        const double eps = std::cbrt(1e-18) * dmax(1.0, std::fabs(x[i]));

        xt[i] = x[i] + eps;
        const double f1 = evaluate_core(xt);

        xt[i] = x[i] - eps;
        const double f2 = evaluate_core(xt);

        g[i] = (f1 - f2) / (2.0 * eps);
        xt[i] = x[i];
    }
}

} // namespace optimsolution
