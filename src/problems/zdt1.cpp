#include "zdt1.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace {
double zdt1_g(const Vec& x) {
    const int D = static_cast<int>(x.size());
    if (D <= 1) return 1.0;
    double s = 0.0;
    for (int i = 1; i < D; ++i) s += x[i];
    return 1.0 + 9.0 * s / double(D - 1);
}
} // namespace

ZDT1::ZDT1() {
    setName("zdt1");
    setFullName("ZDT1 (Zitzler-Deb-Thiele 1) bi-objective benchmark");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("multi-objective benchmark");
}

void ZDT1::init(int dim) {
    if (dim < 2) dim = 2;
    // Literature default is D=30; keep that as the fallback when no explicit
    // dimension is requested, but honor whatever the caller asks for.
    Problem::init(dim);
    Vec lo(dim, 0.0), hi(dim, 1.0);
    setBounds(lo, hi);
}

Vec ZDT1::evaluateMultiCore(const Vec& x) {
    const double f1 = x.empty() ? 0.0 : x[0];
    const double g  = zdt1_g(x);
    const double ratio = (g > 0.0) ? std::max(0.0, f1 / g) : 0.0;
    const double f2 = g * (1.0 - std::sqrt(ratio));
    return { f1, f2 };
}

double ZDT1::evaluate_core(const Vec& x) {
    const Vec f = evaluateMultiCore(x);
    return f[0] + f[1];
}

void ZDT1::gradient_core(const Vec& x, Vec& g) {
    const int D = static_cast<int>(x.size());
    g.assign(D, 0.0);
    const double eps = 1.0e-6;
    Vec xp = x, xm = x;
    for (int i = 0; i < D; ++i) {
        xp[i] = x[i] + eps;
        xm[i] = x[i] - eps;
        const double fp = evaluate_core(xp);
        const double fm = evaluate_core(xm);
        g[i] = (fp - fm) / (2.0 * eps);
        xp[i] = x[i];
        xm[i] = x[i];
    }
}

} // namespace optimsolution
