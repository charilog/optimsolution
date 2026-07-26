#include "zdt4.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace {
constexpr double kPi = 3.1415926535897932384626433832795029;

double zdt4_g(const Vec& x) {
    const int D = static_cast<int>(x.size());
    double s = 0.0;
    for (int i = 1; i < D; ++i) {
        s += x[i] * x[i] - 10.0 * std::cos(4.0 * kPi * x[i]);
    }
    return 1.0 + 10.0 * double(D - 1) + s;
}
} // namespace

ZDT4::ZDT4() {
    setName("zdt4");
    setFullName("ZDT4 (Zitzler-Deb-Thiele 4) bi-objective benchmark (many local fronts)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("multi-objective benchmark");
}

void ZDT4::init(int dim) {
    if (dim < 2) dim = 2;
    Problem::init(dim);
    // Non-uniform bounds: x_1 in [0,1], x_i in [-5,5] for i = 2..D.
    Vec lo(dim, -5.0), hi(dim, 5.0);
    lo[0] = 0.0;
    hi[0] = 1.0;
    setBounds(lo, hi);
}

Vec ZDT4::evaluateMultiCore(const Vec& x) {
    const double f1 = x.empty() ? 0.0 : x[0];
    const double g  = zdt4_g(x);
    const double ratio = (g > 0.0) ? std::max(0.0, f1 / g) : 0.0;
    const double f2 = g * (1.0 - std::sqrt(ratio));
    return { f1, f2 };
}

double ZDT4::evaluate_core(const Vec& x) {
    const Vec f = evaluateMultiCore(x);
    return f[0] + f[1];
}

void ZDT4::gradient_core(const Vec& x, Vec& g) {
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
