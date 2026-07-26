#include "zdt3.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace {
constexpr double kPi = 3.1415926535897932384626433832795029;

double zdt3_g(const Vec& x) {
    const int D = static_cast<int>(x.size());
    if (D <= 1) return 1.0;
    double s = 0.0;
    for (int i = 1; i < D; ++i) s += x[i];
    return 1.0 + 9.0 * s / double(D - 1);
}
} // namespace

ZDT3::ZDT3() {
    setName("zdt3");
    setFullName("ZDT3 (Zitzler-Deb-Thiele 3) bi-objective benchmark (disconnected front)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("multi-objective benchmark");
}

void ZDT3::init(int dim) {
    if (dim < 2) dim = 2;
    Problem::init(dim);
    Vec lo(dim, 0.0), hi(dim, 1.0);
    setBounds(lo, hi);
}

Vec ZDT3::evaluateMultiCore(const Vec& x) {
    const double f1 = x.empty() ? 0.0 : x[0];
    const double g  = zdt3_g(x);
    const double ratio = (g > 0.0) ? std::max(0.0, f1 / g) : 0.0;
    const double h = 1.0 - std::sqrt(ratio) - ratio * std::sin(10.0 * kPi * f1);
    const double f2 = g * h;
    return { f1, f2 };
}

double ZDT3::evaluate_core(const Vec& x) {
    const Vec f = evaluateMultiCore(x);
    return f[0] + f[1];
}

void ZDT3::gradient_core(const Vec& x, Vec& g) {
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
