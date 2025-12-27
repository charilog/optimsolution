#include "gascycle.h"
#include <algorithm>

namespace optimsolution {

GasCycle::GasCycle() {
    setName("gascycle");
    setFullName("Idealized gas cycle efficiency (Brayton-type)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("thermodynamics / real-world benchmark");
}

void GasCycle::init(int /*dim*/) {
    // Problem is intrinsically 4D
    const int D = 4;
    Problem::init(D);

    Vec lo(D), hi(D);
    lo[0] = 300.0;  hi[0] = 1500.0;  // T1
    lo[1] = 1200.0; hi[1] = 2000.0;  // T3
    lo[2] = 1.0;    hi[2] = 20.0;    // P1
    lo[3] = 1.0;    hi[3] = 20.0;    // P3

    setBounds(lo, hi);
}

double GasCycle::evaluate_core(const Vec& x) {
    const double T1 = x[0];
    const double T3 = x[1];
    const double P1 = x[2];
    const double P3 = x[3];

    // Hard box penalty, consistent with original code
    if (T1 < 300.0 || T1 > 1500.0 ||
        T3 < 1200.0 || T3 > 2000.0 ||
        P1 < 1.0    || P1 > 20.0   ||
        P3 < 1.0    || P3 > 20.0) {
        return 1e20;
    }

    const double gamma = 1.4;
    const double r = P3 / P1;
    const double eta = 1.0 - (1.0 / std::pow(r, (gamma - 1.0) / gamma)) * (T1 / T3);

    return -eta; // maximize η
}

void GasCycle::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    g.assign(D, 0.0);

    const double eps = 1e-6;
    const double f0  = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < D; ++i) {
        xt[i] = x[i] + eps;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / eps;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
