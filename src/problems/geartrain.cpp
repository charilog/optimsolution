#include "geartrain.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

GearTrain::GearTrain()
{
    setName("geartrain");
    setFullName("Gear Train Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(2.7e-12);
}

void GearTrain::init(int /*dim*/) {
    Problem::init(4);

    Vec lo(4, 12.0), hi(4, 60.0);
    setBounds(lo, hi);

    Vec xopt = {19.0, 16.0, 43.0, 49.0};
    setKnownGlobalOptimum(2.7e-12, xopt);
}

double GearTrain::evaluate_core(const Vec& x) {
    const double x0 = clampd(x[0], 12.0, 60.0);
    const double x1 = clampd(x[1], 12.0, 60.0);
    const double x2 = clampd(x[2], 12.0, 60.0);
    const double x3 = clampd(x[3], 12.0, 60.0);

    const double u = 1.0 / 6.931 - (x0 * x1) / (x2 * x3);
    double f = u * u;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

// f(x) = u^2, u = 1/6.931 - x0*x1/(x2*x3)
void GearTrain::gradient_core(const Vec& x, Vec& g) {
    g.assign(4, 0.0);

    const double x0 = clampd(x[0], 12.0, 60.0);
    const double x1 = clampd(x[1], 12.0, 60.0);
    const double x2 = clampd(x[2], 12.0, 60.0);
    const double x3 = clampd(x[3], 12.0, 60.0);

    const double u = 1.0 / 6.931 - (x0 * x1) / (x2 * x3);
    const double twou = 2.0 * u;

    g[0] = twou * (-x1 / (x2 * x3));
    g[1] = twou * (-x0 / (x2 * x3));
    g[2] = twou * ((x0 * x1) / (x2 * x2 * x3));
    g[3] = twou * ((x0 * x1) / (x2 * x3 * x3));
}

} // namespace optimsolution
