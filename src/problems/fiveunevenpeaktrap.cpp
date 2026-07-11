#include "fiveunevenpeaktrap.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace {
    // Returns (F1(x), dF1/dx) for the CEC2013 Five-Uneven-Peak Trap piece
    // that x falls into.
    std::pair<double, double> peakTrap(double x) {
        if (x < 2.5)  return { 80.0 * (2.5 - x),   -80.0 };
        if (x < 5.0)  return { 64.0 * (x - 2.5),    64.0 };
        if (x < 7.5)  return { 64.0 * (7.5 - x),   -64.0 };
        if (x < 12.5) return { 28.0 * (x - 7.5),    28.0 };
        if (x < 17.5) return { 28.0 * (17.5 - x),  -28.0 };
        if (x < 22.5) return { 32.0 * (x - 17.5),   32.0 };
        if (x < 27.5) return { 32.0 * (27.5 - x),  -32.0 };
        return { 80.0 * (x - 27.5), 80.0 };
    }
}

FiveUnevenPeakTrap::FiveUnevenPeakTrap()
{
    setName("fiveunevenpeaktrap");
    setFullName("Five-Uneven-Peak Trap (CEC 2013 niching F1, inverted)");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("multimodal niching benchmark");
}

void FiveUnevenPeakTrap::init(int /*dim*/)
{
    Problem::init(1);

    Vec lo(1, 0.0), hi(1, 30.0);
    setBounds(lo, hi);

    // Two global optima exist (x=0 and x=30); we record one of them.
    Vec xopt = {0.0};
    setKnownGlobalOptimum(-200.0, xopt);
}

double FiveUnevenPeakTrap::evaluate_core(const Vec& x)
{
    const double xv = std::max(0.0, std::min(30.0, x[0]));
    const double F1 = peakTrap(xv).first;
    double f = -F1;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void FiveUnevenPeakTrap::gradient_core(const Vec& x, Vec& g)
{
    g.assign(1, 0.0);
    const double xv = std::max(0.0, std::min(30.0, x[0]));
    const double slope = peakTrap(xv).second;
    g[0] = -slope; // f = -F1(x)
}

} // namespace optimsolution
