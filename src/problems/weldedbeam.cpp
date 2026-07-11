#include "weldedbeam.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

WeldedBeam::WeldedBeam()
    : P_(6000.0), L_(14.0), E_(30.0e6), G_(12.0e6),
      tau_max_(13600.0), sigma_max_(30000.0), delta_max_(0.25),
      w_pen_(1.0e6)
{
    setName("weldedbeam");
    setFullName("Welded Beam Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(1.72485);
}

void WeldedBeam::init(int /*dim*/) {
    Problem::init(4);

    Vec lo = {0.1, 0.1, 0.1, 0.1};
    Vec hi = {2.0, 10.0, 10.0, 2.0};
    setBounds(lo, hi);

    Vec xopt = {0.2057, 3.4705, 9.0366, 0.2057};
    setKnownGlobalOptimum(1.72485, xopt);
}

double WeldedBeam::evaluate_core(const Vec& x) {
    const double h = clampd(x[0], 0.1, 2.0);
    const double l = clampd(x[1], 0.1, 10.0);
    const double t = clampd(x[2], 0.1, 10.0);
    const double b = clampd(x[3], 0.1, 2.0);

    const double cost = 1.10471 * h * h * l + 0.04811 * t * b * (14.0 + l);

    // Weld-shear stress tau(x)
    const double tau1 = P_ / (std::sqrt(2.0) * h * l);
    const double M    = P_ * (L_ + l / 2.0);
    const double R    = std::sqrt(l * l / 4.0 + ((h + t) / 2.0) * ((h + t) / 2.0));
    const double J    = 2.0 * std::sqrt(2.0) * h * l * (l * l / 4.0 + ((h + t) / 2.0) * ((h + t) / 2.0));
    const double tau2 = (J > 1e-12) ? (M * R / J) : 0.0;
    const double tau  = std::sqrt(tau1 * tau1 + 2.0 * tau1 * tau2 * (l / (2.0 * std::max(R, 1e-12))) + tau2 * tau2);

    // Bending stress sigma(x) -- standard sigma = M_bend*c/I = 6PL/(b t^2)
    const double sigma = (6.0 * P_ * L_) / (b * t * t);

    // Tip deflection delta(x) -- standard cantilever deflection PL^3/(3EI)
    // with I = b*t^3/12 gives delta = 4PL^3/(E*b*t^3). (The paper's Eq.(17)
    // prints this with a leading "6" rather than "4"; 4 matches the standard
    // cantilever-beam formula and the widely reproduced literature version
    // of this benchmark, so it is used here.)
    const double delta = (4.0 * P_ * L_ * L_ * L_) / (E_ * b * t * t * t);

    // Buckling load capacity Pc(x). (The paper's Eq.(17) prints an exponent
    // pattern using "w6", but only x0..x3 exist for this problem -- the
    // standard literature formula uses t^2 * b^6, i.e. x2^2 * x3^6, which is
    // used here.)
    const double inner = std::max(0.0, (t * t * std::pow(b, 6.0)) / 36.0);
    const double Pc = (4.013 * E_ * std::sqrt(inner) / (L_ * L_))
                     * (1.0 - (t / (2.0 * L_)) * std::sqrt(E_ / (4.0 * G_)));

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };

    double penalty = 0.0;
    penalty += std::pow(pos(tau - tau_max_)   / tau_max_,   2.0);
    penalty += std::pow(pos(sigma - sigma_max_) / sigma_max_, 2.0);
    penalty += std::pow(pos(h - b) / 2.0,                    2.0);
    penalty += std::pow(pos(cost - 5.0) / 5.0,                2.0);
    penalty += std::pow(pos(0.125 - h) / 0.125,               2.0);
    penalty += std::pow(pos(delta - delta_max_) / delta_max_, 2.0);
    penalty += std::pow(pos(P_ - Pc) / P_,                    2.0);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void WeldedBeam::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
