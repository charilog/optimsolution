#include "vibratingplatform.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.1415926535897932384626433832795; }

VibratingPlatform::VibratingPlatform()
{

    setName("vibratingplatform");
    setFullName("Base-excited SDOF isolation platform design");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("mechanical design / vibration isolation");


    // ---- Sensible defaults (laboratory isolation platform scale) ----
    m_         = 50.0;     // kg
    fmin_      = 5.0;      // Hz
    fmax_      = 200.0;    // Hz
    fwork_     = 30.0;     // Hz (typical disturbance)
    delta_max_ = 0.005;    // 5 mm
    Y0_work_   = 0.0005;   // 0.5 mm base motion
    zeta_min_  = 0.02;     // 2% damping
    zeta_max_  = 0.30;     // 30% damping
    fn_min_    = 2.0;      // Hz
    fn_max_    = 20.0;     // Hz
    Nfreq_     = 400;      // log samples across band

    // Variable bounds
    k_min_ = 1.0e3;   k_max_ = 2.0e5;   // N/m
    c_min_ = 50.0;    c_max_ = 5.0e3;   // N*s/m

    // Penalty weights
    w_delta_ = 100.0;
    w_disp_  = 100.0;
    w_zeta_  = 25.0;
    w_fn_    = 25.0;
}

void VibratingPlatform::build_freq_grid(std::vector<double>& f) const {
    f.resize(Nfreq_);
    const double a = std::log(fmin_);
    const double b = std::log(fmax_);
    for (int i = 0; i < Nfreq_; ++i) {
        const double t = (Nfreq_ > 1) ? (double)i / (Nfreq_ - 1) : 0.0;
        f[i] = std::exp(a + t * (b - a));
    }
}

void VibratingPlatform::init(int /*dim*/) {
    // Force D=2 and set bounds
    Problem::init(2);
    Vec lo = { k_min_, c_min_ };
    Vec hi = { k_max_, c_max_ };
    setBounds(lo, hi);
}

double VibratingPlatform::evaluate_core(const Vec& x) {
    // Read & clamp inside bounds for numerical stability
    double k = clampd(x[0], k_min_, k_max_);
    double c = clampd(x[1], c_min_, c_max_);

    // Derived quantities
    const double wn   = std::sqrt(k / m_);           // rad/s
    const double fn   = wn / (2.0 * PI);             // Hz
    const double zeta = c / (2.0 * std::sqrt(k * m_));

    // Frequency grid and average |H|^2 over [fmin,fmax] (base-excited SDOF)
    std::vector<double> fv;
    build_freq_grid(fv);

    double sumH2 = 0.0;
    for (int i = 0; i < Nfreq_; ++i) {
        const double w = 2.0 * PI * fv[i];
        const double r = w / wn;
        const double a = 2.0 * zeta * r;
        const double num = 1.0 + a*a;
        const double den = (1.0 - r*r) * (1.0 - r*r) + a*a;
        const double H2  = num / den;                // |H|^2
        sumH2 += H2;
    }
    const double avgH2 = sumH2 / std::max(1, Nfreq_);

    // Constraints as smooth penalties

    // (1) Static sag: delta = m g / k <= delta_max
    const double g = 9.80665;
    const double delta = m_ * g / k;
    double pen_delta = 0.0;
    if (delta > delta_max_) {
        const double r = delta / delta_max_ - 1.0;
        pen_delta = w_delta_ * r * r;
    }

    // (2) Displacement at fwork with base amplitude Y0_work: X = |H|*Y0 <= x_max
    const double ww   = 2.0 * PI * fwork_;
    const double rw   = ww / wn;
    const double aw   = 2.0 * zeta * rw;
    const double Hw   = std::sqrt((1.0 + aw*aw) /
                        (((1.0 - rw*rw)*(1.0 - rw*rw)) + aw*aw));
    const double Xw   = Hw * Y0_work_;
    const double x_max = delta_max_ * 2.0;          // relaxed vs static
    double pen_disp = 0.0;
    if (Xw > x_max) {
        const double r = Xw / x_max - 1.0;
        pen_disp = w_disp_ * r * r;
    }

    // (3) Damping ratio bounds
    double pen_zeta = 0.0;
    if (zeta < zeta_min_) {
        const double r = (zeta_min_ - zeta) / zeta_min_;
        pen_zeta += w_zeta_ * r * r;
    }
    if (zeta > zeta_max_) {
        const double r = (zeta - zeta_max_) / zeta_max_;
        pen_zeta += w_zeta_ * r * r;
    }

    // (4) Natural frequency bounds
    double pen_fn = 0.0;
    if (fn < fn_min_) {
        const double r = (fn_min_ - fn) / fn_min_;
        pen_fn += w_fn_ * r * r;
    }
    if (fn > fn_max_) {
        const double r = (fn - fn_max_) / fn_max_;
        pen_fn += w_fn_ * r * r;
    }

    double cost = avgH2 + pen_delta + pen_disp + pen_zeta + pen_fn;
    if (!std::isfinite(cost)) cost = 1e6;
    return cost;
}

void VibratingPlatform::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < 2; ++i) {
        const double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
