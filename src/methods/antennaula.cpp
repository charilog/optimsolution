#include "antennaula.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

AntennaULA::AntennaULA()
{
    // Metadata
    setName("antennaula");
    setFullName("Uniform Linear Array (half-wavelength spacing, amplitude taper)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("electromagnetics / antenna array synthesis");
   
}

void AntennaULA::buildAngleGrid() {
    thetas_rad_.clear();
    thetas_rad_.reserve(ntheta_);
    const double start = -90.0, stop = 90.0;
    const double step = (stop - start) / double(ntheta_ - 1);
    for (int k = 0; k < ntheta_; ++k) {
        const double deg = start + k * step;
        thetas_rad_.push_back(deg2rad(deg));
    }
}

void AntennaULA::init(int dim) {

    const int N = std::max(2, dim < 1 ? 10 : dim); // default 10 αν δοθεί <1
    Problem::init(N);

    // Bounds: amplitudes in [0,1]
    lo_.assign(N, 0.0);
    hi_.assign(N, 1.0);
    setBounds(lo_, hi_);

    // Rebuild angle grid
    buildAngleGrid();
}

// Half-wavelength spacing: d = 0.5λ -> kd = π
// Element index n = 0..N-1, phase_n(θ) = π * n * cos(θ)
double AntennaULA::arrayFactorMag(const Vec& w, double theta_rad) const {
    const int N = static_cast<int>(w.size());
    double csum = 0.0, ssum = 0.0;
    const double cth = std::cos(theta_rad);

    for (int n = 0; n < N; ++n) {
        const double phi = 3.1415926535897932384626433832795 * n * cth; // π * n * cosθ
        const double c = std::cos(phi);
        const double s = std::sin(phi);
        const double a = w[n];
        csum += a * c;
        ssum += a * s;
    }
    return std::sqrt(csum * csum + ssum * ssum);
}

double AntennaULA::evaluate_core(const Vec& x) {
    const int N = dimension();

    // Normalize AF by sum of weights to isolate pattern shape
    double sumw = 0.0;
    for (int i = 0; i < N; ++i) sumw += x[i];
    if (sumw < eps_norm_) sumw = eps_norm_;

    // Mainlobe unit-gain constraint at theta0
    const double AF0 = arrayFactorMag(x, deg2rad(theta0_deg_)) / sumw;
    double main_term = (1.0 - AF0);
    main_term = main_term * main_term;

    // Sidelobe region: outside ±main_win_deg around theta0
    double sll_accum = 0.0;
    int sll_count = 0;
    for (double th : thetas_rad_) {
        const double deg = th * 180.0 / 3.1415926535897932384626433832795;
        if (std::abs(deg - theta0_deg_) <= main_win_deg_) continue; // skip mainlobe window
        const double AF = arrayFactorMag(x, th) / sumw;
        sll_accum += std::pow(std::abs(AF), p_sll_);
        ++sll_count;
    }
    const double sll_term = (sll_count > 0) ? (sll_accum / sll_count) : 0.0;

    // Smoothness of taper (encourage gentle amplitude variation)
    double smooth = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        const double d = x[i + 1] - x[i];
        smooth += d * d;
    }

    // Total objective
    return w_main_ * main_term + w_sll_ * sll_term + w_smooth_ * smooth;
}

// Numerical central-difference gradient (faithful to original implementation)
void AntennaULA::gradient_core(const Vec& x, Vec& g) {
    const int N = dimension();
    g.assign(N, 0.0);

    const double rel = 1e-6, abs = 1e-6;
    Vec xt = x;

    for (int i = 0; i < N; ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);

        // keep inside bounds
        if (x[i] + h > hi_[i]) h = std::min(h, hi_[i] - x[i]);
        if (x[i] - h < lo_[i]) h = std::min(h, x[i] - lo_[i]);
        if (h <= 0.0) { g[i] = 0.0; continue; }

        xt[i] = x[i] + h; const double fp = evaluate_core(xt);
        xt[i] = x[i] - h; const double fm = evaluate_core(xt);
        xt[i] = x[i];

        g[i] = (fp - fm) / (2.0 * h);
    }
}

} // namespace optimsolution
