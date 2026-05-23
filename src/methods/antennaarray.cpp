#include "antennaarray.h"
#include <algorithm>
#include <cmath>

namespace optimsolution {

namespace {
    inline double pi() {
        return 3.1415926535897932384626433832795;
    }
}

// ---- Helper: compute normalized |AF(θ)| and optionally return S, e^{jα_k} ----
static inline double array_factor_norm_at_theta(
    double ct, double st,
    const std::vector<double>& cos_tk, const std::vector<double>& sin_tk,
    const optimsolution::Vec& x,
    std::complex<double>* S_out = nullptr,
    std::vector<std::complex<double>>* ej_out = nullptr
) {
    const int N = 6;
    std::complex<double> S(0.0, 0.0);
    if (ej_out) ej_out->assign(N, std::complex<double>(0.0, 0.0));

    for (int k = 0; k < N; ++k) {
        const double rk  = x[k];
        const double deg = x[k + N];
        const double phi = deg * (pi()/180.0); // radians

        // cos(θ - θ_k) = cosθ cosθ_k + sinθ sinθ_k
        const double cos_t_minus_tk = ct * cos_tk[k] + st * sin_tk[k];
        const double alpha = (2.0 * pi()) * rk * cos_t_minus_tk + phi;

        const std::complex<double> ej(std::cos(alpha), std::sin(alpha));
        if (ej_out) (*ej_out)[k] = ej;
        S += ej;
    }
    if (S_out) *S_out = S;
    return std::abs(S) / static_cast<double>(N);
}

// -----------------------------------------------------------------------------
//  Constructor & initialization
// -----------------------------------------------------------------------------

AntennaArray::AntennaArray()
    : N(6),
      samples_(721),                  // ~0.5° step
      exclude_rad_(pi() / 60.0),      // ~3° main-lobe exclusion
      last_imax_(0)
{
    // metadata for the framework
    setName("antennaarray");
    setFullName("6-element circular antenna array (sidelobe level minimization)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("electromagnetics / antenna array synthesis");
    

    build_geometry();
    build_scan();
}

void AntennaArray::build_geometry() {
    theta_k.resize(N);
    cos_tk.resize(N);
    sin_tk.resize(N);
    for (int k = 0; k < N; ++k) {
        theta_k[k] = (2.0 * pi() * k) / static_cast<double>(N);
        cos_tk[k]  = std::cos(theta_k[k]);
        sin_tk[k]  = std::sin(theta_k[k]);
    }
}

void AntennaArray::build_scan() {
    if (samples_ < 181) samples_ = 181; // not too sparse
    thetas.resize(samples_);
    cos_th.resize(samples_);
    sin_th.resize(samples_);
    for (int i = 0; i < samples_; ++i) {
        const double t = (2.0 * pi()) *
            (static_cast<double>(i) / static_cast<double>(samples_ - 1));
        thetas[i] = t;
        cos_th[i] = std::cos(t);
        sin_th[i] = std::sin(t);
    }
}

void AntennaArray::setScan(int samples, double exclude_deg) {
    samples_    = samples;
    exclude_rad_ = std::max(0.0, exclude_deg) * (pi()/180.0);
    build_scan();
}

void AntennaArray::init(int /*dim*/) {
    // 12 vars: 6 radii + 6 phases(deg)
    Problem::init(12);

    lo_.assign(12, 0.0);
    hi_.assign(12, 0.0);
    for (int i = 0; i < 6; ++i) {
        lo_[i] = 0.2;    hi_[i] = 1.0;     // radii
    }
    for (int i = 6; i < 12; ++i) {
        lo_[i] = -180.0; hi_[i] = 180.0;   // phases in degrees
    }
    setBounds(lo_, hi_);

    build_geometry();
    build_scan();
}

// -----------------------------------------------------------------------------
//  Objective: max sidelobe level over scan (normalized |AF(θ)|)
// -----------------------------------------------------------------------------

double AntennaArray::evaluate_core(const Vec& x) {
    double max_af = -1.0;
    int imax = 0;

    for (int i = 0; i < samples_; ++i) {
        const double t = thetas[i];
        // exclude small main-lobe zone around θ=0 (and 2π)
        if (t < exclude_rad_ || t > (2.0 * pi() - exclude_rad_)) continue;

        const double af = array_factor_norm_at_theta(
            cos_th[i], sin_th[i], cos_tk, sin_tk, x
        );
        if (af > max_af) {
            max_af = af;
            imax   = i;
        }
    }
    last_imax_ = imax;
    return max_af;
}

// -----------------------------------------------------------------------------
//  Analytic subgradient at θ* = argmax (f(x) = |AF(θ*)|/N)
// -----------------------------------------------------------------------------

void AntennaArray::gradient_core(const Vec& x, Vec& g) {
    g.assign(12, 0.0);

    const int i  = last_imax_;      // θ* index
    const double ct = cos_th[i];
    const double st = sin_th[i];

    // Compute S and e^{jα_k} at θ*
    std::complex<double> S;
    std::vector<std::complex<double>> ej;
    const double af = array_factor_norm_at_theta(
        ct, st, cos_tk, sin_tk, x, &S, &ej
    );

    if (af <= 0.0) return; // degenerate case → zero gradient

    const double Nf = 6.0;
    const double invN = 1.0 / Nf;
    const double inv_absS = 1.0 / (af * Nf); // |S| = af * N

    // U = S* / |S|
    const std::complex<double> U = std::conj(S) * inv_absS;

    for (int k = 0; k < 6; ++k) {
        // cos(θ - θ_k)
        const double cos_t_minus_tk = ct * cos_tk[k] + st * sin_tk[k];

        // common scalar = -Im( U * e^{jα_k} )
        const double common = -std::imag(U * ej[k]);

        // d|S|/dr_k = common * (2π cos(...))
        const double d_absS_dr   = common * (2.0 * pi() * cos_t_minus_tk);

        // φ_k = deg * π/180 ⇒ dφ/d(deg) = π/180
        const double d_absS_dphi = common;
        const double d_absS_ddeg = d_absS_dphi * (pi()/180.0);

        // f = |S|/N ⇒ divide by N
        g[k]     = d_absS_dr   * invN; // radii
        g[k + 6] = d_absS_ddeg * invN; // phases in degrees
    }
}

} // namespace optimsolution
