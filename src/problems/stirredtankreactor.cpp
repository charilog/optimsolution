#include "stirredtankreactor.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

StirredTankReactor::StirredTankReactor()
    : Nstages_(10), dt_(0.05),
      V_(1.0), k_(300.0), N_(5.0), Tf_(0.3947), Tc_(0.3816), alpha_c_(1.95e-4),
      c_ss_(0.1367), T_ss_(0.7293), rho_ss_(1.0), F_ss_(390.0),
      c0_(0.2), T0_(0.6),
      rho_min_(0.5), rho_max_(2.0),
      F_min_(0.0),   F_max_(700.0),
      w_c_(1.0), w_T_(1.0), w_rho_(2e-3), w_F_(2e-6)
{
    setName("stirredtankreactor");
    setFullName("Optimal control of a non-linear stirred tank reactor (CSTR)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("dynamic optimal-control benchmark");
}

void StirredTankReactor::init(int /*dim*/) {
    // Force D = 2*Nstages_ (rho_k, F_k pairs) and set bounds
    const int D = 2 * Nstages_;
    Problem::init(D);

    Vec lo(D), hi(D);
    for (int k = 0; k < Nstages_; ++k) {
        lo[2*k + 0] = rho_min_; hi[2*k + 0] = rho_max_;
        lo[2*k + 1] = F_min_;   hi[2*k + 1] = F_max_;
    }
    setBounds(lo, hi);
}

double StirredTankReactor::simulate_cost(const Vec& x) const {
    double c = c0_;
    double T = T0_;
    double J = 0.0;

    for (int k = 0; k < Nstages_; ++k) {
        double rho = clamp(x[2*k + 0], rho_min_, rho_max_);
        double F   = clamp(x[2*k + 1], F_min_,   F_max_);

        // Forward-Euler step of the CSTR ODEs
        const double reaction = c * k_ * std::exp(-N_ / std::max(T, 1e-6));
        const double dc = (1.0 - c) * rho / V_ - reaction;
        const double dT = (Tf_ - T) * rho / V_ + reaction - F * alpha_c_ * (T - Tc_);

        c += dt_ * dc;
        T += dt_ * dT;

        if (!std::isfinite(c) || !std::isfinite(T)) {
            return 1e12;
        }

        const double dcv   = c   - c_ss_;
        const double dTv   = T   - T_ss_;
        const double drho  = rho - rho_ss_;
        const double dF    = F   - F_ss_;

        J += w_c_ * dcv * dcv + w_T_ * dTv * dTv
           + w_rho_ * drho * drho + w_F_ * dF * dF;
    }
    return J;
}

double StirredTankReactor::evaluate_core(const Vec& x) {
    double J = simulate_cost(x);
    if (!std::isfinite(J)) J = 1e12;
    return J;
}

void StirredTankReactor::gradient_core(const Vec& x, Vec& g) {
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
