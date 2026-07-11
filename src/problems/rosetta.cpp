#include "rosetta.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double ROSETTA_PI = 3.141592653589793238462643383279502884; }

Rosetta::Rosetta()
    : t0_min_(1460.0),  t0_max_(1825.0),
      vinf_min_(0.0),   vinf_max_(4.0),
      angle_min_(0.0),  angle_max_(2.0 * ROSETTA_PI),
      dv_ega_gain_(2.2),
      dv_leg_scale_(5.0),
      dsm_scale_(1.1),
      dv_comet_pen_(2.0),
      tof_ref_days_(450.0),
      hard_pen_(5e3)
{
    setName("rosetta");
    setFullName("Rosetta MGA-1DSM surrogate ΔV (Earth-Earth-Mars-Earth-Earth-67P)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");

    // 5 leg-duration bounds (days), loosely matching the real mission's
    // multi-year, multi-flyby structure.
    T_min_ = {50.0, 100.0, 100.0, 100.0, 100.0};
    T_max_ = {700.0, 700.0, 700.0, 700.0, 1850.0};
}

void Rosetta::init(int /*dim*/) {
    // Force D = 22 (as in the real Rosetta benchmark) and set bounds
    Problem::init(22);

    Vec lo(22), hi(22);
    lo[0] = t0_min_;   hi[0] = t0_max_;
    lo[1] = vinf_min_; hi[1] = vinf_max_;
    lo[2] = angle_min_; hi[2] = angle_max_;
    lo[3] = angle_min_; hi[3] = angle_max_;
    for (int i = 0; i < 5; ++i) { lo[4 + i] = T_min_[i]; hi[4 + i] = T_max_[i]; }
    for (int i = 9; i < 22; ++i) { lo[i] = 0.0; hi[i] = 1.0; } // 5 DSM fracs + 4 rp + 4 b

    setBounds(lo, hi);
}

double Rosetta::evaluate_core(const Vec& x) {
    double t0   = x[0];
    double vinf = x[1];
    Vec T(5);
    for (int i = 0; i < 5; ++i) T[i] = x[4 + i];
    double s[5];
    for (int i = 0; i < 5; ++i) s[i] = clamp01(x[9 + i]);
    double rp[4];
    for (int i = 0; i < 4; ++i) rp[i] = clamp01(x[14 + i]);
    double bp[4];
    for (int i = 0; i < 4; ++i) bp[i] = clamp01(x[18 + i]);

    auto out = [](double v, double lo, double hi){ return (v < lo) || (v > hi); };
    double pen = 0.0;
    if (out(t0,   t0_min_,   t0_max_))   pen += hard_pen_;
    if (out(vinf, vinf_min_, vinf_max_)) pen += hard_pen_;
    for (int i = 0; i < 5; ++i) {
        if (out(T[i], T_min_[i], T_max_[i])) pen += hard_pen_;
        T[i] = clamp(T[i], T_min_[i], T_max_[i]);
    }
    vinf = clamp(vinf, vinf_min_, vinf_max_);

    // 1) Launch ΔV: tied directly to the chosen hyperbolic excess speed
    double dv_launch = vinf;

    // 2) Earth/Mars gravity-assist gains at each of the 4 intermediate
    //    flybys (Earth, Mars, Earth, Earth), shaped by flyby radius/B-plane
    double dv_ega = 0.0;
    for (int i = 0; i < 4; ++i) {
        double r = std::clamp(T[i] / 250.0, 0.3, 3.0);
        dv_ega += dv_ega_gain_ * (1.0 / std::sqrt(r)) * (0.3 + 0.7 * rp[i]) * (0.5 + 0.5 * bp[i]);
    }

    // 3) Per-leg cruise cost (all five legs)
    auto leg_cost = [&](double T){
        double r = std::clamp(T / tof_ref_days_, 0.2, 6.0);
        return dv_leg_scale_ * (1.0 / (1.0 + 1.5 * r));
    };
    double dv_legs = 0.0;
    for (int i = 0; i < 5; ++i) dv_legs += leg_cost(T[i]);

    // 4) DSM help/cost trade-off on each of the 5 legs
    double dv_dsm_cost = 0.0, dv_dsm_help = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double shape = (i < 4) ? rp[i] : 0.5; // last leg has no flyby shaping
        dv_dsm_help += s[i] * (0.15 + 0.85 * shape);
        dv_dsm_cost += dsm_scale_ * (0.25 + 0.75 * s[i]);
    }

    // 5) Comet-rendezvous matching difficulty at the end of the last leg
    double rTlast = std::clamp(T[4] / 1200.0, 0.3, 4.0);
    double dv_comet = dv_comet_pen_ * (1.0 / std::sqrt(rTlast));

    // 6) Combine
    double dv_total = dv_launch + dv_legs + dv_dsm_cost + dv_comet - dv_dsm_help - dv_ega;

    // 7) Hard-bound penalties
    dv_total += pen;

    if (!std::isfinite(dv_total)) dv_total = 60.0 + pen;
    return dv_total;
}

void Rosetta::gradient_core(const Vec& x, Vec& g) {
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
