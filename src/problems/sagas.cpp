#include "sagas.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double SAGAS_PI = 3.141592653589793238462643383279502884; }

Sagas::Sagas()
    : t0_min_(7000.0),   t0_max_(9100.0),
      vinf_min_(0.0),    vinf_max_(7.0),
      angle_min_(0.0),   angle_max_(2.0 * SAGAS_PI),
      T1_min_(50.0),     T1_max_(900.0),
      T2_min_(300.0),    T2_max_(3500.0),
      dv_launch_base_(0.0),   // launch dv is folded into vinf directly (see below)
      dv_ega_gain_(3.0),
      dv_leg_scale_(4.0),
      dsm_scale_(1.0),
      tof_ref_days_(500.0),
      hard_pen_(5e3)
{
    setName("sagas");
    setFullName("Sagas ΔV-EGA surrogate ΔV (Earth-Earth-Jupiter, toward 50 AU)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");
}

void Sagas::init(int /*dim*/) {
    // Force D = 12 (as in the real Sagas benchmark) and set bounds
    Problem::init(12);

    Vec lo(12), hi(12);
    lo[0] = t0_min_;   hi[0] = t0_max_;
    lo[1] = vinf_min_; hi[1] = vinf_max_;
    lo[2] = angle_min_; hi[2] = angle_max_;
    lo[3] = angle_min_; hi[3] = angle_max_;
    lo[4] = T1_min_;   hi[4] = T1_max_;
    lo[5] = T2_min_;   hi[5] = T2_max_;
    for (int i = 6; i < 12; ++i) { lo[i] = 0.0; hi[i] = 1.0; }

    setBounds(lo, hi);
}

double Sagas::evaluate_core(const Vec& x) {
    double t0   = x[0];
    double vinf = x[1];
    double T1   = x[4];
    double T2   = x[5];
    double s1   = clamp01(x[6]),  s2  = clamp01(x[7]);
    double rp1  = clamp01(x[8]),  rp2 = clamp01(x[9]);
    double b1   = clamp01(x[10]), b2  = clamp01(x[11]);

    auto out = [](double v, double lo, double hi){ return (v < lo) || (v > hi); };
    double pen = 0.0;
    if (out(t0,   t0_min_,   t0_max_))   pen += hard_pen_;
    if (out(vinf, vinf_min_, vinf_max_)) pen += hard_pen_;
    if (out(T1,   T1_min_,   T1_max_))   pen += hard_pen_;
    if (out(T2,   T2_min_,   T2_max_))   pen += hard_pen_;

    vinf = clamp(vinf, vinf_min_, vinf_max_);
    T1   = clamp(T1,   T1_min_,   T1_max_);
    T2   = clamp(T2,   T2_min_,   T2_max_);

    // 1) Launch ΔV: directly tied to the chosen hyperbolic excess speed
    double dv_launch = dv_launch_base_ + vinf;

    // 2) Earth gravity-assist (EGA) benefit, shaped by flyby radius / B-plane
    double rr1 = std::clamp(T1 / 250.0, 0.3, 3.0);
    double dv_ega = dv_ega_gain_ * (1.0 / std::sqrt(rr1)) * (0.3 + 0.7 * rp1) * (0.5 + 0.5 * b1);

    // 3) Per-leg cruise cost (both legs)
    auto leg_cost = [&](double T){
        double r = std::clamp(T / tof_ref_days_, 0.2, 6.0);
        return dv_leg_scale_ * (1.0 / (1.0 + 1.5 * r));
    };
    double dv_legs = leg_cost(T1) + leg_cost(T2);

    // 4) DSM help/cost trade-off on each leg
    auto dsm_help_cost = [&](double s, double shape){
        double help = s * (0.2 + 0.8 * shape);
        double cost = dsm_scale_ * (0.25 + 0.75 * s);
        return std::pair<double,double>(help, cost);
    };
    auto hc1 = dsm_help_cost(s1, rp1);
    auto hc2 = dsm_help_cost(s2, rp2);
    double dv_dsm_cost = hc1.second + hc2.second;
    double dv_dsm_help = hc1.first  + hc2.first;

    // 5) Combine
    double dv_total = dv_launch + dv_legs + dv_dsm_cost - dv_dsm_help - dv_ega;

    // 6) Hard-bound penalties
    dv_total += pen;

    if (!std::isfinite(dv_total)) dv_total = 60.0 + pen;
    return dv_total;
}

void Sagas::gradient_core(const Vec& x, Vec& g) {
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
