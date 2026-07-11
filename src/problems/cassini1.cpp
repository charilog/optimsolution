#include "cassini1.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Cassini1::Cassini1()
    : t0_min_(-1000.0), t0_max_(0.0),
      T1_min_(30.0),   T1_max_(400.0),
      T2_min_(100.0),  T2_max_(470.0),
      T3_min_(30.0),   T3_max_(400.0),
      T4_min_(400.0),  T4_max_(2000.0),
      T5_min_(1000.0), T5_max_(6000.0),
      dv_launch_base_(8.5),
      dv_ga_gain_(1.4),
      dv_leg_scale_(6.0),
      dv_saturn_pen_(3.0),
      tof_ref_days_(600.0),
      soft_total_span_(6500.0),
      hard_pen_(5e3)
{
    setName("cassini1");
    setFullName("Cassini1 MGA surrogate ΔV (Earth-Venus-Venus-Earth-Jupiter-Saturn)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");
}

void Cassini1::init(int /*dim*/) {
    // Force D = 6 (as in the real Cassini1 benchmark) and set bounds
    Problem::init(6);

    Vec lo(6), hi(6);
    lo[0] = t0_min_; hi[0] = t0_max_;
    lo[1] = T1_min_; hi[1] = T1_max_;
    lo[2] = T2_min_; hi[2] = T2_max_;
    lo[3] = T3_min_; hi[3] = T3_max_;
    lo[4] = T4_min_; hi[4] = T4_max_;
    lo[5] = T5_min_; hi[5] = T5_max_;

    setBounds(lo, hi);
}

double Cassini1::evaluate_core(const Vec& x) {
    double t0 = x[0];
    double T1 = x[1], T2 = x[2], T3 = x[3], T4 = x[4], T5 = x[5];

    auto out = [](double v, double lo, double hi){ return (v < lo) || (v > hi); };
    double pen = 0.0;
    if (out(t0, t0_min_, t0_max_)) pen += hard_pen_;
    if (out(T1, T1_min_, T1_max_)) pen += hard_pen_;
    if (out(T2, T2_min_, T2_max_)) pen += hard_pen_;
    if (out(T3, T3_min_, T3_max_)) pen += hard_pen_;
    if (out(T4, T4_min_, T4_max_)) pen += hard_pen_;
    if (out(T5, T5_min_, T5_max_)) pen += hard_pen_;

    T1 = clamp(T1, T1_min_, T1_max_);
    T2 = clamp(T2, T2_min_, T2_max_);
    T3 = clamp(T3, T3_min_, T3_max_);
    T4 = clamp(T4, T4_min_, T4_max_);
    T5 = clamp(T5, T5_min_, T5_max_);

    // 1) Launch surrogate (decreases with longer first leg)
    double dv_launch = dv_launch_base_ - 2.0 * std::log(1.0 + T1 / 120.0);
    dv_launch = std::max(4.0, dv_launch);

    // 2) Gravity-assist gains from the two Venus flybys and the Earth flyby
    //    (three intermediate legs T1, T2, T3 each contribute a GA benefit)
    auto ga_gain_of = [&](double T, double ref){
        double r = std::clamp(T / ref, 0.3, 3.0);
        return dv_ga_gain_ * (1.0 / std::sqrt(r));
    };
    double dv_gain = ga_gain_of(T1, 150.0) + ga_gain_of(T2, 280.0) + ga_gain_of(T3, 150.0);

    // 3) Per-leg cruise cost (all five legs)
    auto leg_cost = [&](double T){
        double r = std::clamp(T / tof_ref_days_, 0.2, 5.0);
        return dv_leg_scale_ * (1.0 / (1.0 + 1.5 * r));
    };
    double dv_legs = leg_cost(T1) + leg_cost(T2) + leg_cost(T3) + leg_cost(T4) + leg_cost(T5);

    // 4) Saturn orbit-insertion difficulty: eases with a longer final leg
    //    (more time to shape the approach) but never disappears entirely.
    double rT5 = std::clamp(T5 / 2500.0, 0.3, 3.0);
    double dv_saturn = dv_saturn_pen_ * (1.0 / std::sqrt(rT5));

    // 5) Combine
    double dv_total = dv_launch + dv_legs + dv_saturn - dv_gain;

    // 6) Soft penalty for excessive total time of flight
    const double total_days = T1 + T2 + T3 + T4 + T5;
    if (total_days > soft_total_span_) {
        dv_total += 0.006 * (total_days - soft_total_span_);
    }

    // 7) Hard-bound penalties
    dv_total += pen;

    if (!std::isfinite(dv_total)) dv_total = 40.0 + pen;
    return dv_total;
}

void Cassini1::gradient_core(const Vec& x, Vec& g) {
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
