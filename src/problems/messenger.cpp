#include "messenger.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Messenger::Messenger()
    : t0_min_(7000.0),  t0_max_(10000.0),
      T1_min_(30.0),    T1_max_(400.0),
      T2_min_(30.0),    T2_max_(400.0),
      T3_min_(30.0),    T3_max_(600.0),
      T4_min_(30.0),    T4_max_(600.0),
      T5_min_(30.0),    T5_max_(700.0),
      // surrogate tuning (faithful copy)
      dv_launch_base_(10.0),
      dv_venus_gain_(1.6),
      dv_mercury_pen_(6.0),
      dv_leg_scale_(16.0),
      tof_ref_inner_(220.0),
      dsm_scale_(1.1),
      soft_total_span_(1400.0),
      hard_pen_(5e3)
{
   
    setName("messenger");
    setFullName("MESSENGER MGA-1DSM surrogate ΔV");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");

 
}

void Messenger::init(int /*dim*/) {
    // Force D = 14 (as in reference) and set bounds
    Problem::init(14);

    Vec lo(14), hi(14);
    lo[0]=t0_min_; hi[0]=t0_max_;
    lo[1]=T1_min_; hi[1]=T1_max_;
    lo[2]=T2_min_; hi[2]=T2_max_;
    lo[3]=T3_min_; hi[3]=T3_max_;
    lo[4]=T4_min_; hi[4]=T4_max_;
    lo[5]=T5_min_; hi[5]=T5_max_;
    for (int i=6; i<14; ++i) { lo[i]=0.0; hi[i]=1.0; }

    setBounds(lo, hi);
}

double Messenger::evaluate_core(const Vec& x) {
    // Local copy to clamp DSM/shaping for stability
    double t0 = x[0];
    double T1 = x[1], T2 = x[2], T3 = x[3], T4 = x[4], T5 = x[5];
    double s1 = clamp01(x[6]),  s2 = clamp01(x[7]),  s3 = clamp01(x[8]),
           s4 = clamp01(x[9]),  s5 = clamp01(x[10]);
    double rp = clamp01(x[11]), k1 = clamp01(x[12]), k2 = clamp01(x[13]);

    // Hard-bounds penalty (km/s add)
    auto out = [](double v, double lo, double hi){ return (v < lo) || (v > hi); };
    double pen = 0.0;
    if (out(t0, t0_min_, t0_max_)) pen += hard_pen_;
    if (out(T1, T1_min_, T1_max_)) pen += hard_pen_;
    if (out(T2, T2_min_, T2_max_)) pen += hard_pen_;
    if (out(T3, T3_min_, T3_max_)) pen += hard_pen_;
    if (out(T4, T4_min_, T4_max_)) pen += hard_pen_;
    if (out(T5, T5_min_, T5_max_)) pen += hard_pen_;

    // Clamp times to safe ranges for surrogate stability
    T1 = clamp(T1, T1_min_, T1_max_);
    T2 = clamp(T2, T2_min_, T2_max_);
    T3 = clamp(T3, T3_min_, T3_max_);
    T4 = clamp(T4, T4_min_, T4_max_);
    T5 = clamp(T5, T5_min_, T5_max_);

    // --- Surrogate ΔV components (as in reference) ---

    // 1) Launch surrogate (decreases with longer first leg)
    double dv_launch = dv_launch_base_ - 2.5 * std::log(1.0 + T1 / 180.0);
    if (dv_launch < 6.5) dv_launch = 6.5;

    // 2) Venus flyby gains (two Venus flybys)
    auto venus_gain_of = [](double T, double dv_gain){
        double r = std::clamp(T / 200.0, 0.3, 3.0);
        return dv_gain * (1.0 / std::sqrt(r));
    };
    double dv_gain_venus = venus_gain_of(T1, dv_venus_gain_) + venus_gain_of(T2, dv_venus_gain_);

    // 3) Per-leg transfer costs (inner system)
    auto leg_cost = [&](double T){
        double r = std::clamp(T / tof_ref_inner_, 0.25, 4.0);
        return dv_leg_scale_ * (1.0 / (1.0 + 2.0 * r));
    };
    double dv_legs = leg_cost(T1) + leg_cost(T2) + leg_cost(T3) + leg_cost(T4) + leg_cost(T5);

    // 4) DSMs: benefit (help) vs cost
    auto dsm_help_cost = [&](double s, double strength){
        // help ~ s*(0.2 + 0.8*rp)*strength
        // cost ~ dsm_scale*(0.25 + 0.75*s)*(0.5 + 0.5*(k1+k2))
        double help = strength * s * (0.2 + 0.8 * rp);
        double cost = dsm_scale_ * (0.25 + 0.75 * s) * (0.5 + 0.5 * (k1 + k2));
        return std::pair<double,double>(help, cost);
    };
    auto h1c1 = dsm_help_cost(s1, 1.0);
    auto h2c2 = dsm_help_cost(s2, 1.0);
    auto h3c3 = dsm_help_cost(s3, 1.35);
    auto h4c4 = dsm_help_cost(s4, 1.35);
    auto h5c5 = dsm_help_cost(s5, 1.50);

    double dv_dsm_cost = h1c1.second + h2c2.second + h3c3.second + h4c4.second + h5c5.second;
    double dv_dsm_help = h1c1.first  + h2c2.first  + h3c3.first  + h4c4.first  + h5c5.first;

    // 5) Mercury rendezvous difficulty (lower with longer T5, higher shaping helps)
    double rT5 = std::clamp(T5 / 300.0, 0.2, 4.0);
    double dv_mercury = dv_mercury_pen_ * (1.0 / std::sqrt(rT5))
                        * (1.0 - 0.3 * rp)
                        * (1.0 - 0.2 * (k1 + k2));

    // 6) Combine
    double dv_total =
          dv_launch
        + dv_legs
        + dv_dsm_cost
        + dv_mercury
        - dv_dsm_help
        - dv_gain_venus;

    // 7) Soft penalty for too long total ToF
    const double total_days = T1 + T2 + T3 + T4 + T5;
    if (total_days > soft_total_span_) {
        dv_total += 0.008 * (total_days - soft_total_span_);
    }

    // 8) Add hard-bound penalties
    dv_total += pen;

    if (!std::isfinite(dv_total)) dv_total = 50.0 + pen;
    return dv_total;
}

void Messenger::gradient_core(const Vec& x, Vec& g) {
    // forward differences (consistent with your reference)
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
