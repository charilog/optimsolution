#include "tandem.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Tandem::Tandem()
{
    setName("tandem");
    setFullName("Tandem MGA-1DSM surrogate ΔV");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");
}

void Tandem::init(int /*dim*/) {
    // Force D=18 
    Problem::init(18);

    Vec lo(18), hi(18);
    lo[0] = t0_min_;  hi[0] = t0_max_;

    lo[1] = T1_min_;  hi[1] = T1_max_;
    lo[2] = T2_min_;  hi[2] = T2_max_;
    lo[3] = T3_min_;  hi[3] = T3_max_;
    lo[4] = T4_min_;  hi[4] = T4_max_;
    lo[5] = T5A_min_; hi[5] = T5A_max_;
    lo[6] = T5B_min_; hi[6] = T5B_max_;

    for (int i = 7; i < 18; ++i) {
        lo[i] = 0.0;
        hi[i] = 1.0;
    }

    setBounds(lo, hi);
}

double Tandem::evaluate_core(const Vec& x) {
    // Clamp DSM/shaping for stability 
    double t0   = x[0];
    double T1   = x[1], T2 = x[2], T3 = x[3], T4 = x[4], T5A = x[5], T5B = x[6];
    double s1   = clamp01(x[7]),  s2  = clamp01(x[8]),  s3  = clamp01(x[9]),  s4  = clamp01(x[10]);
    double s5A  = clamp01(x[11]), s5B = clamp01(x[12]);
    double rp   = clamp01(x[13]);
    double kA1  = clamp01(x[14]), kA2 = clamp01(x[15]);
    double kB1  = clamp01(x[16]), kB2 = clamp01(x[17]);

    // Hard-bounds penalty (scaled to km/s at the end)
    auto out = [](double v,double lo,double hi){ return (v<lo)||(v>hi); };
    double pen = 0.0;
    if (out(t0,  t0_min_, t0_max_))
        pen += p_pen_base_ * std::abs(t0 - clamp(t0,t0_min_,t0_max_));
    if (out(T1,  T1_min_, T1_max_))   pen += p_pen_base_;
    if (out(T2,  T2_min_, T2_max_))   pen += p_pen_base_;
    if (out(T3,  T3_min_, T3_max_))   pen += p_pen_base_;
    if (out(T4,  T4_min_, T4_max_))   pen += p_pen_base_;
    if (out(T5A, T5A_min_, T5A_max_)) pen += p_pen_base_;
    if (out(T5B, T5B_min_, T5B_max_)) pen += p_pen_base_;

    // Clamp times for surrogate stability before using
    T1  = clamp(T1,  T1_min_,  T1_max_);
    T2  = clamp(T2,  T2_min_,  T2_max_);
    T3  = clamp(T3,  T3_min_,  T3_max_);
    T4  = clamp(T4,  T4_min_,  T4_max_);
    T5A = clamp(T5A, T5A_min_, T5A_max_);
    T5B = clamp(T5B, T5B_min_, T5B_max_);

    // ---- Surrogate ΔV terms  ----

    // 1) Launch surrogate (decreases with longer first leg)
    double dv_launch = dv_launch_base_ - 3.0 * std::log(1.0 + T1 / 200.0);
    dv_launch = std::max(6.0, dv_launch);

    // 2) Gravity-assist gains before Jupiter (E->V1->E1->E2)
    auto ga_gain_of = [&](double T, double ref){
        double r = std::clamp(T / ref, 0.3, 3.0);
        return dv_ga_gain_ * (1.0 / std::sqrt(r));
    };
    double dv_gain_preJ = ga_gain_of(T1, 200.0)
                        + ga_gain_of(T2, 250.0)
                        + ga_gain_of(T3, 300.0);

    // 3) Help from Jupiter transit
    double dv_gain_J = dv_jupiter_aid_ *
                       (1.0 / std::sqrt(std::clamp(T4 / 500.0, 0.5, 4.0)));

    // 4) Cost per leg (common legs)
    auto leg_cost = [&](double T){
        double r = std::clamp(T / tof_ref_days_, 0.2, 4.0);
        return dv_leg_scale_ * (1.0 / (1.0 + 2.0 * r));
    };
    double dv_legs_common = leg_cost(T1) + leg_cost(T2)
                          + leg_cost(T3) + leg_cost(T4);

    // 5) Cost/benefit on branches to Saturn
    auto branch_cost = [&](double T5, double s5, double k1, double k2){
        double base      = leg_cost(T5);
        double dsm_help  = 1.5 * s5 * (0.2 + 0.8 * rp);
        double dsm_cost  = dsm_scale_ * (0.3 + 0.7 * s5)
                         * (0.5 + 0.5 * (k1 + k2));
        return std::max(0.0, base - dsm_help) + dsm_cost;
    };
    double dv_branchA = branch_cost(T5A, s5A, kA1, kA2);
    double dv_branchB = branch_cost(T5B, s5B, kB1, kB2);

    // 6) DSMs on intermediate common legs
    auto mid_dsm = [&](double s){
        double help = 1.0 * s * (0.1 + 0.9 * rp);
        double cost = dsm_scale_ * (0.25 + 0.75 * s);
        return std::make_pair(help, cost);
    };
    auto h1c1 = mid_dsm(s1);
    auto h2c2 = mid_dsm(s2);
    auto h3c3 = mid_dsm(s3);
    auto h4c4 = mid_dsm(s4);

    double dv_dsm_common  = h1c1.second + h2c2.second
                          + h3c3.second + h4c4.second;
    double dv_help_common = h1c1.first  + h2c2.first
                          + h3c3.first  + h4c4.first;

    // 7) Combination
    double dv_total =
          dv_launch
        + dv_legs_common
        + dv_branchA + dv_branchB
        + dv_dsm_common
        - dv_help_common
        - dv_gain_preJ
        - dv_gain_J;

    // 8) Mild total time penalty
    double total_days = T1 + T2 + T3 + T4 + 0.5 * (T5A + T5B);
    if (total_days > soft_tof_span_) {
        dv_total += 0.01 * (total_days - soft_tof_span_);
    }

    // 9) hard-bound penalty (scaled)
    dv_total += pen * 1e-6;

    if (!(dv_total > 0.0) || std::isnan(dv_total) || std::isinf(dv_total))
        dv_total = 50.0 + pen * 1e-6;

    return dv_total;
}

void Tandem::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
