#include "gtoc1.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

GTOC1::GTOC1()
    : t0_min_(-1000.0), t0_max_(0.0),
      launch_base_(4.0),
      ga_gain_(0.9),
      leg_pen_scale_(1.2),
      impact_gain_(1e6),
      tof_ref_days_(500.0),
      soft_total_span_(9000.0),
      hard_pen_(5e3)
{
    setName("gtoc1");
    setFullName("GTOC1 surrogate (Earth-Venus-Earth-Jupiter-Saturn-asteroid TW229)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("space trajectory optimization benchmark");

    // 7 leg-duration variables, bounds loosely following the flyby sequence
    // (each leg progressively longer as the mission reaches the outer
    // solar system, matching the qualitative shape of the real benchmark).
    T_min_ = {50.0, 50.0, 50.0, 300.0, 300.0, 300.0, 300.0};
    T_max_ = {700.0, 700.0, 700.0, 1850.0, 2500.0, 2500.0, 2500.0};
}

void GTOC1::init(int /*dim*/) {
    // Force D = 8 (as in the real GTOC1 benchmark) and set bounds
    Problem::init(8);

    Vec lo(8), hi(8);
    lo[0] = t0_min_; hi[0] = t0_max_;
    for (int i = 0; i < 7; ++i) {
        lo[i + 1] = T_min_[i];
        hi[i + 1] = T_max_[i];
    }
    setBounds(lo, hi);
}

double GTOC1::evaluate_core(const Vec& x) {
    double t0 = x[0];
    Vec T(7);
    for (int i = 0; i < 7; ++i) T[i] = x[i + 1];

    auto out = [](double v, double lo, double hi){ return (v < lo) || (v > hi); };
    double pen = 0.0;
    if (out(t0, t0_min_, t0_max_)) pen += hard_pen_;
    for (int i = 0; i < 7; ++i) {
        if (out(T[i], T_min_[i], T_max_[i])) pen += hard_pen_;
        T[i] = clamp(T[i], T_min_[i], T_max_[i]);
    }

    // 1) Launch cost surrogate (a fixed baseline, since t0 mainly affects
    //    ephemeris phasing which this surrogate does not model in detail)
    double launch_cost = launch_base_;

    // 2) Gravity-assist gains from the three inner flybys (Venus, Earth, Jupiter)
    auto ga_gain_of = [&](double T, double ref){
        double r = std::clamp(T / ref, 0.3, 3.0);
        return ga_gain_ * (1.0 / std::sqrt(r));
    };
    double gain = ga_gain_of(T[0], 200.0) + ga_gain_of(T[1], 200.0) + ga_gain_of(T[2], 200.0);

    // 3) Per-leg timing penalty (poorly-timed long/short legs cost efficiency)
    double leg_pen = 0.0;
    for (int i = 0; i < 7; ++i) {
        double r = std::clamp(T[i] / tof_ref_days_, 0.2, 6.0);
        leg_pen += leg_pen_scale_ / (1.0 + r);
    }

    // 4) Terminal impact benefit: grows with the relative speed built up
    //    over the final (Saturn -> asteroid) leg, but saturates.
    double rT7 = std::clamp(T[6] / 2000.0, 0.2, 4.0);
    double impact_benefit = impact_gain_ * (1.0 - std::exp(-1.0 / rT7));

    // 5) Combine into a surrogate "benefit" (semi-major-axis-change proxy),
    //    then negate for minimization.
    double benefit = impact_benefit + 2e5 * gain - 2e5 * leg_pen - 2e4 * launch_cost;

    // 6) Soft penalty for excessive total time of flight (reduces benefit)
    double total_days = 0.0;
    for (double v : T) total_days += v;
    if (total_days > soft_total_span_) {
        benefit -= 50.0 * (total_days - soft_total_span_);
    }

    double f = -benefit + pen;

    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void GTOC1::gradient_core(const Vec& x, Vec& g) {
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
