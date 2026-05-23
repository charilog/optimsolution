#include "smartportenergy.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace optimsolution {

namespace {
constexpr double PI = 3.14159265358979323846;

inline double sqr(double v) { return v * v; }

inline double clamp01(double v)
{
    return std::max(0.0, std::min(1.0, v));
}

inline double soft_clip(double v, double lo, double hi)
{
    return std::max(lo, std::min(hi, v));
}

inline double sin2(double v)
{
    const double s = std::sin(v);
    return s * s;
}

inline double logistic(double z)
{
    if (z > 40.0) return 1.0;
    if (z < -40.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

inline double softplus(double z)
{
    if (z > 40.0) return z;
    if (z < -40.0) return std::exp(z);
    return std::log1p(std::exp(z));
}

// Converter scheduling in practical systems often induces quasi-discrete operating bands.
// This mapping creates broad neutral/flat regions with light ripples inside each band,
// making early progress much less informative for an optimizer.
inline double banded_control(double raw, double phase)
{
    raw = clamp01(raw);

    if (raw < 0.12)
        return 0.02 * raw;
    if (raw < 0.32)
        return clamp01(0.16 + 0.010 * std::sin(17.0 * raw + phase));
    if (raw < 0.54)
        return clamp01(0.43 + 0.012 * std::sin(19.0 * raw + phase));
    if (raw < 0.77)
        return clamp01(0.70 + 0.013 * std::sin(21.0 * raw + phase));
    return clamp01(0.91 + 0.010 * std::sin(23.0 * raw + phase));
}

inline double saturating_penalty(double z, double scale)
{
    return sqr(z) / (scale + sqr(z));
}

} // namespace

SmartPortEnergy::SmartPortEnergy()
    : smax_(0.0), hmax_(0.0), s0_(0.0), h0_(0.0),
      starget_(0.0), htarget_(0.0), seasonal_budget_(0.0)
{
    setName("smartportenergy");
    setFullName("Hybrid Smart Port Shore-Power Scheduling Problem");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("real-world inspired energy-infrastructure optimization problem");
}

void SmartPortEnergy::init(int dim)
{
    if (dim < 6)
        dim = 6;

    Problem::init(dim);

    Vec l(dim, 0.0);
    Vec u(dim, 1.0);
    setBounds(l, u);

    demand_.assign(dim, 0.0);
    renewable_.assign(dim, 0.0);
    tariff_.assign(dim, 0.0);
    criticality_.assign(dim, 0.0);
    pmax_.assign(dim, 0.0);
    bcap_.assign(dim, 0.0);
    hcap_.assign(dim, 0.0);
    peakcap_.assign(dim, 0.0);
    pref_.assign(dim, 0.0);

    const int D = dim;
    for (int i = 0; i < D; ++i) {
        const double t = static_cast<double>(i + 1);
        const double tau = t / static_cast<double>(D);

        demand_[i] = 58.0
                   + 16.0 * std::sin(2.0 * PI * tau)
                   + 8.0  * std::sin(7.0 * PI * tau + 0.45)
                   + 6.0  * sin2(5.0 * PI * tau + 0.20);

        renewable_[i] = 12.0
                      + 7.0 * sin2(4.0 * PI * tau + 0.30)
                      + 3.0 * std::sin(11.0 * PI * tau + 0.80);
        renewable_[i] = std::max(2.0, renewable_[i]);

        tariff_[i] = 0.17
                   + 0.11 * sin2(3.0 * PI * tau + 0.90)
                   + 0.03 * sin2(13.0 * PI * tau + 0.15);

        criticality_[i] = 1.0
                        + 0.40 * sin2(5.0 * PI * tau + 0.10)
                        + 0.25 * sin2(9.0 * PI * tau + 1.10);

        pmax_[i] = 34.0
                 + 10.0 * sin2(2.0 * PI * tau + 0.25)
                 + 5.0  * sin2(8.0 * PI * tau + 0.65);

        bcap_[i] = 12.0
                 + 4.0 * sin2(6.0 * PI * tau + 0.55)
                 + 1.8 * std::sin(10.0 * PI * tau + 0.35);
        bcap_[i] = std::max(5.0, bcap_[i]);

        hcap_[i] = 9.0
                 + 3.0 * sin2(7.0 * PI * tau + 0.25)
                 + 1.4 * std::sin(12.0 * PI * tau + 1.20);
        hcap_[i] = std::max(4.0, hcap_[i]);

        peakcap_[i] = 46.0
                    + 4.0 * sin2(2.0 * PI * tau + 0.60)
                    + 2.5 * sin2(9.0 * PI * tau + 0.10);

        pref_[i] = 0.42
                 + 0.18 * std::sin(2.0 * PI * tau + 0.50)
                 + 0.10 * std::sin(8.0 * PI * tau + 0.90);
        pref_[i] = soft_clip(pref_[i], 0.05, 0.95);
    }

    smax_ = 52.0 + 0.35 * static_cast<double>(D);
    hmax_ = 46.0 + 0.25 * static_cast<double>(D);
    s0_ = 0.62 * smax_;
    h0_ = 0.58 * hmax_;
    starget_ = 0.56 * smax_;
    htarget_ = 0.50 * hmax_;

    seasonal_budget_ = 0.54;
    for (int i = 0; i < D; ++i)
        seasonal_budget_ += 0.44 * pmax_[i];
}

double SmartPortEnergy::safe_x(const Vec& x, int i) const
{
    if (i < 0)
        return x.front();
    if (i >= static_cast<int>(x.size()))
        return x.back();
    return x[i];
}

double SmartPortEnergy::evaluate_with_point(const Vec& x) const
{
    const int D = static_cast<int>(x.size());

    double soc = s0_;
    double h2 = h0_;

    // Long-memory latent states create delayed consequences and reduce the amount of
    // useful local information available early in the search.
    double reserve_debt = 0.12 * (smax_ + 0.6 * hmax_);
    double maintenance_stress = 0.08 * (smax_ + hmax_);
    double mode_lock = 0.0;
    double corridor_debt = 0.0;
    double compliance_debt = 0.0;

    double energy_cost = 0.0;
    double emergency_cost = 0.0;
    double degradation_cost = 0.0;
    double curtailment_cost = 0.0;
    double switching_cost = 0.0;
    double peak_cost = 0.0;
    double rugged_cost = 0.0;
    double deferred_cost = 0.0;

    double cumulative_dispatch = 0.0;
    double cumulative_emergency = 0.0;
    double prev_dispatch = 0.0;
    double prev_total_draw = 0.0;
    double prev_u = 0.0;

    const int checkpoint_step = std::max(4, D / 5);

    for (int i = 0; i < D; ++i) {
        const double tau = static_cast<double>(i + 1) / static_cast<double>(D);
        const double xi  = clamp01(x[i]);
        const double xm1 = clamp01(safe_x(x, i - 1));
        const double xm2 = clamp01(safe_x(x, i - 2));
        const double xm3 = clamp01(safe_x(x, i - 3));
        const double xp1 = clamp01(safe_x(x, i + 1));

        const double raw_u = clamp01(0.46 * xi + 0.22 * xm1 + 0.14 * xm2 + 0.10 * xm3 + 0.08 * xp1);
        const double u = banded_control(raw_u, 0.35 * static_cast<double>(i + 1) + 1.8 * xm1 - 1.2 * xm2);
        const double dispatch = pmax_[i] * u;

        const double center = soft_clip(
            0.18
          + 0.44 * pref_[i]
          + 0.08 * std::sin(6.0 * PI * tau + 0.40)
          + 0.05 * std::sin(13.0 * PI * tau + 0.20),
            0.08, 0.92);
        const double corridor_half_width = 0.055 + 0.012 * sin2(5.0 * PI * tau + 0.30);
        const double corridor_err = std::fabs(u - center) - corridor_half_width;
        corridor_debt = 0.93 * corridor_debt + sqr(std::max(0.0, corridor_err));

        const double eff_b = 0.95 - 0.15 * logistic(0.18 * maintenance_stress - 4.6)
                                   - 0.06 * sin2(7.5 * u + 0.8 * xm1 + 0.3 * tau * D);
        const double eff_h = 0.90 - 0.13 * logistic(0.16 * maintenance_stress - 4.2)
                                   - 0.05 * sin2(6.8 * u + 0.5 * xm2 + 0.4 * tau * D);
        const double eta_charge = soft_clip(eff_b, 0.70, 0.96);
        const double eta_discharge = soft_clip(0.97 * eff_b, 0.68, 0.94);
        const double eta_el = soft_clip(eff_h, 0.68, 0.92);
        const double eta_fc = soft_clip(0.96 * eff_h, 0.66, 0.90);

        const double reserve_norm = reserve_debt / std::max(1.0, smax_ + hmax_);
        const double mb = logistic(2.9 * (2.0 * u - 1.0)
                                 + 1.4 * (xm1 - xm2)
                                 - 2.0 * reserve_norm
                                 + 0.9 * mode_lock
                                 + 0.35 * std::sin(8.0 * xi + 3.0 * xm1));
        const double mh = logistic(2.5 * (2.0 * u - 1.0)
                                 - 1.2 * (xm1 - xm3)
                                 - 1.6 * reserve_norm
                                 - 0.8 * mode_lock
                                 + 0.30 * std::cos(7.0 * xi - 2.0 * xm2));

        const double shortage = std::max(0.0, demand_[i] - renewable_[i] - dispatch);
        const double surplus  = std::max(0.0, renewable_[i] + dispatch - demand_[i]);

        const double bdis_cap = std::min(bcap_[i], soc);
        const double hdis_cap = std::min(hcap_[i], h2);
        const double reserve_floor_s = 0.10 * smax_ + 0.12 * criticality_[i];
        const double reserve_floor_h = 0.08 * hmax_ + 0.10 * criticality_[i];

        const double discharge_b = std::min(shortage,
            std::max(0.0, mb * std::max(0.0, bdis_cap - reserve_floor_s)));
        const double rem1 = shortage - discharge_b;

        const double discharge_h = std::min(rem1,
            std::max(0.0, mh * std::max(0.0, hdis_cap - reserve_floor_h)));
        const double rem2 = rem1 - discharge_h;
        const double emergency_grid = std::max(0.0, rem2);

        const double bchg_cap = std::max(0.0, smax_ - soc);
        const double hchg_cap = std::max(0.0, hmax_ - h2);
        const double charge_b = std::min(surplus,
            std::max(0.0, (0.18 + 0.82 * (1.0 - mb)) * std::min(bcap_[i], bchg_cap)));
        const double rems = surplus - charge_b;
        const double charge_h = std::min(rems,
            std::max(0.0, (0.20 + 0.80 * (1.0 - mh)) * std::min(hcap_[i], hchg_cap)));
        const double curtail = std::max(0.0, rems - charge_h);

        const double total_draw = dispatch + emergency_grid;
        const double rolling_peak = 0.60 * total_draw + 0.28 * prev_total_draw + 0.12 * prev_dispatch;

        const double batt_cycle = charge_b + 1.15 * discharge_b;
        const double h2_cycle = charge_h + 1.10 * discharge_h;
        maintenance_stress = 0.92 * maintenance_stress
                           + 0.18 * batt_cycle
                           + 0.14 * h2_cycle
                           + 0.55 * sqr(u - prev_u)
                           + 0.10 * logistic(5.0 * (rolling_peak / std::max(1.0, peakcap_[i]) - 0.95));

        soc = std::clamp(0.965 * soc
                       + eta_charge * charge_b
                       - discharge_b / std::max(0.60, eta_discharge)
                       - 0.010 * soc
                       - 0.030 * logistic(0.20 * maintenance_stress - 5.0) * soc,
                         0.0, smax_);

        h2 = std::clamp(0.975 * h2
                      + eta_el * charge_h
                      - discharge_h / std::max(0.58, eta_fc)
                      - 0.006 * h2
                      - 0.022 * logistic(0.18 * maintenance_stress - 4.8) * h2,
                        0.0, hmax_);

        const double reserve_target = 0.12 * smax_ + 0.10 * hmax_
                                    + 0.06 * demand_[i] + 0.55 * criticality_[i];
        const double reserve_gap = std::max(0.0, reserve_target - soc - 0.65 * h2);
        reserve_debt = 0.94 * reserve_debt + reserve_gap;

        mode_lock = 0.90 * mode_lock
                  + 0.28 * std::tanh(3.6 * (charge_b + charge_h - discharge_b - discharge_h))
                  + 0.18 * std::tanh(5.0 * (u - 0.5));

        energy_cost += tariff_[i] * (dispatch + 2.6 * emergency_grid)
                     + 0.012 * dispatch * dispatch
                     + 0.008 * dispatch * reserve_norm * demand_[i];

        emergency_cost += (110.0 + 22.0 * criticality_[i]) * emergency_grid * emergency_grid
                        + 8.0 * emergency_grid * reserve_norm;

        degradation_cost += 0.78 * batt_cycle
                          + 0.58 * h2_cycle
                          + 0.016 * sqr(batt_cycle + 0.8 * h2_cycle)
                          + 0.11 * sqr(std::max(0.0, maintenance_stress - 0.26 * (smax_ + hmax_)));

        curtailment_cost += 1.05 * curtail * curtail
                          + 0.08 * curtail * reserve_norm * reserve_norm;

        if (i > 0) {
            switching_cost += 0.70 * sqr(dispatch - prev_dispatch)
                            + 1.25 * sqr(u - prev_u)
                            + 0.30 * softplus(8.0 * (std::fabs(u - prev_u) - 0.18));
        }

        peak_cost += 12.0 * sqr(std::max(0.0, rolling_peak - peakcap_[i]))
                   + 1.1 * saturating_penalty(rolling_peak - 0.92 * peakcap_[i], 4.0);

        rugged_cost += 5.5 * sin2(0.18 * dispatch + 0.09 * (i + 1)
                                 + 1.2 * xm1 - 0.6 * xm2 + 0.5 * reserve_norm)
                     + 4.2 * sin2(0.052 * (dispatch + 1.8 * emergency_grid)
                                 * (1.0 + 0.045 * maintenance_stress)
                                 + 0.40 * std::cos(6.0 * u + 2.4 * xm1))
                     + 1.8 * saturating_penalty(u - center, 0.0015);

        // Weak immediate penalty, strong deferred penalty: this slows down convergence
        // because locally the landscape provides only limited information.
        deferred_cost += 0.20 * corridor_debt
                       + 0.10 * softplus(7.0 * (reserve_norm - 0.35));

        cumulative_dispatch += dispatch;
        cumulative_emergency += emergency_grid;

        if (((i + 1) % checkpoint_step) == 0 || i == D - 1) {
            const double frac = static_cast<double>(i + 1) / static_cast<double>(D);
            const double target = seasonal_budget_ * std::pow(frac, 1.08)
                                * (0.93 + 0.10 * sin2(3.0 * PI * frac + 0.45));
            const double envelope = 3.0 + 0.020 * seasonal_budget_ + 0.6 * (1.0 - frac) * seasonal_budget_ / D;
            const double dev = std::fabs(cumulative_dispatch - target) - envelope;
            compliance_debt = 0.88 * compliance_debt + sqr(std::max(0.0, dev));
        }

        prev_dispatch = dispatch;
        prev_total_draw = total_draw;
        prev_u = u;
    }

    const double terminal_gap = sqr(soc - starget_) + 0.85 * sqr(h2 - htarget_);
    const double terminal_cost = 28.0 * terminal_gap
                               + 90.0 * reserve_debt * reserve_debt / std::max(1.0, smax_ + hmax_)
                               + 55.0 * compliance_debt
                               + 48.0 * corridor_debt
                               + 0.95 * sqr(std::max(0.0, maintenance_stress - 0.22 * (smax_ + hmax_)));

    const double budget_penalty = 18.0 * sqr(std::max(0.0, cumulative_dispatch - seasonal_budget_));
    const double emergency_tail_penalty = 35.0 * sqr(std::max(0.0, cumulative_emergency - 0.10 * seasonal_budget_));

    // Narrow final reward corridor: only highly coordinated trajectories receive it.
    const double alignment_score = corridor_debt + 0.8 * compliance_debt
                                 + 0.015 * reserve_debt + 0.010 * maintenance_stress;
    const double coordination_reward = 160.0 * std::exp(-8.0 * alignment_score);

    return energy_cost
         + emergency_cost
         + degradation_cost
         + curtailment_cost
         + switching_cost
         + peak_cost
         + rugged_cost
         + deferred_cost
         + terminal_cost
         + budget_penalty
         + emergency_tail_penalty
         - coordination_reward;
}

double SmartPortEnergy::evaluate_core(const Vec& x)
{
    return evaluate_with_point(x);
}

void SmartPortEnergy::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i) {
        Vec xp = x;
        Vec xm = x;

        const double lo = 0.0;
        const double hi = 1.0;
        const double eps = 1e-6 * (hi - lo);

        xp[i] = std::min(hi, x[i] + eps);
        xm[i] = std::max(lo, x[i] - eps);

        if (xp[i] == xm[i]) {
            g[i] = 0.0;
            continue;
        }

        const double fp = evaluate_with_point(xp);
        const double fm = evaluate_with_point(xm);
        g[i] = (fp - fm) / (xp[i] - xm[i]);
    }
}

} // namespace optimsolution
