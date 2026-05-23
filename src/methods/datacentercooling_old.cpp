#include "datacentercooling.h"
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

inline double softplus(double z)
{
    if (z > 40.0) return z;
    if (z < -40.0) return std::exp(z);
    return std::log1p(std::exp(z));
}

inline double logistic(double z)
{
    if (z > 40.0) return 1.0;
    if (z < -40.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

inline double banded_control(double raw, double phase)
{
    raw = clamp01(raw);

    if (raw < 0.10)
        return 0.02 * raw;
    if (raw < 0.28)
        return clamp01(0.15 + 0.010 * std::sin(19.0 * raw + phase));
    if (raw < 0.49)
        return clamp01(0.41 + 0.011 * std::sin(21.0 * raw + phase));
    if (raw < 0.71)
        return clamp01(0.68 + 0.012 * std::sin(23.0 * raw + phase));
    if (raw < 0.88)
        return clamp01(0.87 + 0.010 * std::sin(25.0 * raw + phase));
    return clamp01(0.95 + 0.008 * std::sin(29.0 * raw + phase));
}

inline double sat_penalty(double z, double scale)
{
    return sqr(z) / (scale + sqr(z));
}

} // namespace

DataCenterCooling::DataCenterCooling()
    : t0_(0.0), m0_(0.0), ttarget_(0.0), mtarget_(0.0), carbon_budget_(0.0)
{
    setName("datacentercooling");
    setFullName("Adaptive Data Center Thermal Orchestration Problem");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("real-world inspired thermal-infrastructure optimization problem");
}

void DataCenterCooling::init(int dim)
{
    if (dim < 6)
        dim = 6;

    Problem::init(dim);

    Vec l(dim, 0.0);
    Vec u(dim, 1.0);
    setBounds(l, u);

    workload_.assign(dim, 0.0);
    ambient_.assign(dim, 0.0);
    tariff_.assign(dim, 0.0);
    humidity_.assign(dim, 0.0);
    tlimit_.assign(dim, 0.0);
    coolcap_.assign(dim, 0.0);
    flowcap_.assign(dim, 0.0);
    pref_.assign(dim, 0.0);
    criticality_.assign(dim, 0.0);

    const int D = dim;
    for (int i = 0; i < D; ++i) {
        const double tau = static_cast<double>(i + 1) / static_cast<double>(D);

        workload_[i] = 0.62
                     + 0.18 * std::sin(2.0 * PI * tau + 0.35)
                     + 0.09 * std::sin(9.0 * PI * tau + 0.70)
                     + 0.06 * sin2(15.0 * PI * tau + 0.15);
        workload_[i] = soft_clip(workload_[i], 0.28, 0.96);

        ambient_[i] = 28.0
                    + 5.2 * std::sin(2.0 * PI * tau + 0.40)
                    + 2.1 * std::sin(7.0 * PI * tau + 1.10)
                    + 1.8 * sin2(13.0 * PI * tau + 0.20);

        tariff_[i] = 0.13
                   + 0.08 * sin2(3.0 * PI * tau + 0.80)
                   + 0.03 * sin2(11.0 * PI * tau + 0.25);

        humidity_[i] = 0.42
                     + 0.11 * std::sin(2.0 * PI * tau + 1.10)
                     + 0.08 * sin2(10.0 * PI * tau + 0.35);
        humidity_[i] = soft_clip(humidity_[i], 0.24, 0.76);

        tlimit_[i] = 23.0
                   + 0.9 * sin2(4.0 * PI * tau + 0.20)
                   + 0.7 * sin2(9.0 * PI * tau + 1.00);

        coolcap_[i] = 8.6
                    + 1.6 * sin2(5.0 * PI * tau + 0.55)
                    + 0.8 * std::sin(12.0 * PI * tau + 0.90);
        coolcap_[i] = std::max(5.8, coolcap_[i]);

        flowcap_[i] = 4.6
                    + 0.9 * sin2(6.0 * PI * tau + 0.25)
                    + 0.5 * std::sin(10.0 * PI * tau + 0.60);
        flowcap_[i] = std::max(3.0, flowcap_[i]);

        pref_[i] = 0.38
                 + 0.16 * std::sin(2.0 * PI * tau + 0.65)
                 + 0.09 * std::sin(8.0 * PI * tau + 0.50);
        pref_[i] = soft_clip(pref_[i], 0.06, 0.94);

        criticality_[i] = 1.0
                        + 0.45 * sin2(5.0 * PI * tau + 0.15)
                        + 0.20 * sin2(12.0 * PI * tau + 0.95);
    }

    t0_ = 24.3;
    m0_ = 0.36;
    ttarget_ = 22.4;
    mtarget_ = 0.40;

    carbon_budget_ = 0.0;
    for (int i = 0; i < D; ++i)
        carbon_budget_ += 2.85 + 1.55 * workload_[i];
}

double DataCenterCooling::safe_x(const Vec& x, int i) const
{
    if (i < 0)
        return x.front();
    if (i >= static_cast<int>(x.size()))
        return x.back();
    return x[i];
}

double DataCenterCooling::evaluate_with_point(const Vec& x) const
{
    const int D = static_cast<int>(x.size());

    double temp = t0_;
    double moisture = m0_;
    double hotspot_debt = 0.18;
    double recirc_debt = 0.12;
    double wear_debt = 0.10;
    double comfort_debt = 0.08;
    double mode_lock = 0.0;

    double energy_cost = 0.0;
    double hotspot_cost = 0.0;
    double moisture_cost = 0.0;
    double switching_cost = 0.0;
    double rugged_cost = 0.0;
    double deferred_cost = 0.0;
    double reliability_cost = 0.0;

    double cumulative_power = 0.0;
    double prev_u = 0.0;
    double prev_cooling = 0.0;
    double prev_temp = temp;

    const int checkpoint_step = std::max(4, D / 4);

    for (int i = 0; i < D; ++i) {
        const double tau = static_cast<double>(i + 1) / static_cast<double>(D);
        const double xi  = clamp01(x[i]);
        const double xm1 = clamp01(safe_x(x, i - 1));
        const double xm2 = clamp01(safe_x(x, i - 2));
        const double xm3 = clamp01(safe_x(x, i - 3));
        const double xp1 = clamp01(safe_x(x, i + 1));

        const double raw_u = clamp01(0.43 * xi + 0.21 * xm1 + 0.16 * xm2 + 0.10 * xm3 + 0.10 * xp1);
        const double u = banded_control(raw_u, 0.40 * static_cast<double>(i + 1) + 1.6 * xm1 - 1.1 * xm2);

        const double fan = flowcap_[i] * (0.22 + 0.78 * u);
        const double cooling = coolcap_[i] * (0.10 + 0.90 * u);

        const double corridor_center = soft_clip(
            0.17
          + 0.50 * pref_[i]
          + 0.07 * std::sin(7.0 * PI * tau + 0.25)
          + 0.05 * std::sin(15.0 * PI * tau + 0.65),
            0.06, 0.94);
        const double corridor_half_width = 0.050 + 0.015 * sin2(6.0 * PI * tau + 0.45);
        const double corridor_err = std::fabs(u - corridor_center) - corridor_half_width;
        comfort_debt = 0.92 * comfort_debt + sqr(std::max(0.0, corridor_err));

        const double effective_load = workload_[i]
                                    * (1.0 + 0.40 * recirc_debt)
                                    * (1.0 + 0.14 * sin2(9.0 * PI * u + 1.20 * tau));

        const double recirc = 0.14
                            + 0.18 * logistic(2.8 * (effective_load - 0.70) + 1.9 * (0.45 - u))
                            + 0.05 * sin2(13.0 * PI * u + 0.70 * static_cast<double>(i + 1));

        const double dtemp =
            0.58 * (ambient_[i] - temp)
          + 9.8  * effective_load
          + 1.35 * recirc
          + 1.80 * hotspot_debt
          - 1.16 * cooling
          - 0.19 * fan;

        double next_temp = temp + 0.15 * dtemp;
        next_temp += 0.38 * std::sin(8.0 * PI * u + 3.0 * tau)
                   + 0.24 * std::sin(17.0 * PI * (u - prev_u) + 0.50);
        next_temp = soft_clip(next_temp, 16.0, 40.0);

        const double supply_bias = 0.03 + 0.12 * u + 0.02 * std::sin(10.0 * PI * tau + 0.30);
        double next_moisture = 0.80 * moisture + 0.20 * humidity_[i] - 0.12 * supply_bias;
        next_moisture += 0.015 * std::sin(11.0 * PI * u + 0.90 * tau)
                       - 0.010 * std::sin(17.0 * PI * prev_u + 0.40);
        next_moisture = soft_clip(next_moisture, 0.08, 0.82);

        const double dewpoint = 7.5 + 18.5 * next_moisture + 1.3 * humidity_[i];
        const double dew_risk = std::max(0.0, dewpoint + 1.1 - next_temp);
        const double thermal_excess = std::max(0.0, next_temp - tlimit_[i]);
        const double thermal_short = std::max(0.0, ttarget_ - next_temp - 0.8);

        hotspot_debt = 0.86 * hotspot_debt
                     + 0.30 * thermal_excess
                     + 0.12 * sat_penalty(next_temp - tlimit_[i], 1.0)
                     + 0.08 * sin2(5.0 * PI * tau + 6.0 * u);

        recirc_debt = 0.90 * recirc_debt
                    + 0.20 * std::max(0.0, recirc - 0.22)
                    + 0.08 * sqr(u - pref_[i]);

        wear_debt = 0.94 * wear_debt
                  + 0.20 * sqr(u - prev_u)
                  + 0.08 * sqr(cooling - prev_cooling)
                  + 0.05 * sqr(next_temp - prev_temp);

        mode_lock = 0.88 * mode_lock + std::fabs(u - prev_u);

        const double cooling_power = 2.4 + 0.75 * cooling + 0.16 * sqr(cooling) + 0.09 * sqr(fan);
        const double carbon = 0.72 * cooling_power + 0.24 * tariff_[i] * cooling_power;
        cumulative_power += cooling_power;

        energy_cost += tariff_[i] * cooling_power;
        hotspot_cost += 16.0 * criticality_[i] * sqr(thermal_excess)
                      + 7.0  * criticality_[i] * sqr(thermal_short);
        moisture_cost += 26.0 * sqr(dew_risk)
                       + 8.0  * sqr(std::max(0.0, 0.18 - next_moisture))
                       + 7.0  * sqr(std::max(0.0, next_moisture - 0.62));
        switching_cost += 2.0 * sqr(u - prev_u)
                        + 1.2 * sqr(cooling - prev_cooling)
                        + 0.7 * mode_lock * sqr(u - prev_u);
        rugged_cost += 1.4 * sin2(10.0 * PI * u + 3.0 * tau)
                     + 1.1 * sin2(17.0 * PI * (u - 0.5 * prev_u) + 1.5 * tau)
                     + 0.9 * sin2(7.0 * PI * clamp01(0.5 * xi + 0.3 * xm1 + 0.2 * xp1) + 0.4 * static_cast<double>(i + 1));

        deferred_cost += 8.0 * sqr(hotspot_debt)
                       + 3.4 * sqr(recirc_debt)
                       + 4.0 * sqr(comfort_debt)
                       + 2.6 * sqr(std::max(0.0, wear_debt - 0.65));
        reliability_cost += 2.2 * sqr(wear_debt)
                          + 2.5 * sqr(std::max(0.0, carbon - 5.8));

        if (((i + 1) % checkpoint_step) == 0 || i == D - 1) {
            const double target_ratio = soft_clip(
                0.30 + 0.52 * static_cast<double>(i + 1) / static_cast<double>(D)
                     + 0.04 * std::sin(2.0 * PI * tau + 0.40),
                0.18, 0.92);
            const double actual_ratio = cumulative_power / std::max(1e-12, carbon_budget_);
            const double checkpoint_err = actual_ratio - target_ratio;
            deferred_cost += 44.0 * sqr(checkpoint_err) / (0.030 + sqr(checkpoint_err));
        }

        prev_u = u;
        prev_cooling = cooling;
        prev_temp = temp;
        temp = next_temp;
        moisture = next_moisture;
    }

    const double terminal_penalty =
        16.0 * sqr(temp - ttarget_)
      + 18.0 * sqr(std::max(0.0, temp - (ttarget_ + 0.8)))
      + 20.0 * sqr(std::max(0.0, 0.22 - moisture))
      + 16.0 * sqr(std::max(0.0, moisture - 0.58))
      + 38.0 * sqr(hotspot_debt)
      + 16.0 * sqr(recirc_debt)
      + 12.0 * sqr(comfort_debt)
      + 18.0 * sqr(wear_debt);

    const double carbon_budget_penalty = 28.0 * sqr(std::max(0.0, cumulative_power - carbon_budget_));

    const double narrow_reward =
        160.0 * std::exp(-8.5 * sqr(temp - ttarget_) - 20.0 * sqr(moisture - mtarget_)
                         - 3.5 * sqr(hotspot_debt) - 2.5 * sqr(recirc_debt) - 1.8 * sqr(comfort_debt));

    return energy_cost
         + hotspot_cost
         + moisture_cost
         + switching_cost
         + rugged_cost
         + deferred_cost
         + reliability_cost
         + terminal_penalty
         + carbon_budget_penalty
         - narrow_reward;
}

double DataCenterCooling::evaluate_core(const Vec& x)
{
    return evaluate_with_point(x);
}

void DataCenterCooling::gradient_core(const Vec& x, Vec& g)
{
    const int D = static_cast<int>(x.size());
    g.assign(D, 0.0);

    if (D == 0)
        return;

    Vec xp = x;
    Vec xm = x;

    for (int i = 0; i < D; ++i) {
        const double xi = x[i];
        const double h = 1e-6 * (1.0 + std::fabs(xi));

        xp[i] = std::min(1.0, xi + h);
        xm[i] = std::max(0.0, xi - h);

        const double fp = evaluate_with_point(xp);
        const double fm = evaluate_with_point(xm);
        const double denom = xp[i] - xm[i];

        g[i] = (denom > std::numeric_limits<double>::epsilon())
             ? (fp - fm) / denom
             : 0.0;

        xp[i] = xi;
        xm[i] = xi;
    }
}

} // namespace optimsolution
