#include "cassini.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Cassini::Cassini()
{
    setName("cassini");
    setFullName("Cassini interplanetary transfer timing problem");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("real-world trajectory optimization problem");

    // Problem has no known analytic optimum → do not call setKnownGlobalOptimum()

    // Default mission window parameters (from old version)
    tmin_ = {   0.0,  300.0,  600.0,  900.0, 1500.0, 2200.0 };
    tmax_ = {1000.0, 1600.0, 2200.0, 2600.0, 3200.0, 4200.0 };

    mu_    = { 200.0, 700.0, 1100.0, 1500.0, 2200.0, 3000.0 };
    sigma_ = { 120.0, 150.0, 180.0, 180.0, 240.0, 300.0 };

    mingap_ = { 80.0, 120.0, 150.0, 300.0, 400.0 };
}

void Cassini::init(int /*dim*/)
{
    // This problem has fixed dimension = 6
    Problem::init(K);

    Vec lo(K, 0.0), hi(K, 1.0);
    setBounds(lo, hi);
}

// User-supplied overrides
void Cassini::setTimeWindows(const std::vector<double>& tmin,
                             const std::vector<double>& tmax)
{
    if (tmin.size() == K && tmax.size() == K)
        for (int i = 0; i < K; ++i) {
            tmin_[i] = tmin[i];
            tmax_[i] = tmax[i];
        }
}

void Cassini::setPreferredTimes(const std::vector<double>& mu,
                                const std::vector<double>& sigma)
{
    if (mu.size() == K && sigma.size() == K)
        for (int i = 0; i < K; ++i) {
            mu_[i] = mu[i];
            sigma_[i] = std::max(1e-9, sigma[i]);
        }
}

void Cassini::setMinGaps(const std::vector<double>& mingap)
{
    if (mingap.size() == K - 1)
        for (int i = 0; i < K - 1; ++i)
            mingap_[i] = mingap[i];
}

void Cassini::setDurationLimits(double Tmin, double Tmax)
{
    if (Tmin > 0.0 && Tmax > Tmin) {
        Tmin_ = Tmin;
        Tmax_ = Tmax;
    }
}

void Cassini::setWeights(double w_pref, double w_gap,
                         double w_dur, double w_smooth)
{
    w_pref_   = w_pref;
    w_gap_    = w_gap;
    w_dur_    = w_dur;
    w_smooth_ = w_smooth;
}

// x ∈ [0,1]^6 → t in mission windows
inline void Cassini::map_to_times(const Vec& x,
                                  std::array<double, K>& t,
                                  std::array<double, K>& dt_dx) const
{
    for (int i = 0; i < K; ++i) {
        const double span = tmax_[i] - tmin_[i];
        dt_dx[i] = span;
        t[i] = tmin_[i] + span * x[i];
    }
}

double Cassini::evaluate_core(const Vec& x)
{
    std::array<double, K> t{};
    std::array<double, K> dt_dx{};
    map_to_times(x, t, dt_dx);

    // 1) Preference term
    double cost_pref = 0.0;
    for (int i = 0; i < K; ++i) {
        const double z = (t[i] - mu_[i]) / sigma_[i];
        cost_pref += z * z;
    }
    cost_pref *= w_pref_;

    // 2) Gap violations
    double cost_gap = 0.0;
    for (int i = 0; i < K - 1; ++i) {
        const double v = mingap_[i] - (t[i+1] - t[i]);
        if (v > 0.0)
            cost_gap += v * v;
    }
    cost_gap *= w_gap_;

    // 3) Mission duration penalty: T = t5 - t0
    const double T = t[K - 1] - t[0];
    double cost_dur = 0.0;
    if (T < Tmin_)
        cost_dur += (Tmin_ - T) * (Tmin_ - T);
    if (T > Tmax_)
        cost_dur += (T - Tmax_) * (T - Tmax_);
    cost_dur *= w_dur_;

    // 4) Smoothness term (second differences)
    double cost_smooth = 0.0;
    for (int i = 1; i < K - 1; ++i) {
        const double d2 = t[i+1] - 2.0 * t[i] + t[i-1];
        cost_smooth += d2 * d2;
    }
    cost_smooth *= w_smooth_;

    return cost_pref + cost_gap + cost_dur + cost_smooth;
}

void Cassini::gradient_core(const Vec& x, Vec& g)
{
    g.assign(K, 0.0);

    std::array<double, K> t{};
    std::array<double, K> dt_dx{};
    map_to_times(x, t, dt_dx);

    std::array<double, K> gt{};
    gt.fill(0.0);

    // 1) Preference
    for (int i = 0; i < K; ++i) {
        const double z = (t[i] - mu_[i]) / sigma_[i];
        gt[i] += w_pref_ * (2.0 * z / sigma_[i]);
    }

    // 2) Gap penalties
    for (int i = 0; i < K - 1; ++i) {
        const double v = mingap_[i] - (t[i+1] - t[i]);
        if (v > 0.0) {
            const double d = 2.0 * v * w_gap_;
            gt[i]     += d;
            gt[i + 1] -= d;
        }
    }

    // 3) Duration
    const double T = t[K - 1] - t[0];
    if (T < Tmin_) {
        const double d = -2.0 * (Tmin_ - T) * w_dur_;
        gt[0]     += -d;
        gt[K - 1] +=  d;
    }
    if (T > Tmax_) {
        const double d =  2.0 * (T - Tmax_) * w_dur_;
        gt[0]     += -d;
        gt[K - 1] +=  d;
    }

    // 4) Smoothness
    for (int i = 1; i < K - 1; ++i) {
        const double d2 = t[i+1] - 2.0 * t[i] + t[i-1];
        const double c = 2.0 * d2 * w_smooth_;
        gt[i - 1] +=  c;
        gt[i]     += -2.0 * c;
        gt[i + 1] +=  c;
    }

    // Chain rule: df/dx_i = df/dt_i * dt_i/dx_i
    for (int i = 0; i < K; ++i)
        g[i] = gt[i] * dt_dx[i];
}

} // namespace optimsolution
