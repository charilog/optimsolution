#include "ded1.h"

namespace optimsolution {

DED1::DED1()
    : U(5), T(24), Dv(U * 24),
      w_balance(0.05), w_ramp(0.02), w_bounds(10.0)
{
    setName("ded1");
    setFullName("Dynamic Economic Dispatch - Case 1 (quadratic cost)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("dynamic economic dispatch benchmark");

    // No known analytic global optimum; leave as unknown (no setKnownGlobalOptimum)
    set_default_data();
}

void DED1::set_default_data()
{
    // Typical 5-unit quadratic costs & limits (same as old version)
    a = {0.00028, 0.00056, 0.00056, 0.00071, 0.00071};
    b = {   8.10,    8.10,    8.10,    7.74,    7.74};
    c = { 550.00,  309.00,  307.00,  240.00,  240.00};

    Pmin = { 10.0,  10.0,  10.0,  20.0,  20.0};
    Pmax = {125.0, 150.0, 135.0, 160.0, 160.0};

    // Ramp limits (MW per hour)
    UR   = { 50.0, 60.0, 40.0, 50.0, 50.0};
    DR   = { 50.0, 60.0, 40.0, 50.0, 50.0};

    // Default 24h demand (MW), within total capacity (sum Pmax = 730 MW)
    Dload = {
        420, 430, 440, 450, 460, 480, 500, 520, 540, 560, 580, 600,
        610, 600, 590, 570, 560, 550, 540, 530, 510, 490, 470, 450
    };
}

void DED1::make_demand_profile()
{
    if (T == 24) {
        // Already filled in set_default_data()
        return;
    }

    // If horizon differs, synthesize a smooth profile around similar average
    const double avg = 530.0;    // near the center of default profile
    const double amp = 90.0;     // swing
    const double PI  = 3.14159265358979323846;

    Dload.resize(T);
    for (int t = 0; t < T; ++t) {
        double phase = 2.0 * PI * static_cast<double>(t) / static_cast<double>(T);
        // morning-evening peaks with 2nd harmonic
        double val = avg
                   + amp * std::sin(phase)
                   + 0.3 * amp * std::sin(2.0 * phase + 0.7);
        // keep within feasible bounds
        val = clamp(val, 400.0, 650.0);
        Dload[t] = val;
    }
}

void DED1::build_bounds()
{
    // Replicate Pmin/Pmax per hour
    Vec lo(Dv), hi(Dv);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            lo[idx(i,t)] = Pmin[i];
            hi[idx(i,t)] = Pmax[i];
        }
    }
    setBounds(lo, hi);
}

void DED1::init(int dim)
{
    if (dim > 0 && (dim % U) == 0) {
        T  = dim / U;
        Dv = U * T;
    } else {
        T  = 24;
        Dv = U * T;
    }

    Problem::init(Dv);

    // Build/resize demand for this T
    make_demand_profile();

    // Set bounds from Pmin/Pmax replicated per hour
    build_bounds();
}

double DED1::evaluate_core(const Vec& x)
{
    // Use clamped values for all penalties/costs (as in old version)
    auto Px = [&](int i, int t) -> double {
        const double P = x[idx(i,t)];
        return clamp(P, Pmin[i], Pmax[i]);
    };

    // 1) Fuel cost
    double fuel = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            const double P = Px(i,t);
            fuel += a[i] * P * P + b[i] * P + c[i];
        }
    }

    // 2) Power balance penalties
    double pen_bal = 0.0;
    for (int t = 0; t < T; ++t) {
        double sumP = 0.0;
        for (int i = 0; i < U; ++i)
            sumP += Px(i,t);
        const double diff = sumP - Dload[t];
        pen_bal += w_balance * diff * diff;
    }

    // 3) Ramp penalties
    double pen_ramp = 0.0;
    for (int i = 0; i < U; ++i) {
        for (int t = 1; t < T; ++t) {
            const double Pnow = Px(i,t);
            const double Ppre = Px(i,t-1);
            const double dP   = Pnow - Ppre;

            if (dP > UR[i]) {
                const double over = dP - UR[i];
                pen_ramp += w_ramp * over * over;
            }
            if (-dP > DR[i]) {
                const double over = (-dP) - DR[i];
                pen_ramp += w_ramp * over * over;
            }
        }
    }

    // 4) Soft bound penalty (if optimizer proposes out-of-bounds values)
    double pen_bounds = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            const double P = x[idx(i,t)];
            if (P < Pmin[i]) {
                const double d = Pmin[i] - P;
                pen_bounds += w_bounds * d * d;
            } else if (P > Pmax[i]) {
                const double d = P - Pmax[i];
                pen_bounds += w_bounds * d * d;
            }
        }
    }

    double cost = fuel + pen_bal + pen_ramp + pen_bounds;
    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost))
        cost = 1e12;

    return cost;
}

void DED1::gradient_core(const Vec& x, Vec& g)
{
    g.assign(x.size(), 0.0);

    const double rel = 1e-6;
    const double f0  = evaluate_core(x);

    Vec xh = x;
    for (int k = 0; k < static_cast<int>(x.size()); ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);
        xh[k] += h;
        const double f1 = evaluate_core(xh);
        g[k] = (f1 - f0) / h;
        xh[k] = x[k];
    }
}

} // namespace optimsolution
