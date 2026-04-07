#include "ded2.h"

namespace optimsolution {

DED2::DED2()
    : U(9), T(24), Dv(U * 24),
      use_valve(true),
      use_losses(false),
      w_balance(0.08),
      w_ramp(0.03),
      w_bounds(20.0),
      B00(0.0)
{
    setName("ded2");
    setFullName("Dynamic Economic Dispatch - Case 2 (9-unit system)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("dynamic economic dispatch benchmark");

    // No known analytic optimum for DED -> no setKnownGlobalOptimum(...)
    set_default_data();
}

void DED2::set_default_data()
{
    // Quadratic cost coefficients
    a = {0.00050, 0.00060, 0.00055, 0.00040, 0.00045, 0.00050, 0.00048, 0.00052, 0.00050};
    b = {   7.0,     7.2,     6.8,     7.5,     7.3,     6.9,     7.4,     7.1,     7.0};
    c = {   150,      160,     140,     170,     165,     155,     160,     150,     152};

    // Limits
    Pmin = { 15, 20, 20, 25, 25, 30, 30, 35, 35 };
    Pmax = {120,130,110,140,135,150,145,160,155 };

    // Ramp limits
    UR   = { 50, 45, 40, 55, 50, 60, 50, 55, 50 };
    DR   = { 50, 45, 40, 55, 50, 60, 50, 55, 50 };

    // Valve-point terms
    e_vp = {100,120,110,100,120,130,110,120,100};
    f_vp = {0.041,0.040,0.039,0.042,0.040,0.038,0.041,0.039,0.040};

    // Default 24h demand
    Dload = {
        680,690,700,710,720,740,760,780,800,820,840,860,
        880,870,860,840,820,800,780,760,740,720,700,690
    };

    // Loss matrices (default OFF = zeros)
    B.assign(U, std::vector<double>(U, 0.0));
    B0.assign(U, 0.0);
    B00 = 0.0;
}

void DED2::synthesize_demand()
{
    if (T == 24)
        return;

    const double PI = 3.14159265358979323846;

    double sumPmin = 0.0, sumPmax = 0.0;
    for (int i = 0; i < U; ++i) {
        sumPmin += Pmin[i];
        sumPmax += Pmax[i];
    }

    double lo = sumPmin + 0.1 * (sumPmax - sumPmin);
    double hi = sumPmax - 0.1 * (sumPmax - sumPmin);

    double avg = 770.0;
    double amp = 120.0;

    Dload.resize(T);
    for (int t = 0; t < T; ++t) {
        double ph1 = 2.0 * PI * t / T;
        double ph2 = 2.0 * PI * t / T * 2.0 + 0.6;
        double val = avg + amp * std::sin(ph1) + 0.3 * amp * std::sin(ph2);
        Dload[t] = clamp(val, lo, hi);
    }
}

void DED2::build_bounds()
{
    Vec lo(Dv), hi(Dv);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            lo[idx(i,t)] = Pmin[i];
            hi[idx(i,t)] = Pmax[i];
        }
    }
    setBounds(lo, hi);
}

void DED2::init(int dim)
{
    if (dim > 0 && (dim % U) == 0) {
        T = dim / U;
        Dv = U * T;
    } else {
        T = 24;
        Dv = U * T;
    }

    Problem::init(Dv);

    synthesize_demand();
    build_bounds();
}

double DED2::evaluate_core(const Vec& x)
{
    // 1) Fuel cost (quadratic + optional smooth valve-point)
    double fuel = 0.0;

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            const double P = x[idx(i,t)];

            fuel += a[i] * P * P + b[i] * P + c[i];

            if (use_valve) {
                const double arg = f_vp[i] * (Pmin[i] - P);
                fuel += e_vp[i] * smooth_abs(std::sin(arg));
            }
        }
    }

    // 2) Power-balance penalties (supports losses if enabled)
    double pen_bal = 0.0;

    for (int t = 0; t < T; ++t) {
        double PL = 0.0;

        if (use_losses) {
            for (int i = 0; i < U; ++i)
                for (int j = 0; j < U; ++j)
                    PL += x[idx(i,t)] * B[i][j] * x[idx(j,t)];

            double tmp = 0.0;
            for (int i = 0; i < U; ++i)
                tmp += B0[i] * x[idx(i,t)];
            PL += tmp + B00;
        }

        double sumP = 0.0;
        for (int i = 0; i < U; ++i)
            sumP += x[idx(i,t)];

        const double diff = sumP - (Dload[t] + PL);
        pen_bal += w_balance * diff * diff;
    }

    // 3) Ramp-rate penalties
    double pen_ramp = 0.0;

    for (int i = 0; i < U; ++i) {
        for (int t = 1; t < T; ++t) {
            const double dP = x[idx(i,t)] - x[idx(i,t-1)];

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

    // 4) Soft bound penalties
    double pen_bounds = 0.0;

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < U; ++i) {
            const double P = x[idx(i,t)];

            if (P < Pmin[i]) {
                const double d = Pmin[i] - P;
                pen_bounds += w_bounds * d * d;
            }
            else if (P > Pmax[i]) {
                const double d = P - Pmax[i];
                pen_bounds += w_bounds * d * d;
            }
        }
    }

    // 5) Tiny regularizer
    double reg = 0.0;
    for (int k = 0; k < Dv; ++k)
        reg += 1e-6 * x[k];

    double cost = fuel + pen_bal + pen_ramp + pen_bounds + reg;
    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost))
        cost = 1e12;

    return cost;
}

void DED2::gradient_core(const Vec& x, Vec& g)
{
    g.assign(x.size(), 0.0);

    const double rel = 1e-6;
    const double f0  = evaluate_core(x);

    Vec xh = x;
    for (int k = 0; k < (int)x.size(); ++k) {
        const double h = std::max(1e-6, std::abs(x[k]) * rel);
        xh[k] += h;
        const double f1 = evaluate_core(xh);
        g[k] = (f1 - f0) / h;
        xh[k] = x[k];
    }
}

} // namespace optimsolution
