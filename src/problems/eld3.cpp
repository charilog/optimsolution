#include "eld3.h"

namespace optimsolution {

ELD3::ELD3()
    : NG(15),
      PD(2630.0),
      use_valve(true),
      use_losses(false),
      B00(0.0),
      w_balance(0.15),
      w_bounds(40.0),
      w_poz(1500.0)
{
    setName("eld3");
    setFullName("Economic Load Dispatch - 3 (CEC2011 15-unit instance)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("real-world power system benchmark");

    
    set_default_data();
}

void ELD3::set_default_data()
{
    // ---- Bounds (CEC2011 ELD instance, 15-unit) ----
    // Pmin/Pmax (MW) — faithful to uploaded reference
    Pmin = {150,150, 20, 20,150,135,135, 60, 25, 25, 20, 20, 25, 55, 55};
    Pmax = {455,455,130,130,470,460,465,300,162,160, 80, 85,115,115,115};

    // ---- Quadratic fuel & valve-point coefficients (15-unit dataset) ----
    a = {
        2.99e-4, 1.83e-4, 1.13e-3, 1.13e-3, 2.05e-4,
        3.01e-4, 3.64e-4, 3.38e-4, 8.07e-4, 1.20e-3,
        3.59e-3, 5.51e-3, 3.71e-4, 1.93e-3, 4.45e-3
    };

    b = {
        10.1,    10.2,    8.8,     8.8,     10.4,
        10.1,    9.8,     11.2,    11.2,    10.7,
        10.2,    9.9,     13.1,    12.1,    12.4
    };

    c = {
        671,     574,     374,     374,     461,
        630,     548,     227,     173,     175,
        186,     230,     225,     309,     323
    };

    e_vp = {300,200,150,150,150,150,150,150,150,100,100,100,100,100,100};
    f_vp = {
        0.035,0.042,0.042,0.063,0.063,
        0.063,0.063,0.063,0.063,0.084,
        0.084,0.084,0.084,0.084,0.084
    };

    // ---- POZ ----
    poz.clear();
    poz.assign(NG, {});

    // G2: [185,225], [305,335], [420,450]
    poz[1].push_back({185.0,225.0});
    poz[1].push_back({305.0,335.0});
    poz[1].push_back({420.0,450.0});

    // G5: [180,200], [305,335], [390,420]
    poz[4].push_back({180.0,200.0});
    poz[4].push_back({305.0,335.0});
    poz[4].push_back({390.0,420.0});

    // G6: [230,255], [365,395], [430,455]
    poz[5].push_back({230.0,255.0});
    poz[5].push_back({365.0,395.0});
    poz[5].push_back({430.0,455.0});

    // G12: [30,40], [55,65]
    poz[11].push_back({30.0,40.0});
    poz[11].push_back({55.0,65.0});

    // Losses OFF by default (zeros)
    B.assign(NG, std::vector<double>(NG, 0.0));
    B0.assign(NG, 0.0);
    B00 = 0.0;
}

void ELD3::build_bounds()
{
    Vec lo(NG), hi(NG);
    for (int i = 0; i < NG; ++i) {
        lo[i] = Pmin[i];
        hi[i] = Pmax[i];
    }
    setBounds(lo, hi);
}

void ELD3::init(int /*dim*/)
{
    // Fixed NG = 15 per reference
    Problem::init(NG);
    build_bounds();
}

double ELD3::evaluate_core(const Vec& x)
{
    // 1) Fuel (quadratic) + valve-point (smooth |sin|)
    double fuel = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        fuel += a[i] * P * P + b[i] * P + c[i];

        if (use_valve) {
            const double s = std::sin(f_vp[i] * (Pmin[i] - P));
            fuel += e_vp[i] * smooth_abs(s);
        }
    }

    // 2) Losses (optional; default OFF → PL = 0)
    double PL = 0.0;
    if (use_losses) {
        for (int i = 0; i < NG; ++i)
            for (int j = 0; j < NG; ++j)
                PL += x[i] * B[i][j] * x[j];

        double tmp = 0.0;
        for (int i = 0; i < NG; ++i)
            tmp += B0[i] * x[i];

        PL += tmp + B00;
    }

    // 3) Power balance penalty
    double sumP = 0.0;
    for (int i = 0; i < NG; ++i)
        sumP += x[i];

    const double bal         = sumP - (PD + PL);
    const double pen_balance = w_balance * bal * bal;

    // 4) Soft bounds penalties
    double pen_bounds = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        if (P < Pmin[i]) {
            const double d = Pmin[i] - P;
            pen_bounds += w_bounds * d * d;
        } else if (P > Pmax[i]) {
            const double d = P - Pmax[i];
            pen_bounds += w_bounds * d * d;
        }
    }

    // 5) POZ penalties (inside zone → dist^2 to nearest boundary)
    double pen_poz = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        for (const auto& z : poz[i]) {
            const double L = z.first;
            const double U = z.second;
            if (P > L && P < U) {
                const double d = std::min(P - L, U - P);
                pen_poz += w_poz * d * d;
            }
        }
    }

    double cost = fuel + pen_balance + pen_bounds + pen_poz;
    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost))
        cost = 1e12;

    return cost;
}

void ELD3::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences
    const int n = static_cast<int>(x.size());
    g.assign(n, 0.0);

    const double rel = 1e-6;
    const double f0  = evaluate_core(x);

    Vec xh = x;
    for (int k = 0; k < n; ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);

        
        if (k < NG && x[k] + h > Pmax[k])
            h = std::min(h, Pmax[k] - x[k]);

        if (h <= 0.0) {
            g[k] = 0.0;
            continue;
        }

        xh[k] += h;
        const double f1 = evaluate_core(xh);
        g[k] = (f1 - f0) / h;
        xh[k] = x[k];
    }
}

} // namespace optimsolution
