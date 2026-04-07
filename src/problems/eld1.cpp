#include "eld1.h"

namespace optimsolution {

ELD1::ELD1()
    : NG(6),
      PD(700.0),
      use_losses(false),
      B00(0.0),
      use_valve(true),
      w_balance(1.0),
      w_bounds(500.0),
      w_poz(1000.0)
{
    setName("eld1");
    setFullName("Economic Load Dispatch - 1 (single period)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("real-world power system benchmark");

    // No known analytic optimum → leave default
    set_defaults();
}

void ELD1::set_defaults()
{
    // EXACT values from old version
    a   = {0.00375, 0.0175, 0.0625, 0.00834, 0.025, 0.025};
    b   = {2.00,    1.75,   1.00,   3.25,    3.00,  3.00};
    c   = {0.0,     0.0,    0.0,    0.0,     0.0,   0.0};

    Pmin= {50, 50, 50, 50, 50, 50};
    Pmax= {200,210,260,180,200,200};

    e_vp = {100, 140, 160, 100, 120, 150};
    f_vp = {0.042,0.040,0.038,0.042,0.040,0.039};

    poz.clear();
    poz.resize(6);
    poz[1].push_back({90.0,110.0});
    poz[2].push_back({120.0,140.0});
    poz[2].push_back({180.0,200.0});
    poz[4].push_back({90.0,105.0});

    B.assign(NG, std::vector<double>(NG,0.0));
    B0.assign(NG,0.0);
    B00 = 0.0;
}

void ELD1::expand_to_NG()
{
    auto grow_cycle = [&](std::vector<double>& v){
        if ((int)v.size() == NG) return;
        int old = (int)v.size();
        v.reserve(NG);
        for (int i = old; i < NG; ++i)
            v.push_back(v[i % old]);
    };

    auto grow_zero = [&](std::vector<double>& v){
        v.assign(NG, 0.0);
    };

    grow_cycle(a);
    grow_cycle(b);
    grow_cycle(c);

    grow_cycle(Pmin);
    grow_cycle(Pmax);

    if (use_valve) {
        grow_cycle(e_vp);
        grow_cycle(f_vp);
    } else {
        grow_zero(e_vp);
        grow_zero(f_vp);
    }

    if ((int)poz.size() != NG) {
        auto old = poz;
        poz.clear();
        poz.resize(NG);
        for (int i = 0; i < NG; ++i)
            poz[i] = old[i % old.size()];
    }

    // Reset losses for new dimension
    B.assign(NG, std::vector<double>(NG, 0.0));
    B0.assign(NG, 0.0);
    B00 = 0.0;
}

void ELD1::build_bounds()
{
    Vec lo(NG), hi(NG);
    for (int i = 0; i < NG; ++i) {
        lo[i] = Pmin[i];
        hi[i] = Pmax[i];
    }
    setBounds(lo, hi);
}

void ELD1::init(int dim)
{
    if (dim >= 1)
        NG = dim;

    expand_to_NG();
    Problem::init(NG);
    build_bounds();
}

double ELD1::evaluate_core(const Vec& x)
{
    // 1) Fuel + valve-point
    double fuel = 0.0;

    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        fuel += a[i]*P*P + b[i]*P + c[i];

        if (use_valve) {
            const double arg = f_vp[i] * (Pmin[i] - P);
            fuel += e_vp[i] * smooth_abs(std::sin(arg));
        }
    }

    // 2) Losses (off by default)
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

    // 3) Power balance
    double sumP = 0.0;
    for (int i = 0; i < NG; ++i)
        sumP += x[i];

    const double bal = sumP - (PD + PL);
    const double pen_balance = w_balance * bal * bal;

    // 4) Soft bounds
    double pen_bounds = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        if (P < Pmin[i]) {
            const double d = Pmin[i] - P;
            pen_bounds += w_bounds * d * d;
        }
        else if (P > Pmax[i]) {
            const double d = P - Pmax[i];
            pen_bounds += w_bounds * d * d;
        }
    }

    // 5) POZ penalties
    double pen_poz = 0.0;

    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        for (const auto& z : poz[i]) {
            const double L = z.first;
            const double U = z.second;

            if (U <= L) continue;
            if (P > L && P < U) {
                const double d = std::min(P - L, U - P);
                pen_poz += w_poz * d * d;
            }
        }
    }

    // Regularizer
    double reg = 0.0;
    for (int i = 0; i < NG; ++i)
        reg += 1e-6 * x[i];

    double cost = fuel + pen_balance + pen_bounds + pen_poz + reg;

    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost))
        cost = 1e12;

    return cost;
}

void ELD1::gradient_core(const Vec& x, Vec& g)
{
    const int n = (int)x.size();
    g.assign(n, 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int k = 0; k < n; ++k) {
        const double h = std::max(1e-6, std::abs(x[k]) * 1e-6);
        xt[k] += h;
        const double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;
        xt[k] = x[k];
    }
}

} // namespace optimsolution
