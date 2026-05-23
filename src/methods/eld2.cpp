#include "eld2.h"

namespace optimsolution {

ELD2::ELD2()
    : NG(13),
      Pd(1800.0),
      w_balance(0.2),
      w_bounds(50.0)
{
    setName("eld2");
    setFullName("Economic Load Dispatch - 2 (13-unit single-period)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("real-world power system benchmark");

    // Δεν υπάρχει γνωστό κλειστό global optimum → δεν καλούμε setKnownGlobalOptimum(x*)
    set_default_data();
}

void ELD2::set_default_data()
{
    // Limits (CEC/13-unit style) – faithful to reference
    Pmin = { 0,   0,   0,  60, 60, 60, 60, 60, 60, 40, 40, 55, 55 };
    Pmax = { 680, 360, 360,180,180,180,180,180,180,120,120,120,120 };

    // Quadratic fuel + valve-point coefficients (as in reference)
    a = {0.00028,0.00056,0.00056,0.00324,0.00324,0.00324,0.00324,0.00324,0.00324,
         0.00284,0.00284,0.00284,0.00284};
    b = {8.10,    8.10,   8.10,   7.74,   7.74,   7.74,   7.74,   7.74,   7.74,
         8.60,   8.60,   8.60,   8.60};
    c = {550,     309,    307,    240,    240,    240,    240,    240,    240,
         126,    126,    126,    126};
    e = {300,     200,    150,    150,    150,    150,    150,    150,    150,
         100,    100,    100,    100};
    f = {0.035,   0.042,  0.042,  0.063,  0.063,  0.063,  0.063,  0.063,  0.063,
         0.084,  0.084,  0.084,  0.084};
}

void ELD2::build_bounds()
{
    Vec lo(NG), hi(NG);
    for (int i = 0; i < NG; ++i) {
        lo[i] = Pmin[i];
        hi[i] = Pmax[i];
    }
    setBounds(lo, hi);
}

void ELD2::init(int /*dim*/)
{
    // ELD2 έχει fixed NG = 13
    Problem::init(NG);
    build_bounds();
}

double ELD2::evaluate_core(const Vec& x)
{
    // 1) Fuel + valve-point
    double fuel = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P = x[i];
        const double base = a[i]*P*P + b[i]*P + c[i];
        const double vp   = std::abs(e[i] * std::sin(f[i] * (Pmin[i] - P)));
        fuel += base + vp;
    }

    // 2) Power balance penalty (no losses in this instance)
    double sumP = 0.0;
    for (int i = 0; i < NG; ++i)
        sumP += x[i];

    const double diff        = sumP - Pd;
    const double pen_balance = w_balance * diff * diff;

    // 3) Soft bounds penalties
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

    double fval = fuel + pen_balance + pen_bounds;
    if (!(fval >= 0.0) || std::isnan(fval) || std::isinf(fval))
        fval = 1e12;

    return fval;
}

void ELD2::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences (faithful to reference style)
    g.assign(x.size(), 0.0);
    const double rel = 1e-6;
    const double f0  = evaluate_core(x);

    Vec xh = x;
    for (int k = 0; k < (int)x.size(); ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);
        // keep step inside bounds if possible
        if (x[k] + h > Pmax[k])
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
