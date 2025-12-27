#include "eld4.h"

namespace optimsolution {

ELD4::ELD4()
    : NG(40),
      PD(10500.0),
      use_losses(false),
      B00(0.0),
      w_balance(0.15),
      w_bounds(40.0)
{
    setName("eld4");
    setFullName("Economic Load Dispatch - 4 (40-unit CEC benchmark)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("real-world power system benchmark");

    // no known global optimum → do not call setKnownGlobalOptimum
    set_default_data();
}

void ELD4::set_default_data()
{
    // ---- Pmin/Pmax (faithfully copied from original file) ----
    Pmin = {36,36,60,80,47,68,110,135,135,130,94,94,125,125,125,125,
            220,220,242,242,254,254,254,254,254,254,10,10,10,47,
            60,60,60,90,90,90,25,25,25,242};

    Pmax = {114,114,120,190,97,140,300,300,300,300,375,375,500,500,500,500,
            500,500,550,550,550,550,550,550,550,550,150,150,150,97,
            190,190,190,200,200,200,110,110,110,550};

    // ---- Quadratic coefficients ----
    a = {0.00690,0.00690,0.02028,0.00942,0.01140,0.01142,0.00357,0.00492,0.00573,0.00605,
         0.00515,0.00569,0.00421,0.00752,0.00708,0.00708,0.00313,0.00313,0.00313,0.00313,
         0.00298,0.00298,0.00284,0.00284,0.00277,0.00277,0.52124,0.52124,0.52124,0.01140,
         0.00160,0.00160,0.00160,0.00010,0.00010,0.00010,0.01610,0.01610,0.01610,0.00313};

    b = {6.73,6.73,7.07,8.18,5.35,8.05,8.03,6.99,6.60,12.90,
         12.90,12.80,12.50,8.84,9.15,9.15,7.97,7.95,7.97,7.97,
         6.63,6.63,6.66,6.66,7.10,7.10,3.33,3.33,3.33,5.35,
         6.43,6.43,6.43,8.95,8.62,8.62,5.88,5.88,5.88,7.97};

    c = {94.705,94.705,309.54,369.03,148.89,220.33,287.71,391.98,455.76,722.82,
         635.20,654.69,913.40,1760.40,1728.30,1728.30,647.85,649.69,647.83,647.81,
         785.96,785.96,794.53,794.53,801.32,801.32,1055.10,1055.10,1055.10,148.89,
         222.92,222.92,222.92,107.87,116.58,116.58,307.45,307.45,307.45,647.83};

    // ---- Valve-point coefficients ----
    d = {100,100,100,150,120,100,200,200,200,200,
         200,200,300,300,300,300,300,300,300,300,
         300,300,300,300,300,300,120,120,120,120,
         150,150,150,200,200,200,80,80,80,300};

    e = {0.084,0.084,0.084,0.063,0.077,0.084,0.042,0.042,0.042,0.042,
         0.042,0.042,0.035,0.035,0.035,0.035,0.035,0.035,0.035,0.035,
         0.035,0.035,0.035,0.035,0.035,0.035,0.077,0.077,0.077,0.077,
         0.063,0.063,0.063,0.042,0.042,0.042,0.098,0.098,0.098,0.035};

    // ---- Transmission losses default OFF ----
    B.assign(NG, std::vector<double>(NG, 0.0));
    B0.assign(NG, 0.0);
    B00 = 0.0;
}

void ELD4::build_bounds()
{
    Vec lo(NG), hi(NG);
    for (int i = 0; i < NG; ++i) {
        lo[i] = Pmin[i];
        hi[i] = Pmax[i];
    }
    setBounds(lo, hi);
}

void ELD4::init(int /*dim*/)
{
    Problem::init(NG);
    build_bounds();
}

double ELD4::evaluate_core(const Vec& x)
{
    // ----- 1) Fuel + valve-point -----
    double fuel = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P  = x[i];
        const double vp = std::sin(e[i] * (Pmin[i] - P));
        fuel += a[i]*P*P + b[i]*P + c[i]
              + d[i] * smooth_abs(vp);
    }

    // ----- 2) Losses (optional) -----
    double PL = 0.0;
    if (use_losses) {
        for (int i = 0; i < NG; ++i)
            for (int j = 0; j < NG; ++j)
                PL += x[i] * B[i][j] * x[j];
        double tmp = 0.0;
        for (int i = 0; i < NG; ++i) tmp += B0[i] * x[i];
        PL += tmp + B00;
    }

    // ----- 3) Power balance penalty -----
    double sumP = 0.0;
    for (int i = 0; i < NG; ++i)
        sumP += x[i];

    const double bal = sumP - (PD + PL);
    const double pen_balance = w_balance * bal * bal;

    // ----- 4) Soft bounds penalties -----
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

    double f = fuel + pen_balance + pen_bounds;
    if (!(f >= 0.0) || std::isnan(f) || std::isinf(f))
        f = 1e12;
    return f;
}

void ELD4::gradient_core(const Vec& x, Vec& g)
{
    g.assign(x.size(), 0.0);

    const double f0  = evaluate_core(x);
    const double rel = 1e-6;

    Vec xt = x;
    for (int k = 0; k < (int)x.size(); ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);

        // Try to remain inside global bounds
        if (x[k] + h > Pmax[k])
            h = std::min(h, Pmax[k] - x[k]);
        if (h <= 0.0) {
            g[k] = 0.0;
            continue;
        }

        xt[k] += h;
        const double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;
        xt[k] = x[k];
    }
}

} // namespace optimsolution
