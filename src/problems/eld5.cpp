#include "eld5.h"

namespace optimsolution {

ELD5::ELD5()
    : NG(140),
      PD(49342.0),
      use_valve(false),   // faithful to reference defaults
      use_losses(false),
      B00(0.0),
      w_balance(0.15),
      w_bounds(40.0)
{
    setName("eld5");
    setFullName("Economic Load Dispatch - 5 (CEC2011 140-unit case)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("real-world power system benchmark");

    

    set_default_limits_from_cec();  // Pmin/Pmax from reference
    set_placeholder_costs();        // a,b,c (+ d,e) placeholders as στο παλιό
}

void ELD5::set_default_limits_from_cec()
{
    // Pmin/Pmax from the uploaded reference (CEC-2011, 140-unit instance)
    // (Same lists and order as in the original files.)
    Pmin = {
        71,120,125,125,90,90,280,280,260,260,260,260,260,260,260,260,260,260,260,260,
        260, 60,260,260,280,280,280,280,260,260,260,260,260,260,260,260,120,120,423,423,
          3,  3,160,160,160,160,160,160,160,160,165,165,165,165,180,180,103,198,100,153,
        163, 95,160,160,196,196,196,196,130,130,137,137,195,175,175,175,175,330,160,160,
        200, 56,115,115,115,207,207,175,175,175,175,360,415,795,795,578,615,612,612,758,
        755,750,750,713,718,791,786,795,795,795,795, 94, 94, 94,244,244,244, 95, 95,116,
        175,  2,  4, 15,  9, 12, 10,112,  4,  5,  5, 50,  5, 42, 42, 41, 17,  7,  7, 26
    };

    Pmax = {
        119,189,190,190,190,190,490,490,496,496,496,496,506,509,506,505,506,506,505,505,
        505,505,505,505,537,537,549,549,501,501,506,506,506,506,500,500,241,241,774,769,
         19, 28,250,250,250,250,250,250,250,250,504,504,504,504,471,561,341,617,312,471,
        500,302,511,511,490,490,490,490,432,432,455,455,541,536,540,538,540,574,531,531,
        542,132,245,245,245,307,307,345,345,345,345,580,645,984,978,682,720,718,720,964,
        958,1007,1006,1013,1020,954,952,1006,1013,1021,1015,203,203,203,379,379,379,190,
        189,194,321, 19, 59, 83, 53, 37, 34,373, 20, 38, 19, 98, 10, 74, 74,105, 51, 19,
         19, 40
    };
}

void ELD5::set_placeholder_costs()
{
    //  140-unit case.
    a.assign(NG, 0.0010);
    b.assign(NG, 9.0);
    c.assign(NG, 300.0);

    // Valve-point placeholders (kept, .. use_valve = false by default)
    d.assign(NG, 100.0);
    e.assign(NG, 0.04);

    // Loss coefficients placeholders (all zerow, losses off)
    B.assign(NG, std::vector<double>(NG, 0.0));
    B0.assign(NG, 0.0);
    B00 = 0.0;
}

void ELD5::build_bounds()
{
    Vec lo(NG), hi(NG);
    for (int i = 0; i < NG; ++i) {
        lo[i] = Pmin[i];
        hi[i] = Pmax[i];
    }
    setBounds(lo, hi);
}

void ELD5::init(int /*dim*/)
{
    // Fixed NG = 140, as in the reference implementation
    Problem::init(NG);
    build_bounds();
}

double ELD5::evaluate_core(const Vec& x)
{
    // 1) Fuel cost + optional valve-point
    double fuel = 0.0;
    for (int i = 0; i < NG; ++i) {
        const double P  = x[i];
        fuel += a[i]*P*P + b[i]*P + c[i];

        if (use_valve) {
            const double s = std::sin(e[i] * (Pmin[i] - P));
            fuel += d[i] * smooth_abs(s);  // smooth |sin|
        }
    }

    // 2) Losses (optional; default OFF -> PL = 0)
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

    // 4) Soft bounds penalties (no clamp)
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

    double f = fuel + pen_balance + pen_bounds;
    if (!(f >= 0.0) || std::isnan(f) || std::isinf(f))
        f = 1e12;

    return f;
}

void ELD5::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences (faithful to the reference style)
    g.assign(x.size(), 0.0);
    const double rel = 1e-6;
    const double f0  = evaluate_core(x);

    Vec xh = x;
    for (int k = 0; k < (int)x.size(); ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);

        // try to keep step inside bounds
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
