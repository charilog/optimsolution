#include "hydrothermal.h"

namespace optimsolution {

Hydrothermal::Hydrothermal()
: T_(24), NG_(3), NH_(2),
  // penalties (tuned to give feasibility preference, but smooth)
  w_balance_(0.1), w_V_(5.0), w_QP_(1.0), w_ramp_(0.0), w_final_(10.0)
{
    // Metadata για το νέο template
    setName("hydrothermal");
    setFullName("Hydrothermal scheduling (smooth penalty model)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("dynamic economic dispatch / hydrothermal scheduling problem");

    // ---- Thermal data (kept close to your file) ----
    a_ = {0.0012, 0.0010, 0.0015};
    b_ = {7.0,    6.3,    6.8   };
    c_ = {240.0,  200.0,  220.0 };

    Pmin_ = { 50,  60, 100 };
    Pmax_ = {200, 250, 300 };

    // optional ramps (0 => off)
    UR_ = {0.0, 0.0, 0.0};
    DR_ = {0.0, 0.0, 0.0};

    // ---- Hydro storage/discharge (same scales as your file) ----
    Qmin_ = { 20, 25 };
    Qmax_ = {100,120 };

    Vmin_ = {100,100};
    Vmax_ = {500,500};

    Vinit_ = {200,250};
    Vfinal_= {200,250};

    // ---- Hydro power model parameters ----
    // Choose head baseline h0 and sensitivity beta so that hydro can supply ~50 MW per unit at high head/flow.
    // kappa ≈ 9.81e-3 * η ; we absorb units into kappa for consistency with abstract Q,V scales.
    kappa_ = {0.0085, 0.0085}; // MW per (m * flow_unit)
    h0_    = {50.0,   50.0  }; // base head [m]
    beta_  = {0.05,   0.05  }; // head per storage unit

    // ---- Demand and inflow profiles (baseline feasible) ----
    demand_.assign(T_, 800.0);
    inflow_.assign(NH_*T_, 30.0); // per (j,t)
}

void Hydrothermal::init(int /*dim*/) {
    // Decision vector: P (NG*T) then Q (NH*T)
    const int D = NG_*T_ + NH_*T_;
    Problem::init(D);

    Vec lo(D), hi(D);
    for (int t = 0; t < T_; ++t) {
        for (int i = 0; i < NG_; ++i) {
            lo[idxP(t,i)] = Pmin_[i];
            hi[idxP(t,i)] = Pmax_[i];
        }
        for (int j = 0; j < NH_; ++j) {
            lo[idxQ(t,j)] = Qmin_[j];
            hi[idxQ(t,j)] = Qmax_[j];
        }
    }
    setBounds(lo, hi);
}

double Hydrothermal::evaluate_core(const Vec& x) {
    double obj = 0.0;

    // initialize storages
    std::vector<double> V = Vinit_; // size NH_

    for (int t = 0; t < T_; ++t) {
        // ---- thermal cost + (soft) bound penalties ----
        double sumP = 0.0;
        for (int i = 0; i < NG_; ++i) {
            const int k = idxP(t,i);
            const double P = x[k];
            sumP += P;

            // quadratic fuel
            obj += a_[i]*P*P + b_[i]*P + c_[i];

            // optional ramp penalties
            if (w_ramp_ > 0.0 && t > 0) {
                const double dP = x[idxP(t,i)] - x[idxP(t-1,i)];
                if (UR_[i] > 0.0 && dP >  UR_[i]) obj += w_ramp_ * sqr(dP - UR_[i]);
                if (DR_[i] > 0.0 && -dP > DR_[i]) obj += w_ramp_ * sqr((-dP) - DR_[i]);
            }

            // soft bounds (in case a solver wanders outside)
            if (P < Pmin_[i]) obj += w_QP_ * sqr(Pmin_[i]-P);
            if (P > Pmax_[i]) obj += w_QP_ * sqr(P - Pmax_[i]);
        }

        // ---- hydro power with head dependence ----
        double sumH = 0.0;
        for (int j = 0; j < NH_; ++j) {
            const int k = idxQ(t,j);
            const double Q = x[k];

            // storage update
            const double Vin  = V[j];
            const double Vout = Vin + inflow_[j*T_ + t] - Q;
            const double Vbar = 0.5 * (Vin + Vout);
            V[j] = Vout;

            // head-dependent power
            const double head = h0_[j] + beta_[j] * Vbar;
            const double H = kappa_[j] * std::max(0.0, head) * std::max(0.0, Q);
            sumH += H;

            // hydro bound penalties
            if (Q < Qmin_[j]) obj += w_QP_ * sqr(Qmin_[j]-Q);
            if (Q > Qmax_[j]) obj += w_QP_ * sqr(Q - Qmax_[j]);

            // storage bound penalties
            if (Vout < Vmin_[j]) obj += w_V_ * sqr(Vmin_[j] - Vout);
            if (Vout > Vmax_[j]) obj += w_V_ * sqr(Vout - Vmax_[j]);
        }

        // ---- power balance penalty ----
        const double diff = (sumP + sumH) - demand_[t];
        obj += w_balance_ * diff * diff;
    }

    // final storage targets
    for (int j = 0; j < NH_; ++j) {
        const double miss = V[j] - Vfinal_[j];
        obj += w_final_ * miss * miss;
    }

    if (std::isnan(obj) || std::isinf(obj)) obj = 1e12;
    return obj;
}

void Hydrothermal::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int k = 0; k < (int)x.size(); ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * rel);
        xt[k] = x[k] + h;
        const double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;
        xt[k] = x[k];
    }
}

} // namespace optimsolution
