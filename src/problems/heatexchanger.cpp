#include "heatexchanger.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;
}

HeatExchanger::HeatExchanger()
: Dv_(Dcore_),
  // --- Process specs (same defaults as reference) ---
  m_hot_(2.0),  m_cold_(2.0),
  cp_hot_(4180.0), cp_cold_(4180.0),
  Th_in_(150.0), Th_out_(110.0),
  Tc_in_(30.0),  Tc_out_(70.0),
  rho_hot_(995.0), rho_cold_(995.0),
  mu_hot_(0.001),  mu_cold_(0.001),
  k_hot_(0.6),     k_cold_(0.6),
  // These are computed below (same logic as reference)
  Q_req_(0.0), LMTD_(0.0), UA_req_(0.0), Rf_(2.0e-4),
  pump_eta_(0.7),
  sec_per_year_(3600.0*24.0*330.0),
  elec_cost_(0.12),
  C_area_(1000.0),
  w_pump_(1.0e-5),
  pen_hard_(1.0e5),
  pen_soft_(200.0),
  a_tube_(500.0), a_shell_(300.0),
  k_tube_wall_(16.0),
  di_factor_(0.90),
  kt_dp_(800.0), ks_dp_(500.0),
  Ds_min_(0.30), Ds_max_(1.50),
  Dt_min_(0.010), Dt_max_(0.050),
  L_min_(1.00),   L_max_(6.00),
  pr_min_(1.25),  pr_max_(2.00),
  B_min_(0.10),   B_max_(0.60),
  cb_min_(0.15),  cb_max_(0.45)
{
    // Metadata για το νέο template
    setName("heatexchanger");
    setFullName("Heat Exchanger Design Optimization Problem");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("engineering design benchmark");

    // Required thermal duty and LMTD exactly like the reference
    Q_req_ = m_cold_ * cp_cold_ * (Tc_out_ - Tc_in_);
    const double dT1 = Th_in_ - Tc_out_;
    const double dT2 = Th_out_ - Tc_in_;
    if (std::fabs(dT1 - dT2) < 1e-9) LMTD_ = dT1;
    else LMTD_ = (dT1 - dT2) / std::log(dT1 / dT2);
    UA_req_ = Q_req_ / std::max(1e-6, LMTD_);
}

void HeatExchanger::set_core_bounds(Vec& lo, Vec& hi) const {
    lo[0] = Ds_min_; hi[0] = Ds_max_;
    lo[1] = Dt_min_; hi[1] = Dt_max_;
    lo[2] = L_min_;  hi[2] = L_max_;
    lo[3] = pr_min_; hi[3] = pr_max_;
    lo[4] = B_min_;  hi[4] = B_max_;
    lo[5] = cb_min_; hi[5] = cb_max_;
}

void HeatExchanger::init(int dim) {
    // core is 6 
    Dv_ = (dim >= Dcore_) ? dim : Dcore_;
    Problem::init(Dv_);

    Vec lo(Dv_, 0.0), hi(Dv_, 1.0);
    set_core_bounds(lo, hi); // πρώτες 6
    setBounds(lo, hi);
}

double HeatExchanger::evaluate_core(const Vec& x) {
    // clamp core vars 
    const double Ds = clamp(x[0], Ds_min_, Ds_max_);
    const double Dt = clamp(x[1], Dt_min_, Dt_max_);
    const double L  = clamp(x[2], L_min_,  L_max_);
    const double pr = clamp(x[3], pr_min_, pr_max_);
    const double B  = clamp(x[4], B_min_,  B_max_);
    const double cb = clamp(x[5], cb_min_, cb_max_);


    const double Di = di_factor_ * Dt;
    const double pt = pr * Dt;
    const double packing = 0.6;
    int Nt = (int)std::floor(packing * std::pow(Ds / std::max(1e-6, pt), 2.0) * 100.0);
    if (Nt < 10) Nt = 10;
    const int Npass = 2;

    // Heat transfer areas
    const double A     = PI * Dt * L * Nt * 0.9;          // 0.9 inefficiency
    const double Atube = PI * Di * Di / 4.0;


    const double vol_cold = m_cold_ / rho_cold_;
    const double vol_hot  = m_hot_  / rho_hot_;
    const double Aflow_t  = Nt * Atube / Npass;
    const double v_t      = vol_cold / std::max(1e-6, Aflow_t);

    // Shell crossflow area surrogate
    const double Ashell   = Ds * B * std::max(0.05, cb);
    const double v_s      = vol_hot / std::max(1e-6, Ashell);

    // Convective surrogates
    const double hi = a_tube_  * std::pow(std::max(1e-6, v_t), 0.8) / std::pow(std::max(1e-6, Di), 0.2);
    const double ho = a_shell_ * std::pow(std::max(1e-6, v_s), 0.8) / std::pow(std::max(1e-6, Ds), 0.2);

    // Wall resistance (thin tube wall)
    const double t_wall = 0.05 * Dt;
    const double Rw = t_wall / std::max(1e-6, k_tube_wall_);

    // Overall U and UA
    const double Rtot = (1.0/hi) + Rw + (1.0/ho) + Rf_;
    const double U    = 1.0 / std::max(1e-9, Rtot);
    const double UA   = U * A;

    // 1) UA requirement penalty (heavy)
    double pen_UA = 0.0;
    if (UA < UA_req_) {
        const double shortfall = (UA_req_ - UA) / UA_req_;
        pen_UA = pen_hard_ * shortfall * shortfall;
    }

    // 2) Pressure drop surrogates
    const double dP_tube  = kt_dp_ * L * v_t * v_t / std::max(1e-6, Di);
    const double dP_shell = ks_dp_ * (Ds / std::max(1e-6, B)) * v_s * v_s;

    // Pumping power
    const double P_pump = (dP_tube * vol_cold + dP_shell * vol_hot) / std::max(1e-6, pump_eta_);

    // 3) Soft penalties
    double pen_soft_sum = 0.0;
    if (v_t > 3.0) { const double r = v_t - 3.0; pen_soft_sum += pen_soft_ * r*r; }
    if (v_s > 5.0) { const double r = v_s - 5.0; pen_soft_sum += pen_soft_ * 0.5 * r*r; }
    if (B < 0.2*Ds) { const double r = 0.2*Ds - B; pen_soft_sum += pen_soft_ * r*r; }


    const double cost_area = C_area_ * A;
    const double cost_pump = w_pump_ * P_pump * sec_per_year_;

    double cost = cost_area + cost_pump + pen_UA + pen_soft_sum;
    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost)) cost = 1e9;
    return cost;
}

void HeatExchanger::gradient_core(const Vec& x, Vec& g) {

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
