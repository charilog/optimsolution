// ============================================================================
//  datacentercooling.cpp
//  Adaptive Data Center Thermal Orchestration (ADCTO) Benchmark — v5
//
//  Three-mechanism implementation. Every numerical coefficient appearing in
//  Section 2.2 of the paper is exposed as a cfg-tunable parameter so that
//  the benchmark integrates with the OptimSolution sensitivity-analysis
//  pipeline. Default values match the paper specification exactly.
// ============================================================================

#include "datacentercooling.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace optimsolution {

// ----------------------------------------------------------------------------
//  Numerical helpers (file-local)
// ----------------------------------------------------------------------------
namespace {

constexpr double PI = 3.14159265358979323846;

inline double sqr(double v)            { return v * v; }
inline double clamp01(double v)        { return std::max(0.0, std::min(1.0, v)); }
inline double soft_clip(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }
inline double sin2(double v)           { const double s = std::sin(v); return s * s; }
inline double sat(double z, double a)  { const double z2 = z * z; return z2 / (a + z2); }

inline double banded_control(double raw, double phase)
{
    raw = clamp01(raw);
    if (raw < 0.10) return 0.02 * raw;
    if (raw < 0.28) return clamp01(0.15 + 0.010 * std::sin(19.0 * raw + phase));
    if (raw < 0.49) return clamp01(0.41 + 0.011 * std::sin(21.0 * raw + phase));
    if (raw < 0.71) return clamp01(0.68 + 0.012 * std::sin(23.0 * raw + phase));
    if (raw < 0.88) return clamp01(0.87 + 0.010 * std::sin(25.0 * raw + phase));
    return            clamp01(0.95 + 0.008 * std::sin(29.0 * raw + phase));
}

} // anonymous namespace

// ============================================================================
//  Constructor — initialise all members to zero, defaults set in load_config
// ============================================================================
DataCenterCooling::DataCenterCooling()
    : Problem(),
      carbon_budget_(0.0),
      t_init_(0.0), t_target_(0.0), h_init_(0.0), c_init_(0.0),
      base_excess_weight_(0.0), base_terminal_temp_weight_(0.0),
      base_energy_weight_(0.0), base_rugged_weight_(0.0), base_rugged_freq_(0.0),
      hotspot_decay_(0.0), hotspot_gain_coeff_(0.0),
      hotspot_sat_coeff_(0.0), hotspot_temp_coupling_(0.0),
      thermal_load_gain_(0.0), thermal_cooling_gain_(0.0),
      thermal_airflow_gain_(0.0), ambient_relax_(0.0),
      temp_update_scale_(0.0),
      m1_extra_excess_weight_(0.0), m1_terminal_hotspot_weight_(0.0),
      m2_w1_(0.0), m2_w2_(0.0), m2_w3_(0.0),
      comfort_decay_(0.0),
      budget_base_per_stage_(0.0), budget_workload_factor_(0.0),
      checkpoint_weight_(0.0), checkpoint_saturation_(0.0),
      budget_overrun_weight_(0.0),
      m3_corridor_stage_weight_(0.0), m3_corridor_term_weight_(0.0),
      m3_multimodal_weight_(0.0), m3_multimodal_freq_(0.0),
      power_base_(0.0), power_q_linear_(0.0),
      power_q_quad_(0.0), power_f_quad_(0.0),
      config_{true, true, true},
      config_loaded_(false),
      loaded_config_path_()
{
    setName("datacentercooling");
}

// ============================================================================
//  init(dim) — called by the framework
// ============================================================================
void DataCenterCooling::init(int dim)
{
    if (dim < 6) dim = 6;

    load_config();
    load_main_config_overrides();
    refresh_metadata();

    std::fprintf(stderr,
        "[ADCTO v5 LOADED]  cfg=%s\n"
        "  flags:    M1=%d  M2=%d  M3=%d\n"
        "  init:     T1=%.2f  T*=%.2f  H1=%.3f  C1=%.3f\n"
        "  base:     ex_w=%.2f termT_w=%.2f en_w=%.2f rug_w=%.3f rug_f=%.2f\n"
        "  M1:       hd=%.3f hg=%.3f hsc=%.3f htc=%.3f tlg=%.2f\n"
        "  M1cost:   extra_ex_w=%.2f termH_w=%.2f\n"
        "  M2:       w1=%.3f w2=%.3f w3=%.3f\n"
        "  M3:       cd=%.3f bb=%.2f bw=%.2f cw=%.2f bow=%.2f\n"
        "  M3cost:   cs_w=%.2f ct_w=%.2f  mm_w=%.2f mm_f=%.2f\n"
        "  dim=%d\n",
        loaded_config_path_.empty() ? "(not found)" : loaded_config_path_.c_str(),
        (int)config_.M1_enable, (int)config_.M2_enable, (int)config_.M3_enable,
        t_init_, t_target_, h_init_, c_init_,
        base_excess_weight_, base_terminal_temp_weight_, base_energy_weight_,
        base_rugged_weight_, base_rugged_freq_,
        hotspot_decay_, hotspot_gain_coeff_, hotspot_sat_coeff_,
        hotspot_temp_coupling_, thermal_load_gain_,
        m1_extra_excess_weight_, m1_terminal_hotspot_weight_,
        m2_w1_, m2_w2_, m2_w3_,
        comfort_decay_, budget_base_per_stage_, budget_workload_factor_,
        checkpoint_weight_, budget_overrun_weight_,
        m3_corridor_stage_weight_, m3_corridor_term_weight_,
        m3_multimodal_weight_, m3_multimodal_freq_,
        dim);

    Problem::init(dim);

    Vec l(dim, 0.0), u(dim, 1.0);
    setBounds(l, u);

    workload_   .assign(dim, 0.0);
    ambient_    .assign(dim, 0.0);
    tariff_     .assign(dim, 0.0);
    tlimit_     .assign(dim, 0.0);
    coolcap_    .assign(dim, 0.0);
    flowcap_    .assign(dim, 0.0);
    pref_       .assign(dim, 0.0);
    criticality_.assign(dim, 0.0);

    for (int i = 0; i < dim; ++i) {
        const double tau = static_cast<double>(i + 1) / dim;
        workload_[i]    = soft_clip(0.62
                                     + 0.18 * std::sin(2.0 * PI * tau + 0.35)
                                     + 0.09 * std::sin(9.0 * PI * tau + 0.70)
                                     + 0.06 * sin2(15.0 * PI * tau + 0.15),
                                     0.28, 0.96);
        ambient_[i]     = 28.0
                          + 5.2 * std::sin(2.0  * PI * tau + 0.40)
                          + 2.1 * std::sin(7.0  * PI * tau + 1.10)
                          + 1.8 * sin2     (13.0 * PI * tau + 0.20);
        tariff_[i]      = 0.13
                          + 0.08 * sin2(3.0  * PI * tau + 0.80)
                          + 0.03 * sin2(11.0 * PI * tau + 0.25);
        tlimit_[i]      = 23.0
                          + 0.9 * sin2(4.0 * PI * tau + 0.20)
                          + 0.7 * sin2(9.0 * PI * tau + 1.00);
        coolcap_[i]     = std::max(5.8,
                                   8.6
                                   + 1.6 * sin2(5.0  * PI * tau + 0.55)
                                   + 0.8 * std::sin(12.0 * PI * tau + 0.90));
        flowcap_[i]     = std::max(3.0,
                                   4.6
                                   + 0.9 * sin2(6.0  * PI * tau + 0.25)
                                   + 0.5 * std::sin(10.0 * PI * tau + 0.60));
        pref_[i]        = soft_clip(0.38
                                     + 0.16 * std::sin(2.0 * PI * tau + 0.65)
                                     + 0.09 * std::sin(8.0 * PI * tau + 0.50),
                                     0.06, 0.94);
        criticality_[i] = 1.0
                          + 0.45 * sin2(5.0  * PI * tau + 0.15)
                          + 0.20 * sin2(12.0 * PI * tau + 0.95);
    }

    carbon_budget_ = 0.0;
    for (int i = 0; i < dim; ++i)
        carbon_budget_ += budget_base_per_stage_
                        + budget_workload_factor_ * workload_[i];
}

// ============================================================================
//  refresh_metadata — encode mechanism flags in problem name
// ============================================================================
void DataCenterCooling::refresh_metadata()
{
    std::string full_name = "datacentercooling";
    full_name += " [M1=" + std::string(config_.M1_enable ? "1" : "0");
    full_name += ",M2="  + std::string(config_.M2_enable ? "1" : "0");
    full_name += ",M3="  + std::string(config_.M3_enable ? "1" : "0") + "]";
    setName(full_name);
}

double DataCenterCooling::safe_x(const Vec& x, int i) const
{
    const int D = static_cast<int>(x.size());
    if (i < 0)    return x[0];
    if (i >= D)   return x[D - 1];
    return x[i];
}

// ============================================================================
//  evaluate_with_point — Section 2.2 of the paper, parameter-driven
// ============================================================================
double DataCenterCooling::evaluate_with_point(const Vec& x) const
{
    const int D = static_cast<int>(x.size());

    double T = t_init_;
    double H = h_init_;
    double C = c_init_;
    double cum_power = 0.0;

    double J_M1 = 0.0, J_M2 = 0.0, J_M3 = 0.0;
    double J_baseline = 0.0;     // always-active, ungated by any mechanism

    const int checkpoint_step = std::max(4, D / 4);

    double prev_u = 0.0;

    for (int i = 0; i < D; ++i) {
        const double tau = static_cast<double>(i + 1) / D;
        const double xi  = clamp01(x[i]);

        // ── Control derivation ────────────────────────────────────────────
        double u   = xi;
        double r_i = xi;

        if (config_.M2_enable) {
            const double xm1 = clamp01(safe_x(x, i - 1));
            const double xm2 = clamp01(safe_x(x, i - 2));
            const double xm3 = clamp01(safe_x(x, i - 3));
            const double xp1 = clamp01(safe_x(x, i + 1));

            r_i = clamp01(0.43 * xi + 0.21 * xm1 + 0.16 * xm2
                        + 0.10 * xm3 + 0.10 * xp1);

            const double phi = 0.40 * static_cast<double>(i + 1)
                             + 1.6  * xm1 - 1.1 * xm2;

            u = banded_control(r_i, phi);
        }

        // ── Airflow / cooling thermal outputs ─────────────────────────────
        const double F_i = flowcap_[i] * (0.22 + 0.78 * u);
        const double Q_i = coolcap_[i] * (0.10 + 0.90 * u);

        // ── M1 — thermal dynamics + hotspot debt ──────────────────────────
        const double H_feedback = config_.M1_enable
                                ? hotspot_temp_coupling_ * H
                                : 0.0;

        const double dT = ambient_relax_       * (ambient_[i] - T)
                        + thermal_load_gain_   * workload_[i]
                        + H_feedback
                        - thermal_cooling_gain_* Q_i
                        - thermal_airflow_gain_* F_i;

        const double T_next = soft_clip(T + temp_update_scale_ * dT, 16.0, 40.0);

        const double E_plus = std::max(0.0, T_next - tlimit_[i]);

        // ── BASELINE (ALWAYS active) — provides D-scaling difficulty ───────
        //  Per-stage: mild excess penalty, energy cost, mild ruggedness.
        //  These accumulate with D so the problem becomes non-trivial in
        //  large dimensions even when no mechanism is enabled.
        double J_base_stage = 0.0;
        J_base_stage += base_excess_weight_ * criticality_[i] * sqr(E_plus);

        // ── M1 — extra hotspot-debt dynamics on top of baseline ───────────
        if (config_.M1_enable) {
            J_M1 += m1_extra_excess_weight_ * criticality_[i] * sqr(E_plus);
            H = hotspot_decay_ * H
              + hotspot_gain_coeff_ * E_plus
              + hotspot_sat_coeff_  * sat(T_next - tlimit_[i], 1.0);
        }
        T = T_next;

        // ── M2 — banded actuation ruggedness ──────────────────────────────
        if (config_.M2_enable) {
            J_M2 += m2_w1_ * sin2(10.0 * PI * u + 3.0 * tau)
                  + m2_w2_ * sin2(17.0 * PI * (u - 0.5 * prev_u) + 1.5 * tau)
                  + m2_w3_ * sin2( 7.0 * PI * r_i + 0.4 * static_cast<double>(i + 1));
        }

        // ── Cooling power (always computed; baseline energy cost and M3) ─
        const double P_cool = power_base_
                            + power_q_linear_ * Q_i
                            + power_q_quad_   * sqr(Q_i)
                            + power_f_quad_   * sqr(F_i);
        cum_power += P_cool;

        // ── BASELINE: per-stage energy cost + mild ruggedness ─────────────
        J_base_stage += base_energy_weight_ * tariff_[i] * P_cool;
        J_base_stage += base_rugged_weight_
                      * sin2(base_rugged_freq_ * PI * u + 2.0 * tau);

        // Add this stage's baseline cost to the total (we book it under J_M1
        // accumulator simply for bookkeeping — semantically it is always-on).
        J_M1 += J_base_stage;

        // ── M3 — budget + deceptive corridor ─────────────────────────────
        if (config_.M3_enable) {
            const double gamma_center = soft_clip(
                0.17
                + 0.50 * pref_[i]
                + 0.07 * std::sin(7.0  * PI * tau + 0.25)
                + 0.05 * std::sin(15.0 * PI * tau + 0.65),
                0.06, 0.94);
            const double delta_half = 0.050 + 0.015 * sin2(6.0 * PI * tau + 0.45);
            const double e_corr     = std::fabs(u - gamma_center) - delta_half;

            C = comfort_decay_ * C + sqr(std::max(0.0, e_corr));
            J_M3 += m3_corridor_stage_weight_ * sqr(C);

            // Multimodal regulation cost: breaks the smooth corridor bowl into
            // multiple local minima at discrete actuation levels. The phase
            // depends on the stage coordinate (so optima shift across stages)
            // and on accumulated comfort debt (so the multimodality intensifies
            // when the trajectory drifts from the corridor centre).
            J_M3 += m3_multimodal_weight_
                  * sin2(m3_multimodal_freq_ * PI * u + 1.0 * tau)
                  * (1.0 + 0.3 * C);

            const bool at_checkpoint = (((i + 1) % checkpoint_step) == 0)
                                   ||  (i == D - 1);
            if (at_checkpoint) {
                const double rho_target = soft_clip(
                    0.30
                    + 0.52 * (static_cast<double>(i + 1) / D)
                    + 0.04 * std::sin(2.0 * PI * tau + 0.40),
                    0.18, 0.92);
                const double rho_hat = cum_power
                                     / std::max(1e-12, carbon_budget_);
                const double dev = rho_hat - rho_target;
                J_M3 += checkpoint_weight_ * sqr(dev)
                      / (checkpoint_saturation_ + sqr(dev));
            }
        }

        prev_u = u;
    }

    // ── Terminal terms ──────────────────────────────────────────────────────
    // BASELINE terminal: always-active temperature target accuracy.
    J_M1 += base_terminal_temp_weight_ * sqr(T - t_target_);

    // M1 extra terminal: hotspot debt resolution.
    if (config_.M1_enable) {
        J_M1 += m1_terminal_hotspot_weight_ * sqr(H);
    }
    if (config_.M3_enable) {
        J_M3 += budget_overrun_weight_     * sqr(std::max(0.0, cum_power - carbon_budget_));
        J_M3 += m3_corridor_term_weight_   * sqr(C);
    }

    return J_M1 + J_M2 + J_M3;
}

double DataCenterCooling::evaluate_core(const Vec& x)
{
    return evaluate_with_point(x);
}

void DataCenterCooling::gradient_core(const Vec& x, Vec& g)
{
    const int    D  = static_cast<int>(x.size());
    const double h  = 1e-6;
    const double f0 = evaluate_with_point(x);
    Vec xp = x;
    g.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        const double saved = xp[i];
        xp[i] = std::min(1.0, saved + h);
        const double fp = evaluate_with_point(xp);
        g[i] = (fp - f0) / (xp[i] - saved + 1e-30);
        xp[i] = saved;
    }
}

// ============================================================================
//  Configuration
// ============================================================================
void DataCenterCooling::load_config()
{
    // Mechanism flags — default: all ON
    config_.M1_enable = true;
    config_.M2_enable = true;
    config_.M3_enable = true;

    // Initial and target states
    t_init_   = 24.3;
    t_target_ = 22.4;
    h_init_   = 0.18;
    c_init_   = 0.08;

    // Baseline cost (ALWAYS active)
    // Very small per-stage contributions so smooth ALLOFF stays easy up to D≈40.
    // The primary difficulty driver is the terminal temperature term, which
    // requires coordinating the full control trajectory regardless of dimension.
    base_excess_weight_         = 0.3;   // tiny per-stage thermal-excess
    base_terminal_temp_weight_  = 2.0;   // terminal (T-T*)^2 — main baseline cost
    base_energy_weight_         = 0.1;   // negligible per-stage energy cost
    base_rugged_weight_         = 0.0;   // NO baseline ruggedness — keeps ALLOFF smooth
    base_rugged_freq_           = 5.0;   // irrelevant when weight=0, kept for sensitivity

    // M1 — sequential thermal coupling (strong, clearly visible)
    hotspot_decay_              = 0.75;
    hotspot_gain_coeff_         = 0.02;
    hotspot_sat_coeff_          = 0.01;
    hotspot_temp_coupling_      = 0.35;
    thermal_load_gain_          = 2.8;
    thermal_cooling_gain_       = 1.16;
    thermal_airflow_gain_       = 0.19;
    ambient_relax_              = 0.58;
    temp_update_scale_          = 0.15;
    m1_extra_excess_weight_     = 9.0;   // strong extra per-stage cost
    m1_terminal_hotspot_weight_ = 7.0;   // strong terminal debt cost

    // M2 — banded actuation ruggedness (moderate)
    m2_w1_ = 0.45;
    m2_w2_ = 0.35;
    m2_w3_ = 0.18;

    // M3 — budget and deceptive corridor (moderate)
    comfort_decay_             = 0.80;
    budget_base_per_stage_     = 20.0;
    budget_workload_factor_    = 5.0;
    checkpoint_weight_         = 14.0;
    checkpoint_saturation_     = 0.030;
    budget_overrun_weight_     = 9.0;
    m3_corridor_stage_weight_  = 4.0;
    m3_corridor_term_weight_   = 14.0;
    m3_multimodal_weight_      = 1.8;
    m3_multimodal_freq_        = 9.0;
    power_base_                = 2.4;
    power_q_linear_            = 0.75;
    power_q_quad_              = 0.16;
    power_f_quad_              = 0.09;

    config_loaded_ = true;
}

void DataCenterCooling::load_main_config_overrides()
{
    const std::string candidates[] = {
        "optimsolution.cfg",
        "./optimsolution.cfg",
        "../optimsolution.cfg",
    };
    std::ifstream in;
    for (const auto& path : candidates) {
        in.open(path);
        if (in.good()) { loaded_config_path_ = path; break; }
        in.clear();
    }
    if (!in.good()) return;

    std::string line, section;
    while (std::getline(in, line)) {
        const auto cut = line.find_first_of(";#");
        if (cut != std::string::npos) line.erase(cut);
        line = trim_copy(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = to_lower_copy(line.substr(1, line.size() - 2));
            continue;
        }
        if (section != "datacentercooling") continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key   = to_lower_copy(trim_copy(line.substr(0, eq)));
        const std::string value = trim_copy(line.substr(eq + 1));
        const double dval = std::atof(value.c_str());

        // ── Mechanism flags ────────────────────────────────────────────────
        if      (key == "m1_enable") config_.M1_enable = parse_bool_value(value, config_.M1_enable);
        else if (key == "m2_enable") config_.M2_enable = parse_bool_value(value, config_.M2_enable);
        else if (key == "m3_enable") config_.M3_enable = parse_bool_value(value, config_.M3_enable);

        // ── Initial / target states ───────────────────────────────────────
        else if (key == "t_init")     t_init_   = dval;
        else if (key == "t_target")   t_target_ = dval;
        else if (key == "h_init")     h_init_   = dval;
        else if (key == "c_init")     c_init_   = dval;

        // ── Baseline (ALWAYS active) ──────────────────────────────────────
        else if (key == "base_excess_weight")          base_excess_weight_         = dval;
        else if (key == "base_terminal_temp_weight")   base_terminal_temp_weight_  = dval;
        else if (key == "base_energy_weight")          base_energy_weight_         = dval;
        else if (key == "base_rugged_weight")          base_rugged_weight_         = dval;
        else if (key == "base_rugged_freq")            base_rugged_freq_           = dval;

        // ── M1 — sequential thermal coupling (extra when ON) ──────────────
        else if (key == "hotspot_decay")              hotspot_decay_              = dval;
        else if (key == "hotspot_gain_coeff")         hotspot_gain_coeff_         = dval;
        else if (key == "hotspot_sat_coeff")          hotspot_sat_coeff_          = dval;
        else if (key == "hotspot_temp_coupling")      hotspot_temp_coupling_      = dval;
        else if (key == "thermal_load_gain")          thermal_load_gain_          = dval;
        else if (key == "thermal_cooling_gain")       thermal_cooling_gain_       = dval;
        else if (key == "thermal_airflow_gain")       thermal_airflow_gain_       = dval;
        else if (key == "ambient_relax")              ambient_relax_              = dval;
        else if (key == "temp_update_scale")          temp_update_scale_          = dval;
        else if (key == "m1_extra_excess_weight")     m1_extra_excess_weight_     = dval;
        else if (key == "m1_terminal_hotspot_weight") m1_terminal_hotspot_weight_ = dval;

        // ── M2 — banded actuation ruggedness ──────────────────────────────
        else if (key == "m2_w1") m2_w1_ = dval;
        else if (key == "m2_w2") m2_w2_ = dval;
        else if (key == "m2_w3") m2_w3_ = dval;

        // ── M3 — budget + deceptive corridor ──────────────────────────────
        else if (key == "comfort_decay")              comfort_decay_              = dval;
        else if (key == "budget_base_per_stage")      budget_base_per_stage_      = dval;
        else if (key == "budget_workload_factor")     budget_workload_factor_     = dval;
        else if (key == "checkpoint_weight")          checkpoint_weight_          = dval;
        else if (key == "checkpoint_saturation")      checkpoint_saturation_      = dval;
        else if (key == "budget_overrun_weight")      budget_overrun_weight_      = dval;
        else if (key == "m3_corridor_stage_weight")   m3_corridor_stage_weight_   = dval;
        else if (key == "m3_corridor_term_weight")    m3_corridor_term_weight_    = dval;
        else if (key == "m3_multimodal_weight")       m3_multimodal_weight_       = dval;
        else if (key == "m3_multimodal_freq")         m3_multimodal_freq_         = dval;
        else if (key == "power_base")                 power_base_                 = dval;
        else if (key == "power_q_linear")             power_q_linear_             = dval;
        else if (key == "power_q_quad")               power_q_quad_               = dval;
        else if (key == "power_f_quad")               power_f_quad_               = dval;
    }
}

// ----------------------------------------------------------------------------
//  String / parsing utilities
// ----------------------------------------------------------------------------
std::string DataCenterCooling::trim_copy(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string DataCenterCooling::to_lower_copy(const std::string& s)
{
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return r;
}

bool DataCenterCooling::parse_bool_value(const std::string& value, bool default_value)
{
    const std::string v = to_lower_copy(trim_copy(value));
    if (v == "1" || v == "true"  || v == "yes" || v == "on")  return true;
    if (v == "0" || v == "false" || v == "no"  || v == "off") return false;
    return default_value;
}

std::string DataCenterCooling::dirname_copy(const std::string& path)
{
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

std::string DataCenterCooling::join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    const char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

} // namespace optimsolution
