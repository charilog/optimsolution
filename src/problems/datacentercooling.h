// ============================================================================
//  datacentercooling.h
//  Adaptive Data Center Thermal Orchestration (ADCTO) Benchmark — v5
//
//  Three-mechanism design matching Section 2.2 of the paper:
//      M1 — Sequential Thermal State Coupling     (hotspot debt)
//      M2 — Discontinuous Banded Actuation        (multimodal landscape)
//      M3 — Cumulative Budget with Deceptive Corridor (global planning)
//
//  ALL coefficients are exposed as cfg-tunable parameters so that the
//  benchmark can be used for sensitivity analysis through the standard
//  OptimSolution sensitivity-evaluation pipeline. Default values match
//  the paper specification exactly.
// ============================================================================

#pragma once
#include "problem.h"
#include <string>
#include <vector>

namespace optimsolution {

class DataCenterCooling : public Problem {
public:
    DataCenterCooling();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    struct MechanismConfig {
        bool M1_enable;
        bool M2_enable;
        bool M3_enable;
    };

    // ── Exogenous profiles (precomputed at init) ────────────────────────────
    std::vector<double> workload_;
    std::vector<double> ambient_;
    std::vector<double> tariff_;
    std::vector<double> tlimit_;
    std::vector<double> coolcap_;
    std::vector<double> flowcap_;
    std::vector<double> pref_;
    std::vector<double> criticality_;
    double              carbon_budget_;

    // ── Initial / target states ─────────────────────────────────────────────
    double t_init_;            // 24.3
    double t_target_;          // 22.4
    double h_init_;            // 0.18
    double c_init_;            // 0.08

    // ── Baseline cost (ALWAYS active — provides D-scaling difficulty) ──────
    double base_excess_weight_;        // 3.0  per-stage Crit·(E+)^2
    double base_terminal_temp_weight_; // 10.0 terminal (T-T*)^2
    double base_energy_weight_;        // 1.0  per-stage tariff·P_cool
    double base_rugged_weight_;        // 0.10 mild ruggedness (D-scaling)
    double base_rugged_freq_;          // 5.0  base ruggedness frequency

    // ── M1: Sequential thermal state coupling ───────────────────────────────
    double hotspot_decay_;             // 0.75
    double hotspot_gain_coeff_;        // 0.02
    double hotspot_sat_coeff_;         // 0.01
    double hotspot_temp_coupling_;     // 0.35
    double thermal_load_gain_;         // 2.8
    double thermal_cooling_gain_;      // 1.16
    double thermal_airflow_gain_;      // 0.19
    double ambient_relax_;             // 0.58
    double temp_update_scale_;         // 0.15
    double m1_extra_excess_weight_;    // 6.0  EXTRA per-stage E+^2 (on top of baseline)
    double m1_terminal_hotspot_weight_;// 5.0  terminal H^2

    // ── M2: Banded actuation ruggedness ─────────────────────────────────────
    double m2_w1_;                     // 0.50
    double m2_w2_;                     // 0.40
    double m2_w3_;                     // 0.20

    // ── M3: Budget + deceptive corridor ─────────────────────────────────────
    double comfort_decay_;             // 0.80
    double budget_base_per_stage_;     // 20.0
    double budget_workload_factor_;    // 5.0
    double checkpoint_weight_;         // 15.0
    double checkpoint_saturation_;     // 0.030
    double budget_overrun_weight_;     // 10.0
    double m3_corridor_stage_weight_;  // 4.0
    double m3_corridor_term_weight_;   // 12.0
    double m3_multimodal_weight_;      // 1.5  — multimodal regulation amplitude
    double m3_multimodal_freq_;        // 9.0  — multimodal oscillation count in u∈[0,1]
    double power_base_;                // 2.4
    double power_q_linear_;            // 0.75
    double power_q_quad_;              // 0.16
    double power_f_quad_;              // 0.09

    // ── Always-active baseline (provides natural D-scaling) ─────────────────
    double baseline_tracking_weight_;  // 0.3
    double baseline_rugged_weight_;    // 0.10
    double baseline_rugged_freq_;      // 3.0

    MechanismConfig config_;
    bool            config_loaded_;
    std::string     loaded_config_path_;

    double safe_x(const Vec& x, int i) const;
    double evaluate_with_point(const Vec& x) const;
    void   load_config();
    void   load_main_config_overrides();
    void   refresh_metadata();

    static std::string trim_copy(const std::string& s);
    static std::string to_lower_copy(const std::string& s);
    static bool        parse_bool_value(const std::string& value, bool default_value);
    static std::string dirname_copy(const std::string& path);
    static std::string join_path(const std::string& a, const std::string& b);
};

} // namespace optimsolution
