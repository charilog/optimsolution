#pragma once
#include "problem.h"
#include <string>

namespace optimsolution {

class WeatherIrrigation : public Problem {
public:
    WeatherIrrigation();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    struct MechanismConfig {
        bool recursive_storage_enable;
        bool oscillatory_efficiency_enable;
        bool threshold_losses_enable;
        bool multiplicative_yield_enable;
        bool neighbor_coupling_enable;
        bool preferred_window_enable;
    };

    double objective_value(const Vec& x) const;
    void   build_weather_profile(int dim);
    void   load_config();
    void   load_main_config_overrides();
    void   apply_config_key(const std::string& key, const std::string& value);
    void   refresh_metadata();

    static double clamp_value(double v, double lo, double hi);
    static double sqr(double v);
    static std::string trim_copy(const std::string& s);
    static std::string to_lower_copy(const std::string& s);
    static bool parse_bool_value(const std::string& value, bool default_value);
    static std::string dirname_copy(const std::string& path);
    static std::string join_path(const std::string& a, const std::string& b);

private:
    Vec eto_;             // Reference evapotranspiration per stage
    Vec kc_;              // Crop coefficient per stage
    Vec etc_;             // Crop evapotranspiration demand per stage
    Vec rain_;            // Expected rainfall per stage
    Vec ky_;              // Yield sensitivity to water stress per stage
    Vec heat_;            // Heat-stress intensity per stage
    Vec wind_;            // Wind exposure per stage
    Vec smax_;            // Maximum soil-water storage per stage
    Vec soil_retention_;  // Fraction of surplus retained in the root zone
    Vec eff_base_;        // Baseline irrigation efficiency per stage
    Vec tariff_;          // Stage-dependent water/energy tariff
    Vec freq_;            // Stage-dependent ruggedness frequency
    Vec phase_;           // Stage-dependent ruggedness phase
    Vec pref_;            // Preferred irrigation window center (soft target)

    double max_irrigation_;              // 80.0 mm (paper: x_i ∈ [0, 80])
    double initial_storage_ratio_;       // 0.35 (paper: S1 = 0.35 * Smax_1)
    double yield_reward_;                // 350.0 (paper: −350 * Y_rel)
    double terminal_storage_weight_;     // 1.2 (paper: P_term = 1.2 * (...)²)
    double peak_weight_;                 // 0.035 (paper: P_peak = 0.035 * Σ...)
    double seasonal_budget_target_;      // 28*D (paper: BD = 28D)
    double seasonal_budget_weight_;      // 0.010 (paper: P_budget = 0.010 * (...)²)

    // ── Group A: High-priority tunable parameters ────────────────────────
    double osc_eff_amplitude_;       // oscillatory efficiency amplitude in η_i(x)          default: 0.22
    double resonance_weight_;        // resonance penalty weight in P_res                   default: 8.0
    double osc_freq_base_;           // oscillation frequency base component (freq_i)       default: 0.22
    double osc_freq_amp_;            // oscillation frequency amplitude (freq_i)            default: 0.09
    double ky_base_;                 // yield-stress sensitivity base (Ky_i)                default: 0.85
    double ky_amp_;                  // yield-stress sensitivity amplitude (Ky_i)           default: 0.55

    // ── Group B: Medium-priority tunable parameters ──────────────────────
    double smax_base_;               // max soil storage base (S_max)                       default: 50.0
    double smax_amp_;                // max soil storage amplitude (S_max)                  default: 20.0
    double deficit_weight_;          // deficit penalty weight in P_def                     default: 0.12
    double pref_weight_;             // preferred-window penalty base weight in P_win       default: 0.014

    // ── Group C: Lower-priority tunable parameters ───────────────────────
    double soil_retention_base_;     // soil water retention coefficient base (ρ_i)         default: 0.52
    double soil_retention_amp_;      // soil water retention coefficient amplitude (ρ_i)    default: 0.18
    double smoothness_weight_;       // smoothness penalty weight in P_smooth               default: 0.06
    double peak_threshold_;          // peak-load threshold in P_peak                       default: 62.0

    MechanismConfig config_;
    bool config_loaded_;
    std::string loaded_config_path_;
};

} // namespace optimsolution
