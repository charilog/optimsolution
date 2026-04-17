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

    double max_irrigation_;
    double initial_storage_ratio_;
    double yield_reward_;
    double terminal_storage_weight_;
    double peak_weight_;
    double seasonal_budget_target_;
    double seasonal_budget_weight_;

    MechanismConfig config_;
    bool config_loaded_;
    std::string loaded_config_path_;
};

} // namespace optimsolution
