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
        bool recursive_state_enable;        // path-dependent debt carry-over + terminal debt penalties
        bool oscillatory_ruggedness_enable; // sinusoidal ruggedness in cost + thermal/moisture perturbations
        bool threshold_carbon_enable;       // checkpoint power-budget penalties + carbon overrun penalty
        bool deferred_reliability_enable;   // deferred debt cost + reliability cost terms
        bool neighbor_coupling_enable;      // multi-step blending of control inputs + banded phase coupling
        bool preferred_corridor_enable;     // comfort-corridor tracking + preferred operating-window penalty
    };

    std::vector<double> workload_;
    std::vector<double> ambient_;
    std::vector<double> tariff_;
    std::vector<double> humidity_;
    std::vector<double> tlimit_;
    std::vector<double> coolcap_;
    std::vector<double> flowcap_;
    std::vector<double> pref_;
    std::vector<double> criticality_;

    double t0_;
    double m0_;
    double ttarget_;
    double mtarget_;
    double carbon_budget_;

    MechanismConfig config_;
    bool            config_loaded_;
    std::string     loaded_config_path_;

    double safe_x(const Vec& x, int i) const;
    double evaluate_with_point(const Vec& x) const;
    void   load_config();
    void   refresh_metadata();

    static std::string trim_copy(const std::string& s);
    static std::string to_lower_copy(const std::string& s);
    static bool        parse_bool_value(const std::string& value, bool default_value);
    static std::string dirname_copy(const std::string& path);
    static std::string join_path(const std::string& a, const std::string& b);
};

} // namespace optimsolution
