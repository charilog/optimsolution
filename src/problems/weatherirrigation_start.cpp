#include "weatherirrigation.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>


namespace optimsolution {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr int    kMinDim = 6;
}

WeatherIrrigation::WeatherIrrigation()
    : max_irrigation_(80.0),
      initial_storage_ratio_(0.42),
      yield_reward_(480.0),
      terminal_storage_weight_(8.0),
      peak_weight_(1.40),
      seasonal_budget_target_(0.0),
      seasonal_budget_weight_(0.018),
      config_{true, true, true, true, true, true},
      config_loaded_(false),
      loaded_config_path_()
{
    setName("weatherirrigation");
    refresh_metadata();
    setModality("highly multimodal");
    setSeparability("strongly non-separable");
    setCategory("real-world weather-aware agricultural optimization");
}

void WeatherIrrigation::init(int dim)
{
    if (dim < kMinDim)
        dim = kMinDim;

    load_config();
    refresh_metadata();

    Problem::init(dim);

    Vec l(dim, 0.0);
    Vec u(dim, max_irrigation_);
    setBounds(l, u);

    build_weather_profile(dim);
}

void WeatherIrrigation::refresh_metadata()
{
    std::string full_name = "Hard weather-aware irrigation scheduling problem";

    full_name += config_loaded_ ? " [cfg loaded]" : " [cfg not found: defaults]";
    full_name += " [RS=" + std::string(config_.recursive_storage_enable ? "1" : "0");
    full_name += ",OE=" + std::string(config_.oscillatory_efficiency_enable ? "1" : "0");
    full_name += ",TL=" + std::string(config_.threshold_losses_enable ? "1" : "0");
    full_name += ",MY=" + std::string(config_.multiplicative_yield_enable ? "1" : "0");
    full_name += ",NC=" + std::string(config_.neighbor_coupling_enable ? "1" : "0");
    full_name += ",PW=" + std::string(config_.preferred_window_enable ? "1" : "0") + "]";

    setFullName(full_name);
}

void WeatherIrrigation::build_weather_profile(int dim)
{
    eto_.resize(dim);
    kc_.resize(dim);
    etc_.resize(dim);
    rain_.resize(dim);
    ky_.resize(dim);
    heat_.resize(dim);
    wind_.resize(dim);
    smax_.resize(dim);
    soil_retention_.resize(dim);
    eff_base_.resize(dim);
    tariff_.resize(dim);
    freq_.resize(dim);
    phase_.resize(dim);
    pref_.resize(dim);

    seasonal_budget_target_ = 0.0;

    const double stage_days = 150.0 / static_cast<double>(dim);

    for (int i = 0; i < dim; ++i) {
        const double s = static_cast<double>(i + 1) / static_cast<double>(dim + 1);
        const double sin1 = std::sin(kPi * s);
        const double cos1 = std::cos(kPi * s);
        const double sin2 = std::sin(2.0 * kPi * s + 0.35);
        const double cos2 = std::cos(2.0 * kPi * s - 0.20);

        eto_[i] = 3.0 + 3.8 * sin1 * sin1 + 0.4 * std::max(0.0, sin2);
        kc_[i]  = 0.45 + 0.85 * sin1;
        etc_[i] = eto_[i] * kc_[i] * stage_days;
        rain_[i] = 5.0 + 24.0 * cos1 * cos1 + 6.0 * std::max(0.0, sin2);
        ky_[i] = 0.90 + 0.70 * sin1;
        heat_[i] = 0.15 + 0.85 * std::pow(std::max(0.0, sin1), 1.35);
        wind_[i] = 0.25 + 0.75 * (0.5 + 0.5 * cos2);
        smax_[i] = 26.0 + 22.0 * (0.5 + 0.5 * cos1) + 4.0 * sin2 * sin2;
        soil_retention_[i] = 0.58 + 0.24 * (0.5 + 0.5 * cos1);
        eff_base_[i] = clamp_value(0.72 - 0.10 * wind_[i] - 0.06 * heat_[i], 0.48, 0.76);
        tariff_[i] = 0.62 + 0.22 * (0.5 + 0.5 * sin2) + 0.12 * heat_[i];
        freq_[i] = 0.18 + 0.11 * (0.5 + 0.5 * cos2);
        phase_[i] = 0.60 * kPi * s + 0.35 * sin2;
        pref_[i] = 15.0 + 22.0 * sin1 + 4.0 * cos2;

        seasonal_budget_target_ += std::max(0.0, 0.58 * (etc_[i] - 0.55 * rain_[i]));
    }
}

void WeatherIrrigation::load_config()
{
    config_.recursive_storage_enable = true;
    config_.oscillatory_efficiency_enable = true;
    config_.threshold_losses_enable = true;
    config_.multiplicative_yield_enable = true;
    config_.neighbor_coupling_enable = true;
    config_.preferred_window_enable = true;
    config_loaded_ = false;
    loaded_config_path_.clear();

    std::vector<std::string> candidate_paths;
    candidate_paths.reserve(64);

    const std::vector<std::string> prefixes = {
        "",
        ".",
        "..",
        "../..",
        "../../..",
        "../../../..",
        "../../../../.."
    };

    const std::vector<std::string> suffixes = {
        "weatherirrigation.cfg",
        "cfg/weatherirrigation.cfg",
        "problems/weatherirrigation.cfg",
        "src/problems/weatherirrigation.cfg",
        "src/problems/realworld/weatherirrigation.cfg",
        "src/weatherirrigation.cfg"
    };

    for (const std::string& prefix : prefixes) {
        for (const std::string& suffix : suffixes) {
            candidate_paths.push_back(prefix.empty() ? suffix : join_path(prefix, suffix));
        }
    }

#ifdef __FILE__
    {
        const std::string source_dir = dirname_copy(__FILE__);
        if (!source_dir.empty()) {
            candidate_paths.push_back(join_path(source_dir, "weatherirrigation.cfg"));
            candidate_paths.push_back(join_path(source_dir, "../weatherirrigation.cfg"));
            candidate_paths.push_back(join_path(source_dir, "../../weatherirrigation.cfg"));
        }
    }
#endif

    std::ifstream fin;
    for (const std::string& path : candidate_paths) {
        fin.close();
        fin.clear();
        fin.open(path.c_str());
        if (fin.is_open() && fin.good()) {
            config_loaded_ = true;
            loaded_config_path_ = path;
            break;
        }
    }

    if (!config_loaded_) {
        return;
    }

    std::string line;
    while (std::getline(fin, line)) {
        std::size_t comment_pos = line.find_first_of(";#");
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos);
        }

        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }

        if (line.front() == '[' && line.back() == ']') {
            continue;
        }

        std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = to_lower_copy(trim_copy(line.substr(0, eq)));
        const std::string value = trim_copy(line.substr(eq + 1));

        if (key == "recursive_storage_enable") {
            config_.recursive_storage_enable = parse_bool_value(value, config_.recursive_storage_enable);
        }
        else if (key == "oscillatory_efficiency_enable") {
            config_.oscillatory_efficiency_enable = parse_bool_value(value, config_.oscillatory_efficiency_enable);
        }
        else if (key == "threshold_losses_enable") {
            config_.threshold_losses_enable = parse_bool_value(value, config_.threshold_losses_enable);
        }
        else if (key == "multiplicative_yield_enable") {
            config_.multiplicative_yield_enable = parse_bool_value(value, config_.multiplicative_yield_enable);
        }
        else if (key == "neighbor_coupling_enable") {
            config_.neighbor_coupling_enable = parse_bool_value(value, config_.neighbor_coupling_enable);
        }
        else if (key == "preferred_window_enable") {
            config_.preferred_window_enable = parse_bool_value(value, config_.preferred_window_enable);
        }
    }
}

double WeatherIrrigation::clamp_value(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

double WeatherIrrigation::sqr(double v)
{
    return v * v;
}

std::string WeatherIrrigation::trim_copy(const std::string& s)
{
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }

    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(begin, end - begin);
}

std::string WeatherIrrigation::to_lower_copy(const std::string& s)
{
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool WeatherIrrigation::parse_bool_value(const std::string& value, bool default_value)
{
    const std::string v = to_lower_copy(trim_copy(value));

    if (v == "1" || v == "true" || v == "yes" || v == "on") {
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "off") {
        return false;
    }

    return default_value;
}

std::string WeatherIrrigation::dirname_copy(const std::string& path)
{
    const std::size_t p = path.find_last_of("/\\");
    if (p == std::string::npos) {
        return std::string();
    }
    return path.substr(0, p);
}

std::string WeatherIrrigation::join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) {
        return b;
    }
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') {
        return a + b;
    }
    return a + "/" + b;
}

double WeatherIrrigation::objective_value(const Vec& x) const
{
    const int D = dimension();

    double storage = config_.recursive_storage_enable ? (initial_storage_ratio_ * smax_[0]) : 0.0;

    double water_cost = 0.0;
    double pump_cost = 0.0;
    double deficit_term = 0.0;
    double excess_term = 0.0;
    double smoothness_term = 0.0;
    double resonance_term = 0.0;
    double interaction_term = 0.0;
    double pref_term = 0.0;
    double peak_load = 0.0;
    double seasonal_total = 0.0;
    double yield_signal = config_.multiplicative_yield_enable ? 1.0 : 0.0;

    for (int i = 0; i < D; ++i) {
        const double xi = clamp_value(x[i], 0.0, max_irrigation_);
        const double xim1 = (i > 0)     ? clamp_value(x[i - 1], 0.0, max_irrigation_) : 0.0;
        const double xip1 = (i + 1 < D) ? clamp_value(x[i + 1], 0.0, max_irrigation_) : 0.0;

        const double network_load = config_.neighbor_coupling_enable
                                  ? (0.60 * xi + 0.25 * xim1 + 0.15 * xip1)
                                  : xi;
        peak_load = std::max(peak_load, network_load);

        double eff_mult = 1.0;
        if (config_.oscillatory_efficiency_enable) {
            eff_mult -= 0.22 * sqr(std::sin(freq_[i] * xi + phase_[i]));
            eff_mult -= 0.10 * sqr(std::sin(0.07 * network_load + 0.50 * phase_[i]));
        }
        eff_mult = clamp_value(eff_mult, 0.45, 1.0);
        const double effective_irrigation = eff_base_[i] * eff_mult * xi;

        const double storage_contribution = config_.recursive_storage_enable ? (0.55 * storage) : 0.0;
        const double available = rain_[i] + effective_irrigation + storage_contribution;
        const double demand = etc_[i];
        const double actual_et = std::min(demand, available);
        const double deficit = std::max(0.0, demand - available);
        const double surplus = std::max(0.0, available - demand);

        double deep_loss = 0.0;
        double recharge = 0.0;
        double next_storage = storage;
        if (config_.recursive_storage_enable || config_.threshold_losses_enable) {
            const double deep_threshold = 0.35 * smax_[i];
            if (config_.threshold_losses_enable) {
                const double deep_mod = config_.oscillatory_efficiency_enable
                                      ? (0.55 + 0.20 * wind_[i] + 0.15 * sqr(std::sin(0.11 * xi + phase_[i])))
                                      : (0.55 + 0.20 * wind_[i]);
                deep_loss = std::max(0.0, surplus - deep_threshold) * deep_mod;
            }

            if (config_.recursive_storage_enable) {
                recharge = soil_retention_[i] * std::max(0.0, surplus - deep_loss);
                const double evap_loss = (0.08 + 0.10 * heat_[i] + 0.04 * wind_[i]) * storage;
                next_storage = clamp_value(0.82 * storage + recharge - evap_loss, 0.0, smax_[i]);
            }
        }

        const double ratio = (demand > 1e-12) ? (actual_et / demand) : 1.0;
        const double stress = 1.0 - ratio;

        double stage_yield = 1.0
                           - ky_[i] * std::pow(stress, 1.35)
                           - 0.08 * heat_[i] * std::exp(-available / (0.35 * demand + 1.0));
        stage_yield = clamp_value(stage_yield, 0.02, 1.0);

        if (config_.multiplicative_yield_enable) {
            yield_signal *= stage_yield;
        }
        else {
            yield_signal += stage_yield;
        }

        const double weather_shock = config_.oscillatory_efficiency_enable
                                   ? (1.0 + 0.18 * heat_[i] * sqr(std::sin(0.09 * xi + 1.3 * wind_[i] + phase_[i])))
                                   : 1.0;
        const double pump_mod = config_.oscillatory_efficiency_enable
                              ? (1.0 + 0.35 * sqr(std::sin(0.08 * xi + 0.60 * phase_[i])))
                              : 1.0;
        const double preferred_threshold = config_.preferred_window_enable ? (pref_[i] + 18.0) : 58.0;
        const double runoff_trigger = (config_.neighbor_coupling_enable && config_.threshold_losses_enable)
                                    ? std::max(0.0, xi + 0.55 * xim1 - preferred_threshold)
                                    : 0.0;

        water_cost += tariff_[i] * xi;
        pump_cost += 0.0045 * sqr(network_load) * pump_mod;
        deficit_term += (0.12 + 0.06 * heat_[i]) * sqr(deficit) * weather_shock;
        excess_term += (0.05 + 0.05 * wind_[i]) * (sqr(deep_loss) + 0.30 * sqr(surplus));

        if (config_.oscillatory_efficiency_enable) {
            resonance_term += 8.0 * (1.0 + 0.70 * heat_[i]) * sqr(std::sin(freq_[i] * xi + phase_[i]));
        }

        if (config_.neighbor_coupling_enable) {
            const double interaction_mod = config_.oscillatory_efficiency_enable
                                         ? (1.0 + 0.25 * sqr(std::sin(0.15 * xi - 0.11 * xim1 + phase_[i])))
                                         : 1.0;
            interaction_term += (0.08 + 0.04 * wind_[i]) * sqr(runoff_trigger) * interaction_mod;

            if (i > 0) {
                const double dx = xi - xim1;
                smoothness_term += (0.06 + 0.02 * wind_[i]) * sqr(dx);
                interaction_term += 0.025 * xi * xim1;
            }
        }

        if (config_.preferred_window_enable) {
            const double pref_dev = xi - pref_[i];
            const double pref_mod = config_.oscillatory_efficiency_enable
                                  ? (1.0 + 0.35 * sqr(std::sin(0.12 * pref_dev + 0.5 * phase_[i])))
                                  : 1.0;
            pref_term += 0.015 * (1.0 + wind_[i]) * sqr(pref_dev) * pref_mod;
        }

        seasonal_total += xi;
        storage = next_storage;
    }

    if (!config_.multiplicative_yield_enable) {
        yield_signal /= static_cast<double>(D);
    }

    double terminal_storage_penalty = 0.0;
    if (config_.recursive_storage_enable) {
        const double terminal_target = 0.58 * smax_.back();
        terminal_storage_penalty = terminal_storage_weight_ * sqr(storage - terminal_target);
    }

    const double over_budget = std::max(0.0, seasonal_total - seasonal_budget_target_);
    const double under_budget = std::max(0.0, 0.72 * seasonal_budget_target_ - seasonal_total);
    const double seasonal_budget_penalty = seasonal_budget_weight_ * (sqr(over_budget) + 0.45 * sqr(under_budget));

    const double peak_penalty = config_.threshold_losses_enable
                              ? (peak_weight_ * sqr(std::max(0.0, peak_load - 34.0)))
                              : 0.0;

    return water_cost
         + pump_cost
         + deficit_term
         + excess_term
         + smoothness_term
         + resonance_term
         + interaction_term
         + pref_term
         + terminal_storage_penalty
         + seasonal_budget_penalty
         + peak_penalty
         - yield_reward_ * yield_signal;
}

double WeatherIrrigation::evaluate_core(const Vec& x)
{
    return objective_value(x);
}

void WeatherIrrigation::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);

    Vec xp = x;
    Vec xm = x;

    for (int i = 0; i < D; ++i) {
        const double h = 1e-5 * (1.0 + std::fabs(x[i]));

        xp[i] = x[i];
        xm[i] = x[i];

        const bool can_forward = (x[i] + h <= max_irrigation_);
        const bool can_backward = (x[i] - h >= 0.0);

        if (can_forward && can_backward) {
            xp[i] = x[i] + h;
            xm[i] = x[i] - h;
            g[i] = (objective_value(xp) - objective_value(xm)) / (2.0 * h);
        }
        else if (can_forward) {
            xp[i] = x[i] + h;
            g[i] = (objective_value(xp) - objective_value(x)) / h;
        }
        else if (can_backward) {
            xm[i] = x[i] - h;
            g[i] = (objective_value(x) - objective_value(xm)) / h;
        }
        else {
            g[i] = 0.0;
        }

        xp[i] = x[i];
        xm[i] = x[i];
    }
}

} // namespace optimsolution
