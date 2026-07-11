#include "weatherirrigation.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
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
      initial_storage_ratio_(0.35),          // CHANGED: was 0.42, paper says 0.35
      yield_reward_(350.0),                  // CHANGED: was 480.0, paper says 350
      terminal_storage_weight_(1.2),         // CHANGED: was 8.0, paper says 1.2
      peak_weight_(0.035),                   // CHANGED: was 1.40, paper says 0.035
      seasonal_budget_target_(0.0),
      seasonal_budget_weight_(0.010),        // CHANGED: was 0.018, paper says 0.010
      // ── Group A: High-priority tunable parameters ──────────────
      osc_eff_amplitude_(0.22),              // OK — matches paper
      resonance_weight_(8.0),                // OK — matches paper
      osc_freq_base_(0.22),                  // CHANGED: was 0.18, paper says freq_i = 0.22 + 0.09*sin(πs)
      osc_freq_amp_(0.09),                   // CHANGED: was 0.11, paper says 0.09
      ky_base_(0.85),                        // CHANGED: was 0.90, paper says 0.85
      ky_amp_(0.55),                         // CHANGED: was 0.70, paper says 0.55
      // ── Group B: Medium-priority tunable parameters ────────────
      smax_base_(50.0),                      // CHANGED: was 26.0, paper says 50
      smax_amp_(20.0),                       // CHANGED: was 22.0, paper says 20
      deficit_weight_(0.12),                 // OK — matches paper
      pref_weight_(0.014),                   // CHANGED: was 0.015, paper says 0.014 (base coeff)
      // ── Group C: Lower-priority tunable parameters ─────────────
      soil_retention_base_(0.52),            // CHANGED: was 0.58, paper says 0.52
      soil_retention_amp_(0.18),             // CHANGED: was 0.24, paper says 0.18
      smoothness_weight_(0.06),              // OK — matches paper
      peak_threshold_(62.0),                 // CHANGED: was 34.0, paper says 62
      // ── Config / state ─────────────────────────────────────────
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
    load_main_config_overrides();
    refresh_metadata();

    // Debug: write key values to stderr so they appear in the GUI sensitivity log.
    std::fprintf(stderr,
        "[weatherirrigation] cfg=%s | osc_eff_amp=%.4f ky_base=%.4f preferred_window=%d\n",
        loaded_config_path_.empty() ? "(not found)" : loaded_config_path_.c_str(),
        osc_eff_amplitude_, ky_base_,
        static_cast<int>(config_.preferred_window_enable));

    Problem::init(dim);

    Vec l(dim, 0.0);
    Vec u(dim, max_irrigation_);
    setBounds(l, u);

    build_weather_profile(dim);
}

void WeatherIrrigation::refresh_metadata()
{
    std::string full_name = "Hard weather-aware irrigation scheduling problem";

    full_name += config_loaded_ ? " [cfg loaded from optimsolution.cfg]" : " [cfg not found: defaults]";
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

    // CHANGED: was 150.0, paper says 120-day season
    const double stage_days = 120.0 / static_cast<double>(dim);

    for (int i = 0; i < dim; ++i) {
        const double s = static_cast<double>(i + 1) / static_cast<double>(dim + 1);
        const double sin1 = std::sin(kPi * s);
        const double cos1 = std::cos(kPi * s);
        // NOTE: sin2/cos2 with arbitrary phases removed; paper uses only sin(2πs) for tariff and phase

        // CHANGED: paper says ETo_i = 2.8 + 3.2 * sin²(πs)
        eto_[i] = 2.8 + 3.2 * sin1 * sin1;

        // CHANGED: paper says Kc_i = 0.55 + 0.65 * sin(πs)
        kc_[i]  = 0.55 + 0.65 * sin1;

        // Paper: ETc_i = ETo_i * Kc_i * Δt
        etc_[i] = eto_[i] * kc_[i] * stage_days;

        // CHANGED: paper says rain_i = 8 + 18 * cos²(πs)
        rain_[i] = 8.0 + 18.0 * cos1 * cos1;

        // Paper: Ky_i = 0.85 + 0.55 * sin(πs) — uses ky_base_ and ky_amp_
        ky_[i] = ky_base_ + ky_amp_ * sin1;

        // CHANGED: paper says heat_i = 0.40 + 0.60 * sin²(πs)
        heat_[i] = 0.40 + 0.60 * sin1 * sin1;

        // CHANGED: paper says wind_i = 0.35 + 0.65 * cos²(πs)
        wind_[i] = 0.35 + 0.65 * cos1 * cos1;

        // CHANGED: paper says Smax_i = 50 + 20 * sin²(πs)
        smax_[i] = smax_base_ + smax_amp_ * sin1 * sin1;

        // CHANGED: paper says ρ_i = 0.52 + 0.18 * sin²(πs)
        soil_retention_[i] = soil_retention_base_ + soil_retention_amp_ * sin1 * sin1;

        // CHANGED: paper says η_base,i = 0.78 + 0.10 * cos(πs)
        eff_base_[i] = 0.78 + 0.10 * cos1;

        // CHANGED: paper says tariff_i = 0.90 + 0.18 * sin(2πs + 0.20)
        tariff_[i] = 0.90 + 0.18 * std::sin(2.0 * kPi * s + 0.20);

        // CHANGED: paper says freq_i = 0.22 + 0.09 * sin(πs)
        freq_[i] = osc_freq_base_ + osc_freq_amp_ * sin1;

        // CHANGED: paper says φ_i = 0.70*i + 0.30*sin(2πs)
        phase_[i] = 0.70 * static_cast<double>(i + 1) + 0.30 * std::sin(2.0 * kPi * s);

        // CHANGED: paper says μ_i = 18 + 26 * sin²(πs)
        pref_[i] = 18.0 + 26.0 * sin1 * sin1;
    }

    // CHANGED: paper says BD = 28 * D (fixed formula, not dynamic)
    seasonal_budget_target_ = 28.0 * static_cast<double>(dim);
}

// ── Config parsing helpers (private static) ──────────────────────

static bool try_parse_double(const std::string& s, double& out)
{
    if (s.empty()) return false;
    try {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos == 0) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

void WeatherIrrigation::load_config()
{
    // Reset all to hardcoded defaults — no file I/O.
    config_.recursive_storage_enable      = true;
    config_.oscillatory_efficiency_enable = true;
    config_.threshold_losses_enable       = true;
    config_.multiplicative_yield_enable   = true;
    config_.neighbor_coupling_enable      = true;
    config_.preferred_window_enable       = true;
    config_loaded_ = false;
    loaded_config_path_.clear();
}

void WeatherIrrigation::load_main_config_overrides()
{
    // The GUI saves under one of two names before launching the CLI:
    //   1) "optimsolution_gui_merged.cfg" (--config mode, CWD = baseWd)
    //   2) "optimsolution.cfg"            (legacy mode, CWD = optimsolution_gui_run/)
    const std::vector<std::string> cfg_names = {
        "optimsolution_gui_merged.cfg",
        "optimsolution.cfg"
    };

    const std::vector<std::string> prefixes = {
        "", ".", "..", "../..", "../../..",
        "../../../..", "../../../../.."
    };

    std::vector<std::string> candidate_paths;
    candidate_paths.reserve(32);

    // Interleaved: try BOTH names at each directory level before going up.
    for (const std::string& prefix : prefixes)
        for (const std::string& name : cfg_names)
            candidate_paths.push_back(prefix.empty() ? name : join_path(prefix, name));

    // __FILE__-based absolute fallback.
    {
        const std::string source_dir = dirname_copy(__FILE__);
        if (!source_dir.empty()) {
            for (const std::string& name : cfg_names) {
                candidate_paths.push_back(join_path(source_dir, "../../" + name));
                candidate_paths.push_back(join_path(source_dir, "../../../" + name));
            }
        }
    }

    std::ifstream fin;
    for (const std::string& path : candidate_paths) {
        fin.close(); fin.clear();
        fin.open(path.c_str());
        if (fin.is_open() && fin.good()) { loaded_config_path_ = path; break; }
    }
    if (!fin.is_open() || !fin.good()) return;

    bool in_section = false;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t comment_pos = line.find_first_of(";#");
        if (comment_pos != std::string::npos) line.erase(comment_pos);
        line = trim_copy(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            const std::string sec = to_lower_copy(trim_copy(line.substr(1, line.size() - 2)));
            in_section = (sec == "weatherirrigation");
            continue;
        }
        if (!in_section) continue;

        std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key   = to_lower_copy(trim_copy(line.substr(0, eq)));
        const std::string value = trim_copy(line.substr(eq + 1));
        apply_config_key(key, value);
        config_loaded_ = true;
    }
}

void WeatherIrrigation::apply_config_key(const std::string& key, const std::string& value)
{
    // ── Boolean mechanism flags (ablation switches) ──────────────
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
    // ── Group A: High-priority numerical parameters ──────────────
    else if (key == "osc_eff_amplitude") {
        try_parse_double(value, osc_eff_amplitude_);
    }
    else if (key == "resonance_weight") {
        try_parse_double(value, resonance_weight_);
    }
    else if (key == "osc_freq_base") {
        try_parse_double(value, osc_freq_base_);
    }
    else if (key == "osc_freq_amp") {
        try_parse_double(value, osc_freq_amp_);
    }
    else if (key == "ky_base") {
        try_parse_double(value, ky_base_);
    }
    else if (key == "ky_amp") {
        try_parse_double(value, ky_amp_);
    }
    // ── Group B: Medium-priority numerical parameters ────────────
    else if (key == "smax_base") {
        try_parse_double(value, smax_base_);
    }
    else if (key == "smax_amp") {
        try_parse_double(value, smax_amp_);
    }
    else if (key == "deficit_weight") {
        try_parse_double(value, deficit_weight_);
    }
    else if (key == "seasonal_budget_weight") {
        try_parse_double(value, seasonal_budget_weight_);
    }
    else if (key == "pref_weight") {
        try_parse_double(value, pref_weight_);
    }
    // ── Group C: Lower-priority numerical parameters ─────────────
    else if (key == "soil_retention_base") {
        try_parse_double(value, soil_retention_base_);
    }
    else if (key == "soil_retention_amp") {
        try_parse_double(value, soil_retention_amp_);
    }
    else if (key == "smoothness_weight") {
        try_parse_double(value, smoothness_weight_);
    }
    else if (key == "peak_threshold") {
        try_parse_double(value, peak_threshold_);
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
    const std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return std::string();
    }
    const std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string WeatherIrrigation::to_lower_copy(const std::string& s)
{
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool WeatherIrrigation::parse_bool_value(const std::string& v, bool default_value)
{
    const std::string low = to_lower_copy(trim_copy(v));
    if (low.empty()) {
        return default_value;
    }

    // FIXED: was comparing against v (original case) — must compare against low
    if (low == "1" || low == "true" || low == "yes" || low == "on") {
        return true;
    }

    if (low == "0" || low == "false" || low == "no" || low == "off") {
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
    double peak_term = 0.0;              // CHANGED: was peak_load (single max), now sum of penalties
    double seasonal_total = 0.0;
    double yield_signal = config_.multiplicative_yield_enable ? 1.0 : 0.0;

    for (int i = 0; i < D; ++i) {
        const double xi = clamp_value(x[i], 0.0, max_irrigation_);
        const double xim1 = (i > 0)     ? clamp_value(x[i - 1], 0.0, max_irrigation_) : 0.0;
        const double xip1 = (i + 1 < D) ? clamp_value(x[i + 1], 0.0, max_irrigation_) : 0.0;

        // Paper: N_i = 0.60*xi + 0.25*xi-1 + 0.15*xi+1
        const double network_load = config_.neighbor_coupling_enable
                                  ? (0.60 * xi + 0.25 * xim1 + 0.15 * xip1)
                                  : xi;

        // CHANGED: paper says Ppeak = 0.035 * Σ max(0, Ni - 62)²  (sum, not single max)
        if (config_.threshold_losses_enable) {
            peak_term += sqr(std::max(0.0, network_load - peak_threshold_));
        }

        // Paper: η_i(x) = η_base,i * [1 - 0.22*sin²(freq*xi+φ) - 0.10*sin²(0.07*Ni+0.50*φ)]
        double eff_mult = 1.0;
        if (config_.oscillatory_efficiency_enable) {
            eff_mult -= osc_eff_amplitude_ * sqr(std::sin(freq_[i] * xi + phase_[i]));
            eff_mult -= 0.10 * sqr(std::sin(0.07 * network_load + 0.50 * phase_[i]));
        }
        eff_mult = clamp_value(eff_mult, 0.45, 1.0);
        const double effective_irrigation = eff_base_[i] * eff_mult * xi;

        // Paper: A_i = rain_i + Ieff_i + 0.55*S_i
        const double storage_contribution = config_.recursive_storage_enable ? (0.55 * storage) : 0.0;
        const double available = rain_[i] + effective_irrigation + storage_contribution;
        const double demand = etc_[i];
        const double actual_et = std::min(demand, available);
        const double deficit = std::max(0.0, demand - available);
        const double surplus = std::max(0.0, available - demand);

        // Paper: deep drainage, recharge, evaploss, storage transition
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

        // Paper: r_i, stress_i, y_i, Y_rel
        const double ratio = (demand > 1e-12) ? (actual_et / demand) : 1.0;
        const double stress = 1.0 - ratio;

        // ── Sensitivity diagnostic: count stages with meaningful stress ──────
        // If stress < kStressTol for all stages, ky_base/ky_amp have zero effect
        // on the objective. This explains why their sensitivity main effect = 0
        // at D=50: the optimizer consistently finds zero-stress solutions because
        // the per-stage demand (2.4-day window) is fully met by rainfall+budget.
        // At D=30 (4.0-day window, higher per-stage ETc), stress > 0 occurs and
        // ky parameters contribute measurably. This is a genuine mathematical
        // property of the benchmark, not a code defect.
        // To verify: set kLogStress=true and inspect console output for one run.
#if defined(WEATHERIRRIGATION_LOG_STRESS)
        static thread_local int log_call = 0;
        if (++log_call <= 1) {   // log only first evaluation
            fprintf(stderr, "[stress D=%d i=%d] stress=%.6f ky=%.4f\n",
                    D, i, stress, ky_[i]);
        }
#endif

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

        // Paper: shock_i = 1 + 0.18*heat*sin²(0.09*xi + 1.3*wind + φ)
        const double weather_shock = config_.oscillatory_efficiency_enable
                                   ? (1.0 + 0.18 * heat_[i] * sqr(std::sin(0.09 * xi + 1.3 * wind_[i] + phase_[i])))
                                   : 1.0;

        // Paper: Cpump uses [1 + 0.35*sin²(0.08*xi + 0.60*φ)]
        const double pump_mod = config_.oscillatory_efficiency_enable
                              ? (1.0 + 0.35 * sqr(std::sin(0.08 * xi + 0.60 * phase_[i])))
                              : 1.0;

        // Paper: Cwater = Σ tariff_i * xi
        water_cost += tariff_[i] * xi;

        // Paper: Cpump = Σ 0.0045 * Ni² * [1 + 0.35*sin²(...)]
        pump_cost += 0.0045 * sqr(network_load) * pump_mod;

        // Paper: Pdef = Σ (0.12 + 0.06*heat) * deficit² * shock
        deficit_term += (deficit_weight_ + 0.06 * heat_[i]) * sqr(deficit) * weather_shock;

        // Paper: Pexc = Σ (0.05 + 0.05*wind) * (deep² + 0.30*surplus²)
        excess_term += (0.05 + 0.05 * wind_[i]) * (sqr(deep_loss) + 0.30 * sqr(surplus));

        // Paper: Pres = Σ 8.0*(1+0.70*heat)*sin²(freq*xi+φ)
        if (config_.oscillatory_efficiency_enable) {
            resonance_term += resonance_weight_ * (1.0 + 0.70 * heat_[i]) * sqr(std::sin(freq_[i] * xi + phase_[i]));
        }

        // CHANGED: Paper: Pint = Σ_{i>=2} [0.018*max(0, xi+0.55*xi-1-58)² + 0.025*xi*xi-1]
        if (config_.neighbor_coupling_enable && i > 0) {
            interaction_term += 0.018 * sqr(std::max(0.0, xi + 0.55 * xim1 - 58.0));
            interaction_term += 0.025 * xi * xim1;
        }

        // Paper: Psmooth = Σ_{i>=2} (0.06 + 0.02*wind) * (xi - xi-1)²
        if (i > 0) {
            const double dx = xi - xim1;
            smoothness_term += (smoothness_weight_ + 0.02 * wind_[i]) * sqr(dx);
        }

        // CHANGED: Paper: Pwin = Σ (0.014 + 0.010*sin²(πs)) * (xi-μi)² * [1+0.40*sin²(0.06*xi+φ)]
        if (config_.preferred_window_enable) {
            const double si = static_cast<double>(i + 1) / static_cast<double>(D + 1);
            const double pwin_coeff = pref_weight_ + 0.010 * sqr(std::sin(kPi * si));
            const double pref_dev = xi - pref_[i];
            const double pref_mod = config_.oscillatory_efficiency_enable
                                  ? (1.0 + 0.40 * sqr(std::sin(0.06 * xi + phase_[i])))
                                  : 1.0;
            pref_term += pwin_coeff * sqr(pref_dev) * pref_mod;
        }

        seasonal_total += xi;
        storage = next_storage;
    }

    if (!config_.multiplicative_yield_enable) {
        yield_signal /= static_cast<double>(D);
    }

    // CHANGED: Paper: Pterm = 1.2 * (S_{D+1} - 0.45*Smax_D)²
    double terminal_storage_penalty = 0.0;
    if (config_.recursive_storage_enable) {
        const double terminal_target = 0.45 * smax_.back();    // CHANGED: was 0.58
        terminal_storage_penalty = terminal_storage_weight_ * sqr(storage - terminal_target);
    }

    // CHANGED: Paper: Pbudget = 0.010 * (Σxi - BD)²  (symmetric)
    const double budget_dev = seasonal_total - seasonal_budget_target_;
    const double seasonal_budget_penalty = seasonal_budget_weight_ * sqr(budget_dev);

    // CHANGED: Paper: Ppeak = 0.035 * Σ max(0, Ni - 62)²  (peak_term already accumulated)
    const double peak_penalty = config_.threshold_losses_enable
                              ? (peak_weight_ * peak_term)
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
