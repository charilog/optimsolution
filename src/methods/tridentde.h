#pragma once
#include "optimizer.h"
#include "init.h"
#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace optimsolution {

class TRIDENTDE : public Optimizer {
public:
    TRIDENTDE() = default;
    ~TRIDENTDE() override = default;
	std::string methodShortName() const override { return "tridentde"; }
	std::string methodFullName()  const override { return "TRIDENT Differential Evolution"; }

    std::string name() const { return "tridentde"; }

    void configure(const MethodConfig& mc) override;

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    using Pop = std::vector<Vec>;

    // ---- Hyper-params ----
    int    trials_per_agent{4};
    double agent_fraction{0.55};

    // jDE self-adaptation
    double tauF{0.10}, tauCR{0.10};
    double F_lo{0.10}, F_hi{1.20};
    double CR_lo{0.00}, CR_hi{0.95};

    // pbest fraction
    double pbest_frac{0.10};

    // Stagnation → micro-restart
    int    stagnation_trigger{18};
    double restart_frac{0.10};
    double restart_sigma{0.20}; // As a percentage of the box width.

    // In-run local
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // Internal state
    int D_{0};
    int N_{50};
    int startAgent_{0};
    int stagn_iters_{0};

    Pop                 X_;
    std::vector<double> FX_;

    // Per-individual jDE params
    std::vector<double> Fi_, CRi_;

    // RNG
    std::mt19937_64 rng_{std::random_device{}()};
    std::uniform_real_distribution<double> U01_{0.0, 1.0};
    std::normal_distribution<double> N01_{0.0, 1.0};
    std::uniform_int_distribution<int> DIpop_;
    std::uniform_int_distribution<int> DIdim_;

    // Helpers
    inline double eval_safe(const Vec& x);
    inline Vec    clamp_to_bounds(const Vec& x) const;
    inline double range_j(int j) const;
    void          ensure_finite_best(); // NEW: as in BHO.

    // Operators
    void make_trial_best1   (int i, const Vec& xi, Vec& trial, double F, double CR);
    void make_trial_ctobest1(int i, const Vec& xi, Vec& trial, double F, double CR);
    void make_trial_pbest1  (int i, const Vec& xi, Vec& trial, double F, double CR);

    bool line_refine(const Vec& base, const Vec& trial_in, Vec& trial_out);

    int  eliteIndex_() const;
    void microRestart_();

    // end() options
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
