#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class BHO : public Optimizer {
public:
    BHO() = default;
    ~BHO() override = default;
	std::string methodShortName() const override { return "bho"; }
	std::string methodFullName()  const override { return "BioHealing Optimization (BHO)"; }

    // Optional name
    std::string name() const { return "bho"; }

    // Settings from [bho] (as in GA)
    void configure(const MethodConfig& mc) override;

    // Final local search from [global] (as in GA)
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

    // ------- Tunables (read from [bho]) -------
    double heal_prob{0.65};          // P(healing) vs wounding
    double heal_rate{0.30};          // step towards elite
    double wound_strength_init{0.40};
    int    stagnation_kick{20};      // stagnation steps before kick
    int    stagnation_restart{300};  // stagnation steps before restart
    double elite_kick_sigma{0.01};   // std for small Gaussian around elite
    double restart_frac{0.25};       // population fraction at restart
    int    print_stride{2};          // print every n iterations (>=1)

    // In-run local search (only after acceptance), from [bho]
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final polishing in end(), from [global]
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // Internal state
    int D_{0};
    int N_{50}; // applied in init from the base pop_
    Pop                 X_;      // N x D
    std::vector<double> FX_;     // N
    Pop                 archive_;
    int                 sinceBest_{0};

    // RNG
    std::mt19937_64 rng_{std::random_device{}()};
    std::uniform_real_distribution<double> U01_{0.0, 1.0};
    std::normal_distribution<double> N01_{0.0, 1.0};

    // helpers
    inline double eval_safe(const Vec& x);
    inline Vec    clamp_to_bounds(const Vec& x) const;
    void          seed_uniform();
    void          seed_midpoint(Vec& out) const;
    void          ensure_finite_best();
    void          elite_gaussian_kick();
    void          soft_kick_population();
    void          restart_partial();
};

} // namespace optimsolution
