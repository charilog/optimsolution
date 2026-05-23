// garq.h
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

struct MethodConfig;

// Golden ARQ with RL (GARQ)
// - ARQ-style exploration (quarantine + micro-restart + diversity-aware)
// - gDE-rl-style exploitation (LSHADE + RL + 5 DE strategies)
class GARQ : public Optimizer {
public:
    GARQ() = default;
    ~GARQ() override = default;

    std::string methodShortName() const override { return "garq"; }

    std::string methodFullName() const override {
        return "Golden ARQ with Reinforcement Learning";
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    // ---- Core helpers ----
    double eval(const Vec& x);
    void   ensureBounds(Vec& x);
    double bnDistance(const Vec& a, const Vec& b) const;
    double computeDiversity() const;

    // ---- LSHADE F/CR memory ----
    void sample_F_CR(double& F, double& CR);
    void updateMemory(const std::vector<double>& succF,
                      const std::vector<double>& succCR);

    // ---- Reinforcement learning over DE strategies ----
    // zone = 0 -> explore, zone = 1 -> exploit (near the best)
    int  determineZone(const Vec& x) const;
    int  selectStrategy(int zone, double progress, double diversity);
    void recordOutcome(int zone, int strat, double gain);
    void updateRL();

    // Strategies:
    // 0: DE/rand/1/bin
    // 1: current-to-pbest/1 + archive
    // 2: DE/best/1/bin
    // 3: Gaussian around best_x_
    // 4: Long-jump (global / around best_x_)
    void make_trial_rand1   (int i, Vec& trial, double F, double CR);
    void make_trial_pbest   (int i, Vec& trial, double F, double CR, double pbest_fraction);
    void make_trial_best1   (int i, Vec& trial, double F, double CR);
    void make_trial_gaussian(Vec& trial, double sigma);
    void make_trial_longjump(Vec& trial);

    // ---- Low-level helpers ----
    int  randomIndexExcept(int n, int forbid);
    int  pickPBestIndex(double pbest_fraction);
    void pushArchive(const Vec& x);
    int  pickArchiveIndex();
    void restartIndividuals();   // Micro-restart of the worst individuals (half near the best, half global)

    // ---- ARQ-style quarantine of outliers ----
    void quarantineOutliers();

private:
    // Dimensions / population
    int D_{0};
    int pop_init_{0};
    int pop_min_{4};

    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;
    double              archive_rate_{1.0};

    // LSHADE memory
    int                 H_{10};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 mem_idx_{0};
    double              c_mem_{0.1};
    double              pmin_{0.05};
    double              pmax_{0.25};

    // RL: 2 zones, 5 strategies
    static constexpr int NUM_ZONES_ = 2;
    static constexpr int NUM_STRAT_ = 5;

    double zone_radius_{0.2};

    double strat_weight_[NUM_ZONES_][NUM_STRAT_]{};
    double strat_reward_[NUM_ZONES_][NUM_STRAT_]{};
    int    strat_uses_  [NUM_ZONES_][NUM_STRAT_]{};
    double rl_alpha_{0.3};
    double min_weight_{0.05};

    // Diversity-aware control
    double diversity_ema_{0.0};
    bool   diversity_initialized_{false};
    double div_low_{0.08};   // If it drops below this value, diversity is considered very low.
    double div_high_{0.25};  // Above this value, diversity is considered acceptable.

    // Stagnation / restart
    double restart_frac_{0.18};
    double restart_sigma_{0.35};
    int    stagnation_window_{25};
    int    stagnation_counter_{0};
    double last_best_{std::numeric_limits<double>::infinity()};

    // ARQ-style quarantine parameters
    double quarantine_rate_{0.30};
    double quarantine_sigma_{0.20};
    double outlier_alpha_{1.7};
    double relocate_rate_{0.25}; // Placeholder for future fine-tuning

    // In-run local search
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // Final local refinement
    bool        end_local_refine_{false};
    std::string end_local_method_;

    // RNG
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> U01_{0.0, 1.0};
};

} // namespace optimsolution
