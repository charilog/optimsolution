// gderl.h
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

// ---------------------------------------------------------------------
// Golden Differential Evolution with Reinforcement Learning (gDE-rl)
// Dual-zone DE + RL strategies (weights-based selection)
// ---------------------------------------------------------------------
class GDERL : public Optimizer {
public:
    GDERL() = default;
    ~GDERL() override = default;

    // CLI short name: "gderl"  ->  optimsolution.exe gderl problem ...
    std::string methodShortName() const override { return "gderl"; }

    // Full descriptive name for logs / summaries
    std::string methodFullName() const override {
        return "Golden Differential Evolution with Reinforcement Learning (gDE-rl)";
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    // Helpers
    double eval(const Vec& x);
    void   ensureBounds(Vec& x);
    double bnDistance(const Vec& a, const Vec& b) const;

    // LSHADE-like memory for F / CR
    void sample_F_CR(double& F, double& CR);
    void updateMemory(const std::vector<double>& succF,
                      const std::vector<double>& succCR);

    // Zones & RL
    // zone = 0 -> explore, zone = 1 -> exploit
    int  determineZone(const Vec& x) const;
    int  selectStrategy(int zone, double progress); // progress in [0,1]
    void recordOutcome(int zone, int strat, double gain);
    void updateRL();

    // Strategies:
    // 0: DE/rand/1/bin
    // 1: current-to-pbest/1 + archive
    // 2: DE/best/1/bin
    // 3: Gaussian around best_x_
    // 4: Long-jump (global / around a promising point)
    void make_trial_rand1(int i, Vec& trial, double F, double CR);
    void make_trial_pbest(int i, Vec& trial, double F, double CR, double pbest_fraction);
    void make_trial_best1(int i, Vec& trial, double F, double CR);
    void make_trial_gaussian(Vec& trial, double sigma);
    void make_trial_longjump(Vec& trial);

    // helpers
    int  randomIndexExcept(int n, int forbid);
    int  pickPBestIndex(double pbest_fraction);
    void pushArchive(const Vec& x);
    int  pickArchiveIndex();

    void restartIndividuals();

private:
    int D_{0};
    int pop_init_{0};
    int pop_min_{4};

    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;
    double              archive_rate_{1.0};

    // LSHADE memories
    int                 H_{10};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 mem_idx_{0};
    double              c_mem_{0.1};
    double              pmin_{0.05};
    double              pmax_{0.25};

    // Dual-zone control
    double zone_radius_{0.2}; // normalized distance to be considered "close" to best

    static constexpr int NUM_ZONES_ = 2;
    static constexpr int NUM_STRAT_ = 5;

    double strat_weight_[NUM_ZONES_][NUM_STRAT_]{};
    double strat_reward_[NUM_ZONES_][NUM_STRAT_]{};
    int    strat_uses_[NUM_ZONES_][NUM_STRAT_]{};
    double rl_alpha_{0.3};   // learning/smoothing
    double min_weight_{0.05};

    // Stagnation & restart
    double restart_frac_{0.10};
    double restart_sigma_{0.30};
    int    stagnation_window_{30};
    int    stagnation_counter_{0};
    double last_best_{std::numeric_limits<double>::infinity()};

    // Extra parameters (reserved for outlier relocation etc.)
    double outlier_alpha_{1.5};
    double relocate_rate_{0.2};

    // In-run local search
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // Final local refine at end()
    bool        end_local_refine_{false};
    std::string end_local_method_;

    std::uniform_real_distribution<double> U01_{0.0, 1.0};
};

} // namespace optimsolution
