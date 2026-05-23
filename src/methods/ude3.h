#pragma once
#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>
#include <numeric>
#include <cstddef>

namespace optimsolution {

struct MethodConfig; // Forward declaration, as in options.h.

class UDE3 : public Optimizer {
public:
    UDE3() = default;
    ~UDE3() override = default;
	std::string methodShortName() const override { return "ude3"; }
	std::string methodFullName()  const override { return "Enhanced Unified Differential Evolution Algorithm 3"; }

    // Comment translated from Greek.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    // Settings from the [ude3] block.
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);

    int selectRankedIndex(const std::vector<int>& sorted_idx,
                          const std::vector<double>& rank_prob,
                          int avoid1, int avoid2, int avoid3);

    // strat: 0 = DE/rand/1, 1 = DE/current-to-rand/1, 2 = DE/current-to-pbest/1
    bool generateTrial(int strat, int idx,
                       const std::vector<int>& sorted_idx,
                       int p_best_size,
                       const std::vector<double>& rank_prob,
                       double Fi, double CRi,
                       Vec& ui,
                       std::uniform_real_distribution<double>& U01);

    int  selectStrategy(std::uniform_real_distribution<double>& U01);

private:
    // --- Config ---
    int    pop_cfg_{-1};      // per-method population (override global)
    int    pop_init_{100};

    int    H_{10};            // SHADE memory size
    double c_mem_{0.1};       // Learning rate for MF/MCR.

    double top_frac_{0.5};    // Comment translated from Greek.
    int    Lp_{50};           // Learning period for strategy adaptation.

    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    double p_best_rate_{0.2}; // p-best rate for current-to-pbest/1.

    int stagnation_limit_{50}; // Comment translated from Greek.

    // --- In-run local search (from [ude3]) ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Comment translated from Greek.
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- BDVS-style archive ---
    std::vector<Vec>    archive_;
    std::vector<double> archive_f_;
    std::size_t         maxArc_{0};

    // best discarded per individual
    std::vector<Vec>    best_disc_;
    std::vector<double> best_disc_f_;

    // --- SHADE memory (MF, MCR) ---
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 mem_idx_{0};

    // --- Strategy adaptation (SaDE-style) ---
    static constexpr int NUM_STRAT_ = 3;
    std::vector<double> strat_prob_;    // Probabilities for each strategy.
    std::vector<int>    strat_success_;
    std::vector<int>    strat_attempt_;
    int                 gen_counter_{0};

    // --- Stagnation per individual ---
    std::vector<int> stagnation_;

    // Comment translated from Greek.
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
