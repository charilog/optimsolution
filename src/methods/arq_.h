#pragma once
#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace optimsolution {

// Full ARQ (Archive + RTR + Quarantine) adapted to optimsolution.
class ARQ : public Optimizer {
public:
    ARQ() = default;
    ~ARQ() override = default;
	
    std::string methodShortName() const override { return "arq"; }
    std::string methodFullName()  const override {
        return "Archive-based Quarantine Differential Evolution (ARQ)";
    }

    // Forward end-local settings from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Read method-specific options from [arq]
    void configure(const MethodConfig& mc) override;

    // Core lifecycle
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // ----- Parameters -----
    // NOTE: The actual N is determined in init() from pop_override_ or the base population()
    int    pop_{200};
    double agent_fraction_{0.60}; // fraction of population updated per iteration

    // Success-History (global, SHADE-like)
    double muF_{0.6}, muCR_{0.85};
    double sh_c_{0.10};
    double F_lo_{0.05}, F_hi_{1.4};
    double CR_lo_{0.0}, CR_hi_{1.0};

    // pbest & archive (JADE style)
    double pbest_frac_{0.12};
    double archive_rate_{1.5}; // × population

    // RTR parameters
    int    rtr_pool_{14};
    double rtr_min_replace_gain_{0.0}; // relative gain threshold

    // Outlier Quarantine (robust mean pull)
    double outlier_alpha_{1.0};
    double quarantine_rate_{0.08};
    double quarantine_sigma_{0.10};

    // Stagnation & micro-restart
    int    stagnation_trigger_{24};
    double restart_frac_{0.08};
    double restart_sigma_{0.18};

    // In–run local search (optional)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // End local search (from [global])
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // ----- State -----
    std::vector<Vec>     X_;
    std::vector<double>  FX_;
    std::vector<Vec>     archive_;

    // Mini-batch window
    int start_agent_{0};

    // RNG and helpers
    std::mt19937_64 rng_{};
    std::uniform_real_distribution<double> U01_{0.0,1.0};

    // User/run seeding (optional, from the [arq] section)
    uint64_t user_seed_{0};
    int      run_id_hint_{-1};
    uint64_t seed_used_{0};
    uint64_t runs_started_{0};

    // Stagnation counter
    int stagn_iters_{0};

    // Population override from [arq]
    int pop_override_{-1};

private:
    inline double clamp_(double v, double lo, double hi) const {
        return v<lo?lo:(v>hi?hi:v);
    }
    void   ensureBounds(Vec& v);
    double eval(const Vec& v){ return prob_->evaluate(v); }

    int    eliteIndexFinite() const;
    int    pickDistinct(int n, int a=-1, int b=-1, int c=-1);
    void   pushArchive_(const Vec& x);
    int    pickArchiveIndex_() const;
    int    pickPbestIndex_() const;
    int    pickRTRNeighbor_(const Vec& trial, const std::vector<int>& pool) const;
    double bnDistance_(const Vec& a, const Vec& b) const;

    void   quarantineOutliers_();
    void   microRestart_();  // <--- No external idx is used anymore; performs its own sort

    // Operator
    void trial_pbest1A_bin_(int i, const Vec& xi, Vec& tr, double F, double CR);
};

} // namespace optimsolution
