
#pragma once

#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig;

/**\n * GDE:\n * Simplified Golden Differential Evolution for optimsolution.\n *\n * - Single core strategy: DE/pbest/1/bin with archive (SHADE-style)\n * - Self-adaptive F and CR via global means (muF, muCR)\n * - Archive for diversity\n * - Micro-restart and quarantine for escaping stagnation\n 
 */
class GDE : public Optimizer {
public:
    GDE()  = default;
    ~GDE() override = default;

    std::string methodShortName() const override { return "gde"; }
    std::string methodFullName()  const override {
        return "Golden Differential Evolution (pbest/1+archive)";
    }

    void configure(const MethodConfig &mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override {}   // the final polishing is performed by the Optimizer (final_local)

private:
    using Vec = std::vector<double>;

    // --- population ---
    int pop_init_{40};   // default initial population
    int pop_min_{20};    // minimum population
    int N_{0};           // effective population size

    // --- self-adaptive F/CR (global means) ---
    double muF_{0.5};
    double muCR_{0.9};
    double F_lo_{0.1};
    double F_hi_{1.2};
    double CR_lo_{0.0};
    double CR_hi_{1.0};
    double sh_c_{0.1};   // learning rate for muF/muCR

    // --- population & fitness ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- archive for diversity ---
    std::vector<Vec> archive_;
    std::size_t      archive_max_{300};

    // --- stagnation control ---
    int iter_{0};
    int no_improv_iters_{0};
    int micro_restart_period_{40};
    int quarantine_period_{20};

    // === utilities ===
    void ensureBounds(Vec &x) const;
    void pushArchive_(const Vec &x);
    int  pickPbestIndex_(double pfrac);

    void microRestart_();
    void quarantineOutliers_();

    void trial_pbest1A_bin_(int i,
                            const Vec &xi,
                            Vec &tr,
                            double F,
                            double CR,
                            double pfrac,
                            bool useArchive);
};

} // namespace optimsolution


