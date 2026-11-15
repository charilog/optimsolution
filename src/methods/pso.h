#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>

namespace optimsolution {

class PSO : public Optimizer {
public:
    PSO() = default;
    ~PSO() override = default;
	std::string methodShortName() const override { return "pso"; }
	std::string methodFullName()  const override { return "Particle Swarm Optimization"; }	

    // Obtain final end-local from [global] (as in DE)
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Opt-in to "[global] max_evals = <N>xD" (budget scales with the
    // problem's dimension) -- see options.h. Every other method is
    // unaffected by this override existing here.
    void setGeneralOptions(const GeneralOptions& g) override {
        Optimizer::setGeneralOptions(g);
        if (g.max_evals_per_dim > 0 && prob_) {
            max_evals_ = g.max_evals_per_dim * (long long)prob_->dimension();
        }
    }

    // configure(): reads method-specific parameters from [pso]
    void configure(const MethodConfig& mc) override {
        // population=0 (or unset): the original 1995 Kennedy & Eberhart
        // paper does not fix one canonical swarm size; the literature
        // consistently describes a "typically 20 to 50" range. We use the
        // midpoint, 30, a very commonly used single-value convention.
        // Previously population=0 silently fell back to [global]
        // population (default 100) instead of this documented default -- fixed here.
        int pop_override = mc.getInt("population", 0);
        pop_ = (pop_override > 3) ? pop_override : 30;

        // PSO params
        w_  = mc.getDbl("w",  w_);
        c1_ = mc.getDbl("c1", c1_);
        c2_ = mc.getDbl("c2", c2_);

        vmax_frac_ = mc.getDbl("vmax_frac", vmax_frac_);
        clamp_pos_ = mc.getBool("clamp_pos", clamp_pos_);

        topology_ = mc.getStr("topology", topology_);
        for (char& c : topology_) c = (char)std::tolower((unsigned char)c);
        ring_k_   = mc.getInt("ring_k", ring_k_);

        // In-run local (as in DE: only after a successful improvement)
        local_method_ = mc.getStr("local_method", local_method_);
        for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override; // Final polishing controlled by [global]

private:
    // helpers
    int  neighborhoodBestIndex(int i) const; // For ring topology
    void clampVelocity(std::vector<double>& v);
    void ensureBounds(std::vector<double>& x);
    double eval(const std::vector<double>& v){ return prob_->evaluate(v); }

private:
    // Population
    std::vector<std::vector<double>> X_;
    std::vector<std::vector<double>> V_;
    std::vector<double>              FX_;

    // Personal/global bests (for PSO)
    std::vector<std::vector<double>> Pbest_;
    std::vector<double>              Fpbest_;
    std::vector<double>              Gbest_;
    double                           Fgbest_{std::numeric_limits<double>::infinity()};

    // Velocity bounds
    std::vector<double> Vmax_;

    // PSO parameters (Clerc-ish)
    double w_{0.7298};
    double c1_{1.49618};
    double c2_{1.49618};
    double vmax_frac_{0.2};
    bool   clamp_pos_{true};
    std::string topology_{"gbest"}; // "gbest" or "ring"
    int    ring_k_{2};

    // In-run local (as in DE)
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // Final polishing (in end) from [global]
    bool        end_local_refine_ = false;
    std::string end_local_method_ = "";
};

} // namespace optimsolution
