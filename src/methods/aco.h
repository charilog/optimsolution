#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// Classic Ant Colony Optimization (discrete construction on per-dimension levels)
// (continuous problems supported via discretization grid per dimension)
class ACO : public Optimizer {
public:
    ACO() = default;
    ~ACO() override = default;
	std::string methodShortName() const override { return "aco"; }
	std::string methodFullName()  const override { return "Ant Colony Optimization (ACO)"; }

    // Final local-search settings from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [aco] in the configuration file
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        levels_      = mc.getInt   ("levels",      levels_);
        alpha_       = mc.getDbl   ("alpha",       alpha_);
        beta_        = mc.getDbl   ("beta",        beta_);
        rho_         = mc.getDbl   ("rho",         rho_);
        Q_           = mc.getDbl   ("Q",           Q_);
        deposit_top_ = mc.getInt   ("deposit_top", deposit_top_);
        tau0_        = mc.getDbl   ("tau0",        tau0_);
        tau_min_     = mc.getDbl   ("tau_min",     tau_min_);
        tau_max_     = mc.getDbl   ("tau_max",     tau_max_);

        // in-run local
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // helpers
    double eval(const std::vector<double>& x) { return prob_->evaluate(x); }
    void ensureBounds(std::vector<double>& x);
    double valueAtLevel(int j, int l) const;
    int    sampleLevel(int j, double gbest_j);
    void   evaporate();
    void   deposit(const std::vector<int>& order);

private:
    // Pheromone matrix tau[j][l], j=0..D-1, l=0..levels_-1
    std::vector<std::vector<double>> tau_;

    // grid levels per dimension
    int    levels_{16};
    double alpha_{1.0};     // pheromone exponent
    double beta_{2.0};      // heuristic exponent
    double rho_{0.1};       // evaporation  (0<rho<=1)
    double Q_{1.0};         // deposit intensity (Q/f)
    int    deposit_top_{5}; // #top ants depositing
    double tau0_{1.0};      // initial pheromone
    double tau_min_{1e-6};
    double tau_max_{1e6};

    // working buffers per iteration
    std::vector<std::vector<double>> X_;  // candidate solutions
    std::vector<double>              FX_; // fitnesses

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
