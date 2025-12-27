#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// Continuous GA with greedy (elitist steady-state) per-individual replacement.
class GA : public Optimizer {
public:
    GA() = default;
    ~GA() override = default;
	std::string methodShortName() const override { return "ga"; }
	std::string methodFullName()  const override { return "Genetic Algorithm (GA)"; }

    // Final local search is taken from [global].
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings are read from [ga].
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 3) pop_ = pop_override;

        sel_type_ = mc.getStr("selection", sel_type_);
        for (auto& c: sel_type_) c = (char)std::tolower((unsigned char)c);
        tournament_k_ = mc.getInt("tournament_k", tournament_k_);

        cx_type_ = mc.getStr("crossover", cx_type_);
        for (auto& c: cx_type_) c = (char)std::tolower((unsigned char)c);
        pc_      = mc.getDbl("pc", pc_);
        eta_c_   = mc.getDbl("eta_c", eta_c_);
        blx_a_   = mc.getDbl("blx_alpha", blx_a_);
        uox_p_   = mc.getDbl("uox_p", uox_p_);

        pm_        = mc.getDbl("pm", pm_);
        mut_sigma_ = mc.getDbl("mut_sigma", mut_sigma_);

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
    int  tournamentSelect(); // Non-const (uses non-const rng_).
    void mutate(std::vector<double>& child);
    void crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                   std::vector<double>& child);

private:
    // Population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;

    // Selection / crossover / mutation
    std::string sel_type_{"tournament"};
    int         tournament_k_{3};

    std::string cx_type_{"sbx"}; // "sbx" | "blx" | "uniform"
    double      pc_{0.9};
    double      eta_c_{20.0};
    double      blx_a_{0.5};
    double      uox_p_{0.5};

    double      pm_{0.1};
    double      mut_sigma_{0.1};

    // In-run local search (only after acceptance)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final polishing is performed in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
