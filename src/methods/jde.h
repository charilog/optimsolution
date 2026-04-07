#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig; // forward declaration

// Classic self-adaptive Differential Evolution jDE (Brest et al. 2006-style)
class jDE : public Optimizer {
public:
    jDE() = default;
    ~jDE() override = default;
	std::string methodShortName() const override { return "jde"; }
	std::string methodFullName()  const override { return "Self-adaptive Differential Evolution (jDE)"; }

    // Allow global [end_local_*] settings to propagate from Optimizer
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    // Called by factory after construction
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);

private:
    // --- Config / population ---
    int    pop_cfg_{-1};      // per-method population (overrides the global setting)
    int    pop_init_{100};    // initial N (fallback when not specified elsewhere)

    // --- jDE parameter self-adaptation ---
    double F_min_{0.1};       // lower bound for F_i
    double F_max_{0.9};       // upper bound for F_i
    double CR_init_{0.9};     // initial CR_i
    double tau1_{0.1};        // prob. to reset F_i
    double tau2_{0.1};        // prob. to reset CR_i

    // In-run local search (as in the other methods)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final local @ end
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // --- State ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<double> F_i_;
    std::vector<double> CR_i_;
};

} // namespace optimsolution
