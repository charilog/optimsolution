#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// ACOR: Ant Colony Optimization for Continuous Domains (Archetti, Socha et al.)
class ACOR : public Optimizer {
public:
    ACOR() = default;
    ~ACOR() override = default;
	std::string methodShortName() const override { return "acor"; }
	std::string methodFullName()  const override { return "Ant Colony Optimization for Continuous Domains"; }

    // Enablement/method of final local refinement from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Load settings from the [acor] section
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        archive_ = mc.getInt("archive", archive_);
        if (archive_ < 2) archive_ = 2;

        q_  = mc.getDbl("q",  q_);
        if (q_ <= 0.0) q_ = 0.0001;

        xi_ = mc.getDbl("xi", xi_);
        if (xi_ <= 0.0) xi_ = 0.1;

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

    // Rank-based weights for selecting centers (solutions) from the archive
    void   buildWeights();
    int    sampleIndexByWeights();

    // Sigma for each (i,j): based on the mean distance from A[i] in dimension j
    double sigma_ij(int i, int j) const;

private:
    // Archive A (k best solutions)
    int archive_{50}; // k
    std::vector<std::vector<double>> A_;   // A_[i][j]
    std::vector<double>              FA_;  // fitness

    // Selection weights (rank-based) for archive elements
    std::vector<double> W_;    // unnormalized
    std::vector<double> WCDF_; // cumulative (0..1)

    // Current "ants" per iteration (for stop/print)
    std::vector<std::vector<double>> X_;   // ants
    std::vector<double>              FX_;  // ants fitnesses

    // ACOR parameters
    double q_{0.5};   // shape of the rank-based weights (typically ~0.1..0.5)
    double xi_{0.85}; // learning rate/coefficient in sigma (typically 0.5..1.0)

    // in-run local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
