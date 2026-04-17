#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class SFCDE : public Optimizer {
public:
    SFCDE() = default;
    ~SFCDE() override = default;

    std::string methodShortName() const override { return "sfcde"; }
    std::string methodFullName()  const override { return "Success-Failure Competitive Differential Evolution (SFCDE)"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

    // Hook from the global layer for final local refinement.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

private:
    // Population size. The public reference implementation uses 50 by default.
    int pop_init_{50};

    // Success-history memory.
    int H_{5};
    double c_mem_{0.1};
    double mu_f_init_{0.5};
    double mu_cr_init_{0.5};

    // Sampling widths.
    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    // Maximum number of attempts in the SFA rejection-sampling loops.
    // Prevents potential infinite loops when mu_fail ≈ MF_ (e.g., late search).
    static constexpr int kMaxSfaTrials = 20;

    // Parameter memories (success history, SHADE-style).
    std::vector<double> MF_;
    std::vector<double> MCR_;

    // Sequential write pointer for memory slots (cycles 0 … H_-1).
    // FIX #3: replaces random slot selection so every slot is updated regularly.
    int k_{0};

    // Failure history of the previous generation.
    std::vector<double> fail_F_;
    std::vector<double> fail_CR_;

    // Population.
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // Final local refinement at the end.
    bool        end_local_refine_{false};
    std::string end_local_method_;

    // Helpers.
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }

    double meanLehmer(const std::vector<double>& values) const;
    double meanArithmetic(const std::vector<double>& values) const;
    double sampleF(double mu);
    double sampleCR(double mu);
    void repairRandom(Vec& x);
    int bestIndex() const;

    // FIX #4: draw 3 distinct indices from [0, N) excluding `exclude`,
    // using a partial Fisher-Yates shuffle — O(N) worst case, no rejection loops.
    void sampleThreeDistinct(int N, int exclude,
                             int& a, int& b, int& c);
};

} // namespace optimsolution
