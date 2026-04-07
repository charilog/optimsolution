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

    // Parameter memories.
    std::vector<double> MF_;
    std::vector<double> MCR_;

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
};

} // namespace optimsolution
