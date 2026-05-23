#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class BJSO : public Optimizer {
public:
    BJSO() = default;
    ~BJSO() override = default;
    std::string methodShortName() const override { return "bjso"; }
    std::string methodFullName()  const override { return "Band-guided jSO"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

private:
    int pop_init_{0};
    int pop_min_{4};

    int H_{5};
    double c_mem_{0.1};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int mem_idx_{0};

    double pmin_{0.05};
    double pmax_{0.25};

    double arc_rate_{1.4};
    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    // Optional donor correction over the 11%-50% band of the ranked population.
    bool   bj_enable_{false};
    double bj_band_beta_{0.02};

    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;

    std::string local_method_{"none"};
    double      local_rate_{0.0};

    bool        end_local_refine_{false};
    std::string end_local_method_;

    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }
    void ensureInBounds(Vec& x);
    bool isInBounds(const Vec& x) const;
    void trimArchive(int max_size);
    Vec  meanOfRange(const std::vector<int>& idx_sorted, int begin_idx, int end_idx) const;
};

} // namespace optimsolution
