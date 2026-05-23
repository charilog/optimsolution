#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>

namespace optimsolution {

class SO : public Optimizer {
public:
    SO() = default;
    ~SO() override = default;

    std::string methodShortName() const override { return "SO"; }
    std::string methodFullName()  const override { return "Spiral Optimization"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    void ensureBounds(Vec& v);
    void injectBestIntoWorst();
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // SO parameters
    double shrink_{0.95};          // contraction factor r in (0,1]
    double theta_{0.78539816339};  // rotation angle (radians), default pi/4
    bool   random_direction_{true};
    bool   greedy_{false};         // if true, accept only improving moves; otherwise accept all

    // In-run local
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // Final local
    bool        end_local_refine_ = false;
    std::string end_local_method_;

    // misc
    int         debug_so_ = 0;
    int         pop_override_ = -1;
};

} // namespace optimsolution
