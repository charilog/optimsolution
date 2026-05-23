#pragma once
#include "problem.h"

namespace optimsolution {

class DataCenterCooling : public Problem {
public:
    DataCenterCooling();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    std::vector<double> workload_;
    std::vector<double> ambient_;
    std::vector<double> tariff_;
    std::vector<double> humidity_;
    std::vector<double> tlimit_;
    std::vector<double> coolcap_;
    std::vector<double> flowcap_;
    std::vector<double> pref_;
    std::vector<double> criticality_;

    double t0_;
    double m0_;
    double ttarget_;
    double mtarget_;
    double carbon_budget_;

    double safe_x(const Vec& x, int i) const;
    double evaluate_with_point(const Vec& x) const;
};

} // namespace optimsolution
