#pragma once
#include "problem.h"

namespace optimsolution {

class SmartPortEnergy : public Problem {
public:
    SmartPortEnergy();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    Vec demand_;
    Vec renewable_;
    Vec tariff_;
    Vec criticality_;
    Vec pmax_;
    Vec bcap_;
    Vec hcap_;
    Vec peakcap_;
    Vec pref_;

    double smax_;
    double hmax_;
    double s0_;
    double h0_;
    double starget_;
    double htarget_;
    double seasonal_budget_;

    double evaluate_with_point(const Vec& x) const;
    double safe_x(const Vec& x, int i) const;
};

} // namespace optimsolution
