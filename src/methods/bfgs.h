#pragma once
#include <vector>
#include <random>
#include <limits>
#include <cstddef>
#include "optimizer.h"
#include "init.h"

namespace optimsolution {

struct BFGSParams {
    double alpha0       = 1.0;   // initial step size
    double c1           = 1e-4;  // Armijo
    int    backtracks   = 20;    // maximum backtracking steps
    double tol_grad     = 1e-8;  // ||grad f|| tolerance
    int    max_iters    = 2000;  // safety cap (the stop controller has its own)
    bool   reset_on_nan = true;  // resets H to I if the update fails
};

class BFGS : public Optimizer {
public:
    BFGS();
    ~BFGS() = default;
	
	std::string methodShortName() const override { return "bfgs"; }

    std::string methodFullName() const override {
        return "Broyden-Fletcher-Goldfarb-Shanno (Local Method)";
    }

    void configure(const MethodConfig& mc);

    // required by the base class
    void init();
    void one_iteration();

    int population() const { return 1; }
    int getPopulation() const { return 1; }

private:
    // helpers
    static double dot(const std::vector<double>& a, const std::vector<double>& b);
    static double l2norm(const std::vector<double>& a);
    double evalPoint(const std::vector<double>& x);
    void project_to_bounds(std::vector<double>& x) const;
    bool bfgs_update(const std::vector<double>& s, const std::vector<double>& y);
    LineSearchResult backtrackingArmijo(const std::vector<double>& x,
                                        const std::vector<double>& g,
                                        const std::vector<double>& d,
                                        double alpha0, double c1, int max_bk);

private:
    BFGSParams prm_;

    // state
    std::vector<double> x_;
    std::vector<double> g_;
    std::vector<double> x_new_;
    std::vector<double> g_new_;
    std::vector<double> H_;   // inverse-Hessian (n x n)
    double f_{std::numeric_limits<double>::infinity()};
    double f_new_{std::numeric_limits<double>::infinity()};

    std::vector<double> best_x_;
    double best_f_{std::numeric_limits<double>::infinity()};

    int n_{0};
    int it_{0};

    // buffer for stopping/logging
    std::vector<double> FX_{1, std::numeric_limits<double>::infinity()};
};

} // namespace optimsolution
