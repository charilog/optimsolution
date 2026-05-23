#pragma once
#include <vector>
#include <limits>
#include <random>
#include "optimizer.h"
#include "init.h"

namespace optimsolution {

// Gradient Descent settings
struct GDParams {
    double alpha0     = 1.0;    // initial step size for backtracking
    double c1         = 1e-4;   // Armijo condition
    int    backtracks = 20;     // maximum number of backtracking steps
    double momentum   = 0.0;    // 0 => pure GD, >0 Heavy-Ball
    double tol_grad   = 1e-8;   // ||g||_2 stopping criterion
};

// Standalone GD method (runs as "gd" from the terminal)
class GD : public Optimizer {
public:
    GD() = default;
    ~GD() override = default;
	
	std::string methodShortName() const override { return "gd"; }

    std::string methodFullName() const override { return "Gradient Descent (Local Method)"; }

    // Reads parameters from the [gd] section of the cfg (or from the integration command line)
    void configure(const MethodConfig& mc) override;

    // Optional getters for debugging/printing
    double step0()    const { return prm_.alpha0; }
    double momentum() const { return prm_.momentum; }
    double tol()      const { return prm_.tol_grad; }

protected:
    void init() override;
    void one_iteration() override;

private:
    // Helpers
    static double l2norm(const std::vector<double>& g);
    double eval(const std::vector<double>& x); // wrapper for prob_->evaluate(x)
    void project_to_bounds(std::vector<double>& x) const;

    GDParams prm_;
    std::vector<double> x_;          // current point
    std::vector<double> v_;          // momentum state (dx)
    double f_{std::numeric_limits<double>::infinity()};
    std::vector<double> FX_{1, std::numeric_limits<double>::infinity()}; // used by the stop controller
};

} // namespace optimsolution
