#pragma once
#include <vector>
#include <random>
#include <limits>
#include <cstddef>
#include "optimizer.h"
#include "init.h"

namespace optimsolution {

// L-BFGS settings
struct LBFGSParams {
    int    m_history    = 10;     // number of (s, y) pairs retained
    double alpha0       = 1.0;    // initial step size for line search
    double c1           = 1e-4;   // Armijo constant
    int    backtracks   = 20;     // maximum number of backtracking steps
    double tol_grad     = 1e-8;   // termination criterion based on ||g||_2
};

// L-BFGS optimizer (invoked as "lbfgs")
class LBFGS : public Optimizer {
public:
    LBFGS() = default;
    ~LBFGS() override = default;

    // parameter loading from the [lbfgs] section in the configuration
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;

private:
    // Core logic
    void   updateHistory(const std::vector<double>& s,
                         const std::vector<double>& y);
    void   computeDirection(const std::vector<double>& g,
                            std::vector<double>& d) const; // d = -H*g approx
    double evalPoint(const std::vector<double>& x);
    void   project_to_bounds(std::vector<double>& x) const;
    static double dot(const std::vector<double>& a,
                      const std::vector<double>& b);
    static double l2norm(const std::vector<double>& a);

private:
    LBFGSParams prm_;

    // current state
    std::vector<double> x_;         // current position
    std::vector<double> g_;         // ∇f(x_)
    double f_{std::numeric_limits<double>::infinity()};

    // L-BFGS history
    // s_k = x_{k+1} - x_k
    // y_k = g_{k+1} - g_k
    std::vector<std::vector<double>> S_; // most recent s vectors
    std::vector<std::vector<double>> Y_; // most recent y vectors
    std::vector<double> rho_;            // 1 / (y^T s)

    // buffer used for stopping criteria / logging
    std::vector<double> FX_{1, std::numeric_limits<double>::infinity()};
};

} // namespace optimsolution
