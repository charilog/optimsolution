#pragma once
#include <vector>
#include <random>
#include <limits>
#include "optimizer.h"
#include "init.h"

namespace optimsolution {

// Nelder–Mead settings (classic simplex)
struct NMParams {
    double alpha = 1.0;        // reflection coeff
    double gamma = 2.0;        // expansion coeff
    double beta  = 0.5;        // contraction coeff
    double delta = 0.5;        // shrink coeff
    double simplex_delta = 1e-2; // Initial simplex "spread" around x0
    int    simplex_iters = 1;    // Number of Nelder–Mead steps executed per one_iteration()
};

class NM : public Optimizer {
public:
    NM() = default;
    ~NM() override = default;
	
	std::string methodShortName() const override { return "nd"; }

    std::string methodFullName() const override { return "Nelder-Mead Simplex (Local Method)"; }

    // Reads parameters from [nm] in the configuration file
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;

private:
    // Builds the initial simplex around an initial point
    void buildInitialSimplex();

    // Evaluates f(x) via prob_->evaluate and returns the result
    double evalPoint(const std::vector<double>& x);

    // Projects a point onto the problem bounds (to stay within lb/ub)
    void project_to_bounds(std::vector<double>& x) const;

    // Performs a full Nelder–Mead step (reflection, expansion, contraction, shrink)
    void nelderMeadStep();

    // Sorts simplex_ and fSimplex_ by f (ascending)
    void sortSimplex();

    // Computes the centroid of all vertices except the worst one
    std::vector<double> centroidExceptWorst() const;

    // Combines two vectors: coeff_a * a + coeff_b * b (with projection onto bounds)
    std::vector<double> combine(const std::vector<double>& a,
                                const std::vector<double>& b,
                                double coeff_a,
                                double coeff_b) const;

    // Squared Euclidean distance (useful for optional convergence tests)
    double vecDist2(const std::vector<double>& a,
                    const std::vector<double>& b) const;

private:
    NMParams prm_;

    // Simplex points: dimension()+1 vertices
    std::vector<std::vector<double>> simplex_;
    // Objective values at each vertex
    std::vector<double> fSimplex_;

    // Optional buffer (if needed)
    std::vector<double> tmp_point_;

    // For stopping/logging: stores the best f in a small vector
    // so that updateStop(FX_) can be called
    std::vector<double> FX_{1, std::numeric_limits<double>::infinity()};
};

} // namespace optimsolution
