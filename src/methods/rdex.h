#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution
// (S. Tao, Y. Yang, R. Zhao, K. Wang, S. Liu, S. Gao) -- used in the IEEE
// CEC 2025 numerical optimisation competition (C06, bound-constrained SOP
// track). Reference: "RDEx-SOP: Exploitation-Biased Reconstructed
// Differential Evolution for Fixed-Budget Bound-Constrained Single-Objective
// Optimization", arXiv:2603.27089. Source repository:
// https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx (RDEx_SOP).
//
// This is a from-scratch C++ reconstruction of the algorithm as specified by
// the technical report's equations/pseudocode, adapted to this codebase's
// Optimizer interface (see optimizer.h / woa.h for the conventions followed).
// A few numeric details are not spelled out in the report (e.g. the exact
// Gaussian sigma used for CR sampling); those are filled in with the
// standard SHADE/L-SHADE/iLSHADE-RSP convention and flagged in rdex.cpp.
//
// Design (see the report for the corresponding equation numbers):
//   - an elite "front" population with linear size reduction, N0 -> Nmin
//     as the evaluation budget is consumed (L-SHADE style, Eq. 11);
//   - two mutation branches with an adaptive hybrid rate rho_EB:
//       * standard: current-to-pbest/1 with an extra difference term (Eq. 2)
//       * EB (exploitation-biased): ordered best/mid/worst donor set (Eq. 5)
//     rho_EB is updated after each generation from the relative fitness
//     gains contributed by each branch (Eq. 6);
//   - success-history memories M_F, M_CR (H slots) updated via a
//     Delta-f-weighted Lehmer mean (Eq. 8-9);
//   - binomial crossover with bound repair by resampling (Eq. 4);
//   - a light Cauchy local perturbation on dimensions crossover did not
//     touch (Eq. 10).
class RDEx : public Optimizer {
public:
    RDEx() = default;
    ~RDEx() override = default;
    std::string methodShortName() const override { return "rdex"; }
    std::string methodFullName()  const override { return "RDEx-SOP (Reconstructed DE, exploitation-biased hybrid)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [rdex] in the configuration file. Defaults match the
    // reference implementation's reported configuration (report, Sec. III-B).
    void configure(const MethodConfig& mc) override {
        int n0 = mc.getInt("population", N0_);
        if (n0 > 3) {
            N0_ = n0;
            this->setPopulation(N0_); // so StopController sees the right size, cf. WOA::configure()
        }

        Nmin_      = mc.getInt("min_population", Nmin_);
        H_         = mc.getInt("memory_size", H_);
        rho_eb0_   = mc.getDbl("rho_eb0", rho_eb0_);
        p_r_       = mc.getDbl("local_prob", p_r_);
        sigma_loc_ = mc.getDbl("local_sigma", sigma_loc_);
        sigma_F_   = mc.getDbl("sigma_f", sigma_F_);
        sigma_cr_  = mc.getDbl("sigma_cr", sigma_cr_);
        xi_        = mc.getDbl("xi", xi_);
        k_         = mc.getDbl("k", k_);

        if (Nmin_ < 4) Nmin_ = 4;
        if (H_ < 1) H_ = 1;
        rho_eb0_ = std::min(std::max(rho_eb0_, 0.0), 1.0);

        // Optional in-run local search hook, same convention as other methods.
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    double eval(const Vec& x) { return prob_->evaluate(x); }

    // Random index into the current front, excluding up to two given indices.
    int pickRandomExcept(int exclude1, int exclude2 = -1);

    // Resample-until-in-range samplers (standard SHADE-style truncation).
    double sampleTruncGaussian(double mean, double sigma, double lo, double hi);
    double sampleCauchyParam(double loc, double scale, double lo, double hi);

private:
    // Current front (population); shrinks over the run via linear reduction.
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // ---- Parameters (defaults per the RDEx-SOP report, Sec. III-B) ----
    int    N0_{600};
    int    Nmin_{4};
    int    H_{5};
    double rho_eb0_{0.7};
    double p_r_{0.1};
    double sigma_loc_{0.1};
    double sigma_F_{0.02};
    double sigma_cr_{0.1};   // standard SHADE-style default; not explicitly numbered in the report.
    double xi_{0.7};
    double k_{7.0};

    // ---- Adaptive / success-history state ----
    std::vector<double> mem_F_;
    std::vector<double> mem_CR_;
    int    mem_pos_{0};
    double rho_eb_{0.7};
    double sr_prev_{0.0}; // success rate of the previous generation

    // in-run local search (same convention as woa.h)
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
