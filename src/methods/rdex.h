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
// (S. Tao, Y. Yang, R. Zhao, K. Wang, S. Liu, S. Gao) -- rank 1 in the IEEE
// CEC 2025 bound-constrained SOP track. Report: arXiv:2603.27089. Reference
// implementation by the authors:
// https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx,
// RDEx_SOP/code/rdex-sop/RDEx.cpp, version 1.0.0 (2026-10-10). The 2025
// competition code (commit 644e334) is the same algorithm except p_r = 0.4 and
// no bound repair after the Cauchy perturbation; v1.0.0 aligns both with the
// report.
//
// This file follows THE AUTHORS' CODE. The code is built on LSRTDE (Stanovov
// et al., CEC 2024 winner); several of its mechanisms are not described in
// the report but determine the algorithm's behaviour:
//   - two populations: the "front" (size N, shrinks linearly N0 -> Nmin, the
//     population that is evolved) and the "current" population (size N plus
//     the accepted trials of the generation, then truncated to its best N);
//   - each of the N trials of a generation uses a target drawn UNIFORMLY WITH
//     REPLACEMENT from the front, and an accepted trial overwrites the front
//     slot PFIndex, which advances cyclically (not the target's slot);
//   - standard branch: v = xF_t + F (xC_pbest - xF_t) + F (xF_r1 - xC_r2),
//     pbest uniform among the best max(2, N*xi*exp(-k*SR)) of the current
//     population, r1 from the front with rank weights exp(-3 i/N), r2 uniform
//     from the current population; F ~ N(0.4+0.25 tanh(5 SR), 0.02) redrawn
//     into [0,1], Cr ~ N(M_Cr, 0.05) clipped to [0,1];
//   - EB branch (ordered best/mid/worst donors) is used ONLY AFTER 70% OF THE
//     BUDGET, and then with probability rho_EB; pbest from the best
//     0.17*N*(1-0.5*nfe/maxfe) of the current population, r1 (front) and r2
//     (current) all with RSP weights 3*(N-i); F ~ Cauchy(M_F or 0.9, 0.1),
//     Cr ~ N(M_Cr or 0.9, 0.1) using H+1 memory choices (the extra one = 0.9);
//   - rho_EB = sum of EB improvements / (sum EB + sum standard); reset to 0.7
//     when either sum is 0 (therefore always 0.7 during the first 70%);
//   - memories initialised to 1.0, SR to 0.5; the Cr stored for the memory
//     update is the ACTUAL fraction of dimensions taken from the donor;
//     M_Cr <- (Lehmer + old)/2, M_F <- Lehmer (fallback 1.0);
//   - Cauchy perturbation: with probability p_r = 0.1 per TRIAL, every
//     non-crossover dimension is drawn from Cauchy(x_tj, sigma); components
//     outside the box are resampled uniformly (also for the donor);
//   - N0 = 18*D (the code uses PopSize = 18 -> 540 at D = 30; the report's
//     "600" corresponds to the 20*D used only in the time-complexity test).
//
// Intended adaptation to optimsolution: the authors' Cauchy scale is an
// absolute 0.1 on [-100,100]; here it is a fraction of each dimension's range,
// default 0.1/200 = 0.0005 (identical on [-100,100]).
// optimsolution additions: "max_evals = N x D" support, in-run / end
// local-search hooks.
class RDEx : public Optimizer {
public:
    RDEx() = default;
    ~RDEx() override = default;
    std::string methodShortName() const override { return "rdex"; }
    std::string methodFullName()  const override { return "RDEx-SOP (Reconstructed DE, exploitation-biased hybrid)"; }

    // Opt-in to "[global] max_evals = <N>xD", as in rde.h. The authors'
    // protocol is max_nfes = 10000*D, and N0 = 18*D is tuned for it.
    void setGeneralOptions(const GeneralOptions& g) override {
        Optimizer::setGeneralOptions(g);
        if (g.max_evals_per_dim > 0 && prob_) {
            max_evals_ = g.max_evals_per_dim * (long long)prob_->dimension();
        }
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [rdex] in the configuration file. Defaults = the authors' code.
    void configure(const MethodConfig& mc) override {
        int fixedPop = mc.getInt("population", -1);
        if (fixedPop > 3) {
            fixed_population_ = fixedPop;
            this->setPopulation(fixedPop); // so StopController sees the right size, cf. WOA::configure()
        }

        Nmax_mult_   = mc.getDbl("nmax_multiplier", Nmax_mult_);
        Nmin_        = mc.getInt("min_population", Nmin_);
        H_           = mc.getInt("memory_size", H_);
        rho_eb0_     = mc.getDbl("rho_eb0", rho_eb0_);
        eb_start_    = mc.getDbl("eb_start", eb_start_);
        eb_pfrac_    = mc.getDbl("eb_pbest", eb_pfrac_);
        p_r_         = mc.getDbl("local_prob", p_r_);
        sigma_loc_   = mc.getDbl("local_sigma", sigma_loc_);
        sigma_F_     = mc.getDbl("sigma_f", sigma_F_);
        sigma_cr_    = mc.getDbl("sigma_cr", sigma_cr_);
        sigma_cr_eb_ = mc.getDbl("sigma_cr_eb", sigma_cr_eb_);
        xi_          = mc.getDbl("xi", xi_);
        k_           = mc.getDbl("k", k_);
        mem0_        = mc.getDbl("memory_init", mem0_);
        sr0_         = mc.getDbl("sr0", sr0_);

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
    double progress() const;
    double meanWL(const std::vector<double>& v, const std::vector<double>& w) const;

private:
    // Front (evolved population) and current population (see class comment).
    std::vector<Vec>    XF_;
    std::vector<double> FF_;
    std::vector<Vec>    XC_;
    std::vector<double> FC_;

    // ---- Parameters (defaults = authors' RDEx.cpp v1.0.0) ----
    int    fixed_population_{-1};
    double Nmax_mult_{18.0};   // N0 = 18*D
    int    Nmin_{4};
    int    H_{5};
    double rho_eb0_{0.7};      // initial / reset EB rate
    double eb_start_{0.7};     // EB branch only when nfe/maxfe >= 0.7
    double eb_pfrac_{0.17};    // EB p-best fraction, *(1-0.5*nfe/maxfe)
    double p_r_{0.1};          // Cauchy perturbation probability (per trial)
    double sigma_loc_{0.0005}; // Cauchy scale as fraction of the box range (0.1 on [-100,100])
    double sigma_F_{0.02};     // standard-branch F spread
    double sigma_cr_{0.05};    // standard-branch Cr spread
    double sigma_cr_eb_{0.1};  // EB-branch Cr spread
    double xi_{0.7};
    double k_{7.0};
    double mem0_{1.0};         // initial M_F and M_Cr
    double sr0_{0.5};          // initial success rate

    // ---- Adaptive state ----
    int    N0_{0};
    std::vector<double> mem_F_;
    std::vector<double> mem_CR_;
    int    mem_pos_{0};
    double rho_eb_{0.7};
    double sr_{0.5};
    int    pf_index_{0};
    std::vector<double> fit_mass_;  // front fitness at the end of the previous generation
    std::vector<double> fit_temp_;  // last trial fitness per front slot (persists, as in the authors' code)
    std::vector<char>   eb_flag_;   // branch of that last trial per front slot

    // in-run local search (same convention as woa.h)
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
