#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// RDE (Reconstructed Differential Evolution)
// S. Tao, R. Zhao, K. Wang, S. Gao, "An Efficient Reconstructed Differential
// Evolution Variant by Some of the Current State-of-the-art Strategies for
// Solving Single Objective Bound Constrained Problems", arXiv:2404.16280
// (IEEE CEC 2024, runner-up). Reference implementation by the authors:
// https://github.com/SichenTao/RDE, code/rde-cec2024/RDE.cpp, version 1.1.0
// (2026-10-10; identical to the 2024 competition code v1.0.0 except for one
// added bound repair after the Cauchy perturbation).
//
// This file follows THE AUTHORS' CODE, not only the paper. Where the two
// differ, the code wins (it is what produced the published results):
//   - gamma1 (share of the order-pbest/EB branch) = SUM of the improvements of
//     the EB trials / SUM over all improving trials; if either sum is 0 it is
//     reset to 0.5 (the paper speaks of average improvements);
//   - the "fixed 0.9" memory entry is an EXTRA (H+1-th) choice: the memory
//     index is drawn from {0..H}, and index H gives F~C(0.9,0.1), Cr~N(0.9,0.1);
//     all H real slots are updated;
//   - the Cr lower bounds are applied by stage only (Cr>=0.7 before 25% and
//     Cr>=0.6 before 50% of the budget), independently of F;
//   - M_Cr <- (Lehmer + old)/2, M_F <- Lehmer; a Cr memory slot becomes the
//     terminal value -1 (=> Cr = 0) when all successful Cr were 0;
//   - RSP weights are kr*(N-i), i = 0..N-1 (best first); the p-best choice
//     uses the first psize*N of these SAME weights (no re-ranking);
//   - the archive is used by BOTH branches: with probability |A|/(|A|+N) the
//     third donor (r2) is taken from the archive, chosen by RSP over the
//     archive's own ranking; otherwise from the population;
//   - the archive size shrinks as Ar*(Nmax-Nmin)*(1-nfe/maxfe) (>= Nmin);
//     parents replaced by an equal-or-better trial enter the archive;
//   - bound repair = midpoint between the violated bound and the parent value
//     (applied to the donor and, again, to the trial vector);
//   - Cauchy perturbation (iLSHADE-RSP): with probability pr per INDIVIDUAL,
//     EVERY dimension not taken from the donor is drawn from
//     Cauchy(x_ij, sigma); otherwise those dimensions keep x_ij.
//
// The only intended adaptation to optimsolution: the authors' Cauchy scale is
// an absolute 0.1 on CEC's [-100,100] boxes. Here it is expressed as a
// fraction of each dimension's range, with the default 0.1/200 = 0.0005 that
// reproduces the original exactly on [-100,100].
//
// Other optimsolution-specific additions kept from the previous version:
// "max_evals = N x D" support (setGeneralOptions), the optional
// min_generations cap on Nmax, and the in-run / end local-search hooks.
class RDE : public Optimizer {
public:
    RDE() = default;
    ~RDE() override = default;
    std::string methodShortName() const override { return "rde"; }
    std::string methodFullName()  const override { return "RDE (Reconstructed Differential Evolution)"; }

    // Opt-in to "[global] max_evals = <N>xD" (budget scales with the
    // problem's dimension) -- see options.h. The authors' protocol is
    // max_nfes = 10000*D.
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

    // Settings from [rde] in the configuration file. Defaults = the authors' code.
    void configure(const MethodConfig& mc) override {
        int fixedPop = mc.getInt("population", -1);
        if (fixedPop > 3) {
            fixed_population_ = fixedPop;
            this->setPopulation(fixedPop); // so StopController sees a sane size immediately
        }

        Nmax_mult_ = mc.getDbl("nmax_multiplier", Nmax_mult_);
        Nmin_      = mc.getInt("min_population", Nmin_);
        min_generations_ = mc.getInt("min_generations", min_generations_);
        H_         = mc.getInt("memory_size", H_);
        kr_        = mc.getDbl("kr", kr_);
        Ar_        = mc.getDbl("archive_rate", Ar_);
        pmax_      = mc.getDbl("pmax", pmax_);
        pr_        = mc.getDbl("local_prob", pr_);
        sigma_loc_ = mc.getDbl("local_sigma", sigma_loc_);
        muF0_      = mc.getDbl("mu_f0", muF0_);
        muCR0_     = mc.getDbl("mu_cr0", muCR0_);
        gamma1_0_  = mc.getDbl("gamma1_0", gamma1_0_);

        if (Nmin_ < 4) Nmin_ = 4;
        if (H_ < 1) H_ = 1;
        if (Ar_ < 0.0) Ar_ = 0.0;
        gamma1_0_ = std::min(std::max(gamma1_0_, 0.0), 1.0);

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
    // Weighted Lehmer mean of the successful values (authors' MeanWL).
    double meanWL(const std::vector<double>& v, const std::vector<double>& w) const;
    // Midpoint bound repair against the parent (authors' FindLimits).
    void findLimits(Vec& x, const Vec& parent) const;
    void addToArchive(const Vec& x, double f);

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archiveX_;
    std::vector<double> archiveF_;
    int                 archiveCap_{0};

    // ---- Parameters (defaults = authors' RDE.cpp) ----
    int    fixed_population_{-1}; // -1 = use Nmax_mult_*D scaling instead
    double Nmax_mult_{18.0};      // Nmax = 18*D
    int    min_generations_{100}; // optimsolution safeguard; see init()
    int    Nmin_{4};
    int    H_{5};
    double kr_{3.0};              // RSP weight slope
    double Ar_{1.0};              // archive size factor
    double pmax_{0.25};           // p-best fraction, p = pmax*(1-0.5*nfe/maxfe)
    double pr_{0.2};              // jumping_rate (Cauchy perturbation, per individual)
    double sigma_loc_{0.0005};    // Cauchy scale as fraction of the box range (0.1 on [-100,100])
    double muF0_{0.3};
    double muCR0_{0.8};
    double gamma1_0_{0.5};        // initial / reset value of the EB share

    // ---- Adaptive state ----
    std::vector<double> mem_F_;
    std::vector<double> mem_CR_;
    int    mem_pos_{0};
    double gamma1_{0.5}; // probability of the order-pbest (EB) branch
    int    Nmax_{0};

    std::string local_method_{"none"};
    double      local_rate_{0.0};

    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
