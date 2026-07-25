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
// (IEEE CEC 2024 benchmark). This is RDEx-SOP's predecessor: same success-
// history/linear-reduction lineage, but WITHOUT the exploitation-biased (EB)
// hybrid-rate schedule that RDEx tunes for early convergence speed. RDE
// instead hybridizes two mutation strategies via a fitness-improvement-ratio
// resource allocation (gamma1/gamma2), keeps a JADE/SHADE-style external
// archive, and uses rank-based selective pressure (RSP) for essentially all
// random donor selections (not just the p-best window). Per the CEC2025
// U-score results reported in the RDEx-SOP report (arXiv:2603.27089), RDE
// trades slower convergence speed for better final accuracy versus RDEx-SOP.
//
// Components (see the paper for the corresponding equation numbers):
//   - external archive (JADE/SHADE-style), used for r2 in the
//     current-to-pbest branch (Sec. II-B-1);
//   - two hybridized mutation branches -- current-to-pbest/1 (Eq. 2) and
//     current-to-order-pbest/1 (Eq. 5) -- with an adaptive resource-
//     allocation ratio gamma1/gamma2 based on relative average fitness
//     improvement (Eq. 6-8);
//   - Extended Rank-based Selective Pressure (RSP): the p-best window, r1,
//     r2, and the order-pbest donor set are all drawn with rank-weighted
//     probability (favoring fitter individuals) rather than uniformly
//     (Eq. 9-10);
//   - jSO-style success-history memory for F/Cr with a fixed terminal slot
//     and early-generation F/Cr clamps (Eq. 11-19);
//   - linear reduction of both the pbest window p and the population size,
//     with the population scaled to problem dimension: Nmax = 18*D
//     (Eq. 20-21) -- unlike RDEx-SOP's fixed front size, this scales
//     naturally across the wide dimension range in this codebase;
//   - Cauchy local perturbation on a non-crossover dimension (Eq. 22).
//
// Adaptation note: Eq. 22 specifies a FIXED absolute Cauchy scale (0.1),
// tuned for CEC's [-100,100] domains, applied independently to every
// non-crossover dimension. optimsolution's problems span vastly different
// natural scales and dimensions (D=5..216), so -- exactly mirroring the fix
// applied to rdex.cpp after diagnosing that a per-dimension gate makes the
// perturbation's disruptiveness grow with D (the wrong direction for a
// "light" local operator) -- this implementation instead perturbs at most
// ONE randomly chosen non-crossover dimension per individual, with the
// scale expressed as a fraction of that dimension's box range.
//
// Second adaptation note: the report's own experimental protocol scales the
// evaluation budget with dimension (max_nfes = 10000*D), so Nmax = 18*D
// yields a roughly D-INDEPENDENT number of generations there. optimsolution
// instead uses a single FIXED max_evals regardless of D (see [global] in the
// .cfg), so pairing that with an unmodified Nmax = 18*D starves
// high-dimensional runs of generations (e.g. ~2700 generations at D=6 vs.
// only ~77 at D=216 under a 150000-eval budget) -- exactly the kind of
// architectural mismatch that would make RDE look artificially weak on this
// benchmark suite regardless of the algorithm's own merit. To compensate,
// Nmax is additionally capped so that at least `min_generations` generations
// remain feasible given the actual max_evals budget (see init()).
class RDE : public Optimizer {
public:
    RDE() = default;
    ~RDE() override = default;
    std::string methodShortName() const override { return "rde"; }
    std::string methodFullName()  const override { return "RDE (Reconstructed Differential Evolution)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [rde] in the configuration file. Defaults match the
    // reference implementation's reported configuration (report, Sec. III-B).
    void configure(const MethodConfig& mc) override {
        // Optional fixed-population override; default behavior scales with D
        // (Nmax = nmax_multiplier * D), computed once the problem is known
        // (see init()).
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

    // Rank-based selective pressure (Eq. 9-10): builds a discrete
    // distribution over POSITIONS in `order` (ascending-fitness indices,
    // best first) using Rank_i = kr*(M-i)+1, i = 1-based rank.
    std::discrete_distribution<int> buildRspDistribution(const std::vector<int>& order) const;

    double sampleTruncGaussian(double mean, double sigma, double lo, double hi);
    double sampleCauchyParam(double loc, double scale, double lo, double hi);

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archiveX_;
    std::vector<double> archiveF_;

    // ---- Parameters (defaults per the RDE report, Sec. III-B) ----
    int    fixed_population_{-1}; // -1 = use Nmax_mult_*D scaling instead
    double Nmax_mult_{18.0};      // Nmax = Nmax_mult_ * D (report default: 18)
    int    min_generations_{100}; // budget-aware cap; see init() and the class comment below
    int    Nmin_{4};
    int    H_{5};
    double kr_{3.0};
    double Ar_{1.0};
    double pmax_{0.25};
    double pr_{0.2};
    double sigma_loc_{0.1};       // fraction of box range (see adaptation note above)
    double muF0_{0.3};
    double muCR0_{0.8};
    double gamma1_0_{0.5};

    // ---- Adaptive state ----
    std::vector<double> mem_F_;
    std::vector<double> mem_CR_;
    int    mem_pos_{0};
    double gamma1_{0.5}; // resource share for the order-pbest branch
    int    Nmax_{0};

    std::string local_method_{"none"};
    double      local_rate_{0.0};

    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
