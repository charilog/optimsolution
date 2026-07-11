#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// ============================================================================
// MSCSO -- Modified Sand Cat Swarm Optimization (multi-strategy fusion).
//
// Peng, H.; Zhang, X.; Li, Y.; Qi, J.; Kan, Z.; Meng, H.
// "A Modified Sand Cat Swarm Optimization Algorithm Based on Multi-Strategy
// Fusion and Its Application in Engineering Problems."
// Mathematics 2024, 12, 2153. https://doi.org/10.3390/math12142153
//
// Base algorithm: Sand Cat Swarm Optimization (SCSO), Seyyedabbasi & Kiani
// (2022), improved here with three fused strategies:
//
//  (1) Good point set population initialization (Eqs. 9-12 in the paper):
//      replaces uniform-random initialization with Hua Luogeng's "good
//      point set" (good lattice point) construction, giving a more evenly
//      spread initial population.
//
//  (2) Nonlinear adjustment of the sensitivity range rg (Eqs. 13-14):
//        f  = sin( (pi/4) * t/Tmax )
//        rg = SM - SM*(t/Tmax)*f              (replaces the plain linear
//                                                rg = SM - SM*t/Tmax of the
//                                                base SCSO, Eq. 3)
//      rg no longer decays fully to 0 by t=Tmax, preserving some global
//      search capability even in late iterations.
//
//  (3) Sparrow Search Algorithm early-warning mechanism (Eq. 15), applied
//      to a randomly chosen warning_frac_ fraction of the population each
//      iteration, with a greedy accept-if-better rule.
//
// NOTE on two equations that are ambiguous/inconsistent as printed in the
// paper (each fixed here to the mathematically/algorithmically coherent
// reading, consistent with the paper's own prose and pseudocode):
//
//  - Eq. (6) (exploration/search step) is printed as
//      x = r * (xbc - rand(0,1)) * xc
//    which is dimensionally inconsistent (a vector minus a bare scalar,
//    then multiplied by another vector). This is implemented as the
//    standard SCSO exploration update
//      x = r * (xbc - rand(0,1) * xc)
//    a well-formed convex-like blend of the best and current positions.
//
//  - Eq. (15) (sparrow warning mechanism) is printed with left-hand side
//    "x_b(t+1) = ...", i.e. updating the GLOBAL BEST position. Both the
//    right-hand-side expressions and the paper's own explanation ("the
//    sand cat ... can randomly move within this range", referring to the
//    individual sand cat, not the shared best) make clear that it is the
//    CURRENT WARNING INDIVIDUAL's own position that gets updated. This is
//    also how the equivalent step is defined in the original Sparrow
//    Search Algorithm (Xue & Shen, 2020) that this mechanism is borrowed
//    from. Implemented accordingly.
//
//  - R, r, and rg are computed once per ITERATION (a single value shared
//    across the whole population that iteration), not once per individual
//    and not only once before the main loop. The paper's pseudocode shows
//    them computed outside the "for every individual" loop, and Figures 3
//    and 4 plot exactly one R value per iteration across all Tmax=1000
//    iterations, confirming this reading (computing R,r,rg only once
//    before the very first iteration, as a strict pseudocode reading might
//    suggest, would freeze the exploration/exploitation balance for the
//    entire run and contradicts both figures and the surrounding prose).
//
// beta (step-size control) ~ N(0,1) and K ~ U(-1,1), following the
// conventions of the original Sparrow Search Algorithm that Eq. (15) is
// borrowed from (the paper itself only names these without giving exact
// distributions).
// ============================================================================
class MSCSO : public Optimizer {
public:
    MSCSO() = default;
    ~MSCSO() override = default;
    std::string methodShortName() const override { return "mscso"; }
    std::string methodFullName()  const override { return "Modified Sand Cat Swarm Optimization (multi-strategy fusion)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // configure(): reads method-specific parameters from [mscso]
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 3) {
            pop_ = pop_override;
        }

        SM_           = mc.getDbl("SM", SM_);                     // auditory feature (paper: SM = 2)
        warning_frac_ = mc.getDbl("warning_frac", warning_frac_); // paper: 30% of the population
        eps_          = mc.getDbl("eps", eps_);                   // small constant, avoids div-by-zero in Eq.(15)

        // In-run local (as in DE/PSO: only after a successful improvement)
        local_method_ = mc.getStr("local_method", local_method_);
        for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override; // Final polishing controlled by [global]

private:
    void ensureBounds(std::vector<double>& x);
    double eval(const std::vector<double>& v){ return prob_->evaluate(v); }

    // Good point set (Hua Luogeng): fills F (n x z) with n points in [0,1]^z.
    void goodPointSet(int n, int z, std::vector<std::vector<double>>& F) const;

private:
    // Population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;

    // Global best (x_b) -- SCSO tracks only a swarm-wide best, no personal bests
    std::vector<double> Xbest_;
    double               Fbest_{std::numeric_limits<double>::infinity()};

    // SCSO / MSCSO parameters
    double SM_{2.0};            // auditory feature
    double warning_frac_{0.3};  // fraction of the population treated as "warning" sand cats
    double eps_{1e-10};         // Eq.(15) denominator guard

    // In-run local (as in DE/PSO)
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // Final polishing (in end) from [global]
    bool        end_local_refine_ = false;
    std::string end_local_method_ = "";
};

} // namespace optimsolution
