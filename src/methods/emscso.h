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
// EMSCSO -- Enhanced Modified Sand Cat Swarm Optimization.
//
// Builds on MSCSO (Peng et al., "A Modified Sand Cat Swarm Optimization
// Algorithm Based on Multi-Strategy Fusion and Its Application in
// Engineering Problems", Mathematics 2024, 12, 2153 -- good point set init,
// nonlinear exploration/exploitation adjustment, sparrow-search early
// warning) and adds two stagnation-response mechanisms carried over from
// this framework's SPARQ optimizer, adapted to MSCSO's much simpler
// population model (no SHADE memory, no archive, no eigen-basis -- SPARQ's
// elitePolish()/rejuvenate() are deeply tied into that DE-specific
// machinery, so what is ported here is the CONCEPT, re-implemented against
// plain X_/FX_/Xbest_ state, not a verbatim copy):
//
//  (4) elitePolish  -- stagnation-gated local refinement of the incumbent
//      best. When no_improve_ (consecutive iterations without ANY
//      improvement of Fbest_) reaches polish_trigger_, a short burst of
//      polish_frac_*pop_ probes is tried around best_x_: half single-
//      coordinate steps (opposition or adaptive Gaussian -- good at
//      repairing "one dimension off" local minima), half full-D isotropic
//      Gaussian steps. Any improving probe is injected in place of the
//      population's current worst individual, propagating the refinement
//      back into the swarm. Step sizes adapt by the classic 1.5x-on-success
//      / 0.87x-on-failure rule. Two safety nets bound the risk: a hard
//      per-run evaluation-budget cap (polish_budget_ fraction of evals so
//      far) and an efficiency check comparing the burst's own gain-per-eval
//      against the main loop's gain-per-eval since the last burst -- once
//      polish is measured to be the worse investment on this landscape, it
//      backs off exponentially instead of continuing to skim budget.
//
//  (5) rejuvenate -- hard-stagnation partial restart. Once no_improve_
//      reaches rejuv_trigger_ (well past the point where elitePolish alone
//      can help), the worst (1 - rejuv_keep_) fraction of the population is
//      re-initialised with a fresh good-point-set sample while the best
//      rejuv_keep_ fraction survives untouched. best_x_/Xbest_ are never
//      part of the population and are therefore never at risk from this --
//      the incumbent best is preserved by construction, exactly as in
//      SPARQ's version of this mechanism.
//
//  (6) Opposition-Based Learning (OBL) basin escape (Tizhoosh, 2005),
//      adapted from SPARQ's oblBasinEscape(). Population-comparison data
//      (30-run benchmarks against sparq) showed MSCSO/EMSCSO trailing sparq
//      mainly on SUCCESS RATE / RUN-TO-RUN RELIABILITY (e.g. eld1: 3% vs
//      53%; rastrigin: 87% vs 100%; test2n: 3% vs 100%; test30n: 3% vs
//      87%) rather than on the best value found -- the classic signature of
//      premature convergence: whole populations collapsing onto an inferior
//      basin with no mechanism strong enough to escape it before the
//      (much rarer) hard-stagnation rejuvenate fires. OBL targets exactly
//      this: whenever the population's own spread has collapsed
//      (normalizedPopSpread() below var_collapse_ratio_) AND no_improve_
//      has reached obl_trigger_, the worst obl_frac_ fraction of the
//      population is replaced by quasi-opposition points -- for each
//      affected individual, a 50/50 mix of (a) the pure box-opposite point
//      (lb+ub-x, which by construction lands in the "other half" of the
//      search space from a converged cluster) and (b) a point on the
//      segment between that opposite and the current best. This is a
//      fundamentally different DIRECTION of repair than elitePolish (local,
//      small steps around best) or rejuvenate (uniform-random reseeding) --
//      it deliberately jumps to the OTHER SIDE of the box, which is where a
//      converged-elsewhere population is, almost by definition, least
//      likely to have looked.
//
//  (7) Levy-flight exploration jumps (Mantegna, 1994; popularised for
//      metaheuristics by Yang & Deb's Cuckoo Search, 2009). The base SCSO
//      "search" step (|R|>1, Eq. 6) is a tame convex-ish blend of the
//      current position and the best; it has no mechanism for an
//      occasional LONG jump, so once a run's whole population has drifted
//      into a basin, ordinary exploration steps rarely have the reach to
//      leave it. With probability levy_prob_, the exploration step is
//      replaced by a heavy-tailed Levy flight around the current position
//      biased toward the best (x_i + levy_scale_ * Levy(beta) * (x_i -
//      Xbest_)): most draws are modest, but the tail occasionally produces
//      a large jump, giving MSCSO the same long-range escape capability
//      that Levy-flight-based algorithms are specifically known for.
// ============================================================================
class EMSCSO : public Optimizer {
public:
    EMSCSO() = default;
    ~EMSCSO() override = default;
    std::string methodShortName() const override { return "emscso"; }
    std::string methodFullName()  const override {
        return "Enhanced Modified Sand Cat Swarm Optimization "
               "(MSCSO + stagnation-gated elite polish + hard-stagnation rejuvenation)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // configure(): reads method-specific parameters from [emscso]
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 3) {
            pop_ = pop_override;
        }

        SM_           = mc.getDbl("SM", SM_);
        warning_frac_ = mc.getDbl("warning_frac", warning_frac_);
        eps_          = mc.getDbl("eps", eps_);

        // elitePolish parameters
        polish_trigger_    = mc.getInt("polish_trigger", polish_trigger_);
        polish_frac_       = mc.getDbl("polish_frac", polish_frac_);
        ps_sigma_          = mc.getDbl("ps_sigma", ps_sigma_);
        ps_sigma_c_        = mc.getDbl("ps_sigma_c", ps_sigma_c_);
        ps_sigma_min_      = mc.getDbl("ps_sigma_min", ps_sigma_min_);
        ps_sigma_max_      = mc.getDbl("ps_sigma_max", ps_sigma_max_);
        polish_budget_     = mc.getDbl("polish_budget", polish_budget_);
        polish_min_relgain_= mc.getDbl("polish_min_relgain", polish_min_relgain_);

        // rejuvenate parameters
        rejuv_trigger_      = mc.getInt("rejuv_trigger", rejuv_trigger_);
        rejuv_keep_         = mc.getDbl("rejuv_keep", rejuv_keep_);
        rejuv_cooldown_init_= mc.getInt("rejuv_cooldown", rejuv_cooldown_init_);

        // OBL basin-escape parameters
        obl_trigger_        = mc.getInt("obl_trigger", obl_trigger_);
        obl_frac_           = mc.getDbl("obl_frac", obl_frac_);
        obl_cooldown_init_  = mc.getInt("obl_cooldown", obl_cooldown_init_);
        var_collapse_ratio_ = mc.getDbl("var_collapse_ratio", var_collapse_ratio_);

        // Levy-flight exploration parameters
        levy_beta_  = mc.getDbl("levy_beta", levy_beta_);
        levy_prob_  = mc.getDbl("levy_prob", levy_prob_);
        levy_scale_ = mc.getDbl("levy_scale", levy_scale_);
        levy_ramp_  = mc.getInt("levy_ramp", levy_ramp_);

        // In-run local (as in DE/PSO/MSCSO: only after a successful improvement)
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
    double randU() { std::uniform_real_distribution<double> d(0.0, 1.0); return d(rng_); }
    double gaussN() { std::normal_distribution<double> d(0.0, 1.0); return d(rng_); }

    // Good point set (Hua Luogeng): fills F (n x z) with n points in [0,1]^z.
    void goodPointSet(int n, int z, std::vector<std::vector<double>>& F) const;

    void elitePolish();
    void rejuvenate();
    void oblBasinEscape();
    double normalizedPopSpread() const;
    double sampleLevy();

private:
    // Population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;

    // Global best (x_b)
    std::vector<double> Xbest_;
    double               Fbest_{std::numeric_limits<double>::infinity()};

    // SCSO / MSCSO parameters
    double SM_{2.0};
    double warning_frac_{0.3};
    double eps_{1e-10};

    // In-run local (as in DE/PSO)
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // Final polishing (in end) from [global]
    bool        end_local_refine_ = false;
    std::string end_local_method_ = "";

    // ------------------------------------------------------------
    // Stagnation tracking (drives both elitePolish and rejuvenate)
    // ------------------------------------------------------------
    int no_improve_{0};

    // ------------------------------------------------------------
    // elitePolish state (adapted from SPARQ's elitePolish())
    // ------------------------------------------------------------
    int    polish_trigger_{10};       // no_improve_ iterations before polishing
    double polish_frac_{0.10};        // evals per activation = frac * pop_
    double ps_sigma_{0.02};           // adaptive full-D step (fraction of box range)
    double ps_sigma_c_{0.10};         // adaptive single-coordinate step
    double ps_sigma_min_{1e-9};
    double ps_sigma_max_{0.25};
    double polish_budget_{0.12};      // hard cap: fraction of evals spent so far
    double polish_min_relgain_{1e-3};
    int    polish_coord_ptr_{0};      // round-robin coordinate sweep pointer
    int    polish_low_streak_{0};
    int    polish_cooldown_{0};
    int    polish_backoff_{0};
    long long polish_used_{0};
    double polish_mark_f_{std::numeric_limits<double>::infinity()};
    long long polish_mark_calls_{0};

    // ------------------------------------------------------------
    // rejuvenate state (adapted from SPARQ's rejuvenate())
    // ------------------------------------------------------------
    int    rejuv_trigger_{120};       // no_improve_ iterations before hard-stagnation rejuvenation
    double rejuv_keep_{0.25};         // elite fraction preserved (best-ranked)
    int    rejuv_cooldown_init_{150};
    int    rejuv_cooldown_{0};

    // ------------------------------------------------------------
    // OBL basin-escape state (adapted from SPARQ's oblBasinEscape())
    // ------------------------------------------------------------
    int    obl_trigger_{15};          // no_improve_ iterations before OBL may fire
    double obl_frac_{0.30};           // fraction of the (worst-ranked) population replaced
    int    obl_cooldown_init_{80};
    int    obl_cooldown_{0};
    double var_collapse_ratio_{1e-3}; // population-spread collapse threshold

    // ------------------------------------------------------------
    // Levy-flight exploration state (Mantegna, 1994)
    // ------------------------------------------------------------
    double levy_beta_{1.5};           // stability index (1.5 = standard Cuckoo Search value)
    double levy_prob_{0.30};          // FULL probability of a Levy jump once fully ramped
    double levy_scale_{0.30};         // scales the jump relative to |x_i - Xbest_|
    int    levy_ramp_{8};             // no_improve_ iterations to reach full levy_prob_ (0 = always full)
};

} // namespace optimsolution
