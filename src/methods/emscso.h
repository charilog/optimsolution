#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>

namespace optimsolution {

// ============================================================================
// EMSCSO -- Enhanced Modified Sand Cat Swarm Optimization.
//
// Base algorithm: MSCSO (Peng, H.; Zhang, X.; Li, Y.; Qi, J.; Kan, Z.; Meng, H.
// "A Modified Sand Cat Swarm Optimization Algorithm Based on Multi-Strategy
// Fusion and Its Application in Engineering Problems." Mathematics 2024, 12,
// 2153), itself built on the original Sand Cat Swarm Optimization (SCSO,
// Seyyedabbasi & Kiani, 2022). EMSCSO keeps all three of MSCSO's fused
// strategies unchanged:
//
//  (1) Good point set population initialization (Hua Luogeng).
//  (2) Nonlinear adjustment of the sensitivity range rg.
//  (3) Sparrow Search Algorithm early-warning mechanism.
//
// (See mscso.h for the detailed derivation and the two paper-ambiguity fixes
// that both MSCSO and EMSCSO share -- Eq. 6 and Eq. 15.)
//
// EMSCSO adds three further mechanisms (not part of the Peng et al. paper;
// original to this framework). An earlier revision also added RTR
// (Restricted Tournament Replacement) selection on the main SCSO step; RTR
// has been REMOVED (it competed with the swarm's own always-move dynamics
// and did not clearly pay for itself), and replaced with two new mechanisms
// designed around a single explicit goal: push the best-ever value found
// (across independent runs) lower, WITHOUT degrading the mean/success-rate
// of a typical run. Both new mechanisms achieve this the same structural
// way: they are strictly elitist (an action is kept only if it improves on
// what it replaces -- never a regression), they act on the incumbent best
// point rather than on the bulk of the swarm, and their cost is either
// hard-capped or self-throttling -- so in a run that is already converging
// well (the runs that make up "mean"/"rate"), they are cheap-to-free and
// essentially invisible; only in a run that is stuck do they spend real
// budget trying to pull the incumbent best lower.
//
//  (4) NLPSR -- Non-Linear Population Size Reduction, ported from SPARQ.
//      The swarm starts at Ninit_ individuals (the configured "population",
//      or popscale_*D if popscale_ > 0) and is shrunk once per iteration
//      toward Nmin_ following
//          N(t) = round( Ninit + (Nmin - Ninit) * progress(t)^expo )
//          expo = 1 - (1 - nlpsr_alpha_) * progress(t)
//      (progress(t) = evaluations used so far / evaluation budget). alpha=1
//      gives a linear LSHADE-style shrink; alpha<1 tracks NL-SHADE-RSP's
//      p^(1-p) shape (near-linear early, accelerating late). Shrinking
//      removes the worst-ranked individuals only (shrinkTo()), so
//      Xbest_/best_x_ -- kept outside the population array -- are never at
//      risk.
//
//  (5) Quarantine -- ported from ARQ's quarantine_and_restart() (the RTR
//      selection half of ARQ's "Adaptive RTR with Quarantine" is what was
//      removed; this is the other, unrelated half). IQR-based outlier
//      detection (any individual with fitness >= Q3 + outlier_alpha_*IQR is
//      a "quarantine" candidate) followed by re-seeding an outlier_rho_
//      fraction of those outliers as Gaussian perturbations (qsigma_ *
//      box-range) around the robust center (mean of the best half of the
//      swarm). Additionally, once no_improve_ reaches stagnation_trigger_,
//      a worst_frac_ fraction of the worst-ranked individuals are
//      micro-restarted as Gaussian perturbations (rsigma_ * box-range)
//      around the current best, each accepted only if it improves on the
//      slot it replaces.
//
//  (6) Elite Polish (NEW, replaces RTR) -- stagnation-gated multi-probe local
//      intensification of best_x_ ONLY (not the swarm at large). Once
//      no_improve_ reaches polish_trigger_, a burst of polish_frac_*N probes
//      is tried around best_x_: half single-coordinate Gaussian steps
//      (round-robin over dimensions -- cheap at repairing "one dimension
//      off" local optima), half full-D isotropic Gaussian steps. Every probe
//      is accept-only-if-strictly-better-than-Fbest_ -- it can only pull the
//      incumbent best down, never move it. Step sizes self-adapt
//      (x1.5 on an accepted probe, x0.87 otherwise, both clamped). A
//      successful burst's improvement is also written into the population's
//      current worst slot, letting the swarm inherit the refinement; an
//      unsuccessful burst touches nothing but its own probe evaluations.
//      Two guards keep this from ever being able to hurt the mean: a hard
//      cumulative eval cap (polish_budget_ fraction of evaluations spent so
//      far may ever go to polish across the whole run) and an
//      efficiency check -- if a burst's relative gain falls below
//      polish_min_relgain_, the mechanism backs off with an exponentially
//      growing cooldown instead of retrying every time it is merely
//      eligible, so a landscape where polish doesn't help stops spending
//      budget on it.
//
//  (7) Best-Anchored Levy Jump (NEW, replaces RTR) -- a single heavy-tailed
//      Levy-flight probe (Mantegna, 1994) launched from best_x_ itself:
//          x_probe = best_x_ + levy_scale_ * Levy(beta) * box_range
//      Unlike Elite Polish's local probes, this can occasionally reach far
//      enough to land in a different, possibly better, basin near the
//      incumbent -- the same long-range-escape property that makes
//      Levy-flight-based metaheuristics (e.g. Cuckoo Search) effective.
//      It is attempted with a small, stagnation-RAMPED probability
//      (levy_prob_, ramped in linearly over levy_ramp_ stagnant iterations,
//      so it is essentially inactive while the run is still improving
//      normally) and costs at most ONE extra evaluation per iteration when
//      it does fire. It is accepted only if it beats Fbest_, in which case
//      it also overwrites the population's current worst slot; otherwise it
//      is discarded and touches nothing else. Because it only ever competes
//      against the incumbent best and never against a population member
//      directly, a failed jump cannot make any run's outcome worse than it
//      would already have been.
//
// NLPSR, Quarantine, Elite Polish, and the Levy Jump are each independently
// toggleable from optimsolution.cfg's [emscso] section (enable_nlpsr,
// enable_quarantine, enable_polish, enable_levy_jump), defaulting to ON.
// ============================================================================
class EMSCSO : public Optimizer {
public:
    EMSCSO() = default;
    ~EMSCSO() override = default;
    std::string methodShortName() const override { return "emscso"; }
    std::string methodFullName()  const override {
        return "Enhanced Modified Sand Cat Swarm Optimization "
               "(MSCSO + NLPSR population reduction + ARQ-style quarantine + "
               "stagnation-gated elite polish + best-anchored Levy jump)";
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

        // --- NLPSR: non-linear population size reduction ---
        popscale_     = mc.getInt("popscale", popscale_);
        Nmin_         = mc.getInt("Nmin", Nmin_);
        nlpsr_alpha_  = mc.getDbl("nlpsralpha", nlpsr_alpha_);
        enable_nlpsr_ = mc.getInt("enable_nlpsr", enable_nlpsr_) ? 1 : 0;

        // --- Quarantine (ported from ARQ; RTR half removed) ---
        outlier_alpha_      = mc.getDbl("alpha", outlier_alpha_);
        outlier_rho_        = mc.getDbl("rho", outlier_rho_);
        qsigma_             = mc.getDbl("qsigma", qsigma_);
        worst_frac_         = mc.getDbl("w", worst_frac_);
        rsigma_             = mc.getDbl("rsigma", rsigma_);
        stagnation_trigger_ = mc.getInt("stagnationtrigger", stagnation_trigger_);
        enable_quarantine_  = mc.getInt("enable_quarantine", enable_quarantine_) ? 1 : 0;

        // --- Elite Polish (NEW) ---
        polish_trigger_     = mc.getInt("polish_trigger", polish_trigger_);
        polish_frac_        = mc.getDbl("polish_frac", polish_frac_);
        ps_sigma_           = mc.getDbl("ps_sigma", ps_sigma_);
        ps_sigma_c_         = mc.getDbl("ps_sigma_c", ps_sigma_c_);
        ps_sigma_min_       = mc.getDbl("ps_sigma_min", ps_sigma_min_);
        ps_sigma_max_       = mc.getDbl("ps_sigma_max", ps_sigma_max_);
        polish_budget_      = mc.getDbl("polish_budget", polish_budget_);
        polish_min_relgain_ = mc.getDbl("polish_min_relgain", polish_min_relgain_);
        enable_polish_      = mc.getInt("enable_polish", enable_polish_) ? 1 : 0;

        // --- Best-Anchored Levy Jump (NEW) ---
        levy_beta_       = mc.getDbl("levy_beta", levy_beta_);
        levy_prob_       = mc.getDbl("levy_prob", levy_prob_);
        levy_scale_      = mc.getDbl("levy_scale", levy_scale_);
        levy_ramp_       = mc.getInt("levy_ramp", levy_ramp_);
        enable_levy_jump_= mc.getInt("enable_levy_jump", enable_levy_jump_) ? 1 : 0;

        // In-run local (as in DE/PSO/MSCSO: only after a successful improvement)
        local_method_ = mc.getStr("local_method", local_method_);
        for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;

        // Sanity clamps
        if (Nmin_ < 4) Nmin_ = 4;
        if (nlpsr_alpha_ <= 0.0) nlpsr_alpha_ = 0.5;
        if (outlier_rho_ < 0.0) outlier_rho_ = 0.0;
        if (outlier_rho_ > 1.0) outlier_rho_ = 1.0;
        if (worst_frac_ < 0.0) worst_frac_ = 0.0;
        if (worst_frac_ > 1.0) worst_frac_ = 1.0;
        if (stagnation_trigger_ < 1) stagnation_trigger_ = 1;
        if (polish_trigger_ < 1) polish_trigger_ = 1;
        if (polish_frac_ < 0.0) polish_frac_ = 0.0;
        if (ps_sigma_min_ <= 0.0) ps_sigma_min_ = 1e-9;
        if (ps_sigma_max_ < ps_sigma_min_) ps_sigma_max_ = ps_sigma_min_;
        if (polish_budget_ < 0.0) polish_budget_ = 0.0;
        if (polish_budget_ > 1.0) polish_budget_ = 1.0;
        if (levy_beta_ <= 0.0 || levy_beta_ >= 2.0) levy_beta_ = 1.5;
        if (levy_prob_ < 0.0) levy_prob_ = 0.0;
        if (levy_prob_ > 1.0) levy_prob_ = 1.0;
        if (levy_ramp_ < 0) levy_ramp_ = 0;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override; // Final polishing controlled by [global]

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& x);
    double eval(const Vec& v){ return prob_->evaluate(v); }
    double randU();
    double gaussN(double mu, double sig);

    // Good point set (Hua Luogeng): fills F (n x z) with n points in [0,1]^z.
    void goodPointSet(int n, int z, std::vector<std::vector<double>>& F) const;

    // --- NLPSR (ported from SPARQ) ---
    double progress01() const;
    int    targetPopulationSize() const;
    void   shrinkTo(int Ntarget);

    // --- Quarantine (ported from ARQ; RTR half removed) ---
    static double quantile(std::vector<double> v, double q01);
    void   quarantineAndRestart();

    // --- Elite Polish (NEW) ---
    void   elitePolish();

    // --- Best-Anchored Levy Jump (NEW) ---
    double sampleLevy(); // Mantegna 1994
    void   bestLevyJump();

    // shared helper: index of the current worst individual
    int worstIndex() const;

private:
    // Population
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // Global best (x_b) -- kept outside the population array
    Vec    Xbest_;
    double Fbest_{std::numeric_limits<double>::infinity()};

    // SCSO / MSCSO parameters
    double SM_{2.0};            // auditory feature
    double warning_frac_{0.3};  // fraction of the population treated as "warning" sand cats
    double eps_{1e-10};         // Eq.(15) denominator guard

    // ------------------------------------------------------------
    // NLPSR: non-linear population size reduction (from SPARQ)
    // ------------------------------------------------------------
    int    Ninit_{0};            // initial swarm size (from "population", or popscale_*D)
    int    Nmin_{10};            // floor the swarm may shrink to
    double nlpsr_alpha_{0.5};    // 1.0 = linear shrink; <1 = slower early, faster late
    int    popscale_{0};         // if > 0, Ninit_ = popscale_ * D (overrides "population")
    int    enable_nlpsr_{1};     // master switch (cfg: enable_nlpsr)

    // ------------------------------------------------------------
    // Quarantine (from ARQ; RTR half removed)
    // ------------------------------------------------------------
    double outlier_alpha_{1.0};       // quarantine: IQR multiplier for the outlier threshold
    double outlier_rho_{0.08};        // quarantine: fraction of outliers re-seeded per iteration
    double qsigma_{0.10};             // quarantine: re-seed Gaussian scale (fraction of box range)
    double worst_frac_{0.08};         // micro-restart: fraction of worst individuals restarted
    double rsigma_{0.18};             // micro-restart: Gaussian scale (fraction of box range)
    int    stagnation_trigger_{24};   // no_improve_ iterations before a micro-restart fires
    int    enable_quarantine_{1};     // master switch (cfg: enable_quarantine)

    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // ------------------------------------------------------------
    // Elite Polish (NEW: replaces RTR)
    // ------------------------------------------------------------
    int    polish_trigger_{10};       // no_improve_ iterations before a polish burst fires
    double polish_frac_{0.10};        // burst size = polish_frac_ * current population
    double ps_sigma_{0.02};           // adaptive full-D isotropic step (fraction of box range)
    double ps_sigma_c_{0.10};         // adaptive single-coordinate step (fraction of box range)
    double ps_sigma_min_{1e-9};
    double ps_sigma_max_{0.25};
    double polish_budget_{0.12};      // hard cap: fraction of evals-so-far polish may ever consume
    double polish_min_relgain_{1e-3}; // below this relative gain, back off
    int    enable_polish_{1};         // master switch (cfg: enable_polish)

    int       polish_coord_ptr_{0};   // round-robin coordinate sweep pointer
    int       polish_cooldown_{0};    // iterations to skip before reconsidering
    int       polish_backoff_{0};     // consecutive unproductive bursts (drives exponential cooldown)
    long long polish_used_{0};        // cumulative evals spent on polish, whole run

    // ------------------------------------------------------------
    // Best-Anchored Levy Jump (NEW: replaces RTR)
    // ------------------------------------------------------------
    double levy_beta_{1.5};           // stability index (1.5 = standard Cuckoo Search value)
    double levy_prob_{0.30};          // FULL fire probability once fully ramped
    double levy_scale_{0.05};         // jump scale, fraction of box range
    int    levy_ramp_{8};             // no_improve_ iterations to reach full levy_prob_ (0 = always full)
    int    enable_levy_jump_{1};      // master switch (cfg: enable_levy_jump)

    // In-run local (as in DE/PSO)
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    // Final polishing (in end) from [global]
    bool        end_local_refine_ = false;
    std::string end_local_method_ = "";
};

} // namespace optimsolution
