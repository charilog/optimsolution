#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// UDE (v6) -- Differential Evolution with 1/5-rule-regulated age-based
// turnover, age-guided donors, a success-rate-coupled F, a Cr memory with a
// success-quantile entry, and a self-triggered switch to lineage mode when the
// turnover collapses the population too early.
//
// Origin. UDE is derived from RDE (S. Tao et al., arXiv:2404.16280, CEC 2024),
// using the authors' code (github.com/SichenTao/RDE, v1.1.0) as reference.
// An ablation of that code (CEC 2017, D = 30, 14 functions, 15 runs per
// variant, one mechanism removed at a time) sorted RDE's mechanisms into:
//   category 1, clearly important -> KEPT:
//     - linear population size reduction, N: 18*D -> 4 (L-SHADE);
//     - success-history memory (H = 5 entries): for Cr always (start 0.8,
//       Cr ~ N(M_Cr, 0.1) clipped to [0,1], weighted Lehmer mean with weights
//       |f_parent - f_trial|, M_Cr <- (Lehmer + old)/2, terminal M_Cr = -1 ->
//       Cr = 0); for F only in lineage mode (mechanism 4; start 0.3, F ~
//       Cauchy(M_F, 0.1) redrawn while < 0 and clipped at 1, M_F <- Lehmer).
//       In turnover mode F follows the success rate instead (see "F" below);
//       this change was made with the user's agreement;
//   category 2, small mixed effect -> REPLACED (jSO stage limits, the fixed
//     extra memory entry F = Cr = 0.9; see mechanism 1);
//   category 3, no measurable effect when removed one at a time -> REMOVED
//     (EB branch, external archive, RSP, linear decrease of p, Cauchy
//     perturbation); on all 29 functions they turned out to matter together
//     at D >= 30, and their role is taken over by mechanisms 2 and 3.
// Also kept: current-to-pbest/1 with the pbest pool = the best p*N (p = 0.25),
// binomial crossover, midpoint bound repair, greedy (<=) test of every trial
// against its own target.
//
// Mechanism 1 -- success-quantile entry. The extra (H+1-th) memory choice,
// drawn with probability 1/(H+1), holds Cr_q (and, for lineage mode, F_q), the
// improvement-weighted q-quantiles (q = 0.95) of the values of the successful
// trials, updated as X_q <- (X_q + Q_q(S_X)) / 2. The regular entries follow
// the centre of what succeeds, this one its upper edge; no schedule, no fixed
// values.
//
// Mechanism 2 -- 1/5-rule-regulated age-based turnover (new in v4). A share
// pi in [0, 1] is regulated by Rechenberg's 1/5 success rule,
//     pi <- clip( pi + g * (1/5 - SR) / (1/5), 0, 1 ),   g = turnover_gain,
// SR = success rate of the last generation, pi starts at 0. Each successful
// trial then becomes, with probability pi, a NEW member: its parent is kept
// and the trial takes the slot of the stalest unsuccessful member (largest
// age, see mechanism 3; ties: the worse one). With probability 1 - pi it
// replaces its parent as in classic DE. So while more than 1/5 of the trials
// succeed (descent, final refinement) UDE works lineage by lineage; when
// fewer succeed (plateaus, rugged multimodal landscapes) successful lineages
// multiply and stuck ones give up their places.
// Diagnosis that led to it: the user's runs and ours showed RDEx (CEC 2025
// winner, built on LSRTDE) far ahead of UDE v3 at D = 30 and 50, mostly on
// functions with Schwefel, Rastrigin and Lunacek parts (F10, F11-F14, F16,
// F17, F20, F22, F26). Removing RDEx's mechanisms one at a time left only two
// that matter, and only together: its F rule (below) and its population
// renewal, where every success is written into the oldest slot of the
// population and its parent survives. Under one-to-one replacement a member
// that sits in an old, mediocre basin of such a landscape stays there for
// ever and keeps feeding large, failing differences; renewal removes it.
// The 1/5-rule share and the "stalest member" criterion are UDE's own:
// on our screens the cyclic (oldest slot) renewal and the always-on variant
// were both worse. In turnover mode the trail of v2-v3 is not used (adding it
// back improved nothing); the 1/5 rule regulates the turnover instead. The
// trail returns only in lineage mode (mechanism 4).
//
// Mechanism 3 -- age-guided donors (as in v3). Every member i carries an age
// a_i, the number of generations since its last success (0 = it succeeded,
// or was born, in the last generation). Members are ranked from the freshest
// to the stalest, ties broken by fitness rank, and linear ranking weights
// (N - k for freshness rank k) are used three times: (a) pbest is drawn from
// the best p*N with weights on the freshness rank inside that pool; (b) r1 is
// drawn from the population with weights on the freshness rank, r2 uniformly;
// (c) if r2 is fresher than r1 they swap, so x_r1 - x_r2 points from the
// staler member to the fresher one. Related work: stagnation counters in DE
// mostly trigger replacement or perturbation of stagnant members (TVRS, BDVS,
// ESM, GAR-DE); ETI-DE ranks members by stagnation count, RUSDE draws donors
// from an archive of members whose rank went up. Using the freshness rank to
// choose pbest and r1 and to orient the difference is, as far as we could
// find, new.
//
// Mechanism 4 -- switch to lineage mode on premature collapse (new in v5).
// Once, when a quarter of the budget is used (the first generations with
// progress in [0.25, 0.27]), UDE measures how fast the population spread s
// (root of the mean per-dimension variance, relative to the box) shrank over
// the last 10% of the budget, rate = d ln s / d(budget). If
//     rate < -switch_rate (4: s shrinks by e in less than 1/4 of the budget)
//     and the turnover share pi >= switch_share (0.5: turnover is engaged),
// the turnover is collapsing the population long before the budget ends, and
// UDE works in LINEAGE MODE for the rest of the run: every success replaces
// its parent, F comes from the SHADE memory (Cauchy around M_F or F_q), and
// the 1/5 share pi decides how often r2 is drawn from a trail of the parents
// replaced in lineage mode (the mechanism of v2-v3). Otherwise nothing changes.
// The second condition separates a collapse inside the turnover regime from a
// fast but genuine descent (many successes, pi near 0).
// Refinement (v6): the contraction must be fast over the whole window AND
// over its second half (the collapse is still going on, not slowing down),
// and the turnover share must be >= switch_share now AND on average over the
// window (the collapse was driven by the turnover). In v5 the switch also
// fired at D = 30 F18 / F15 (contraction mostly while pi was low, window mean
// ~0.2-0.35) and at D = 50 F11 (contraction already slowing, second-half rate
// ~ -1.9), which cost D30 F18 and D50 F11; with the refinement it no longer
// fires there, while the D = 10 cases that gain keep switching.
// Diagnosis that led to it: at D = 10 v4 converged at 40-60% of the budget
// into local optima right next to the global one (F19: 0.0194 in every run,
// F20: 0.31/0.62, F17: 0.3-1.3) and wasted the rest; v3 kept the population
// open with large F and reached the global optimum. Restarts converged back
// into the same optima. Running v4 for the first 30% and v3-style lineage
// mode afterwards fixed D = 10 but hurt D = 30 badly, so the switch must be
// triggered by the run itself. Signals that did not separate the cases:
// success rate and turnover share in mid-run, spread level, a deadline
// (contraction needed to reach a final precision), a projection of the
// collapse time from the start, and continuous checking (it fires at D = 30
// too, because every run contracts fast in its first phase). At the 25-30%
// point the contraction rate separates them: about -4 to -18 per unit budget
// for the D = 10 cases that benefit, -0.4 to -3.9 for the D = 30 cases that
// would be harmed. Decision at 20% or 35%, or threshold 3, were worse on our
// screens; thresholds 4-5 and decision at 25-30% behaved alike.
//
// F -- success-rate-coupled in turnover mode (NOT new; LSRTDE, Stanovov & Semenkin, CEC 2024,
// as used in RDEx): F ~ N(0.4 + 0.25 tanh(5 SR), 0.02), redrawn into [0,1],
// SR = success rate of the previous generation (0.5 before the first). With
// turnover, the improvement-weighted F memory of SHADE collapses the
// population (D = 30 Rastrigin converged at half of the budget); what is
// needed is negative feedback -- larger F when many trials succeed, smaller
// when few do. Six original controllers were tried and all were clearly
// worse (success-balance of bold vs cautious trials, quantile tracking of the
// successful F, a variance-balance model, F coupled to the target's age in
// two forms, a 1/5-rule switch); the exact shape of this rule matters (a
// constant F, a linear or a square-root map all failed). The constants are
// exposed as f_base, f_range, f_slope, f_sigma.
//
// optimsolution additions kept from rde.h: "max_evals = N x D" support (in the
// base class), the optional min_generations cap on Nmax, in-run / end-of-run
// local-search hooks.
class UDE : public Optimizer {
public:
    UDE() = default;
    ~UDE() override = default;
    std::string methodShortName() const override { return "ude"; }
    std::string methodFullName()  const override { return "UDE (DE with 1/5-rule age-based turnover, age-guided donors, success-rate F, quantile entries and a self-triggered lineage mode)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [ude] in the configuration file.
    void configure(const MethodConfig& mc) override {
        int fixedPop = mc.getInt("population", -1);
        if (fixedPop > 3) {
            fixed_population_ = fixedPop;
            this->setPopulation(fixedPop); // so StopController sees a sane size immediately
        }
        Nmax_mult_       = mc.getDbl("nmax_multiplier", Nmax_mult_);
        Nmin_            = mc.getInt("min_population", Nmin_);
        min_generations_ = mc.getInt("min_generations", min_generations_);
        H_               = mc.getInt("memory_size", H_);
        muCR0_           = mc.getDbl("mu_cr0", muCR0_);
        pbest_           = mc.getDbl("pbest", pbest_);
        use_qentry_      = mc.getBool("quantile_entry", use_qentry_);
        q_               = mc.getDbl("quantile", q_);
        use_turnover_    = mc.getBool("turnover", use_turnover_);
        turnover_gain_   = mc.getDbl("turnover_gain", turnover_gain_);
        use_age_         = mc.getBool("age_guidance", use_age_);
        muF0_            = mc.getDbl("mu_f0", muF0_);
        use_switch_      = mc.getBool("lineage_switch", use_switch_);
        switch_at_       = mc.getDbl("switch_at", switch_at_);
        switch_window_   = mc.getDbl("switch_window", switch_window_);
        switch_rate_     = mc.getDbl("switch_rate", switch_rate_);
        switch_share_    = mc.getDbl("switch_share", switch_share_);
        f_base_          = mc.getDbl("f_base", f_base_);
        f_range_         = mc.getDbl("f_range", f_range_);
        f_slope_         = mc.getDbl("f_slope", f_slope_);
        f_sigma_         = mc.getDbl("f_sigma", f_sigma_);
        sr0_             = mc.getDbl("sr0", sr0_);

        if (Nmin_ < 4) Nmin_ = 4;
        if (H_ < 1) H_ = 1;
        pbest_ = std::min(std::max(pbest_, 0.0), 1.0);
        q_     = std::min(std::max(q_, 0.0), 1.0);
        if (turnover_gain_ < 0.0) turnover_gain_ = 0.0;
        if (f_sigma_ < 0.0) f_sigma_ = 0.0;
        sr0_ = std::min(std::max(sr0_, 0.0), 1.0);

        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
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
    double weightedQuantile(const std::vector<double>& v, const std::vector<double>& w) const;
    void   findLimits(Vec& x, const Vec& parent) const;

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // ---- parameters ----
    int    fixed_population_{-1};
    double Nmax_mult_{18.0};
    int    min_generations_{100};
    int    Nmin_{4};
    int    H_{5};
    double muCR0_{0.8};
    double pbest_{0.25};
    bool   use_qentry_{true};    // false: no quantile entry for Cr, for ablations
    double q_{0.95};             // quantile level of the extra Cr entry
    bool   use_turnover_{true};  // false: classic one-to-one replacement, for ablations
    double turnover_gain_{0.3};  // gain g of the 1/5-rule controller of the turnover share
    bool   use_age_{true};       // false: uniform pbest / r1 / r2, for ablations
    double muF0_{0.3};           // initial F memory (lineage mode)
    bool   use_switch_{true};    // false: never switch to lineage mode (UDE v4), for ablations
    double switch_at_{0.25};     // decision when this fraction of the budget is used
    double switch_window_{0.1};  // window (fraction of the budget) for the contraction rate
    double switch_rate_{4.0};    // switch if d ln(spread)/d(budget) < -switch_rate ...
    double switch_share_{0.5};   // ... and the turnover share >= switch_share
    double f_base_{0.4};         // F centre = f_base + f_range * tanh(f_slope * SR)  (LSRTDE)
    double f_range_{0.25};
    double f_slope_{5.0};
    double f_sigma_{0.02};       // F ~ N(centre, f_sigma)
    double sr0_{0.5};            // success rate assumed before the first generation

    // ---- adaptive state ----
    int    Nmax_{0};
    std::vector<double> mem_CR_;
    std::vector<double> mem_F_;  // F memory (lineage mode)
    int    mem_pos_{0};
    double qCr_{0.8};            // success-quantile entry of Cr
    double qF_{0.3};             // success-quantile entry of F (lineage mode)
    bool   lineage_{false};      // switched to lineage mode
    std::vector<Vec> trail_;     // parents replaced in lineage mode
    std::vector<double> histT_, histS_, histP_;  // progress, log spread, turnover share per generation (until the decision)
    double sr_{0.5};             // success rate of the last generation
    double piT_{0.0};            // turnover share (1/5-rule controller)
    std::vector<int> age_;       // generations since each member's last success

    std::string local_method_{"none"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
