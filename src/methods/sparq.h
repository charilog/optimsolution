#pragma once
#include "optimizer.h"
#include "init.h"
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <utility>

namespace optimsolution {

// ============================================================================
// SPARQ -- rebuilt directly from the CURRENT production arq3 (including the
// elite-polish / rejuvenation / CR-sorting upgrades), so it starts from
// exact parity with arq3 rather than an outdated pre-upgrade snapshot.
//
// Core additions over ARQ2:
//  (1) NLPSR : non-linear population size reduction (LSHADE/NL-SHADE style)
//  (2) SHADE : circular memory of size H for (muF, muCR) with jSO terminal slot
//  (3) RSP   : rank-based selection of r1/r2 with time-varying pressure
//  (4) jSO   : pbest schedule, CR floor, F cap by progress; K(F) time-varying
//  (5) EIG   : eigen-coordinate binomial crossover (rotation invariance)
//  (6) TS    : Thompson sampling bandit for strategy selection (ARQ vs IDE)
//  (7) LEVY  : Levy-flight (Mantegna) perturbations in quarantine
//  (8) OBL   : on-demand opposition-based basin escape when variance collapses
//  (9) Adaptive rtr_pool / agent_fraction scaling with current N
// ============================================================================

class SPARQ : public Optimizer {
public:
    SPARQ() = default;
    ~SPARQ() override = default;

    std::string methodShortName() const override { return "sparq"; }
    std::string methodFullName()  const override {
        return "SPARQ: NLPSR + SHADE-H + LSHADE-RSP + jSO + Eigen-crossover + "
               "Thompson bandit + Levy quarantine + OBL basin-escape + "
               "stagnation-gated elite polish (incl. Solis-Wets adaptive "
               "momentum search) + hard-stagnation rejuvenation";
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override {}

private:
    using Vec = std::vector<double>;
    using Mat = std::vector<std::vector<double>>;

    // ------------------------------------------------------------
    // Population state
    // ------------------------------------------------------------
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    A_;          // external archive (JADE-style)

    // per-individual F/CR for IDE strategy (self-adaptive)
    std::vector<double> CBF_;
    std::vector<double> CBCR_;

    // ------------------------------------------------------------
    // NLPSR: non-linear population size reduction
    // ------------------------------------------------------------
    int    Ninit_{0};
    int    Nmin_{4};
    double nlpsr_alpha_{0.5};        // 1.0 = linear, <1 = slower shrink early
    int    pop_scale_{18};           // Ninit = pop_scale * D if not given

    // ------------------------------------------------------------
    // SHADE: circular memory for F/CR adaptation
    // ------------------------------------------------------------
    int    H_{6};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int    memK_{0};
    double MF_terminal_{0.9};        // jSO fixed terminal slot value
    double MCR_terminal_{0.9};

    // ARQ sampling bounds
    double Flo_{0.05};
    double Fhi_{1.40};

    // ------------------------------------------------------------
    // jSO-style schedules
    // ------------------------------------------------------------
    double pbest_max_{0.25};
    double pbest_min_{0.10};

    // ------------------------------------------------------------
    // LSHADE-RSP: rank-based selective pressure
    // ------------------------------------------------------------
    double kr_init_{2.0};            // selective pressure at start
    double kr_final_{3.0};           // and at end of budget

    // ------------------------------------------------------------
    // Eigen-coordinate crossover (EA4Eig / CoBiDE style)
    // ------------------------------------------------------------
    double p_eig_{0.40};
    int    eig_period_{10};
    double eig_frac_{0.50};
    Mat    B_rot_;                   // D x D eigenvectors (columns)
    Vec    eig_scale_;               // per-axis sqrt(eigenvalue) scales in [0.05,1]
    bool   eig_valid_{false};
    int    iters_since_eig_{0};
    int    eig_min_D_{2};

    // ------------------------------------------------------------
    // RTR (Restricted Tournament Replacement)
    // ------------------------------------------------------------
    int    rtr_pool_{14};
    double rtr_pool_frac_{0.10};     // dynamic scaling cap

    // ------------------------------------------------------------
    // Archive
    // ------------------------------------------------------------
    double archive_rate_{1.5};

    // ------------------------------------------------------------
    // Thompson sampling bandit (replaces EA4Eig roulette)
    // ------------------------------------------------------------
    int                 h_{2};       // two heuristics: [0]=ARQ, [1]=IDE
    std::vector<double> bandit_a_;
    std::vector<double> bandit_b_;
    double              bandit_decay_{0.97};  // non-stationary forgetting
    int                 bootstrap_arq_iters_{2};
    int                 bootstrap_left_{0};

    // ------------------------------------------------------------
    // Quarantine (Levy flight Mantegna)
    // ------------------------------------------------------------
    double outlier_alpha_{1.0};
    double outlier_rho_{0.08};
    double levy_beta_{1.5};
    double qscale_{0.10};

    // ------------------------------------------------------------
    // On-demand OBL basin escape
    // ------------------------------------------------------------
    int    stag_trigger_{30};
    double var_collapse_ratio_{1e-3};
    int    obl_cooldown_{0};
    int    obl_cooldown_init_{80};
    double obl_frac_{0.30};
    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // ------------------------------------------------------------
    // agent_fraction (how many parents to try per ARQ pass)
    // ------------------------------------------------------------
    double agent_fraction_{1.0};     // default: evaluate all parents

    // ------------------------------------------------------------
    // Exploitation-tail progress thresholds, now configurable (were
    // hardcoded 0.75/0.80/0.90). These are the mechanisms behind the
    // "sudden late improvement" pattern: polish_progress_trig_ drops the
    // polish trigger to 1 iteration of stagnation (from polish_trigger_)
    // past this fraction of the budget; polish_progress_burst_ scales up
    // each polish burst's probe count past this fraction (needed for
    // D-coverage in high dimension); rejuv_progress_cutoff_ disables the
    // hard-stagnation restart past this fraction (too disruptive that late).
    // Moving polish_progress_trig_/polish_progress_burst_ EARLIER gives the
    // intensification phase more total budget to work with, but also risks
    // locking onto a basin before the core has finished exploring for a
    // better one — the classic explore/exploit trade-off. Worth sweeping
    // empirically (e.g. 0.6-0.7) rather than assuming lower is better.
    // ------------------------------------------------------------
    double polish_progress_trig_{0.75};
    double polish_progress_burst_{0.80};
    double rejuv_progress_cutoff_{0.90};

    // ------------------------------------------------------------
    // INNOVATION: D-dependent agentfraction scaling.
    //
    // Motivation: `agentfraction` (evals per DE generation = agentfraction*N)
    // is the single most effective lever found for the giant-D, tightly
    // penalty-coupled dispatch/storage family (ded1/ded2/eld5/hydrothermal,
    // D=120-216) — reducing it from 1.0 to 0.5 lets those problems run
    // roughly twice as many generations in the same evaluation budget,
    // which measurably helped several moderate-D problems too (tersoffc,
    // potential, polyphase, weatherirrigation, test2n's reliability).
    // But it is a single GLOBAL constant, and small-D, densely multimodal
    // problems (gallagher21 at D=16 lost ~63% on Mean) don't need — and
    // actively suffer from — fewer individuals being tried per generation:
    // they already get thousands of generations even at agentfraction=1.0,
    // so shrinking the per-generation sample there only reduces the DE
    // core's own statistical power for no benefit.
    //
    // Fix: the CONFIGURED agentfraction becomes the value used once D
    // reaches agentfractionDthreshold (default 100, itself configurable);
    // below that, it is linearly interpolated back toward 1.0 as D shrinks
    // toward 0. This is a direct generalization of the existing, already-
    // validated knob — not a new competing mechanism — so problems at or
    // above the threshold (the giant-D family this was tuned for) see
    // EXACTLY the same behaviour as before, while small-D problems
    // automatically recover full per-generation sampling.
    // ------------------------------------------------------------
    double agent_fraction_Dthreshold_{100.0};
    double effectiveAgentFraction(int D) const {
        if (D >= agent_fraction_Dthreshold_ || agent_fraction_Dthreshold_ <= 0.0)
            return agent_fraction_;
        const double t = (double)D / agent_fraction_Dthreshold_;   // 0..1
        return 1.0 - t * (1.0 - agent_fraction_);                  // 1.0 at D=0
    }

    // ------------------------------------------------------------
    // UPGRADE: NL-SHADE-RSP CR sorting (smaller CR to better-ranked parents)
    // ------------------------------------------------------------
    int    cr_sort_{1};

    // ------------------------------------------------------------
    // UPGRADE: stagnation-gated elite (1+1)-ES polish with 1/5 success rule
    // ------------------------------------------------------------
    int    polish_trigger_{5};       // no_improve_ iterations before polishing
    double polish_frac_{0.10};       // evals per activation = frac * N
    double ps_sigma_{0.02};          // adaptive step (relative to box range)
    double ps_sigma_c_{0.10};        // adaptive step for single-coordinate mode
    double ps_sigma_min_{1e-9};
    double ps_sigma_max_{0.25};
    int    polish_cooldown_{0};      // exponential backoff when polish keeps failing
    int    polish_backoff_{0};
    long long polish_used_{0};       // total evals consumed by the polish
    double polish_budget_{0.12};     // hard cap: fraction of consumed budget
    int    polish_low_streak_{0};    // consecutive low-value activations
    bool   polish_disabled_{false};  // one-way landscape self-selection
    double polish_min_relgain_{1e-3};
    int    polish_coord_ptr_{0};     // round-robin coordinate sweep pointer
    double polish_mark_f_{std::numeric_limits<double>::infinity()};
    long long polish_mark_calls_{0};

    // ------------------------------------------------------------
    // INNOVATION: Solis-Wets adaptive random search with directional bias
    // and reflection (Solis & Wets, 1981) — replaces the memoryless
    // isotropic probe mode inside elitePolish(). Unlike the CMA-ES arm that
    // was tried and removed twice, this NEVER combines information from two
    // DIFFERENT individuals (no weighted-mean recombination across the
    // population) — it only ever evolves a single running trajectory
    // starting from the current incumbent best_x_, so it cannot produce the
    // structurally-incoherent "averaged" points that were catastrophic on
    // permutation-symmetric cluster problems (potential, tersoff family).
    // It is a pure, monotonic refinement of best_x_ that plugs into the
    // SAME injection-only-if-improves pathway elitePolish already uses, so
    // it inherits all of that mechanism's existing safety guarantees.
    //
    // Mechanism: maintain a bias vector b (direction memory). Each attempt
    // draws a random step delta and tries x + b + delta; if that fails, it
    // tries the REFLECTED step x - b - delta (reusing the same random draw,
    // no extra sampling cost beyond one more evaluation) before giving up.
    // A run of successes lets b accumulate into a genuine "momentum" that
    // lets the search race down a productive valley far faster than blind
    // isotropic resampling (helps Best); a run of failures decays b toward
    // zero and contracts the step (helps Mean/reliability — the classic
    // Solis-Wets convergence guarantee against wasted wandering).
    // ------------------------------------------------------------
    Vec    sw_bias_;
    double sw_rho_{0.02};

    // ------------------------------------------------------------
    // INNOVATION: "Trajectory Echo" — a lightweight memory of the DE core's
    // OWN recent successful step directions, replayed (as random linear
    // combinations) inside elitePolish() to bias local refinement toward
    // directions that have proven productive for THIS run.
    //
    // Motivation: ded1/ded2/eld5/hydrothermal (large, tightly quadratic-
    // penalty-coupled dispatch/storage problems) proved completely
    // unresponsive to every population/budget-scheduling tweak tried
    // (agentfraction sweeps from 1.0 down to 0.25 — 39 to 278 generations —
    // changed nothing). The working hypothesis: their per-time-step
    // near-equality power-balance penalty carves a narrow ridge in the
    // landscape — an improving move usually has to shift several
    // coordinates together in a compensating way — and blind isotropic /
    // single-coordinate probing keeps missing that ridge regardless of how
    // many attempts it gets, because it has no way to know which combined
    // direction the ridge runs along.
    //
    // Design (deliberately NOT CMA-ES-shaped, to avoid the two failure modes
    // that made that experiment unsafe): this NEVER combines information
    // from two DIFFERENT population individuals (no weighted-mean
    // recombination — the mechanism that produced structurally-incoherent
    // points on permutation-symmetric cluster problems), and it is not a
    // separate bandit arm competing for iterations (the mechanism that
    // proved fragile in the real batch twice). It is purely an alternative
    // PROBE-DIRECTION SOURCE inside the already-proven elitePolish() probe
    // burst, subject to the exact same "evaluate, inject only if it beats
    // best_f_" safety net every other probe mode already uses.
    //
    // Mechanics: whenever the DE core (stepARQ's selectionRTR, or stepIDE's
    // acceptance loop) replaces an individual with a strictly better trial,
    // the ACCEPTED DISPLACEMENT (trial - old individual) is pushed into a
    // small fixed-size ring buffer — a live, cheap record of "directions
    // that recently paid off," entirely different in kind from the
    // covariance/eigenbasis machinery elsewhere in this file (O(1) update,
    // no matrix, no periodic recomputation). elitePolish() can then draw a
    // probe step as a random linear combination of the buffered directions
    // — literally replaying an echo of the ridge the DE core has already
    // been walking — normalized so the combined step's expected magnitude
    // stays comparable to a single one of its ingredients.
    // ------------------------------------------------------------
    static constexpr int kEchoCapacity = 8;
    std::vector<Vec> echo_steps_;
    int    echo_ptr_{0};
    int    echo_count_{0};
    double echo_scale_{1.0};   // adaptive multiplier, same 1.5x/0.87x rule as ps_sigma_
    // Circuit breaker: on some landscapes (proven via faithful proxy tests
    // against real problem source — test2n, potential) replaying combined
    // historical directions is actively harmful, not just unhelpful. Same
    // pattern used for the CMA-ES arm: require it to prove itself against
    // its own recent track record, and stand down for good if it can't.
    int    echo_fail_streak_{0};
    bool   echo_disabled_{false};

    void recordEchoStep(const Vec& from, const Vec& to);

    // FIX (large-N starvation): no_improve_ increments only when best_f_ does
    // NOT improve at ALL (even by 1e-18). With population scaling as
    // pop_scale_*D, high-D problems run huge populations (e.g. N=2160 at
    // D=120) and therefore only a few dozen total generations fit in the
    // evaluation budget; some individual improves the incumbent by a
    // vanishing amount almost every single generation, so no_improve_
    // essentially never reaches polish_trigger_ and elite polish is starved
    // exactly on the large-N/high-D family (hydrothermal, ded*, eld4/5)
    // where its coordinate-refinement was proven to help (Lennard-Jones
    // proxy testing). This is a SEPARATE counter, gating ONLY polish's
    // trigger below — no_improve_ itself (used by OBL and rejuvenate) is
    // untouched, so this cannot affect either of those mechanisms.
    double polish_stag_mark_f_{std::numeric_limits<double>::infinity()};
    int    polish_stag_count_{0};
    static constexpr double kPolishStagRelGain = 1e-6;

    // ------------------------------------------------------------
    // UPGRADE: hard-stagnation rejuvenation (partial restart)
    // ------------------------------------------------------------
    int    rejuv_factor_{4};         // trigger at rejuv_factor_ * stag_trigger_
    double rejuv_keep_{0.25};        // elite fraction preserved
    int    rejuv_cooldown_init_{150};
    int    rejuv_cooldown_{0};

    // FIX (escalation): on deceptive-basin problems (fmsynth-type) the weak
    // survival reseed fires repeatedly (every ~20 iterations, hundreds of
    // times over a run) because each shallow 5% refresh briefly revives the
    // population's spread — which prevents no_improve_ from ever building up
    // the 120+ consecutive stagnant iterations hard_stag requires, so the
    // strong 75% escape is starved even though the population is genuinely,
    // durably trapped. This tracks consecutive weak firings without a
    // MEANINGFUL improvement and forces one strong escape once the streak
    // gets long — independent of the raw no_improve_ timing.
    double rejuv_watch_f_{std::numeric_limits<double>::infinity()};
    int    rejuv_weak_streak_{0};
    int    rejuv_weak_streak_limit_{8};
    int    rejuv_strong_fail_streak_{0};
    int    rejuv_strong_fail_limit_{2};
    double rejuv_strong_watch_f_{std::numeric_limits<double>::infinity()};

    // ------------------------------------------------------------
    // IDE scheduling (from EA4Eig IDE strategy)
    // ------------------------------------------------------------
    int    gmax_{0};
    double T_{0.0};
    int    g_{0};
    int    gt_{0};
    int    Tcurr_{0};
    int    ide_progress_sync_{1};
    int    ide_strict_improve_{1};

    // debug
    int    debug_{0};

    // ------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------
    inline double eval(const Vec& v) {
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    // utilities
    void   ensureBounds(Vec& v);
    int    pickDistinct(int n, int a = -1, int b = -1, int c = -1);
    int    randInt(int lo, int hi);
    double randU();
    double cauchy(double loc, double scale);
    double gaussN(double mu, double sig);
    double sampleLevy();                  // Mantegna 1994
    double progress01() const;

    // sorting / population management
    void   sortByFitness();
    int    targetPopulationSize() const;
    void   shrinkTo(int N);

    // rank-based r-index picking (ord is sorted-by-fitness order)
    double currentKR() const;
    double currentPbest() const;
    int    rankBasedPick(const std::vector<int>& ord, int forbid) const;

    // SHADE memory
    void   initMemory();
    void   sampleFCR(double& F, double& CR);
    void   updateMemoryFromSuccess(const std::vector<double>& SF,
                                   const std::vector<double>& SCR,
                                   const std::vector<double>& SG);

    // archive
    void   archivePush(const Vec& x);
    void   archiveTrim(int N);

    // eigen machinery
    void   jacobiEigen(const Mat& Ain, Mat& V, std::vector<double>& w) const;
    void   recomputeEigenBasis();
    void   applyBt(const Mat& B, const Vec& x, Vec& out) const;   // out = B^T x
    void   applyB (const Mat& B, const Vec& x, Vec& out) const;   // out = B   x
    void   eigenBinomialCrossover(int D, const Vec& base, const Vec& v,
                                  double CR, Vec& u);

    // Thompson bandit
    int    thompsonPick();
    void   banditDecay();
    void   banditRecord(int k, int successes, int attempts);

    // RTR selection
    double distBN(const Vec& a, const Vec& b) const;
    bool   selectionRTR(int parentIndex, const Vec& u, double fu,
                        double F, double CR,
                        std::vector<double>& SF,
                        std::vector<double>& SCR,
                        std::vector<double>& SG);

    // ARQ/IDE trial construction
    double computeK(double F) const;
    void   makeTrialARQ(int i, const std::vector<int>& ord,
                        double F, double CR, Vec& u);
    void   stepARQ();
    void   stepIDE();

    // quarantine and restart
    static double quantile(std::vector<double> v, double q01);
    void   quarantineLevy();
    void   oblBasinEscape();
    void   elitePolish();            // UPGRADE (B)
    void   rejuvenate();             // UPGRADE (C)

    // IDE parameter re-seed / inherit
    void   sampleIDEParamsAt(int idx);
    void   inheritIDEParams(int dst, int src);

    // Population std-dev relative to box size (for OBL trigger)
    double normalizedPopSpread() const;
};

} // namespace optimsolution
