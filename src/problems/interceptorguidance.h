// ============================================================================
//  interceptorguidance.h
//  Adaptive Divert-Budget Interceptor Guidance under Deceptive Threat
//  Maneuvering (ADIG) Benchmark — v4 (fixed global optimum + tunable
//  dimension-gated difficulty onset)
//
//  A SURROGATE optimization benchmark inspired by terminal-phase
//  interceptor guidance. This is NOT a physics-accurate, classified, or
//  deployable guidance model -- see the design note below for what
//  actually changed and why, and interceptorguidance.cpp's simulate() for
//  the exact math.
//
//  ============================================================================
//  WHY v4 EXISTS (design history, most recent problem first)
//  ============================================================================
//  v1/v2/v3 built the cost around a physically-flavoured ZEM (zero-effort-
//  miss) simulation: the achievable minimum was "whatever the dynamics
//  happen to produce", which is fine for realism but broke two things a
//  benchmark actually needs:
//    (1) Turning M1/M2/M3 on or off changed the ACHIEVABLE MINIMUM itself
//        (different weighted terms were added to the same cost), not just
//        the difficulty of finding it -- comparisons across mechanism
//        settings, or interpreting "success rate", were not meaningful.
//    (2) Even after v3's baseline non-separability + ruggedness fix,
//        empirically the solvers only started struggling around D=512 --
//        far later than intended, with no direct way to dial in exactly
//        where the onset should be.
//
//  v4 restructures the benchmark around a REFERENCE-TRACKING design, the
//  same pattern used by CEC-style shifted/rotated benchmark suites: define
//  a known reference guidance schedule r(tau) (a fixed, deterministic
//  function of engagement-progress tau in [0,1], NOT of x), work entirely
//  in the DEVIATION e_i = x_i - r(tau_i), and build every cost term so it
//  is EXACTLY zero at e == 0 (i.e. x_i == r(tau_i) for every stage) in
//  EVERY dimension and EVERY M1/M2/M3 setting: mechanisms and dimension
//  only change how deep/wide/coupled the terrain is AROUND that point, not
//  where the bottom is or how low it goes. See simulate()'s comments for
//  the (short) positive-definiteness argument that makes this provable
//  rather than just "usually true in practice". refresh_metadata() now
//  registers setKnownGlobalOptimum(0.0), which is exact for every D and
//  every mechanism combination.
//
//  Difficulty no longer emerges implicitly from combinatorics alone --
//  it's explicitly dialled in with two smooth logistic "severity" gates,
//  keyed to dimension D (see sigmoid_severity() in the .cpp):
//    - baseline_severity(D), pivot D0_base_ (default 60): scales the
//      ALWAYS-ON non-separable coupling + multimodal ripple terms, so the
//      landscape stays close to a simple convex bowl for D well below 60
//      and gets progressively rugged/coupled for D above it -- with EVERY
//      M-flag off.
//    - mechanism_severity(D), pivot D0_mech_ (default 20): scales the
//      EXTRA per-mechanism terms (only added when that mechanism's flag is
//      on), so enabling M1/M2/M3 shifts the SAME kind of difficulty onset
//      much earlier (around D=20 by default).
//  Both pivots and both transition widths are cfg-tunable
//  ([interceptorguidance] D0_base / width_base / D0_mech / width_mech) so
//  the onset dimensions can be recalibrated from empirical batch runs
//  without touching the code.
//
//  Three toggleable AMPLIFIER mechanisms (each adds an EXTRA term on top
//  of the always-on baseline coupling/ruggedness, gated by
//  mechanism_severity(D), and each is exactly zero at e == 0):
//      M1 — extra sequential-coupling penalty (deviations must be
//           corrected in a coordinated, sequence-aware way, not stage by
//           stage independently)
//      M2 — extra, higher-frequency ripple (denser local optima around
//           the reference schedule)
//      M3 — a finite divert-fuel budget that only depletes when x departs
//           from the reference schedule, plus pacing-corridor checkpoints
//           (entirely absent, not just small, when M3 is off)
//
//  The only thing this file actually contributes is an OPTIMIZATION
//  BENCHMARK: given a per-stage "guidance schedule" x in [0,1]^D, find the
//  (known, fixed) reference schedule r(tau) stage by stage, with the
//  search made harder by dimension and by which mechanisms are enabled.
// ============================================================================

#pragma once
#include "problem.h"
#include <string>
#include <vector>

namespace optimsolution {

class InterceptorGuidance : public Problem {
public:
    InterceptorGuidance();
    void init(int dim) override;

    int numObjectives() const override { return 2; }
    Vec evaluateMultiCore(const Vec& x) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    struct MechanismConfig {
        bool M1_enable;
        bool M2_enable;
        bool M3_enable;
    };

    // Every field below is already fully weighted (including its severity
    // gate, if any) by the time simulate() returns it -- evaluate_with_point()
    // and evaluateMultiCore() just sum/split these, applying no further
    // weighting except term2 (see .cpp).
    struct Outcome {
        double term2;          // RAW squared final-stage deviation (weight applied once, outside)
        double core_cost;
        double coupling_cost;  // baseline (always) + M1 extra, combined
        double rugged_cost;    // baseline (always) + M2 extra, combined
        double fuel_cost;      // always-on, purely deviation-driven effort
        double overrun_cost;   // M3 only
        double corridor_cost;  // M3 only
    };

    // ── Reference schedule + per-stage weighting (precomputed at init) ──────
    std::vector<double> reference_;    // r(tau_i) in [0,1]: the KNOWN, fixed optimum
    std::vector<double> criticality_;  // per-stage weight on tracking cost

    // ── Reference schedule shape ─────────────────────────────────────────
    double ref_center_, ref_amp_, ref_freq_, ref_phase_;

    // ── Always-on core (defines the strictly-convex "pull to truth"; NOT
    //    severity-gated -- this term alone already guarantees e==0 is the
    //    unique minimum no matter what else is added) ───────────────────
    double core_weight_;
    double terminal_weight_;          // extra weight on the FINAL stage's deviation

    // ── Always-on baseline coupling + ruggedness (severity-gated by
    //    baseline_severity(D), pivot D0_base_) ──────────────────────────
    double base_couple_weight_, base_couple_rho_;
    double base_rugged_weight_, base_rugged_freq_;

    // ── M1 — extra sequential-coupling penalty (mechanism_severity(D)) ──
    double m1_extra_couple_weight_, m1_extra_couple_rho_, m1_extra_couple_freq_;
    double m1_extra_terminal_weight_;

    // ── M2 — extra, denser ripple (mechanism_severity(D)) ───────────────
    double m2_extra_rugged_weight_, m2_extra_rugged_freq_;

    // ── Always-on effort / control cost (purely a function of the
    //    deviation; zero at e==0 regardless of D or any M-flag) ─────────
    double base_fuel_weight_;
    double fuel_burn_gain_, fuel_burn_quad_;
    double fuel_burn_ripple_weight_, fuel_burn_ripple_freq_;

    // ── M3 — divert-fuel budget + deceptive pacing corridor
    //    (mechanism_severity(D); entirely absent, not just small, when
    //    M3 is off) ────────────────────────────────────────────────────
    double fuel_init_;
    double overrun_weight_;
    double corridor_weight_;
    double corridor_slack_;

    // ── Difficulty-onset gates (dimension-keyed logistic severity) ──────
    double D0_base_, width_base_;   // baseline onset pivot / transition width (default: 60, 12)
    double D0_mech_, width_mech_;   // mechanism onset pivot / transition width (default: 20, 6)

    MechanismConfig config_;
    bool             config_loaded_;
    std::string      loaded_config_path_;

    Outcome simulate(const Vec& x) const;
    double  evaluate_with_point(const Vec& x) const;
    double  safe_x(const Vec& x, int i) const;
    static double sigmoid_severity(double D, double D0, double width);
    void    load_config();
    void    load_main_config_overrides();
    void    refresh_metadata();

    static std::string trim_copy(const std::string& s);
    static std::string to_lower_copy(const std::string& s);
    static bool        parse_bool_value(const std::string& value, bool default_value);
};

} // namespace optimsolution
