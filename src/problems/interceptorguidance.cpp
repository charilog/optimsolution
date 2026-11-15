// ============================================================================
//  interceptorguidance.cpp
//  Adaptive Divert-Budget Interceptor Guidance under Deceptive Threat
//  Maneuvering (ADIG) Benchmark — v5 (fixed mechanism-overlap bugs: M1
//  shortcut, M3 near-optimum blindness) + tunable
//  dimension-gated difficulty onset)
//
//  See interceptorguidance.h for the full design rationale. Summary of the
//  math actually implemented here:
//
//  Let r(tau_i) be the fixed, known reference schedule and e_i = x_i -
//  r(tau_i) the deviation at stage i. Every cost term below is built from
//  e (never from x directly), and every term is EITHER
//    (a) a strictly positive-definite quadratic in e (the "core" term:
//        positive weight * positive per-stage criticality * e_i^2), or
//    (b) a positive-SEMI-definite quadratic in e (the coupling terms:
//        (e_i - rho*e_{i-1})^2 is a perfect square, so >= 0 everywhere and
//        == 0 along the whole e_i = rho*e_{i-1} family, not just at e=0
//        on its own), or
//    (c) non-negative and == 0 exactly at e_i == 0 (the ripple terms:
//        sin^2(freq*PI*e_i), and the fuel-burn terms: gain*e_i^2 +
//        quad*e_i^4).
//  Because (a) is strictly positive everywhere except e==0, and (b)/(c)
//  are merely non-negative (never negative), the TOTAL cost
//      total(e) = (a) + (b) + (c) + ... >= (a)(e) > 0  for any e != 0
//  and total(e) == 0 exactly at e == 0. So e == 0 (i.e. x_i == r(tau_i)
//  for every stage) is the UNIQUE global minimum, with value exactly 0,
//  for ANY weight on the (b)/(c) terms, ANY dimension, and ANY M1/M2/M3
//  setting -- turning a mechanism on/off or changing D only rescales how
//  the terrain looks AROUND that fixed point, never where or how low the
//  true bottom is. That is what setKnownGlobalOptimum(0.0) in
//  refresh_metadata() certifies.
// ============================================================================

#include "interceptorguidance.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace optimsolution {

namespace {
constexpr double PI = 3.14159265358979323846;
inline double sqr(double v)     { return v * v; }
inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
} // anonymous namespace

// ============================================================================
//  Constructor
// ============================================================================
InterceptorGuidance::InterceptorGuidance()
    : Problem(),
      ref_center_(0.0), ref_amp_(0.0), ref_freq_(0.0), ref_phase_(0.0),
      core_weight_(0.0), terminal_weight_(0.0),
      base_couple_weight_(0.0), base_couple_rho_(0.0),
      base_rugged_weight_(0.0), base_rugged_freq_(0.0),
      m1_extra_couple_weight_(0.0), m1_extra_couple_rho_(0.0), m1_extra_couple_freq_(0.0),
      m1_extra_terminal_weight_(0.0),
      m2_extra_rugged_weight_(0.0), m2_extra_rugged_freq_(0.0),
      base_fuel_weight_(0.0), fuel_burn_gain_(0.0), fuel_burn_quad_(0.0),
      fuel_burn_ripple_weight_(0.0), fuel_burn_ripple_freq_(0.0),
      fuel_init_(0.0), overrun_weight_(0.0), corridor_weight_(0.0), corridor_slack_(0.0),
      D0_base_(0.0), width_base_(0.0), D0_mech_(0.0), width_mech_(0.0),
      config_{true, true, true},
      config_loaded_(false),
      loaded_config_path_()
{
    setName("interceptorguidance");
}

// ============================================================================
//  Dimension-keyed logistic "how much should this mechanism bite here"
//  gate. sigmoid_severity(D, D0, width) -> 0 for D well below D0, -> 1 for
//  D well above D0, ~0.5 exactly at D == D0, with a smooth transition of
//  scale `width`. Used to make the difficulty ONSET an explicit, tunable
//  dimension (D0_base_ / D0_mech_) rather than an implicit combinatorial
//  side-effect.
// ============================================================================
double InterceptorGuidance::sigmoid_severity(double D, double D0, double width)
{
    const double w = (width > 1e-9) ? width : 1e-9;
    return 1.0 / (1.0 + std::exp(-(D - D0) / w));
}

// ============================================================================
//  init(dim)
// ============================================================================
void InterceptorGuidance::init(int dim)
{
    if (dim < 1) dim = 1;

    load_config();
    load_main_config_overrides();
    refresh_metadata();

    const double bsev = sigmoid_severity(dim, D0_base_, width_base_);
    const double msev = sigmoid_severity(dim, D0_mech_, width_mech_);
    std::fprintf(stderr,
        "[ADIG v9 LOADED]  cfg=%s\n"
        "  flags:      M1(extra coupling)=%d  M2(extra ripple)=%d  M3(fuel+corridor)=%d\n"
        "  severity:   baseline(D0=%.1f,w=%.1f)=%.4f   mechanism(D0=%.1f,w=%.1f)=%.4f   [at dim=%d]\n"
        "  reference:  center=%.2f amp=%.2f freq=%.2f phase=%.2f\n"
        "  core:       weight=%.2f  terminal_w=%.2f\n"
        "  baseline:   couple_w=%.2f couple_rho=%.2f  rugged_w=%.2f rugged_f=%.2f\n"
        "  M1 extra:   couple_w=%.2f couple_rho=%.2f  terminal_w=%.2f\n"
        "  M2 extra:   rugged_w=%.2f rugged_f=%.2f\n"
        "  fuel:       base_w=%.2f burn_g=%.2f burn_q=%.2f init=%.2f\n"
        "  M3 extra:   overrun_w=%.2f corridor_w=%.2f slack=%.2f\n"
        "  known global optimum = 0.0 (exact, for this dim and these flags)\n"
        "  dim=%d\n",
        loaded_config_path_.empty() ? "(not found)" : loaded_config_path_.c_str(),
        (int)config_.M1_enable, (int)config_.M2_enable, (int)config_.M3_enable,
        D0_base_, width_base_, bsev, D0_mech_, width_mech_, msev, dim,
        ref_center_, ref_amp_, ref_freq_, ref_phase_,
        core_weight_, terminal_weight_,
        base_couple_weight_, base_couple_rho_, base_rugged_weight_, base_rugged_freq_,
        m1_extra_couple_weight_, m1_extra_couple_rho_, m1_extra_terminal_weight_,
        m2_extra_rugged_weight_, m2_extra_rugged_freq_,
        base_fuel_weight_, fuel_burn_gain_, fuel_burn_quad_, fuel_init_,
        overrun_weight_, corridor_weight_, corridor_slack_,
        dim);

    Problem::init(dim);

    Vec l(dim, 0.0), u(dim, 1.0);
    setBounds(l, u);

    reference_.assign(dim, 0.0);
    criticality_.assign(dim, 0.0);

    for (int i = 0; i < dim; ++i) {
        const double tau = static_cast<double>(i + 1) / dim;   // (0, 1]
        reference_[i]   = clamp01(ref_center_ + ref_amp_ * std::sin(2.0 * PI * ref_freq_ * tau + ref_phase_));
        criticality_[i] = 0.3 + 0.9 * tau * tau;
    }
}

// ============================================================================
//  simulate() — see the top-of-file comment for the positive-definiteness
//  argument. One deterministic forward pass over all D stages, shared by
//  evaluate_core() and evaluateMultiCore().
// ============================================================================
InterceptorGuidance::Outcome InterceptorGuidance::simulate(const Vec& x) const
{
    Outcome o{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const int D = dimension();
    const double dt = 1.0 / static_cast<double>(D);

    const double bsev = sigmoid_severity(D, D0_base_, width_base_);
    const double msev = sigmoid_severity(D, D0_mech_, width_mech_);

    // ── Pass 1: raw per-stage deviation from the known reference ───────────
    std::vector<double> e(D);
    for (int i = 0; i < D; ++i) {
        const double xi = clamp01(safe_x(x, i));
        e[i] = xi - reference_[i];
    }

    // ── Pass 2: fixed, x-independent MIXING of the deviations, used by
    //    M1/M2/M3 (coupling, ruggedness, fuel burn) instead of the raw,
    //    axis-aligned e[i]. core_cost and the terminal term below still use
    //    raw e[] directly.
    //
    //    v6 FIX: v5 (zigzag coupling + strict corridor) reduced but did not
    //    eliminate a deeper problem -- SHADE/CMA-ES-family search (jso,
    //    rde, lmcmaes, ...) is specifically good at exploiting ANY smooth,
    //    axis-aligned structure that still points toward the SAME known
    //    optimum, turning "more penalty" into "a clearer gradient" and so
    //    converging FASTER, not slower, even when a term is nominally a
    //    difficulty mechanism (confirmed empirically: jso/rde/lmcmaes
    //    scored consistently *better* with M1/M3 added on top of M2, across
    //    several dimensions at once -- not run-to-run noise). This is the
    //    same reason CEC-style benchmark suites rotate their harder
    //    functions: a fixed rotation/mixing breaks the coordinate-aligned
    //    structure that adaptive algorithms lock onto, without touching
    //    where the true optimum is or how low it goes.
    //
    //    m[i] pairs stage i with a FAR-AWAY stage (i + D/2, wrapping
    //    around) rather than its immediate neighbour, and blends them with
    //    a fixed per-stage angle from a golden-ratio low-discrepancy
    //    sequence (irrational step -> no short repeating pattern for any
    //    D). Being a fixed LINEAR combination of e[], m[i] == 0 for every i
    //    whenever e == 0, so the true optimum and its value (exactly 0)
    //    are completely unaffected; only the terrain away from it changes.
    std::vector<double> m(D);
    {
        const int stride = std::max(1, D / 2);
        for (int i = 0; i < D; ++i) {
            const int j = (i + stride) % D;
            const double theta = 2.0 * PI * std::fmod((i + 1) * 0.6180339887498949, 1.0);
            m[i] = std::cos(theta) * e[i] + std::sin(theta) * e[j];
        }
    }

    double fuel = fuel_init_;
    bool   fuel_exhausted = false;
    double m_prev = m[D - 1];   // wrap: stage 0's "previous" is the last mixed value

    for (int i = 0; i < D; ++i) {
        const double tau = static_cast<double>(i + 1) / D;
        const double ei  = e[i];
        const double mi  = m[i];

        // ── Core tracking cost (ALWAYS on, never severity-gated; RAW e) ────
        // Strictly positive-definite in e: this term alone already makes
        // e==0 the unique minimum, no matter what else below is added.
        o.core_cost += core_weight_ * criticality_[i] * ei * ei * dt;

        // ── Baseline coupling (ALWAYS on, gated by baseline_severity;
        //    operates on the MIXED m, not raw e -- see note above) ────────
        {
            const double c = mi - base_couple_rho_ * m_prev;
            o.coupling_cost += base_couple_weight_ * bsev * c * c * dt;
        }
        // M1: an EXTRA coupling penalty on top, only when M1 is enabled,
        // gated by mechanism_severity instead (earlier onset with D).
        //
        // v8 FIX: v7 tried simply increasing the weight on the same
        // quadratic-in-c penalty, but that measurably made things WORSE
        // (empirically: jso's and rde's "shortcut" advantage over baseline
        // *grew*, not shrank, when the weight went up) -- a smooth,
        // single-valley penalty just gives success-history/covariance-
        // adaptive search (jso, rde, lmcmaes, ...) a CLEARER gradient to
        // lock onto, the stronger it is. M2's ruggedness never had this
        // problem because sin^2 creates many local optima, not one
        // sharpened valley. M1 now borrows that same idea: the penalty is
        // sin^2(freq*c) instead of c^2, so satisfying the coupling
        // relationship has MANY valid local configurations (multimodal in
        // the relative-offset direction) instead of a single, ever-easier-
        // to-follow target. Sign still alternates every stage on top of
        // the mixing, and c==0 at e==0 as before, so sin^2(freq*0)==0 too
        // -- the true optimum's value is completely unaffected.
        if (config_.M1_enable) {
            const double signed_rho = (i % 2 == 0) ? m1_extra_couple_rho_ : -m1_extra_couple_rho_;
            const double c = mi - signed_rho * m_prev;
            o.coupling_cost += m1_extra_couple_weight_ * msev * sqr(std::sin(m1_extra_couple_freq_ * PI * c)) * dt;
        }

        // ── Baseline ruggedness (ALWAYS on, gated by baseline_severity;
        //    operates on the MIXED m) ───────────────────────────────────
        // sin^2(freq*PI*m) has many zeros, but only e==0 (which forces
        // m==0 too, being linear in e) also zeroes the strictly-positive
        // core term above, so e==0 remains the unique GLOBAL minimum even
        // though this term alone is multimodal (see top-of-file proof).
        o.rugged_cost += base_rugged_weight_ * bsev * sqr(std::sin(base_rugged_freq_ * PI * mi)) * dt;
        // M2: an EXTRA, denser ripple on top, only when M2 is enabled.
        if (config_.M2_enable)
            o.rugged_cost += m2_extra_rugged_weight_ * msev * sqr(std::sin(m2_extra_rugged_freq_ * PI * mi)) * dt;

        // ── Effort / control cost (ALWAYS on; purely a function of the
        //    mixed deviation, so it is exactly 0 at e==0 for any D or
        //    flags) ─────────────────────────────────────────────────────
        // v9 FIX: added a ripple component (sin^2) on top of the smooth
        // gain/quad burn -- lmcmaes was found exploiting the previously
        // purely-smooth burn/corridor signal as a shortcut (consistently
        // *better* with M3 alone than baseline, growing with D: -0.002 at
        // D=32 to -0.034 at D=256), the same "clearer gradient" problem
        // that affected M1 before its sin^2 fix. Both the smooth and
        // ripple parts are 0 at mi==0, so the true optimum is unaffected.
        const double burn = fuel_burn_gain_ * mi * mi + fuel_burn_quad_ * sqr(mi * mi)
                           + fuel_burn_ripple_weight_ * sqr(std::sin(fuel_burn_ripple_freq_ * PI * mi));
        o.fuel_cost += base_fuel_weight_ * burn * dt;

        // ── M3: divert-fuel budget + deceptive pacing corridor (entirely
        //    absent, not just small, when M3 is off). Because burn == 0
        //    exactly at e==0 (m==0 too), the budget never actually
        //    depletes when x follows the reference schedule -- overrun_cost
        //    and corridor_cost are guaranteed 0 at the true optimum for
        //    any D. ─────────────────────────────────────────────────────
        if (config_.M3_enable && !fuel_exhausted) {
            fuel -= msev * burn * dt;   // deviations burn fuel faster once D exceeds D0_mech_
            if (fuel < 0.0) {
                o.overrun_cost += overrun_weight_ * fuel * fuel;
                fuel = 0.0;
                fuel_exhausted = true;
            }
            // v5: checkpoint EVERY stage (was: only ~3 checkpoints at
            // 25/50/75% of D), with a much tighter corridor_slack_, so M3
            // also bites near-optimal, low-deviation candidates.
            if ((i + 1) < D) {

                const double required = fuel_init_ * (1.0 - corridor_slack_ * tau);
                if (fuel < required) {
                    const double gap = required - fuel;
                    o.corridor_cost += corridor_weight_ * gap * gap;
                }
            }
        }

        m_prev = mi;
    }

    o.term2 = e[D - 1] * e[D - 1];   // RAW; terminal weight applied once, outside
    return o;
}

double InterceptorGuidance::evaluate_with_point(const Vec& x) const
{
    const Outcome o = simulate(x);
    const double term_w = terminal_weight_ + (config_.M1_enable ? m1_extra_terminal_weight_ : 0.0);
    return term_w * o.term2
         + o.core_cost
         + o.coupling_cost
         + o.rugged_cost
         + o.fuel_cost
         + o.overrun_cost
         + o.corridor_cost;
}

double InterceptorGuidance::evaluate_core(const Vec& x)
{
    return evaluate_with_point(x);
}

// f1 = tracking-accuracy side (terminal + core + coupling + ruggedness)
// f2 = effort/budget side (fuel + overrun + corridor)
// f1 + f2 == evaluate_core(x). Both are individually exactly 0 at the
// known reference schedule, for any D and any M1/M2/M3 setting, so (0,0)
// is always a genuine, reachable ideal point for the bi-objective view.
Vec InterceptorGuidance::evaluateMultiCore(const Vec& x)
{
    const Outcome o = simulate(x);
    const double term_w = terminal_weight_ + (config_.M1_enable ? m1_extra_terminal_weight_ : 0.0);
    const double f1 = term_w * o.term2 + o.core_cost + o.coupling_cost + o.rugged_cost;
    const double f2 = o.fuel_cost + o.overrun_cost + o.corridor_cost;
    return { f1, f2 };
}

void InterceptorGuidance::gradient_core(const Vec& x, Vec& g)
{
    const int    D  = static_cast<int>(x.size());
    const double h  = 1e-6;
    const double f0 = evaluate_with_point(x);
    Vec xp = x;
    g.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        const double saved = xp[i];
        xp[i] = std::min(1.0, saved + h);
        const double fp = evaluate_with_point(xp);
        g[i] = (fp - f0) / (xp[i] - saved + 1e-30);
        xp[i] = saved;
    }
}

double InterceptorGuidance::safe_x(const Vec& x, int i) const
{
    const int D = static_cast<int>(x.size());
    if (D == 0)   return 0.0;
    if (i < 0)    return x[0];
    if (i >= D)   return x[D - 1];
    return x[i];
}

// ============================================================================
//  Defaults — mirrors the [interceptorguidance] section shipped in
//  optimsolution.cfg.
// ============================================================================
void InterceptorGuidance::load_config()
{
    config_.M1_enable = true;
    config_.M2_enable = true;
    config_.M3_enable = true;

    // Reference schedule r(tau) = clamp01(center + amp*sin(2*PI*freq*tau + phase)).
    // A fixed, non-trivial (not merely constant) curve so the benchmark has
    // some real shape to find, while staying perfectly deterministic/known.
    ref_center_ = 0.50;
    ref_amp_    = 0.30;
    ref_freq_   = 1.00;
    ref_phase_  = 0.40;

    // Always-on core: dominant term, defines the convex "pull to truth".
    core_weight_     = 6.0;
    terminal_weight_ = 4.0;

    // Always-on baseline coupling + ruggedness (onset ~ D0_base_). Kept
    // deliberately small relative to core_weight_ so that even a modest
    // severity fraction well below D0_base_ does not already create real
    // traps -- difficulty should stay ~0 until D actually approaches the
    // pivot, then grow.
    base_couple_weight_ = 0.20;
    base_couple_rho_    = 0.50;
    base_rugged_weight_ = 0.60;
    base_rugged_freq_   = 6.0;

    // M1 extra coupling (onset ~ D0_mech_, only when M1 on).
    // v8 FIX: v7 tried a much larger weight on a plain quadratic penalty,
    // which backfired (see simulate()'s comment) -- increasing weight on a
    // smooth, single-valley penalty just gives adaptive search a clearer
    // gradient to exploit. The fix that actually works is injecting
    // multimodality (sin^2(freq*c) instead of c^2), the same proven
    // ingredient behind M2. Because sin^2 is bounded in [0,1] per stage
    // (vs. an unbounded c^2), the weight needed to make M1 comparable in
    // magnitude to M2 is much smaller than v7's attempt.
    m1_extra_couple_weight_  = 2.2;
    m1_extra_couple_rho_     = 0.85;
    m1_extra_couple_freq_    = 5.0;    // ripple density of the coupling constraint itself
    m1_extra_terminal_weight_ = 3.0;

    // M2 extra, denser ripple (onset ~ D0_mech_, only when M2 on).
    m2_extra_rugged_weight_ = 1.20;
    m2_extra_rugged_freq_   = 20.0;

    // Always-on effort/control cost (purely deviation-driven).
    base_fuel_weight_ = 0.5;
    // v7 tried much stronger burn + a tighter budget to close the gap with
    // M2 (M3's contribution above baseline measured ~18x smaller than
    // M2's), but -- like the M1 quadratic-weight attempt -- simply
    // amplifying a smooth, threshold-based penalty made the shortcut
    // WORSE for some algorithms (rde), not better, once its weight got
    // large enough to dominate the landscape with a still-basically-smooth
    // signal. v8 settles on a more modest strengthening than v7's
    // (checked empirically to at least not worsen monotonicity), while
    // acknowledging M3 remains a comparatively gentler mechanism than M2
    // by nature (a resource *budget*, not inherently multimodal).
    fuel_burn_gain_   = 1.2;   // was 1.0 (v6) / 3.0 (v7, reverted)
    fuel_burn_quad_   = 0.7;   // was 0.5 (v6) / 2.5 (v7, reverted)
    // v9 FIX: a ripple component on the burn rate itself, so fuel
    // depletion is not a purely smooth function of the deviation --
    // lmcmaes was measured exploiting the smooth version as a shortcut
    // (see simulate() comment). Zero at mi==0, so the true optimum is
    // unaffected.
    fuel_burn_ripple_weight_ = 0.8;
    fuel_burn_ripple_freq_   = 8.0;

    // M3 divert-fuel budget + corridor (onset ~ D0_mech_, only when M3 on).
    fuel_init_       = 0.85;   // was 1.0 -- a somewhat tighter overall budget
    overrun_weight_  = 9.0;    // was 6.0
    corridor_weight_ = 2.0;    // was 3.0 -- kept modest since it now applies every stage
    // v5 FIX: was 0.50 (required-fuel curve drops to 50% by the end of the
    // window, checked at only ~3 points) -- so loose + so rarely checked
    // that only large, sustained deviation ever tripped it, meaning M3
    // added almost nothing once an algorithm was already close to
    // converged (confirmed empirically: 74/80 identical mean values
    // between m1_m2 and m1_m2_m3). Combined with checking every stage
    // (see simulate()), a much tighter slack keeps the required curve
    // close to fuel_init_ throughout, so even brief, modest deviation now
    // produces a real corridor penalty.
    corridor_slack_  = 0.06;

    // Difficulty-onset pivots (dimension). Baseline mechanisms (coupling +
    // ruggedness, always on) ramp up around D0_base_; the M1/M2/M3 EXTRA
    // terms ramp up around the earlier D0_mech_.
    D0_base_    = 60.0;
    width_base_ = 15.0;
    D0_mech_    = 20.0;
    width_mech_ = 6.0;

    config_loaded_ = true;
}

void InterceptorGuidance::load_main_config_overrides()
{
    const std::string candidates[] = {
        "optimsolution.cfg",
        "./optimsolution.cfg",
        "../optimsolution.cfg",
    };
    std::ifstream in;
    for (const auto& path : candidates) {
        in.open(path);
        if (in.good()) { loaded_config_path_ = path; break; }
        in.clear();
    }
    if (!in.good()) return;

    std::string line, section;
    while (std::getline(in, line)) {
        const auto cut = line.find_first_of(";#");
        if (cut != std::string::npos) line.erase(cut);
        line = trim_copy(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = to_lower_copy(line.substr(1, line.size() - 2));
            continue;
        }
        if (section != "interceptorguidance") continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key   = to_lower_copy(trim_copy(line.substr(0, eq)));
        const std::string value = trim_copy(line.substr(eq + 1));
        const double dval = std::atof(value.c_str());

        if      (key == "m1_enable") config_.M1_enable = parse_bool_value(value, config_.M1_enable);
        else if (key == "m2_enable") config_.M2_enable = parse_bool_value(value, config_.M2_enable);
        else if (key == "m3_enable") config_.M3_enable = parse_bool_value(value, config_.M3_enable);

        else if (key == "ref_center") ref_center_ = dval;
        else if (key == "ref_amp")    ref_amp_    = dval;
        else if (key == "ref_freq")   ref_freq_   = dval;
        else if (key == "ref_phase")  ref_phase_  = dval;

        else if (key == "core_weight")     core_weight_     = dval;
        else if (key == "terminal_weight") terminal_weight_ = dval;

        else if (key == "base_couple_weight") base_couple_weight_ = dval;
        else if (key == "base_couple_rho")    base_couple_rho_    = dval;
        else if (key == "base_rugged_weight") base_rugged_weight_ = dval;
        else if (key == "base_rugged_freq")   base_rugged_freq_   = dval;

        else if (key == "m1_extra_couple_weight")  m1_extra_couple_weight_  = dval;
        else if (key == "m1_extra_couple_rho")     m1_extra_couple_rho_     = dval;
        else if (key == "m1_extra_terminal_weight") m1_extra_terminal_weight_ = dval;

        else if (key == "m2_extra_rugged_weight") m2_extra_rugged_weight_ = dval;
        else if (key == "m2_extra_rugged_freq")   m2_extra_rugged_freq_   = dval;

        else if (key == "base_fuel_weight") base_fuel_weight_ = dval;
        else if (key == "fuel_burn_gain")   fuel_burn_gain_   = dval;
        else if (key == "fuel_burn_quad")   fuel_burn_quad_   = dval;
        else if (key == "fuel_burn_ripple_weight") fuel_burn_ripple_weight_ = dval;
        else if (key == "fuel_burn_ripple_freq")   fuel_burn_ripple_freq_   = dval;

        else if (key == "fuel_init")       fuel_init_       = dval;
        else if (key == "overrun_weight")  overrun_weight_  = dval;
        else if (key == "corridor_weight") corridor_weight_ = dval;
        else if (key == "corridor_slack")  corridor_slack_  = dval;

        else if (key == "d0_base")    D0_base_    = dval;
        else if (key == "width_base") width_base_ = dval;
        else if (key == "d0_mech")    D0_mech_    = dval;
        else if (key == "width_mech") width_mech_ = dval;
    }
}

void InterceptorGuidance::refresh_metadata()
{
    setFullName("Adaptive Divert-Budget Interceptor Guidance under Deceptive Threat Maneuvering (ADIG)");
    setCategory("aerospace guidance & control (surrogate benchmark)");
    setModality("multimodal");
    setSeparability("non-separable");
    // Exact for every dimension and every M1/M2/M3 combination -- see the
    // positive-definiteness argument at the top of this file.
    setKnownGlobalOptimum(0.0);
}

// ----------------------------------------------------------------------------
//  String / parsing utilities
// ----------------------------------------------------------------------------
std::string InterceptorGuidance::trim_copy(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string InterceptorGuidance::to_lower_copy(const std::string& s)
{
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return r;
}

bool InterceptorGuidance::parse_bool_value(const std::string& value, bool default_value)
{
    const std::string v = to_lower_copy(trim_copy(value));
    if (v == "1" || v == "true"  || v == "yes" || v == "on")  return true;
    if (v == "0" || v == "false" || v == "no"  || v == "off") return false;
    return default_value;
}

} // namespace optimsolution
