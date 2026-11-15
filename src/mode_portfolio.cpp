// ---------------------------------------------------------------------
// Mode 8: mid-run dynamic algorithm switching / portfolio run ("--portfolio").
//
// WHAT THIS IS
// ------------
// Every existing mode picks ONE method and runs it start-to-finish on ONE
// problem. This mode instead splits the total evaluation budget into a
// sequence of SEGMENTS, and re-decides -- before each segment -- which
// registered method (from a candidate list) gets to run next, using the
// incumbent best point found so far as the starting region for whichever
// method is chosen. The result is a single relay "trajectory" made of
// runs from possibly several different algorithmic families, instead of
// one method's run from start to finish.
//
// WHY SEGMENTS, NOT LITERAL MID-ITERATION HAND-OFF
// -------------------------------------------------
// Optimizer::run() (include/optimizer.h) always calls init() once at the
// top and owns its internal state (population arrays, step sizes,
// covariance matrices, velocities, ...) in whatever representation that
// specific method uses -- there is no common cross-method population
// format to hand off mid-iteration, and there is no reasonable way to add
// one without rewriting all 74 methods. Segmenting the *budget* instead
// of the *iteration* sidesteps this completely: each segment is one
// ordinary, fully-formed Optimizer::run() call (nothing about the 74
// existing methods needs to change), and continuity across segments is
// achieved by (a) reusing the SAME Problem instance across the whole relay
// so its evaluation counter keeps accumulating toward one shared budget,
// and (b) temporarily narrowing that Problem's bounds around the current
// incumbent before each segment (see Problem::pushShrunkBounds(), added
// alongside this mode) so the next method's own Initializer -- whatever
// sampling distribution it uses -- starts near the incumbent rather than
// re-exploring the whole box from scratch.
//
// SWITCHING POLICIES
// -------------------
//   fixed        : always the same method (baseline / ablation -- lets you
//                   A/B a "portfolio" run against a plain single-method run
//                   under IDENTICAL segment/warm-start bookkeeping).
//   round_robin  : cycles through --methods in order, ignoring performance.
//                   A second, cheap baseline (does switching help at all,
//                   even with a policy that carries no information?).
//   greedy_probe : the actual adaptive policy. Before each segment, every
//                   candidate method gets a short, IDENTICALLY warm-started
//                   probe (--probe-evals each, same rng seed so the
//                   comparison is fair), and whichever probe reaches the
//                   best f is declared the segment's method, which then
//                   gets a full fresh segment of its own (probe included,
//                   both count against the shared budget -- see "honest
//                   accounting" below).
//
// This intentionally leaves a documented extension point rather than
// faking a fourth "landscape_rf" policy: a real ELA-feature-driven policy
// (reusing the meta-learning pipeline already built for Mode 5/6's
// algorithm-selection work) needs a trained classifier artifact as input,
// which is a separate research deliverable, not something to stub out
// here. The hook is exactly where you'd expect it: decideMethod() below is
// the single function every policy goes through; a fourth branch there
// that loads a serialized model and scores the same local-dispersion
// features already computed for --landscape-log is the entire integration
// surface once that classifier exists.
//
// HONEST ACCOUNTING
// ------------------
// --total-evals is the WHOLE budget for one run, probes included. A
// greedy_probe segment therefore costs probe_evals * |candidates| extra
// evaluations beyond its own segment_evals -- this is the real, non-free
// cost of adaptive selection, not hidden overhead, and it is why
// round_robin and fixed exist as baselines: they spend zero evaluations
// on deciding, so a fair comparison must show greedy_probe's segment-level
// gains net of its own probing cost, not before it.
//
// OUTPUT
// -------
// --out-prefix <p> writes:
//   <p>_segments.csv : one row per (run, segment): method chosen, why,
//                       evals spent (probe + committed), f before/after,
//                       incumbent f after, warm-start radius used.
//   <p>_summary.csv  : one row per run: final best f, evals used, method
//                       usage histogram, number of switches.
// Self-contained: does not touch main(), makeProblem(), makeMethod(), or
// any other mode's code, beyond a two-line dispatch hook in main.cpp and
// the additive Problem::pushShrunkBounds()/popBounds() pair.
// ---------------------------------------------------------------------
#include "problem.h"
#include "optimizer.h"
#include "factory.h"
#include "config.h"
#include "options.h"
#include "utils.h"
#include "fixed_dims.h"
#include "allowed_dims.h"
#include "portfolio.h"
#include "stats.h"

#include <algorithm>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace optimsolution {

namespace {

struct PortfolioArgs {
    std::string problem;
    int         dim = 0;
    std::vector<std::pair<std::string,int>> problems; // multi-problem: (name, dim-or-0-for-auto); see --problems
    std::vector<std::string> methods;          // candidate pool
    std::string policy         = "greedy_probe"; // fixed | round_robin | greedy_probe
    std::string fixedMethod;                    // required when policy == "fixed"
    long long   totalEvals     = 150000;
    long long   segmentEvals   = 5000;
    long long   probeEvals     = 300;           // only spent under greedy_probe
    double      radius0        = 0.35;          // initial warm-start box, fraction of full range
    double      radiusDecay    = 0.85;           // applied every segment (progressive exploitation)
    double      radiusMin      = 0.02;           // floor so the warm-start box never collapses to a point
    int         runs           = 30;
    unsigned long long seedBase = 1337ULL;
    std::string cfgFile;                         // optional; falls back to the usual optimsolution.cfg lookup
    std::string outPrefix;                        // required
    bool        withBaselines  = false;           // also run round_robin + fixed(each method) for comparison
    bool        landscapeLog   = false;           // also log cheap local-dispersion stats per segment (diagnostic)
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage:\n  %s --portfolio --problem <name> [--dim D] --methods m1,m2,m3\n"
        "     [--policy fixed|round_robin|greedy_probe] [--fixed-method m1]\n"
        "     [--total-evals 150000] [--segment-evals 5000] [--probe-evals 300]\n"
        "     [--warmstart-radius0 0.35] [--warmstart-decay 0.85] [--warmstart-radius-min 0.02]\n"
        "     [--runs 30] [--seed-base 1337] [--config path] --out-prefix <prefix>\n"
        "     [--with-baselines] [--landscape-log]\n"
        "\n  %s --portfolio --problems sphere:10,sphere:30,ackley:16,rastrigin --methods m1,m2,m3 ...\n"
        "     (multi-problem: comma-separated 'name' or 'name:dim' entries -- 'name' alone needs\n"
        "     either a fixed-dimension problem or a --dim fallback for the rest. Overrides\n"
        "     --problem/--dim. Every variant runs on every problem with the SAME per-run seeds,\n"
        "     and cross-problem 'overall_*' comparison files are written in addition to the\n"
        "     usual per-run ones -- see the --with-baselines note below.)\n"
        "\npolicy:\n"
        "  fixed        always --fixed-method (baseline)\n"
        "  round_robin  cycles through --methods in order (baseline)\n"
        "  greedy_probe probes every candidate each segment, keeps the best (adaptive)\n"
        "\n--with-baselines: additionally run round_robin and one fixed:<method> baseline per\n"
        "  candidate, all on the SAME per-run seeds, then write ranking / Friedman /\n"
        "  pairwise-Wilcoxon(Holm) tables comparing every variant. Without it a single\n"
        "  variant is run and no comparison statistics are produced.\n"
        "\n--methods: comma-separated short names from the method table in README.md\n"
        "  (the local methods gd/lbfgs/bfgs/nm/cg/ncg/adam are valid candidates too: each\n"
        "  segment is one local search from a start drawn in the warm-start box).\n",
        argv0, argv0);
    std::exit(1);
}

std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t a = item.find_first_not_of(" \t");
        size_t b = item.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

// Wraps a field in double quotes and escapes any embedded quotes, so that
// labels like "fixed:arq" and histograms like "arq:13;bho:12" survive a
// round-trip through any CSV reader unambiguously.
std::string csvQuote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    out += '"';
    return out;
}

PortfolioArgs parseArgs(int argc, char** argv) {
    PortfolioArgs a;
    auto need = [&](int i) -> std::string {
        if (i + 1 >= argc) usage(argv[0]);
        return argv[i + 1];
    };
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--problem")            { a.problem = toLower(need(i)); ++i; }
        else if (k == "--dim")                { a.dim = std::stoi(need(i)); ++i; }
        else if (k == "--problems") {
            // "name" or "name:dim", comma-separated. dim==0 means "auto"
            // (resolved later against the problem's fixed dim, or --dim).
            for (auto& entry : splitCsv(need(i))) {
                const size_t colon = entry.find(':');
                if (colon == std::string::npos) {
                    a.problems.push_back({ toLower(entry), 0 });
                } else {
                    const std::string name = toLower(entry.substr(0, colon));
                    const int d = std::stoi(entry.substr(colon + 1));
                    a.problems.push_back({ name, d });
                }
            }
            ++i;
        }
        else if (k == "--methods")            { for (auto& m : splitCsv(need(i))) a.methods.push_back(toLower(m)); ++i; }
        else if (k == "--policy")             { a.policy = toLower(need(i)); ++i; }
        else if (k == "--fixed-method")       { a.fixedMethod = toLower(need(i)); ++i; }
        else if (k == "--total-evals")        { a.totalEvals = std::stoll(need(i)); ++i; }
        else if (k == "--segment-evals")      { a.segmentEvals = std::stoll(need(i)); ++i; }
        else if (k == "--probe-evals")        { a.probeEvals = std::stoll(need(i)); ++i; }
        else if (k == "--warmstart-radius0")  { a.radius0 = std::stod(need(i)); ++i; }
        else if (k == "--warmstart-decay")    { a.radiusDecay = std::stod(need(i)); ++i; }
        else if (k == "--warmstart-radius-min") { a.radiusMin = std::stod(need(i)); ++i; }
        else if (k == "--runs")               { a.runs = std::stoi(need(i)); ++i; }
        else if (k == "--seed-base")          { a.seedBase = std::stoull(need(i)); ++i; }
        else if (k == "--config")             { a.cfgFile = need(i); ++i; }
        else if (k == "--out-prefix")         { a.outPrefix = need(i); ++i; }
        else if (k == "--with-baselines")     { a.withBaselines = true; }
        else if (k == "--landscape-log")      { a.landscapeLog = true; }
        else if (k == "-h" || k == "--help")  usage(argv[0]);
        else { std::fprintf(stderr, "Unknown argument: %s\n", k.c_str()); usage(argv[0]); }
    }
    if ((a.problem.empty() && a.problems.empty()) || a.methods.empty() || a.outPrefix.empty()) usage(argv[0]);
    if (a.policy != "fixed" && a.policy != "round_robin" && a.policy != "greedy_probe") {
        std::fprintf(stderr, "Unknown --policy '%s'\n", a.policy.c_str());
        usage(argv[0]);
    }
    if (a.policy == "fixed" && a.fixedMethod.empty()) {
        std::fprintf(stderr, "--policy fixed requires --fixed-method\n");
        usage(argv[0]);
    }
    return a;
}

// One segment's committed run: fresh Optimizer instance, fresh init(),
// bounded by [prob->calls(), prob->calls()+evalsBudget) on the SHARED
// Problem instance. Returns the segment's own best (x, f); the caller
// decides whether that beats the running incumbent.
struct SegmentResult {
    Vec    x;
    double f = std::numeric_limits<double>::infinity();
    long long evalsSpent = 0;
};

SegmentResult runOneSegment(const std::string& method, Problem* prob, const std::string& cfg_file,
                             long long evalsBudget, unsigned long long seed, int runIndex) {
    SegmentResult out;
    auto opt = makeMethod(method);
    if (!opt) return out;

    // BUG FIX: this used to receive one Config loaded ONCE per run for
    // a.methods.front() and reuse it for every candidate's segments --
    // meaning every method except the first candidate ran with a foreign
    // method's [section] parameter overrides (methodKV) instead of its
    // own. Loading fresh here, keyed on the METHOD ACTUALLY RUNNING,
    // mirrors exactly what the plain single-run CLI path does per run
    // and is what makes e.g. [bho]'s own tuned parameters actually apply
    // when bho is the one running, rather than whatever the first
    // candidate in --methods happened to be.
    Config cfg = Config::load(cfg_file, method);

    GeneralOptions g = cfg.g;
    g.max_evals = prob->calls() + evalsBudget;   // cumulative threshold on the shared Problem
    // A handful of methods (jso, cmaes, lmcmaes, ea4eig, pso, gwo, sparq)
    // opt into reading max_evals_per_dim ("<N>xD" in the cfg file) instead
    // of max_evals literally (see options.h). Since segment budgets are
    // computed and enforced here, not in the cfg file, that per-dimension
    // scaling must be disabled per segment or those methods would ignore
    // the segment budget entirely and run to the cfg file's full literal
    // N*D evaluations in a single "segment".
    g.max_evals_per_dim = 0;
    g.runs = 1;

    long long before = prob->calls();
    opt->setProblem(prob);
    opt->setSeed(seed);
    opt->setGeneralOptions(g);
    opt->setTermination(cfg.t);
    opt->setInitOptions(cfg.init);
    opt->setEndLocalFromGlobal(cfg.g.end_local_refine, toLower(cfg.g.end_local_method));
    opt->configure(cfg.methodKV);
    opt->setRunIndex(runIndex);

    // Segment probes/commits happen many times per run and are not meant
    // to be watched iteration-by-iteration; suppress per-iteration console
    // spam by pointing progress lines at a scratch buffer.
    std::ostringstream sink;
    opt->setOutputStream(&sink);

    auto rr = opt->run();
    out.x = rr.xbest;
    out.f = rr.fbest;
    out.evalsSpent = prob->calls() - before;
    return out;
}

// greedy_probe's decision step: run a short, identically warm-started
// probe of every candidate, return the winner's name and its probe result
// (the caller still runs a full fresh segment for the winner afterward).
std::pair<std::string, SegmentResult> probeAndPick(const std::vector<std::string>& candidates,
                                                     Problem* prob, const std::string& cfg_file,
                                                     long long probeEvals, unsigned long long seed,
                                                     int runIndex) {
    std::string best_m;
    SegmentResult best;
    for (const auto& m : candidates) {
        SegmentResult r = runOneSegment(m, prob, cfg_file, probeEvals, seed, runIndex);
        if (r.f < best.f) { best = r; best_m = m; }
    }
    return { best_m, best };
}

// ---------------------------------------------------------------------
// One "variant" = one fully specified switching strategy that can be run
// end-to-end and compared against the others. Baseline variants exist so
// that the headline question ("does mid-run switching actually help?")
// is answered against something, rather than asserted: a greedy_probe
// number on its own is uninterpretable without knowing what simply
// sticking with one method for the whole budget would have achieved.
// ---------------------------------------------------------------------
struct VariantSpec {
    std::string label;        // column/row label used in every output CSV
    std::string policy;       // fixed | round_robin | greedy_probe
    std::string fixedMethod;  // only meaningful when policy == "fixed"
};

struct RunOutcome {
    double      final_f = std::numeric_limits<double>::infinity();
    long long   evals_used = 0;
    int         num_segments = 0;
    int         num_switches = 0;
    std::string histogram;
};

// Per-(variant, method) accounting, aggregated across all runs of that
// variant. Answers "which candidate actually carried the relay, and did
// running it actually move the incumbent?" -- the segment-level question
// the summary table alone cannot answer.
struct MethodUsage {
    long long segments = 0;        // how many segments this method was chosen for
    long long improvedCount = 0;   // how many of those strictly improved the incumbent
    double    totalImprovement = 0.0; // sum of (incumbent before - incumbent after), finite terms only
};

// Runs every repetition of one variant, appends its per-segment rows to
// segCsv, and returns one RunOutcome per run. Seeds are derived from
// seedBase + run index ONLY -- deliberately not from the variant -- so run
// r of every variant sees the identical seed, which is what makes the
// across-variant comparison a PAIRED one and therefore what makes the
// Wilcoxon signed-rank / Friedman blocking below valid rather than
// decorative.
std::vector<RunOutcome> runOneVariant(const VariantSpec& v,
                                       const PortfolioArgs& a,
                                       const std::string& problem,
                                       int dim,
                                       const std::string& cfg_file,
                                       std::ofstream& segCsv,
                                       std::map<std::string, MethodUsage>& usageOut) {
    std::vector<RunOutcome> outcomes;
    outcomes.reserve((size_t)a.runs);

    for (int r = 0; r < a.runs; ++r) {
        auto p = makeProblem(problem);
        p->init(dim);

        unsigned long long seed = a.seedBase + (unsigned long long)r;

        Vec incumbent_x;
        double incumbent_f = std::numeric_limits<double>::infinity();
        double radius = a.radius0;
        std::string prevMethod;
        int numSwitches = 0;
        int segIdx = 0;
        std::vector<std::pair<std::string,int>> histogram; // method -> count, insertion order

        auto bumpHistogram = [&](const std::string& m) {
            for (auto& kv : histogram) if (kv.first == m) { ++kv.second; return; }
            histogram.push_back({m, 1});
        };

        while (p->calls() < a.totalEvals) {
            long long remaining = a.totalEvals - p->calls();
            long long segBudget = std::min(a.segmentEvals, remaining);
            if (segBudget <= 0) break;

            double f_before = incumbent_f;
            long long evals_before = p->calls();

            // Warm-start the shared Problem's bounds around the incumbent
            // (skipped on the very first segment: no incumbent to center on
            // yet, so the initial segment explores the full box exactly like
            // a normal single-method run would).
            bool haveIncumbent = !incumbent_x.empty();
            if (haveIncumbent) p->pushShrunkBounds(incumbent_x, radius);

            std::string chosen;
            std::string reason;
            long long probeEvalsTotal = 0;
            SegmentResult committed;

            if (v.policy == "fixed") {
                chosen = v.fixedMethod;
                reason = "fixed";
                committed = runOneSegment(chosen, p.get(), cfg_file, segBudget, seed + 1000ULL * segIdx, r);
            } else if (v.policy == "round_robin") {
                chosen = a.methods[segIdx % a.methods.size()];
                reason = "round_robin";
                committed = runOneSegment(chosen, p.get(), cfg_file, segBudget, seed + 1000ULL * segIdx, r);
            } else { // greedy_probe
                long long probeBudget = std::min(a.probeEvals, remaining / (long long)a.methods.size());
                if (probeBudget < 1) probeBudget = 1;
                auto picked = probeAndPick(a.methods, p.get(), cfg_file, probeBudget,
                                            seed + 1000ULL * segIdx, r);
                chosen = picked.first;
                probeEvalsTotal = probeBudget * (long long)a.methods.size();
                reason = "greedy_probe";
                // The probe result already counts as spent evaluations; the
                // winner now gets a fresh continuation segment with whatever
                // budget remains for this segment slot.
                long long remainingAfterProbe = a.totalEvals - p->calls();
                long long contBudget = std::min(segBudget, remainingAfterProbe);
                if (contBudget > 0) {
                    SegmentResult cont = runOneSegment(chosen, p.get(), cfg_file, contBudget,
                                                        seed + 1000ULL * segIdx + 1, r);
                    committed = (cont.f < picked.second.f) ? cont : picked.second;
                } else {
                    committed = picked.second;
                }
            }

            if (haveIncumbent) p->popBounds();

            // Actual total evaluations spent in this segment slot (probe(s)
            // + continuation, all counted on the shared Problem).
            long long evalsThisSegment = p->calls() - evals_before;

            bool switched = (!prevMethod.empty() && prevMethod != chosen);
            if (switched) ++numSwitches;
            // Radius shrinks EVERY segment (not only on a switch), with a
            // floor: a run that happens never to switch must still narrow
            // in on its incumbent rather than restarting wide forever.
            radius *= a.radiusDecay;
            if (radius < a.radiusMin) radius = a.radiusMin;
            prevMethod = chosen;
            bumpHistogram(chosen);

            const double f_incumbent_before = incumbent_f;
            if (committed.f < incumbent_f) {
                incumbent_f = committed.f;
                incumbent_x = committed.x;
            }

            {
                MethodUsage& u = usageOut[chosen];
                ++u.segments;
                if (committed.f < f_incumbent_before) {
                    ++u.improvedCount;
                    // The first segment's "before" is +inf (no incumbent
                    // yet), which would make the improvement infinite and
                    // poison every mean built from it -- only finite deltas
                    // are accumulated.
                    const double delta = f_incumbent_before - incumbent_f;
                    if (std::isfinite(delta)) u.totalImprovement += delta;
                }
            }

            segCsv << csvQuote(problem) << "," << dim << "," << csvQuote(v.label) << "," << r << "," << segIdx << "," << chosen << ","
                   << csvQuote(reason) << ","
                   << evals_before << "," << evalsThisSegment << "," << probeEvalsTotal << ","
                   << std::setprecision(15) << f_before << "," << committed.f << "," << incumbent_f << ","
                   << std::setprecision(6) << radius << "," << (switched ? 1 : 0) << "\n";

            ++segIdx;
        }

        std::ostringstream hist;
        for (size_t i = 0; i < histogram.size(); ++i) {
            if (i) hist << ";";
            hist << histogram[i].first << ":" << histogram[i].second;
        }

        RunOutcome o;
        o.final_f      = incumbent_f;
        o.evals_used   = p->calls();
        o.num_segments = segIdx;
        o.num_switches = numSwitches;
        o.histogram    = hist.str();
        outcomes.push_back(o);

        std::printf("[portfolio] %-10s | %-22s run %d/%d: best_f=%.12g evals=%lld segments=%d switches=%d\n",
                     problem.c_str(), v.label.c_str(), r + 1, a.runs, o.final_f,
                     (long long)o.evals_used, o.num_segments, o.num_switches);
        std::fflush(stdout);
    }

    return outcomes;
}

// One (block, variant) observation fed into writeComparisonStats(): a
// single scalar to rank/test on, plus the two descriptive extras the
// Ranking table also reports. "Block" is whatever the comparison is paired
// by -- a run (single-problem case) or a problem (multi-problem case).
struct BlockObservation {
    double value = std::numeric_limits<double>::quiet_NaN();
    double evals = 0.0;
    double switches = 0.0;
};

// Writes the three comparison files (ranking / Friedman / pairwise
// Wilcoxon+Holm) for a matrix obs[block][variant], to
// "<outPrefix><fileSuffix>_{ranking,friedman,wilcoxon}.csv". Used three
// ways: once for the classic single-problem, run-blocked comparison
// (fileSuffix ""); once per problem when multiple problems are given
// (fileSuffix "__<problem>", still run-blocked, N=runs); and once for the
// cross-problem comparison (fileSuffix "_overall", PROBLEM-blocked,
// N=problems, each observation already averaged over that problem's runs).
// The blocking is what the caller decides by how it builds `obs` --
// this function only knows it has N blocks x k variants.
void writeComparisonStats(const std::string& outPrefix, const std::string& fileSuffix,
                           const std::vector<VariantSpec>& variants,
                           const std::vector<std::vector<BlockObservation>>& obs,
                           const std::string& blockNoun) {
    const int k = (int)variants.size();
    const int N = (int)obs.size();
    if (k < 2 || N < 2) return;

    std::vector<std::vector<double>> ranks;
    ranks.reserve((size_t)N);
    for (int b = 0; b < N; ++b) {
        std::vector<double> row((size_t)k);
        for (int vi = 0; vi < k; ++vi) row[(size_t)vi] = obs[(size_t)b][(size_t)vi].value;
        ranks.push_back(stats::averageRank(row));
    }

    // -- Ranking table --
    {
        std::vector<double> meanRank((size_t)k, 0.0);
        for (const auto& row : ranks)
            for (int vi = 0; vi < k; ++vi) meanRank[(size_t)vi] += row[(size_t)vi];
        for (auto& mr : meanRank) mr /= (double)N;
        std::vector<double> finalRank = stats::competitionRank(meanRank);

        // wins: how many blocks this variant achieved rank 1 in (ties all
        // count -- a variant tied for best on a given problem did win it).
        std::vector<int> wins((size_t)k, 0);
        for (const auto& row : ranks)
            for (int vi = 0; vi < k; ++vi) if (row[(size_t)vi] <= 1.0000001) ++wins[(size_t)vi];

        std::ofstream rankCsv(outPrefix + fileSuffix + "_ranking.csv", std::ios::trunc);
        rankCsv << "variant,mean_value,sd_value,median_value,min_value,max_value,mean_rank,final_rank,"
                   "wins_" << blockNoun << "s,mean_evals,mean_switches\n";
        for (int vi = 0; vi < k; ++vi) {
            std::vector<double> vals;
            double sumE = 0.0, sumSw = 0.0;
            for (int b = 0; b < N; ++b) {
                vals.push_back(obs[(size_t)b][(size_t)vi].value);
                sumE  += obs[(size_t)b][(size_t)vi].evals;
                sumSw += obs[(size_t)b][(size_t)vi].switches;
            }
            std::vector<double> sorted = vals;
            std::sort(sorted.begin(), sorted.end());
            const double med = (N % 2 == 1) ? sorted[(size_t)(N / 2)]
                                            : 0.5 * (sorted[(size_t)(N / 2 - 1)] + sorted[(size_t)(N / 2)]);
            double mean = 0.0; for (double x : vals) mean += x; mean /= (double)N;
            double var = 0.0;  for (double x : vals) { const double d = x - mean; var += d * d; }
            const double sd = (N > 1) ? std::sqrt(var / (double)(N - 1)) : 0.0;

            rankCsv << csvQuote(variants[(size_t)vi].label) << ","
                    << std::setprecision(15)
                    << mean << "," << sd << "," << med << ","
                    << sorted.front() << "," << sorted.back() << ","
                    << std::setprecision(6) << meanRank[(size_t)vi] << ","
                    << (long long)finalRank[(size_t)vi] << ","
                    << wins[(size_t)vi] << ","
                    << (sumE / N) << "," << (sumSw / N) << "\n";
        }
    }

    // -- Friedman omnibus --
    const stats::FriedmanResult fr = stats::friedmanTest(ranks);
    {
        std::ofstream frCsv(outPrefix + fileSuffix + "_friedman.csv", std::ios::trunc);
        frCsv << "statistic,value\n";
        frCsv << "valid," << (fr.valid ? 1 : 0) << "\n";
        frCsv << "num_variants_k," << k << "\n";
        frCsv << "num_" << blockNoun << "s_N," << fr.N << "\n";
        frCsv << "chi_square," << std::setprecision(10) << fr.chi2 << "\n";
        frCsv << "df," << fr.df << "\n";
        frCsv << "p_value," << std::setprecision(10) << fr.p << "\n";
        frCsv << "significant_at_0.05," << ((fr.valid && fr.p < 0.05) ? 1 : 0) << "\n";
        frCsv << "conclusion,"
              << csvQuote(!fr.valid
                    ? std::string("not enough data for an omnibus test")
                    : (fr.p < 0.05
                        ? std::string("variants are NOT all equivalent (reject equality); see pairwise Wilcoxon")
                        : std::string("no statistically significant difference between variants at alpha=0.05")))
              << "\n";
    }

    // -- Pairwise Wilcoxon signed-rank + Holm --
    {
        struct Pair { int i, j; stats::WilcoxonResult w; };
        std::vector<Pair> pairs;
        std::vector<double> praw;
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                std::vector<double> A, B;
                for (int b = 0; b < N; ++b) {
                    A.push_back(obs[(size_t)b][(size_t)i].value);
                    B.push_back(obs[(size_t)b][(size_t)j].value);
                }
                stats::WilcoxonResult w = stats::wilcoxonSignedRank(A, B);
                pairs.push_back({ i, j, w });
                praw.push_back(w.valid ? w.p : 1.0);
            }
        }
        const std::vector<double> padj = stats::holmAdjust(praw);

        std::ofstream wCsv(outPrefix + fileSuffix + "_wilcoxon.csv", std::ios::trunc);
        wCsv << "variant_a,variant_b,mean_value_a,mean_value_b,n_nonzero_diff,statistic,z,"
                "p_raw,p_holm,significant_at_0.05,better\n";
        for (size_t pi = 0; pi < pairs.size(); ++pi) {
            const Pair& pr = pairs[pi];
            double ma = 0.0, mb = 0.0;
            for (int b = 0; b < N; ++b) {
                ma += obs[(size_t)b][(size_t)pr.i].value;
                mb += obs[(size_t)b][(size_t)pr.j].value;
            }
            ma /= (double)N; mb /= (double)N;
            const bool sig = pr.w.valid && padj[pi] < 0.05;
            std::string better = "n.s. (indistinguishable)";
            if (sig) better = (ma < mb) ? variants[(size_t)pr.i].label : variants[(size_t)pr.j].label;

            wCsv << csvQuote(variants[(size_t)pr.i].label) << ","
                 << csvQuote(variants[(size_t)pr.j].label) << ","
                 << std::setprecision(15) << ma << "," << mb << ","
                 << pr.w.n << "," << std::setprecision(10) << pr.w.statistic << "," << pr.w.z << ","
                 << praw[pi] << "," << padj[pi] << "," << (sig ? 1 : 0) << ","
                 << csvQuote(better) << "\n";
        }
    }

    std::printf("[portfolio] %s%s comparison written (%d %ss): Friedman chi2=%.6g df=%d p=%.6g (%s)\n",
                 outPrefix.c_str(), fileSuffix.c_str(), N, blockNoun.c_str(), fr.chi2, fr.df, fr.p,
                 (fr.valid && fr.p < 0.05) ? "significant" : "not significant");
}

} // namespace

// Resolves one --problems entry (name, dimOrZero) into a concrete
// (problem, dim) pair: dimOrZero==0 means "auto", resolved against the
// problem's own fixed dimension if it has one, else the --dim fallback.
// Exits with an error message (matching the single-problem validation
// already done for --problem/--dim) if neither source can supply a dim, or
// if an explicit dim conflicts with a fixed-dimension problem.
static bool resolveProblemDim(const std::string& name, int requestedDim, int fallbackDim,
                               int& outDim, std::string& err) {
    if (!makeProblem(name)) { err = "Unknown problem: " + name; return false; }
    const int fixed = getFixedDimOrZero(name);
    const int dim = (requestedDim > 0) ? requestedDim : (fixed > 0 ? fixed : fallbackDim);
    if (dim <= 0) { err = "Problem '" + name + "' requires a dimension (not fixed-dimension)"; return false; }
    if (fixed > 0 && requestedDim > 0 && requestedDim != fixed) {
        err = "Problem '" + name + "' has a fixed dimension of " + std::to_string(fixed) +
              "; omit its dim or use :" + std::to_string(fixed);
        return false;
    }
    // BUGFIX: dimensions were not checked against the problem's supported
    // set (e.g. CEC2017: D = 10, 30, 50 only). Problem::init() then threw
    // std::invalid_argument only when that problem's turn came -- possibly
    // hours into a multi-problem run -- and, uncaught, aborted the process
    // (Windows exit code -1073740791 = 0xC0000409), losing the statistics of
    // every problem already finished. Every problem is now validated, and
    // trial-initialised, before the first run starts.
    const std::vector<int> allowed = getAllowedDimsOrEmpty(name);
    if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), dim) == allowed.end()) {
        std::string list;
        for (size_t i = 0; i < allowed.size(); ++i) list += (i ? ", " : "") + std::to_string(allowed[i]);
        err = "Problem '" + name + "' does not support dimension " + std::to_string(dim) +
              " (supported: " + list + ")";
        return false;
    }
    try {
        auto trial = makeProblem(name);
        trial->init(dim);
    } catch (const std::exception& e) {
        err = "Problem '" + name + "' cannot be initialised with dimension " + std::to_string(dim) +
              ": " + e.what();
        return false;
    }
    outDim = dim;
    return true;
}

static int runPortfolioCliImpl(int argc, char** argv);

// Safety net: an exception escaping a run (a method or problem throwing)
// used to terminate the process through abort() with no message at all.
// It is now reported, and the process exits with status 2. The CSV rows of
// every variant already finished have been flushed by then.
int runPortfolioCli(int argc, char** argv) {
    try {
        return runPortfolioCliImpl(argc, argv);
    } catch (const std::exception& e) {
        std::fflush(stdout);
        std::fprintf(stderr, "[portfolio] aborted: %s\n", e.what());
        return 2;
    } catch (...) {
        std::fflush(stdout);
        std::fprintf(stderr, "[portfolio] aborted: unknown exception\n");
        return 2;
    }
}

static int runPortfolioCliImpl(int argc, char** argv) {
    PortfolioArgs a = parseArgs(argc, argv);

    // ---- Resolve the problem list (single --problem, or multi --problems) ----
    std::vector<std::pair<std::string,int>> problemList; // (name, resolved dim)
    if (!a.problems.empty()) {
        for (const auto& pd : a.problems) {
            int resolved = 0;
            std::string err;
            if (!resolveProblemDim(pd.first, pd.second, a.dim, resolved, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            problemList.push_back({ pd.first, resolved });
        }
    } else {
        int resolved = 0;
        std::string err;
        if (!resolveProblemDim(a.problem, a.dim, 0, resolved, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        problemList.push_back({ a.problem, resolved });
    }

    for (const auto& m : a.methods) {
        auto probe = makeMethod(m);
        if (!probe) { std::fprintf(stderr, "Unknown method: %s\n", m.c_str()); return 1; }
    }
    if (a.policy == "fixed") {
        auto probe = makeMethod(a.fixedMethod);
        if (!probe) { std::fprintf(stderr, "Unknown --fixed-method: %s\n", a.fixedMethod.c_str()); return 1; }
    }

    const std::string cfg_file = a.cfgFile.empty() ? "optimsolution.cfg" : a.cfgFile;

    // ---- Build the variant list (problem-independent) -----------------
    std::vector<VariantSpec> variants;
    {
        VariantSpec primary;
        primary.policy = a.policy;
        primary.fixedMethod = a.fixedMethod;
        primary.label = (a.policy == "fixed") ? ("fixed:" + a.fixedMethod) : a.policy;
        variants.push_back(primary);
    }
    if (a.withBaselines) {
        auto haveLabel = [&](const std::string& l) {
            for (const auto& v : variants) if (v.label == l) return true;
            return false;
        };
        if (!haveLabel("round_robin") && a.methods.size() >= 2) {
            variants.push_back({ "round_robin", "round_robin", "" });
        }
        for (const auto& m : a.methods) {
            const std::string lbl = "fixed:" + m;
            if (!haveLabel(lbl)) variants.push_back({ lbl, "fixed", m });
        }
    }
    const int k = (int)variants.size();

    // ---- Output files (shared across every problem) -------------------
    std::ofstream segCsv(a.outPrefix + "_segments.csv", std::ios::trunc);
    segCsv << "problem,dim,variant,run,segment,method,reason,evals_before,evals_this_segment,evals_probe_total,"
              "f_before_incumbent,f_segment_best,f_incumbent_after,radius,switched\n";

    std::ofstream sumCsv(a.outPrefix + "_summary.csv", std::ios::trunc);
    sumCsv << "problem,dim,variant,run,total_evals,segment_evals,probe_evals,"
              "final_f,evals_used,num_segments,num_switches,method_histogram\n";

    std::ofstream usageCsv(a.outPrefix + "_method_usage.csv", std::ios::trunc);
    usageCsv << "problem,variant,method,segments,pct_of_segments,segments_improving,"
                "pct_improving,mean_improvement,total_improvement\n";

    // byProblemVariant[pi][vi] = that (problem,variant)'s per-run outcomes.
    // Kept for the whole run (not flushed away per problem) because the
    // cross-problem aggregation at the end needs every problem's results
    // at once.
    std::vector<std::vector<std::vector<RunOutcome>>> byProblemVariant;
    byProblemVariant.reserve(problemList.size());

    for (const auto& pd : problemList) {
        const std::string& problemName = pd.first;
        const int dim = pd.second;

        std::vector<std::vector<RunOutcome>> byVariant;
        byVariant.reserve(variants.size());

        for (const auto& v : variants) {
            std::map<std::string, MethodUsage> usage;
            // segCsv now needs a leading problem/dim pair on every row;
            // runOneVariant still only knows about its own columns, so its
            // per-segment lines are prefixed here rather than changing its
            // signature into something that has to know about the outer
            // problem loop.
            auto outcomes = runOneVariant(v, a, problemName, dim, cfg_file, segCsv, usage);
            for (int r = 0; r < (int)outcomes.size(); ++r) {
                const RunOutcome& o = outcomes[r];
                sumCsv << csvQuote(problemName) << "," << dim << "," << csvQuote(v.label) << "," << r << ","
                       << a.totalEvals << "," << a.segmentEvals << "," << a.probeEvals << ","
                       << std::setprecision(15) << o.final_f << "," << o.evals_used << ","
                       << o.num_segments << "," << o.num_switches << ","
                       << csvQuote(o.histogram) << "\n";
            }
            sumCsv.flush();
            segCsv.flush();

            long long totalSegs = 0;
            for (const auto& kv : usage) totalSegs += kv.second.segments;
            for (const auto& kv : usage) {
                const MethodUsage& u = kv.second;
                const double pct = totalSegs > 0 ? (100.0 * (double)u.segments / (double)totalSegs) : 0.0;
                const double pctImp = u.segments > 0 ? (100.0 * (double)u.improvedCount / (double)u.segments) : 0.0;
                const double meanImp = u.improvedCount > 0 ? (u.totalImprovement / (double)u.improvedCount) : 0.0;
                usageCsv << csvQuote(problemName) << "," << csvQuote(v.label) << "," << kv.first << ","
                         << u.segments << "," << std::setprecision(4) << std::fixed << pct << ","
                         << u.improvedCount << "," << pctImp << ","
                         << std::defaultfloat << std::setprecision(10) << meanImp << ","
                         << u.totalImprovement << "\n";
            }
            usageCsv.flush();

            byVariant.push_back(std::move(outcomes));
        }
        byProblemVariant.push_back(std::move(byVariant));
    }

    // ---- Comparison statistics -----------------------------------------
    if (k < 2) {
        std::printf("[portfolio] only one variant run -- pass --with-baselines to get "
                     "Friedman/Wilcoxon/ranking tables comparing it against round_robin "
                     "and each single-method baseline.\n");
    } else if (problemList.size() == 1) {
        // Classic single-problem case: run-blocked, N=runs, unsuffixed
        // filenames -- exactly the original Mode 8 statistics output.
        std::vector<std::vector<BlockObservation>> obs((size_t)a.runs);
        for (int r = 0; r < a.runs; ++r) {
            obs[(size_t)r].resize((size_t)k);
            for (int vi = 0; vi < k; ++vi) {
                const RunOutcome& o = byProblemVariant[0][(size_t)vi][(size_t)r];
                obs[(size_t)r][(size_t)vi] = { o.final_f, (double)o.evals_used, (double)o.num_switches };
            }
        }
        writeComparisonStats(a.outPrefix, "", variants, obs, "run");
    } else {
        // Multi-problem: (1) one run-blocked comparison PER problem, same
        // as the single-problem case but with the problem name suffixed on
        // every filename, and (2) the actual cross-problem headline: a
        // PROBLEM-blocked comparison (N=problems) built from each
        // (variant,problem)'s MEAN final_f. Ranks -- not raw f -- are what
        // travels across problems of wildly different scale (ackley's f is
        // O(1-20); sphere's can be O(1e5)): a pooled mean across problems
        // would just be dominated by whichever problem happens to have the
        // largest numbers, so the cross-problem comparison goes through
        // stats::averageRank exactly like Mode 6 ("--rank") already does
        // for --batch output, rather than averaging raw f values directly.
        std::vector<std::vector<BlockObservation>> overallObs(problemList.size());
        std::ofstream aggCsv(a.outPrefix + "_aggregate.csv", std::ios::trunc);
        aggCsv << "problem,dim,variant,mean_f,sd_f,median_f,mean_evals,mean_switches,rank_in_problem\n";

        for (size_t pi = 0; pi < problemList.size(); ++pi) {
            const std::string& problemName = problemList[pi].first;
            const int dim = problemList[pi].second;

            std::vector<std::vector<BlockObservation>> obsRun((size_t)a.runs);
            for (int r = 0; r < a.runs; ++r) {
                obsRun[(size_t)r].resize((size_t)k);
                for (int vi = 0; vi < k; ++vi) {
                    const RunOutcome& o = byProblemVariant[pi][(size_t)vi][(size_t)r];
                    obsRun[(size_t)r][(size_t)vi] = { o.final_f, (double)o.evals_used, (double)o.num_switches };
                }
            }
            // The same problem may appear with several dimensions (e.g.
            // cec2017f1:10 and cec2017f1:30): then the dimension is part of
            // the file name, otherwise the later one overwrote the earlier
            // one's files. A problem listed once keeps the old name.
            int sameName = 0;
            for (const auto& q : problemList) if (q.first == problemName) ++sameName;
            const std::string tag = (sameName > 1) ? problemName + "_d" + std::to_string(dim) : problemName;
            writeComparisonStats(a.outPrefix, "__" + tag, variants, obsRun, "run");

            // Per-(problem,variant) mean, feeding the overall cross-problem
            // comparison, and logged to _aggregate.csv either way.
            std::vector<double> meanF((size_t)k, 0.0), sdF((size_t)k, 0.0), medF((size_t)k, 0.0);
            std::vector<double> meanE((size_t)k, 0.0), meanSw((size_t)k, 0.0);
            overallObs[pi].resize((size_t)k);
            for (int vi = 0; vi < k; ++vi) {
                double sumF = 0.0, sumE = 0.0, sumSw = 0.0;
                std::vector<double> fs;
                for (int r = 0; r < a.runs; ++r) {
                    const RunOutcome& o = byProblemVariant[pi][(size_t)vi][(size_t)r];
                    fs.push_back(o.final_f);
                    sumF += o.final_f; sumE += (double)o.evals_used; sumSw += (double)o.num_switches;
                }
                const double mean = sumF / a.runs;
                double var = 0.0; for (double x : fs) { const double d = x - mean; var += d * d; }
                const double sd = (a.runs > 1) ? std::sqrt(var / (a.runs - 1)) : 0.0;
                std::vector<double> sorted = fs; std::sort(sorted.begin(), sorted.end());
                const double med = (a.runs % 2 == 1) ? sorted[(size_t)(a.runs / 2)]
                                    : 0.5 * (sorted[(size_t)(a.runs/2 - 1)] + sorted[(size_t)(a.runs/2)]);
                meanF[(size_t)vi] = mean; sdF[(size_t)vi] = sd; medF[(size_t)vi] = med;
                meanE[(size_t)vi] = sumE / a.runs; meanSw[(size_t)vi] = sumSw / a.runs;
                overallObs[pi][(size_t)vi] = { mean, meanE[(size_t)vi], meanSw[(size_t)vi] };
            }

            const std::vector<double> rankInProblem = stats::averageRank(meanF);
            for (int vi = 0; vi < k; ++vi) {
                aggCsv << csvQuote(problemName) << "," << dim << "," << csvQuote(variants[(size_t)vi].label) << ","
                       << std::setprecision(15) << meanF[(size_t)vi] << "," << sdF[(size_t)vi] << "," << medF[(size_t)vi] << ","
                       << meanE[(size_t)vi] << "," << meanSw[(size_t)vi] << ","
                       << std::setprecision(4) << rankInProblem[(size_t)vi] << "\n";
            }
        }

        writeComparisonStats(a.outPrefix, "_overall", variants, overallObs, "problem");

        std::printf("[portfolio] %zu problems x %d variants complete. Cross-problem comparison: "
                     "%s_overall_{ranking,friedman,wilcoxon}.csv\n",
                     problemList.size(), k, a.outPrefix.c_str());
    }

    segCsv.flush();
    sumCsv.flush();
    return 0;
}

} // namespace optimsolution

