#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <ctime>
#include <filesystem>
#include <cstdlib>
#include <random>
#include <map>
#include <set>
#include <tuple>

#ifdef OPTIM_HAVE_OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "problem.h"
#include "optimizer.h"
#include "factory.h"
#include "config.h"
#include "stats.h"
#include "sensitivity.h"
#include "options.h"
#include "fixed_dims.h"
#include "init.h"


static std::filesystem::path findOptimsolutionCfg(const char* argv0)
{
    namespace fs = std::filesystem;

    std::vector<fs::path> start_dirs;
    try { start_dirs.push_back(fs::current_path()); } catch (...) {}

    if (argv0 && *argv0) {
        try {
            fs::path exe = fs::absolute(fs::path(argv0));
            if (!exe.empty()) {
                start_dirs.push_back(exe.parent_path());
            }
        } catch (...) {}
    }

    for (const auto& start : start_dirs) {
        fs::path dir = start;
        for (int depth = 0; depth < 32; ++depth) {
            fs::path cand = dir / "optimsolution.cfg";
            std::error_code ec;
            if (fs::exists(cand, ec) && fs::is_regular_file(cand, ec)) {
                return cand;
            }
            if (!dir.has_parent_path()) break;
            fs::path parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
    }

    // Fallback: preserve historical behavior (relative lookup).
    return fs::path("../optimsolution.cfg");
}

using namespace optimsolution;

namespace optimsolution {
    int runNNLandscapeCli(int argc, char** argv);   // defined in src/mode_nnlandscape.cpp
    int runPortfolioCli(int argc, char** argv);     // defined in src/mode_portfolio.cpp (Mode 8)
    int runComplexityCli(int argc, char** argv, const std::string& cfgFile); // src/mode_complexity.cpp (CEC 2017 T0/T1/T2)
}

// ---------- helpers ----------
static inline double mean(const std::vector<double>& v){
    double s=0.0; for (double x: v) s+=x; return v.empty()?0.0:s/v.size();
}
static inline double stdev(const std::vector<double>& v){
    if (v.size()<2) return 0.0;
    double m=mean(v), a=0.0;
    for(double x:v){double d=x-m;a+=d*d;}
    return std::sqrt(a/(v.size()-1));
}
static void usage(const char* argv0){
    std::cout << "Usage:\n  " << argv0 << " <method> <problem> [dimension]\n"
              << "  " << argv0 << " --multi <method> <problem> [dimension]\n"
              << "  " << argv0 << " --batch <batch_list_file> [output_dir]   (default output_dir: optimsolution_gui_run)\n"
              << "  " << argv0 << " --features <problem> [dimension] [n_samples] [output_file]\n"
              << "  " << argv0 << " --rank <batch_master.csv> [output_prefix] [metric_column]\n"
              << "  " << argv0 << " --nnlandscape --dataset <path.train> --out-prefix <prefix> --out-summary <summary.csv> [options]\n"
              << "                 (train a neural network as a Problem and sample the loss landscape locally around\n"
              << "                  the found minimum; run with --nnlandscape -h for the full option list)\n"
              << "  " << argv0 << " --portfolio --problem <name> [--dim D] --methods m1,m2,m3 --out-prefix <prefix> [options]\n"
              << "                 (Mode 8: mid-run dynamic algorithm switching -- relays between candidate methods\n"
              << "                  across evaluation-budget segments, warm-starting each segment near the current\n"
              << "                  incumbent; run with --portfolio -h for the full option list)\n"
              << "  " << argv0 << " --complexity --methods m1,m2,m3 [--dims 10,30,50] [--problem cec2017f18] [--evals 200000] [--runs 5] [--out complexity.csv]\n"
              << "                 (algorithm complexity T0/T1/T2/C with the CEC 2017 procedure)\n";
}

// ---------------------------------------------------------------------
// Headless batch mode (mode 4): allows scripted, GUI-free execution of
// many (method, problem, dimension) combinations, each producing the
// same *_summary.csv row format already written by the normal single-run
// path (see the [global] csv_summary block below). This is what lets us
// build large reproducible meta-datasets (e.g. for algorithm-selection /
// landscape-analysis studies) on a headless server or cluster, without
// depending on the Qt "Batch run" mode in the GUI.
//
// This function is intentionally self-contained: it does NOT touch the
// existing single-run execution path in main() at all. It only re-invokes
// the same executable (argv0) as a child process, once per combination,
// via std::system(). Per-combination CSV output is routed into a single
// folder via the OPTIMSOLUTION_BATCH_PREFIX environment variable, which
// applyBatchEnvOverrides() below picks up.
// ---------------------------------------------------------------------

// Read env overrides for batch children. Called once, right after
// Config::load(), in the normal single-run path. If OPTIMSOLUTION_BATCH_PREFIX
// is not set, this is a strict no-op and normal (non-batch) behavior is
// completely unaffected.
static void applyBatchEnvOverrides(optimsolution::GeneralOptions& g){
    const char* pfx = std::getenv("OPTIMSOLUTION_BATCH_PREFIX");
    if (!pfx || !*pfx) return;
    g.csv_enable      = true;
    g.csv_summary     = true;
    g.csv_convergence = false;   // keep batch runs light on disk for large grids
    g.csv_prefix      = pfx;
}

struct BatchSpec {
    std::vector<std::string> methods;
    std::vector<std::string> problems;
    std::vector<int>         dims;
    // Explicit (problem, dim) pairs -- used when the GUI generates this file
    // from its per-problem dimension selection (Batch run's "[d=N]" override
    // mechanism). dim==0 means "use that problem's fixed/default dimension".
    // When non-empty, this REPLACES the problems x dims cartesian product
    // below (methods still combine with every pair via a plain cartesian
    // product on methods only).
    std::vector<std::pair<std::string,int>> pairs;
    // Explicit (method, problem, dim) triples -- a fully sparse execution
    // list, bypassing every cartesian product above. Used by the GUI's
    // "load previous experiment, then add/remove methods/problems" pipeline
    // workflow: only the genuinely NEW combinations need to actually run
    // (ones already sitting on disk as a *_summary.csv from an earlier
    // session are skipped GUI-side), so a plain methods x pairs product
    // would needlessly redo already-computed work. An explicitly EMPTY
    // "triples:" line (sawTriplesKey true, triples still empty) is valid
    // and means "run nothing new, just re-merge whatever summary CSVs are
    // currently on disk" -- e.g. after the GUI deleted some removed
    // combinations' summary files.
    std::vector<std::tuple<std::string,std::string,int>> triples;
    bool sawTriplesKey = false;
};

// Minimal parser for the batch list file. Format (one directive per line):
//   methods: pso,de,arq,jso,gwo
//   problems: ackley,rastrigin,sphere
//   dims: 2,5,10,20,30
// -- OR, equivalently/additionally, explicit pairs (one entry per exact
//    problem+dimension combination, e.g. as generated by the GUI):
//   pairs: rastrigin:30, sphere:10, hartmann6:0
// -- OR a fully explicit, sparse (method,problem,dim) list (bypasses every
//    cartesian product above entirely):
//   triples: jso:sphere:10, cmaes:sphere:10, jso:ackley:30
// Blank lines and lines starting with '#' are ignored.
static bool loadBatchSpec(const std::string& path, BatchSpec& out, std::string& err){
    std::ifstream in(path);
    if (!in) { err = "Cannot open batch list file: " + path; return false; }
    std::string line;
    auto trimStr = [](const std::string& s){
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) return std::string();
        return s.substr(a, b - a + 1);
    };
    auto splitCsv = [&](const std::string& s){
        std::vector<std::string> parts;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            std::string t = trimStr(tok);
            if (!t.empty()) parts.push_back(t);
        }
        return parts;
    };
    while (std::getline(in, line)) {
        // strip comments/blank
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trimStr(line.substr(0, colon));
        std::string val = line.substr(colon + 1);
        if (key.empty()) continue;
        for (auto& c : key) c = (char)std::tolower((unsigned char)c);

        if (key == "methods") {
            for (auto& m : splitCsv(val)) out.methods.push_back(toLower(m));
        } else if (key == "problems") {
            for (auto& p : splitCsv(val)) out.problems.push_back(toLower(p));
        } else if (key == "dims") {
            for (auto& d : splitCsv(val)) {
                try { out.dims.push_back(std::max(1, std::stoi(d))); }
                catch (...) { /* ignore malformed entry */ }
            }
        } else if (key == "pairs") {
            for (auto& tok : splitCsv(val)) {
                size_t pc = tok.find(':');
                if (pc == std::string::npos) continue;
                std::string name = toLower(trimStr(tok.substr(0, pc)));
                std::string dimStr = trimStr(tok.substr(pc + 1));
                if (name.empty()) continue;
                int dim = 0;
                try { dim = std::stoi(dimStr); } catch (...) { dim = 0; }
                out.pairs.push_back({name, dim});
            }
        } else if (key == "triples") {
            out.sawTriplesKey = true;
            for (auto& tok : splitCsv(val)) {
                size_t c1 = tok.find(':');
                if (c1 == std::string::npos) continue;
                size_t c2 = tok.find(':', c1 + 1);
                if (c2 == std::string::npos) continue;
                std::string method  = toLower(trimStr(tok.substr(0, c1)));
                std::string problem = toLower(trimStr(tok.substr(c1 + 1, c2 - c1 - 1)));
                std::string dimStr  = trimStr(tok.substr(c2 + 1));
                if (method.empty() || problem.empty()) continue;
                int dim = 0;
                try { dim = std::stoi(dimStr); } catch (...) { dim = 0; }
                out.triples.push_back({method, problem, dim});
            }
        }
    }
    if (out.sawTriplesKey) {
        // "triples:" (even if it parsed to zero usable entries) is a
        // complete, self-sufficient work specification on its own -- no
        // methods:/problems:/pairs: required alongside it.
        return true;
    }
    if (out.methods.empty()) { err = "batch list file defines no methods"; return false; }
    if (out.problems.empty() && out.pairs.empty()) {
        err = "batch list file defines no problems and no pairs";
        return false;
    }
    if (out.dims.empty()) out.dims.push_back(0); // 0 => "use fixed/default dim"
    return true;
}

// Merge every "*_summary.csv" file found directly inside outDir into one
// combined "batch_master.csv" (single shared header, one row per
// method/problem/dim combination). This is the file to hand to the
// Python-side statistics / ELA / meta-learning pipeline.
static bool mergeBatchSummaries(const std::filesystem::path& outDir, std::string& err){
    namespace fs = std::filesystem;
    fs::path masterPath = outDir / "batch_master.csv";
    std::ofstream master(masterPath, std::ios::trunc);
    if (!master) { err = "Cannot create " + masterPath.string(); return false; }

    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(outDir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() <= 12 || name.rfind("_summary.csv") != name.size() - 12) continue;
        // A genuine per-combination summary file always looks like
        // "<method>_<problem>_d<digits>_summary.csv". This additionally
        // excludes the CLI's OWN aggregate outputs that happen to also end
        // in "_summary.csv" (currently just "batch_method_summary.csv") --
        // without this check, that aggregate file (one row per METHOD, with
        // completely different columns: method,groups_used,avg_rank,wins,
        // mean_success_rate) gets swept into this merge as if it were a
        // per-combination row, injecting one garbage row per method with a
        // blank problem/dim into batch_master.csv. Those blank-problem rows
        // then silently group together as a single extra "(problem=blank,
        // dim=blank)" group that Friedman/Wilcoxon may or may not count as
        // "complete" depending on exactly which methods' aggregate rows are
        // present -- which is exactly what made the reported "N (complete
        // groups)" count drift by +/-1 between two otherwise-identical runs.
        const std::string base = name.substr(0, name.size() - 12); // strip "_summary.csv"
        const size_t dPos = base.rfind("_d");
        if (dPos == std::string::npos) continue;
        const std::string dimPart = base.substr(dPos + 2);
        if (dimPart.empty() || !std::all_of(dimPart.begin(), dimPart.end(),
                                             [](unsigned char c){ return std::isdigit(c); })) {
            continue;
        }
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    // Split a raw CSV line on commas. The summary CSVs this reads/writes
    // never contain quoted/embedded commas (every field is a plain number,
    // identifier, or simple word), so a plain split is safe and matches
    // exactly how these files are written elsewhere in this program.
    auto splitCsvLine = [](const std::string& line) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : line) {
            if (c == ',') { out.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        out.push_back(cur);
        return out;
    };

    // The FIRST file (alphabetically) fixes the canonical column order for
    // the whole merged batch_master.csv. Every other file's columns are
    // remapped BY NAME into that order before being written -- not simply
    // concatenated as raw text -- because a summary CSV's column set/order
    // can legitimately differ between files (e.g. one written by an older
    // build, or a method-specific column). Concatenating mismatched rows
    // verbatim under someone else's header silently shifts every field
    // after the first divergence into the WRONG column (a numeric
    // "population" value ending up under "init", "mean_f" ending up under
    // an unrelated column, etc.) -- this was a real, confirmed bug: it
    // corrupted a newly-added method's row whenever its summary CSV's
    // header didn't byte-for-byte match the first file's.
    std::vector<std::string> canonicalHeader;
    int  mergedCount = 0;
    int  remappedFiles = 0;
    int  droppedRows = 0;

    for (auto& f : files) {
        std::ifstream in(f);
        if (!in) continue;
        std::string headerLine, dataLine;
        if (!std::getline(in, headerLine)) continue;
        std::vector<std::string> header = splitCsvLine(headerLine);

        if (canonicalHeader.empty()) {
            canonicalHeader = header;
            master << headerLine << "\n";
        }

        const bool sameOrder = (header == canonicalHeader);
        // Build a "canonical column index -> this file's column index" map
        // once per file (not per row) when the header differs.
        std::vector<int> colMap; // colMap[i] = source column for canonical column i, or -1 if absent
        if (!sameOrder) {
            ++remappedFiles;
            colMap.resize(canonicalHeader.size(), -1);
            for (size_t i = 0; i < canonicalHeader.size(); ++i) {
                for (size_t j = 0; j < header.size(); ++j) {
                    if (header[j] == canonicalHeader[i]) { colMap[i] = (int)j; break; }
                }
            }
        }

        while (std::getline(in, dataLine)) {
            if (dataLine.empty()) continue;
            if (sameOrder) {
                master << dataLine << "\n";
                ++mergedCount;
                continue;
            }
            std::vector<std::string> fields = splitCsvLine(dataLine);
            std::string remapped;
            bool anyMissing = false;
            for (size_t i = 0; i < canonicalHeader.size(); ++i) {
                if (i > 0) remapped += ',';
                const int srcCol = colMap[i];
                if (srcCol >= 0 && (size_t)srcCol < fields.size()) {
                    remapped += fields[(size_t)srcCol];
                } else {
                    anyMissing = true; // canonical column not present in this file -> leave blank
                }
            }
            if (anyMissing) ++droppedRows; // still written (with blanks), just flagged for visibility
            master << remapped << "\n";
            ++mergedCount;
        }
    }
    std::cout << "[batch] merged " << mergedCount << " summary rows from "
              << files.size() << " files into " << masterPath.string() << "\n";
    if (remappedFiles > 0) {
        std::cout << "[batch] note: " << remappedFiles << " summary file(s) had a different "
                  << "column order/set than the first file and were remapped by column name"
                  << (droppedRows > 0 ? " (" + std::to_string(droppedRows) + " row(s) had a column missing from their source file, left blank)" : "")
                  << ".\n";
    }
    return true;
}

static int runBatchCli(int argc, char** argv){
    // argv[0]=exe, argv[1]="--batch", argv[2]=batch_list_file, [argv[3]=output_dir]
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage:\n  %s --batch <batch_list_file> [output_dir]\n", argv[0]);
        return 1;
    }
    namespace fs = std::filesystem;
    std::string listFile = argv[2];
    // Default output folder matches the GUI's own runtime folder
    // ("optimsolution_gui_run"), so headless --batch output lands in the
    // same place the GUI already knows how to browse/import/clean up.
    // Override with an explicit output_dir argument if you want elsewhere.
    fs::path outDir = (argc >= 4) ? fs::path(argv[3]) : fs::path("optimsolution_gui_run");

    BatchSpec spec;
    std::string err;
    if (!loadBatchSpec(listFile, spec, err)) {
        std::fprintf(stderr, "[batch] error: %s\n", err.c_str());
        return 1;
    }

    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        std::fprintf(stderr, "[batch] error: cannot create output dir '%s': %s\n",
                     outDir.string().c_str(), ec.message().c_str());
        return 1;
    }

    fs::path exePath = fs::absolute(fs::path(argv[0]));

    // Build the final flat (method, problem, dim) execution list, from
    // whichever directive was supplied. Priority: explicit triples (fully
    // sparse, each with its own method) > explicit pairs (combined with
    // every method) > problems x dims cartesian (combined with every
    // method). Triples exist specifically for callers that already know
    // exactly which combinations are missing (e.g. the GUI's "load
    // previous experiment, then add/remove" workflow) and don't want to
    // redo a full cartesian product.
    struct FlatTarget { std::string method, problem; int dim; };
    std::vector<FlatTarget> flatTargets;
    if (spec.sawTriplesKey) {
        for (const auto& t : spec.triples)
            flatTargets.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t)});
    } else {
        std::vector<std::pair<std::string,int>> pairTargets;
        if (!spec.pairs.empty()) {
            pairTargets = spec.pairs;
        } else {
            for (const auto& problem : spec.problems)
                for (int dim : spec.dims)
                    pairTargets.push_back({problem, dim});
        }
        for (const auto& method : spec.methods)
            for (const auto& pt : pairTargets)
                flatTargets.push_back({method, pt.first, pt.second});
    }

    long long total = (long long)flatTargets.size();
    long long done = 0;
    long long skipped = 0;
    long long failed = 0;

    std::cout << "[batch] " << total << " (method, problem, dim) combination(s) to run\n";
    std::cout << "[batch] output dir: " << fs::absolute(outDir).string() << "\n";

    for (const auto& t : flatTargets) {
        const std::string& method = t.method;
        const std::string& problem = t.problem;
        const int requestedDim = t.dim;

        auto methodCheck = makeMethod(method);
        if (!methodCheck) {
            std::cout << "[batch] skip: unknown/unregistered method '" << method << "'\n";
            ++skipped;
            continue;
        }

        // Skip problems the current build doesn't actually know about,
        // rather than letting the run fail loudly.
        auto probCheck = makeProblem(problem);
        if (!probCheck) {
            std::cout << "[batch] skip: unknown problem '" << problem << "'\n";
            ++skipped;
            continue;
        }
        int fixed = optimsolution::getFixedDimOrZero(problem);

        int effDim = requestedDim;
        if (fixed > 0) {
            if (requestedDim != 0 && requestedDim != fixed) {
                std::cout << "[batch] skip: '" << problem << "' has fixed dim " << fixed
                          << ", requested " << requestedDim << "\n";
                ++skipped;
                continue;
            }
            effDim = fixed;
        } else if (effDim <= 0) {
            std::cout << "[batch] skip: '" << problem << "' needs an explicit dimension\n";
            ++skipped;
            continue;
        }

        {
            std::ostringstream prefixSs;
            prefixSs << (outDir / (method + "_" + problem + "_d" + std::to_string(effDim))).string();
            const std::string prefix = prefixSs.str();

#if defined(_WIN32)
            _putenv_s("OPTIMSOLUTION_BATCH_PREFIX", prefix.c_str());
#else
            setenv("OPTIMSOLUTION_BATCH_PREFIX", prefix.c_str(), 1);
#endif

            std::ostringstream cmd;
            cmd << "\"" << exePath.string() << "\" " << method << " " << problem;
            if (fixed <= 0) cmd << " " << effDim;

            ++done;
            std::cout << "[batch] (" << done << "/" << total << ") "
                      << method << " " << problem << " d" << effDim << " ... " << std::flush;

            int rc = std::system(cmd.str().c_str());
            if (rc != 0) {
                std::cout << "FAILED (exit " << rc << ")\n";
                ++failed;
            } else {
                std::cout << "ok\n";
            }
        }
    }

#if defined(_WIN32)
    _putenv_s("OPTIMSOLUTION_BATCH_PREFIX", "");
#else
    unsetenv("OPTIMSOLUTION_BATCH_PREFIX");
#endif

    std::cout << "[batch] finished: " << done << " run, " << skipped << " skipped, "
              << failed << " failed\n";

    if (!mergeBatchSummaries(outDir, err)) {
        std::fprintf(stderr, "[batch] error: %s\n", err.c_str());
        return 1;
    }
    return failed > 0 ? 1 : 0;
}
static const char* stopRuleName(StopRule r){
    switch(r){
        case StopRule::NONE:       return "none";
        case StopRule::BSS:        return "bss";
        case StopRule::WSS:        return "wss";
        case StopRule::TSS:        return "tss";
        case StopRule::BOSS:       return "boss";
        case StopRule::SRS:        return "srs";
        case StopRule::IRS:        return "irs";
        case StopRule::DOUBLEBOX:  return "doublebox";
        case StopRule::MAXEVALS:   return "maxevals";
        case StopRule::ALL:        return "all";
        default:                   return "?";
    }
}

// ---------- ANSI colors ----------
static const char* C_RESET   = "\033[0m";
static const char* C_BOLD    = "\033[1m";
static const char* C_LABEL   = "\033[36m";  // bright cyan for all labels

// For success rate:
static const char* C_GOOD    = "\033[32m";  // green (>= 90%)
static const char* C_MED     = "\033[34m";  // blue  (50%–90%)
static const char* C_BAD     = "\033[31m";  // red   (< 50%)

// for highlights in end 
static const char* C_VALUE   = "\033[32m";  // green

// for timings/memory 
static const char* C_WARN    = "\033[33m";  // yellow


static const char* C_PROBVAL = "\033[96m";

// ---------- Tee for capturing convergence lines ----------
class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* a, std::ostream* tap, std::function<void(const std::string&)> onLine)
        : sb1_(a), tap_(tap), onLine_(std::move(onLine)) {}

protected:
    int overflow(int ch) override {
        if (ch == EOF) return !EOF;
        if (sb1_->sputc((char)ch) == EOF) return EOF;
        if (tap_) tap_->put((char)ch);
        line_.push_back((char)ch);
        if (ch == '\n') {
            if (onLine_) onLine_(line_);
            line_.clear();
        }
        return ch;
    }
    int sync() override {
        int r1 = sb1_->pubsync();
        if (tap_) tap_->flush();
        return r1;
    }

private:
    std::streambuf* sb1_;
    std::ostream* tap_;
    std::function<void(const std::string&)> onLine_;
    std::string line_;
};

// --- parse line with [Run k | iter ... | evals ... best_f=...] ---
static bool parse_iter_line(const std::string& s, int& iter, long long& evals, double& f) {
    auto lb = s.find('[');
    auto rb = s.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;
    std::string inside = s.substr(lb+1, rb-lb-1);

    std::vector<std::string> parts;
    {
        std::size_t start = 0;
        while (true) {
            auto pos = inside.find('|', start);
            if (pos == std::string::npos) {
                parts.push_back(inside.substr(start));
                break;
            }
            parts.push_back(inside.substr(start, pos - start));
            start = pos + 1;
        }
    }

    auto clean = [](std::string x){
        x.erase(std::remove_if(x.begin(), x.end(),
                               [](unsigned char c){return std::isspace(c);}), x.end());
        return x;
    };

    std::string iterPart, evalPart;
    for (auto &p : parts) {
        std::string c = clean(p);
        if (c.rfind("iter", 0) == 0)   iterPart = c;
        if (c.rfind("evals", 0) == 0)  evalPart = c;
    }

    if (iterPart.empty() || evalPart.empty()) return false;

    try{
        iter  = std::stoi(iterPart.substr(4));
        evals = std::stoll(evalPart.substr(5));
    }catch(...){ return false; }

    auto pos = s.find("best_f");
    if (pos == std::string::npos) return false;
    pos = s.find('=', pos);
    if (pos == std::string::npos) return false;
    try{
        f = std::stod(s.substr(pos+1));
    }catch(...){ return false; }

    return true;
}

static std::string make_timestamp()
{
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    tmv = *std::localtime(&t);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tmv);
    return std::string(buf);
}

// Read boolean flag from the [global] section of optimsolution.cfg.
// Values: 1/0, true/false, yes/no, on/off (case-insensitive).
static bool readGlobalBoolOption(const std::string& filename,
                                 const std::string& key,
                                 bool defaultVal)
{
    std::ifstream in(filename);
    if (!in) return defaultVal;

    auto trim = [](std::string s) {
        auto notspace = [](int ch){ return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
        s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
        return s;
    };

    bool inGlobal = false;
    std::string line;
    while (std::getline(in, line)) {
        std::string raw = line;
        // cat commends ';' or '#'
        auto sc = raw.find_first_of(";#");
        if (sc != std::string::npos) raw = raw.substr(0, sc);
        raw = trim(raw);
        if (raw.empty()) continue;

        if (raw.front() == '[' && raw.back() == ']') {
            std::string section = raw.substr(1, raw.size()-2);
            std::string sectionLower = toLower(trim(section));
            inGlobal = (sectionLower == "global");
            continue;
        }

        if (!inGlobal) continue;

        auto eq = raw.find('=');
        if (eq == std::string::npos) continue;
        std::string name  = trim(raw.substr(0, eq));
        std::string value = trim(raw.substr(eq+1));

        if (toLower(name) != toLower(key)) continue;

        std::string vlow = toLower(value);
        if (vlow=="1" || vlow=="true" || vlow=="yes" || vlow=="on")  return true;
        if (vlow=="0" || vlow=="false" || vlow=="no"  || vlow=="off") return false;
        return defaultVal;
    }
    return defaultVal;
}

// --------------------------------- MAIN ---------------------------------
// ---------------------------------------------------------------------------
// Multi-objective CLI path -- deliberately self-contained and separate from
// the rest of main()'s single-objective flow below. It never touches conv_
// csv/summ_csv/landscape_csv, Config::load's [global]/[sensitivity] machinery,
// or the Optimizer::run() loop, so Single/Batch/Sensitivity runs are completely
// unaffected by anything in this function.
//
//   optimsolution_cli --multi <method> <problem> [dim]
//
// Population and generations are read from optimsolution.cfg, from the
// chosen method's own section (e.g. [nsga2] population=... / generations=...),
// exactly like every single-objective method reads its own settings --
// they are deliberately NOT CLI arguments here.
//
// Output: "<prefix>_pareto.csv" with columns x1..xD,f1..fK (one row per
// Pareto-optimal solution found).
// ---------------------------------------------------------------------------
static int runMultiObjectiveCli(int argc, char** argv) {
    // argv[0]=exe, argv[1]="--multi", argv[2]=method, argv[3]=problem, [argv[4]=dim]
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage:\n  %s --multi <method> <problem> [dimension]\n",
            argv[0]);
        return 1;
    }

    const std::string method  = toLower(argv[2]);
    const std::string problem = toLower(argv[3]);

    auto moo = makeMultiObjectiveMethod(method);
    if (!moo) {
        std::fprintf(stderr, "Unknown (or not yet multi-objective) method: %s\n", method.c_str());
        return 1;
    }
    auto prob = makeProblem(problem);
    if (!prob) {
        std::fprintf(stderr, "Unknown problem: %s\n", problem.c_str());
        return 1;
    }

    int dim = (argc >= 5) ? std::atoi(argv[4]) : prob->dimension();
    if (dim <= 0) dim = 2;
    prob->init(dim);

    if (prob->numObjectives() < 2) {
        std::fprintf(stderr,
            "Problem '%s' is single-objective (numObjectives()=%d) and has no "
            "multi-objective formulation yet.\n",
            problem.c_str(), prob->numObjectives());
        return 1;
    }

    const std::string cfg_file = findOptimsolutionCfg(argv[0]).string();
    const auto cfg = Config::load(cfg_file, method);
    const int population  = cfg.methodKV.getInt("population", 100);
    const int generations  = cfg.methodKV.getInt("generations", 200);

    moo->setProblem(prob.get());
    moo->setPopulationSize(population);
    moo->setGenerations(generations);
    moo->setSeed(std::random_device{}());

    // Method-specific extra parameters, all optional (defaults live in each
    // method's own class comment); harmless to set even for the methods
    // that don't use a given name, since setParam() just stores it.
    moo->setParam("crossover_prob", cfg.methodKV.getDbl("crossover_prob", 0.9));
    moo->setParam("eta_c",          cfg.methodKV.getDbl("eta_c", 15.0));
    moo->setParam("eta_m",          cfg.methodKV.getDbl("eta_m", 20.0));
    moo->setParam("neighbor_size",  cfg.methodKV.getDbl("neighbor_size", 20.0));
    moo->setParam("de_F",           cfg.methodKV.getDbl("de_F", 0.5));
    moo->setParam("max_replace",    cfg.methodKV.getDbl("max_replace", 2.0));
    moo->setParam("inertia_w",      cfg.methodKV.getDbl("inertia_w", 0.4));
    moo->setParam("c1",             cfg.methodKV.getDbl("c1", 1.5));
    moo->setParam("c2",             cfg.methodKV.getDbl("c2", 1.5));
    moo->setParam("archive_size",   cfg.methodKV.getDbl("archive_size", double(population)));
    const double mutProb = cfg.methodKV.getDbl("mutation_prob", -1.0);
    if (mutProb >= 0.0) moo->setParam("mutation_prob", mutProb);

    std::fprintf(stdout, "[multi] method=%s problem=%s dim=%d objectives=%d population=%d generations=%d\n",
                 method.c_str(), problem.c_str(), dim, prob->numObjectives(), population, generations);

    const MOOResult result = moo->run();

    std::ostringstream ss;
    ss << method << "_" << problem << "_d" << dim << "_multi_" << make_timestamp();
    const std::string prefix = ss.str();
    const std::string outPath = prefix + "_pareto.csv";

    std::ofstream out(outPath, std::ios::trunc);
    if (!out.is_open()) {
        std::fprintf(stderr, "Could not open output file: %s\n", outPath.c_str());
        return 1;
    }
    for (int i = 0; i < dim; ++i) out << "x" << (i + 1) << ",";
    for (int k = 0; k < prob->numObjectives(); ++k) {
        out << "f" << (k + 1);
        if (k + 1 < prob->numObjectives()) out << ",";
    }
    out << "\n";

    for (size_t r = 0; r < result.paretoX.size(); ++r) {
        const Vec& x = result.paretoX[r];
        const Vec& f = result.paretoF[r];
        for (int i = 0; i < dim; ++i) out << std::setprecision(15) << x[i] << ",";
        for (size_t k = 0; k < f.size(); ++k) {
            out << std::setprecision(15) << f[k];
            if (k + 1 < f.size()) out << ",";
        }
        out << "\n";
    }
    out.close();

    std::fprintf(stdout, "[multi] Pareto front size: %zu, evals=%lld, generations=%lld\n",
                 result.paretoX.size(), result.evals, result.generations);
    std::fprintf(stdout, "[multi] Written: %s\n", outPath.c_str());
    return 0;
}

// ---------------------------------------------------------------------
// Mode 5: Landscape sampling (Exploratory Landscape Analysis input).
//
// This mode does NOT compute ELA features itself (skewness, meta-model
// fit, information content, nearest-better-clustering, etc.) — that is
// intentionally left to a companion Python step using an established
// library (pflacco), so we are not re-deriving well-validated statistics
// by hand. What this mode DOES do, and is the only thing it needs to do
// correctly, is:
//   1. sample N points inside a problem's actual bounds, using the exact
//      same Initializer/InitOptions machinery ([init] section of
//      optimsolution.cfg) already used by every optimizer in the
//      framework, so the sample is consistent with how the rest of the
//      framework explores the space (uniform, LHS, Halton, etc.);
//   2. evaluate the problem's real objective function at each point
//      (through Problem::evaluate(), the same public entry point every
//      optimizer uses, so call-counting semantics stay consistent);
//   3. write (x_1..x_d, f) to CSV, alongside the problem's own declared
//      metadata (modality, separability, known global optimum if any),
//      which later gives us an independent, hand-curated cross-check
//      against whatever the data-driven ELA features suggest.
//
// It does not touch any method/optimizer code at all -- Problem and
// Initializer are used exactly as they already exist.
// ---------------------------------------------------------------------
static int runFeaturesCli(int argc, char** argv){
    // argv[0]=exe, argv[1]="--features", argv[2]=problem,
    // [argv[3]=dimension], [argv[4]=n_samples], [argv[5]=output_file]
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage:\n  %s --features <problem> [dimension] [n_samples] [output_file]\n",
            argv[0]);
        return 1;
    }
    namespace fs = std::filesystem; // used only via findOptimsolutionCfg() below

    std::string problem = toLower(argv[2]);
    const std::string cfg_file = findOptimsolutionCfg(argv[0]).string();

    auto prob = makeProblem(problem);
    if (!prob) {
        std::fprintf(stderr, "Unknown problem: %s\n", problem.c_str());
        return 1;
    }

    // -------- Dimension handling: identical rules to the single-run path,
    // so a problem behaves the same way here as everywhere else in the
    // framework (fixed-dim problems reject an explicit dim, and vice versa).
    int fixed = optimsolution::getFixedDimOrZero(problem);
    int dim = 0;
    bool dim_from_cli = (argc >= 4) && std::string(argv[3]) != "0";
    if (dim_from_cli) {
        if (fixed > 0) {
            std::fprintf(stderr,
                "Problem '%s' has a fixed dimension of %d.\n"
                "Run it without a dimension, e.g.:\n"
                "  %s --features %s\n",
                problem.c_str(), fixed, argv[0], problem.c_str());
            return 1;
        }
        dim = std::max(1, std::atoi(argv[3]));
    } else {
        if (fixed > 0) {
            dim = fixed;
        } else {
            std::fprintf(stderr,
                "Error: problem '%s' requires a dimension (e.g. %s --features %s 30)\n",
                problem.c_str(), argv[0], problem.c_str());
            return 1;
        }
    }

    prob->init(dim);
    prob->setMaxEvaluations(std::numeric_limits<long long>::max());

    // -------- Effective dimension.
    // init() is allowed to adjust the REQUESTED dimension: e.g. Powell is
    // defined in blocks of four variables, so init(10/30/50) really builds
    // a 12/32/52-dimensional problem, and that is the dimension the
    // optimizers actually run at. The sampler draws points of the problem's
    // own dimension and evaluate() uses all of their coordinates, so this
    // stage must sample, evaluate AND export exactly that many coordinates.
    // (It used to size the sample and write the x-columns from the
    // requested dimension instead, which for Powell exported only the first
    // 10/30/50 of 12/32/52 coordinates while f was computed from all of
    // them -- the exported design matrix then no longer determined f.)
    // The "dim" column stays the REQUESTED dimension: it is the (problem,
    // dim) key shared with batch_ranked.csv, so downstream joins are
    // unchanged; the effective dimension is recorded in "eff_dim".
    const int effDim = prob->dimension();
    if (effDim != dim) {
        std::fprintf(stderr,
            "[features] note: problem '%s' adjusted the requested dimension %d to %d; "
            "sampling, evaluating and exporting all %d coordinates "
            "(column 'dim' keeps the requested value %d as the join key).\n",
            problem.c_str(), dim, effDim, effDim, dim);
    }

    // -------- Sample count: literature default for ELA is on the order of
    // 50*d-100*d points (Bossek/flacco convention). We default to 100*d,
    // with a hard floor so tiny dimensions still get a usable sample.
    // d is the EFFECTIVE dimension (see above).
    long long n_samples = (argc >= 5) ? std::atoll(argv[4]) : (100LL * effDim);
    if (n_samples < 50) n_samples = 50;

    auto cfg = optimsolution::Config::load(cfg_file, "features");

    // -------- Output path precedence: explicit CLI arg > OPTIMSOLUTION_BATCH_PREFIX
    // (so a future batch driver can route --features output into the same
    // folder as --batch summaries) > auto name in the current directory.
    std::string outPath;
    if (argc >= 6) {
        outPath = argv[5];
    } else if (const char* pfx = std::getenv("OPTIMSOLUTION_BATCH_PREFIX")) {
        if (*pfx) outPath = std::string(pfx) + "_features.csv";
    }
    if (outPath.empty()) {
        outPath = problem + "_d" + std::to_string(dim) + "_features.csv";
    }

    optimsolution::Initializer initializer;
    initializer.configure(cfg.init);

    std::mt19937_64 rng(cfg.g.seed_base);
    std::vector<Vec> X = initializer.samplePopulation(*prob, rng, (int)n_samples);
    if (!X.empty() && (int)X.front().size() != effDim) {
        std::fprintf(stderr,
            "[features] internal error: sampled points have %d coordinates but the problem's "
            "effective dimension is %d.\n", (int)X.front().size(), effDim);
        return 1;
    }

    std::ofstream out(outPath, std::ios::trunc);
    if (!out.is_open()) {
        std::fprintf(stderr, "[features] Could not open output file: %s\n", outPath.c_str());
        return 1;
    }

    out << "problem,dim,n_samples,init_type,modality,separability,"
           "has_known_global_optimum,known_global_optimum,eff_dim,";
    for (int i = 0; i < effDim; ++i) out << "x" << (i + 1) << ",";
    out << "f\n";

    const std::string modality     = prob->modality();
    const std::string separability = prob->separability();
    const bool hasOpt  = prob->hasKnownGlobalOptimum();
    const double fopt  = hasOpt ? prob->globalOptimum() : 0.0;

    long long evaluated = 0;
    for (const auto& x : X) {
        double f = prob->evaluate(x);
        out << problem << "," << dim << "," << n_samples << "," << cfg.init.type << ","
            << modality << "," << separability << ","
            << (hasOpt ? 1 : 0) << "," << std::setprecision(15) << fopt << ","
            << effDim << ",";
        for (int i = 0; i < effDim; ++i) out << std::setprecision(15) << x[i] << ",";
        out << std::setprecision(15) << f << "\n";
        ++evaluated;
    }
    out.close();

    std::fprintf(stdout,
        "[features] problem=%s dim=%d eff_dim=%d init=%s samples=%lld -> %s (evals used: %lld)\n",
        problem.c_str(), dim, effDim, cfg.init.type.c_str(), evaluated, outPath.c_str(), prob->calls());
    return 0;
}

// ---------------------------------------------------------------------
// Mode 6: statistical ranking / labeling ("--rank").
//
// This is the piece that turns raw --batch output into the LABELED
// meta-dataset an algorithm-selection study actually needs: for every
// (problem, dim) group, which method performed best, its rank among all
// methods tried on that group, and whether the overall differences between
// methods are statistically significant (Friedman omnibus test + pairwise
// Wilcoxon signed-rank with Holm correction) -- the standard protocol in
// the metaheuristics-comparison literature.
//
// All statistics below (regularized incomplete gamma / chi-square p-value,
// Friedman chi-square, Wilcoxon signed-rank normal approximation with
// continuity correction and tie handling, Holm-Bonferroni adjustment,
// average-rank and competition-rank with tie handling) were independently
// validated against scipy.stats / pandas on synthetic reference data before
// being wired in here (chi-square vs scipy.stats.chi2.sf, Friedman vs a
// hand-computed chi2/p, Wilcoxon vs scipy.stats.wilcoxon(mode="approx") both
// with and without continuity correction, Holm vs a manual reference
// computation, and both ranking conventions vs pandas .rank()) -- every
// check matched to floating-point precision.
// ---------------------------------------------------------------------

// The statistical machinery this mode uses (chi-square p-value, Friedman
// with tie correction, Wilcoxon signed-rank, Holm adjustment, average /
// competition ranking) now lives in include/stats.h, so that Mode 8
// ("--portfolio") compares its switching policies against their baselines
// with the exact same validated implementations rather than a second copy.
// The call sites below are unchanged: the using-directive brings the names
// into scope exactly as when they were file-local statics here.
using namespace optimsolution::stats;

// Minimal CSV reader for the plain, unquoted, comma-separated files this
// framework itself generates (--batch's batch_master.csv and per-run
// *_summary.csv share this exact simple format -- no embedded commas or
// quoting is ever produced by them, so a straightforward split is correct
// and matches how every other part of this codebase reads its own CSVs).
struct SimpleCsv {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    int col(const std::string& name) const {
        for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return (int)i;
        return -1;
    }
};
// Robust double parser: unlike std::stod, this never throws -- in particular
// it correctly handles legitimate subnormal values (e.g. an optimizer
// converging to something like 4.94065645841e-324, effectively the smallest
// positive double) that glibc's std::stod flags as ERANGE/out_of_range even
// though strtod itself returns the correct value. Unparseable input yields 0.0.
static double parseDoubleSafe(const std::string& s) {
    if (s.empty()) return 0.0;
    return std::strtod(s.c_str(), nullptr);
}

static bool loadSimpleCsv(const std::string& path, SimpleCsv& out, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "Cannot open CSV: " + path; return false; }
    std::string line;
    auto stripCr = [](std::string& s){
        // Defensive: strip a trailing '\r' so this reads correctly regardless
        // of whether the CSV was generated on Windows (CRLF) and later read
        // on a system that doesn't do CRLF->LF translation on input (or vice
        // versa). Without this, whichever column happens to be last would
        // silently carry a trailing '\r' into its value.
        if (!s.empty() && s.back() == '\r') s.pop_back();
    };
    auto splitLine = [&](std::string s){
        stripCr(s);
        std::vector<std::string> parts;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) parts.push_back(tok);
        return parts;
    };
    if (!std::getline(in, line)) { err = "CSV is empty: " + path; return false; }
    out.header = splitLine(line);
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        out.rows.push_back(splitLine(line));
    }
    return true;
}

static int runRankCli(int argc, char** argv) {
    // argv[0]=exe, argv[1]="--rank", argv[2]=batch_master.csv,
    // [argv[3]=output_prefix], [argv[4]=metric_column]
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage:\n  %s --rank <batch_master.csv> [output_prefix] [metric_column]\n"
            "  (metric_column defaults to 'mean_f', lower = better)\n", argv[0]);
        return 1;
    }
    namespace fs = std::filesystem;
    const std::string inputCsv = argv[2];
    const std::string metricCol = (argc >= 5) ? argv[4] : "mean_f";

    std::string outPrefix;
    if (argc >= 4) {
        outPrefix = argv[3];
    } else {
        fs::path p(inputCsv);
        outPrefix = (p.parent_path() / p.stem()).string();
    }

    SimpleCsv csv;
    std::string err;
    if (!loadSimpleCsv(inputCsv, csv, err)) {
        std::fprintf(stderr, "[rank] error: %s\n", err.c_str());
        return 1;
    }
    const int cMethod = csv.col("method");
    const int cProblem = csv.col("problem");
    const int cDim = csv.col("dim");
    const int cMetric = csv.col(metricCol);
    const int cSuccessRate = csv.col("success_rate");
    if (cMethod < 0 || cProblem < 0 || cDim < 0 || cMetric < 0) {
        std::fprintf(stderr,
            "[rank] error: input CSV is missing required column(s) "
            "(need 'method', 'problem', 'dim', '%s')\n", metricCol.c_str());
        return 1;
    }

    // Group row indices by (problem, dim).
    std::map<std::string, std::vector<int>> groups;
    std::vector<std::string> groupKeyOf(csv.rows.size());
    for (size_t i = 0; i < csv.rows.size(); ++i) {
        const auto& row = csv.rows[i];
        std::string key = row[cProblem] + "@" + row[cDim];
        groupKeyOf[i] = key;
        groups[key].push_back((int)i);
    }

    // Per-row rank/is_best (computed within each (problem,dim) group,
    // independent of whether the group is "complete" -- these are always
    // meaningful even for partial coverage).
    std::vector<double> rowRank(csv.rows.size(), 0.0);
    std::vector<int> rowIsBest(csv.rows.size(), 0);
    // Per-method accumulators for the summary table.
    std::map<std::string, double> sumRank, sumSuccessRate;
    std::map<std::string, int> countGroups, countWins, countSuccessRate;
    // Full set of methods seen anywhere, and which groups contain which methods
    // (needed to find the "complete" subset of groups for Friedman/Wilcoxon).
    std::set<std::string> allMethods;

    for (const auto& kv : groups) {
        const auto& idxs = kv.second;
        std::vector<double> vals(idxs.size());
        for (size_t t = 0; t < idxs.size(); ++t) {
            vals[t] = parseDoubleSafe(csv.rows[idxs[t]][cMetric]);
        }
        std::vector<double> avgR = averageRank(vals);
        std::vector<double> compR = competitionRank(vals);
        for (size_t t = 0; t < idxs.size(); ++t) {
            int i = idxs[t];
            rowRank[i] = avgR[t];
            rowIsBest[i] = (compR[t] == 1.0) ? 1 : 0;
            const std::string& m = csv.rows[i][cMethod];
            allMethods.insert(m);
            sumRank[m] += avgR[t];
            countGroups[m] += 1;
            if (rowIsBest[i]) countWins[m] += 1;
            if (cSuccessRate >= 0) {
                sumSuccessRate[m] += parseDoubleSafe(csv.rows[i][cSuccessRate]); countSuccessRate[m] += 1;
            }
        }
    }

    // Write the per-row ranked CSV (original columns + rank + is_best).
    {
        std::ofstream out(outPrefix + "_ranked.csv", std::ios::trunc);
        for (size_t c = 0; c < csv.header.size(); ++c) out << csv.header[c] << ",";
        out << "rank,is_best\n";
        for (size_t i = 0; i < csv.rows.size(); ++i) {
            for (const auto& v : csv.rows[i]) out << v << ",";
            out << std::setprecision(10) << rowRank[i] << "," << rowIsBest[i] << "\n";
        }
    }

    // Write the per-method summary CSV.
    {
        std::ofstream out(outPrefix + "_method_summary.csv", std::ios::trunc);
        out << "method,groups_used,avg_rank,wins" << (cSuccessRate >= 0 ? ",mean_success_rate" : "") << "\n";
        for (const auto& m : allMethods) {
            double avgRank = sumRank[m] / std::max(1, countGroups[m]);
            out << m << "," << countGroups[m] << "," << std::setprecision(6) << avgRank << "," << countWins[m];
            if (cSuccessRate >= 0) {
                double msr = countSuccessRate[m] > 0 ? sumSuccessRate[m] / countSuccessRate[m] : 0.0;
                out << "," << std::setprecision(6) << msr;
            }
            out << "\n";
        }
    }

    // Determine the "complete" subset of groups: those containing every
    // method in allMethods. Friedman/Wilcoxon are only statistically valid
    // on a complete block design, so we restrict to those and say clearly
    // when there aren't enough to run the test, rather than silently
    // computing a number on mismatched data.
    std::vector<std::string> methodList(allMethods.begin(), allMethods.end());
    const int k = (int)methodList.size();
    std::vector<std::vector<double>> completeRanks;   // [group][method] avg rank
    std::vector<std::vector<double>> completeMetrics; // [group][method] raw metric (for Wilcoxon)
    for (const auto& kv : groups) {
        const auto& idxs = kv.second;
        if ((int)idxs.size() != k) continue; // not complete for this group
        std::map<std::string, double> valByMethod, rankByMethod;
        std::vector<double> vals(idxs.size());
        for (size_t t = 0; t < idxs.size(); ++t) vals[t] = parseDoubleSafe(csv.rows[idxs[t]][cMetric]);
        std::vector<double> avgR = averageRank(vals);
        for (size_t t = 0; t < idxs.size(); ++t) {
            valByMethod[csv.rows[idxs[t]][cMethod]] = vals[t];
            rankByMethod[csv.rows[idxs[t]][cMethod]] = avgR[t];
        }
        std::vector<double> rankRow, metricRow;
        bool ok = true;
        for (const auto& m : methodList) {
            if (!valByMethod.count(m)) { ok = false; break; }
            rankRow.push_back(rankByMethod[m]);
            metricRow.push_back(valByMethod[m]);
        }
        if (ok) { completeRanks.push_back(rankRow); completeMetrics.push_back(metricRow); }
    }

    // Friedman omnibus test.
    {
        std::ofstream out(outPrefix + "_friedman.txt", std::ios::trunc);
        auto fr = friedmanTest(completeRanks);
        if (!fr.valid) {
            out << "Friedman test: N/A -- need >=2 methods present in every group and "
                   ">=2 complete groups; found " << completeRanks.size() << " complete group(s) "
                   "out of " << groups.size() << " total, across " << k << " method(s).\n";
        } else {
            out << "Friedman omnibus test\n";
            out << "N (complete problem/dim groups) = " << fr.N << "\n";
            out << "k (methods)                     = " << k << "\n";
            out << "chi^2 = " << std::setprecision(10) << fr.chi2 << "\n";
            out << "df    = " << fr.df << "\n";
            out << "p     = " << std::setprecision(10) << fr.p << "\n";
            out << (fr.p < 0.05 ? "-> significant differences among methods at alpha=0.05\n"
                                : "-> no significant difference detected at alpha=0.05\n");
            out << "\nRank sums (lower = better):\n";
            for (int j = 0; j < k; ++j)
                out << "  " << methodList[j] << ": " << std::setprecision(8) << fr.rankSums[j] << "\n";
        }
    }

    // Pairwise Wilcoxon signed-rank (Holm-adjusted) on the raw metric values
    // across the same complete groups.
    {
        std::vector<std::string> pairA, pairB;
        std::vector<WilcoxonResult> results;
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                std::vector<double> a, b;
                for (const auto& row : completeMetrics) { a.push_back(row[i]); b.push_back(row[j]); }
                results.push_back(wilcoxonSignedRank(a, b));
                pairA.push_back(methodList[i]);
                pairB.push_back(methodList[j]);
            }
        }
        std::vector<double> pvals;
        for (const auto& r : results) pvals.push_back(r.valid ? r.p : 1.0);
        std::vector<double> adjusted = holmAdjust(pvals);

        std::ofstream out(outPrefix + "_wilcoxon.csv", std::ios::trunc);
        out << "method_a,method_b,n_pairs,statistic,z,p_raw,p_holm,significant_at_0.05\n";
        for (size_t idx = 0; idx < results.size(); ++idx) {
            const auto& r = results[idx];
            out << pairA[idx] << "," << pairB[idx] << "," << r.n << ",";
            if (r.valid) {
                out << std::setprecision(8) << r.statistic << "," << r.z << ","
                    << r.p << "," << adjusted[idx] << ","
                    << (adjusted[idx] < 0.05 ? 1 : 0) << "\n";
            } else {
                out << "NA,NA,NA,NA,NA\n";
            }
        }
    }

    std::cout << "[rank] " << csv.rows.size() << " rows, " << groups.size() << " problem/dim group(s), "
              << k << " method(s), " << completeRanks.size() << " complete group(s)\n";
    std::cout << "[rank] wrote " << outPrefix << "_ranked.csv, " << outPrefix << "_method_summary.csv, "
              << outPrefix << "_friedman.txt, " << outPrefix << "_wilcoxon.csv\n";
    return 0;
}

int main(int argc, char** argv){
    // Local searches must read the same settings file as everything else
    // (it used to be opened relative to the current directory only).
    optimsolution::setLocalSearchConfigPath(findOptimsolutionCfg(argc > 0 ? argv[0] : nullptr).string());

    if (argc >= 2 && std::string(argv[1]) == "--multi") {
        return runMultiObjectiveCli(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "--batch") {
        return runBatchCli(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "--features") {
        return runFeaturesCli(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "--rank") {
        return runRankCli(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "--nnlandscape") {
        // Mode 7: neural-network training + local landscape sampling.
        // Implementation lives in src/mode_nnlandscape.cpp (see there for details).
        return optimsolution::runNNLandscapeCli(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "--complexity") {
        // Algorithm complexity with the CEC 2017 procedure (T0, T1, T2, C).
        // Implementation lives in src/mode_complexity.cpp.
        return optimsolution::runComplexityCli(argc, argv, findOptimsolutionCfg(argv[0]).string());
    }
    if (argc >= 2 && std::string(argv[1]) == "--portfolio") {
        // Mode 8: mid-run dynamic algorithm switching / portfolio relay.
        // Implementation lives in src/mode_portfolio.cpp (see there for details).
        return optimsolution::runPortfolioCli(argc, argv);
    }

    if (argc < 3){ usage(argv[0]); return 1; }

    std::string method  = toLower(argv[1]);
    std::string problem = toLower(argv[2]);
    std::string method_full_name = method;  // Default; replaced by the method when available.
    const std::string cfg_file = findOptimsolutionCfg(argv[0]).string();


    auto prob = makeProblem(problem);
    if (!prob) {
        std::fprintf(stderr, "Unknown problem: %s\n", problem.c_str());
        return 1;
    }

    // -------- FIXED-DIM HANDLING --------
    int fixed = optimsolution::getFixedDimOrZero(problem);
    int dim = 0;
    bool dim_from_cli = (argc >= 4);

    if (dim_from_cli) {
        if (fixed > 0) {
            std::fprintf(stderr,
                "Problem '%s' has a fixed dimension of %d.\n"
                "Run it without a dimension, e.g.:\n"
                "  %s %s %s\n",
                problem.c_str(), fixed, argv[0], argv[1], argv[2]);
            return 1;
        }
        dim = std::max(1, std::atoi(argv[3]));
    } else {
        if (fixed > 0) {
            dim = fixed;
        } else {
            std::fprintf(stderr,
                "Error: problem '%s' requires a dimension (e.g. %s %s %s 30)\n",
                problem.c_str(), argv[0], argv[1], argv[2]);
            return 1;
        }
    }
    // ------------------------------------

    prob->init(dim);

    auto cfg = Config::load(cfg_file, method);
    applyBatchEnvOverrides(cfg.g);
    if (cfg.sens.enabled) {
        return run_sensitivity(method, problem, dim, cfg);
    }

    const int RUNS = cfg.g.runs;
    const bool multiRun = (RUNS > 1);

    std::string prefix;
    if (cfg.g.csv_enable) {
        if (!cfg.g.csv_prefix.empty()) {
            prefix = cfg.g.csv_prefix;
        } else {
            std::ostringstream ss;
            ss << method << "_" << problem << "_d" << dim << "_" << make_timestamp();
            prefix = ss.str();
        }
    }

    // CSV files
    std::ofstream conv_csv;
    std::ofstream summ_csv;
    std::ofstream landscape_csv;

    if (cfg.g.csv_enable && cfg.g.csv_convergence) {
        conv_csv.open(prefix + "_convergence.csv", std::ios::trunc);
        conv_csv << "method,problem,dim,run,iter,evals,best_f\n";
    }
    if (cfg.g.csv_enable && cfg.g.csv_summary) {
        summ_csv.open(prefix + "_summary.csv", std::ios::trunc);
        summ_csv
            << "method,problem,dim,population,init,runs,max_evals,max_iters,stop_rule,eps,sim,sumRate,"
            << "success_criterion,min_f,mean_f,sd_f,mean_evals,sd_evals,mean_grad,sd_grad,"
            << "success_count,success_rate,iter_mean_ms_mean,iter_mean_ms_sd,iter_p95_ms_mean,iter_p95_ms_sd,"
            << "mem_peak_mb_mean,mem_peak_mb_sd,inrun_local,final_local\n";
    }
    // 2D-only: best-so-far (x1,x2) trajectory, used by the GUI's "Landscape" tab to
    // animate exploration -> exploitation over a contour of the objective. Only the
    // first run (run==0) is logged (see executeOneRun below) since this is meant as a
    // single illustrative animation, not a per-run comparison.
    if (cfg.g.csv_enable && dim == 2) {
        landscape_csv.open(prefix + "_landscape.csv", std::ios::trunc);
        landscape_csv << "run,iter,evals,best_f,x1,x2\n";
    }

    // stats over runs
    std::vector<double> best_vals;    best_vals.reserve(RUNS);
    std::vector<double> eval_counts;  eval_counts.reserve(RUNS);
    std::vector<double> grad_counts;  grad_counts.reserve(RUNS);
    std::vector<double> iter_mean_ms; iter_mean_ms.reserve(RUNS);
    std::vector<double> iter_p95_ms;  iter_p95_ms.reserve(RUNS);
    std::vector<double> peak_kb;      peak_kb.reserve(RUNS);

    double total_seconds = 0.0;
    int effective_pop = -1;
    long long effective_max_evals = -1;

    int current_run = -1;
    auto onLine = [&](const std::string& line){
        if (!(cfg.g.csv_enable && cfg.g.csv_convergence)) return;
        int it; long long ev; double bf;
        if (parse_iter_line(line, it, ev, bf) && current_run >= 0) {
            conv_csv << method << "," << problem << "," << dim << ","
                     << current_run << "," << it << "," << ev << ","
                     << std::setprecision(15) << bf << "\n";
        }
    };

    TeeBuf* tee = nullptr;
    std::streambuf* old_cout_buf = nullptr;
    if (cfg.g.csv_enable && cfg.g.csv_convergence) {
        tee = new TeeBuf(std::cout.rdbuf(), /*tap*/nullptr, onLine);
        old_cout_buf = std::cout.rdbuf(tee);
    }

    // ------------------------------------------------------------------
    // Execute the independent runs.
    //
    // Two execution paths that produce IDENTICAL numerical results:
    //   * serial   : historical behavior, live console output per iteration.
    //   * parallel : OpenMP over independent runs ([global] parallel_runs=1).
    //                Every run owns its Problem instance, its optimizer and
    //                its RNG seeded with seed_base + run_index — exactly as
    //                in serial mode — so per-run results are bit-identical.
    //                Console text of each run is buffered privately and
    //                replayed in run order afterwards (so the convergence
    //                CSV tee sees a clean, ordered stream).
    // ------------------------------------------------------------------
#ifdef OPTIM_HAVE_OPENMP
    // The classic GKLS generator (gkls250/gkls350/gkls2100) is C code with
    // global mutable state and is NOT thread-safe: those problems always
    // run serially, regardless of the parallel_runs setting.
    const bool problem_is_thread_safe = (problem.rfind("gkls", 0) != 0);
    const bool use_parallel_runs = cfg.g.parallel_runs && RUNS > 1 && problem_is_thread_safe;
    if (cfg.g.parallel_runs && RUNS > 1 && !problem_is_thread_safe) {
        std::cout << "[info] parallel_runs disabled for problem '" << problem
                  << "' (GKLS generator uses non-thread-safe global state); running serially.\n";
    }
#else
    const bool use_parallel_runs = false;
#endif

    // Per-run result slots (index-addressed so ordering never depends on
    // thread scheduling).
    struct RunSlot {
        double fbest = 0.0;
        Vec    xbest;
        double evals = 0.0;
        double grads = 0.0;
        double seconds = 0.0;
        double iter_mean_ms = 0.0;
        double iter_p95_ms  = 0.0;
        double peak_kb      = 0.0;
        bool   has_profile  = false;
        std::string console;      // buffered output (parallel mode only)
    };
    std::vector<RunSlot> slots((size_t)RUNS);

    // NOTE: run_error is declared further down, immediately inside the
    // #ifdef OPTIM_HAVE_OPENMP block where it is actually used -- it used to
    // live here unconditionally, which produced an "unused variable" warning
    // on any build without OpenMP (the only place it's read/written is
    // inside that #ifdef).

    auto executeOneRun = [&](int r, std::ostream* sink, RunSlot& out) -> int {
        auto p = makeProblem(problem);
        if (!p) return 2;

        p->init(dim);
        p->setMaxEvaluations(cfg.g.max_evals);

        // Every method, the local methods gd/lbfgs/bfgs/nm/cg/ncg/adam included,
        // runs through the same makeMethod()+run() path (see
        // src/methods/local_methods.h).
        {
            auto opt = makeMethod(method);
            if (!opt) return 3;

            if (r == 0) {
                method_full_name = opt->methodFullName();
            }

            opt->setProblem(p.get());
            opt->setSeed(cfg.g.seed_base + (unsigned long long)r);
            opt->setGeneralOptions(cfg.g);
            opt->setTermination(cfg.t);
            opt->setInitOptions(cfg.init);
            // NOTE: setEndLocalFromGlobal() must run BEFORE configure().
            // It seeds end_local_refine_/end_local_method_ with the [global]
            // default; configure() then overrides them if the method's own
            // section (e.g. [de], [peao]) sets end_local_refine/end_local_method.
            // Calling them in the opposite order (as before) made configure()'s
            // per-method value get silently clobbered by the global default
            // right afterwards, so end_local_refine=1 in [de]/[peao] never took effect.
            opt->setEndLocalFromGlobal(cfg.g.end_local_refine, toLower(cfg.g.end_local_method));
            opt->configure(cfg.methodKV);
            opt->setRunIndex(r);
            if (sink) opt->setOutputStream(sink);
            // Only the first run's trajectory is logged (a single illustrative
            // animation, not a per-run comparison); safe under parallel_runs too
            // since only whichever thread handles r==0 ever touches this stream.
            if (r == 0 && landscape_csv.is_open()) opt->setLandscapeStream(&landscape_csv);

            auto rr = opt->run();
            // Read initialPopulation() (snapshotted right after init(), see
            // Optimizer::run()) rather than population(): for methods that
            // resolve a literature-recommended, dimension-dependent default
            // inside init() (jSO, mLSHADE-RL, NL-SHADE-LBC, LM-CMA-ES, SFCDE,
            // ...), population() immediately after configure() still holds
            // the global/default value, since it hasn't been overridden yet.
            if (r == 0) effective_pop = opt->initialPopulation();
            if (r == 0) effective_max_evals = opt->maxEvaluations();
            out.fbest   = rr.fbest;
            out.xbest   = rr.xbest;
            out.seconds = rr.seconds;

            const auto& prof = opt->profiler();
            out.iter_mean_ms = prof.meanIterMS();
            out.iter_p95_ms  = prof.p95IterMS();
            out.peak_kb      = static_cast<double>(prof.peakRSSKB());
            out.has_profile  = true;
        }

        out.evals = (double)p->calls();
        out.grads = (double)p->gradCalls();
        return 0;
    };

    if (!use_parallel_runs) {
        // -------- serial path (identical to previous behavior) --------
        for (int r=0; r<RUNS; ++r){
            current_run = r;
            int rc = executeOneRun(r, /*sink*/nullptr, slots[(size_t)r]);
            if (rc != 0) {
                std::cerr << (rc == 2 ? "Unknown problem: " + problem
                                      : "Unknown method: "  + method) << "\n";
                if (old_cout_buf) std::cout.rdbuf(old_cout_buf);
                delete tee; return rc;
            }
        }
    }
#ifdef OPTIM_HAVE_OPENMP
    else {
        // -------- OpenMP parallel path --------
        int run_error = 0; // 2 = unknown problem, 3 = unknown method
        if (cfg.g.omp_threads > 0) omp_set_num_threads(cfg.g.omp_threads);

        #pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < RUNS; ++r) {
            std::ostringstream local_out;
            int rc = executeOneRun(r, &local_out, slots[(size_t)r]);
            if (rc != 0) {
                #pragma omp critical(optim_run_error)
                { if (run_error == 0) run_error = rc; }
            }
            slots[(size_t)r].console = local_out.str();
        }

        if (run_error != 0) {
            std::cerr << (run_error == 2 ? "Unknown problem: " + problem
                                         : "Unknown method: "  + method) << "\n";
            if (old_cout_buf) std::cout.rdbuf(old_cout_buf);
            delete tee; return run_error;
        }

        // Replay buffered console output in run order (feeds the tee so
        // the convergence CSV is populated exactly as in serial mode).
        for (int r = 0; r < RUNS; ++r) {
            current_run = r;
            std::cout << slots[(size_t)r].console;
        }
        std::cout.flush();
    }
#endif

    // Collect per-run statistics in run order (same as serial history).
    for (int r = 0; r < RUNS; ++r) {
        const RunSlot& sl = slots[(size_t)r];
        best_vals.push_back(sl.fbest);
        eval_counts.push_back(sl.evals);
        grad_counts.push_back(sl.grads);
        total_seconds += sl.seconds;
        if (sl.has_profile) {
            iter_mean_ms.push_back(sl.iter_mean_ms);
            iter_p95_ms.push_back(sl.iter_p95_ms);
            peak_kb.push_back(sl.peak_kb);
        }
    }
    if (effective_pop < 0) effective_pop = 1;
    if (effective_max_evals < 0) effective_max_evals = cfg.g.max_evals; // defensive fallback, e.g. RUNS==0

    if (old_cout_buf) std::cout.rdbuf(old_cout_buf);
    delete tee;
    current_run = -1;

    // === statistics ===
    const double min_f  = best_vals.empty() ? 0.0 : *std::min_element(best_vals.begin(), best_vals.end());
    const double tol    = std::max(1e-12, 1e-9 * std::fabs(min_f));
    int success = 0;
    for (double f : best_vals) if (std::fabs(f - min_f) <= tol) ++success;
    const double success_rate = 100.0 * (double)success / (double)RUNS;

    const double mean_f = mean(best_vals);
    const double sd_f   = stdev(best_vals);
    const double mean_e = mean(eval_counts), sd_e = stdev(eval_counts);
    const double mean_g = mean(grad_counts), sd_g = stdev(grad_counts);
    const double m_iter_ms  = mean(iter_mean_ms), sd_iter_ms = stdev(iter_mean_ms);
    const double m_iter95   = mean(iter_p95_ms),  sd_iter95  = stdev(iter_p95_ms);
    const double m_peak_kb  = mean(peak_kb),      sd_peak_kb = stdev(peak_kb);

    double grad_func_ratio = 0.0;
    if (mean_e > 0.0) {
        grad_func_ratio = mean_g / mean_e;
    }

    // quantiles for best_f (median, Q1, Q3)
    double q1_f = min_f, median_f = min_f, q3_f = min_f;
    if (!best_vals.empty()){
        std::vector<double> sorted = best_vals;
        std::sort(sorted.begin(), sorted.end());
        auto quant = [&](double q){
            if (sorted.size()==1) return sorted[0];
            double pos = q * (sorted.size() - 1);
            size_t lo = (size_t)std::floor(pos);
            size_t hi = (size_t)std::ceil(pos);
            double a = sorted[lo], b = sorted[hi];
            return a + (b - a) * (pos - lo);
        };
        q1_f     = quant(0.25);
        median_f = quant(0.50);
        q3_f     = quant(0.75);
    }

    // evals to reach target (successful runs)
    std::vector<double> evals_success;
    for (size_t i=0; i<best_vals.size(); ++i){
        if (std::fabs(best_vals[i] - min_f) <= tol){
            evals_success.push_back(eval_counts[i]);
        }
    }
    double mean_eval_target = mean(evals_success);
    double sd_eval_target   = stdev(evals_success);

    // best / worst run index
    int best_run_idx  = -1;
    int worst_run_idx = -1;
    if (!best_vals.empty()){
        best_run_idx  = (int)(std::min_element(best_vals.begin(), best_vals.end()) - best_vals.begin());
        worst_run_idx = (int)(std::max_element(best_vals.begin(), best_vals.end()) - best_vals.begin());
    }

    std::cout << std::endl;

    // --------- GROUPING ROWS TO SECTIONS ---------
    std::vector<std::pair<std::string,std::string>> rows_general;
    std::vector<std::pair<std::string,std::string>> rows_problem;
    std::vector<std::pair<std::string,std::string>> rows_config;
    std::vector<std::pair<std::string,std::string>> rows_perf;
    std::vector<std::pair<std::string,std::string>> rows_calls;
    std::vector<std::pair<std::string,std::string>> rows_time;
    std::vector<std::pair<std::string,std::string>> rows_local;

    // GENERAL
    rows_general.emplace_back("Method:",            method);
    rows_general.emplace_back("Method full name:",  method_full_name);
    rows_general.emplace_back("Problem:",           problem + " (Dim=" + std::to_string(dim) + ")");

    // PROBLEM INFO
    if (prob) {
        if (!prob->fullName().empty()) {
            rows_problem.emplace_back("Problem full name:", prob->fullName());
        }

        if (!prob->modality().empty() && prob->modality() != "unknown") {
            rows_problem.emplace_back("Problem modality:", prob->modality());
        }

        if (!prob->separability().empty() && prob->separability() != "unknown") {
            rows_problem.emplace_back("Problem separability:", prob->separability());
        }

        if (!prob->category().empty()) {
            rows_problem.emplace_back("Problem category:", prob->category());
        }

        if (prob->hasKnownGlobalOptimum()) {
            {
                std::ostringstream ss;
                ss << prob->globalOptimum();
                rows_problem.emplace_back("Known global optimum f*:", ss.str());
            }
            const Vec& xopt = prob->globalArgOptimum();
            if (!xopt.empty()) {
                std::ostringstream ss;
                ss << "(";
                int D = (int)xopt.size();
                int show = std::min(D, 10);
                for (int i = 0; i < show; ++i) {
                    if (i > 0) ss << ", ";
                    ss << xopt[i];
                }
                if (D > 10) ss << ", ...";
                ss << ")";
                rows_problem.emplace_back("Known global minimizer x*:", ss.str());
            }
        }

        if (prob->hasUniformBounds()) {
            double lo = prob->uniformLowerBound();
            double hi = prob->uniformUpperBound();
            std::ostringstream ss;
            ss << "[" << lo << ", " << hi << "]^" << dim;
            rows_problem.emplace_back("Bounds:", ss.str());
        } else {
            rows_problem.emplace_back("Bounds:", "non-uniform per dimension");
        }
    }

    // EXPERIMENT CONFIG
    rows_config.emplace_back("Population:",        std::to_string(std::max(effective_pop,0)));
    rows_config.emplace_back("Init distribution:", cfg.init.type);
    rows_config.emplace_back("Runs:",              std::to_string(RUNS));
    rows_config.emplace_back("Max evals/run:",     std::to_string(effective_max_evals));
    rows_config.emplace_back("Max iters/run:",     std::to_string(cfg.g.max_iters));
    {
        std::string ruleStr = stopRuleName(cfg.t.rule);
        std::ostringstream ss;
        ss << ruleStr;
        
        if (ruleStr != "maxevals" && ruleStr != "maxiters") {
            ss << " (eps=" << cfg.t.eps << ", sim=" << cfg.t.sim
               << ", sumRate=" << cfg.t.sumRate << ")";
        }
        rows_config.emplace_back("Stop rule:", ss.str());
    }
    {
        std::ostringstream ss;
        ss << "f within " << std::setprecision(12) << tol
           << " of best-of-runs (" << min_f << ")";
        rows_config.emplace_back("Success criterion:", ss.str());
    }

    // PERFORMANCE ON f
    {
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(12) << min_f;
		rows_perf.emplace_back("Best f (min):", ss.str());
	}
    if (multiRun) {
		{
			std::ostringstream ss;
			ss << std::fixed << std::setprecision(12)
			<< mean_f << " (+- " << sd_f << ")";
			rows_perf.emplace_back("Best f (mean +- sd):", ss.str());
		}
        {
            std::ostringstream ss;
            ss << "median=" << median_f
               << " [Q1=" << q1_f << ", Q3=" << q3_f << "]";
            rows_perf.emplace_back("Best f (median, Q1-Q3):", ss.str());
        }
    }

    {
        std::ostringstream ss;
        if (!evals_success.empty()){
            ss << mean_eval_target << " (+- " << sd_eval_target << ")";
        } else {
            ss << "n/a";
        }
        rows_perf.emplace_back("Evals to reach target:", ss.str());
    }

    // CALLS & SUCCESS
    if (multiRun) {
        std::ostringstream ssE; ssE << mean_e << " (+- " << sd_e << ")";
        rows_calls.emplace_back("Evals (mean over runs):", ssE.str());

        std::ostringstream ssG; ssG << mean_g << " (+- " << sd_g << ")";
        rows_calls.emplace_back("Grad calls (mean over runs):", ssG.str());
    } else {
        double e0 = eval_counts.empty() ? 0.0 : eval_counts[0];
        double g0 = grad_counts.empty() ? 0.0 : grad_counts[0];
        {
            std::ostringstream ss; ss << e0;
            rows_calls.emplace_back("Evals:", ss.str());
        }
        {
            std::ostringstream ss; ss << g0;
            rows_calls.emplace_back("Grad calls:", ss.str());
        }
    }

    {
        std::ostringstream ss;
        ss << grad_func_ratio;
        rows_calls.emplace_back("Grad/Func calls ratio (mean):", ss.str());
    }

    {
        std::ostringstream ss;
        ss << success_rate << "%  (" << success << "/" << RUNS << ")";
        rows_calls.emplace_back("Success rate:", ss.str());
    }

    if (best_run_idx >= 0){
        std::ostringstream ss;
        ss << "#" << (best_run_idx+1)
           << " (f=" << best_vals[best_run_idx]
           << ", evals=" << eval_counts[best_run_idx] << ")";
        rows_calls.emplace_back("Best run (index,f,evals):", ss.str());
    }
    if (worst_run_idx >= 0){
        std::ostringstream ss;
        ss << "#" << (worst_run_idx+1)
           << " (f=" << best_vals[worst_run_idx]
           << ", evals=" << eval_counts[worst_run_idx] << ")";
        rows_calls.emplace_back("Worst run (index,f,evals):", ss.str());
    }

    // TIMING & MEMORY
    auto fmt_s = [](double ms){
        std::ostringstream os; os.setf(std::ios::fixed);
        os << std::setprecision(6) << (ms / 1000.0);
        return os.str();
    };
    rows_time.emplace_back(
        "Iter time (mean of means):",
        fmt_s(m_iter_ms) + " s (+- " + fmt_s(sd_iter_ms) + " s)"
    );
    rows_time.emplace_back(
        "Iter time (p95 across runs):",
        fmt_s(m_iter95) + " s (+- " + fmt_s(sd_iter95) + " s)"
    );
    rows_time.emplace_back(
        "Time per run (mean):",
        fmt_s(1000.0 * (RUNS>0 ? total_seconds / RUNS : 0.0)) + " s"
    );
    rows_time.emplace_back(
        "Total wall time (all runs):",
        fmt_s(1000.0 * total_seconds) + " s"
    );
    {
        std::ostringstream ss; ss << (m_peak_kb/1024.0) << " MB (+- " << (sd_peak_kb/1024.0) << " MB)";
        rows_time.emplace_back("Memory peak (mean):", ss.str());
    }

    // LOCAL SEARCH (in-run & in-end)
    std::string inrun_local = "none";
    {
        bool        local_refine = false;
        std::string local_method;
        double      local_rate   = 0.0;

        const auto& kv = cfg.methodKV.kv;

        if (auto it = kv.find("local_refine"); it != kv.end()) {
            try { local_refine = (std::stoi(it->second) != 0); } catch (...) {}
        }
        if (auto it = kv.find("local_method"); it != kv.end()) {
            local_method = it->second;
            for (char& c : local_method) c = (char)std::tolower((unsigned char)c);
        }
        if (auto it = kv.find("local_rate"); it != kv.end()) {
            try { local_rate = std::stod(it->second); } catch (...) { local_rate = 0.0; }
            if (local_rate < 0.0) local_rate = 0.0;
            if (local_rate > 1.0) local_rate = 1.0;
        }

        if ((!local_method.empty()) && (local_rate>0.0 || local_refine)) {
            int pct = (int)std::round(local_rate * 100.0);
            std::ostringstream ss; ss << toLower(local_method) << " (rate=" << pct << "%)";
            inrun_local = ss.str();
            rows_local.emplace_back("Local search (in-run):", ss.str());
        } else {
            rows_local.emplace_back("Local search (in-run):", "none");
        }
    }

    std::string final_local = "none";
    {
        // Effective end-of-run refine = [global] default, overridden by the
        // method's own section if it sets end_local_refine/end_local_method
        // (mirrors the precedence applied in DE::configure()/PEAO::configure()).
        bool        eff_refine = cfg.g.end_local_refine;
        std::string eff_method = cfg.g.end_local_method;

        const auto& kv = cfg.methodKV.kv;
        if (auto it = kv.find("end_local_refine"); it != kv.end()) {
            try { eff_refine = (std::stoi(it->second) != 0); } catch (...) {}
        }
        if (auto it = kv.find("end_local_method"); it != kv.end()) {
            eff_method = it->second;
        }
        eff_method = toLower(eff_method);
        if (eff_method.empty() || eff_method == "none" || eff_method == "off" || eff_method == "0")
            eff_refine = false;

        if (eff_refine && !eff_method.empty()){
            final_local = eff_method;
            rows_local.emplace_back("Local search (in-end):", final_local);
        } else {
            rows_local.emplace_back("Local search (in-end):", "none");
        }
    }

    // --------- COMPUTE label_ FOR ALL GROUPS ---------
    std::size_t label_w = 0;
    auto upd_label_w = [&](const std::vector<std::pair<std::string,std::string>>& rows){
        for (const auto& r : rows) {
            label_w = std::max(label_w, r.first.size());
        }
    };
    upd_label_w(rows_general);
    upd_label_w(rows_problem);
    upd_label_w(rows_config);
    upd_label_w(rows_perf);
    upd_label_w(rows_calls);
    upd_label_w(rows_time);
    upd_label_w(rows_local);
    label_w += 1;

    // --------- GLOBAL summary toggles from [global] of optimsolution.cfg ---------
    bool summary_enable    = readGlobalBoolOption(cfg_file, "summary_enable",           true);
    bool show_general      = readGlobalBoolOption(cfg_file, "summary_show_general",     true);
    bool show_problem      = readGlobalBoolOption(cfg_file, "summary_show_problem",     true);
    bool show_config       = readGlobalBoolOption(cfg_file, "summary_show_config",      true);
    bool show_performance  = readGlobalBoolOption(cfg_file, "summary_show_performance", true);
    bool show_calls        = readGlobalBoolOption(cfg_file, "summary_show_calls",       true);
    bool show_timing       = readGlobalBoolOption(cfg_file, "summary_show_timing",      true);
    bool show_local        = readGlobalBoolOption(cfg_file, "summary_show_local",       true);
    bool show_highlights   = readGlobalBoolOption(cfg_file, "summary_show_highlights",  true);

    std::cout << std::setprecision(12) << std::fixed;

    // helper 
    auto print_row = [&](const std::pair<std::string,std::string>& r){
        const std::string& label = r.first;
        const std::string& value = r.second;

        std::size_t pad = (label_w > label.size()) ? (label_w - label.size()) : 1;

        
        std::cout
            << C_LABEL << label << C_RESET
            << std::string(pad, ' ')
            << C_RESET << value << C_RESET
            << "\n";
    };

    auto print_section_header = [&](const std::string& title){
        const int TOTAL = 58; 
        int inner = (int)title.size() + 2; 
        if (inner > TOTAL) inner = (int)title.size() + 2;
        int hyphens = TOTAL - inner;
        if (hyphens < 0) hyphens = 0;
        int left  = hyphens / 2;
        int right = hyphens - left;

        std::cout << "\n" << C_BOLD << C_LABEL;
        for (int i = 0; i < left; ++i)  std::cout << "-";
        std::cout << " " << title << " ";
        for (int i = 0; i < right; ++i) std::cout << "-";
        std::cout << C_RESET << "\n";
    };

    auto print_section = [&](const std::string& title,
                             const std::vector<std::pair<std::string,std::string>>& rows){
        if (rows.empty()) return;
        print_section_header(title);
        for (const auto& r : rows) {
            print_row(r);
        }
    };

    if (summary_enable) {
        std::cout << C_BOLD << C_LABEL
                  << "\n====================== RUN SUMMARY =======================\n"
                  << C_RESET;

        if (show_general)     print_section("GENERAL",           rows_general);
        if (show_problem)     print_section("PROBLEM INFO",      rows_problem);
        if (show_config)      print_section("EXPERIMENT CONFIG", rows_config);
        if (show_performance) print_section("PERFORMANCE",       rows_perf);
        if (show_calls)       print_section("CALLS & SUCCESS",   rows_calls);
        if (show_timing)      print_section("TIMING & MEMORY",   rows_time);
        if (show_local)       print_section("LOCAL SEARCH",      rows_local);

        if (show_highlights) {
            // SUMMARY HIGHLIGHTS 
            print_section_header("SUMMARY HIGHLIGHTS");
           
            auto print_summary_line = [&](const std::string& label,
                                          const std::string& value){
                std::size_t pad = (label_w > label.size()) ? (label_w - label.size()) : 1;

                std::cout
                    << C_LABEL << label << C_RESET
                    << std::string(pad, ' ')
                    << C_RESET << value << C_RESET
                    << "\n";
            };

            {
                std::ostringstream val;
                val << min_f;
                print_summary_line("Best f (min):", val.str());
            }
            {
                std::ostringstream val;
                val << success_rate << "% (" << success << "/" << RUNS << ")";
                print_summary_line("Success rate:", val.str());
            }
        }
    }

    // CSV summary (same logic as before)
    if (cfg.g.csv_enable && cfg.g.csv_summary) {
        summ_csv
            << method << ","
            << problem << ","
            << dim << ","
            << std::max(effective_pop,0) << ","
            << cfg.init.type << ","
            << RUNS << ","
            << effective_max_evals << ","
            << cfg.g.max_iters << ","
            << stopRuleName(cfg.t.rule) << ","
            << cfg.t.eps << ","
            << cfg.t.sim << ","
            << cfg.t.sumRate << ","
            << "\"f within " << std::setprecision(12) << tol << " of best-of-runs (" << min_f << ")\"" << ","
            << std::setprecision(12) << min_f << ","
            << mean_f << ","
            << sd_f << ","
            << mean_e << ","
            << sd_e << ","
            << mean_g << ","
            << sd_g << ","
            << success << ","
            << success_rate << ","
            << m_iter_ms << ","
            << sd_iter_ms << ","
            << m_iter95 << ","
            << sd_iter95 << ","
            << (m_peak_kb/1024.0) << ","
            << (sd_peak_kb/1024.0) << ","
            << inrun_local << ","
            << final_local
            << "\n";
        summ_csv.flush();
    }

    if (cfg.g.csv_enable && cfg.g.csv_convergence) {
        conv_csv.flush();
    }

    return 0;
}
