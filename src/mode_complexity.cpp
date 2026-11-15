// =====================================================================
// Mode "--complexity": algorithm complexity with the procedure of the
// CEC 2017 bound-constrained competition (Awad et al., technical report,
// Nanyang Technological University, 2016).
//
//   T0 : time of the fixed test loop below (1,000,000 iterations).
//   T1 : time of <evals> evaluations of the test problem (default F18)
//        alone, in dimension D.
//   T2 : time of one complete run of the method with a budget of <evals>
//        evaluations of the same problem in dimension D; it is measured
//        over <runs> independent runs (default 5) and T2^ is their mean.
//   C  = (T2^ - T1) / T0
//
// Every run is executed serially on the calling thread, with its own
// Problem and Optimizer exactly as in a normal single run, the method
// settings of its [section] in optimsolution.cfg, and the per-iteration
// progress lines discarded so that no console I/O enters the timings.
// T0 and T1 are the median of <reps> repetitions; with --repeat k the
// whole T2^ measurement is repeated k times and the median is reported.
//
// Usage:
//   optimsolution --complexity --methods m1,m2,... [--dims 10,30,50]
//                 [--problem cec2017f18] [--evals 200000] [--runs 5]
//                 [--reps 5] [--repeat 1] [--out complexity.csv]
// =====================================================================
#include "problem.h"
#include "optimizer.h"
#include "factory.h"
#include "config.h"
#include "options.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace optimsolution {

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// T0: the test loop of the CEC 2017 technical report.
double measureT0() {
    const auto a = Clock::now();
    volatile double x = 0.55;
    for (int i = 0; i < 1000000; ++i) {
        double y = x;
        y = y + y; y = y / 2.0; y = y * y; y = std::sqrt(y);
        y = std::log(y); y = std::exp(y); y = y / (y + 2.0);
        x = y;
    }
    return seconds(a, Clock::now());
}

// T1: <evals> evaluations of the problem alone (points drawn beforehand).
double measureT1(const std::string& problem, int D, long long evals) {
    auto p = makeProblem(problem);
    if (!p) return -1.0;
    p->init(D);
    std::mt19937_64 rng(12345);
    std::vector<Vec> pts(1000, Vec(D));
    for (auto& v : pts)
        for (int j = 0; j < D; ++j) {
            std::uniform_real_distribution<double> U(p->lb()[j], p->ub()[j]);
            v[j] = U(rng);
        }
    volatile double sink = 0.0;
    const auto a = Clock::now();
    double s = 0.0;
    for (long long k = 0; k < evals; ++k) s += p->evaluate(pts[(size_t)(k % 1000)]);
    const double t = seconds(a, Clock::now());
    sink = s;
    (void)sink;
    return t;
}

// One complete run of a method, set up exactly as a normal single run.
double measureRun(const std::string& method, const std::string& problem, int D,
                  long long evals, int runIndex, const std::string& cfgFile) {
    auto p = makeProblem(problem);
    auto opt = makeMethod(method);
    if (!p || !opt) return -1.0;
    p->init(D);

    Config cfg = Config::load(cfgFile, method);
    cfg.g.max_evals = evals;          // fixed budget of the protocol ...
    cfg.g.max_evals_per_dim = 0;      // ... not the "<N>xD" budget of the cfg file
    cfg.g.runs = 1;
    p->setMaxEvaluations(evals);

    opt->setProblem(p.get());
    opt->setSeed(cfg.g.seed_base + (unsigned long long)runIndex);
    opt->setGeneralOptions(cfg.g);
    opt->setTermination(cfg.t);
    opt->setInitOptions(cfg.init);
    opt->setEndLocalFromGlobal(cfg.g.end_local_refine, lower(cfg.g.end_local_method));
    opt->configure(cfg.methodKV);
    opt->setRunIndex(runIndex);

    std::ostream discard(nullptr);    // progress lines are dropped, not formatted
    opt->setOutputStream(&discard);

    const auto a = Clock::now();
    auto r = opt->run();
    const double t = seconds(a, Clock::now());
    if (r.evals < evals)
        std::fprintf(stderr, "[complexity] warning: %s D=%d run %d used %lld of %lld evaluations\n",
                     method.c_str(), D, runIndex, (long long)r.evals, evals);
    return t;
}

void usage() {
    std::printf(
        "Usage:\n"
        "  optimsolution --complexity --methods m1,m2,... [--dims 10,30,50]\n"
        "                [--problem cec2017f18] [--evals 200000] [--runs 5]\n"
        "                [--reps 5] [--repeat 1] [--out complexity.csv]\n"
        "Computes T0, T1, T2^ and C = (T2^ - T1)/T0 with the CEC 2017 procedure.\n");
}

} // namespace

int runComplexityCli(int argc, char** argv, const std::string& cfgFile) {
    std::vector<std::string> methods;
    std::vector<int> dims = {10, 30, 50};
    std::string problem = "cec2017f18", outFile = "complexity.csv";
    long long evals = 200000;
    int runs = 5, reps = 5, repeat = 1;

    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(1); }
            return argv[++i];
        };
        if      (k == "--methods") { for (auto& m : splitList(next())) methods.push_back(lower(m)); }
        else if (k == "--dims")    { dims.clear(); for (auto& d : splitList(next())) dims.push_back(std::atoi(d.c_str())); }
        else if (k == "--problem") problem = lower(next());
        else if (k == "--evals")   evals = std::atoll(next().c_str());
        else if (k == "--runs")    runs = std::max(1, std::atoi(next().c_str()));
        else if (k == "--reps")    reps = std::max(1, std::atoi(next().c_str()));
        else if (k == "--repeat")  repeat = std::max(1, std::atoi(next().c_str()));
        else if (k == "--out")     outFile = next();
        else if (k == "-h" || k == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "Unknown option: %s\n", k.c_str()); usage(); return 1; }
    }
    if (methods.empty()) { usage(); return 1; }
    if (!makeProblem(problem)) { std::fprintf(stderr, "Unknown problem: %s\n", problem.c_str()); return 1; }
    for (auto& m : methods)
        if (!makeMethod(m)) { std::fprintf(stderr, "Unknown method: %s\n", m.c_str()); return 1; }

    // T0
    std::vector<double> t0s;
    measureT0();                                         // warm-up
    for (int r = 0; r < reps; ++r) t0s.push_back(measureT0());
    const double T0 = median(t0s);
    std::printf("T0 = %.6f s (median of %d)\n", T0, reps);

    std::ofstream csv(outFile, std::ios::trunc);
    csv << "method,problem,dim,evals,runs,repeat,T0_s,T1_s,T2_s,C,T2_repeats_s\n";
    csv << std::setprecision(9);

    for (int D : dims) {
        std::vector<double> t1s;
        measureT1(problem, D, evals);                    // warm-up
        for (int r = 0; r < reps; ++r) t1s.push_back(measureT1(problem, D, evals));
        const double T1 = median(t1s);
        std::printf("D = %d: T1 = %.6f s (median of %d)\n", D, T1, reps);

        for (const auto& m : methods) {
            std::vector<double> t2means;
            for (int k = 0; k < repeat; ++k) {
                double sum = 0.0;
                for (int r = 0; r < runs; ++r) sum += measureRun(m, problem, D, evals, r, cfgFile);
                t2means.push_back(sum / runs);
            }
            const double T2 = median(t2means);
            const double C = (T2 - T1) / T0;
            std::printf("  %-14s T2^ = %9.4f s   C = %10.2f\n", m.c_str(), T2, C);
            std::fflush(stdout);
            csv << m << ',' << problem << ',' << D << ',' << evals << ',' << runs << ',' << repeat << ','
                << T0 << ',' << T1 << ',' << T2 << ',' << C << ',';
            for (size_t k = 0; k < t2means.size(); ++k) csv << (k ? ";" : "") << t2means[k];
            csv << '\n';
            csv.flush();
        }
    }
    std::printf("Written: %s\n", outFile.c_str());
    return 0;
}

} // namespace optimsolution
