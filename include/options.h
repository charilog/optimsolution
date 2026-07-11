#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace optimsolution {

struct GeneralOptions {
    int    population   = 100;
    int    max_iters    = 1000;
    long long max_evals = 150000;
    unsigned long long seed_base = 1337ULL;
    int    runs         = 30;

    // --- OpenMP parallelization of independent runs ---
    // parallel_runs: 0 = serial (default, identical to previous behavior),
    //                1 = execute independent runs in parallel with OpenMP.
    // omp_threads:   0 = auto (OpenMP default), >0 = explicit thread count.
    // Results are bit-identical to serial mode: every run keeps its own
    // Problem instance, its own RNG seeded with seed_base + run_index, and
    // console/CSV output is emitted in run order after completion.
    bool   parallel_runs = false;
    int    omp_threads   = 0;


    double success_tol  = 1e-8;


    bool        end_local_refine = false;   // 0/1
    std::string end_local_method = "";      // "lbfgs","bfgs","nm","gd"

    // CSV export — από [global]
    bool        csv_enable      = false;    // master switch
    bool        csv_convergence = true;     //  *_convergence.csv
    bool        csv_summary     = true;     //  *_summary.csv
    std::string csv_prefix      = "";       // if null, auto (method_problem_d<dim>_YYYYmmdd-HHMMSS)
};

enum class StopRule {
    NONE, BSS, WSS, TSS, BOSS, SRS, IRS, DOUBLEBOX, MAXEVALS, ALL
};

struct TerminationOptions {
    StopRule rule     = StopRule::NONE;
    double   eps      = 1e-6;
    int      sim      = 5;
    double   sumRate  = 0.1;
};

struct MethodConfig {
    std::unordered_map<std::string, std::string> kv;
    int    getInt (const std::string& k, int def) const;
    double getDbl (const std::string& k, double def) const;
    bool   getBool(const std::string& k, bool def) const;
    std::string getStr(const std::string& k, const std::string& def) const;
};

struct InitOptions {
    std::string type = "uniform";  // "uniform", "normal", "cauchy", "laplace", "lognormal", "exponential", "beta", "levy", "lhs", "halton", "oppositional"
    std::unordered_map<std::string,std::string> kv;

    int    getInt (const std::string& k, int def) const;
    double getDbl (const std::string& k, double def) const;
    bool   getBool(const std::string& k, bool def) const;
    std::string getStr(const std::string& k, const std::string& def) const;
};

} // namespace optimsolution
