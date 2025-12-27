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
