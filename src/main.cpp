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

#include "utils.h"
#include "problem.h"
#include "optimizer.h"
#include "factory.h"
#include "config.h"
#include "sensitivity.h"
#include "options.h"
#include "fixed_dims.h"


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
    std::pair<Vec,double> localGD   (Problem* prob, std::mt19937_64& rng, const Vec& x0);
    std::pair<Vec,double> localLBFGS(Problem* prob, std::mt19937_64& rng, const Vec& x0);
    std::pair<Vec,double> localBFGS (Problem* prob, std::mt19937_64& rng, const Vec& x0);
    std::pair<Vec,double> localNM   (Problem* prob, std::mt19937_64& rng, const Vec& x0);
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
    std::cout << "Usage:\n  " << argv0 << " <method> <problem> [dimension]\n";
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
static inline bool isLocalMethod(const std::string& m){
    std::string s = toLower(m);
    return (s=="gd" || s=="lbfgs" || s=="bfgs" || s=="nm");
}
static Vec makeInitialX0(int dim){ return Vec((size_t)dim, 0.0); }

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
int main(int argc, char** argv){
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

    // stats over runs
    std::vector<double> best_vals;    best_vals.reserve(RUNS);
    std::vector<double> eval_counts;  eval_counts.reserve(RUNS);
    std::vector<double> grad_counts;  grad_counts.reserve(RUNS);
    std::vector<double> iter_mean_ms; iter_mean_ms.reserve(RUNS);
    std::vector<double> iter_p95_ms;  iter_p95_ms.reserve(RUNS);
    std::vector<double> peak_kb;      peak_kb.reserve(RUNS);

    double total_seconds = 0.0;
    int effective_pop = -1;

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

    for (int r=0; r<RUNS; ++r){
        current_run = r;

        auto p = makeProblem(problem);
        if (!p){
            std::cerr<<"Unknown problem: "<<problem<<"\n";
            if (old_cout_buf) std::cout.rdbuf(old_cout_buf);
            delete tee; return 2;
        }

        p->init(dim);
        p->setMaxEvaluations(cfg.g.max_evals);

        std::mt19937_64 rng(cfg.g.seed_base + (unsigned long long)r);

        double fbest = 0.0;
        Vec    xbest;

        if (isLocalMethod(method)){
            Vec x0 = makeInitialX0(dim);
            std::pair<Vec,double> res;
            if (method=="gd")         res = localGD    (p.get(), rng, x0);
            else if (method=="lbfgs") res = localLBFGS (p.get(), rng, x0);
            else if (method=="bfgs")  res = localBFGS  (p.get(), rng, x0);
            else                      res = localNM    (p.get(), rng, x0);

            xbest = std::move(res.first);
            fbest = res.second;
            if (effective_pop < 0) effective_pop = 1;
        } else {
            auto opt = makeMethod(method);
            if (!opt){
                std::cerr<<"Unknown method: "<<method<<"\n";
                if (old_cout_buf) std::cout.rdbuf(old_cout_buf);
                delete tee; return 3;
            }

            if (r == 0) {
                method_full_name = opt->methodFullName();
            }

            opt->setProblem(p.get());
            opt->setSeed(cfg.g.seed_base + (unsigned long long)r);
            opt->setGeneralOptions(cfg.g);
            opt->setTermination(cfg.t);
            opt->setInitOptions(cfg.init);
            opt->configure(cfg.methodKV);
            opt->setEndLocalFromGlobal(cfg.g.end_local_refine, toLower(cfg.g.end_local_method));
            opt->setRunIndex(r);

            if (effective_pop < 0) effective_pop = opt->population();

            auto rr = opt->run();
            fbest   = rr.fbest;
            xbest   = rr.xbest;

            total_seconds += rr.seconds;

            const auto& prof = opt->profiler();
            iter_mean_ms.push_back(prof.meanIterMS());
            iter_p95_ms.push_back(prof.p95IterMS());
            peak_kb.push_back(static_cast<double>(prof.peakRSSKB()));
        }

        best_vals.push_back(fbest);
        eval_counts.push_back((double)p->calls());
        grad_counts.push_back((double)p->gradCalls());
    }

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
    rows_config.emplace_back("Max evals/run:",     std::to_string(cfg.g.max_evals));
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
    if (cfg.g.end_local_refine && !cfg.g.end_local_method.empty()){
        final_local = toLower(cfg.g.end_local_method);
        rows_local.emplace_back("Local search (in-end):", final_local);
    } else {
        rows_local.emplace_back("Local search (in-end):", "none");
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
            << cfg.g.max_evals << ","
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
            << "none" << ","
            << "none"
            << "\n";
        summ_csv.flush();
    }

    if (cfg.g.csv_enable && cfg.g.csv_convergence) {
        conv_csv.flush();
    }

    return 0;
}
