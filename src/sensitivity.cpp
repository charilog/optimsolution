#ifdef OPTIM_HAVE_OPENMP
#include <omp.h>
#endif
#include "sensitivity.h"
#include "factory.h"
#include "optimizer.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cctype>

namespace optimsolution
{

    static std::string to_lower_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return (char)std::tolower(c); });
        return s;
    }
    static std::string to_upper_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return (char)std::toupper(c); });
        return s;
    }

    static double meanv(const std::vector<double> &v)
    {
        if (v.empty())
            return 0.0;
        double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / (double)v.size();
    }
    static double stdevv(const std::vector<double> &v)
    {
        if (v.size() < 2)
            return 0.0;
        double m = meanv(v);
        double acc = 0.0;
        for (double x : v)
        {
            double d = x - m;
            acc += d * d;
        }
        return std::sqrt(acc / (double)(v.size() - 1));
    }

    // Helper to enumerate the cartesian grid of parameter values
    static void enumerate_grid(const std::vector<std::string> &params,
                               const std::unordered_map<std::string, std::vector<double>> &values,
                               std::vector<std::unordered_map<std::string, double>> &out)
    {
        if (params.empty())
        {
            out.push_back({});
            return;
        }
        // recursive product
        std::vector<size_t> idx(params.size(), 0);
        std::vector<size_t> sizes(params.size(), 0);
        for (size_t i = 0; i < params.size(); ++i)
        {
            auto it = values.find(params[i]);
            if (it == values.end() || it->second.empty())
                return;
            sizes[i] = it->second.size();
        }
        // total combinations
        size_t total = 1;
        for (size_t s : sizes)
            total *= s;
        out.reserve(out.size() + total);

        for (size_t t = 0; t < total; ++t)
        {
            std::unordered_map<std::string, double> comb;
            size_t carry = t;
            for (size_t i = 0; i < params.size(); ++i)
            {
                size_t sz = sizes[i];
                size_t pos = carry % sz;
                carry /= sz;
                comb[params[i]] = values.at(params[i])[pos];
            }
            out.push_back(std::move(comb));
        }
    }

    int run_sensitivity(const std::string &method,
                        const std::string &problem,
                        int dim,
                        const Config &cfg)
    {
        if (!cfg.sens.enabled)
        {
            std::cerr << "[sensitivity] Not enabled. Nothing to do.\n";
            return 0;
        }
        if (cfg.sens.params.empty())
        {
            std::cerr << "[sensitivity] No params specified (sensitivity.params). Nothing to do.\n";
            return 1;
        }

        // Build parameter combinations
        std::vector<std::unordered_map<std::string, double>> combos;
        enumerate_grid(cfg.sens.params, cfg.sens.values, combos);
        if (combos.empty())
        {
            std::cerr << "[sensitivity] No valid combinations.\n";
            return 2;
        }

        std::ofstream ofs(cfg.sens.output_csv);
        if (!ofs)
        {
            std::cerr << "[sensitivity] Cannot open output CSV: " << cfg.sens.output_csv << "\n";
            return 3;
        }

        // CSV header
        ofs << "method,problem,dim";
        for (const auto &p : cfg.sens.params)
            ofs << "," << p;
        ofs << ",runs,mean_f,stdev_f,min_f,mean_evals,stdev_evals,success_rate\n";

        std::cout << "\n=== Sensitivity Analysis (" << method << " on " << problem
                  << ", D=" << dim << ") ===\n";

        for (const auto &comb : combos)
        {
            // Prepare a copy of methodKV with overrides
            MethodConfig methodKV = cfg.methodKV;
            for (const auto &kv : comb)
            {
                const std::string k_l = to_lower_copy(kv.first);
                const std::string k_u = to_upper_copy(kv.first);
                const std::string val = std::to_string(kv.second);

                
                methodKV.kv[kv.first] = val; // (lower names)
                methodKV.kv[k_l] = val;      // lower
                methodKV.kv[k_u] = val;      // UPPER
            }

            const int RUNS = cfg.g.runs;
            std::vector<double> best_vals((size_t)RUNS, 0.0);
            std::vector<double> eval_counts((size_t)RUNS, 0.0);
            int success = 0;

            int effective_pop = -1;
            int sens_error = 0; // 4 = unknown problem, 5 = unknown method

            // One sensitivity run: fully independent (own Problem, own
            // optimizer, deterministic per-run seed). Results are written
            // into index-addressed slots so serial and OpenMP-parallel
            // execution produce identical numbers.
            auto runOne = [&](int r, std::ostream* sink) -> int {
                auto prob = makeProblem(problem);
                if (!prob) return 4;
                prob->init(dim);
                prob->setMaxEvaluations(cfg.g.max_evals);

                auto opt = makeMethod(method);
                if (!opt) return 5;

                opt->setProblem(prob.get());
                opt->setSeed(cfg.g.seed_base + (unsigned long long)r);
                opt->setGeneralOptions(cfg.g);
                opt->setTermination(cfg.t);
                opt->setInitOptions(cfg.init);
                opt->configure(methodKV);
                // In parallel mode each run writes its progress lines into a
                // private discarded buffer so console output cannot interleave;
                // serial mode keeps the historical live output on stdout.
                if (sink) opt->setOutputStream(sink);

                if (r == 0) effective_pop = opt->population();

                auto res = opt->run();
                best_vals[(size_t)r]   = res.fbest;
                eval_counts[(size_t)r] = (double)res.evals;
                return (res.fbest <= cfg.g.success_tol) ? -1 : 0; // -1 marks success
            };

#ifdef OPTIM_HAVE_OPENMP
            const bool par = cfg.g.parallel_runs && RUNS > 1 && (problem.rfind("gkls", 0) != 0);
#else
            const bool par = false;
#endif
            if (!par)
            {
                for (int r = 0; r < RUNS; ++r)
                {
                    int rc = runOne(r, nullptr);
                    if (rc > 0)
                    {
                        std::cerr << (rc == 4 ? "Unknown problem: " + problem
                                              : "Unknown method: "  + method) << "\n";
                        return rc;
                    }
                    if (rc == -1) ++success;
                }
            }
#ifdef OPTIM_HAVE_OPENMP
            else
            {
                if (cfg.g.omp_threads > 0) omp_set_num_threads(cfg.g.omp_threads);
                int succ_count = 0;
                #pragma omp parallel for schedule(dynamic) reduction(+:succ_count)
                for (int r = 0; r < RUNS; ++r)
                {
                    std::ostringstream sink; // discarded per-run progress text
                    int rc = runOne(r, &sink);
                    if (rc > 0)
                    {
                        #pragma omp critical(optim_sens_error)
                        { if (sens_error == 0) sens_error = rc; }
                    }
                    else if (rc == -1) ++succ_count;
                }
                if (sens_error != 0)
                {
                    std::cerr << (sens_error == 4 ? "Unknown problem: " + problem
                                                  : "Unknown method: "  + method) << "\n";
                    return sens_error;
                }
                success = succ_count;
            }
#endif

            double m_f = meanv(best_vals);
            double sd_f = stdevv(best_vals);
            double min_f = best_vals.empty() ? 0.0 : *std::min_element(best_vals.begin(), best_vals.end());
            double m_e = meanv(eval_counts);
            double sd_e = stdevv(eval_counts);
            double succ_rate = 100.0 * (double)success / (double)RUNS;

            // Console row
            std::ostringstream tag;
            bool first = true;
            for (const auto &p : cfg.sens.params)
            {
                auto it = comb.find(p);
                if (!first)
                    tag << ", ";
                tag << p << "=" << (it == comb.end() ? 0.0 : it->second);
                first = false;
            }
            std::cout << "  [" << tag.str() << "]  "
                      << "mean=" << std::setprecision(10) << m_f
                      << "  min=" << min_f
                      << "  evals=" << (long long)m_e
                      << "  succ=" << std::fixed << std::setprecision(1) << succ_rate << "%\n";

            // CSV row
            ofs << method << "," << problem << "," << dim;
            for (const auto &p : cfg.sens.params)
            {
                auto it = comb.find(p);
                ofs << "," << (it == comb.end() ? 0.0 : it->second);
            }
            ofs << "," << RUNS
                << "," << std::setprecision(10) << m_f
                << "," << sd_f
                << "," << min_f
                << "," << m_e
                << "," << sd_e
                << "," << std::fixed << std::setprecision(3) << succ_rate
                << "\n";
        }

        std::cout << "\nSaved sensitivity CSV to: " << cfg.sens.output_csv << "\n";
        return 0;
    }

} // namespace optimsolution
