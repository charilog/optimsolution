#pragma once

#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <utility>
#include <string>
#include <cctype>   

#include "problem.h"
#include "stop.h"
#include "options.h"
#include "localsearch.h"
#include "profiler.h"

namespace optimsolution
{

    struct Result
    {
        Vec xbest;
        double fbest{std::numeric_limits<double>::infinity()};
        long long evals{0};
        double seconds{0.0};
        int iters{0};
    };

    class Optimizer
    {
    public:
        Optimizer() = default;
        virtual ~Optimizer() = default;
		
		virtual std::string methodShortName() const { return "unknown"; }
		virtual std::string methodFullName()  const { return methodShortName(); }

        const Profiler &profiler() const { return profiler_; }

        void setProblem(Problem *p) { prob_ = p; }
        void setSeed(unsigned long long s) { rng_.seed(s); }
        void setPopulation(int n) { pop_ = (n > 3 ? n : 50); }
        int  population() const { return pop_; }

        void setGeneralOptions(const GeneralOptions &g)
        {
            pop_       = g.population;
            max_iters_ = g.max_iters;
            max_evals_ = g.max_evals;
        }

        void setTermination(const TerminationOptions &t) { topt_ = t; }
        void setInitOptions(const InitOptions &io)       { initopt_ = io; }

        // method-specific options (e.g.: [de], [bho], [nm], [lbfgs]...)
        virtual void configure(const MethodConfig &) {}

        // --- Final local @ end (GLOBAL hook) ---
     
        virtual void setEndLocalFromGlobal(bool enable, const std::string& method)
        {
            std::string m  = method;
            std::string ml = m;
            for (auto &c : ml) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

            if (ml.empty() || ml == "none" || ml == "off" || ml == "0") {
                final_local_enabled_ = false;
                final_local_method_.clear();
            } else {
                final_local_enabled_ = enable;
                if (!final_local_enabled_) final_local_method_.clear();
                else final_local_method_ = m;
            }
        }

        // Getters 
        bool finalLocalEnabled() const        { return final_local_enabled_; }
        const std::string& finalLocalMethod() const { return final_local_method_; }

        
        void setRunIndex(int r) { run_index_ = r; }

        Result run()
        {
            stop_.configure(topt_, pop_);
            profiler_.startRun(prob_ ? prob_->calls() : 0);

            init();

            while (!terminated())
            {
                profiler_.startIter();
                one_iteration();
                profiler_.endIter(prob_ ? prob_->calls() : 0);
                if (terminated()) break;
            }

            end();

            profiler_.endRun(prob_ ? prob_->calls() : 0);

            Result r;
            r.xbest  = best_x_;
            r.fbest  = best_f_;
            r.evals  = (prob_ ? prob_->calls() : 0);
            r.seconds= profiler_.runSeconds();
            r.iters  = iters_;
            return r;
        }

    protected:
        virtual void init() = 0;
        virtual void one_iteration() = 0;

        
        virtual void end() {}

        bool terminated() const
        {
            if (!prob_) return true;
            if (prob_->calls() >= max_evals_) return true;
            if (iters_ >= max_iters_) return true;
            return stopReached_;
        }

        void updateStop(const std::vector<double> &fx)
        {
            stopReached_ = stop_.updateAndCheck(fx);
            ++iters_;
        }

        
        void printBest() const
        {
            const long long evals_now = prob_ ? prob_->calls() : 0;
            std::cout << "[";
            if (run_index_ >= 0) {
                std::cout << "Run " << (run_index_ + 1) << " | ";
            }
            std::cout << "iter " << iters_
                      << " | evals " << evals_now
                      << "] best_f = " << std::setprecision(12) << std::fixed
                      << best_f_ << "\n";
        }

        
        std::pair<Vec, double> localSearch(const std::string &localName, const Vec &x0)
        {
            std::string s = localName;
            for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (s.empty() || s == "none" || s == "off" || s == "0")
            {
                double f = std::numeric_limits<double>::infinity();
                if (prob_) f = prob_->evaluate(x0);
                return { x0, f };
            }
            return runLocalSearch(localName, prob_, rng_, x0);
        }

        
        Problem *prob_{nullptr};
        std::mt19937_64 rng_{std::random_device{}()};

        int        pop_{100};
        int        max_iters_{1000};
        long long  max_evals_{150000};

        StopController      stop_;
        TerminationOptions  topt_;
        bool                stopReached_{false};
        int                 iters_{0};

        InitOptions         initopt_;

        Vec    best_x_;
        double best_f_{std::numeric_limits<double>::infinity()};

        Profiler profiler_;

        // --- effective final-local  ---
        bool        final_local_enabled_{false};
        std::string final_local_method_;

        // --- NEW: index of run for logging ---
        int         run_index_{-1};
    };

} // namespace optimsolution
