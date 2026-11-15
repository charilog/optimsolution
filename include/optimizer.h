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

        // The population size actually in effect once init() has resolved it
        // (e.g. a method-specific, dimension-dependent default computed once
        // the problem is known). Captured right after init() in run(), i.e.
        // before any population-size-reduction schedule (NLPSR/LPSR) has had
        // a chance to shrink pop_. Reporting code should read this instead of
        // population() when it needs the *initial* population size, since
        // population() alone reflects whatever configure() set (which, for
        // methods that defer sizing to init(), is still the global/default
        // value at that point).
        int  initialPopulation() const { return init_population_; }

        // Reporting code (e.g. the batch summary CSV) should read this
        // instead of GeneralOptions::max_evals when it needs the *actual*
        // per-run budget: when "[global] max_evals = <N>xD" is used,
        // max_evals_ ends up scaled by the problem's dimension for every
        // method uniformly (see setGeneralOptions() below), while
        // GeneralOptions::max_evals stays the raw, unscaled config value
        // (N) for every method uniformly.
        long long maxEvaluations() const { return max_evals_; }

        virtual void setGeneralOptions(const GeneralOptions &g)
        {
            pop_       = g.population;
            max_iters_ = g.max_iters;
            max_evals_ = g.max_evals;
            // BUGFIX: "[global] max_evals = <N>xD" (per-dimension budget)
            // used to be honored only by the handful of methods that
            // individually re-implemented this same check in their own
            // setGeneralOptions() override (jso, cmaes, lmcmaes, ea4eig,
            // pso, gwo, sparq, acor, rde, sa) -- every other method simply
            // inherited this base implementation and silently kept the raw
            // "N" from "NxD" (the "xD" suffix was dropped), instead of
            // N * dimension. Applying the same scaling here, once, makes
            // every method honor "NxD" the same way. This runs BEFORE any
            // per-method override's own body (each override calls
            // Optimizer::setGeneralOptions(g) first), so for the 10
            // methods above this is a harmless no-op recomputation of the
            // exact same value; their own override code is left untouched.
            if (g.max_evals_per_dim > 0 && prob_) {
                max_evals_ = g.max_evals_per_dim * (long long)prob_->dimension();
            }
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

        // Output stream for per-iteration progress lines (printBest).
        // Defaults to std::cout. In OpenMP parallel-runs mode each run is
        // given its own buffer so console output never interleaves between
        // threads; buffered text is replayed in run order afterwards.
        void setOutputStream(std::ostream* os) { out_ = (os ? os : &std::cout); }

        // Optional: when set, the best-so-far (x1, x2) trajectory is appended to this
        // stream once per iteration (only when the problem is 2D). Used by the GUI's
        // "Landscape" tab to animate exploration -> exploitation over a 2D contour.
        // Never owned by the Optimizer.
        void setLandscapeStream(std::ostream* os) { landscape_os_ = os; }
        std::ostream& out() const { return *out_; }

        Result run()
        {
            stop_.configure(topt_, pop_);
            profiler_.startRun(prob_ ? prob_->calls() : 0);

            init();
            // Snapshot the population right after initialization, before any
            // population-size-reduction schedule can shrink it. This is the
            // value that should be reported as "the population used", for
            // methods (jSO, mLSHADE-RL, NL-SHADE-LBC, LM-CMA-ES, SFCDE, ...)
            // that only resolve their literature-recommended, dimension-
            // dependent default inside init() rather than in configure().
            init_population_ = pop_;
            logLandscapePoint();

            while (!terminated())
            {
                profiler_.startIter();
                one_iteration();
                profiler_.endIter(prob_ ? prob_->calls() : 0);
                logLandscapePoint();
                if (terminated()) break;
            }

            // From here on localSearch() is the optional end-of-run refinement
            // ([global]/[method] end_local_refine), which by design runs after
            // the main budget; see localSearch() below.
            in_end_phase_ = true;
            end();
            in_end_phase_ = false;

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

        // Fraction of the run budget already used, in [0,1]: the larger of
        // iterations/max_iters and evaluations/max_evals. Schedules (inertia,
        // step sizes, ...) must use this rather than iters_/max_iters_ alone:
        // max_iters is normally just a large safety cap (5,000,000 in the
        // default cfg) while the real budget is max_evals, so iters_/max_iters_
        // stays ~0 for the whole run and the schedule never advances.
        double progressFraction() const
        {
            double p = 0.0;
            if (max_iters_ > 0) p = std::max(p, (double)iters_ / (double)max_iters_);
            if (max_evals_ > 0 && prob_) p = std::max(p, (double)prob_->calls() / (double)max_evals_);
            return std::min(1.0, std::max(0.0, p));
        }

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
            std::ostream& os = *out_;
            os << "[";
            if (run_index_ >= 0) {
                os << "Run " << (run_index_ + 1) << " | ";
            }
            os << "iter " << iters_
               << " | evals " << evals_now
               << "] best_f = " << std::setprecision(12) << std::fixed
               << best_f_ << "\n";
        }

        // Appends one row (run,iter,evals,best_f,x1,x2) to the landscape stream, if set
        // and the problem is exactly 2D. No-op otherwise.
        void logLandscapePoint() const
        {
            if (!landscape_os_ || best_x_.size() != 2) return;
            (*landscape_os_) << run_index_ << "," << iters_ << ","
                             << (prob_ ? prob_->calls() : 0) << ","
                             << std::setprecision(15) << best_f_ << ","
                             << best_x_[0] << "," << best_x_[1] << "\n";
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
            // BUGFIX: in-run local searches (local_rate > 0) used to run with no
            // evaluation cap, so a method could exceed max_evals by an arbitrary
            // amount. They are now capped at the run budget. The end-of-run
            // refinement (called from end()) keeps its previous, uncapped
            // behaviour, since it is an explicit post-budget option.
            const long long cap = in_end_phase_ ? -1LL : max_evals_;
            if (!in_end_phase_ && prob_ && prob_->calls() >= max_evals_) {
                double f = std::numeric_limits<double>::infinity();
                return { x0, f };
            }
            // Island methods call this from inside "#pragma omp parallel for";
            // the local searches use the shared rng_ (Nelder-Mead), so they are
            // serialised to avoid a data race on it.
            std::pair<Vec, double> res;
#ifdef _OPENMP
#pragma omp critical(optimsolution_local_search)
#endif
            {
                res = runLocalSearch(localName, prob_, rng_, x0, -1, cap);
            }
            return res;
        }

        
        Problem *prob_{nullptr};
        std::mt19937_64 rng_{std::random_device{}()};

        int        pop_{100};
        int        init_population_{0}; // pop_ snapshotted right after init(), see run()
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

        // --- output sink for progress lines (never owned) ---
        std::ostream* out_{&std::cout};

        // --- true while end() runs (end-of-run local refinement) ---
        bool          in_end_phase_{false};

        // --- optional landscape trajectory log sink (never owned) ---
        std::ostream* landscape_os_{nullptr};
    };

} // namespace optimsolution
