#pragma once
#include "optimizer.h"
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <cctype>
#include <cmath>

namespace optimsolution {

// Genetic Algorithm with Harmony Search mechanisms (GAHS)
class GAHS : public Optimizer {
public:
    GAHS() = default;
    ~GAHS() override = default;

    std::string methodShortName() const override { return "gahs"; }
    std::string methodFullName()  const override { return "Genetic Algorithm with Harmony Search (GAHS)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        auto to_lower = [](std::string s){ for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; };
        auto trim = [](std::string s){
            size_t a = 0, b = s.size();
            while (a < b && std::isspace((unsigned char)s[a])) ++a;
            while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
            return s.substr(a, b - a);
        };
        auto parse_bool = [&](std::string s, bool fb)->bool{
            s = to_lower(trim(s));
            if (s == "1" || s == "true" || s == "on" || s == "yes") return true;
            if (s == "0" || s == "false" || s == "off" || s == "no") return false;
            return fb;
        };
        auto parse_int = [&](std::string s, int fb)->int{
            s = trim(s);
            if (s.empty()) return fb;
            try {
                size_t pos = 0;
                long v = std::stol(s, &pos);
                if (pos == s.size()) return (int)v;
            } catch (...) {}
            return fb;
        };
        auto parse_double = [&](std::string s, double fb)->double{
            s = trim(s);
            if (s.empty()) return fb;
            try {
                size_t pos = 0;
                double v = std::stod(s, &pos);
                if (pos == s.size() && std::isfinite(v)) return v;
            } catch (...) {}
            return fb;
        };

        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;
        chromosome_count_ = pop_;

        max_generations_ = mc.getInt("max_generations", max_generations_);
        if (max_generations_ < 1) max_generations_ = 200;

        selection_rate_ = mc.getDbl("selection_rate", selection_rate_);
        mutation_rate_  = mc.getDbl("mutation_rate",  mutation_rate_);
        local_rate_     = mc.getDbl("local_rate",     local_rate_);

        if (selection_rate_ < 0.0) selection_rate_ = 0.0;
        if (selection_rate_ > 1.0) selection_rate_ = 1.0;
        if (mutation_rate_  < 0.0) mutation_rate_  = 0.0;
        if (mutation_rate_  > 1.0) mutation_rate_  = 1.0;
        if (local_rate_     < 0.0) local_rate_     = 0.0;
        if (local_rate_     > 1.0) local_rate_     = 1.0;

        tournament_size_ = mc.getInt("tournament_size", tournament_size_);
        if (tournament_size_ < 2) tournament_size_ = 2;

        lsearch_items_ = mc.getInt("lsearch_items", lsearch_items_);
        if (lsearch_items_ < 0) lsearch_items_ = 0;

        lsearch_gens_ = mc.getInt("lsearch_gens", lsearch_gens_);
        if (lsearch_gens_ < 1) lsearch_gens_ = 20;

        selection_method_ = mc.getStr("selection_method", selection_method_);
        crossover_method_ = mc.getStr("crossover_method", crossover_method_);
        mutation_method_  = mc.getStr("mutation_method",  mutation_method_);
        lsearch_method_   = mc.getStr("lsearch_method",   lsearch_method_);
        local_method_     = mc.getStr("local_method",     local_method_);

        for (auto& c : selection_method_) c = (char)std::tolower((unsigned char)c);
        for (auto& c : crossover_method_) c = (char)std::tolower((unsigned char)c);
        for (auto& c : mutation_method_)  c = (char)std::tolower((unsigned char)c);
        for (auto& c : lsearch_method_)   c = (char)std::tolower((unsigned char)c);
        for (auto& c : local_method_)     c = (char)std::tolower((unsigned char)c);

        // HS mechanisms
        HMCR_ = parse_double(mc.getStr("hmcr", mc.getStr("HMCR", std::to_string(HMCR_))), HMCR_);
        PAR_  = parse_double(mc.getStr("par",  mc.getStr("PAR",  std::to_string(PAR_ ))), PAR_ );

        std::string bw_s = mc.getStr("bw",
                           mc.getStr("bandwidth",
                           mc.getStr("bw_scale", std::string{})));
        if (!trim(bw_s).empty()) {
            double v = parse_double(bw_s, bw_scale_);
            if (std::isfinite(v)) {
                if (v <= 1.0) {
                    bw_scale_ = std::max(0.0, v);
                    bw_abs_   = std::numeric_limits<double>::quiet_NaN();
                } else {
                    bw_abs_   = std::max(0.0, v);
                }
            }
        }

        std::string bwmin_s = mc.getStr("bw_min",
                              mc.getStr("bandwidth_min",
                              mc.getStr("bw_min_scale", std::string{})));
        if (!trim(bwmin_s).empty()) {
            double v = parse_double(bwmin_s, bw_min_scale_);
            if (std::isfinite(v)) {
                if (v <= 1.0) {
                    bw_min_scale_ = std::max(0.0, v);
                    bw_min_abs_   = std::numeric_limits<double>::quiet_NaN();
                } else {
                    bw_min_abs_   = std::max(0.0, v);
                }
            }
        }

        int abw = mc.getInt("adaptive_bw", adaptive_bw_ ? 1 : 0);
        abw = parse_int(mc.getStr("adaptive_bw", std::to_string(abw)), abw);
        adaptive_bw_ = (abw != 0);

        int imp = mc.getInt("improvisations",
                  mc.getInt("new_harmonies",
                  mc.getInt("iters_per_gen", improvisations_)));
        if (imp <= 0) imp = parse_int(mc.getStr("improvisations", ""), imp);
        improvisations_ = imp;

        if (HMCR_ < 0.0) HMCR_ = 0.0;
        if (HMCR_ > 1.0) HMCR_ = 1.0;
        if (PAR_  < 0.0) PAR_  = 0.0;
        if (PAR_  > 1.0) PAR_  = 1.0;
        if (bw_scale_ < 0.0) bw_scale_ = 0.0;
        if (bw_min_scale_ < 0.0) bw_min_scale_ = 0.0;

        int flg = mc.getInt("end_local_refine",
                  mc.getInt("final_local",
                  mc.getInt("final.local", end_local_refine_ ? 1 : 0)));
        flg = mc.getInt("end_local_refin", flg);
        std::string flg_s = mc.getStr("end_local_refine",
                            mc.getStr("final_local",
                            mc.getStr("final.local",
                            mc.getStr("end_local_refin", std::string{}))));
        bool fl_enable = parse_bool(flg_s, flg != 0);

        std::string flm = mc.getStr("end_local_method",
                          mc.getStr("final_local_method",
                          mc.getStr("final.method", end_local_method_)));
        flm = to_lower(trim(flm));

        end_local_refine_ = fl_enable;
        end_local_method_ = flm;

        debug_gahs_ = mc.getInt("debug_gahs", debug_gahs_);
        debug_gahs_ = parse_int(mc.getStr("debug_gahs", std::to_string(debug_gahs_)), debug_gahs_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x) {
        double f = prob_->evaluate(x);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }
    bool   inBounds(const Vec& x);
    void   ensureBounds(Vec& x);
    size_t worstIndex() const;
    double bandwidthForDim(int j) const;

    // GA phases
    void calcFitnessArray();
    void selectionSort();
    void crossoverPhase();
    void mutatePhase();

    // HS phase
    void harmonyPhase();

    // Internal local search dispatcher
    void localSearchAt(int pos);
    void localCrossover(int pos);
    void localMutate(int pos);
    void localSiman(int pos);
    void localDE(int pos);

    struct RouletteEntry {
        Vec x;
        double weight{0.0};
    };
    std::vector<RouletteEntry> makeChromosomesForRoulette();
    int selectWithRoulette(const std::vector<RouletteEntry>& roulette);
    int selectWithTournament();

    void makeChildrenUniform(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2);
    void makeChildrenOnePoint(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2);
    void makeChildrenDouble(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2);

    double deltaIter(double y);

private:
    std::vector<Vec> population_;
    std::vector<double> fitness_;
    std::vector<Vec> children_array_;

    int         chromosome_count_{200};
    int         max_generations_{200};
    double      selection_rate_{0.10};
    double      mutation_rate_{0.05};
    double      local_rate_{0.005};
    int         tournament_size_{8};
    int         lsearch_items_{20};
    int         lsearch_gens_{20};
    std::string selection_method_{"roulette"};
    std::string crossover_method_{"double"};
    std::string mutation_method_{"double"};
    std::string lsearch_method_{"none"};
    std::string local_method_{"lbfgs"};

    int generation_{0};

    // HS parameters
    double HMCR_{0.95};
    double PAR_{0.30};
    double bw_scale_{0.01};
    double bw_min_scale_{0.001};
    double bw_abs_{std::numeric_limits<double>::quiet_NaN()};
    double bw_min_abs_{std::numeric_limits<double>::quiet_NaN()};
    bool   adaptive_bw_{true};
    int    improvisations_{-1};

    bool        end_local_refine_{false};
    std::string end_local_method_{};
    int         debug_gahs_{0};
};

} // namespace optimsolution
