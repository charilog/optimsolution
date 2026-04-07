#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <string>
#include <cctype>
#include <cmath>

namespace optimsolution {

// Krill Herd Algorithm (KH)
// Continuous, population-based algorithm with three motion components:
// (1) induced motion by other krill, (2) foraging motion, (3) physical diffusion.
// This implementation is a robust, practical variant designed for continuous benchmarks.
class KH : public Optimizer {
public:
    KH() = default;
    ~KH() override = default;

    std::string methodShortName() const override { return "KH"; }
    std::string methodFullName()  const override { return "Krill Herd Algorithm (KH)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        auto to_lower = [](std::string s){ for (auto &c: s) c = (char)std::tolower((unsigned char)c); return s; };
        auto trim = [](std::string s){
            size_t a = 0, b = s.size();
            while (a < b && std::isspace((unsigned char)s[a])) ++a;
            while (b > a && std::isspace((unsigned char)s[b-1])) --b;
            return s.substr(a, b-a);
        };
        auto parse_bool = [&](std::string s, bool fb)->bool{
            s = to_lower(trim(s));
            if (s=="1"||s=="true"||s=="on"||s=="yes") return true;
            if (s=="0"||s=="false"||s=="off"||s=="no") return false;
            return fb;
        };
        auto parse_int = [&](std::string s, int fb)->int{
            s = trim(s); if (s.empty()) return fb;
            try{ size_t pos=0; long v=std::stol(s,&pos); if(pos==s.size()) return (int)v; }catch(...) {}
            return fb;
        };
        auto parse_double = [&](std::string s, double fb)->double{
            s = trim(s); if (s.empty()) return fb;
            try{ size_t pos=0; double v=std::stod(s,&pos); if(pos==s.size() && std::isfinite(v)) return v; }catch(...) {}
            return fb;
        };

        // In-run local (aliases)
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local",
                         local_method_)));
        lm = to_lower(trim(lm));

        double lr = mc.getDbl("local_rate",
                      mc.getDbl("local.rate",
                      mc.getDbl("inrun_rate",
                      local_rate_)));
        lr = parse_double(mc.getStr("local_rate",
                   mc.getStr("local.rate",
                   mc.getStr("inrun_rate", std::to_string(lr)))), lr);

        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        if (lm == "none" || lm == "off" || lm == "0") {
            local_method_.clear();
            local_rate_ = 0.0;
        } else {
            local_method_ = lm;
            local_rate_   = lr;
        }

        // Population override (aliases)
        int p = -1;
        p = mc.getInt("population",
            mc.getInt("Population",
            mc.getInt("pop",
            mc.getInt("Pop", -1))));
        if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("Population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("pop", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("Pop", ""), -1);
        if (p >= 4) {
            pop_override_ = p;
            this->setPopulation(pop_override_);
        }

        // KH parameters
        nmax_ = parse_double(mc.getStr("nmax", std::to_string(nmax_)), mc.getDbl("nmax", nmax_));
        vf_   = parse_double(mc.getStr("vf",   std::to_string(vf_)),   mc.getDbl("vf",   vf_));
        dmax0_= parse_double(mc.getStr("dmax0",std::to_string(dmax0_)),mc.getDbl("dmax0",dmax0_));
        dt_   = parse_double(mc.getStr("dt",   std::to_string(dt_)),   mc.getDbl("dt",   dt_));
        wn_   = parse_double(mc.getStr("wn",   std::to_string(wn_)),   mc.getDbl("wn",   wn_));
        wf_   = parse_double(mc.getStr("wf",   std::to_string(wf_)),   mc.getDbl("wf",   wf_));
        c_best_=parse_double(mc.getStr("c_best",std::to_string(c_best_)),mc.getDbl("c_best",c_best_));
        c_food_=parse_double(mc.getStr("c_food",std::to_string(c_food_)),mc.getDbl("c_food",c_food_));
        neighbor_k_ = mc.getInt("neighbor_k", neighbor_k_);
        neighbor_k_ = parse_int(mc.getStr("neighbor_k", std::to_string(neighbor_k_)), neighbor_k_);
        greedy_ = parse_bool(mc.getStr("greedy", std::to_string(greedy_ ? 1 : 0)), greedy_);

        if (!std::isfinite(nmax_)  || nmax_  <= 0.0) nmax_  = 0.01;
        if (!std::isfinite(vf_)    || vf_    <= 0.0) vf_    = 0.02;
        if (!std::isfinite(dmax0_) || dmax0_ < 0.0)  dmax0_ = 0.005;
        if (!std::isfinite(dt_)    || dt_    <= 0.0) dt_    = 1.0;
        if (!std::isfinite(wn_)) wn_ = 0.9;
        if (!std::isfinite(wf_)) wf_ = 0.8;
        if (wn_ < 0.0) wn_ = 0.0; if (wn_ > 1.0) wn_ = 1.0;
        if (wf_ < 0.0) wf_ = 0.0; if (wf_ > 1.0) wf_ = 1.0;
        if (!std::isfinite(c_best_) || c_best_ < 0.0) c_best_ = 2.0;
        if (!std::isfinite(c_food_) || c_food_ < 0.0) c_food_ = 1.0;
        if (neighbor_k_ < 1) neighbor_k_ = 5;

        // Final local at end (aliases; accepts typo end_local_refin)
        int flg = mc.getInt("end_local_refine",
                  mc.getInt("final_local",
                  mc.getInt("final.local",
                  end_local_refine_ ? 1 : 0)));
        flg = mc.getInt("end_local_refin", flg);
        std::string flg_s = mc.getStr("end_local_refine",
                            mc.getStr("final_local",
                            mc.getStr("final.local",
                            mc.getStr("end_local_refin", std::string{}))));
        bool fl_enable = parse_bool(flg_s, flg != 0);

        std::string flm = mc.getStr("end_local_method",
                          mc.getStr("final_local_method",
                          mc.getStr("final.method",
                          end_local_method_)));
        flm = to_lower(trim(flm));

        end_local_refine_ = fl_enable;
        end_local_method_ = flm;

        debug_kh_ = mc.getInt("debug_kh", debug_kh_);
        debug_kh_ = parse_int(mc.getStr("debug_kh", std::to_string(debug_kh_)), debug_kh_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    double meanRange() const;
    Vec    unitDir(const Vec& a, const Vec& b) const;

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    Nprev_;
    std::vector<Vec>    Fprev_;

    // params
    double nmax_{0.01};
    double vf_{0.02};
    double dmax0_{0.005};
    double dt_{1.0};
    double wn_{0.9};
    double wf_{0.8};
    double c_best_{2.0};
    double c_food_{1.0};
    int    neighbor_k_{5};
    bool   greedy_{true};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_kh_ = 0;
    int         pop_override_ = -1;

    long long   iter_ = 0;
};

} // namespace optimsolution
