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

// Water Cycle Algorithm (WCA)
// Canonical continuous-domain implementation with sea/rivers/streams,
// movement toward better solutions and evaporation/raining.
class WCA : public Optimizer {
public:
    WCA() = default;
    ~WCA() override = default;

    std::string methodShortName() const override { return "WCA"; }
    std::string methodFullName()  const override { return "Water Cycle Algorithm (WCA)"; }

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

        // WCA parameters
        nsr_ = mc.getInt("nsr", mc.getInt("NSR", nsr_));
        nsr_ = parse_int(mc.getStr("nsr", std::to_string(nsr_)), nsr_);
        if (nsr_ < 2) nsr_ = 2;

        C_ = parse_double(mc.getStr("C", std::to_string(C_)), mc.getDbl("C", C_));
        if (!std::isfinite(C_) || C_ <= 0.0) C_ = 2.0;

        // dmax0 and dmax_min: if <= 1 treat as fraction of mean range, else absolute
        dmax0_ = parse_double(mc.getStr("dmax0", std::to_string(dmax0_)), mc.getDbl("dmax0", dmax0_));
        dmax0_ = parse_double(mc.getStr("dmax", std::to_string(dmax0_)), mc.getDbl("dmax", dmax0_));
        dmax_min_ = parse_double(mc.getStr("dmax_min", std::to_string(dmax_min_)), mc.getDbl("dmax_min", dmax_min_));
        if (!std::isfinite(dmax0_) || dmax0_ <= 0.0) dmax0_ = 0.01;
        if (!std::isfinite(dmax_min_) || dmax_min_ <= 0.0) dmax_min_ = 1e-5;
        if (dmax_min_ > dmax0_) std::swap(dmax_min_, dmax0_);

        rain_prob_ = parse_double(mc.getStr("rain_prob", std::to_string(rain_prob_)), mc.getDbl("rain_prob", rain_prob_));
        if (!std::isfinite(rain_prob_)) rain_prob_ = 0.1;
        if (rain_prob_ < 0.0) rain_prob_ = 0.0;
        if (rain_prob_ > 1.0) rain_prob_ = 1.0;

        greedy_ = mc.getInt("greedy", greedy_ ? 1 : 0) != 0;
        greedy_ = parse_bool(mc.getStr("greedy", greedy_ ? "1" : "0"), greedy_);

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

        debug_wca_ = mc.getInt("debug_wca", debug_wca_);
        debug_wca_ = parse_int(mc.getStr("debug_wca", std::to_string(debug_wca_)), debug_wca_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    Vec  randomVec();
    double dist(const Vec& a, const Vec& b) const;
    double currentDmaxAbs() const;

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // params
    int    nsr_{4};
    double C_{2.0};
    double dmax0_{0.01};
    double dmax_min_{1e-5};
    double rain_prob_{0.1};
    bool   greedy_{true};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_wca_ = 0;
    int         pop_override_ = -1;

    long long   iter_{0};
    double      mean_range_{1.0};
};

} // namespace optimsolution
