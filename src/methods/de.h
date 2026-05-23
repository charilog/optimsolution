#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>

namespace optimsolution {

class DE : public Optimizer {
public:
    DE() = default;
    ~DE() override = default;

    std::string methodShortName() const override { return "DE"; }
    std::string methodFullName()  const override { return "Differential Evolution (DE/rand/1/bin)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        // helpers
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

        // DE params
        F_  = parse_double(mc.getStr("F",  std::to_string(F_)),  F_);
        CR_ = parse_double(mc.getStr("CR", std::to_string(CR_)), CR_);

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

        // Population override from [de] (aliases & robust parsing)
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
            // CRITICAL: the base value is updated immediately so it is printed correctly in the header
            this->setPopulation(pop_override_);
        }

        // Final local at end (aliases; also accepts the typo end_local_refin)
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

        // Optional echo flag
        debug_de_ = mc.getInt("debug_de", debug_de_);
        debug_de_ = parse_int(mc.getStr("debug_de", std::to_string(debug_de_)), debug_de_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    int  pickDistinct(int n, int a=-1, int b=-1, int c=-1);
    void ensureBounds(Vec& v);
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    double F_{0.6};
    double CR_{0.9};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_de_ = 0;
    int         pop_override_ = -1;
};

} // namespace optimsolution
