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

// Marine Predators Algorithm (MPA)
class MPA : public Optimizer {
public:
    MPA() = default;
    ~MPA() override = default;

    std::string methodShortName() const override { return "MPA"; }
    std::string methodFullName()  const override { return "Marine Predators Algorithm"; }

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

        // Population override from [mpa] (aliases)
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

        // MPA parameters
        FADs_ = parse_double(mc.getStr("fads", std::to_string(FADs_)), mc.getDbl("fads", FADs_));
        if (FADs_ < 0.0) FADs_ = 0.0;
        if (FADs_ > 1.0) FADs_ = 1.0;

        P_ = parse_double(mc.getStr("P", std::to_string(P_)), mc.getDbl("P", P_));
        if (!std::isfinite(P_) || P_ <= 0.0) P_ = 0.5;
        if (P_ > 1.0) P_ = 1.0;

        levy_beta_ = parse_double(mc.getStr("levy_beta", std::to_string(levy_beta_)), mc.getDbl("levy_beta", levy_beta_));
        if (!std::isfinite(levy_beta_) || levy_beta_ < 1.01) levy_beta_ = 1.5;
        if (levy_beta_ > 2.0) levy_beta_ = 2.0;

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

        debug_mpa_ = mc.getInt("debug_mpa", debug_mpa_);
        debug_mpa_ = parse_int(mc.getStr("debug_mpa", std::to_string(debug_mpa_)), debug_mpa_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    int  pickDistinct(int n, int a=-1);
    double levyStep();

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_mpa_ = 0;
    int         pop_override_ = -1;

    double      FADs_ = 0.2;      // Fish Aggregating Devices effect probability
    double      P_    = 0.5;      // movement scaling
    double      levy_beta_ = 1.5; // Levy exponent

    long long   iter_ = 0;
};

} // namespace optimsolution
