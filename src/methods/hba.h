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

// Honey Badger Algorithm (HBA)
// Continuous population-based metaheuristic with two phases:
// (1) digging (exploration) using a cardioid-like movement, (2) honey (exploitation)
// close to the prey (global best). This implementation follows the canonical two-phase
// update with a decaying density factor alpha.
class HBA : public Optimizer {
public:
    HBA() = default;
    ~HBA() override = default;

    std::string methodShortName() const override { return "HBA"; }
    std::string methodFullName()  const override { return "Honey Badger Algorithm"; }

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

        // HBA parameters
        beta_ = parse_double(mc.getStr("beta", std::to_string(beta_)), mc.getDbl("beta", beta_));
        C_    = parse_double(mc.getStr("C",    std::to_string(C_)),    mc.getDbl("C",    C_));
        pdig_ = parse_double(mc.getStr("p_dig",std::to_string(pdig_)), mc.getDbl("p_dig", pdig_));
        eps_  = parse_double(mc.getStr("eps",  std::to_string(eps_)),  mc.getDbl("eps",  eps_));
        i_cap_= parse_double(mc.getStr("i_cap",std::to_string(i_cap_)),mc.getDbl("i_cap", i_cap_));

        if (!std::isfinite(beta_) || beta_ < 1.0) beta_ = 6.0;
        if (!std::isfinite(C_)    || C_    <= 0.0) C_ = 2.0;
        if (!std::isfinite(pdig_)) pdig_ = 0.5;
        pdig_ = (pdig_ < 0.0 ? 0.0 : (pdig_ > 1.0 ? 1.0 : pdig_));
        if (!std::isfinite(eps_)  || eps_ <= 0.0) eps_ = 1e-12;
        if (!std::isfinite(i_cap_)|| i_cap_ <= 0.0) i_cap_ = 1e3;

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

        debug_hba_ = mc.getInt("debug_hba", debug_hba_);
        debug_hba_ = parse_int(mc.getStr("debug_hba", std::to_string(debug_hba_)), debug_hba_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    double meanRange() const;

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // params
    double beta_{6.0};
    double C_{2.0};
    double pdig_{0.5};
    double eps_{1e-12};
    double i_cap_{1e3};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_hba_ = 0;
    int         pop_override_ = -1;

    long long   iter_ = 0;
};

} // namespace optimsolution
