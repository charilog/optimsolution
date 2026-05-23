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

class HS : public Optimizer {
public:
    HS() = default;
    ~HS() override = default;

    std::string methodShortName() const override { return "HS"; }
    std::string methodFullName()  const override { return "Harmony Search"; }

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

        // HS core params
        HMCR_ = parse_double(mc.getStr("hmcr", mc.getStr("HMCR", std::to_string(HMCR_))), HMCR_);
        PAR_  = parse_double(mc.getStr("par",  mc.getStr("PAR",  std::to_string(PAR_ ))), PAR_ );

        // bandwidth: accepts either scale (<=1) or absolute (>1). Explicit forms also supported.
        std::string bw_s = mc.getStr("bw",
                           mc.getStr("bandwidth",
                           mc.getStr("bw_scale", std::string{})));
        if (!trim(bw_s).empty()) {
            double v = parse_double(bw_s, bw_scale_);
            if (std::isfinite(v)) {
                if (v <= 1.0) { bw_scale_ = std::max(0.0, v); bw_abs_ = std::numeric_limits<double>::quiet_NaN(); }
                else          { bw_abs_   = std::max(0.0, v); }
            }
        }

        std::string bwmin_s = mc.getStr("bw_min",
                              mc.getStr("bandwidth_min",
                              mc.getStr("bw_min_scale", std::string{})));
        if (!trim(bwmin_s).empty()) {
            double v = parse_double(bwmin_s, bw_min_scale_);
            if (std::isfinite(v)) {
                if (v <= 1.0) { bw_min_scale_ = std::max(0.0, v); bw_min_abs_ = std::numeric_limits<double>::quiet_NaN(); }
                else          { bw_min_abs_   = std::max(0.0, v); }
            }
        }

        // adaptive bandwidth flag
        int abw = mc.getInt("adaptive_bw", adaptive_bw_ ? 1 : 0);
        abw = parse_int(mc.getStr("adaptive_bw", std::to_string(abw)), abw);
        adaptive_bw_ = (abw != 0);

        // improv count per iteration
        int imp = mc.getInt("improvisations",
                  mc.getInt("new_harmonies",
                  mc.getInt("iters_per_gen", improvisations_)));
        if (imp <= 0) imp = parse_int(mc.getStr("improvisations", ""), imp);
        improvisations_ = imp;

        // clamp main params
        if (HMCR_ < 0.0) HMCR_ = 0.0;
        if (HMCR_ > 1.0) HMCR_ = 1.0;
        if (PAR_  < 0.0) PAR_  = 0.0;
        if (PAR_  > 1.0) PAR_  = 1.0;
        if (bw_scale_ < 0.0) bw_scale_ = 0.0;
        if (bw_min_scale_ < 0.0) bw_min_scale_ = 0.0;

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

        // Population override from [hs] (aliases)
        int p = -1;
        p = mc.getInt("population",
            mc.getInt("Population",
            mc.getInt("hms",
            mc.getInt("HMS",
            mc.getInt("memory_size", -1)))));
        if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("hms", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("HMS", ""), -1);
        if (p >= 3) {
            pop_override_ = p;
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
        debug_hs_ = mc.getInt("debug_hs", debug_hs_);
        debug_hs_ = parse_int(mc.getStr("debug_hs", std::to_string(debug_hs_)), debug_hs_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    void   ensureBounds(Vec& v);
    size_t worstIndex() const;
    double bandwidthForDim(int j) const;
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    double HMCR_{0.95};
    double PAR_{0.3};

    // bandwidth is either absolute (bw_abs_) or scale of (ub-lb) (bw_scale_)
    double bw_scale_{0.01};
    double bw_min_scale_{0.001};
    double bw_abs_{std::numeric_limits<double>::quiet_NaN()};
    double bw_min_abs_{std::numeric_limits<double>::quiet_NaN()};
    bool   adaptive_bw_{true};

    int improvisations_{-1};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_hs_ = 0;
    int         pop_override_ = -1;
};

} // namespace optimsolution
