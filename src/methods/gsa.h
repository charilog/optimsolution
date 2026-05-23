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

// Gravitational Search Algorithm (GSA)
class GSA : public Optimizer {
public:
    GSA() = default;
    ~GSA() override = default;

    std::string methodShortName() const override { return "GSA"; }
    std::string methodFullName()  const override { return "Gravitational Search Algorithm"; }

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

        // Core GSA params
        G0_    = parse_double(mc.getStr("G0", std::to_string(G0_)), G0_);
        alpha_ = parse_double(mc.getStr("alpha", std::to_string(alpha_)), alpha_);
        eps_   = parse_double(mc.getStr("eps", std::to_string(eps_)), eps_);
        w_     = parse_double(mc.getStr("w", std::to_string(w_)), w_);
        if (!std::isfinite(w_)) w_ = 0.7;
        if (w_ < 0.0) w_ = 0.0;
        if (w_ > 1.0) w_ = 1.0;

        // kbest control: either fixed kbest count, or ratio, or adaptive
        kbest_ratio_ = mc.getDbl("kbest_ratio", kbest_ratio_);
        kbest_ratio_ = parse_double(mc.getStr("kbest_ratio", std::to_string(kbest_ratio_)), kbest_ratio_);
        if (kbest_ratio_ < 0.0) kbest_ratio_ = 0.0;
        if (kbest_ratio_ > 1.0) kbest_ratio_ = 1.0;

        fixed_kbest_ = mc.getInt("kbest", fixed_kbest_);
        fixed_kbest_ = parse_int(mc.getStr("kbest", std::to_string(fixed_kbest_)), fixed_kbest_);
        adaptive_kbest_ = mc.getInt("adaptive_kbest", adaptive_kbest_ ? 1 : 0) != 0;
        adaptive_kbest_ = parse_bool(mc.getStr("adaptive_kbest", adaptive_kbest_ ? "1" : "0"), adaptive_kbest_);

        vmax_scale_ = mc.getDbl("vmax_scale", vmax_scale_);
        vmax_scale_ = parse_double(mc.getStr("vmax_scale", std::to_string(vmax_scale_)), vmax_scale_);
        if (vmax_scale_ < 0.0) vmax_scale_ = 0.0;

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
        debug_gsa_ = mc.getInt("debug_gsa", debug_gsa_);
        debug_gsa_ = parse_int(mc.getStr("debug_gsa", std::to_string(debug_gsa_)), debug_gsa_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    void clampVelocity(Vec& vel);
    int currentKbest(int N) const;

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<Vec>    V_;
    std::vector<double> FX_;

    // params
    double G0_{100.0};
    double alpha_{20.0};
    double eps_{1e-12};
    double w_{0.7};

    double kbest_ratio_{1.0};  // used if fixed_kbest_ < 2
    int    fixed_kbest_{-1};
    bool   adaptive_kbest_{true};

    double vmax_scale_{0.0};   // 0 -> no clamp, else vmax = vmax_scale*(ub-lb)

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int         debug_gsa_ = 0;
    int         pop_override_ = -1;

    long long iter_{0};
    long long max_iters_est_{1};
};

} // namespace optimsolution
