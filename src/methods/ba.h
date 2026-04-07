#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>

namespace optimsolution {

class BA : public Optimizer {
public:
    BA() = default;
    ~BA() override = default;

    std::string methodShortName() const override { return "BA"; }
    std::string methodFullName()  const override { return "Bat Algorithm (BA)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        auto to_lower = [](std::string s){
            for (auto &c: s) c = (char)std::tolower((unsigned char)c);
            return s;
        };
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

        // Core BA parameters
        fmin_ = parse_double(mc.getStr("fmin", mc.getStr("f_min", std::to_string(fmin_))), fmin_);
        fmax_ = parse_double(mc.getStr("fmax", mc.getStr("f_max", std::to_string(fmax_))), fmax_);
        alpha_ = parse_double(mc.getStr("alpha", mc.getStr("loudness_decay", std::to_string(alpha_))), alpha_);
        gamma_ = parse_double(mc.getStr("gamma", mc.getStr("pulse_gamma", std::to_string(gamma_))), gamma_);
        A0_ = parse_double(mc.getStr("A0", mc.getStr("a0", mc.getStr("loudness", std::to_string(A0_)))), A0_);
        r0_ = parse_double(mc.getStr("r0", mc.getStr("R0", mc.getStr("pulse", std::to_string(r0_)))), r0_);
        vmax_scale_ = parse_double(mc.getStr("vmax_scale", mc.getStr("v_max_scale", mc.getStr("vmax", std::to_string(vmax_scale_)))), vmax_scale_);
        walk_scale_ = parse_double(mc.getStr("walk_scale", mc.getStr("rw_scale", std::to_string(walk_scale_))), walk_scale_);

        if (!std::isfinite(fmin_)) fmin_ = 0.0;
        if (!std::isfinite(fmax_)) fmax_ = 2.0;
        if (fmax_ < fmin_) std::swap(fmin_, fmax_);
        if (fmax_ == fmin_) fmax_ = fmin_ + 1.0;

        if (!std::isfinite(alpha_) || alpha_ <= 0.0) alpha_ = 0.9;
        if (alpha_ > 1.0) alpha_ = 1.0;
        if (!std::isfinite(gamma_) || gamma_ < 0.0) gamma_ = 0.9;

        if (!std::isfinite(A0_) || A0_ <= 0.0) A0_ = 1.0;
        if (!std::isfinite(r0_)) r0_ = 0.5;
        if (r0_ < 0.0) r0_ = 0.0;
        if (r0_ > 1.0) r0_ = 1.0;

        if (!std::isfinite(vmax_scale_) || vmax_scale_ < 0.0) vmax_scale_ = 0.2;
        if (vmax_scale_ > 10.0) vmax_scale_ = 10.0;

        if (!std::isfinite(walk_scale_) || walk_scale_ < 0.0) walk_scale_ = 0.01;
        if (walk_scale_ > 10.0) walk_scale_ = 10.0;

        // Optional: accept improvements even if rand>=A (useful for deterministic benchmarks)
        int aimp = mc.getInt("accept_improvement_always", accept_improvement_always_ ? 1 : 0);
        std::string aimp_s = mc.getStr("accept_improvement_always", std::string{});
        accept_improvement_always_ = parse_bool(aimp_s, aimp != 0);

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

        // Population override from [ba]
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

        // Final local at end (aliases; accepts the typo end_local_refin)
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

        debug_ba_ = mc.getInt("debug_ba", debug_ba_);
        debug_ba_ = parse_int(mc.getStr("debug_ba", std::to_string(debug_ba_)), debug_ba_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    void ensureBounds(Vec& v);
    void clampVelocity(Vec& v);

    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

private:
    std::vector<Vec>    X_;
    std::vector<Vec>    V_;
    std::vector<double> FX_;
    std::vector<double> A_;
    std::vector<double> R_;

    double fmin_{0.0};
    double fmax_{2.0};
    double alpha_{0.9};
    double gamma_{0.9};
    double A0_{1.0};
    double r0_{0.5};
    double vmax_scale_{0.2};
    // Random-walk intensity around the current best is expressed as a fraction of the variable range.
    // A conservative default helps on easy continuous benchmarks (otherwise the walk can be too large).
    double walk_scale_{0.01};

    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    bool        accept_improvement_always_ = false;

    int         debug_ba_ = 0;
    int         pop_override_ = -1;
    long long   it_ = 0;
};

} // namespace optimsolution
