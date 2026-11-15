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

// ---------------------------------------------------------------------------
//  HCDE : Hysteretic-Credit Differential Evolution
//
//  A fully self-adaptive DE variant. Neither F nor CR is ever set by the user
//  as a fixed value: both are produced, per individual and per generation, by
//  two independent mechanisms, and the population size is driven by a third.
//
//   (1) DAHM  - Diversity-Anchored Hysteretic Momentum          -> controls F
//       A population-level "anchor" F* is derived from the mismatch between
//       the measured diversity and a non-linear diversity schedule, and every
//       individual deviates from that anchor according to its own hysteretic
//       success momentum (fast rise on failure, slow decay on success).
//
//   (2) ECCA  - Entropy-regulated Coordinate Credit Allocation  -> controls CR
//       A credit vector over the D coordinates learns which dimensions
//       actually produced improvements. Its normalised entropy sets the target
//       mean of CR, while a softmax tilt of the credit vector decides *which*
//       coordinates are crossed, keeping the expected number of crossed
//       coordinates equal to CR_i * D.
//
//   (3) PS-NLPSR - Population State + Non-Linear Population Size Reduction
//       Classic NLPSR, N(t) = round((Nmin-Nmax)*p^(1-p) + Nmax), with the
//       progress variable p warped by an estimated population state
//       (diversity mismatch + success-rate EMA).
//
//  With beta=0 and the credit vector uniform, ECCA degenerates exactly to
//  standard binomial crossover; with eta=0 and kappa=0, DAHM degenerates to a
//  constant-F scheme. The classic DE/current-to-pbest/1/bin is therefore a
//  strict special case of HCDE.
// ---------------------------------------------------------------------------

class HCDE : public Optimizer {
public:
    HCDE() = default;
    ~HCDE() override = default;

    std::string methodShortName() const override { return "HCDE"; }
    std::string methodFullName()  const override {
        return "Hysteretic-Credit Differential Evolution (DAHM-F / ECCA-CR / PS-NLPSR)";
    }

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
        // ConfigFile lowercases every key on load, so all lookups below use
        // lowercase keys only (this is the bug that silently disabled f/cr in DE).
        auto D = [&](const char* k, double fb){ return parse_double(mc.getStr(k, std::to_string(fb)), fb); };
        auto I = [&](const char* k, int    fb){ return parse_int   (mc.getStr(k, std::to_string(fb)), fb); };

        // ---- (1) DAHM : adaptive F -------------------------------------
        f_min_    = D("dahm_f_min",    f_min_);     // hard clamp on any sampled F
        f_max_    = D("dahm_f_max",    f_max_);
        f_lo_     = D("dahm_f_lo",     f_lo_);      // range of the anchor F*
        f_hi_     = D("dahm_f_hi",     f_hi_);
        kappa_    = D("dahm_kappa",    kappa_);     // diversity -> F* sigmoid gain
        eta_      = D("dahm_eta",      eta_);       // momentum -> F gain
        c_up_     = D("dahm_c_up",     c_up_);      // momentum rate on FAILURE (fast)
        c_dn_     = D("dahm_c_dn",     c_dn_);      // momentum rate on SUCCESS (slow)
        sigmaF0_  = D("dahm_sigma",    sigmaF0_);   // Cauchy scale for F sampling
        div_exp_  = D("dahm_div_exp",  div_exp_);   // exponent of diversity schedule

        // ---- (2) ECCA : adaptive CR ------------------------------------
        cr_lo_    = D("ecca_cr_lo",    cr_lo_);
        cr_hi_    = D("ecca_cr_hi",    cr_hi_);
        c_cr_     = D("ecca_c",        c_cr_);      // mu_CR learning rate
        alpha_    = D("ecca_alpha",    alpha_);     // Lehmer vs entropy-target blend
        beta_     = D("ecca_beta",     beta_);      // softmax tilt of credit vector
        lam_      = D("ecca_lambda",   lam_);       // credit learning rate
        lam_f_    = D("ecca_forget",   lam_f_);     // credit forgetting on stagnation
        sigmaCR_  = D("ecca_sigma",    sigmaCR_);   // per-individual CR jitter
        pcross_min_ = D("ecca_p_min",  pcross_min_);
        pcross_max_ = D("ecca_p_max",  pcross_max_);
        w_min_    = D("ecca_w_min",    w_min_);
        w_max_    = D("ecca_w_max",    w_max_);

        // ---- (3) PS-NLPSR : population state + non-linear reduction ----
        use_nlpsr_ = parse_bool(mc.getStr("nlpsr", std::string{}), I("nlpsr", use_nlpsr_?1:0) != 0);
        n_min_     = I("nlpsr_n_min",  n_min_);
        ps_gain_   = D("nlpsr_ps_gain", ps_gain_);  // how strongly the state warps progress
        ps_a_      = D("ps_div_gain",  ps_a_);      // diversity term in the state
        ps_b_      = D("ps_sr_gain",   ps_b_);      // success-rate term in the state
        sr_ref_    = D("ps_sr_ref",    sr_ref_);    // reference success rate
        c_sr_      = D("ps_sr_c",      c_sr_);      // success-rate EMA rate

        // ---- mutation / archive ----------------------------------------
        arc_rate_  = D("arc_rate",     arc_rate_);
        pbest_hi_  = D("pbest_hi",     pbest_hi_);
        pbest_lo_  = D("pbest_lo",     pbest_lo_);
        p_rand_max_= D("rand1_max",    p_rand_max_); // stagnation-triggered DE/rand/1

        // ---- in-run local search (aliases, same contract as DE) --------
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local", local_method_)));
        lm = to_lower(trim(lm));
        double lr = mc.getDbl("local_rate",
                      mc.getDbl("local.rate",
                      mc.getDbl("inrun_rate", local_rate_)));
        lr = parse_double(mc.getStr("local_rate",
                   mc.getStr("local.rate",
                   mc.getStr("inrun_rate", std::to_string(lr)))), lr);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        if (lm == "none" || lm == "off" || lm == "0") { local_method_.clear(); local_rate_ = 0.0; }
        else { local_method_ = lm; local_rate_ = lr; }

        // ---- population override ---------------------------------------
        int p = mc.getInt("population", mc.getInt("pop", -1));
        if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("pop", ""), -1);
        if (p >= 4) { pop_override_ = p; this->setPopulation(pop_override_); }

        // ---- final local refinement ------------------------------------
        int flg = mc.getInt("end_local_refine",
                  mc.getInt("final_local",
                  mc.getInt("final.local", end_local_refine_ ? 1 : 0)));
        flg = mc.getInt("end_local_refin", flg);
        std::string flg_s = mc.getStr("end_local_refine",
                            mc.getStr("final_local",
                            mc.getStr("final.local",
                            mc.getStr("end_local_refin", std::string{}))));
        end_local_refine_ = parse_bool(flg_s, flg != 0);
        std::string flm = mc.getStr("end_local_method",
                          mc.getStr("final_local_method",
                          mc.getStr("final.method", end_local_method_)));
        end_local_method_ = to_lower(trim(flm));

        debug_ = I("debug_hcde", debug_);

        sanitize();
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    // ---- small helpers ------------------------------------------------
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }
    void   sanitize();
    void   clampBounds(Vec& v);
    void   repairMidpoint(Vec& v, const Vec& parent);
    int    randDistinct(int n, int a=-1, int b=-1, int c=-1);
    double sampleCauchyPositive(double mu, double sigma);

    // ---- the three mechanisms -----------------------------------------
    double measureDiversity() const;         // PS
    void   updatePopulationState(double progress);
    double anchorF() const;                  // DAHM  (population level)
    double sampleF(int i);                   // DAHM  (individual level)
    void   updateMomentum(int i, bool success);
    double creditEntropy() const;            // ECCA
    void   buildCrossoverTilt();             // ECCA
    void   updateCredit(const std::vector<double>& num,
                        const std::vector<double>& den);
    void   updateMuCR(const std::vector<double>& S_CR,
                      const std::vector<double>& S_dF);
    void   applyNLPSR(double progress);      // PS-NLPSR
    void   buildOrder();
    void   trimArchive();

private:
    // population & archive
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    ARC_;
    std::vector<int>    order_;

    // per-individual adaptive state
    std::vector<double> Fi_, CRi_, mom_;

    // ECCA state
    std::vector<double> W_;      // coordinate credit weights, size D
    std::vector<double> q_;      // softmax tilt, size D, sums to 1
    double muCR_   {0.50};
    double Hnorm_  {1.00};

    // DAHM / PS state
    double Fstar_  {0.55};
    double div_    {1.0}, div0_{1.0}, divTarget_{1.0}, diagScale_{1.0};
    double srEma_  {0.50};
    double state_  {0.0};        // population state in [-1, +1]

    // sizing
    int N0_   {0};
    int n_min_{6};
    long long gen_{0};

    // ---- tunables (all have sane defaults; none of them is F or CR) ----
    double f_min_{0.05}, f_max_{1.10}, f_lo_{0.15}, f_hi_{0.95};
    double kappa_{3.0},  eta_{0.55}, c_up_{0.35}, c_dn_{0.10};
    double sigmaF0_{0.15}, div_exp_{1.5};

    double cr_lo_{0.05}, cr_hi_{0.95}, c_cr_{0.15}, alpha_{0.5};
    double beta_{1.5}, lam_{0.25}, lam_f_{0.05}, sigmaCR_{0.10};
    double pcross_min_{0.02}, pcross_max_{0.98};
    double w_min_{0.05}, w_max_{8.0};

    bool   use_nlpsr_{true};
    double ps_gain_{0.35}, ps_a_{1.0}, ps_b_{2.0}, sr_ref_{0.20}, c_sr_{0.25};

    double arc_rate_{1.5}, pbest_hi_{0.25}, pbest_lo_{0.08}, p_rand_max_{0.30};

    // shared with the DE contract
    std::string local_method_ = "lbfgs";
    double      local_rate_   = 0.0;
    bool        end_local_refine_ = false;
    std::string end_local_method_;
    int         debug_ = 0;
    int         pop_override_ = -1;
};

} // namespace optimsolution
