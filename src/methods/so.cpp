#include "so.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static constexpr double kPi = 3.141592653589793238462643383279502884;

static inline std::string to_lower(std::string s){
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static inline std::string trim(std::string s){
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

void SO::configure(const MethodConfig& mc) {
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

    // Spiral parameters
    {
        std::string rs = mc.getStr("shrink",
                        mc.getStr("r",
                        mc.getStr("contraction",
                        mc.getStr("radius_factor",
                        mc.getStr("radius", std::to_string(shrink_))))));
        double r = parse_double(rs, shrink_);
        if (!std::isfinite(r)) r = shrink_;
        if (r <= 0.0) r = 1e-6;
        if (r > 1.0) r = 1.0;
        shrink_ = r;
    }

    {
        std::string ts = mc.getStr("theta",
                        mc.getStr("angle",
                        mc.getStr("rotation",
                        mc.getStr("rotation_angle", std::to_string(theta_)))));
        double t = parse_double(ts, theta_);
        if (std::isfinite(t)) {
            double at = std::fabs(t);
            if (at > 2.0*kPi + 1e-9 && at <= 360.0) {
                t = t * (kPi / 180.0); // degrees -> radians
            }
            // keep within a reasonable range
            if (!std::isfinite(t)) t = theta_;
            if (std::fabs(t) < 1e-12) t = 1e-12;
            theta_ = t;
        }
    }

    random_direction_ = parse_bool(mc.getStr("random_direction", std::string{}), random_direction_);
    random_direction_ = parse_bool(mc.getStr("random_dir", std::string{}), random_direction_);

    greedy_ = parse_bool(mc.getStr("greedy", std::string{}), greedy_);
    greedy_ = parse_bool(mc.getStr("accept_improve_only", std::string{}), greedy_);
    {
        // accept_all overrides greedy if explicitly provided
        std::string aa = mc.getStr("accept_all", std::string{});
        if (!trim(aa).empty()) {
            bool accept_all = parse_bool(aa, !greedy_);
            greedy_ = !accept_all;
        }
    }

    // In-run local (aliases)
    {
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
    }

    // Population override from [so]
    {
        int p = -1;
        p = mc.getInt("population",
            mc.getInt("Population",
            mc.getInt("pop",
            mc.getInt("Pop", -1))));
        if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("Population", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("pop", ""), -1);
        if (p < 0) p = parse_int(mc.getStr("Pop", ""), -1);
        if (p >= 3) {
            pop_override_ = p;
            this->setPopulation(pop_override_);
        }
    }

    // Final local at end (aliases; accepts typo end_local_refin)
    {
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
    }

    // Optional echo flag
    debug_so_ = mc.getInt("debug_so", debug_so_);
    debug_so_ = parse_int(mc.getStr("debug_so", std::to_string(debug_so_)), debug_so_);
}

void SO::ensureBounds(Vec& v){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(v[j])) v[j] = 0.5*(lo + hi);
        if (v[j] < lo) v[j] = lo;
        if (v[j] > hi) v[j] = hi;
    }
}

void SO::injectBestIntoWorst(){
    if (X_.empty() || FX_.empty()) return;
    size_t worst_idx = 0;
    double worst_val = FX_[0];
    for (size_t k=1; k<FX_.size(); ++k){
        if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
    }
    if (worst_idx < X_.size()) {
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }
}

void SO::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(3, (pop_override_ >= 3 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_so_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[so] cfg -> N=%d (population() now=%d, override=%d), r=%.6f, theta=%.6f rad, rand_dir=%s, greedy=%s, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, shrink_, theta_,
            random_direction_ ? "on" : "off",
            greedy_ ? "on" : "off",
            lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void SO::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    // Determine current best index from population (robust even if best_x_ was set externally)
    int best_idx = 0;
    double bf = FX_[0];
    for (int i=1; i<N; ++i){
        if (FX_[i] < bf) { bf = FX_[i]; best_idx = i; }
    }
    if (bf < best_f_) {
        best_f_ = bf;
        best_x_ = X_[best_idx];
    }

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double c = std::cos(theta_);
    const double s0 = std::sin(theta_);

    for (int i=0; i<N; ++i){
        if (i == best_idx) continue;

        double s = s0;
        if (random_direction_) {
            if (U01(rng_) < 0.5) s = -s0;
        }

        Vec u(D, 0.0);
        // relative vector
        Vec y(D, 0.0);
        for (int j=0; j<D; ++j) y[j] = X_[i][j] - best_x_[j];

        // apply pairwise 2D rotations + contraction
        int j = 0;
        for (; j+1 < D; j += 2) {
            const double a = y[j];
            const double b = y[j+1];
            const double na = shrink_ * (c*a - s*b);
            const double nb = shrink_ * (s*a + c*b);
            u[j]   = best_x_[j]   + na;
            u[j+1] = best_x_[j+1] + nb;
        }
        if (j < D) {
            // leftover dimension (or D==1)
            u[j] = best_x_[j] + shrink_ * y[j];
        }

        ensureBounds(u);
        double fu = eval(u);

        // optional in-run local
        if (local_rate_ > 0.0 && !local_method_.empty()) {
            if (U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, u);
                if (std::isfinite(floc) && floc < fu) {
                    u  = std::move(xloc);
                    fu = floc;
                }
            }
        }

        const bool accept = (!greedy_) ? true : (fu < FX_[i]);
        if (accept) {
            X_[i]  = std::move(u);
            FX_[i] = fu;
        }

        // Best is updated even if the move is not accepted (greedy mode)
        if (fu < best_f_) {
            best_f_ = fu;
            best_x_ = accept ? X_[i] : u;
            best_idx = i;
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Elitism: keep best inside the population so stop/reporting sees the correct minimum
    injectBestIntoWorst();

    printBest();
    updateStop(FX_);
}

void SO::end() {
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty()) {
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = std::move(xloc);
        }
    }

    injectBestIntoWorst();
    printBest();
}

} // namespace optimsolution
