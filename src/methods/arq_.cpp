#include "arq.h"

#include <cmath>
#include <chrono>
#include <cctype>
#include <limits>
#include <random>      // for std::random_device
#include <cstdio>      // optional debug

namespace optimsolution {

// Small mixer for seed stabilization
static inline uint64_t splitmix64(uint64_t z){
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Helpers for robust parsing (same pattern as DE)
static inline std::string to_lower(std::string s){
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static inline std::string trim(std::string s){
    size_t a=0, b=s.size();
    while (a<b && std::isspace((unsigned char)s[a])) ++a;
    while (b>a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a,b-a);
}
static inline int parse_int_str(const std::string& s, int fb){
    std::string t = trim(s);
    if (t.empty()) return fb;
    try{ size_t pos=0; long v=std::stol(t,&pos); if (pos==t.size()) return (int)v; }catch(...) {}
    return fb;
}
static inline double parse_double_str(const std::string& s, double fb){
    std::string t = trim(s);
    if (t.empty()) return fb;
    try{ size_t pos=0; double v=std::stod(t,&pos); if (pos==t.size() && std::isfinite(v)) return v; }catch(...) {}
    return fb;
}

// -----------------------------------------------------------------------------
// configure
// -----------------------------------------------------------------------------

void ARQ::configure(const MethodConfig& mc){
    // -------- Population override (aliases as in DE) --------
    int p = -1;
    p = mc.getInt("population",
        mc.getInt("Population",
        mc.getInt("pop",
        mc.getInt("Pop", -1))));
    if (p < 0) p = parse_int_str(mc.getStr("population",""), -1);
    if (p < 0) p = parse_int_str(mc.getStr("Population",""), -1);
    if (p < 0) p = parse_int_str(mc.getStr("pop",""), -1);
    if (p < 0) p = parse_int_str(mc.getStr("Pop",""), -1);
    if (p >= 4) {
        pop_override_ = p;
        // Critical: The base state is synchronized immediately so the reporter prints correctly
        this->setPopulation(pop_override_);
    }

    agent_fraction_ = std::clamp(mc.getDbl("agent_fraction", agent_fraction_), 0.01, 1.0);

    // Success-History
    muF_   = mc.getDbl("muF_init",  muF_);
    muCR_  = mc.getDbl("muCR_init", muCR_);
    sh_c_  = mc.getDbl("sh_c",      sh_c_);
    F_lo_  = mc.getDbl("F_lo",      F_lo_);
    F_hi_  = mc.getDbl("F_hi",      F_hi_);
    CR_lo_ = mc.getDbl("CR_lo",     CR_lo_);
    CR_hi_ = mc.getDbl("CR_hi",     CR_hi_);

    // pbest & archive
    pbest_frac_   = std::clamp(mc.getDbl("pbest_frac", pbest_frac_), 0.01, 0.9);
    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);

    // RTR
    rtr_pool_             = mc.getInt("rtr_pool", rtr_pool_);
    rtr_min_replace_gain_ = mc.getDbl("rtr_min_replace_gain", rtr_min_replace_gain_);

    // Quarantine
    outlier_alpha_    = mc.getDbl("outlier_alpha", outlier_alpha_);
    quarantine_rate_  = mc.getDbl("quarantine_rate", quarantine_rate_);
    quarantine_sigma_ = mc.getDbl("quarantine_sigma", quarantine_sigma_);

    // Stagnation & restart
    stagnation_trigger_ = mc.getInt("stagnation_trigger", stagnation_trigger_);
    restart_frac_       = mc.getDbl("restart_frac", restart_frac_);
    restart_sigma_      = mc.getDbl("restart_sigma", restart_sigma_);

    // In–run local search (optional)
    {
        std::string lm = mc.getStr("local_method",
                         mc.getStr("local.method",
                         mc.getStr("inrun_local", local_method_)));
        lm = to_lower(trim(lm));

        std::string lr_str = mc.getStr("local_rate",
                             mc.getStr("local.rate",
                             mc.getStr("inrun_rate", std::to_string(local_rate_))));
        double lr = parse_double_str(lr_str, local_rate_);
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

    // ---- optional seeding from config ----
    user_seed_   = static_cast<uint64_t>(mc.getInt("seed",     static_cast<int>(user_seed_)));
    run_id_hint_ = mc.getInt("run_id", run_id_hint_);
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

void ARQ::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j]+U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

int ARQ::eliteIndexFinite() const{
    int ei = 0;
    double bf = std::numeric_limits<double>::infinity();
    for (int i=0;i<(int)FX_.size();++i){
        double f = FX_[i];
        if (std::isfinite(f) && f<bf){ bf=f; ei=i; }
    }
    return ei;
}

int ARQ::pickDistinct(int n, int a, int b, int c){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a || r==b || r==c);
    return r;
}

void ARQ::pushArchive_(const Vec& x){
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    const size_t cap = (size_t)std::max(0.0, std::round(archive_rate_ * (double)population()));
    if (cap>0 && archive_.size() > cap){
        std::uniform_int_distribution<size_t> A(0, archive_.size()-1);
        size_t pos = A(rng_);
        archive_.erase(archive_.begin()+static_cast<long>(pos));
    }
}

int ARQ::pickArchiveIndex_() const{
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> A(0, (int)archive_.size()-1);
    return A(const_cast<std::mt19937_64&>(rng_));
}

int ARQ::pickPbestIndex_() const{
    const int N = (int)FX_.size();
    const int P = std::max(1, (int)std::round(pbest_frac_ * (double)N));

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+P, idx.end(),
        [&](int a,int b){ return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> J(0, P-1);
    return idx[J(const_cast<std::mt19937_64&>(rng_))];
}

double ARQ::bnDistance_(const Vec& a, const Vec& b) const{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double s=0.0, eps=1e-12;
    for (size_t j=0;j<a.size();++j){
        double r = U[j]-L[j];
        double d = (a[j]-b[j]) / (r + eps);
        s += d*d;
    }
    return std::sqrt(s);
}

int ARQ::pickRTRNeighbor_(const Vec& trial, const std::vector<int>& pool) const{
    int best = pool[0];
    double bd = bnDistance_(trial, X_[best]);
    for (size_t t=1;t<pool.size();++t){
        int k = pool[t];
        double d = bnDistance_(trial, X_[k]);
        if (d < bd){ bd=d; best=k; }
    }
    return best;
}

// -----------------------------------------------------------------------------
// DE operator: pbest/1 + archive + binomial crossover
// -----------------------------------------------------------------------------

void ARQ::trial_pbest1A_bin_(int i, const Vec& xi, Vec& tr, double F, double CR){
    const int D = prob_->dimension();
    const int pbest = pickPbestIndex_();
    const int r1    = pickDistinct(population(), i, pbest);

    // r2 from archive or population
    bool useArch = (!archive_.empty() && U01_(rng_) < 0.5);
    Vec base_r2;
    if (useArch){
        int ia = pickArchiveIndex_();
        base_r2 = archive_[ia];
    } else {
        int rtemp = pickDistinct(population(), i, pbest, r1);
        base_r2 = X_[rtemp];
    }

    tr = xi;
    std::uniform_int_distribution<int> J(0, D-1);
    const int jrand = J(rng_);
    for (int j=0;j<D;++j){
        if (U01_(rng_) < CR || j==jrand){
            double v = xi[j] + F*(X_[pbest][j] - xi[j]) + F*(X_[r1][j] - base_r2[j]);
            tr[j] = v;
        }
    }
    ensureBounds(tr);
}

// -----------------------------------------------------------------------------
// Quarantine: outlier repair with a robust center
// -----------------------------------------------------------------------------

void ARQ::quarantineOutliers_(){
    if (quarantine_rate_ <= 0.0) return;
    if ((int)FX_.size() < 8) return;

    // collect finite
    std::vector<double> f;
    f.reserve(FX_.size());
    for (double v: FX_) if (std::isfinite(v)) f.push_back(v);
    if ((int)f.size() < 8) return;

    std::sort(f.begin(), f.end());
    auto qpos = [&](double q){
        double pos=(f.size()-1)*q; size_t lo=(size_t)std::floor(pos), hi=(size_t)std::ceil(pos);
        double a=f[lo], b=f[hi]; return a + (b-a)*(pos-lo);
    };
    double q1 = qpos(0.25), q3 = qpos(0.75), iqr = std::max(0.0, q3-q1);
    double thr = q3 + outlier_alpha_ * iqr;

    // robust center: mean of best half
    const int N = (int)FX_.size();
    const int D = prob_->dimension();
    std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
    int cut = std::max(1, (int)std::round(0.5 * (double)N));
    std::partial_sort(idx.begin(), idx.begin()+cut, idx.end(),
        [&](int a,int b){ return FX_[a] < FX_[b]; });

    Vec center(D, 0.0);
    int counted=0;
    for (int t=0;t<cut;++t){
        int i = idx[t];
        double fi = FX_[i];
        if (!std::isfinite(fi)) continue;
        for (int j=0;j<D;++j) center[j] += X_[i][j];
        counted++;
    }
    if (counted<=0) return;
    for (int j=0;j<D;++j) center[j] /= (double)counted;

    int maxFix = std::max(1, (int)std::round(quarantine_rate_ * (double)N));
    std::normal_distribution<double> N0(0.0,1.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    int fixed=0;
    for (int i=0;i<N && fixed<maxFix; ++i){
        double fi = FX_[i];
        if (!std::isfinite(fi) || fi <= thr) continue;

        Vec cand = X_[i];
        for (int j=0;j<D;++j){
            double v = cand[j]
                     + quarantine_sigma_ * (center[j] - cand[j])
                     + 0.01*(U[j]-L[j])*N0(rng_);
            cand[j] = clamp_(v, L[j], U[j]);
        }
        ensureBounds(cand);
        double fnew = eval(cand);
        if (fnew < FX_[i]){
            pushArchive_(X_[i]);
            X_[i]  = std::move(cand);
            FX_[i] = fnew;
            if (fnew < best_f_){ best_f_ = fnew; best_x_ = X_[i]; }
        }
        fixed++;
    }
}

// -----------------------------------------------------------------------------
// Micro-restart: unconditional restart of the worst individuals around best_x_
// -----------------------------------------------------------------------------

void ARQ::microRestart_(){
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N <= 1) return;

    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    // Sort indices by *current* fitness
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a,int b){
        double fa=FX_[a], fb=FX_[b];
        bool A=std::isfinite(fa), B=std::isfinite(fb);
        if (A && B) return fa<fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a<b;
    });

    int elite = idx[0];
    std::normal_distribution<double> N0(0.0,1.0);

    int nreset = std::max(1, (int)std::round(restart_frac_ * (double)N));
    for (int k=0;k<nreset;++k){
        int i = idx[N-1-k];   // worst individuals
        if (i==elite) continue;

        Vec cand(D);
        for (int j=0;j<D;++j){
            double step = restart_sigma_ * (U[j]-L[j]) * N0(rng_);
            cand[j] = clamp_(best_x_[j] + step, L[j], U[j]);
        }
        ensureBounds(cand);
        double f = eval(cand);

        // Archive displaced point (as described in the paper)
        pushArchive_(X_[i]);
        X_[i]  = std::move(cand);
        FX_[i] = f;

        if (f < best_f_){
            best_f_ = f;
            best_x_ = X_[i];
        }
    }
}

// -----------------------------------------------------------------------------
// init
// -----------------------------------------------------------------------------

void ARQ::init(){
    if (!prob_) return;

    // If an override from [arq] exists, it takes precedence; otherwise [global] population() is used
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));

    // Synchronizes the base state as well so the header shows the correct Population
    this->setPopulation(N);

    // seeding
    uint64_t now  = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device rd;
    uint64_t entropy = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    uint64_t mix = now ^ entropy ^ (uint64_t)(uintptr_t)this
                   ^ (runs_started_ * 0x9e3779b97f4a7c15ULL)
                   ^ (uint64_t)prob_->calls();

    if (user_seed_ != 0 || run_id_hint_ >= 0){
        uint64_t base = (user_seed_ ? user_seed_ : entropy)
                        ^ (uint64_t)(run_id_hint_ < 0 ? 0 : run_id_hint_);
        seed_used_ = splitmix64(base ^ runs_started_);
    } else {
        seed_used_ = splitmix64(mix);
    }
    rng_.seed(seed_used_);
    runs_started_++;

    stagn_iters_ = 0;
    start_agent_ = 0;

    const int D = prob_->dimension();
    X_.clear(); FX_.clear(); archive_.clear();

    // Initial sampling
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0;i<N; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_){ best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // print once
    printBest();
    updateStop(FX_);
}

// -----------------------------------------------------------------------------
// one_iteration
// -----------------------------------------------------------------------------

void ARQ::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D<=0) { updateStop(FX_); return; }

    const int N = (int)X_.size();
    if (N <= 0) { updateStop(FX_); return; }

    // sort indices by fitness (finite first)
    std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a,int b){
        double fa=FX_[a], fb=FX_[b];
        bool A=std::isfinite(fa), B=std::isfinite(fb);
        if (A && B) return fa<fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a<b;
    });
    int elite = eliteIndexFinite();
    double prevBest = best_f_;
    if (FX_[elite] < best_f_){ best_f_ = FX_[elite]; best_x_ = X_[elite]; }

    // Mini-batch window
    int batch = std::max(1, (int)std::floor(agent_fraction_ * N));
    int s = start_agent_;
    int e = std::min(N, s + batch);
    if (s>=e){ start_agent_=0; s=0; e=std::min(N, batch); }
    int wsize = e - s;

    // Success-history lists
    std::vector<double> sF;    sF.reserve(wsize);
    std::vector<double> sCR;   sCR.reserve(wsize);
    std::vector<double> wGain; wGain.reserve(wsize);

    // loop
    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::normal_distribution<double> N0(0.0,1.0);

    for (int t=0; t<wsize; ++t){
        int i = idx[s + t];
        if (i==elite) continue; // keep absolute elite

        // sample F ~ Cauchy around muF, CR ~ Normal around muCR
        double F  = muF_ + 0.10 * std::tan(3.141592653589793*(U01(rng_) - 0.5));
        if (F<=0.0 || !std::isfinite(F)) F = muF_;
        F = std::clamp(F, F_lo_, F_hi_);

        double CR = muCR_ + 0.10 * N0(rng_);
        if (!std::isfinite(CR)) CR = muCR_;
        CR = std::clamp(CR, CR_lo_, CR_hi_);

        // parent
        Vec x = X_[i];
        ensureBounds(x);
        double fx = FX_[i];
        if (!std::isfinite(fx)) {
            fx = eval(x);
            FX_[i] = fx;
        }

        // trial
        Vec T; T.reserve(D);
        trial_pbest1A_bin_(i, x, T, F, CR);
        double fT = eval(T);

        // RTR pool
        std::vector<int> pool; pool.reserve(std::max(1, rtr_pool_));
        for (int kk=0; kk<rtr_pool_; ++kk){
            int r; do{ r = std::uniform_int_distribution<int>(0, N-1)(rng_);}while(r==i);
            pool.push_back(r);
        }
        int jstar = pickRTRNeighbor_(T, pool);

        bool   replaced   = false;
        int    target_idx = -1;
        double parent_f   = std::numeric_limits<double>::infinity();

        // 1) Direct replacement of parent i
        if (fT + 1e-18 < fx){
            pushArchive_(X_[i]);
            parent_f   = fx;
            X_[i]      = T;
            FX_[i]     = fT;
            replaced   = true;
            target_idx = i;
        } else {
            // 2) RTR replacement on neighbor jstar
            double fj = FX_[jstar];
            if (!std::isfinite(fj)) {
                fj = eval(X_[jstar]);
                FX_[jstar] = fj;
            }
            double denom   = std::abs(fj) + 1e-12;
            double relGain = (fj - fT)/denom;
            if (fT < fj && relGain >= rtr_min_replace_gain_){
                pushArchive_(X_[jstar]);
                parent_f    = fj;
                X_[jstar]   = T;
                FX_[jstar]  = fT;
                replaced    = true;
                target_idx  = jstar;
            }
        }

        if (replaced && target_idx >= 0){
            // Optional in-run local search on the actually improved individual
            if (local_rate_>0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, X_[target_idx]);
                if (floc < FX_[target_idx]){ X_[target_idx]=xloc; FX_[target_idx]=floc; }
            }

            // update global best
            if (FX_[target_idx] < best_f_){
                best_f_ = FX_[target_idx];
                best_x_ = X_[target_idx];
            }

            // Success-history gain from the actual parent to the new fitness
            double gain = std::max(1e-12, parent_f - FX_[target_idx]);
            sF.push_back(F); sCR.push_back(CR); wGain.push_back(gain);
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Success-History (global)
    if (!sF.empty()){
        double sumw=0.0; for (double w:wGain) sumw+=w;
        if (sumw>0.0){
            double wmCR=0.0, wmF=0.0, wmF2=0.0;
            for (size_t k=0;k<sF.size();++k){
                double wk = wGain[k]/sumw;
                wmCR += wk*sCR[k];
                wmF  += wk*sF[k];
                wmF2 += wk*sF[k]*sF[k];
            }
            double LmeanF = (wmF>1e-12)? (wmF2/wmF) : wmF;

            muCR_ = (1.0 - sh_c_)*muCR_ + sh_c_ * std::clamp(wmCR, CR_lo_, CR_hi_);
            muF_  = (1.0 - sh_c_)*muF_  + sh_c_ * std::clamp(LmeanF, F_lo_, F_hi_);
        }
    }

    // Outlier quarantine
    quarantineOutliers_();

    // slide window
    start_agent_ = e;
    if (start_agent_ >= N) start_agent_ = 0;

    // Stagnation → micro-restart
    if (best_f_ < prevBest - 1e-10) {
        stagn_iters_ = 0;
    } else {
        stagn_iters_++;
    }
    if (stagn_iters_ >= stagnation_trigger_){
        microRestart_();
        stagn_iters_ = 0;
    }

    // progress / stop
    printBest();
    updateStop(FX_);
}

// -----------------------------------------------------------------------------
// end
// -----------------------------------------------------------------------------

void ARQ::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_){
        best_f_ = floc; best_x_ = xloc;
    }

    // inject best into worst
    if (!X_.empty() && !FX_.empty()){
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1;k<FX_.size();++k){ if (FX_[k] > fw){ fw=FX_[k]; worst=k; } }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    //
    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
