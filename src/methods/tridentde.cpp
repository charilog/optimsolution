#include "tridentde.h"

namespace optimsolution {

void TRIDENTDE::configure(const MethodConfig& mc)
{
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    trials_per_agent   = mc.getInt("trials_per_agent",   trials_per_agent);
    agent_fraction     = mc.getDbl("agent_fraction",     agent_fraction);

    tauF               = mc.getDbl("tauF",               tauF);
    tauCR              = mc.getDbl("tauCR",              tauCR);
    F_lo               = mc.getDbl("F_lo",               F_lo);
    F_hi               = mc.getDbl("F_hi",               F_hi);
    CR_lo              = mc.getDbl("CR_lo",              CR_lo);
    CR_hi              = mc.getDbl("CR_hi",              CR_hi);

    pbest_frac         = mc.getDbl("pbest_frac",         pbest_frac);

    stagnation_trigger = mc.getInt("stagnation_trigger", stagnation_trigger);
    restart_frac       = mc.getDbl("restart_frac",       restart_frac);
    restart_sigma      = mc.getDbl("restart_sigma",      restart_sigma);

    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    if (agent_fraction <= 0.0) agent_fraction = 0.05;
    if (agent_fraction > 1.0)  agent_fraction = 1.0;
    if (pbest_frac <= 0.0)     pbest_frac = 0.05;
    if (pbest_frac >  0.9)     pbest_frac = 0.9;
    if (F_lo < 0.0) F_lo = 0.0;
    if (F_hi < F_lo) F_hi = F_lo + 1e-9;
    if (CR_lo < 0.0) CR_lo = 0.0;
    if (CR_hi > 1.0) CR_hi = 1.0;
}

// --- Safe evaluation ---
double TRIDENTDE::eval_safe(const Vec& x)
{
    double v = prob_->evaluate(x);
    if (!std::isfinite(v)) v = 1e100;
    return v;
}

TRIDENTDE::Vec TRIDENTDE::clamp_to_bounds(const Vec& x) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    Vec y = x;
    for (int j = 0; j < D_; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (y[j] < lo) y[j] = lo;
        if (y[j] > hi) y[j] = hi;
    }
    return y;
}

inline double TRIDENTDE::range_j(int j) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double lo = (j < (int)L.size() ? L[j] : -1.0);
    double hi = (j < (int)U.size() ? U[j] :  1.0);
    if (lo > hi) std::swap(lo, hi);
    return (hi - lo);
}

// NEW: as in BHO, ensures that best_f_ is finite.
void TRIDENTDE::ensure_finite_best()
{
    if (std::isfinite(best_f_)) return;
    Vec mid(D_, 0.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int j = 0; j < D_; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        mid[j] = 0.5*(lo+hi);
    }
    double v = eval_safe(mid);
    best_f_ = v;
    best_x_ = mid;
}

// ---------------- Operators (bin crossover) ----------------

void TRIDENTDE::make_trial_best1(int i, const Vec& xi, Vec& trial, double F, double CR)
{
    if (N_ < 3) { trial = xi; return; }
    int r1, r2;
    do { r1 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r1==i);
    do { r2 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r2==i || r2==r1);

    trial = xi;
    int jrand = (D_>0) ? std::uniform_int_distribution<int>(0, D_-1)(rng_) : 0;
    for (int j=0;j<D_;++j){
        if (U01_(rng_) < CR || j==jrand){
            double v = best_x_[j] + F*(X_[r1][j] - X_[r2][j]);
            trial[j] = v;
        }
    }
    trial = clamp_to_bounds(trial);
}

void TRIDENTDE::make_trial_ctobest1(int i, const Vec& xi, Vec& trial, double F, double CR)
{
    if (N_ < 3) { trial = xi; return; }
    int r1, r2;
    do { r1 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r1==i);
    do { r2 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r2==i || r2==r1);

    trial = xi;
    int jrand = (D_>0) ? std::uniform_int_distribution<int>(0, D_-1)(rng_) : 0;
    for (int j=0;j<D_;++j){
        if (U01_(rng_) < CR || j==jrand){
            double base = xi[j] + F*(best_x_[j] - xi[j]) + F*(X_[r1][j] - X_[r2][j]);
            trial[j] = base;
        }
    }
    trial = clamp_to_bounds(trial);
}

void TRIDENTDE::make_trial_pbest1(int i, const Vec& xi, Vec& trial, double F, double CR)
{
    if (N_ < 3) { trial = xi; return; }
    int pnum = std::max(1, (int)std::round(pbest_frac * N_));
    std::vector<int> idx(N_); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+pnum, idx.end(),
        [&](int a,int b){ return FX_[a] < FX_[b]; });
    int pbest = idx[ std::uniform_int_distribution<int>(0, pnum-1)(rng_) ];

    int r1, r2;
    do { r1 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r1==i || r1==pbest);
    do { r2 = std::uniform_int_distribution<int>(0, N_-1)(rng_); } while (r2==i || r2==r1 || r2==pbest);

    trial = xi;
    int jrand = (D_>0) ? std::uniform_int_distribution<int>(0, D_-1)(rng_) : 0;
    for (int j=0;j<D_;++j){
        if (U01_(rng_) < CR || j==jrand){
            double v = xi[j] + F*(X_[pbest][j] - xi[j]) + F*(X_[r1][j] - X_[r2][j]);
            trial[j] = v;
        }
    }
    trial = clamp_to_bounds(trial);
}

// trial−base (lambda∈{1,0.5})
bool TRIDENTDE::line_refine(const Vec& base, const Vec& trial_in, Vec& trial_out)
{
    Vec y = base;
    for (int j=0;j<D_;++j) y[j] = base[j] + 1.0*(trial_in[j] - base[j]);
    y = clamp_to_bounds(y);
    double f1 = eval_safe(y);

    Vec z = base;
    for (int j=0;j<D_;++j) z[j] = base[j] + 0.5*(trial_in[j] - base[j]);
    z = clamp_to_bounds(z);
    double f2 = eval_safe(z);

    trial_out = (f1 <= f2) ? y : z;
    return true;
}

int TRIDENTDE::eliteIndex_() const
{
    int idx=0; double bf=FX_[0];
    for (int i=1;i<N_;++i) if (FX_[i] < bf){ bf=FX_[i]; idx=i; }
    return idx;
}

void TRIDENTDE::microRestart_()
{
    int nreset = std::max(1, (int)std::round(restart_frac * N_));
    std::vector<int> idx(N_); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a,int b){ return FX_[a] < FX_[b]; });

    const int elite = idx[0];
    for (int k=0; k<nreset; ++k){
        int i = idx[ N_-1 - k ];
        if (i == elite) continue;

        Vec cand = X_[i];
        for (int j=0;j<D_;++j){
            double step = restart_sigma * range_j(j) * N01_(rng_);
            cand[j] = best_x_[j] + step;
        }
        cand = clamp_to_bounds(cand);
        double f = eval_safe(cand);
        if (f < FX_[i]){
            X_[i]=cand; FX_[i]=f;
            if (f < best_f_){ best_f_=f; best_x_=cand; }
        }
    }
}

// ---------------- Lifecycle ----------------

void TRIDENTDE::init()
{
    if (!prob_) return;

    D_ = prob_->dimension();
    N_ = std::max(4, population());

    DIpop_ = std::uniform_int_distribution<int>(0, std::max(0, N_-1));
    DIdim_ = std::uniform_int_distribution<int>(0, std::max(0, D_-1));

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N_);

    FX_.assign(N_, std::numeric_limits<double>::infinity());
    for (int i=0;i<N_;++i) FX_[i] = eval_safe(X_[i]);

    int elite = eliteIndex_();
    best_x_ = X_[elite];          // Comment translated from Greek.
    best_f_ = FX_[elite];

    ensure_finite_best();         // <-- guarantee, as in BHO.

    Fi_.assign(N_, 0.9);
    CRi_.assign(N_, 0.7);

    iters_ = 0;
    startAgent_ = 0;
    stagn_iters_ = 0;

    printBest();
}

void TRIDENTDE::one_iteration()
{
    if (!prob_) return;
    ++iters_;

    int elite = eliteIndex_();
    if (FX_[elite] < best_f_) { best_f_ = FX_[elite]; best_x_ = X_[elite]; }
    double prevBest = best_f_;

    int batch = std::max(1, (int)std::floor(agent_fraction * N_));
    int s = startAgent_;
    int e = std::min(N_, s + batch);

    std::uniform_real_distribution<double> U01(0.0,1.0);

    for (int i=s; i<e; ++i){
        if (i==elite) continue;

        if (U01(rng_) < tauF)  Fi_[i]  = F_lo  + (F_hi  - F_lo) * U01(rng_);
        if (U01(rng_) < tauCR) CRi_[i] = CR_lo + (CR_hi - CR_lo)* U01(rng_);

        const Vec x = X_[i];
        const double fx = FX_[i];

        Vec bestT = x; double bestTf = std::numeric_limits<double>::infinity();

        for (int t=0; t<trials_per_agent; ++t){
            Vec T;
            int v = (iters_ + i + t) % 3; // Comment translated from Greek.
            if (v==0)      make_trial_best1   (i, x, T, Fi_[i], CRi_[i]);
            else if (v==1) make_trial_ctobest1(i, x, T, Fi_[i], CRi_[i]);
            else           make_trial_pbest1  (i, x, T, Fi_[i], CRi_[i]);

            double fT = eval_safe(T);
            if (fT < bestTf){ bestTf=fT; bestT=T; }
        }

        Vec Tref;
        (void)line_refine(x, bestT, Tref);
        double fRef = eval_safe(Tref);
        if (fRef < bestTf){ bestTf = fRef; bestT = Tref; }

        if (bestTf < fx){
            Vec y = bestT; double fy = bestTf;

            if (local_rate_ > 0.0 && !local_method_.empty() && local_method_!="none" && U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, y);
                if (floc < fy) { y = std::move(xloc); fy = floc; }
            }

            X_[i] = y; FX_[i] = fy;
            if (fy < best_f_) { best_f_ = fy; best_x_ = y; }
        } else {
            int worst = 0; double wf = FX_[0];
            for (int k=1;k<N_;++k) if (FX_[k] > wf){ wf = FX_[k]; worst = k; }
            if (bestTf < wf && U01(rng_) < 0.20){
                X_[worst] = bestT; FX_[worst] = bestTf;
                if (bestTf < best_f_){ best_f_ = bestTf; best_x_ = bestT; }
            }
        }

        if (prob_->calls() >= max_evals_) {
            printBest();
            updateStop(FX_);
            return;
        }
    }

    startAgent_ = e;
    if (startAgent_ >= N_) startAgent_ = 0;

    if (best_f_ + 1e-18 < prevBest) stagn_iters_ = 0; else stagn_iters_++;
    if (stagn_iters_ >= stagnation_trigger){
        microRestart_();
        stagn_iters_ = 0;
    }

    printBest();
    updateStop(FX_);
}

void TRIDENTDE::end()
{
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    if (!X_.empty() && !FX_.empty()){
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > fw){ fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
