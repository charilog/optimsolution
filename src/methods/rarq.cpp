#include "rarq.h"

#include <cmath>
#include <limits>

namespace optimsolution {

/* ============================================================
 * CONFIGURE
 * ============================================================ */
void RARQ::configure(const MethodConfig &mc)
{
    int p = mc.getInt("population", pop_init_);
    if (p > 3) {
        pop_init_ = p;
        Optimizer::setPopulation(pop_init_);
    }

    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;

    muF_  = mc.getDbl("muF",  muF_);
    muCR_ = mc.getDbl("muCR", muCR_);

    F_lo_  = mc.getDbl("F_lo",  F_lo_);
    F_hi_  = mc.getDbl("F_hi",  F_hi_);
    CR_lo_ = mc.getDbl("CR_lo", CR_lo_);
    CR_hi_ = mc.getDbl("CR_hi", CR_hi_);

    if (F_lo_ <= 0.0)  F_lo_ = 0.1;
    if (F_hi_ <= F_lo_) F_hi_ = 1.0;
    if (CR_hi_ <= CR_lo_) {
        CR_lo_ = 0.0;
        CR_hi_ = 1.0;
    }

    n0_    = mc.getDbl("roul_n0",    n0_);
    delta_ = mc.getDbl("roul_delta", delta_);
    if (n0_   <= 0.0) n0_   = 2.0;
    if (delta_ <= 0.0) delta_ = 0.05;

    archive_max_ = (std::size_t)mc.getInt("archive_max", (int)archive_max_);
    if (archive_max_ < 10) archive_max_ = 10;

    micro_restart_period_ = mc.getInt("micro_restart_period", micro_restart_period_);
    if (micro_restart_period_ < 10) micro_restart_period_ = 10;

    local_method_ = mc.getStr("local_method", local_method_);
    local_rate_   = mc.getDbl("local_rate",   local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    resetRoulette();
    iter_ = 0;
    no_improv_iters_ = 0;
}

/* ============================================================
 * INIT
 * ============================================================ */
void RARQ::init()
{
    if (!prob_) return;

    const int D = prob_->dimension();

    N_ = population();
    if (N_ < 4) N_ = std::max(pop_init_, 4);
    if (N_ < pop_min_) N_ = pop_min_;

    Optimizer::setPopulation(N_);

    // Use Initializer (same method as EA4Eig / ARQ)
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N_);
    N_ = (int)X_.size();

    FX_.assign(N_, std::numeric_limits<double>::infinity());
    archive_.clear();

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < N_; ++i) {
        ensureBounds(X_[i]);
        FX_[i] = prob_->evaluate(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
    }

    resetRoulette();
    iter_ = 0;
    no_improv_iters_ = 0;
}

/* ============================================================
 * ensureBounds (no-op)
 * ============================================================ */
void RARQ::ensureBounds(Vec &) const
{
    
}

/* ============================================================
 * Roulette selection
 * ============================================================ */
std::pair<int,double> RARQ::rouletteSelect()
{
    double sum = 0.0;
    for (int h = 0; h < H_; ++h)
        if (ni_[h] > 0.0) sum += ni_[h];

    if (sum <= 0.0) {
        std::uniform_int_distribution<int> dist(0, H_-1);
        int h = dist(rng_);
        return { h, 1.0 / static_cast<double>(H_) };
    }

    std::uniform_real_distribution<double> ur(0.0, 1.0);
    double r = ur(rng_);
    double acc = 0.0;
    double pmin = 1.0;
    int hsel = H_-1;

    for (int h=0; h<H_; ++h) {
        double ph = ni_[h] / sum;
        if (ph < pmin) pmin = ph;
        acc += ph;
        if (r <= acc) {
            hsel = h;
            break;
        }
    }
    return {hsel, pmin};
}

void RARQ::resetRoulette()
{
    for (int h = 0; h < H_; ++h) {
        ni_[h]      = n0_;
        success_[h] = 0.0;
    }
    ++nrst_;
}

/* ============================================================
 * Archive helpers
 * ============================================================ */
void RARQ::pushArchive_(const Vec &x)
{
    if (archive_max_ == 0) return;
    if (archive_.size() < archive_max_) archive_.push_back(x);
    else {
        std::uniform_int_distribution<int> dist(0, (int)archive_.size()-1);
        archive_[dist(rng_)] = x;
    }
}

int RARQ::pickArchiveIndex_()
{
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> dist(0, (int)archive_.size()-1);
    return dist(rng_);
}

int RARQ::pickPbestIndex_(double pfrac)
{
    if (N_ == 0) return 0;
    if (pfrac <= 0.0) pfrac = 0.2;
    if (pfrac >  1.0) pfrac = 1.0;

    int pN = (int)std::round(pfrac * N_);
    if (pN < 2) pN = std::min(N_, 2);
    if (pN > N_) pN = N_;

    std::vector<int> idx(N_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a,int b){ return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> dist(0, pN-1);
    return idx[dist(rng_)];
}

double RARQ::bnDistance_(const Vec &a,const Vec &b) const
{
    double s=0.0;
    const std::size_t D = std::min(a.size(), b.size());
    for (std::size_t i=0;i<D;++i){
        double d=a[i]-b[i];
        s+=d*d;
    }
    return std::sqrt(s);
}

int RARQ::pickRTRNeighbor_(const Vec &trial,
                           const std::vector<int> &pool) const
{
    if (pool.empty()) return -1;

    int    bestIdx = pool[0];
    double bestD   = bnDistance_(trial, X_[bestIdx]);

    for (std::size_t i=1;i<pool.size();++i) {
        int idx = pool[i];
        double d = bnDistance_(trial, X_[idx]);
        if (d < bestD) { bestD = d; bestIdx = idx; }
    }
    return bestIdx;
}

/* ============================================================
 * quarantineOutliers_()
 * ============================================================ */
void RARQ::quarantineOutliers_()
{
    if (!prob_ || N_ < 4) return;

    double mean=0.0;
    for (double f:FX_) mean+=f;
    mean /= N_;

    double var=0.0;
    for (double f:FX_) {
        double d=f-mean; var+=d*d;
    }
    var/=N_;
    double sd=std::sqrt(std::max(0.0,var));
    if (sd<=0.0) return;

    double thresh = mean + 2.0*sd;

    Initializer initSampler;
    initSampler.configure(initopt_);

    auto newOne = initSampler.samplePopulation(*prob_, rng_, 1);

    for (int i=0;i<N_;++i){
        if (FX_[i] > thresh && !newOne.empty()) {
            Vec xi = newOne[0];
            ensureBounds(xi);
            double fi = prob_->evaluate(xi);
            pushArchive_(X_[i]);
            X_[i]  = xi;
            FX_[i] = fi;
            if (fi < best_f_) { best_f_ = fi; best_x_ = xi; }
        }
    }
}

/* ============================================================
 * microRestart_()
 * ============================================================ */
void RARQ::microRestart_()
{
    if (!prob_ || N_ < 4) return;

    std::vector<int> idx(N_);
    std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),
              [&](int a,int b){ return FX_[a] > FX_[b]; }); // worst first

    int n_restart = std::max(1, N_/10);

    Initializer initSampler;
    initSampler.configure(initopt_);
    auto newPts = initSampler.samplePopulation(*prob_, rng_, n_restart);

    int L = (int)newPts.size();
    for (int k=0;k<n_restart && k<L;++k) {
        int i = idx[k];
        Vec xi = newPts[k];
        ensureBounds(xi);
        double fi = prob_->evaluate(xi);
        pushArchive_(X_[i]);
        X_[i]  = xi;
        FX_[i] = fi;
        if (fi < best_f_) { best_f_ = fi; best_x_ = xi; }
    }
}

/* ============================================================
 * trial_pbest1A_bin_
 * ============================================================ */
void RARQ::trial_pbest1A_bin_(int i,
                              const Vec &xi,
                              Vec &tr,
                              double F,
                              double CR,
                              double pfrac,
                              bool useArchive)
{
    int D = prob_->dimension();

    int pbest = pickPbestIndex_(pfrac);

    std::uniform_int_distribution<int> dist(0,N_-1);
    int r1;
    do { r1 = dist(rng_); } while (r1==i || r1==pbest);

    Vec base_r2(D,0.0);
    if (useArchive && !archive_.empty()) {
        int ia = pickArchiveIndex_();
        if (ia >= 0) base_r2 = archive_[ia];
        else         base_r2 = X_[r1];
    } else {
        int r2;
        do { r2 = dist(rng_); }
        while (r2==i || r2==pbest || r2==r1);
        base_r2 = X_[r2];
    }

    tr = xi;

    std::uniform_int_distribution<int> jdist(0, D-1);
    int jrand = jdist(rng_);
    std::uniform_real_distribution<double> ur(0.0,1.0);

    for (int j=0;j<D;++j){
        if (ur(rng_) < CR || j==jrand) {
            tr[j] = xi[j]
                  + F*(X_[pbest][j] - xi[j])
                  + F*(X_[r1][j]    - base_r2[j]);
        }
    }

    ensureBounds(tr);
}

/* ============================================================
 * STRATEGY 0: Core ARQ
 * ============================================================ */
void RARQ::stepCore()
{
    int D = prob_->dimension();

    std::vector<Vec> newX(N_, Vec(D,0.0));
    std::vector<double> newF(N_,0.0);

    int nSuccess = 0;
    double sF=0.0, sCR=0.0;

    std::cauchy_distribution<double> cF(muF_,0.1);
    std::normal_distribution<double> nCR(muCR_,0.1);

    for (int i=0;i<N_;++i){
        const Vec &xi = X_[i];

        double Fi;
        do { Fi = cF(rng_); } while (Fi <= 0.0);
        Fi = std::clamp(Fi, F_lo_, F_hi_);

        double CRi = std::clamp(nCR(rng_), CR_lo_, CR_hi_);

        Vec tr(D,0.0);
        trial_pbest1A_bin_(i, xi, tr, Fi, CRi, 0.2, true);

        double ftrial = prob_->evaluate(tr);

        if (ftrial <= FX_[i]) {
            newX[i] = tr;
            newF[i] = ftrial;
            ++nSuccess;
            sF  += Fi;
            sCR += CRi;
            pushArchive_(X_[i]);
            if (ftrial < best_f_) { best_f_ = ftrial; best_x_ = tr; }
        } else {
            newX[i] = X_[i];
            newF[i] = FX_[i];
        }
    }

    X_.swap(newX);
    FX_.swap(newF);

    if (nSuccess>0){
        double mF  = sF  / nSuccess;
        double mCR = sCR / nSuccess;
        muF_  = (1.0-sh_c_)*muF_  + sh_c_*mF;
        muCR_ = (1.0-sh_c_)*muCR_ + sh_c_*mCR;
    }

    success_[0] += nSuccess;
    ni_[0]      += nSuccess;

    if ((iter_%10)==0) quarantineOutliers_();
}

/* ============================================================
 * STRATEGY 1: Intensify
 * ============================================================ */
void RARQ::stepIntensify()
{
    int D = prob_->dimension();

    std::vector<Vec> newX(N_,Vec(D,0.0));
    std::vector<double> newF(N_,0.0);

    int nSuccess = 0;

    std::normal_distribution<double> nF(0.4,0.1);
    std::normal_distribution<double> nCR(0.9,0.05);

    for (int i=0;i<N_;++i){
        const Vec &xi = X_[i];

        double Fi  = std::clamp(nF(rng_), 0.1,0.8);
        double CRi = std::clamp(nCR(rng_),0.5,1.0);

        Vec tr(D,0.0);
        // Tighter p-best (exploitation) without archive
        trial_pbest1A_bin_(i, xi, tr, Fi, CRi, 0.1, false);

        double ftrial = prob_->evaluate(tr);

        if (ftrial < FX_[i]){
            newX[i] = tr;
            newF[i] = ftrial;
            ++nSuccess;
            if (ftrial < best_f_) { best_f_ = ftrial; best_x_ = tr; }
        } else {
            newX[i]=X_[i]; newF[i]=FX_[i];
        }
    }

    X_.swap(newX);
    FX_.swap(newF);

    success_[1]+=nSuccess;
    ni_[1]+=nSuccess;
}

/* ============================================================
 * STRATEGY 2: Explore
 * ============================================================ */
void RARQ::stepExplore()
{
    int D = prob_->dimension();

    if (no_improv_iters_ > micro_restart_period_){
        microRestart_();
        no_improv_iters_ = 0;
    }

    std::vector<Vec> newX(N_,Vec(D,0.0));
    std::vector<double> newF(N_,0.0);

    int nSuccess=0;

    std::cauchy_distribution<double> cF(0.9,0.2);
    std::normal_distribution<double> nCR(0.5,0.15);
    std::uniform_real_distribution<double> ur(0.0,1.0);

    for (int i=0;i<N_;++i){
        const Vec &xi = X_[i];

        double Fi;
        do { Fi=cF(rng_); } while (Fi<=0.0);
        Fi = std::clamp(Fi,0.5,1.5);

        double CRi = std::clamp(nCR(rng_),0.1,0.9);

        Vec tr(D,0.0);
        // relaxed p-best (p=0.5) + archive → more exploration
        trial_pbest1A_bin_(i, xi, tr, Fi, CRi, 0.5, true);

        double ftrial = prob_->evaluate(tr);

        if (ftrial <= FX_[i]){
            newX[i] = tr;
            newF[i] = ftrial;
            ++nSuccess;
            if (ftrial < best_f_) { best_f_ = ftrial; best_x_ = tr; }
        } else {
            newX[i] = X_[i]; newF[i] = FX_[i];
            if (ur(rng_) < 0.3) pushArchive_(xi);
        }
    }

    X_.swap(newX);
    FX_.swap(newF);

    success_[2]+=nSuccess;
    ni_[2]+=nSuccess;
}

/* ============================================================
 * one_iteration()
 * ============================================================ */
void RARQ::one_iteration()
{
    if (!prob_ || N_<4 || terminated()) return;

    ++iter_;

    auto sel = rouletteSelect();
    int h = sel.first;
    double pmin = sel.second;

    if (pmin < delta_) resetRoulette();

    double prev_best = best_f_;

    switch(h){
        case 0: stepCore();      break;
        case 1: stepIntensify(); break;
        case 2:
        default: stepExplore();  break;
    }

    if (best_f_ < prev_best) no_improv_iters_ = 0;
    else ++no_improv_iters_;

    // In-run local search (method-specific, as in EA4Eig)
    if (local_rate_>0.0 && !local_method_.empty()) {
        std::uniform_real_distribution<double> ur(0.0,1.0);
        for (int i=0;i<N_;++i){
            if (ur(rng_) > local_rate_) continue;

            auto res = localSearch(local_method_, X_[i]);
            const Vec &xr = res.first;
            double fr      = res.second;

            if (!xr.empty() && std::isfinite(fr) && fr < FX_[i]) {
                X_[i]  = xr;
                FX_[i] = fr;
                if (fr < best_f_) { best_f_ = fr; best_x_ = xr; }
            }
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
