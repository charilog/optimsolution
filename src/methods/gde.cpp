#include "gde.h"

#include <limits>
#include <numeric>

namespace optimsolution {

/* ============================================================
 * CONFIGURE
 * ============================================================ */
void GDE::configure(const MethodConfig &mc)
{
    // Reads the main parameters.
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

    // sanity bounds
    if (F_lo_ <= 0.0)   F_lo_ = 0.1;
    if (F_hi_ <= F_lo_) F_hi_ = 1.2;
    if (CR_hi_ <= CR_lo_) {
        CR_lo_ = 0.0;
        CR_hi_ = 1.0;
    }

    sh_c_ = mc.getDbl("sh_c", sh_c_);
    if (sh_c_ <= 0.0 || sh_c_ > 1.0) sh_c_ = 0.1;

    archive_max_ = static_cast<std::size_t>(
        mc.getInt("archive_max", static_cast<int>(archive_max_)));
    if (archive_max_ < 20) archive_max_ = 20;

    micro_restart_period_ = mc.getInt("micro_restart_period", micro_restart_period_);
    if (micro_restart_period_ < 10) micro_restart_period_ = 10;

    quarantine_period_ = mc.getInt("quarantine_period", quarantine_period_);
    if (quarantine_period_ < 0) quarantine_period_ = 0;

    iter_ = 0;
    no_improv_iters_ = 0;
}

/* ============================================================
 * INIT
 * ============================================================ */
void GDE::init()
{
    if (!prob_) return;

    const int D = prob_->dimension();

    N_ = population();
    if (N_ < 4)        N_ = std::max(pop_init_, 4);
    if (N_ < pop_min_) N_ = pop_min_;

    Optimizer::setPopulation(N_);

    // initialization using the framework's standard Initializer
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N_);
    N_ = static_cast<int>(X_.size());

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

    iter_ = 0;
    no_improv_iters_ = 0;
}

/* ============================================================
 * ensureBounds (mirror + clamp)
 * ============================================================ */
void GDE::ensureBounds(Vec &x) const
{
    if (!prob_) return;
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();
    const int D   = static_cast<int>(x.size());

    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j]))
            x[j] = 0.5 * (L[j] + U[j]);

        // mirror correction until the point is within bounds
        while (x[j] < L[j] || x[j] > U[j]) {
            if (x[j] > U[j])
                x[j] = 2.0 * U[j] - x[j];
            else
                x[j] = 2.0 * L[j] - x[j];
        }
    }
}

/* ============================================================
 * Archive helpers
 * ============================================================ */
void GDE::pushArchive_(const Vec &x)
{
    if (archive_max_ == 0) return;

    if (archive_.size() < archive_max_) {
        archive_.push_back(x);
    } else {
        int up = static_cast<int>(archive_.size()) - 1;
        if (up < 0) up = 0;
        std::uniform_int_distribution<int> dist(0, up);
        archive_[dist(rng_)] = x;
    }
}

int GDE::pickPbestIndex_(double pfrac)
{
    if (N_ <= 0) return 0;
    if (pfrac <= 0.0) pfrac = 0.2;
    if (pfrac > 1.0)  pfrac = 1.0;

    int pN = static_cast<int>(std::round(pfrac * N_));
    if (pN < 2)  pN = std::min(N_, 2);
    if (pN > N_) pN = N_;

    std::vector<int> idx(N_);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> dist(0, pN - 1);
    return idx[dist(rng_)];
}

/* ============================================================
 * microRestart_: restart on the worst 20%
 * ============================================================  */
void GDE::microRestart_()
{
    if (!prob_ || N_ < 4) return;

    std::vector<int> idx(N_);
    std::iota(idx.begin(), idx.end(), 0);

    // worst-first
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] > FX_[b]; });

    int n_restart = std::max(1, N_ / 5);

    Initializer initSampler;
    initSampler.configure(initopt_);
    auto news = initSampler.samplePopulation(*prob_, rng_, n_restart);

    int limit = std::min(n_restart, static_cast<int>(news.size()));

    for (int k = 0; k < limit; ++k) {
        int i  = idx[k];
        Vec xi = news[k];

        ensureBounds(xi);
        double fi = prob_->evaluate(xi);

        pushArchive_(X_[i]);
        X_[i]  = xi;
        FX_[i] = fi;

        if (fi < best_f_) {
            best_f_ = fi;
            best_x_ = xi;
        }
    }
}

/* ============================================================
 * quarantineOutliers_: replacement of very poor individuals
 * ============================================================  */
void GDE::quarantineOutliers_()
{
    if (!prob_ || N_ < 4) return;

    double mean = 0.0;
    for (double f : FX_) mean += f;
    mean /= N_;

    double var = 0.0;
    for (double f : FX_) {
        double d = f - mean;
        var += d * d;
    }
    var /= N_;
    double sd = std::sqrt(std::max(0.0, var));
    if (sd <= 0.0) return;

    double thresh = mean + 2.5 * sd;

    Initializer initSampler;
    initSampler.configure(initopt_);

    for (int i = 0; i < N_; ++i) {
        if (FX_[i] > thresh) {
            auto newOne = initSampler.samplePopulation(*prob_, rng_, 1);
            if (newOne.empty()) continue;

            Vec xi = newOne[0];
            ensureBounds(xi);
            double fi = prob_->evaluate(xi);

            pushArchive_(X_[i]);
            X_[i]  = xi;
            FX_[i] = fi;

            if (fi < best_f_) {
                best_f_ = fi;
                best_x_ = xi;
            }
        }
    }
}

/* ============================================================
 * trial_pbest1A_bin_: DE/pbest/1/bin with archive
 * ============================================================  */
void GDE::trial_pbest1A_bin_(int i,
                             const Vec &xi,
                             Vec &tr,
                             double F,
                             double CR,
                             double pfrac,
                             bool useArchive)
{
    int D = prob_->dimension();

    int pbest = pickPbestIndex_(pfrac);

    std::uniform_int_distribution<int> dist(0, N_ - 1);
    int r1;
    do { r1 = dist(rng_); } while (r1 == i || r1 == pbest);

    Vec base_r2(D, 0.0);
    if (useArchive && !archive_.empty()) {
        int ia = std::uniform_int_distribution<int>(
            0, static_cast<int>(archive_.size()) - 1)(rng_);
        base_r2 = archive_[ia];
    } else {
        int r2;
        do { r2 = dist(rng_); } while (r2 == i || r2 == pbest || r2 == r1);
        base_r2 = X_[r2];
    }

    tr = xi;

    std::uniform_int_distribution<int> jdist(0, D - 1);
    int jrand = jdist(rng_);
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    for (int j = 0; j < D; ++j) {
        if (ur(rng_) < CR || j == jrand) {
            tr[j] = xi[j]
                  + F * (X_[pbest][j] - xi[j])
                  + F * (X_[r1][j]   - base_r2[j]);
        }
    }

    ensureBounds(tr);
}

/* ============================================================
 * one_iteration(): single-strategy self-adaptive DE/pbest/1+archive
 * ============================================================ */
void GDE::one_iteration()
{
    if (!prob_ || N_ < 4 || terminated()) return;

    ++iter_;

    const int D = prob_->dimension();

    double prev_best = best_f_;

    std::vector<Vec>    newX(N_, Vec(D, 0.0));
    std::vector<double> newF(N_, 0.0);

    int    nSuccess = 0;
    double sumF     = 0.0;
    double sumCR    = 0.0;

    std::cauchy_distribution<double> cF(muF_, 0.1);
    std::normal_distribution<double> nCR(muCR_, 0.1);

    for (int i = 0; i < N_; ++i) {
        const Vec &xi = X_[i];

        double Fi;
        // Cauchy sampling for F until a positive value is obtained
        do { Fi = cF(rng_); } while (Fi <= 0.0);
        Fi = std::clamp(Fi, F_lo_, F_hi_);

        double CRi = std::clamp(nCR(rng_), CR_lo_, CR_hi_);

        Vec tr(D, 0.0);
        trial_pbest1A_bin_(i, xi, tr, Fi, CRi, 0.2, true);

        double ftrial = prob_->evaluate(tr);

        if (ftrial <= FX_[i]) {
            newX[i] = tr;
            newF[i] = ftrial;
            ++nSuccess;

            sumF  += Fi;
            sumCR += CRi;

            pushArchive_(X_[i]);

            if (ftrial < best_f_) {
                best_f_ = ftrial;
                best_x_ = tr;
            }
        } else {
            newX[i] = X_[i];
            newF[i] = FX_[i];
        }
    }

    X_.swap(newX);
    FX_.swap(newF);

    // updates muF/muCR using a simple EWMA
    if (nSuccess > 0) {
        double mF  = sumF  / std::max(1, nSuccess);
        double mCR = sumCR / std::max(1, nSuccess);
        muF_  = (1.0 - sh_c_) * muF_  + sh_c_ * mF;
        muCR_ = (1.0 - sh_c_) * muCR_ + sh_c_ * mCR;
    }

    // stagnation control
    if (best_f_ < prev_best) no_improv_iters_ = 0;
    else ++no_improv_iters_;

    if (micro_restart_period_ > 0 && no_improv_iters_ >= micro_restart_period_) {
        microRestart_();
        no_improv_iters_ = 0;
    }

    if (quarantine_period_ > 0 && (iter_ % quarantine_period_) == 0) {
        quarantineOutliers_();
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
