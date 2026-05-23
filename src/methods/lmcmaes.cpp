#include "lmcmaes.h"
#include "init.h"
#include "options.h"

#include <numeric>   // iota, accumulate
#include <cassert>

namespace optimsolution {

// =========================================================================
//  configure
// =========================================================================
void LMCMAES::configure(const MethodConfig& mc)
{
    lambda_cfg_ = mc.getInt("lambda",     lambda_cfg_);
    pop_cfg_    = mc.getInt("population", pop_cfg_);

    sigma0_ = mc.getDbl("sigma0", sigma0_);
    if (sigma0_ <= 0.0) sigma0_ = 0.3;

    int mcfg = mc.getInt("m_vectors", -1);
    if (mcfg > 0) m_vec_ = mcfg;

    // PSR parameters
    c_sigma_ = mc.getDbl("c_sigma", c_sigma_);
    {
        double d_cfg = mc.getDbl("d_sigma", -1.0);
        if (d_cfg > 0.0) { d_sigma_ = d_cfg; d_sigma_from_cfg_ = true; }
    }
    zstar_ = mc.getDbl("zstar", zstar_);

    // in-run local search
    local_method_ = mc.getStr("local_method", local_method_);
    local_rate_   = mc.getDbl("local_rate",   local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    // final local refinement
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method",  end_local_method_);
}

// =========================================================================
//  init
// =========================================================================
void LMCMAES::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    iter_     = 0;
    n_stored_ = 0;
    s_        = 0.0;

    // ---- population size ----
    const int lambda_auto =
        std::max(4, 4 + static_cast<int>(3.0 * std::log(static_cast<double>(D))));

    if      (lambda_cfg_ > 0) lambda_ = lambda_cfg_;
    else if (pop_cfg_    > 0) lambda_ = pop_cfg_;
    else                      lambda_ = lambda_auto;
    if (lambda_ < 4) lambda_ = 4;

    mu_ = lambda_ / 2;
    if (mu_ < 1) mu_ = 1;

    // ---- weights: w_i = (ln(mu+1) - ln(i)) / (mu*ln(mu+1) - Σln(j))
    //   Loshchilov 2014, Algorithm 6 ----
    w_.resize(mu_);
    {
        double denom = 0.0;
        for (int i = 1; i <= mu_; ++i) denom += std::log(static_cast<double>(i));
        denom = mu_ * std::log(mu_ + 1.0) - denom;
        for (int i = 1; i <= mu_; ++i)
            w_[i-1] = (std::log(mu_ + 1.0) - std::log(static_cast<double>(i))) / denom;
        // normalize to sum exactly to 1
        double wsum = 0.0;
        for (double wi : w_) wsum += wi;
        for (double& wi : w_) wi /= wsum;
    }
    {
        double wsq = 0.0;
        for (double wi : w_) wsq += wi * wi;
        mu_eff_ = 1.0 / wsq;
    }

    // ---- algorithm parameters (Algorithm 7, Loshchilov 2017) ----
    if (m_vec_ <= 0) m_vec_ = lambda_;
    m_vec_ = std::min(m_vec_, D);
    if (m_vec_ <  1) m_vec_ = 1;

    // nsteps_: temporal target distance between stored vectors.
    // 2014 paper: nsteps = m_vec.  2017 paper: nsteps = n (problem dimension).
    // Using D gives much better temporal diversity, especially for small D.
    nsteps_ = D;

    // c_c_: evolution path learning rate.
    // 2014 paper: 1/m_vec.  2017 paper (Algorithm 7): sqrt(0.5/n).
    // sqrt(0.5/n) gives faster initial adaptation for any D.
    c_c_ = std::sqrt(0.5 / static_cast<double>(D));

    c1_  = 1.0 / (10.0 * std::log(static_cast<double>(D) + 1.0));
    a_   = std::sqrt(1.0 - c1_);

    // d_sigma_: PSR damping. Paper: 1.0 (both editions).
    if (!d_sigma_from_cfg_) {
        d_sigma_ = 1.0;
    }

    setPopulation(lambda_);

    // ---- mean initialization ----
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    m_.assign(D, 0.0);
    const bool have_bounds = (static_cast<int>(L.size()) == D &&
                              static_cast<int>(U.size()) == D);
    if (have_bounds) {
        for (int j = 0; j < D; ++j) m_[j] = 0.5 * (L[j] + U[j]);
    } else {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto X0 = initSampler.samplePopulation(*prob_, rng_, 1);
        if (!X0.empty() && static_cast<int>(X0[0].size()) == D)
            m_ = X0[0];
    }

    // ---- initial sigma ----
    if (have_bounds) {
        double avg_range = 0.0;
        for (int j = 0; j < D; ++j) avg_range += U[j] - L[j];
        avg_range /= static_cast<double>(D);
        if (avg_range <= 0.0) avg_range = 1.0;
        sigma_ = sigma0_ * avg_range;
    } else {
        sigma_ = sigma0_;
    }
    sigma_init_ = sigma_;

    // ---- direction-vector ring buffer ----
    P_.assign(m_vec_, Vec(D, 0.0));
    V_.assign(m_vec_, Vec(D, 0.0));
    b_.assign(m_vec_, 0.0);
    d_.assign(m_vec_, 0.0);
    // j_[i] will be valid only for i < n_stored_.
    // Pre-fill j_ sequentially so az/ainvz can rely on j_[0..n_stored_-1].
    j_.resize(m_vec_);
    l_.assign(m_vec_, -1);  // -1 = slot not yet written
    for (int k = 0; k < m_vec_; ++k) j_[k] = k;

    // ---- evolution path ----
    pc_.assign(D, 0.0);

    // ---- previous generation FX ----
    FX_prev_.clear();
    prev_sampled_ = 0;

    // ---- evaluate initial mean ----
    best_x_ = m_;
    best_f_ = safeEval(best_x_);   // uses safeEval so best is set correctly

    FX_.assign(1, best_f_);
    updateStop(FX_);
    printBest();
}

// =========================================================================
//  safeEval
//  Evaluates the problem and IMMEDIATELY updates best_f_ / best_x_.
//  Always returns a finite double (NaN/inf → +inf).
//  BUG FIX: sanitize before best comparison, not after.
// =========================================================================
double LMCMAES::safeEval(const Vec& x)
{
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

// =========================================================================
//  Az — Cholesky factor × vector  (Algorithm 3, Loshchilov 2014/2017)
//  x = A * z, A reconstructed from n_stored_ direction vectors.
//  Iterates oldest → newest (j_[0] to j_[n_stored_-1]).
//
//  CRITICAL: the dot product at each step MUST use the ORIGINAL z
//  (the function parameter), NOT the running x.
//
//  Paper, Algorithm 3, line 4: "k ← b^{j_it}  V^{(j_it,:)} · z"
//                                                              ^^^
//  z here is the function argument, unchanged throughout the loop.
//
//  Mathematical derivation (Section 4.1):
//    A_1·z = a·z + b^0·(v^0·z)·p^0          ← dot with original z
//    A_2·z = a·(A_1·z) + b^1·(v^1·z)·p^1   ← dot with original z again
//
//  Using the running x instead introduces cross-product terms
//    b^0·(v^0·z)·(v^1·p^0)·b^1·p^1  ≠ 0
//  which corrupt the covariance reconstruction entirely.
//  With n_stored_=1 both formulations coincide (x=z at t=0).
//  With n_stored_≥2 the bug accumulates iteration by iteration.
// =========================================================================
LMCMAES::Vec LMCMAES::az(const Vec& z) const
{
    Vec x = z;
    const int D = static_cast<int>(z.size());
    for (int t = 0; t < n_stored_; ++t) {
        const int  k    = j_[t];
        double     dot  = 0.0;
        for (int j = 0; j < D; ++j) dot += V_[k][j] * z[j]; // original z, not x
        const double coeff = b_[k] * dot;
        for (int j = 0; j < D; ++j) x[j] = a_ * x[j] + coeff * P_[k][j];
    }
    return x;
}

// =========================================================================
//  Ainvz — inverse Cholesky × vector  (Algorithm 4, Loshchilov 2014)
//  x = A^{-1} * z.
//  Iterates in the SAME order as Az (oldest → newest).
// =========================================================================
LMCMAES::Vec LMCMAES::ainvz(const Vec& z) const
{
    Vec x = z;
    const int    D     = static_cast<int>(x.size());
    const double c_inv = 1.0 / a_;          // 1/sqrt(1-c1)
    for (int t = 0; t < n_stored_; ++t) {
        const int  k    = j_[t];
        double     dot  = 0.0;
        for (int j = 0; j < D; ++j) dot += V_[k][j] * x[j];
        const double coeff = d_[k] * dot;
        for (int j = 0; j < D; ++j) x[j] = c_inv * x[j] - coeff * V_[k][j];
    }
    return x;
}

// =========================================================================
//  updateSet — direction vector selection  (Algorithm 5, Loshchilov 2014)
//
//  j_[] stores slot indices ordered oldest (j_[0]) → newest (j_[n_stored_-1]).
//  Returns the slot index (jcur) to be overwritten this iteration.
//  The caller must then write P_[jcur], V_[jcur], l_[jcur], b_[jcur], d_[jcur].
// =========================================================================
int LMCMAES::updateSet(int t)
{
    // ---- Phase 1: fill slots sequentially ----
    if (t < m_vec_) {
        j_[t] = t;          // slot t is the (t+1)-th oldest
        ++n_stored_;        // track number of filled slots
        return t;           // jcur = t
    }

    // ---- Phase 2: buffer full — find which slot to evict ----
    //
    // Find the pair of consecutive j_[] entries with the smallest temporal
    // gap.  imin is the 0-based position of the MORE RECENT element of that
    // pair (the one to be evicted).
    //
    // Per the paper (1-indexed eq.):
    //   imin = 1 + argmin_{i=1..m-1}  (l[j_{i+1}] - l[j_i])
    // If the minimum gap >= Nsteps → evict the oldest instead (imin = 0).

    int imin    = 1;
    int min_gap = l_[j_[1]] - l_[j_[0]];

    for (int i = 1; i < m_vec_ - 1; ++i) {
        int gap = l_[j_[i+1]] - l_[j_[i]];
        if (gap < min_gap) {
            min_gap = gap;
            imin    = i + 1;   // more-recent element of the closest pair
        }
    }

    // All gaps >= Nsteps: enforce FIFO (remove oldest)
    if (min_gap >= nsteps_)
        imin = 0;

    // Rotate j_[] so that j_[imin] moves to the newest position j_[m_vec_-1].
    // Slots before imin and after imin keep their relative order, which
    // preserves the oldest→newest invariant after l_[jcur] is written.
    if (imin != m_vec_ - 1) {
        const int jtmp = j_[imin];
        for (int i = imin; i < m_vec_ - 1; ++i)
            j_[i] = j_[i + 1];
        j_[m_vec_ - 1] = jtmp;
    }

    return j_[m_vec_ - 1];   // jcur = the evicted (now newest) slot
}

// =========================================================================
//  updateSigmaPSR — Population Success Rule  (Section 3.3, Loshchilov 2014)
//
//  Mix current (size actual_sampled) and previous (size prev_sampled_)
//  populations.  Rank all 2λ individuals (rank 1 = worst, 2λ = best).
//  z_PSR = [Σ_i (r_curr[i] - r_prev[i])] / λ² − z*
//
//  Normalise by lambda_² (not actual_sampled²) so that partial sampling
//  does not artificially inflate z_PSR.
//
//  BUG FIX: previously mixed in +inf "ghost" individuals from unsampled
//  slots, making every PSR comparison look like the current generation
//  was catastrophically worse → sigma collapsed immediately.
// =========================================================================
void LMCMAES::updateSigmaPSR(int actual_sampled)
{
    if (prev_sampled_ == 0 || FX_prev_.empty()) {
        // No previous generation to compare against yet.
        return;
    }

    const int n_curr = actual_sampled;            // finite entries in FX_
    const int n_prev = prev_sampled_;             // finite entries in FX_prev_

    // Build the mixed set from actually-sampled individuals only.
    struct Entry { double f; int src; int idx; };
    std::vector<Entry> mixed;
    mixed.reserve(n_curr + n_prev);
    for (int i = 0; i < n_curr; ++i) mixed.push_back({FX_[i],      0, i});
    for (int i = 0; i < n_prev; ++i) mixed.push_back({FX_prev_[i], 1, i});

    // Sort DESCENDING so that rank 1 = worst (highest f), rank N = best.
    std::sort(mixed.begin(), mixed.end(),
              [](const Entry& a, const Entry& b){ return a.f > b.f; });

    // Assign integer ranks (1 = worst, n_curr+n_prev = best).
    std::vector<double> r_curr(n_curr, 0.0);
    std::vector<double> r_prev(n_prev, 0.0);
    const int N = static_cast<int>(mixed.size());
    for (int rank = 1; rank <= N; ++rank) {
        const Entry& e = mixed[rank - 1];
        if (e.src == 0) r_curr[e.idx] = static_cast<double>(rank);
        else            r_prev[e.idx] = static_cast<double>(rank);
    }

    // z_PSR = [Σ(r_curr - r_prev)] / λ² − z*
    // Sum is invariant to ordering within each generation (sum of all ranks
    // equals sum of r_curr + sum of r_prev regardless of individual pairing).
    // Use lambda_² (nominal full population size) for consistent normalisation.
    double sum_diff = 0.0;
    const int n_cmp = std::min(n_curr, n_prev);
    for (int i = 0; i < n_cmp; ++i) sum_diff += r_curr[i] - r_prev[i];

    const double zpsr = sum_diff
                      / (static_cast<double>(lambda_) * static_cast<double>(lambda_))
                      - zstar_;

    // Cumulate and update sigma.
    s_      = (1.0 - c_sigma_) * s_ + c_sigma_ * zpsr;
    sigma_ *= std::exp(s_ / d_sigma_);

    // Clamp: relative lower bound prevents total collapse;
    // upper bound prevents explosion.
    const double sigma_floor = 1e-8 * sigma_init_;
    const double sigma_ceil  = 1e8  * sigma_init_;
    if (sigma_ < sigma_floor) sigma_ = sigma_floor;
    if (sigma_ > sigma_ceil)  sigma_ = sigma_ceil;
}

// =========================================================================
//  ensureBounds — clip (or repair) x to the problem's box bounds
// =========================================================================
void LMCMAES::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = prob_->dimension();
    const int m = std::min(D, static_cast<int>(x.size()));
    const bool have = (static_cast<int>(L.size()) == D &&
                       static_cast<int>(U.size()) == D);

    if (!have) {
        for (auto& v : x) if (!std::isfinite(v)) v = 0.0;
        return;
    }
    for (int j = 0; j < m; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// =========================================================================
//  one_iteration
// =========================================================================
void LMCMAES::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    std::normal_distribution<double> N01(0.0, 1.0);

    // =========================================================
    // 1. Sample λ offspring: x_k = m + σ · Az(z_k)
    //    (Algorithm 6, line 5-7)
    //
    //    BUG FIX: Track actual_sampled separately from lambda_.
    //    When the budget expires mid-loop the remaining X_ slots
    //    are left as zero-vectors and FX_ as +inf.  The old code
    //    included these ghost individuals in both the mean update
    //    and the PSR comparison, corrupting both.
    // =========================================================
    X_.assign(lambda_, Vec(D, 0.0));
    FX_.assign(lambda_, std::numeric_limits<double>::infinity());
    int actual_sampled = 0;

    for (int k = 0; k < lambda_ && prob_->calls() < max_evals_; ++k) {
        Vec z(D);
        for (int j = 0; j < D; ++j) z[j] = N01(rng_);

        Vec x(D);
        Vec Az = az(z);
        for (int j = 0; j < D; ++j) x[j] = m_[j] + sigma_ * Az[j];
        ensureBounds(x);

        X_[k]  = x;
        // BUG FIX: safeEval sanitises immediately AND updates best in one
        // place.  Old code sanitised AFTER the comparison loop, so NaN
        // results were silently skipped.
        FX_[k] = safeEval(x);
        ++actual_sampled;
    }

    if (actual_sampled == 0) return;   // nothing was sampled at all

    // =========================================================
    // 2. Sort X_ and FX_ together (best-first)
    //
    //    BUG FIX: old code sorted only FX_ (via an idx[] array),
    //    leaving X_ in the original sampling order.  The mean update
    //    compensated through idx[], but if actual_sampled < lambda_
    //    the un-touched zero-vector slots polluted the result.
    //    Sort both arrays together to keep them in sync, then use
    //    plain X_[i] / FX_[i] everywhere.
    // =========================================================
    {
        // Sort a range of [0, actual_sampled) by FX_ ascending.
        std::vector<int> order(actual_sampled);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });

        std::vector<Vec>    Xsorted(actual_sampled);
        std::vector<double> Fsorted(actual_sampled);
        for (int i = 0; i < actual_sampled; ++i) {
            Xsorted[i] = std::move(X_[order[i]]);
            Fsorted[i] = FX_[order[i]];
        }
        X_.swap(Xsorted);
        FX_.swap(Fsorted);
        // X_ and FX_ now both have size actual_sampled, sorted best-first.
    }

    // =========================================================
    // 3. Update mean  (Algorithm 6, line 8)
    //
    //    BUG FIX: use min(mu_, actual_sampled) individuals, not mu_,
    //    to avoid pulling the mean toward zero-vector unsampled slots.
    //    Renormalise weights so they still sum to 1.
    // =========================================================
    const int mu_use = std::min(mu_, actual_sampled);

    // Compute the normalisation constant for the used weights.
    double w_sum = 0.0;
    for (int i = 0; i < mu_use; ++i) w_sum += w_[i];
    // (w_sum == 1 when mu_use == mu_, otherwise < 1)

    Vec m_old = m_;
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < mu_use; ++i) s += w_[i] * X_[i][j];
        m_[j] = s / w_sum;   // re-normalise
    }

    // =========================================================
    // 4. Update evolution path p_c  (Algorithm 6, line 9)
    //    Uses σ^t (not yet updated) and m^{t+1} (just computed).
    // =========================================================
    const double cc_fac = std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);
    for (int j = 0; j < D; ++j)
        pc_[j] = (1.0 - c_c_) * pc_[j]
               + cc_fac * (m_[j] - m_old[j]) / sigma_;

    // =========================================================
    // 5-6. Update direction-vector ring buffer  (lines 10-14)
    //
    // OSWIN KRAUSE FIX (July 2014):
    //   The paper (Algorithm 6, line 10) says: v ← Ainvz(p_c).
    //   That is WRONG.  Az and Ainvz use V[t] in both their dot
    //   products, so for the two functions to be exact inverses
    //   of each other, we need V[t] == P[t].  Storing
    //   V[t] = Ainvz(p_c) ≠ p_c breaks the identity
    //   Ainvz(Az(z)) = z, causing the sampled distribution to
    //   diverge from what the b/d coefficients assume.
    //
    //   Fix: set V[jcur] = P[jcur] = p_c  and  use ||p_c||² for
    //   the b/d formulas (not ||Ainvz(p_c)||²).
    //
    //   Proof of consistency (n=1): with V=P=p_c,
    //     Ainvz(Az(z)) = c·(a·z + b·(p_c·z)·p_c)
    //                    - d·(p_c·(a·z+b·(p_c·z)·p_c))·p_c
    //                  = z + [c·b - d·(a+b·‖p_c‖²)]·(p_c·z)·p_c
    //   and substituting the b/d formulas gives c·b = d·(a+b·‖p_c‖²),
    //   so the bracket is 0 and we recover z. ✓
    //   With V ≠ P this cancellation fails, corrupting sampling.
    // =========================================================
    const int jcur = updateSet(iter_);
    P_[jcur] = pc_;
    V_[jcur] = pc_;          // <-- fix: same as P, NOT ainvz(pc)
    l_[jcur] = iter_;

    // Compute b[jcur] and d[jcur] using ||p_c||² (= ||V[jcur]||²).
    {
        double norm2_v = 0.0;
        for (int j = 0; j < D; ++j) norm2_v += pc_[j] * pc_[j];
        if (norm2_v > 1e-30) {
            const double ratio = (c1_ / (1.0 - c1_)) * norm2_v;
            const double sq    = std::sqrt(1.0 + ratio);
            b_[jcur] = (a_       / norm2_v) * (sq - 1.0);
            d_[jcur] = (1.0 / (a_ * norm2_v)) * (1.0 - 1.0 / sq);
        } else {
            b_[jcur] = 0.0;
            d_[jcur] = 0.0;
        }
    }

    // =========================================================
    // 7. PSR step-size adaptation  (lines 15-18)
    //
    //    BUG FIX: pass actual_sampled so PSR uses only finite
    //    individuals.  Old code mixed +inf ghost values into the
    //    rank computation, making z_PSR very negative every time
    //    the budget expired mid-loop → sigma collapsed to ~0.
    // =========================================================
    updateSigmaPSR(actual_sampled);

    // Save current sorted fitnesses for next iteration's PSR.
    // Store only actual_sampled entries (all finite).
    FX_prev_     = FX_;          // FX_ already has size actual_sampled
    prev_sampled_ = actual_sampled;

    // =========================================================
    // 8. Optional in-run local search
    // =========================================================
    if (!local_method_.empty() && local_rate_ > 0.0 &&
        prob_->calls() < max_evals_) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto res = localSearch(local_method_, best_x_);
            const Vec& xloc = res.first;
            const double floc = res.second;
            if (!xloc.empty() &&
                static_cast<int>(xloc.size()) == D &&
                std::isfinite(floc) && floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
            }
        }
    }

    updateStop(FX_);
    printBest();
    ++iter_;
}

// =========================================================================
//  end
// =========================================================================
void LMCMAES::end()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        const Vec& xloc = res.first;
        const double floc = res.second;
        if (!xloc.empty() &&
            static_cast<int>(xloc.size()) == D &&
            std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }
        printBest();
    }

    if (!FX_.empty()) updateStop(FX_);
}

} // namespace optimsolution
