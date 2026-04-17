#include "sfcde.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

// ─────────────────────────────────────────────────────────────────────────────
//  configure
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::configure(const MethodConfig& mc)
{
    int basePop = population();
    if (basePop < 4) basePop = 50;

    const int pop_override = mc.getInt("population", 0);
    if (pop_override >= 4) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }
    setPopulation(pop_init_);

    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    mu_f_init_ = mc.getDbl("mu_f_init", mu_f_init_);
    if (mu_f_init_ <= 0.0) mu_f_init_ = 0.5;
    if (mu_f_init_ > 1.0) mu_f_init_ = 1.0;

    mu_cr_init_ = mc.getDbl("mu_cr_init", mu_cr_init_);
    if (mu_cr_init_ < 0.0) mu_cr_init_ = 0.0;
    if (mu_cr_init_ > 1.0) mu_cr_init_ = 1.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  meanLehmer  (used for F — standard SHADE Lehmer mean)
// ─────────────────────────────────────────────────────────────────────────────
double SFCDE::meanLehmer(const std::vector<double>& values) const
{
    double numer = 0.0;
    double denom = 0.0;
    for (double v : values) {
        numer += v * v;
        denom += v;
    }
    if (denom <= 0.0) return 0.0;
    return numer / denom;
}

// ─────────────────────────────────────────────────────────────────────────────
//  meanArithmetic  (used for CR — standard SHADE arithmetic mean)
// ─────────────────────────────────────────────────────────────────────────────
double SFCDE::meanArithmetic(const std::vector<double>& values) const
{
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  sampleF  —  Cauchy sample, clipped to (0, 1]
//  The inner do-while is safe: Cauchy is centred on mu > 0 with scale 0.1,
//  so P(F > 0) >> 0.5 and the loop terminates rapidly.
// ─────────────────────────────────────────────────────────────────────────────
double SFCDE::sampleF(double mu)
{
    std::cauchy_distribution<double> dist(mu, cauchy_scale_F_);
    double F;
    do {
        F = dist(rng_);
    } while (F <= 0.0);
    if (F > 1.0) F = 1.0;
    return F;
}

// ─────────────────────────────────────────────────────────────────────────────
//  sampleCR  —  Normal sample, clipped to [0, 1]
// ─────────────────────────────────────────────────────────────────────────────
double SFCDE::sampleCR(double mu)
{
    std::normal_distribution<double> dist(mu, normal_std_CR_);
    double cr = dist(rng_);
    if (cr < 0.0) cr = 0.0;
    if (cr > 1.0) cr = 1.0;
    return cr;
}

// ─────────────────────────────────────────────────────────────────────────────
//  repairRandom  —  replace any out-of-bounds or non-finite coordinate
//  with a uniform draw within [lb, ub].
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::repairRandom(Vec& x)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j]) || x[j] < L[j] || x[j] > U[j]) {
            std::uniform_real_distribution<double> Uj(L[j], U[j]);
            x[j] = Uj(rng_);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  bestIndex
// ─────────────────────────────────────────────────────────────────────────────
int SFCDE::bestIndex() const
{
    if (FX_.empty()) return -1;
    return static_cast<int>(std::min_element(FX_.begin(), FX_.end()) - FX_.begin());
}

// ─────────────────────────────────────────────────────────────────────────────
//  sampleThreeDistinct  —  FIX #4
//
//  Draws three distinct indices a, b, c from [0, N) that are all ≠ exclude.
//  Uses a partial Fisher-Yates shuffle over a compact candidate pool, so
//  there is NO rejection-sampling loop and the cost is always O(N).
//
//  Precondition: N >= 4  (guarantees at least 3 candidates after exclusion).
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::sampleThreeDistinct(int N, int exclude,
                                int& a, int& b, int& c)
{
    // Build pool of all indices except exclude.
    // We avoid a heap allocation by reusing a thread-local or local vector;
    // for simplicity we use a local vector here (typically N ≤ 200).
    std::vector<int> pool;
    pool.reserve(static_cast<size_t>(N - 1));
    for (int j = 0; j < N; ++j) {
        if (j != exclude) pool.push_back(j);
    }

    // Partial Fisher-Yates: shuffle only the first 3 positions.
    const int M = static_cast<int>(pool.size()); // M = N-1 >= 3
    for (int s = 0; s < 3; ++s) {
        std::uniform_int_distribution<int> pick(s, M - 1);
        int t = pick(rng_);
        std::swap(pool[s], pool[t]);
    }
    a = pool[0];
    b = pool[1];
    c = pool[2];
}

// ─────────────────────────────────────────────────────────────────────────────
//  init
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (pop_init_ < 4) pop_init_ = 50;
    setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    fail_F_.clear();
    fail_CR_.clear();

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < X_.size(); ++i) {
        repairRandom(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    MF_.assign(H_,  mu_f_init_);
    MCR_.assign(H_, mu_cr_init_);

    // FIX #3: reset the sequential write pointer.
    k_ = 0;

    updateStop(FX_);
    printBest();
}

// ─────────────────────────────────────────────────────────────────────────────
//  one_iteration
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();
    if (N < 4 || D <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui_mem(0, H_ - 1);
    std::uniform_int_distribution<int>     Ui_dim(0, D - 1);

    // ── Failure statistics from the PREVIOUS generation ───────────────────
    const bool   has_fail_F  = !fail_F_.empty();
    const bool   has_fail_CR = !fail_CR_.empty();
    const double mu_fail_F   = has_fail_F  ? meanArithmetic(fail_F_)  : 0.0;
    const double mu_fail_CR  = has_fail_CR ? meanArithmetic(fail_CR_) : 0.0;

    // Discard previous generation's failure lists — we'll rebuild them below.
    fail_F_.clear();
    fail_CR_.clear();

    // ── Accumulate successes for memory update at end of generation ────────
    std::vector<double> success_F;
    std::vector<double> success_CR;
    success_F.reserve(N);
    success_CR.reserve(N);

    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    const int bidx = bestIndex();
    if (bidx < 0) return;
    const Vec xbest = X_[bidx];   // fixed for the whole generation

    // ── Main loop ─────────────────────────────────────────────────────────
    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        // ── FIX #2: Per-individual memory index selection ─────────────────
        //    In SHADE and all its published variants (L-SHADE, CFDE, ISHACDE)
        //    each individual draws its own random memory slot independently.
        //    Using one shared slot per generation reduces parameter diversity.
        const int F_idx  = Ui_mem(rng_);
        const int CR_idx = Ui_mem(rng_);

        // ── FIX #4: Distinct index selection via partial Fisher-Yates ──────
        //    Replaces the rejection-sampling lambda that could spin for many
        //    iterations when N is small (e.g., N = 4).
        int idx, r1, r2;
        sampleThreeDistinct(N, i, idx, r1, r2);
        // idx = competitor for the competitive mechanism
        // r1, r2 = differential pair

        // ── Sample F and CR from the current individual's memory slots ─────
        double F  = sampleF(MF_[F_idx]);
        double CR = sampleCR(MCR_[CR_idx]);

        // ── FIX #1: SFA for F — bounded rejection loop ────────────────────
        //
        //  Goal: avoid parameter values that are geometrically closer to the
        //  failure-mean than to the success-mean (mu_fail_F vs. MF_[F_idx]).
        //  A probabilistic acceptance lets the algorithm occasionally use an
        //  "edge-zone" F, preserving diversity.
        //
        //  FIX: kMaxSfaTrials caps the loop so it can never hang when
        //  mu_fail_F ≈ MF_[F_idx] (convergence phase or cold-start).
        if (has_fail_F) {
            for (int attempt = 0;
                 attempt < kMaxSfaTrials &&
                 std::abs(F - mu_fail_F) < std::abs(F - MF_[F_idx]);
                 ++attempt)
            {
                const double Dfail = std::abs(F - mu_fail_F);
                const double Dsucc = std::abs(F - MF_[F_idx]);

                // Acceptance probability: higher when F is only marginally
                // inside the failure zone (Dsucc ≈ Dfail), near zero when
                // F is deep inside it (Dsucc >> Dfail).
                const double prob = (Dfail <= 1e-15)
                    ? 0.0
                    : std::exp(-std::sqrt(Dsucc / Dfail));

                if (U01(rng_) < prob) break;   // accept with small probability
                F = sampleF(MF_[F_idx]);       // otherwise resample
            }
            // After kMaxSfaTrials the current F is used as-is; in the rare
            // degenerate case this is a reasonable fallback.
        }

        // ── FIX #1: SFA for CR — same bounded-loop treatment ──────────────
        if (has_fail_CR) {
            for (int attempt = 0;
                 attempt < kMaxSfaTrials &&
                 std::abs(CR - mu_fail_CR) < std::abs(CR - MCR_[CR_idx]);
                 ++attempt)
            {
                const double Dfail = std::abs(CR - mu_fail_CR);
                const double Dsucc = std::abs(CR - MCR_[CR_idx]);

                const double prob = (Dfail <= 1e-15)
                    ? 0.0
                    : std::exp(-std::sqrt(Dsucc / Dfail));

                if (U01(rng_) < prob) break;
                CR = sampleCR(MCR_[CR_idx]);
            }
        }

        // ── Competitive mechanism: winner becomes the base vector ──────────
        //    DE/winner-to-best/1 as in the original CDE paper.
        const Vec& winner = (FX_[idx] < FX_[i]) ? X_[idx] : X_[i];

        // ── Mutation ──────────────────────────────────────────────────────
        Vec donor(D, 0.0);
        for (int j = 0; j < D; ++j) {
            donor[j] = winner[j]
                     + F * (xbest[j]    - winner[j])
                     + F * (X_[r1][j]   - X_[r2][j]);
        }

        // ── Binomial crossover ────────────────────────────────────────────
        Vec trial = donor;                      // start from donor
        const int jrand = Ui_dim(rng_);
        for (int j = 0; j < D; ++j) {
            if (!(U01(rng_) < CR || j == jrand)) {
                trial[j] = X_[i][j];           // revert to target on miss
            }
        }

        repairRandom(trial);

        // ── Greedy selection ──────────────────────────────────────────────
        const double f_trial = eval(trial);
        if (f_trial < FX_[i]) {
            newPop[i] = trial;
            newFit[i] = f_trial;
            success_F.push_back(F);
            success_CR.push_back(CR);

            if (f_trial < best_f_) {
                best_f_ = f_trial;
                best_x_ = trial;
            }
        } else {
            fail_F_.push_back(F);
            fail_CR_.push_back(CR);
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // ── FIX #3: Sequential memory update via cycling pointer k_ ───────────
    //
    //  Original code: used the same random F_idx / CR_idx for both reading
    //  (sampling) and writing (update).  This left some slots perpetually
    //  stale while others were overwritten repeatedly.
    //
    //  Fix: maintain a separate write pointer k_ that cycles 0 … H_-1, so
    //  every slot is refreshed at the same average rate (every H_ generations).
    //  The read index (F_idx / CR_idx per individual above) remains random,
    //  exactly as in published SHADE variants.
    if (!success_F.empty()) {
        MF_[k_]  = (1.0 - c_mem_) * MF_[k_]  + c_mem_ * meanLehmer(success_F);
    }
    if (!success_CR.empty()) {
        MCR_[k_] = (1.0 - c_mem_) * MCR_[k_] + c_mem_ * meanArithmetic(success_CR);
    }
    k_ = (k_ + 1) % H_;   // advance pointer regardless of success/failure

    updateStop(FX_);
    printBest();
}

// ─────────────────────────────────────────────────────────────────────────────
//  end
// ─────────────────────────────────────────────────────────────────────────────
void SFCDE::end()
{
    if (!prob_) return;
    if (!end_local_refine_) return;
    if (end_local_method_.empty()) return;
    if (best_x_.empty()) return;

    auto res = localSearch(end_local_method_, best_x_);
    if (!res.first.empty() &&
        std::isfinite(res.second) &&
        res.second < best_f_) {

        best_x_ = std::move(res.first);
        best_f_ = res.second;

        if (!X_.empty() && !FX_.empty()) {
            const int worst = static_cast<int>(
                std::max_element(FX_.begin(), FX_.end()) - FX_.begin());
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
