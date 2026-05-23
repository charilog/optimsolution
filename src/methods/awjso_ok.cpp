#include "awjso.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {
inline double clampRange(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
} // anonymous namespace

double AWJSO::clamp01(double v)
{
    return clampRange(v, 0.0, 1.0);
}

double AWJSO::domainDiagonal() const
{
    if (!prob_) return 1.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    if (L.size() != U.size() || L.empty()) return 1.0;

    double s = 0.0;
    for (size_t j = 0; j < L.size(); ++j) {
        const double d = U[j] - L[j];
        s += d * d;
    }
    const double diag = std::sqrt(std::max(s, 0.0));
    return (diag > 0.0) ? diag : 1.0;
}

double AWJSO::normalizedDistance(const Vec& a, const Vec& b) const
{
    if (a.size() != b.size() || a.empty()) return 0.0;
    double s = 0.0;
    for (size_t j = 0; j < a.size(); ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return std::sqrt(std::max(s, 0.0)) / domainDiagonal();
}

double AWJSO::directionAlignment(const Vec& from, const Vec& to1, const Vec& to2) const
{
    if (from.size() != to1.size() || from.size() != to2.size() || from.empty()) return 0.5;

    double dot = 0.0;
    double n1  = 0.0;
    double n2  = 0.0;
    for (size_t j = 0; j < from.size(); ++j) {
        const double a = to1[j] - from[j];
        const double b = to2[j] - from[j];
        dot += a * b;
        n1  += a * a;
        n2  += b * b;
    }
    if (n1 <= 1e-24 || n2 <= 1e-24) return 0.5;
    const double c = dot / (std::sqrt(n1) * std::sqrt(n2));
    return 0.5 * (clampRange(c, -1.0, 1.0) + 1.0);
}

// ---------------------------------------------------------------------------
// Mechanism A — adaptive Fw multiplier
//
// FIX (Bug 3): The raw score spans [-0.22, +1.00] (span = 1.22).
// Previously "clamp01(0.50 + score)" offset the baseline to 0.28, making
// fw_min_mul_ unreachable. Now we normalise the raw score linearly onto
// [0, 1] before mapping onto [fw_min_mul_, fw_max_mul_], so the full
// multiplier range is exercised.
// ---------------------------------------------------------------------------
double AWJSO::adaptiveWeightMultiplier(const Vec& xi,
                                       const Vec& xp,
                                       int rank_pos,
                                       int N,
                                       double progress) const
{
    const double prog             = clamp01(progress);
    const double rank_ratio       = (N > 1) ? static_cast<double>(rank_pos) /
                                               static_cast<double>(N - 1) : 0.0;
    const double dist_best        = clamp01(normalizedDistance(xi, best_x_) / 0.35);
    const double dist_pbest       = clamp01(normalizedDistance(xi, xp) / 0.25);
    const double align            = directionAlignment(xi, xp, best_x_);
    const double success_pressure = clamp01(success_ema_ /
                                            std::max(fw_target_success_, 1e-9));
    const double stagnation       = clamp01(static_cast<double>(no_best_improve_iters_) / 12.0);

    // Raw score ∈ [-0.22, +1.00]
    //   positive drivers: progress, stagnation, rank position, distances, alignment
    //   negative driver:  high recent-success rate (less pull needed)
    double raw = 0.0;
    raw += 0.24 * prog;
    raw += 0.23 * stagnation;
    raw += 0.16 * rank_ratio;
    raw += 0.14 * dist_best;
    raw += 0.11 * dist_pbest;
    raw += 0.12 * align;
    raw -= 0.22 * success_pressure;
    // Span of raw: min = -0.22 (all zeros except max success_pressure)
    //              max = +1.00 (all positive terms maxed, no penalty)

    // Normalise to [0, 1] so that fw_min_mul_ and fw_max_mul_ are reachable.
    static constexpr double RAW_MIN  = -0.22;
    static constexpr double RAW_SPAN =  1.22;   // 1.00 - (-0.22)
    const double score = clamp01((raw - RAW_MIN) / RAW_SPAN);

    const double mul = fw_min_mul_ + (fw_max_mul_ - fw_min_mul_) * score;
    return clampRange(mul, fw_min_mul_, fw_max_mul_);
}

// ---------------------------------------------------------------------------
// Mechanism B — predictive prescreen
//
// FIX (Bug 1): The original condition was:
//     return (step < step_floor_ && merit < threshold);
// The extra "step < step_floor_" gate prevented skipping large-step moves
// that nonetheless had genuinely low merit (wrong direction, no contraction,
// no novelty).  The merit formula already penalises small steps via its
// first term (0.34 * step/step_floor_), so the gate is redundant AND
// harmful.  Correct condition: merit alone decides.
// ---------------------------------------------------------------------------
bool AWJSO::shouldSkipEvaluation(const Vec& xi,
                                 const Vec& ui,
                                 const Vec& xp,
                                 const Vec& xr1,
                                 const Vec& xr2,
                                 int rank_pos,
                                 int N,
                                 double progress,
                                 int evals_done_this_iter,
                                 int max_skips_this_iter,
                                 int skips_done_this_iter,
                                 bool is_best_candidate)
{
    if (!predictive_prescreen_enable_)              return false;
    if (progress < predictive_prescreen_start_)     return false;
    if (max_skips_this_iter <= 0)                   return false;
    if (skips_done_this_iter >= max_skips_this_iter) return false;
    if (is_best_candidate)                          return false;

    const int min_evals = std::max(1, static_cast<int>(
        std::ceil(predictive_prescreen_min_eval_frac_ * N)));
    if (evals_done_this_iter < min_evals) return false;

    // Stochastic exploration: occasionally evaluate even low-merit candidates.
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    if (U01(rng_) < predictive_prescreen_explore_prob_) return false;

    // --- Merit signal ---
    const double step        = normalizedDistance(xi, ui);
    const double dist_x_best = normalizedDistance(xi, best_x_);
    const double dist_u_best = normalizedDistance(ui, best_x_);

    // Contraction toward best: positive when trial is closer to best than parent.
    const double contraction = (dist_x_best > 1e-15)
        ? clamp01((dist_x_best - dist_u_best) / dist_x_best)
        : 0.0;

    // Novelty: how far ui is from the nearest reference point (xp, xr1, xr2).
    const double novelty_ref = std::min(
        std::min(normalizedDistance(ui, xp), normalizedDistance(ui, xr1)),
        normalizedDistance(ui, xr2));
    const double novelty = clamp01(novelty_ref / 0.08);

    const double align_best = directionAlignment(xi, ui, best_x_);
    const double rank_ratio = (N > 1) ? static_cast<double>(rank_pos) /
                                         static_cast<double>(N - 1) : 0.0;
    const double progress_gain = clamp01(
        (progress - predictive_prescreen_start_) /
        std::max(1.0 - predictive_prescreen_start_, 1e-9));

    // merit ∈ [0, 1]: high = worth evaluating, low = likely wasted evaluation.
    //   step contributes positively (larger step = more worth evaluating),
    //   but is already penalised implicitly when step < step_floor_ because
    //   clamp01(step / step_floor_) < 1.  No additional hard gate needed.
    double merit = 0.0;
    merit += 0.34 * clamp01(step / std::max(predictive_prescreen_step_floor_, 1e-9));
    merit += 0.28 * contraction;
    merit += 0.18 * novelty;
    merit += 0.10 * align_best;
    merit += 0.10 * (1.0 - rank_ratio);   // worse-ranked individuals benefit more

    // Adaptive threshold: rises with progress and with recent skip rate.
    double threshold = predictive_prescreen_threshold_
                     + 0.08 * progress_gain
                     + 0.06 * prescreen_skip_ema_;
    threshold = clampRange(threshold, 0.05, 0.90);

    // FIX: merit alone decides — no extra step gate.
    return (merit < threshold);
}

// ---------------------------------------------------------------------------
// configure
// ---------------------------------------------------------------------------
void AWJSO::configure(const MethodConfig& mc)
{
    int basePop = population();
    if (basePop < 4) basePop = 4;

    const int pop_override = mc.getInt("population", 0);
    if (pop_override > 3) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }
    setPopulation(pop_init_);

    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    pmin_ = mc.getDbl("p_min", pmin_);
    pmax_ = mc.getDbl("p_max", pmax_);
    if (pmin_ <= 0.0) pmin_ = 0.05;
    if (pmax_ <= pmin_) pmax_ = std::max(2.0 * pmin_, 0.25);

    arc_rate_ = mc.getDbl("arc_rate", arc_rate_);
    if (arc_rate_ < 0.0) arc_rate_ = 0.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    double lr = mc.getDbl("local_rate", local_rate_);
    lr = clampRange(lr, 0.0, 1.0);
    local_rate_ = lr;

    // --- Mechanism A ---
    adaptive_weight_enable_ = (mc.getInt("adaptive_weight_enable",
                                          adaptive_weight_enable_ ? 1 : 0) != 0);
    fw_min_mul_        = mc.getDbl("fw_min_mul",        fw_min_mul_);
    fw_max_mul_        = mc.getDbl("fw_max_mul",        fw_max_mul_);
    fw_abs_max_        = mc.getDbl("fw_abs_max",        fw_abs_max_);
    fw_target_success_ = mc.getDbl("fw_target_success", fw_target_success_);

    if (fw_min_mul_ < 0.05)             fw_min_mul_ = 0.05;
    if (fw_max_mul_ < fw_min_mul_)      fw_max_mul_ = fw_min_mul_;
    if (fw_abs_max_ < 0.10)             fw_abs_max_ = 0.10;
    if (fw_target_success_ <= 0.0)      fw_target_success_ = 0.18;

    // --- Mechanism B ---
    predictive_prescreen_enable_    = (mc.getInt("predictive_prescreen_enable",
                                                  predictive_prescreen_enable_ ? 1 : 0) != 0);
    predictive_prescreen_start_     = mc.getDbl("predictive_prescreen_start",
                                                 predictive_prescreen_start_);
    predictive_prescreen_threshold_ = mc.getDbl("predictive_prescreen_threshold",
                                                 predictive_prescreen_threshold_);
    predictive_prescreen_step_floor_   = mc.getDbl("predictive_prescreen_step_floor",
                                                    predictive_prescreen_step_floor_);
    predictive_prescreen_explore_prob_ = mc.getDbl("predictive_prescreen_explore_prob",
                                                    predictive_prescreen_explore_prob_);
    predictive_prescreen_min_eval_frac_ = mc.getDbl("predictive_prescreen_min_eval_frac",
                                                     predictive_prescreen_min_eval_frac_);

    predictive_prescreen_start_      = clamp01(predictive_prescreen_start_);
    predictive_prescreen_threshold_  = clampRange(predictive_prescreen_threshold_, 0.01, 0.95);
    predictive_prescreen_step_floor_ = clampRange(predictive_prescreen_step_floor_, 1e-6, 1.0);
    predictive_prescreen_explore_prob_   = clamp01(predictive_prescreen_explore_prob_);
    predictive_prescreen_min_eval_frac_  = clampRange(predictive_prescreen_min_eval_frac_, 0.05, 1.0);
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
void AWJSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (pop_init_ <= 0) {
        pop_init_ = population();
        if (pop_init_ < 4) pop_init_ = 50;
        setPopulation(pop_init_);
    }
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    archive_.clear();

    X_  = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < X_.size(); ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    MF_.assign(H_, 0.3);
    MCR_.assign(H_, 0.8);
    if (H_ > 0) {
        MF_[H_ - 1]  = 0.9;
        MCR_[H_ - 1] = 0.9;
    }
    mem_idx_ = 0;

    // Mechanism A state
    success_ema_           = 0.0;
    no_best_improve_iters_ = 0;
    iteration_counter_     = 0;

    // Mechanism B state
    prescreen_skip_ema_ = 0.0;

    updateStop(FX_);
    printBest();
}

// ---------------------------------------------------------------------------
// Bound helpers
// ---------------------------------------------------------------------------
void AWJSO::ensureInBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

bool AWJSO::isInBounds(const Vec& x) const
{
    if (!prob_) return false;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    if (x.size() != L.size() || x.size() != U.size()) return false;
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) return false;
        if (x[j] < L[j] || x[j] > U[j]) return false;
    }
    return true;
}

void AWJSO::trimArchive(int max_size)
{
    if (max_size < 0) max_size = 0;
    while ((int)archive_.size() > max_size) {
        std::uniform_int_distribution<int> Ui_arc(0, (int)archive_.size() - 1);
        archive_.erase(archive_.begin() + Ui_arc(rng_));
    }
}

// ---------------------------------------------------------------------------
// one_iteration  —  unified entry point
//
// Mode matrix (controlled by two independent flags):
//
//   adaptive_weight_enable_  | predictive_prescreen_enable_  | Behaviour
//   -------------------------+-------------------------------+-------------------
//   false                    | false                         | Classic jSO
//   true                     | false                         | Adaptive Fw only
//   false                    | true                          | Prescreen only
//   true                     | true                          | Both mechanisms
//
// FIX (Bug 2): MF/MCR memory slots are now REPLACED (not averaged) so that
// the memory adapts at the rate intended by SHADE/L-SHADE/jSO.
// ---------------------------------------------------------------------------
void AWJSO::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    int N = static_cast<int>(X_.size());
    if (N < 4) return;

    const double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));

    // -----------------------------------------------------------------------
    // Linear Population Size Reduction (LPSR)
    // -----------------------------------------------------------------------
    {
        const double nfes = static_cast<double>(prob_->calls());
        int targetN = static_cast<int>(
            std::round(pop_init_ - (pop_init_ - pop_min_) * (nfes / maxFES)));
        targetN = clampRange(targetN, pop_min_, pop_init_);

        if (targetN < N) {
            std::vector<int> idx(N);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return FX_[a] < FX_[b]; });

            std::vector<Vec>    newX;
            std::vector<double> newF;
            newX.reserve(targetN);
            newF.reserve(targetN);
            for (int k = 0; k < targetN; ++k) {
                newX.push_back(X_[idx[k]]);
                newF.push_back(FX_[idx[k]]);
            }
            X_.swap(newX);
            FX_.swap(newF);
            N = targetN;
        }
    }
    if (N < 4) return;

    trimArchive(static_cast<int>(std::round(arc_rate_ * N)));

    // -----------------------------------------------------------------------
    // Progress and p-best size
    // -----------------------------------------------------------------------
    const double progress = clamp01(static_cast<double>(prob_->calls()) / maxFES);

    double p = pmax_ - (pmax_ - pmin_) * progress;
    p = clampRange(p, pmin_, pmax_);

    int p_best_size = static_cast<int>(std::round(p * N));
    if (p_best_size < 2) p_best_size = 2;
    if (p_best_size > N) p_best_size = N;

    // -----------------------------------------------------------------------
    // Sort population by fitness → rank table
    // -----------------------------------------------------------------------
    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<int> rank_pos(N, 0);
    for (int pos = 0; pos < N; ++pos)
        rank_pos[idx_sorted[pos]] = pos;

    // -----------------------------------------------------------------------
    // Working copies of population/fitness
    // -----------------------------------------------------------------------
    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    std::vector<double> SF, SCR, dF;
    SF.reserve(N);
    SCR.reserve(N);
    dF.reserve(N);

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui_dim(0, D - 1);
    std::uniform_int_distribution<int>     Ui_mem(0, H_ - 1);

    // -----------------------------------------------------------------------
    // Lambda helpers for index sampling
    // -----------------------------------------------------------------------
    auto samplePopIndex = [&](int avoid1, int avoid2, int avoid3) -> int {
        std::uniform_int_distribution<int> Ui_pop(0, N - 1);
        int idx;
        do { idx = Ui_pop(rng_); }
        while (idx == avoid1 || idx == avoid2 || idx == avoid3);
        return idx;
    };

    // Sample r2 from population ∪ archive, avoiding three indices.
    auto sampleR2 = [&](int avoid1, int avoid2, int avoid3, Vec& xr2) -> bool {
        const int total = N + static_cast<int>(archive_.size());
        if (total <= 0) return false;

        std::uniform_int_distribution<int> Ui_union(0, total - 1);
        for (int tries = 0; tries < 128; ++tries) {
            const int pick = Ui_union(rng_);
            if (pick < N) {
                if (pick == avoid1 || pick == avoid2 || pick == avoid3) continue;
                xr2 = X_[pick];
                return true;
            }
            xr2 = archive_[pick - N];
            return true;
        }
        // Fallback to population only.
        const int r2 = samplePopIndex(avoid1, avoid2, avoid3);
        xr2 = X_[r2];
        return true;
    };

    // -----------------------------------------------------------------------
    // Prescreen budget for this iteration
    // -----------------------------------------------------------------------
    const int max_skips_this_iter = std::max(
        0,
        N - std::max(1, static_cast<int>(
            std::ceil(predictive_prescreen_min_eval_frac_ * N))));
    int skips_done_this_iter      = 0;
    int evals_done_this_iter      = 0;
    int strict_successes_this_iter = 0;    // used by Mechanism A EMA
    bool best_improved_this_iter   = false; // used by Mechanism A stagnation

    // -----------------------------------------------------------------------
    // Main per-individual loop
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        // --- F/CR from history memory ---
        const int    r_mem = Ui_mem(rng_);
        const double muF   = (H_ > 0 && r_mem == H_ - 1) ? 0.9 : MF_[r_mem];
        const double muCR  = (H_ > 0 && r_mem == H_ - 1) ? 0.9 : MCR_[r_mem];

        // --- Sample Fi from Cauchy(muF, scale) ---
        double Fi;
        {
            std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
            do { Fi = cauchy(rng_); } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
        }

        // --- Sample CRi from Normal(muCR, std) ---
        double CRi;
        if (muCR < 0.0) {
            CRi = 0.0;
        } else {
            std::normal_distribution<double> normal(muCR, normal_std_CR_);
            CRi = clampRange(normal(rng_), 0.0, 1.0);
        }

        // --- jSO early-stage CR/F adjustments (both modes) ---
        if (progress < 0.25) {
            if (CRi < 0.7) CRi = 0.7;
        } else if (progress < 0.5) {
            if (CRi < 0.6) CRi = 0.6;
        }
        if (progress < 0.6 && Fi > 0.7) Fi = 0.7;

        const Vec& xi = X_[i];

        // --- Build trial vector ui ---
        Vec ui(D);
        bool feasible = false;
        Vec xp_sel, xr1_sel, xr2_sel;

        for (int repeat = 0; repeat < 64 && !feasible; ++repeat) {
            std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
            const int  p_idx = idx_sorted[Ui_pbest(rng_)];
            const Vec& xp    = X_[p_idx];

            const int  r1   = samplePopIndex(i, p_idx, -1);
            const Vec& xr1  = X_[r1];

            Vec xr2;
            if (!sampleR2(i, p_idx, r1, xr2)) break;

            // --- Fw: classic jSO schedule OR adaptive multiplier ---
            double Fw;
            if (adaptive_weight_enable_) {
                // Mechanism A active: per-individual adaptive multiplier.
                const double mul = adaptiveWeightMultiplier(xi, xp,
                                                             rank_pos[i], N,
                                                             progress);
                Fw = clampRange(Fi * mul, 0.0, fw_abs_max_);
            } else {
                // Classic jSO three-phase Fw schedule.
                if      (progress < 0.2) Fw = 0.7 * Fi;
                else if (progress < 0.4) Fw = 0.8 * Fi;
                else                     Fw = 1.2 * Fi;
            }

            // --- current-to-pBest/1 crossover ---
            const int jrand = Ui_dim(rng_);
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CRi || j == jrand)
                    ui[j] = xi[j] + Fw * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j]);
                else
                    ui[j] = xi[j];
            }

            feasible = isInBounds(ui);
            if (feasible) {
                xp_sel  = xp;
                xr1_sel = xr1;
                xr2_sel = xr2;
            }
        }

        // Repair out-of-bounds trial vector by clamping.
        if (!feasible) {
            ensureInBounds(ui);
            // Rebuild reference vectors for the prescreen (needed even when
            // feasible=false so that shouldSkipEvaluation has valid inputs).
            std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
            const int p_idx = idx_sorted[Ui_pbest(rng_)];
            xp_sel  = X_[p_idx];
            const int r1 = samplePopIndex(i, p_idx, -1);
            xr1_sel = X_[r1];
            if (!sampleR2(i, p_idx, r1, xr2_sel)) xr2_sel = xi;
        }

        // --- Mechanism B: prescreen (no-op when disabled) ---
        if (shouldSkipEvaluation(xi, ui, xp_sel, xr1_sel, xr2_sel,
                                 rank_pos[i], N, progress,
                                 evals_done_this_iter, max_skips_this_iter,
                                 skips_done_this_iter,
                                 rank_pos[i] == 0)) {
            ++skips_done_this_iter;
            continue;
        }

        // --- Evaluate ---
        const double f_old = FX_[i];
        const double f_new = eval(ui);
        ++evals_done_this_iter;

        const bool accepted = (f_new <= f_old);
        if (accepted) {
            newPop[i] = ui;
            newFit[i] = f_new;

            // Optional in-run local search.
            if (local_rate_ > 0.0 && !local_method_.empty() &&
                prob_->calls() < max_evals_) {
                if (U01(rng_) < local_rate_) {
                    auto res = localSearch(local_method_, newPop[i]);
                    if (!res.first.empty() && std::isfinite(res.second) &&
                        res.second < newFit[i]) {
                        newPop[i] = std::move(res.first);
                        newFit[i] = res.second;
                    }
                }
            }

            const double accepted_f   = newFit[i];
            const bool strict_success = (accepted_f < f_old);
            if (strict_success) {
                ++strict_successes_this_iter;
                if (arc_rate_ > 0.0) archive_.push_back(xi);
                SF.push_back(Fi);
                SCR.push_back(CRi);
                dF.push_back(f_old - accepted_f);
            }

            if (accepted_f < best_f_) {
                best_f_                = accepted_f;
                best_x_                = newPop[i];
                best_improved_this_iter = true;
            }
        }
    } // end per-individual loop

    X_.swap(newPop);
    FX_.swap(newFit);
    trimArchive(static_cast<int>(std::round(arc_rate_ * static_cast<int>(X_.size()))));

    // -----------------------------------------------------------------------
    // MF/MCR memory update
    //
    // FIX (Bug 2): Direct replacement (not averaging).  Each writable slot
    // [0 .. H_-2] is fully overwritten with the Lehmer mean of F and the
    // weighted mean of CR from this iteration's successful trials.  Averaging
    // halved the adaptation speed; full replacement is correct per SHADE.
    // -----------------------------------------------------------------------
    if (!SF.empty()) {
        double sum_dF = 0.0;
        for (double v : dF) sum_dF += v;
        if (sum_dF <= 0.0) sum_dF = 1.0;

        double sum_wF  = 0.0;
        double sum_wF2 = 0.0;
        double sum_wCR = 0.0;
        double sum_w   = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
            const double wk  = dF[k] / sum_dF;
            const double Fk  = SF[k];
            const double CRk = SCR[k];
            sum_w   += wk;
            sum_wF  += wk * Fk;
            sum_wF2 += wk * Fk * Fk;
            sum_wCR += wk * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0) {
            const double meanF_Lehmer = sum_wF2 / sum_wF;  // weighted Lehmer mean
            const double meanCR       = sum_wCR / sum_w;

            if (H_ > 1) {
                // FIX: replace, do not average.
                MF_[mem_idx_]  = meanF_Lehmer;
                MCR_[mem_idx_] = meanCR;
                ++mem_idx_;
                if (mem_idx_ >= H_ - 1) mem_idx_ = 0;  // slot H_-1 is frozen
            }
        }
    }

    // Keep the last memory slot frozen at 0.9 (jSO / iL-SHADE convention).
    if (H_ > 0) {
        MF_[H_ - 1]  = 0.9;
        MCR_[H_ - 1] = 0.9;
    }

    // -----------------------------------------------------------------------
    // Mechanism A bookkeeping (only when adaptive weight is active)
    // -----------------------------------------------------------------------
    if (adaptive_weight_enable_) {
        const double iter_success_rate = (evals_done_this_iter > 0)
            ? static_cast<double>(strict_successes_this_iter) /
              static_cast<double>(evals_done_this_iter)
            : 0.0;
        success_ema_ = 0.85 * success_ema_ + 0.15 * iter_success_rate;

        if (best_improved_this_iter)
            no_best_improve_iters_ = 0;
        else
            ++no_best_improve_iters_;

        ++iteration_counter_;
    }

    // -----------------------------------------------------------------------
    // Mechanism B bookkeeping (only when prescreen is active)
    // -----------------------------------------------------------------------
    if (predictive_prescreen_enable_) {
        const double iter_skip_rate = (N > 0)
            ? static_cast<double>(skips_done_this_iter) /
              static_cast<double>(N)
            : 0.0;
        prescreen_skip_ema_ = 0.80 * prescreen_skip_ema_ + 0.20 * iter_skip_rate;
    }

    updateStop(FX_);
    printBest();
}

// ---------------------------------------------------------------------------
// end  —  optional end-of-run local refinement on best solution
// ---------------------------------------------------------------------------
void AWJSO::end()
{
    if (!prob_)             return;
    if (!end_local_refine_) return;
    if (end_local_method_.empty()) return;
    if (best_x_.empty())   return;

    auto res = localSearch(end_local_method_, best_x_);
    if (!res.first.empty() && std::isfinite(res.second) && res.second < best_f_) {
        best_x_ = std::move(res.first);
        best_f_ = res.second;

        // Replace worst individual in the population with the refined solution.
        if (!X_.empty()) {
            int    worst = 0;
            double fw    = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
