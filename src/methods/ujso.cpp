#include "ujso.h"
#include "init.h"
#include <numeric>
#include <cctype>

namespace optimsolution {

namespace {

inline double clampd(double x, double lo, double hi)
{
    return std::max(lo, std::min(hi, x));
}

inline double weightedArithmeticMean(const std::vector<double>& vals,
                                     const std::vector<double>& w)
{
    if (vals.empty() || vals.size() != w.size()) return 0.0;
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        num += w[i] * vals[i];
        den += w[i];
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

inline double weightedLehmerMean(const std::vector<double>& vals,
                                 const std::vector<double>& w)
{
    if (vals.empty() || vals.size() != w.size()) return 0.0;
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        num += w[i] * vals[i] * vals[i];
        den += w[i] * vals[i];
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

inline double weightedAbsDeviation(const std::vector<double>& vals,
                                   const std::vector<double>& w,
                                   double center)
{
    if (vals.empty() || vals.size() != w.size()) return 0.0;
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        num += w[i] * std::fabs(vals[i] - center);
        den += w[i];
    }
    return (den > 0.0) ? (num / den) : 0.0;
}

} // namespace

void UJSO::configure(const MethodConfig& mc)
{
    int basePop = population();
    if (basePop < 4) basePop = 4;

    int pop_override = mc.getInt("population", 0);
    pop_init_ = (pop_override > 3) ? pop_override : basePop;
    setPopulation(pop_init_);

    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    p_floor_ = mc.getDbl("p_floor", p_floor_);
    p_ceil_  = mc.getDbl("p_ceil",  p_ceil_);
    if (p_floor_ <= 0.0) p_floor_ = 0.03;
    if (p_ceil_ <= p_floor_) p_ceil_ = std::max(0.10, p_floor_ + 0.10);
    if (p_ceil_ > 0.50) p_ceil_ = 0.50;

    k_floor_ = mc.getDbl("k_floor", k_floor_);
    k_ceil_  = mc.getDbl("k_ceil",  k_ceil_);
    if (k_floor_ <= 0.0) k_floor_ = 0.50;
    if (k_ceil_ <= k_floor_) k_ceil_ = k_floor_ + 0.20;

    arc_rate_init_ = mc.getDbl("arc_rate", arc_rate_init_);
    arc_rate_min_  = mc.getDbl("arc_rate_min", arc_rate_min_);
    arc_rate_max_  = mc.getDbl("arc_rate_max", arc_rate_max_);
    if (arc_rate_min_ < 0.0) arc_rate_min_ = 0.0;
    if (arc_rate_max_ < arc_rate_min_) arc_rate_max_ = arc_rate_min_ + 0.5;
    if (arc_rate_init_ < arc_rate_min_) arc_rate_init_ = arc_rate_min_;
    if (arc_rate_init_ > arc_rate_max_) arc_rate_init_ = arc_rate_max_;
    arc_rate_eff_ = arc_rate_init_;

    archive_use_prob_ = mc.getDbl("archive_prob", archive_use_prob_);
    archive_use_prob_ = clampd(archive_use_prob_, 0.10, 0.90);

    np_exp_ = mc.getDbl("np_exp", np_exp_);
    np_exp_min_ = mc.getDbl("np_exp_min", np_exp_min_);
    np_exp_max_ = mc.getDbl("np_exp_max", np_exp_max_);
    if (np_exp_min_ <= 0.0) np_exp_min_ = 0.70;
    if (np_exp_max_ < np_exp_min_) np_exp_max_ = np_exp_min_ + 0.50;
    np_exp_ = clampd(np_exp_, np_exp_min_, np_exp_max_);

    sigma_floor_F_  = mc.getDbl("sigma_floor_f",  sigma_floor_F_);
    sigma_floor_CR_ = mc.getDbl("sigma_floor_cr", sigma_floor_CR_);
    sigma_floor_P_  = mc.getDbl("sigma_floor_p",  sigma_floor_P_);
    sigma_floor_K_  = mc.getDbl("sigma_floor_k",  sigma_floor_K_);
    sigma_floor_F_  = clampd(sigma_floor_F_,  0.001, 0.30);
    sigma_floor_CR_ = clampd(sigma_floor_CR_, 0.001, 0.30);
    sigma_floor_P_  = clampd(sigma_floor_P_,  0.001, 0.20);
    sigma_floor_K_  = clampd(sigma_floor_K_,  0.001, 0.40);

    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    double lr = mc.getDbl("local_rate", local_rate_base_);
    local_rate_base_ = clampd(lr, 0.0, 1.0);
    local_rate_eff_  = local_rate_base_;
    local_rate_max_  = clampd(std::max(local_rate_base_ * 2.0, local_rate_base_ + 0.10),
                              0.0, 0.60);

    eval_gate_enable_       = mc.getInt("eval_gate_enable", eval_gate_enable_ ? 1 : 0) != 0;
    eval_gate_retry_max_    = mc.getInt("eval_gate_retry_max", eval_gate_retry_max_);
    eval_gate_base_         = mc.getDbl("eval_gate_base", eval_gate_base_);
    eval_gate_max_          = mc.getDbl("eval_gate_max", eval_gate_max_);
    eval_gate_random_keep_  = mc.getDbl("eval_gate_random_keep", eval_gate_random_keep_);
    eval_gate_close_radius_ = mc.getDbl("eval_gate_close_radius", eval_gate_close_radius_);
    eval_gate_tiny_step_    = mc.getDbl("eval_gate_tiny_step", eval_gate_tiny_step_);
    eval_gate_margin_       = mc.getDbl("eval_gate_margin", eval_gate_margin_);

    if (eval_gate_retry_max_ < 0) eval_gate_retry_max_ = 0;
    if (eval_gate_retry_max_ > 6) eval_gate_retry_max_ = 6;
    eval_gate_base_ = clampd(eval_gate_base_, 0.0, 0.70);
    eval_gate_max_  = clampd(eval_gate_max_,  eval_gate_base_, 0.90);
    eval_gate_random_keep_  = clampd(eval_gate_random_keep_, 0.0, 0.50);
    eval_gate_close_radius_ = clampd(eval_gate_close_radius_, 0.001, 0.20);
    eval_gate_tiny_step_    = clampd(eval_gate_tiny_step_, 0.0001, 0.05);
    eval_gate_margin_       = clampd(eval_gate_margin_, 0.0, 0.20);
}

void UJSO::init()
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

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
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

    MF_.assign(H_, 0.55);
    MCR_.assign(H_, 0.85);
    MP_.assign(H_, 0.18);
    MK_.assign(H_, 1.00);

    sigmaF_.assign(H_, 0.10);
    sigmaCR_.assign(H_, 0.10);
    sigmaP_.assign(H_, 0.04);
    sigmaK_.assign(H_, 0.10);

    mem_idx_ = 0;
    archive_use_prob_ = clampd(archive_use_prob_, 0.10, 0.90);
    arc_rate_eff_ = clampd(arc_rate_init_, arc_rate_min_, arc_rate_max_);
    local_rate_eff_ = local_rate_base_;
    stagnant_gens_ = 0;
    prev_success_rate_ = 0.0;

    gate_attempts_ = 0;
    gate_skips_ = 0;
    success_mu_step_ = 0.050;
    success_mu_bestgain_ = 0.010;
    success_mu_novelty_ = 0.040;
    success_mu_rank_ = 0.500;
    failure_mu_step_ = 0.015;
    failure_mu_bestgain_ = -0.002;
    failure_mu_novelty_ = 0.010;
    failure_mu_rank_ = 0.350;
    gate_success_count_ = 0;
    gate_failure_count_ = 0;

    updateStop(FX_);
    printBest();
}

void UJSO::ensureInBounds(Vec& x, const Vec& parent)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) {
            x[j] = 0.5 * (L[j] + U[j]);
        }
        if (x[j] < L[j]) {
            x[j] = 0.5 * (L[j] + parent[j]);
            if (x[j] < L[j]) x[j] = L[j];
        } else if (x[j] > U[j]) {
            x[j] = 0.5 * (U[j] + parent[j]);
            if (x[j] > U[j]) x[j] = U[j];
        }
    }
}

double UJSO::normalizedDistanceToBest(const Vec& x) const
{
    if (!prob_ || x.empty() || best_x_.empty()) return 0.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const size_t D = x.size();
    double acc = 0.0;
    for (size_t j = 0; j < D; ++j) {
        double range = U[j] - L[j];
        if (range <= 0.0) range = 1.0;
        double z = (x[j] - best_x_[j]) / range;
        acc += z * z;
    }
    return std::sqrt(acc / static_cast<double>(D));
}

double UJSO::normalizedDistance(const Vec& a, const Vec& b) const
{
    if (!prob_ || a.size() != b.size() || a.empty()) return 0.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const size_t D = a.size();
    double acc = 0.0;
    for (size_t j = 0; j < D; ++j) {
        double range = U[j] - L[j];
        if (range <= 0.0) range = 1.0;
        const double z = (a[j] - b[j]) / range;
        acc += z * z;
    }
    return std::sqrt(acc / static_cast<double>(D));
}

double UJSO::minSampledDistance(const Vec& x, int exclude_idx) const
{
    if (x.empty()) return 0.0;

    double dmin = std::numeric_limits<double>::infinity();

    auto updateMin = [&](const Vec& y) {
        if (y.size() != x.size()) return;
        dmin = std::min(dmin, normalizedDistance(x, y));
    };

    if (!best_x_.empty()) updateMin(best_x_);

    const int N = static_cast<int>(X_.size());
    if (N > 0) {
        const int step = std::max(1, N / 6);
        for (int idx = 0; idx < N; idx += step) {
            if (idx == exclude_idx) continue;
            updateMin(X_[idx]);
        }
    }

    const int A = static_cast<int>(archive_.size());
    if (A > 0) {
        const int stepA = std::max(1, A / 4);
        for (int idx = 0; idx < A; idx += stepA) {
            updateMin(archive_[idx]);
        }
    }

    if (!std::isfinite(dmin)) return 1.0;
    return dmin;
}

double UJSO::scoreTrialForEvaluation(const Vec& trial,
                                     const Vec& parent,
                                     int parent_idx,
                                     int parent_rank,
                                     int pop_size) const
{
    const double step = clampd(normalizedDistance(trial, parent), 0.0, 1.0);
    const double parent_to_best = clampd(normalizedDistanceToBest(parent), 0.0, 1.0);
    const double trial_to_best  = clampd(normalizedDistanceToBest(trial),  0.0, 1.0);
    const double best_gain = clampd(parent_to_best - trial_to_best, -1.0, 1.0);
    const double novelty = clampd(minSampledDistance(trial, parent_idx), 0.0, 1.0);
    const double rank_badness = (pop_size > 1)
        ? clampd(static_cast<double>(parent_rank) / static_cast<double>(pop_size - 1), 0.0, 1.0)
        : 0.0;

    const double ds = std::fabs(step - success_mu_step_)
                    + 1.5 * std::fabs(best_gain - success_mu_bestgain_)
                    + std::fabs(novelty - success_mu_novelty_)
                    + 0.5 * std::fabs(rank_badness - success_mu_rank_);

    const double df = std::fabs(step - failure_mu_step_)
                    + 1.5 * std::fabs(best_gain - failure_mu_bestgain_)
                    + std::fabs(novelty - failure_mu_novelty_)
                    + 0.5 * std::fabs(rank_badness - failure_mu_rank_);

    double score = (df - ds)
                 + 0.90 * best_gain
                 + 0.30 * novelty
                 + 0.15 * step
                 + 0.08 * rank_badness;

    if (step < eval_gate_tiny_step_) score -= 0.25;
    if (novelty < eval_gate_close_radius_) score -= 0.20;
    return score;
}

bool UJSO::shouldEvaluateTrial(const Vec& trial,
                               const Vec& parent,
                               int parent_idx,
                               int parent_rank,
                               int pop_size,
                               double fes_ratio,
                               double diversity,
                               double* out_score)
{
    if (!eval_gate_enable_) {
        if (out_score) *out_score = 1.0;
        return true;
    }

    const double step = clampd(normalizedDistance(trial, parent), 0.0, 1.0);
    const double best_gain = clampd(normalizedDistanceToBest(parent) - normalizedDistanceToBest(trial),
                                    -1.0, 1.0);
    const double novelty = clampd(minSampledDistance(trial, parent_idx), 0.0, 1.0);
    const double rank_badness = (pop_size > 1)
        ? clampd(static_cast<double>(parent_rank) / static_cast<double>(pop_size - 1), 0.0, 1.0)
        : 0.0;

    const double score = scoreTrialForEvaluation(trial, parent, parent_idx, parent_rank, pop_size);
    if (out_score) *out_score = score;

    if (best_gain > 0.020) return true;
    if (novelty > std::max(0.05, 1.6 * eval_gate_close_radius_)) return true;
    if (step > std::max(0.05, 1.8 * success_mu_step_)) return true;
    if (stagnant_gens_ >= 4 && (novelty > eval_gate_close_radius_ || best_gain > -0.005)) return true;

    double gate_strength = eval_gate_base_
                         + 0.20 * std::max(0.0, fes_ratio - 0.25)
                         + 0.15 * std::max(0.0, 0.12 - diversity)
                         + 0.10 * std::max(0.0, 0.10 - prev_success_rate_);
    gate_strength = clampd(gate_strength, 0.0, eval_gate_max_);

    double threshold = eval_gate_margin_
                     + 0.35 * gate_strength
                     - 0.10 * std::min(stagnant_gens_, 6) / 6.0
                     - 0.05 * rank_badness;

    if (step < eval_gate_tiny_step_ && novelty < eval_gate_close_radius_ && best_gain <= 0.0) {
        threshold += 0.10;
    }

    double keep_prob = eval_gate_random_keep_
                     + 0.15 * std::min(stagnant_gens_, 6) / 6.0
                     + 0.08 * diversity;
    keep_prob = clampd(keep_prob, 0.0, 0.60);

    if (score >= threshold) return true;
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < keep_prob;
}

void UJSO::updateEvaluationGateModel(double step,
                                     double best_gain,
                                     double novelty,
                                     double rank_badness,
                                     bool success)
{
    step = clampd(step, 0.0, 1.0);
    best_gain = clampd(best_gain, -1.0, 1.0);
    novelty = clampd(novelty, 0.0, 1.0);
    rank_badness = clampd(rank_badness, 0.0, 1.0);

    if (success) {
        ++gate_success_count_;
        const double a = 1.0 / static_cast<double>(gate_success_count_);
        success_mu_step_      = (1.0 - a) * success_mu_step_      + a * step;
        success_mu_bestgain_  = (1.0 - a) * success_mu_bestgain_  + a * best_gain;
        success_mu_novelty_   = (1.0 - a) * success_mu_novelty_   + a * novelty;
        success_mu_rank_      = (1.0 - a) * success_mu_rank_      + a * rank_badness;
    } else {
        ++gate_failure_count_;
        const double a = 1.0 / static_cast<double>(gate_failure_count_);
        failure_mu_step_      = (1.0 - a) * failure_mu_step_      + a * step;
        failure_mu_bestgain_  = (1.0 - a) * failure_mu_bestgain_  + a * best_gain;
        failure_mu_novelty_   = (1.0 - a) * failure_mu_novelty_   + a * novelty;
        failure_mu_rank_      = (1.0 - a) * failure_mu_rank_      + a * rank_badness;
    }
}

double UJSO::meanNormalizedDistanceToBest() const
{
    if (X_.empty() || best_x_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& x : X_) {
        sum += normalizedDistanceToBest(x);
    }
    return sum / static_cast<double>(X_.size());
}

void UJSO::updateAdaptiveState()
{
    const double div = clampd(meanNormalizedDistanceToBest(), 0.0, 1.0);
    const double succ = clampd(prev_success_rate_, 0.0, 1.0);
    const double stag = static_cast<double>(std::min(stagnant_gens_, 12));

    // Slower shrink when the search stagnates or diversity becomes too small.
    double target_exp = 1.0
                      + 1.10 * std::max(0.0, 0.15 - div)
                      + 0.80 * std::max(0.0, 0.12 - succ)
                      + 0.06 * stag;
    if (div > 0.25 && succ > 0.18) {
        target_exp -= 0.20;
    }
    target_exp = clampd(target_exp, np_exp_min_, np_exp_max_);
    np_exp_ = 0.85 * np_exp_ + 0.15 * target_exp;

    // Archive grows when exploration is needed and shrinks when direct population
    // moves are already productive.
    double target_arc = arc_rate_init_
                      + 1.00 * std::max(0.0, 0.14 - div)
                      + 0.60 * std::max(0.0, 0.10 - succ)
                      + 0.05 * stag;
    if (div > 0.28 && succ > 0.18) {
        target_arc -= 0.25;
    }
    target_arc = clampd(target_arc, arc_rate_min_, arc_rate_max_);
    arc_rate_eff_ = 0.85 * arc_rate_eff_ + 0.15 * target_arc;

    if (local_rate_base_ > 0.0) {
        double target_local = local_rate_base_
                            * (1.0
                               + 0.75 * std::max(0.0, 0.10 - succ)
                               + 0.50 * std::max(0.0, 0.15 - div));
        target_local += 0.01 * stag;
        target_local = clampd(target_local, 0.0, local_rate_max_);
        local_rate_eff_ = 0.85 * local_rate_eff_ + 0.15 * target_local;
    }
}

void UJSO::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    int N = static_cast<int>(X_.size());
    if (N < 4) return;

    updateAdaptiveState();

    const double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));
    const double nfes0  = static_cast<double>(prob_->calls());
    const double fes_ratio = clampd(nfes0 / maxFES, 0.0, 1.0);

    // Adaptive nonlinear population size reduction.
    int targetN = static_cast<int>(std::round(
        pop_init_ - (pop_init_ - pop_min_) * std::pow(fes_ratio, np_exp_)
    ));
    if (targetN < pop_min_)  targetN = pop_min_;
    if (targetN > pop_init_) targetN = pop_init_;

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

    if (N < 4) return;

    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<int> rank_pos(N, 0);
    for (int pos = 0; pos < N; ++pos) {
        rank_pos[idx_sorted[pos]] = pos;
    }

    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> SP;
    std::vector<double> SK;
    std::vector<double> dF;
    SF.reserve(N);
    SCR.reserve(N);
    SP.reserve(N);
    SK.reserve(N);
    dF.reserve(N);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    auto pickDistinct = [&](int count, int avoid1, int avoid2) {
        std::vector<int> pool;
        pool.reserve(N);
        for (int j = 0; j < N; ++j) {
            if (j == avoid1 || j == avoid2) continue;
            pool.push_back(j);
        }
        if (count > static_cast<int>(pool.size())) count = static_cast<int>(pool.size());
        std::shuffle(pool.begin(), pool.end(), rng_);
        pool.resize(count);
        return pool;
    };

    const double diversity = clampd(meanNormalizedDistanceToBest(), 0.0, 1.0);
    double p_lo_eff = std::max(p_floor_, 2.0 / static_cast<double>(std::max(N, 2)));
    double p_hi_eff = 0.10 + 0.20 * diversity + 0.02 * std::min(stagnant_gens_, 8);
    p_hi_eff = clampd(p_hi_eff, std::max(p_lo_eff + 0.02, p_floor_), p_ceil_);

    double k_lo_eff = clampd(k_floor_ - 0.20 * std::max(0.0, diversity - 0.20), 0.30, k_ceil_);
    double k_hi_eff = clampd(k_ceil_ + 0.25 * std::max(0.0, 0.12 - prev_success_rate_)
                                      + 0.05 * std::min(stagnant_gens_, 8),
                             std::max(k_lo_eff + 0.05, k_floor_), 2.00);

    double best_before = best_f_;
    int success_count = 0;
    int arch_uses = 0;
    int pop_uses  = 0;
    double delta_arch = 0.0;
    double delta_pop  = 0.0;
    int ls_attempts = 0;
    int ls_success  = 0;

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        const Vec& xi = X_[i];
        const double f_old = FX_[i];
        const double rank_badness = (N > 1)
            ? clampd(static_cast<double>(rank_pos[i]) / static_cast<double>(N - 1), 0.0, 1.0)
            : 0.0;

        int prescreen_retries = 0;

        while (prescreen_retries <= eval_gate_retry_max_ && prob_->calls() < max_evals_) {
            std::uniform_int_distribution<int> Ui_mem(0, H_ - 1);
            const int r_mem = Ui_mem(rng_);

            const double muF  = clampd(MF_[r_mem],  0.05, 1.00);
            const double muCR = clampd(MCR_[r_mem], 0.00, 1.00);
            const double muP  = clampd(MP_[r_mem],  p_lo_eff, p_hi_eff);
            const double muK  = clampd(MK_[r_mem],  k_lo_eff, k_hi_eff);

            double Fi;
            {
                std::cauchy_distribution<double> cauchy(muF, std::max(sigmaF_[r_mem], sigma_floor_F_));
                do {
                    Fi = cauchy(rng_);
                } while (Fi <= 0.0);
                if (Fi > 1.0) Fi = 1.0;
            }

            std::normal_distribution<double> normalCR(muCR, std::max(sigmaCR_[r_mem], sigma_floor_CR_));
            double CRi = clampd(normalCR(rng_), 0.0, 1.0);

            std::normal_distribution<double> normalP(muP, std::max(sigmaP_[r_mem], sigma_floor_P_));
            double pi = clampd(normalP(rng_), p_lo_eff, p_hi_eff);

            std::normal_distribution<double> normalK(muK, std::max(sigmaK_[r_mem], sigma_floor_K_));
            double Ki = clampd(normalK(rng_), k_lo_eff, k_hi_eff);

            const double di = clampd(normalizedDistanceToBest(xi), 0.0, 1.0);
            if (di < 0.08) {
                Fi = std::min(1.0, Fi * 1.10);
                Ki = std::min(k_hi_eff, Ki * 1.05);
            } else if (di > 0.25) {
                CRi = clampd(CRi * 0.95, 0.0, 1.0);
            }

            int p_best_size = static_cast<int>(std::round(pi * N));
            if (p_best_size < 2) p_best_size = 2;
            if (p_best_size > N) p_best_size = N;

            std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
            const int p_idx = idx_sorted[Ui_pbest(rng_)];

            auto r = pickDistinct(2, i, p_idx);
            if (r.size() < 2) break;
            const int r1 = r[0];
            const int r2 = r[1];

            const Vec& xp  = X_[p_idx];
            const Vec& xr1 = X_[r1];

            Vec xr2;
            bool used_archive = false;
            if (!archive_.empty() && U01(rng_) < archive_use_prob_) {
                std::uniform_int_distribution<int> Ui_arc(0, static_cast<int>(archive_.size()) - 1);
                xr2 = archive_[Ui_arc(rng_)];
                used_archive = true;
            } else {
                xr2 = X_[r2];
            }

            const double Fw = clampd(Ki * Fi, 0.0, 2.0);
            std::uniform_int_distribution<int> Ui_dim(0, D - 1);
            const int jrand = Ui_dim(rng_);

            Vec ui(D);
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CRi || j == jrand) {
                    ui[j] = xi[j]
                          + Fw * (xp[j] - xi[j])
                          + Fi * (xr1[j] - xr2[j]);
                } else {
                    ui[j] = xi[j];
                }
            }

            ensureInBounds(ui, xi);

            double gate_score = 0.0;
            ++gate_attempts_;
            if (!shouldEvaluateTrial(ui, xi, i, rank_pos[i], N, fes_ratio, diversity, &gate_score)) {
                ++gate_skips_;
                ++prescreen_retries;
                continue;
            }
            (void)gate_score;

            if (used_archive) ++arch_uses;
            else              ++pop_uses;

            const double step = clampd(normalizedDistance(ui, xi), 0.0, 1.0);
            const double best_gain = clampd(normalizedDistanceToBest(xi) - normalizedDistanceToBest(ui),
                                            -1.0, 1.0);
            const double novelty = clampd(minSampledDistance(ui, i), 0.0, 1.0);

            const double f_new = eval(ui);
            const bool success = (f_new < f_old);
            updateEvaluationGateModel(step, best_gain, novelty, rank_badness, success);

            if (success) {
                ++success_count;
                newPop[i] = ui;
                newFit[i] = f_new;

                archive_.push_back(xi);
                int max_arc = static_cast<int>(std::round(arc_rate_eff_ * static_cast<double>(N)));
                if (max_arc < 0) max_arc = 0;
                if (max_arc == 0) {
                    archive_.clear();
                } else {
                    while (static_cast<int>(archive_.size()) > max_arc) {
                        std::uniform_int_distribution<int> Ui_arc(0, static_cast<int>(archive_.size()) - 1);
                        archive_.erase(archive_.begin() + Ui_arc(rng_));
                    }
                }

                const double diff = std::max(0.0, f_old - f_new);
                SF.push_back(Fi);
                SCR.push_back(CRi);
                SP.push_back(pi);
                SK.push_back(Ki);
                dF.push_back(diff);

                if (used_archive) delta_arch += diff;
                else              delta_pop  += diff;

                if (local_rate_eff_ > 0.0 && !local_method_.empty() && prob_->calls() < max_evals_) {
                    if (U01(rng_) < local_rate_eff_) {
                        ++ls_attempts;
                        auto res = localSearch(local_method_, newPop[i]);
                        if (!res.first.empty() && std::isfinite(res.second) && res.second < newFit[i]) {
                            newPop[i] = std::move(res.first);
                            newFit[i] = res.second;
                            ++ls_success;
                        }
                    }
                }

                if (newFit[i] < best_f_) {
                    best_f_ = newFit[i];
                    best_x_ = newPop[i];
                }
            }

            break;
        }

    }

    X_.swap(newPop);
    FX_.swap(newFit);

    if (!SF.empty()) {
        double sum_dF = std::accumulate(dF.begin(), dF.end(), 0.0);
        if (sum_dF <= 0.0) sum_dF = 1.0;

        std::vector<double> w(dF.size(), 0.0);
        for (size_t k = 0; k < dF.size(); ++k) {
            w[k] = dF[k] / sum_dF;
        }

        const double meanF  = weightedLehmerMean(SF,  w);
        const double meanCR = weightedArithmeticMean(SCR, w);
        const double meanP  = weightedArithmeticMean(SP,  w);
        const double meanK  = weightedArithmeticMean(SK,  w);

        MF_[mem_idx_]  = clampd(0.5 * MF_[mem_idx_]  + 0.5 * meanF,  0.05, 1.0);
        MCR_[mem_idx_] = clampd(0.5 * MCR_[mem_idx_] + 0.5 * meanCR, 0.0, 1.0);
        MP_[mem_idx_]  = clampd(0.5 * MP_[mem_idx_]  + 0.5 * meanP,  p_floor_, p_ceil_);
        MK_[mem_idx_]  = clampd(0.5 * MK_[mem_idx_]  + 0.5 * meanK,  0.3, 2.0);

        sigmaF_[mem_idx_]  = clampd(0.7 * sigmaF_[mem_idx_]  + 0.3 * weightedAbsDeviation(SF,  w, meanF),
                                    sigma_floor_F_, 0.30);
        sigmaCR_[mem_idx_] = clampd(0.7 * sigmaCR_[mem_idx_] + 0.3 * weightedAbsDeviation(SCR, w, meanCR),
                                    sigma_floor_CR_, 0.30);
        sigmaP_[mem_idx_]  = clampd(0.7 * sigmaP_[mem_idx_]  + 0.3 * weightedAbsDeviation(SP,  w, meanP),
                                    sigma_floor_P_, 0.20);
        sigmaK_[mem_idx_]  = clampd(0.7 * sigmaK_[mem_idx_]  + 0.3 * weightedAbsDeviation(SK,  w, meanK),
                                    sigma_floor_K_, 0.40);

        ++mem_idx_;
        if (mem_idx_ >= H_) mem_idx_ = 0;
    } else {
        sigmaF_[mem_idx_]  = clampd(sigmaF_[mem_idx_]  * 1.03, sigma_floor_F_,  0.30);
        sigmaCR_[mem_idx_] = clampd(sigmaCR_[mem_idx_] * 1.03, sigma_floor_CR_, 0.30);
        sigmaP_[mem_idx_]  = clampd(sigmaP_[mem_idx_]  * 1.03, sigma_floor_P_,  0.20);
        sigmaK_[mem_idx_]  = clampd(sigmaK_[mem_idx_]  * 1.03, sigma_floor_K_,  0.40);
    }

    // Adaptive archive usage probability.
    if (arch_uses > 0 || pop_uses > 0) {
        const double scoreA = (arch_uses > 0) ? (delta_arch / static_cast<double>(arch_uses)) : 0.0;
        const double scoreP = (pop_uses  > 0) ? (delta_pop  / static_cast<double>(pop_uses )) : 0.0;
        double targetPA = 0.5;
        if (scoreA > 0.0 || scoreP > 0.0) {
            targetPA = scoreA / (scoreA + scoreP + 1e-16);
        }
        archive_use_prob_ = clampd(0.8 * archive_use_prob_ + 0.2 * targetPA, 0.10, 0.90);
    }

    if (local_rate_base_ > 0.0 && ls_attempts > 0) {
        const double ls_sr = static_cast<double>(ls_success) / static_cast<double>(ls_attempts);
        if (ls_sr > 0.25) {
            local_rate_eff_ = clampd(local_rate_eff_ + 0.02, 0.0, local_rate_max_);
        } else if (ls_sr < 0.05) {
            local_rate_eff_ = clampd(local_rate_eff_ - 0.02,
                                     0.0,
                                     local_rate_max_);
        }
    }

    prev_success_rate_ = static_cast<double>(success_count) / static_cast<double>(std::max(N, 1));
    if (best_f_ < best_before) stagnant_gens_ = 0;
    else ++stagnant_gens_;

    updateStop(FX_);
    printBest();
}

void UJSO::end()
{
    if (!prob_) return;
    if (!end_local_refine_) return;
    if (end_local_method_.empty()) return;
    if (best_x_.empty()) return;

    auto res = localSearch(end_local_method_, best_x_);
    if (!res.first.empty() && std::isfinite(res.second) && res.second < best_f_) {
        best_x_ = std::move(res.first);
        best_f_ = res.second;

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < static_cast<int>(FX_.size()); ++i) {
                if (FX_[i] > fw) {
                    fw = FX_[i];
                    worst = i;
                }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
