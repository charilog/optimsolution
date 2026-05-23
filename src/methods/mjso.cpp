#include "mjso.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

namespace {

inline double clampd(double x, double lo, double hi)
{
    return std::max(lo, std::min(hi, x));
}

} // namespace

void MJSO::configure(const MethodConfig& mc)
{
    // Base: current global population (from [global]).
    int basePop = population();
    if (basePop < 4) basePop = 4;

    // If a population is provided in [mjso], it takes precedence.
    int pop_override = mc.getInt("population", 0);
    if (pop_override > 3) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }

    // The Optimizer is also configured to appear correctly in the summary.
    setPopulation(pop_init_);

    // np_min
    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    // H (memory size)
    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    // p_min / p_max as in mLSHADE_RL.
    pmin_ = mc.getDbl("p_min", pmin_);
    pmax_ = mc.getDbl("p_max", pmax_);
    if (pmin_ <= 0.0) pmin_ = 0.05;
    if (pmax_ <= pmin_) pmax_ = std::max(pmin_ + 0.05, 0.5);

    arc_rate_ = mc.getDbl("arc_rate", arc_rate_);
    if (arc_rate_ < 0.0) arc_rate_ = 0.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    // In-run local search
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // Evaluation-rejection gate (transferred from UJSO).
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
    eval_gate_max_  = clampd(eval_gate_max_, eval_gate_base_, 0.90);
    eval_gate_random_keep_  = clampd(eval_gate_random_keep_, 0.0, 0.50);
    eval_gate_close_radius_ = clampd(eval_gate_close_radius_, 0.001, 0.20);
    eval_gate_tiny_step_    = clampd(eval_gate_tiny_step_, 0.0001, 0.05);
    eval_gate_margin_       = clampd(eval_gate_margin_, 0.0, 0.20);
}

void MJSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    // Safety: if a value is missing for any reason, the Optimizer value is used.
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

    // Initialize the F/CR memory.
    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

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

void MJSO::ensureInBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

double MJSO::normalizedDistanceToBest(const Vec& x) const
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

double MJSO::normalizedDistance(const Vec& a, const Vec& b) const
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

double MJSO::minSampledDistance(const Vec& x, int exclude_idx) const
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

double MJSO::scoreTrialForEvaluation(const Vec& trial,
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

bool MJSO::shouldEvaluateTrial(const Vec& trial,
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

void MJSO::updateEvaluationGateModel(double step,
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

double MJSO::meanNormalizedDistanceToBest() const
{
    if (X_.empty() || best_x_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& x : X_) {
        sum += normalizedDistanceToBest(x);
    }
    return sum / static_cast<double>(X_.size());
}

void MJSO::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    int N = static_cast<int>(X_.size());
    if (N < 4) return;

    double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));
    double nfes   = static_cast<double>(prob_->calls());

    // --- Linear population size reduction (LPSR) ---
    int targetN = static_cast<int>(
        std::round(pop_init_ - (pop_init_ - pop_min_) * (nfes / maxFES))
    );
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

    // --- p-best size ---
    nfes = static_cast<double>(prob_->calls());
    double fes_ratio = nfes / maxFES;
    if (fes_ratio < 0.0) fes_ratio = 0.0;
    if (fes_ratio > 1.0) fes_ratio = 1.0;

    double p = pmin_ + (pmax_ - pmin_) * fes_ratio;
    if (p < pmin_) p = pmin_;
    if (p > pmax_) p = pmax_;

    int p_best_size = static_cast<int>(std::round(p * N));
    if (p_best_size < 2) p_best_size = 2;
    if (p_best_size > N) p_best_size = N;

    // Indices are sorted by fitness.
    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<int> rank_pos(N, 0);
    for (int pos = 0; pos < N; ++pos) {
        rank_pos[idx_sorted[pos]] = pos;
    }

    const int elite_count = std::max(1, static_cast<int>(std::ceil(0.20 * static_cast<double>(N))));
    Vec elite_mean(D, 0.0);
    for (int k = 0; k < elite_count; ++k) {
        const Vec& xe = X_[idx_sorted[k]];
        for (int j = 0; j < D; ++j) {
            elite_mean[j] += xe[j];
        }
    }
    for (int j = 0; j < D; ++j) {
        elite_mean[j] /= static_cast<double>(elite_count);
    }

    // It starts from the current population/fitness so that if the
    // evaluation budget ends in the middle of the generation, the untouched
    // individuals keep their previous valid state.
    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> dF;

    SF.reserve(N);
    SCR.reserve(N);
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
    const double best_before = best_f_;
    int success_count = 0;

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        const Vec& xi = X_[i];
        const double f_old = FX_[i];
        bool evaluated = false;
        int prescreen_retries = 0;

        while (prescreen_retries <= eval_gate_retry_max_ && prob_->calls() < max_evals_) {
            // --- memory selection for F/CR ---
            std::uniform_int_distribution<int> Ui_mem(0, H_ - 1);
            int r_mem = Ui_mem(rng_);

            // jSO tweak: the last memory slot is pushed toward 0.9.
            if (H_ > 0 && r_mem == H_ - 1) {
                MF_[r_mem]  = 0.9;
                MCR_[r_mem] = 0.9;
            }

            double muF  = MF_[r_mem];
            double muCR = MCR_[r_mem];

            // --- F sampled from Cauchy(muF, scale) ---
            double Fi;
            {
                std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
                do {
                    Fi = cauchy(rng_);
                } while (Fi <= 0.0);
                if (Fi > 1.0) Fi = 1.0;
            }

            // --- CR sampled from Normal(muCR, std) ---
            double CRi;
            if (muCR < 0.0) {
                CRi = 0.0;
            } else {
                std::normal_distribution<double> normal(muCR, normal_std_CR_);
                CRi = normal(rng_);
            }
            if (CRi < 0.0) CRi = 0.0;
            if (CRi > 1.0) CRi = 1.0;

            // --- Early-stage CR/F adjustments (jSO-like) ---
            double g_ratio = static_cast<double>(prob_->calls()) / maxFES;
            if (g_ratio < 0.25) {
                if (CRi < 0.7) CRi = 0.7;
            } else if (g_ratio < 0.5) {
                if (CRi < 0.6) CRi = 0.6;
            }
            if (g_ratio < 0.6 && Fi > 0.7) {
                Fi = 0.7;
            }

            // --- Weighted F_w ---
            double fes_ratio_now = static_cast<double>(prob_->calls()) / maxFES;
            double Fw;
            if (fes_ratio_now < 0.2) {
                Fw = 0.7 * Fi;
            } else if (fes_ratio_now < 0.4) {
                Fw = 0.8 * Fi;
            } else {
                Fw = 1.2 * Fi;
            }
            const double F3 = clampd(std::fabs(Fw - Fi), 0.0, 1.0);

            // --- p-best selection ---
            std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
            int p_sorted_idx = Ui_pbest(rng_);
            int p_idx        = idx_sorted[p_sorted_idx];

            // --- r1, r2 selection ---
            auto r = pickDistinct(2, i, p_idx);
            if (r.size() < 2) {
                break;
            }
            int r1 = r[0];
            int r2 = r[1];

            const Vec& xp  = X_[p_idx];
            const Vec& xr1 = X_[r1];

            Vec xr2;
            if (!archive_.empty() && U01(rng_) < 0.5) {
                std::uniform_int_distribution<int> Ui_arc(0, static_cast<int>(archive_.size()) - 1);
                xr2 = archive_[Ui_arc(rng_)];
            } else {
                xr2 = X_[r2];
            }

            std::uniform_int_distribution<int> Ui_dim(0, D - 1);
            int jrand = Ui_dim(rng_);

            Vec ui(D);
            Vec vi(D);

            // current-to-pbest-w/1 mutation + binomial crossover
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CRi || j == jrand) {
                    vi[j] = xi[j]
                          + Fw * (xp[j] - xi[j])
                          + Fi * (xr1[j] - xr2[j])
                          + F3 * (elite_mean[j] - xi[j]);
                    ui[j] = vi[j];
                } else {
                    ui[j] = xi[j];
                }
            }

            ensureInBounds(ui);

            double gate_score = 0.0;
            ++gate_attempts_;
            if (!shouldEvaluateTrial(ui, xi, i, rank_pos[i], N, fes_ratio, diversity, &gate_score)) {
                ++gate_skips_;
                ++prescreen_retries;
                continue;
            }
            (void)gate_score;

            const double step = clampd(normalizedDistance(ui, xi), 0.0, 1.0);
            const double best_gain = clampd(normalizedDistanceToBest(xi) - normalizedDistanceToBest(ui),
                                            -1.0, 1.0);
            const double novelty = clampd(minSampledDistance(ui, i), 0.0, 1.0);
            const double rank_badness = (N > 1)
                ? clampd(static_cast<double>(rank_pos[i]) / static_cast<double>(N - 1), 0.0, 1.0)
                : 0.0;

            double f_new = eval(ui);
            evaluated = true;

            bool improved = (f_new <= f_old);
            updateEvaluationGateModel(step, best_gain, novelty, rank_badness, improved);

            if (improved) {
                ++success_count;
                newPop[i] = ui;
                newFit[i] = f_new;

                // Archive update.
                if (arc_rate_ > 0.0) {
                    archive_.push_back(xi);
                    int max_arc = static_cast<int>(std::round(arc_rate_ * pop_init_));
                    if (max_arc < 0) max_arc = 0;

                    if (max_arc == 0) {
                        archive_.clear();
                    } else {
                        while (static_cast<int>(archive_.size()) > max_arc) {
                            std::uniform_int_distribution<int> Ui_arc(0, static_cast<int>(archive_.size()) - 1);
                            int idx_remove = Ui_arc(rng_);
                            archive_.erase(archive_.begin() + idx_remove);
                        }
                    }
                }

                double diff = f_old - f_new;
                if (diff < 0.0) diff = 0.0;
                SF.push_back(Fi);
                SCR.push_back(CRi);
                dF.push_back(diff);

                // In-run local search with probability local_rate_.
                if (local_rate_ > 0.0 && !local_method_.empty() && prob_->calls() < max_evals_) {
                    double ru = U01(rng_);
                    if (ru < local_rate_) {
                        auto res = localSearch(local_method_, newPop[i]);
                        if (!res.first.empty() &&
                            std::isfinite(res.second) &&
                            res.second < newFit[i]) {
                            newPop[i] = std::move(res.first);
                            newFit[i] = res.second;
                        }
                    }
                }

                // best_f_ must track the actually accepted individual,
                // including a possibly improved local-search result.
                const double accepted_f = newFit[i];
                if (accepted_f < best_f_) {
                    best_f_ = accepted_f;
                    best_x_ = newPop[i];
                }
            }

            break;
        }

        (void)evaluated;
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // --- MF/MCR memory update ---
    if (!SF.empty()) {
        double sum_dF = 0.0;
        for (double v : dF) sum_dF += v;
        if (sum_dF <= 0.0) sum_dF = 1.0;

        std::vector<double> w(SF.size());
        for (size_t k = 0; k < SF.size(); ++k) {
            w[k] = dF[k] / sum_dF;
        }

        double sum_wF  = 0.0;
        double sum_wF2 = 0.0;
        double sum_wCR = 0.0;
        double sum_w   = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
            double wk  = w[k];
            double Fk  = SF[k];
            double CRk = SCR[k];

            sum_w   += wk;
            sum_wF  += wk * Fk;
            sum_wF2 += wk * Fk * Fk;
            sum_wCR += wk * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0) {
            double meanF_Lehmer = sum_wF2 / sum_wF;
            double meanCR       = sum_wCR / sum_w;

            MF_[mem_idx_]  = (1.0 - c_mem_) * MF_[mem_idx_]  + c_mem_ * meanF_Lehmer;
            MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * meanCR;

            ++mem_idx_;
            if (mem_idx_ >= H_) mem_idx_ = 0;
        }
    }

    prev_success_rate_ = static_cast<double>(success_count) / static_cast<double>(std::max(N, 1));
    if (best_f_ < best_before) stagnant_gens_ = 0;
    else ++stagnant_gens_;

    updateStop(FX_);
    printBest();
}

void MJSO::end()
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

        // Replace the worst individual with the polished best.
        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < static_cast<int>(FX_.size()); ++i) {
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
