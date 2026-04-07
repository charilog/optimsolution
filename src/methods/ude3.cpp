#include "ude3.h"
#include "options.h"
#include "init.h"

#include <cstddef>
#include <numeric>
#include <cfloat>
#include <cstdint>

namespace optimsolution {

void UDE3::configure(const MethodConfig& mc)
{
    // Comment translated from Greek.
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
        pop_init_ = pop_cfg_;
    } else {
        pop_init_ = population() > 0 ? population() : pop_init_;
    }

    H_ = mc.getInt("H", H_);
    if (H_ <= 0) H_ = 10;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    top_frac_ = mc.getDbl("top_frac", top_frac_);
    if (top_frac_ <= 0.0 || top_frac_ >= 1.0) top_frac_ = 0.5;

    Lp_ = mc.getInt("Lp", Lp_);
    if (Lp_ <= 0) Lp_ = 50;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    p_best_rate_ = mc.getDbl("p_best_rate", p_best_rate_);
    if (p_best_rate_ <= 0.0 || p_best_rate_ > 1.0) p_best_rate_ = 0.2;

    stagnation_limit_ = mc.getInt("stagnation_limit", stagnation_limit_);
    if (stagnation_limit_ <= 0) stagnation_limit_ = 50;

    // In-run local search from [ude3].
    local_method_ = mc.getStr("local_method", local_method_);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // Comment translated from Greek.
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

void UDE3::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    if (pop_init_ <= 0) {
        pop_init_ = population() > 0 ? population() : 100;
    }
    setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    archive_.clear();
    archive_f_.clear();
    best_disc_.clear();
    best_disc_f_.clear();

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    const int N = static_cast<int>(X_.size());

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N; ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    strat_prob_.assign(NUM_STRAT_, 1.0 / NUM_STRAT_);
    strat_success_.assign(NUM_STRAT_, 0);
    strat_attempt_.assign(NUM_STRAT_, 0);
    gen_counter_ = 0;

    stagnation_.assign(N, 0);

    maxArc_ = static_cast<std::size_t>(N);
    best_disc_.assign(N, Vec());
    best_disc_f_.assign(N, std::numeric_limits<double>::infinity());

    updateStop(FX_);
    printBest();
}

void UDE3::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int  D = prob_->dimension();
    const int  m = static_cast<int>(std::min<std::size_t>(D, x.size()));

    for (int j = 0; j < m; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
    for (std::size_t j = m; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.0;
    }
}

// Comment translated from Greek.
int UDE3::selectRankedIndex(const std::vector<int>& sorted_idx,
                            const std::vector<double>& rank_prob,
                            int avoid1, int avoid2, int avoid3)
{
    const int N = static_cast<int>(sorted_idx.size());
    if (N <= 1) return 0;

    std::uniform_int_distribution<int>     Ui(0, N - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int t = 0; t < 2 * N; ++t) {
        int pos = Ui(rng_);
        if (pos < 0 || pos >= N) continue;
        int idx = sorted_idx[pos];
        if (idx == avoid1 || idx == avoid2 || idx == avoid3) continue;
        if (idx < 0 || idx >= static_cast<int>(rank_prob.size())) continue;
        double p = rank_prob[idx];
        double r = U01(rng_);
        if (r <= p) return idx;
    }

    // Fallback
    for (int t = 0; t < 2 * N; ++t) {
        int pos = Ui(rng_);
        if (pos < 0 || pos >= N) continue;
        int idx = sorted_idx[pos];
        if (idx != avoid1 && idx != avoid2 && idx != avoid3) return idx;
    }
    return sorted_idx[0];
}

// Creates a trial vector for each sub-strategy.
bool UDE3::generateTrial(int strat, int idx,
                         const std::vector<int>& sorted_idx,
                         int p_best_size,
                         const std::vector<double>& rank_prob,
                         double Fi, double CRi,
                         Vec& ui,
                         std::uniform_real_distribution<double>& U01)
{
    const int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();

    if (N < 4 || idx < 0 || idx >= N) {
        ui = (idx >= 0 && idx < N) ? X_[idx] : Vec(D, 0.0);
        return false;
    }

    std::uniform_int_distribution<int> Ui(0, N - 1);

    auto pick_random = [&](int avoid1, int avoid2, int avoid3) {
        for (int t = 0; t < 2 * N; ++t) {
            int j = Ui(rng_);
            if (j == avoid1 || j == avoid2 || j == avoid3) continue;
            return j;
        }
        return avoid1;
    };

    Vec v(D);

    if (strat == 0) {
        // DE/rand/1 with a ranking-based base vector.
        int r1 = selectRankedIndex(sorted_idx, rank_prob, idx, -1, -1);
        int r2 = pick_random(idx, r1, -1);
        int r3 = pick_random(idx, r1, r2);

        for (int j = 0; j < D; ++j) {
            v[j] = X_[r1][j] + Fi * (X_[r2][j] - X_[r3][j]);
        }
    } else if (strat == 1) {
        // DE/current-to-rand/1
        int r1 = selectRankedIndex(sorted_idx, rank_prob, idx, -1, -1);
        int r2 = pick_random(idx, r1, -1);
        int r3 = pick_random(idx, r1, r2);

        for (int j = 0; j < D; ++j) {
            v[j] = X_[idx][j]
                 + Fi * (X_[r1][j] - X_[idx][j])
                 + Fi * (X_[r2][j] - X_[r3][j]);
        }
    } else { // strat == 2: DE/current-to-pbest/1
        if (p_best_size <= 0 || p_best_size > N) return false;
        std::uniform_int_distribution<int> Up(0, p_best_size - 1);
        int p_index_pos = Up(rng_);
        if (p_index_pos < 0 || p_index_pos >= static_cast<int>(sorted_idx.size())) return false;
        int p_index     = sorted_idx[p_index_pos];

        int r1 = pick_random(idx, p_index, -1);
        int r2 = pick_random(idx, p_index, r1);

        for (int j = 0; j < D; ++j) {
            v[j] = X_[idx][j]
                 + Fi * (X_[p_index][j] - X_[idx][j])
                 + Fi * (X_[r1][j] - X_[r2][j]);
        }
    }

    // Binomial crossover
    ui.resize(D);
    std::uniform_int_distribution<int> UiD(0, D - 1);
    int jrand = UiD(rng_);
    for (int j = 0; j < D; ++j) {
        double r = U01(rng_);
        if (r <= CRi || j == jrand) ui[j] = v[j];
        else                        ui[j] = X_[idx][j];
    }

    return true;
}

int UDE3::selectStrategy(std::uniform_real_distribution<double>& U01)
{
    if (NUM_STRAT_ <= 1) return 0;
    double r = U01(rng_);
    double cum = 0.0;
    for (int s = 0; s < NUM_STRAT_; ++s) {
        double ps = (s < (int)strat_prob_.size()) ? strat_prob_[s] : 0.0;
        cum += ps;
        if (r <= cum) return s;
    }
    return NUM_STRAT_ - 1;
}

void UDE3::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();
    if (N <= 0 || D <= 0) return;

    // --- Defensive size alignment ---
    if ((int)FX_.size() != N)
        FX_.resize(N, std::numeric_limits<double>::infinity());
    if ((int)stagnation_.size() != N)
        stagnation_.assign(N, 0);
    if ((int)best_disc_.size() != N) {
        best_disc_.assign(N, Vec());
        best_disc_f_.assign(N, std::numeric_limits<double>::infinity());
    }

    // Comment translated from Greek.
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int i = 0; i < N; ++i) {
        if ((int)X_[i].size() != D) {
            X_[i].assign(D, 0.0);
            for (int j = 0; j < D; ++j)
                X_[i][j] = 0.5 * (L[j] + U[j]);
        }
    }
    // and for archive_.
    for (auto& a : archive_) {
        if ((int)a.size() != D) {
            a.assign(D, 0.0);
            for (int j = 0; j < D; ++j)
                a[j] = 0.5 * (L[j] + U[j]);
        }
    }

    // Sorts indices by fitness (ascending).
    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    // Ranking-based probabilities (linear ranking)
    std::vector<double> rank_prob(N, 0.0);
    double sumR = 0.0;
    for (int r = 0; r < N; ++r) {
        int Ri = N - r;
        sumR += Ri;
    }
    if (sumR <= 0.0) sumR = 1.0;
    for (int pos = 0; pos < N; ++pos) {
        int i = idx_sorted[pos];
        if (i < 0 || i >= N) continue;
        double Ri = N - pos;
        rank_prob[i] = Ri / sumR;
    }

    // Dual population split: A (top) / B (bottom)
    int A_size = static_cast<int>(std::round(top_frac_ * N));
    if (A_size < 1) A_size = 1;
    if (A_size >= N) A_size = N - 1;

    // p-best size for current-to-pbest.
    int p_best_size = static_cast<int>(std::round(p_best_rate_ * N));
    if (p_best_size < 2) p_best_size = std::min(2, N);
    if (p_best_size > N) p_best_size = N;

    // Random distributions.
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui_mem(0, std::max(0, H_-1));

    // SHADE success history
    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> weights;
    SF.reserve(N);
    SCR.reserve(N);
    weights.reserve(N);

    std::vector<Vec>    newPop(N);
    std::vector<double> newFit(N, std::numeric_limits<double>::quiet_NaN());

    if ((int)strat_success_.size() < NUM_STRAT_) {
        strat_success_.assign(NUM_STRAT_, 0);
        strat_attempt_.assign(NUM_STRAT_, 0);
    }

    gen_counter_++;

    auto storeDiscarded = [&](int i, const Vec& ui, double f_new) {
        if (i < 0 || i >= N) return;
        if (!std::isfinite(f_new)) return;
        // Best is discarded for this individual.
        if (f_new < best_disc_f_[i]) {
            best_disc_f_[i] = f_new;
            best_disc_[i]   = ui;
        }
        // global archive
        if (archive_.size() < maxArc_) {
            archive_.push_back(ui);
            archive_f_.push_back(f_new);
        } else if (!archive_.empty()) {
            std::uniform_int_distribution<std::size_t> UiA(0, archive_.size() - 1);
            std::size_t idx = UiA(rng_);
            // Comment translated from Greek.
            if (f_new < archive_f_[idx]) {
                archive_[idx]   = ui;
                archive_f_[idx] = f_new;
            }
        }
    };

    auto bdvsReplaceIfNeeded = [&](int i, bool& accepted) {
        if (stagnation_limit_ <= 0) return;
        if (i < 0 || i >= N) return;
        if (stagnation_[i] < stagnation_limit_) return;

        Vec cand;
        double f_cand = std::numeric_limits<double>::infinity();
        bool haveCand = false;

        if (!best_disc_[i].empty() &&
            std::isfinite(best_disc_f_[i]))
        {
            cand    = best_disc_[i];
            f_cand  = best_disc_f_[i];
            haveCand = true;
        } else if (!archive_.empty()) {
            std::uniform_int_distribution<std::size_t> UiA(0, archive_.size() - 1);
            std::size_t idx = UiA(rng_);
            if (idx < archive_.size() && idx < archive_f_.size()) {
                cand    = archive_[idx];
                f_cand  = archive_f_[idx];
                haveCand = true;
            }
        }

        if (!haveCand) return;

        // BDVS: allows a worse solution; this is purely diversification.
        if ((int)cand.size() != D) {
            cand.resize(D, 0.0);
            for (int j = 0; j < D; ++j)
                cand[j] = 0.5 * (L[j] + U[j]);
        }

        ensureBounds(cand);

        newPop[i]      = cand;
        newFit[i]      = f_cand;   // Fitness as recorded at the time of storage.
        stagnation_[i] = 0;
        accepted       = true;

        if (std::isfinite(f_cand) && f_cand < best_f_) {
            best_f_ = f_cand;
            best_x_ = cand;
        }
    };

    // Main loop.
    for (int pos = 0; pos < N; ++pos) {
        if (prob_->calls() >= max_evals_) break;

        int i = idx_sorted[pos];
        if (i < 0 || i >= N) continue;

        // Comment translated from Greek.
        int k = Ui_mem(rng_);
        if (k < 0 || k >= H_) k = 0;
        double muF  = (k < (int)MF_.size())  ? MF_[k]  : 0.5;
        double muCR = (k < (int)MCR_.size()) ? MCR_[k] : 0.8;
        if (muF <= 0.0) muF = 0.5;

        double Fi;
        {
            std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
            do {
                Fi = cauchy(rng_);
            } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
        }

        double CRi;
        {
            std::normal_distribution<double> normal(muCR, normal_std_CR_);
            CRi = normal(rng_);
            if (CRi < 0.0) CRi = 0.0;
            if (CRi > 1.0) CRi = 1.0;
        }

        bool accepted = false;

        // Comment translated from Greek.
        if (pos < A_size) {
            Vec   trial_best;
            double f_best = std::numeric_limits<double>::infinity();
            int   best_s  = -1;

            for (int s = 0; s < NUM_STRAT_; ++s) {
                Vec ui;
                bool ok = generateTrial(s, i, idx_sorted, p_best_size, rank_prob, Fi, CRi, ui, U01);
                if (!ok) continue;

                ensureBounds(ui);
                double f_new = eval(ui);
                if (prob_->calls() >= max_evals_) {
                    break;
                }

                strat_attempt_[s]++;

                if (f_new < f_best) {
                    f_best = f_new;
                    trial_best = ui;
                    best_s = s;
                }

                // Stores unsuccessful trials for BDVS.
                if (f_new > FX_[i]) {
                    storeDiscarded(i, ui, f_new);
                }
            }

            if (best_s >= 0 && f_best <= FX_[i]) {
                newPop[i] = trial_best;
                newFit[i] = f_best;
                stagnation_[i] = 0;
                strat_success_[best_s]++;

                double df = FX_[i] - f_best;
                if (df < 0.0) df = 0.0;
                SF.push_back(Fi);
                SCR.push_back(CRi);
                weights.push_back(df);

                accepted = true;
            } else {
                newPop[i] = X_[i];
                newFit[i] = FX_[i];
                stagnation_[i]++;

                // BDVS-style replacement when the stagnation limit is reached.
                bdvsReplaceIfNeeded(i, accepted);
            }
        } else {
            // Bottom sub-population B: one strategy with adaptation.
            int s = selectStrategy(U01);

            Vec ui;
            bool ok = generateTrial(s, i, idx_sorted, p_best_size, rank_prob, Fi, CRi, ui, U01);
            if (!ok) {
                newPop[i] = X_[i];
                newFit[i] = FX_[i];
                continue;
            }

            ensureBounds(ui);
            double f_new = eval(ui);
            if (prob_->calls() >= max_evals_) {
                newPop[i] = X_[i];
                newFit[i] = FX_[i];
                break;
            }

            strat_attempt_[s]++;

            if (f_new <= FX_[i]) {
                newPop[i] = ui;
                newFit[i] = f_new;
                stagnation_[i] = 0;
                strat_success_[s]++;

                double df = FX_[i] - f_new;
                if (df < 0.0) df = 0.0;
                SF.push_back(Fi);
                SCR.push_back(CRi);
                weights.push_back(df);

                accepted = true;
            } else {
                newPop[i] = X_[i];
                newFit[i] = FX_[i];
                stagnation_[i]++;

                // unsuccessful trial → BDVS archive
                storeDiscarded(i, ui, f_new);

                // BDVS-style replacement when stagnant.
                bdvsReplaceIfNeeded(i, accepted);
            }
        }

        if (accepted && newFit[i] < best_f_) {
            best_f_ = newFit[i];
            best_x_ = newPop[i];
        }
    }

    // Comment translated from Greek.
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(newFit[i])) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // SHADE update MF / MCR
    if (!SF.empty()) {
        double sum_w = 0.0;
        double sum_wF = 0.0, sum_wF2 = 0.0;
        double sum_wCR = 0.0;

        for (std::size_t k = 0; k < SF.size(); ++k) {
            double w   = weights[k];
            double Fk  = SF[k];
            double CRk = SCR[k];
            sum_w   += w;
            sum_wF  += w * Fk;
            sum_wF2 += w * Fk * Fk;
            sum_wCR += w * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0) {
            double meanF_Lehmer = sum_wF2 / sum_wF;
            double meanCR       = sum_wCR / sum_w;

            if ((int)MF_.size() < H_) MF_.assign(H_, 0.5);
            if ((int)MCR_.size() < H_) MCR_.assign(H_, 0.8);

            MF_[mem_idx_]  = (1.0 - c_mem_) * MF_[mem_idx_]  + c_mem_ * meanF_Lehmer;
            MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * meanCR;

            mem_idx_++;
            if (mem_idx_ >= H_) mem_idx_ = 0;
        }
    }

    // SaDE-style strategy adaptation every Lp_ generations.
    if (Lp_ > 0 && gen_counter_ >= Lp_) {
        double sumP = 0.0;
        std::vector<double> newProb(NUM_STRAT_, 0.0);

        for (int s = 0; s < NUM_STRAT_; ++s) {
            double succ = static_cast<double>(strat_success_[s]);
            double att  = static_cast<double>(strat_attempt_[s]);
            double ps   = (att > 0.0) ? (succ / att) : (1.0 / NUM_STRAT_);
            newProb[s]  = ps;
            sumP       += ps;
        }

        if (sumP > 0.0) {
            for (int s = 0; s < NUM_STRAT_; ++s) newProb[s] /= sumP;
            strat_prob_ = newProb;
        }

        std::fill(strat_success_.begin(), strat_success_.end(), 0);
        std::fill(strat_attempt_.begin(), strat_attempt_.end(), 0);
        gen_counter_ = 0;
    }

    // -------- in-run LOCAL SEARCH from [ude3] --------
    if (!local_method_.empty() && local_rate_ > 0.0) {
        std::uniform_real_distribution<double> Uloc(0.0, 1.0);
        for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
            if (Uloc(rng_) < local_rate_) {
                auto res = localSearch(local_method_, X_[i]);
                const Vec& xloc = res.first;
                double     floc = res.second;
                if (!xloc.empty() &&
                    xloc.size() == static_cast<std::size_t>(D) &&
                    std::isfinite(floc) && floc < FX_[i]) {
                    X_[i]          = xloc;
                    FX_[i]         = floc;
                    stagnation_[i] = 0;
                    if (floc < best_f_) {
                        best_f_ = floc;
                        best_x_ = xloc;
                    }
                }
            }
        }
    }

    updateStop(FX_);
    printBest();
}

void UDE3::end()
{
    if (!prob_) return;

    const int D = prob_->dimension();

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        const Vec& xloc = res.first;
        double     floc = res.second;
        if (!xloc.empty() &&
            xloc.size() == static_cast<std::size_t>(D) &&
            std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }

        if (!X_.empty()) {
            int    worst = 0;
            double fw    = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) {
                    fw = FX_[i];
                    worst = i;
                }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }

        printBest();
    }

    updateStop(FX_);
}

} // namespace optimsolution
