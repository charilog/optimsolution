#include "mlshaderl.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

void mLSHADE_RL::configure(const MethodConfig& mc)
{
    // --- population as in PSAO ---
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
        pop_init_ = pop_cfg_;
    }

    // Minimum population
    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;

    // H (memory size)
    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    pmin_ = mc.getDbl("p_min", pmin_);
    pmax_ = mc.getDbl("p_max", pmax_);
    if (pmin_ <= 0.0) pmin_ = 0.05;
    if (pmax_ <= pmin_) pmax_ = std::max(pmin_ + 0.05, 0.5);

    arc_rate_ = mc.getDbl("arc_rate", arc_rate_);
    if (arc_rate_ < 0.0) arc_rate_ = 0.0;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    epsilon_ = mc.getDbl("rl_epsilon", epsilon_);
    if (epsilon_ < 0.0) epsilon_ = 0.0;
    if (epsilon_ > 1.0) epsilon_ = 1.0;

    rl_alpha_ = mc.getDbl("rl_alpha", rl_alpha_);
    if (rl_alpha_ <= 0.0 || rl_alpha_ > 1.0) rl_alpha_ = 0.2;

    // In-run local from [mlshaderl]
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // RL weights
    strat_weight_.assign(NUM_STRAT_, 1.0);
    strat_reward_acc_.assign(NUM_STRAT_, 0.0);
    strat_use_count_.assign(NUM_STRAT_, 0);
}

void mLSHADE_RL::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    // If a per-method population exists, it is enforced again.
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    pop_init_ = std::max(4, Optimizer::population());

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

    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    updateStop(FX_);
    printBest();
}

void mLSHADE_RL::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// Strategy selection with epsilon-greedy / weighted roulette
int mLSHADE_RL::selectStrategy()
{
    if (NUM_STRAT_ <= 1) return 0;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    if (r < epsilon_) {
        std::uniform_int_distribution<int> Ui(0, NUM_STRAT_ - 1);
        return Ui(rng_);
    }

    double sumw = 0.0;
    for (double w : strat_weight_) sumw += w;
    if (sumw <= 0.0) {
        std::uniform_int_distribution<int> Ui(0, NUM_STRAT_ - 1);
        return Ui(rng_);
    }
    std::uniform_real_distribution<double> Ur(0.0, sumw);
    double x = Ur(rng_);
    double acc = 0.0;
    for (int k = 0; k < NUM_STRAT_; ++k) {
        acc += strat_weight_[k];
        if (x <= acc) return k;
    }
    return NUM_STRAT_ - 1;
}

void mLSHADE_RL::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    const int D = prob_->dimension();
    const int N = static_cast<int>(X_.size());

    // LSHADE: linear population size reduction
    double evals_now = static_cast<double>(prob_->calls());
    double maxE      = static_cast<double>(std::max<long long>(max_evals_, 1));
    int targetN = static_cast<int>(std::round(pop_init_ - (pop_init_ - pop_min_) * (evals_now / maxE)));
    if (targetN < pop_min_) targetN = pop_min_;
    if (targetN < N) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

        std::vector<Vec>    newPop;
        std::vector<double> newFit;
        newPop.reserve(targetN);
        newFit.reserve(targetN);
        for (int i = 0; i < targetN; ++i) {
            newPop.push_back(X_[idx[i]]);
            newFit.push_back(FX_[idx[i]]);
        }
        X_.swap(newPop);
        FX_.swap(newFit);
    }

    const int Ncurr = static_cast<int>(X_.size());
    if (Ncurr < 4) return;

    // p-best size (linearly from pmax_ -> pmin_)
    double eval_ratio = evals_now / maxE;
    if (eval_ratio < 0.0) eval_ratio = 0.0;
    if (eval_ratio > 1.0) eval_ratio = 1.0;

    double frac = pmin_ + (pmax_ - pmin_) * (1.0 - eval_ratio);
    if (frac < pmin_) frac = pmin_;
    if (frac > pmax_) frac = pmax_;

    int p_best_size = std::max(2, (int)std::round(frac * Ncurr));
    if (p_best_size > Ncurr) p_best_size = Ncurr;
    if (p_best_size < 1) p_best_size = 1;

    // Indices are sorted by fitness
    std::vector<int> idx_sorted(Ncurr);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui_mem(0, std::max(0, H_ - 1));

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> weights;

    std::vector<Vec>    newPop(Ncurr);
    std::vector<double> newFit(Ncurr, std::numeric_limits<double>::quiet_NaN());

    // Rewards/uses are reset
    std::fill(strat_reward_acc_.begin(), strat_reward_acc_.end(), 0.0);
    std::fill(strat_use_count_.begin(),  strat_use_count_.end(),  0);

    const size_t maxArc = (size_t)std::round(arc_rate_ * Ncurr);

    for (int i = 0; i < Ncurr; ++i) {
        if (prob_->calls() >= max_evals_) break;

        int rMem = Ui_mem(rng_);
        double muF  = MF_[rMem];
        double muCR = MCR_[rMem];

        // F is sampled from a Cauchy distribution
        double Fi;
        {
            std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
            do {
                Fi = cauchy(rng_);
            } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
        }

        // CR is sampled from a Normal distribution
        double CRi;
        {
            std::normal_distribution<double> normal(muCR, normal_std_CR_);
            CRi = normal(rng_);
            if (CRi < 0.0) CRi = 0.0;
            if (CRi > 1.0) CRi = 1.0;
        }

        // Strategy selection
        int strat = selectStrategy();
        strat_use_count_[strat]++;

        // The p-best index is selected (based on the sort)
        int p_index = idx_sorted[std::uniform_int_distribution<int>(0, p_best_size - 1)(rng_)];

        // Helpers for indices (in the population)
        auto pickDistinct = [&](int count, int avoid1, int avoid2, int avoid3)->std::vector<int> {
            std::vector<int> pool;
            pool.reserve(Ncurr);
            for (int j = 0; j < Ncurr; ++j) {
                if (j == avoid1 || j == avoid2 || j == avoid3) continue;
                pool.push_back(j);
            }
            if (pool.empty()) return {};
            if (count > (int)pool.size()) count = (int)pool.size();
            std::shuffle(pool.begin(), pool.end(), rng_);
            pool.resize(count);
            return pool;
        };

        const Vec& xi = X_[i];
        Vec vi(D), ui(D);
        bool ok_strategy = true;

        if (strat == 0) {
            // Strategy 0: current-to-pbest/1 (with archive)
            auto r = pickDistinct(2, i, p_index, -1);
            if ((int)r.size() < 2) {
                ok_strategy = false;
            } else {
                int r1 = r[0];
                int r2 = r[1];

                const Vec& xp  = X_[p_index];
                const Vec& xr1 = X_[r1];

                Vec xr2;
                if (!archive_.empty() && U01(rng_) < 0.5) {
                    std::uniform_int_distribution<int> Ui_arc(0, (int)archive_.size() - 1);
                    xr2 = archive_[Ui_arc(rng_)];
                } else {
                    xr2 = X_[r2];
                }

                std::uniform_int_distribution<int> Ui_dim(0, D - 1);
                int jrand = Ui_dim(rng_);

                for (int j = 0; j < D; ++j) {
                    if (U01(rng_) < CRi || j == jrand) {
                        vi[j] = xi[j] + Fi * (xp[j] - xi[j]) + Fi * (xr1[j] - xr2[j]);
                        ui[j] = vi[j];
                    } else {
                        ui[j] = xi[j];
                    }
                }
            }
        } else if (strat == 1) {
            // Strategy 1: rand/1/bin
            auto r = pickDistinct(3, i, -1, -1);
            if ((int)r.size() < 3) {
                ok_strategy = false;
            } else {
                int r0 = r[0];
                int r1 = r[1];
                int r2 = r[2];

                const Vec& xr0 = X_[r0];
                const Vec& xr1 = X_[r1];
                const Vec& xr2 = X_[r2];

                std::uniform_int_distribution<int> Ui_dim(0, D - 1);
                int jrand = Ui_dim(rng_);

                for (int j = 0; j < D; ++j) {
                    if (U01(rng_) < CRi || j == jrand) {
                        vi[j] = xr0[j] + Fi * (xr1[j] - xr2[j]);
                        ui[j] = vi[j];
                    } else {
                        ui[j] = xi[j];
                    }
                }
            }
        } else {
            // Strategy 2: current-to-pbest/2
            auto r = pickDistinct(4, i, p_index, -1);
            if ((int)r.size() < 4) {
                ok_strategy = false;
            } else {
                int r1 = r[0];
                int r2 = r[1];
                int r3 = r[2];
                int r4 = r[3];

                const Vec& xp  = X_[p_index];
                const Vec& xr1 = X_[r1];
                const Vec& xr2 = X_[r2];
                const Vec& xr3 = X_[r3];
                const Vec& xr4 = X_[r4];

                std::uniform_int_distribution<int> Ui_dim(0, D - 1);
                int jrand = Ui_dim(rng_);

                for (int j = 0; j < D; ++j) {
                    if (U01(rng_) < CRi || j == jrand) {
                        vi[j] = xi[j]
                              + Fi * (xp[j]  - xi[j])
                              + Fi * (xr1[j] - xr2[j])
                              + Fi * (xr3[j] - xr4[j]);
                        ui[j] = vi[j];
                    } else {
                        ui[j] = xi[j];
                    }
                }
            }
        }

        // If a valid mutation could not be produced (insufficient indices),
        // the individual is kept as-is.
        if (!ok_strategy) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
            continue;
        }

        ensureBounds(ui);
        double f_new = eval(ui);

        if (f_new <= FX_[i]) {
            newPop[i] = ui;
            newFit[i] = f_new;

            // archive
            if (maxArc > 0) {
                if (archive_.size() < maxArc) {
                    archive_.push_back(X_[i]);
                } else if (!archive_.empty()) {
                    std::uniform_int_distribution<size_t> Ui_arc(0, archive_.size() - 1);
                    archive_[Ui_arc(rng_)] = X_[i];
                }
            } else {
                archive_.clear();
            }

            double diff = FX_[i] - f_new;
            if (diff < 0.0) diff = 0.0;
            SF.push_back(Fi);
            SCR.push_back(CRi);
            weights.push_back(diff + 1e-12);

            strat_reward_acc_[strat] += diff;

            // In-run local search on the improved individual
            if (local_rate_ > 0.0 && !local_method_.empty()) {
                double ru = U01(rng_);
                if (ru < local_rate_) {
                    auto res = localSearch(local_method_, newPop[i]);
                    if (!res.first.empty() && std::isfinite(res.second) && res.second < newFit[i]) {
                        newPop[i] = std::move(res.first);
                        newFit[i] = res.second;
                    }
                }
            }

            if (f_new < best_f_) {
                best_f_ = f_new;
                best_x_ = newPop[i];
            }
        } else {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    // If any element was not updated (due to a break from max_evals_),
    // the previous value is copied for safety.
    for (int i = 0; i < Ncurr; ++i) {
        if (!std::isfinite(newFit[i])) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // MF/MCR is updated if there were successes
    if (!SF.empty()) {
        double sum_w = 0.0;
        double sum_wF = 0.0, sum_wF2 = 0.0;
        double sum_wCR = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
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

            MF_[mem_idx_]  = (1.0 - c_mem_) * MF_[mem_idx_]  + c_mem_ * meanF_Lehmer;
            MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * meanCR;

            mem_idx_++;
            if (mem_idx_ >= H_) mem_idx_ = 0;
        }
    }

    // RL weights are updated
    for (int s = 0; s < NUM_STRAT_; ++s) {
        if (strat_use_count_[s] > 0) {
            double avgReward = strat_reward_acc_[s] / (double)strat_use_count_[s];
            strat_weight_[s] =
                (1.0 - rl_alpha_) * strat_weight_[s] + rl_alpha_ * (avgReward + 1e-8);
        }
    }

    updateStop(FX_);
    printBest();
}

void mLSHADE_RL::end()
{
    if (!prob_) return;

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        if (!res.first.empty() && std::isfinite(res.second) && res.second < best_f_) {
            best_x_ = std::move(res.first);
            best_f_ = res.second;
        }

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
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
