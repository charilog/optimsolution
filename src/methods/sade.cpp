#include "sade.h"
#include "init.h"
#include "options.h"

#include <numeric>   // iota

namespace optimsolution {

void SaDE::configure(const MethodConfig& mc)
{
    // Per-method population; if provided, updates the base.
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    // F handling.
    F_min_ = mc.getDbl("F_min", F_min_);
    F_max_ = mc.getDbl("F_max", F_max_);
    if (F_min_ < 0.0) F_min_ = 0.0;
    if (F_max_ <= F_min_) F_max_ = F_min_ + 0.5;
    if (F_max_ > 1.5) F_max_ = 1.5;

    // Learning period
    Lp_ = mc.getInt("Lp", Lp_);
    if (Lp_ <= 0) Lp_ = 50;

    // Initial CR values per strategy (optionally from cfg).
    CR_init_[0] = mc.getDbl("CR1", CR_init_[0]); // e.g., for DE/rand/1.
    CR_init_[1] = mc.getDbl("CR2", CR_init_[1]); // e.g., for current-to-best/2.
    CR_init_[2] = mc.getDbl("CR3", CR_init_[2]); // e.g., for rand/2.
    CR_init_[3] = mc.getDbl("CR4", CR_init_[3]); // e.g., for current-to-rand/1.
    for (int k = 0; k < NUM_STRAT_; ++k) {
        if (CR_init_[k] < 0.0) CR_init_[k] = 0.0;
        if (CR_init_[k] > 1.0) CR_init_[k] = 1.0;
    }

    // In-run local search
    local_rate_   = mc.getDbl("local_rate", local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;
    local_method_ = mc.getStr("local_method", local_method_);

    // Final local
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

void SaDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    // If a per-method population is provided, it is applied.
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    pop_init_ = std::max(6, Optimizer::population()); // rand/2 requires >= 6.
    setPopulation(pop_init_);

    X_.clear();
    FX_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    int N = static_cast<int>(X_.size());

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N; ++i) {
        if ((int)X_[i].size() != D) {
            X_[i].assign(D, 0.0);
        }
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    // Strategy adaptation initialization.
    strat_prob_.assign(NUM_STRAT_, 1.0 / NUM_STRAT_);
    strat_success_.assign(NUM_STRAT_, 0);
    strat_attempt_.assign(NUM_STRAT_, 0);
    gen_counter_ = 0;

    CR_mean_.resize(NUM_STRAT_);
    CR_success_pool_.resize(NUM_STRAT_);
    for (int s = 0; s < NUM_STRAT_; ++s) {
        CR_mean_[s] = CR_init_[s];
        CR_success_pool_[s].clear();
    }

    updateStop(FX_);
    printBest();
}

void SaDE::ensureBounds(Vec& x)
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

int SaDE::selectStrategy(std::uniform_real_distribution<double>& U01)
{
    if (NUM_STRAT_ <= 1) return 0;
    double r   = U01(rng_);
    double cum = 0.0;
    for (int s = 0; s < NUM_STRAT_; ++s) {
        double ps = (s < (int)strat_prob_.size()) ? strat_prob_[s] : 0.0;
        cum += ps;
        if (r <= cum) return s;
    }
    return NUM_STRAT_ - 1;
}

// strat 0..3
bool SaDE::generateTrial(int strat, int i, double Fi, double CRi,
                         int best_idx, Vec& ui,
                         std::uniform_real_distribution<double>& U01,
                         std::uniform_int_distribution<int>& Ui)
{
    const int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();

    if (N < 4 || i < 0 || i >= N) {
        ui = (i >= 0 && i < N) ? X_[i] : Vec(D, 0.0);
        return false;
    }

    // Helper for unique indices.
    auto randDistinct = [&](int avoid1, int avoid2, int avoid3, int avoid4, int avoid5) {
        int r;
        do {
            r = Ui(rng_);
        } while (r == avoid1 || r == avoid2 || r == avoid3 || r == avoid4 || r == avoid5);
        return r;
    };

    Vec v(D);

    if (strat == 0) {
        // DE/rand/1
        int r1 = randDistinct(i, -1, -1, -1, -1);
        int r2 = randDistinct(i, r1, -1, -1, -1);
        int r3 = randDistinct(i, r1, r2, -1, -1);
        for (int j = 0; j < D; ++j) {
            v[j] = X_[r1][j] + Fi * (X_[r2][j] - X_[r3][j]);
        }
    } else if (strat == 1) {
        // DE/current-to-best/2
        if (best_idx < 0 || best_idx >= N) return false;
        int r1 = randDistinct(i, best_idx, -1, -1, -1);
        int r2 = randDistinct(i, best_idx, r1, -1, -1);
        for (int j = 0; j < D; ++j) {
            v[j] = X_[i][j]
                 + Fi * (X_[best_idx][j] - X_[i][j])
                 + Fi * (X_[r1][j] - X_[r2][j]);
        }
    } else if (strat == 2) {
        // DE/rand/2
        if (N < 5) return false;
        int r1 = randDistinct(i, -1, -1, -1, -1);
        int r2 = randDistinct(i, r1, -1, -1, -1);
        int r3 = randDistinct(i, r1, r2, -1, -1);
        int r4 = randDistinct(i, r1, r2, r3, -1);
        for (int j = 0; j < D; ++j) {
            v[j] = X_[r1][j]
                 + Fi * (X_[r2][j] - X_[r3][j])
                 + Fi * (X_[r4][j] - X_[randDistinct(i, r1, r2, r3, r4)][j]);
        }
    } else {
        // strat == 3: DE/current-to-rand/1
        int r1 = randDistinct(i, -1, -1, -1, -1);
        int r2 = randDistinct(i, r1, -1, -1, -1);
        int r3 = randDistinct(i, r1, r2, -1, -1);
        for (int j = 0; j < D; ++j) {
            v[j] = X_[i][j]
                 + Fi * (X_[r1][j] - X_[i][j])
                 + Fi * (X_[r2][j] - X_[r3][j]);
        }
    }

    // Binomial crossover
    ui.resize(D);
    std::uniform_int_distribution<int> UiD(0, D - 1);
    int jrand = UiD(rng_);
    for (int j = 0; j < D; ++j) {
        double r = U01(rng_);
        if (r <= CRi || j == jrand) ui[j] = v[j];
        else                        ui[j] = X_[i][j];
    }

    return true;
}

void SaDE::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    const int D = prob_->dimension();
    int       N = static_cast<int>(X_.size());
    if (N < 4 || D <= 0) return;

    if ((int)FX_.size() != N)
        FX_.resize(N, std::numeric_limits<double>::infinity());

    // Comment translated from Greek.
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int i = 0; i < N; ++i) {
        if ((int)X_[i].size() != D) {
            X_[i].assign(D, 0.0);
            for (int j = 0; j < D; ++j) {
                X_[i][j] = 0.5 * (L[j] + U[j]);
            }
        }
    }

    // Finds the best index.
    int    best_idx = 0;
    double best_val = FX_[0];
    for (int i = 1; i < N; ++i) {
        if (FX_[i] < best_val) {
            best_val = FX_[i];
            best_idx = i;
        }
    }

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui(0, N - 1);

    std::vector<Vec>    newPop(N);
    std::vector<double> newFit(N, std::numeric_limits<double>::quiet_NaN());

    if ((int)strat_prob_.size() != NUM_STRAT_) {
        strat_prob_.assign(NUM_STRAT_, 1.0 / NUM_STRAT_);
        strat_success_.assign(NUM_STRAT_, 0);
        strat_attempt_.assign(NUM_STRAT_, 0);
        CR_mean_.assign(NUM_STRAT_, 0.5);
        CR_success_pool_.assign(NUM_STRAT_, std::vector<double>());
    }

    gen_counter_++;

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        // Comment translated from Greek.
        int strat = selectStrategy(U01);
        if (strat < 0 || strat >= NUM_STRAT_) strat = 0;

        // Comment translated from Greek.
        double Fi = F_min_ + U01(rng_) * (F_max_ - F_min_);

        // Comment translated from Greek.
        double CRi;
        {
            double mu = (strat < (int)CR_mean_.size()) ? CR_mean_[strat] : 0.5;
            std::normal_distribution<double> normal(mu, 0.1);
            CRi = normal(rng_);
            if (CRi < 0.0) CRi = 0.0;
            if (CRi > 1.0) CRi = 1.0;
        }

        Vec ui;
        bool ok = generateTrial(strat, i, Fi, CRi, best_idx, ui, U01, Ui);
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

        strat_attempt_[strat]++;

        if (f_new <= FX_[i]) {
            newPop[i] = ui;
            newFit[i] = f_new;
            strat_success_[strat]++;

            if (strat < (int)CR_success_pool_.size()) {
                CR_success_pool_[strat].push_back(CRi);
            }

            if (f_new < best_f_) {
                best_f_ = f_new;
                best_x_ = ui;
            }
        } else {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(newFit[i])) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // --- Strategy & CR adaptation every Lp_ generations ---
    if (Lp_ > 0 && gen_counter_ >= Lp_) {
        // Probabilities for strategies.
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

        // CR means from pools of successful trials.
        for (int s = 0; s < NUM_STRAT_; ++s) {
            const auto& pool = CR_success_pool_[s];
            if (!pool.empty()) {
                double sumCR = 0.0;
                for (double c : pool) sumCR += c;
                CR_mean_[s] = sumCR / pool.size();
                if (CR_mean_[s] < 0.0) CR_mean_[s] = 0.0;
                if (CR_mean_[s] > 1.0) CR_mean_[s] = 1.0;
            }
        }

        // reset counters/pools
        std::fill(strat_success_.begin(), strat_success_.end(), 0);
        std::fill(strat_attempt_.begin(), strat_attempt_.end(), 0);
        for (int s = 0; s < NUM_STRAT_; ++s) {
            CR_success_pool_[s].clear();
        }
        gen_counter_ = 0;
    }

    // Comment translated from Greek.
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
                    X_[i]  = xloc;
                    FX_[i] = floc;
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

void SaDE::end()
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
