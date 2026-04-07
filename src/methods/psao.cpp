#include "psao.h"
#include "init.h"

#include <algorithm>
#include <numeric>
#include <thread>
#include <cctype>
#include <cmath>

namespace optimsolution {

static inline std::string tolower_str(std::string s){
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// -------------------------------------------------
// configure — pattern as in SAO/GA (without end_local_* here)
// end_local_* are populated by setEndLocalFromGlobal([global]).
// -------------------------------------------------
void PSAO::configure(const MethodConfig& mc){
    // Per-method population (override of global for correct reporting)
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) Optimizer::setPopulation(pop_cfg_);

    // In–run locals
    local_method_      = mc.getStr("local_method",  local_method_);
    local_rate_        = mc.getDbl("local_rate",    local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    // SAO params
    sniff_w_           = mc.getDbl("sniff_w",   sniff_w_);
    sniff_a1_          = mc.getDbl("sniff_a1",  sniff_a1_);
    sniff_a2_          = mc.getDbl("sniff_a2",  sniff_a2_);

    trail_sigma0_      = mc.getDbl("trail_sigma0", trail_sigma0_);
    trail_decay_       = mc.getDbl("trail_decay",  trail_decay_);

    rand_rate_         = mc.getDbl("rand_rate",  rand_rate_);
    rand_scale_        = mc.getDbl("rand_scale", rand_scale_);

    // Islands / threads / use_openmp
    islands_           = mc.getInt("islands", islands_);
    if (islands_ < 1) islands_ = 1;

    threads_           = mc.getInt("threads", threads_);
    use_openmp_        = mc.getBool("use_openmp", use_openmp_);

    // Migration
    NR_                = mc.getInt("NR", NR_);
    if (NR_ < 0) NR_ = 0;    // 0 => no migration

    Np_                = mc.getInt("Np", Np_);
    if (Np_ < 0) Np_ = 0;

    stop_after_islands_ = mc.getInt("stop_after_islands", stop_after_islands_);
    if (stop_after_islands_ <= 0) stop_after_islands_ = islands_;

    // Propagation (supports "1toN", "NtoN", etc.)
    {
        std::string p = tolower_str( mc.getStr("propagation", "1to1") );
        if (p == "1to1" || p == "ring" || p == "ring1to1"){
            propagation_ = Propagation::Ring1to1;
        } else if (p == "1ton" || p == "one2n" || p == "one_to_n" || p == "oneton"){
            propagation_ = Propagation::OneToN;
        } else if (p == "nton" || p == "n2n" || p == "n_to_n" ||
                   p == "all"  || p == "alltoall" || p == "all_to_all"){
            propagation_ = Propagation::AllToAll;
        } else {
            propagation_ = Propagation::Ring1to1;
        }
    }

    // BSS per-island
    bss_eps_           = mc.getDbl("eps", bss_eps_);
    bss_sim_           = mc.getInt("sim", bss_sim_);
    if (bss_sim_ < 1) bss_sim_ = 1;
}

// -------------------------------------------------
// init
// -------------------------------------------------
void PSAO::init(){
    if (!prob_) return;

    if (pop_cfg_ > 0) Optimizer::setPopulation(pop_cfg_);
    final_population_ = std::max(4, Optimizer::population());

    isl_.clear();
    isl_.resize(islands_);

    int base = final_population_ / islands_;
    int rem  = final_population_ % islands_;

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.clear();

    for (int k = 0; k < islands_; ++k){
        int sz = base + (k < rem ? 1 : 0);
        if (sz <= 0) sz = 1;
        init_island_(isl_[k], sz);
    }

    islands_stopped_.store(0);
    globally_stopped_ = false;
    K_ = 0;

    std::vector<double> allf(final_population_);
    collect_snapshot_(allf);
    updateStop(allf);
    printBest();
}

// -------------------------------------------------
// init_island_
// -------------------------------------------------
void PSAO::init_island_(Island& I, int subpop){
    const int D = prob_->dimension();

    Initializer initSampler;
    initSampler.configure(initopt_);
    I.X = initSampler.samplePopulation(*prob_, rng_, subpop);

    I.V.assign(subpop, Vec(D, 0.0));
    I.fX.assign(subpop, std::numeric_limits<double>::infinity());

    I.best_f        = std::numeric_limits<double>::infinity();
    I.worst_f       = -std::numeric_limits<double>::infinity();
    I.trail_sigma_k = trail_sigma0_;

    I.stopped         = false;
    I.same_best_iters = 0;
    I.last_best_f     = std::numeric_limits<double>::infinity();
    I.size            = subpop;

    I.rng_local.seed( 0xC3A5C85CUL ^ (uint64_t)rng_() );

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < subpop; ++i){
        auto& x = I.X[i];
        if ((int)x.size() < D) x.resize(D, 0.0);

        for (int j = 0; j < D; ++j){
            if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
            if (x[j] < L[j]) x[j] = L[j];
            if (x[j] > U[j]) x[j] = U[j];
        }

        I.fX[i] = eval(I.X[i]);

        if (I.fX[i] < I.best_f){ I.best_f = I.fX[i]; I.best_x = I.X[i]; }
        if (I.fX[i] > I.worst_f){ I.worst_f = I.fX[i]; I.worst_x = I.X[i]; }

        if (I.best_f < best_f_){ best_f_ = I.best_f; best_x_ = I.best_x; }

        if (prob_->calls() >= max_evals_) break;
    }
}

// -------------------------------------------------
// helpers
// -------------------------------------------------
inline void PSAO::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j] + U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

void PSAO::island_eval_and_update_(Island& I, int i){
    double f = eval(I.X[i]);
    I.fX[i] = f;
    if (f < I.best_f){ I.best_f = f; I.best_x = I.X[i]; }
    if (f > I.worst_f){ I.worst_f = f; I.worst_x = I.X[i]; }
    if (I.best_f < best_f_){ best_f_ = I.best_f; best_x_ = I.best_x; }
}

void PSAO::island_recompute_worst_(Island& I){
    I.worst_f = -std::numeric_limits<double>::infinity();
    if (I.fX.empty()) return;
    for (int i = 0; i < (int)I.fX.size(); ++i){
        if (I.fX[i] > I.worst_f){
            I.worst_f = I.fX[i];
            I.worst_x = I.X[i];
        }
    }
}

// -------------------------------------------------
// island_sniffing_
// -------------------------------------------------
void PSAO::island_sniffing_(Island& I){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < (int)I.X.size(); ++i){
        if (I.stopped) break;
        for (int j = 0; j < D; ++j){
            double r1 = I.U01(I.rng_local);
            double r2 = I.U01(I.rng_local);

            I.V[i][j] = sniff_w_ * I.V[i][j]
                      + sniff_a1_ * r1 * (I.best_x[j]  - I.X[i][j])
                      - sniff_a2_ * r2 * (I.X[i][j]    - I.worst_x[j]);

            I.X[i][j] += I.V[i][j];
            if (I.X[i][j] < L[j]) I.X[i][j] = L[j];
            if (I.X[i][j] > U[j]) I.X[i][j] = U[j];
        }
        island_eval_and_update_(I, i);
        if (prob_->calls() >= max_evals_) return;
    }
}

// -------------------------------------------------
// island_trailing_
// -------------------------------------------------
bool PSAO::island_trailing_(Island& I){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    bool improved = false;

    for (int i = 0; i < (int)I.X.size(); ++i){
        if (I.stopped) break;

        Vec xi = I.X[i];
        for (int j = 0; j < D; ++j){
            double scale = I.trail_sigma_k * (std::abs(I.best_x[j] - xi[j]) + 1e-16);
            double step  = scale * I.N01(I.rng_local);
            xi[j] += step;
            if (xi[j] < L[j]) xi[j] = L[j];
            if (xi[j] > U[j]) xi[j] = U[j];
        }

        double f = eval(xi);
        if (f <= I.fX[i]){
            I.X[i]  = std::move(xi);
            I.fX[i] = f;

            if (f < I.best_f){ I.best_f = f; I.best_x = I.X[i]; improved = true; }
            if (I.best_f < best_f_){ best_f_ = I.best_f; best_x_ = I.best_x; }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    I.trail_sigma_k = std::max(1e-12, I.trail_sigma_k * trail_decay_);
    island_recompute_worst_(I);
    return improved;
}

// -------------------------------------------------
// island_random_
// -------------------------------------------------
void PSAO::island_random_(Island& I){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < (int)I.X.size(); ++i){
        if (I.stopped) break;

        Vec xi = I.X[i];
        for (int j = 0; j < D; ++j){
            if (I.U01(I.rng_local) < rand_rate_){
                double range = (U[j] - L[j]);
                double step  = (I.U01(I.rng_local) - 0.5) * 2.0 * rand_scale_ * range;
                xi[j] += step;
                if (xi[j] < L[j]) xi[j] = L[j];
                if (xi[j] > U[j]) xi[j] = U[j];
            }
        }

        double f = eval(xi);
        if (f <= I.fX[i]){
            I.X[i]  = std::move(xi);
            I.fX[i] = f;

            if (f < I.best_f){ I.best_f = f; I.best_x = I.X[i]; }
            if (I.best_f < best_f_){ best_f_ = I.best_f; best_x_ = I.best_x; }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    island_recompute_worst_(I);
}

// -------------------------------------------------
// island_update_bss_
// -------------------------------------------------
void PSAO::island_update_bss_(Island& I){
    if (I.stopped) return;

    if (!std::isfinite(I.last_best_f)){
        I.last_best_f     = I.best_f;
        I.same_best_iters = 1;
        return;
    }

    double diff = std::fabs(I.best_f - I.last_best_f);
    if (diff <= bss_eps_){
        ++I.same_best_iters;
    } else {
        I.last_best_f     = I.best_f;
        I.same_best_iters = 1;
    }

    if (I.same_best_iters >= bss_sim_){
        I.stopped = true;
        islands_stopped_.fetch_add(1);
    }
}

// -------------------------------------------------
// parallel_for_islands_
// -------------------------------------------------
void PSAO::parallel_for_islands_(const std::function<void(int)>& fn){
    if (!use_openmp_ || islands_ <= 1){
        for (int k = 0; k < islands_; ++k) fn(k);
        return;
    }

    int T = threads_;
    if (T <= 0){
        T = (int)std::thread::hardware_concurrency();
        if (T <= 0) T = islands_;
    }

    std::vector<std::thread> pool;
    pool.reserve(islands_);

    for (int k = 0; k < islands_; ++k){
        pool.emplace_back([&, k]{ fn(k); });
    }
    for (auto &th : pool) th.join();
}

// -------------------------------------------------
// migration helpers
// -------------------------------------------------
void PSAO::top_indices_(const std::vector<double>& fX, int take, std::vector<int>& out){
    out.resize(fX.size());
    std::iota(out.begin(), out.end(), 0);

    int t = std::max(0, std::min(take, (int)fX.size()));
    if (t == 0){
        out.clear();
        return;
    }

    std::partial_sort(out.begin(), out.begin() + t, out.end(),
                      [&](int a, int b){ return fX[a] < fX[b]; });
    out.resize(t);
}

void PSAO::replace_worst_with_group_(Island& D, const std::vector<Vec>& srcX, const std::vector<double>& srcF){
    if (D.X.empty() || srcX.empty()) return;

    int m = (int)std::min(srcX.size(), D.X.size());
    for (int w = 0; w < m; ++w){
        int    worst = 0;
        double fw    = D.fX[0];
        for (int i = 1; i < (int)D.fX.size(); ++i){
            if (D.fX[i] > fw){ fw = D.fX[i]; worst = i; }
        }

        D.X[worst]  = srcX[w];
        D.fX[worst] = srcF[w];

        if (D.fX[worst] < D.best_f){
            D.best_f = D.fX[worst];
            D.best_x = D.X[worst];
        }
    }
    island_recompute_worst_(D);
}

// ----- Ring 1->1 -----
void PSAO::migrate_ring_1to1_topN_(){
    if (islands_ <= 1 || Np_ <= 0) return;

    for (int k = 0; k < islands_; ++k){
        int   dst = (k + 1) % islands_;
        auto& S   = isl_[k];
        auto& D   = isl_[dst];

        if (S.stopped || D.stopped) continue;
        if (S.X.empty() || D.X.empty()) continue;

        std::vector<int> topIdx;
        top_indices_(S.fX, Np_, topIdx);

        std::vector<Vec>    srcX; srcX.reserve(topIdx.size());
        std::vector<double> srcF; srcF.reserve(topIdx.size());
        for (int id : topIdx){ srcX.push_back(S.X[id]); srcF.push_back(S.fX[id]); }

        replace_worst_with_group_(D, srcX, srcF);
    }

    for (int k = 0; k < islands_; ++k){
        if (isl_[k].best_f < best_f_){
            best_f_ = isl_[k].best_f;
            best_x_ = isl_[k].best_x;
        }
    }
}

// ----- Best island -> all (1->N) -----
void PSAO::migrate_one_to_N_topN_(){
    if (islands_ <= 1 || Np_ <= 0) return;

    int    src = -1;
    double bf  = std::numeric_limits<double>::infinity();
    for (int k = 0; k < islands_; ++k){
        if (isl_[k].best_f < bf){
            bf  = isl_[k].best_f;
            src = k;
        }
    }
    if (src < 0) return;

    auto& S = isl_[src];
    if (S.X.empty()) return;

    std::vector<int> topIdx;
    top_indices_(S.fX, Np_, topIdx);

    std::vector<Vec>    srcX; srcX.reserve(topIdx.size());
    std::vector<double> srcF; srcF.reserve(topIdx.size());
    for (int id : topIdx){ srcX.push_back(S.X[id]); srcF.push_back(S.fX[id]); }

    for (int k = 0; k < islands_; ++k){
        if (k == src) continue;
        auto& D = isl_[k];
        if (D.stopped || D.X.empty()) continue;
        replace_worst_with_group_(D, srcX, srcF);
    }

    for (int k = 0; k < islands_; ++k){
        if (isl_[k].best_f < best_f_){
            best_f_ = isl_[k].best_f;
            best_x_ = isl_[k].best_x;
        }
    }
}

// ----- AllToAll (NtoN) -----
void PSAO::migrate_all_to_all_topN_(){
    if (islands_ <= 1 || Np_ <= 0) return;

    std::vector<std::vector<Vec>>    topX(islands_);
    std::vector<std::vector<double>> topF(islands_);

    for (int k = 0; k < islands_; ++k){
        auto& S = isl_[k];
        if (S.X.empty()) continue;

        std::vector<int> topIdx;
        top_indices_(S.fX, Np_, topIdx);

        auto& sx = topX[k];
        auto& sf = topF[k];
        sx.reserve(topIdx.size());
        sf.reserve(topIdx.size());
        for (int id : topIdx){ sx.push_back(S.X[id]); sf.push_back(S.fX[id]); }
    }

    for (int src = 0; src < islands_; ++src){
        auto& sx = topX[src];
        auto& sf = topF[src];
        if (sx.empty()) continue;

        for (int dst = 0; dst < islands_; ++dst){
            if (dst == src) continue;
            auto& D = isl_[dst];
            if (D.stopped || D.X.empty()) continue;
            replace_worst_with_group_(D, sx, sf);
        }
    }

    for (int k = 0; k < islands_; ++k){
        if (isl_[k].best_f < best_f_){
            best_f_ = isl_[k].best_f;
            best_x_ = isl_[k].best_x;
        }
    }
}

// -------------------------------------------------
// collect_snapshot_
// -------------------------------------------------
void PSAO::collect_snapshot_(std::vector<double>& allf){
    int ptr = 0;
    for (auto& I : isl_){
        for (double v : I.fX){
            if (ptr < (int)allf.size()) allf[ptr++] = v;
            else allf.push_back(v);
        }
    }
    allf.resize(ptr);
}

// -------------------------------------------------
// one_iteration  (Sniff -> Trail -> Rand -> in-run local)
// -------------------------------------------------
void PSAO::one_iteration(){
    if (!prob_ || globally_stopped_) return;

    if (islands_stopped_.load() >= stop_after_islands_){
        globally_stopped_ = true;
        max_evals_ = prob_->calls();
        return;
    }

    // 1) Sniffing
    parallel_for_islands_([&](int k){
        auto& I = isl_[k];
        if (!I.stopped) island_sniffing_(I);
    });
    if (globally_stopped_ || prob_->calls() >= max_evals_) return;

    // 2) Trailing
    parallel_for_islands_([&](int k){
        auto& I = isl_[k];
        if (!I.stopped) island_trailing_(I);
    });
    if (globally_stopped_ || prob_->calls() >= max_evals_) return;

    // 3) Random
    parallel_for_islands_([&](int k){
        auto& I = isl_[k];
        if (!I.stopped) island_random_(I);
    });
    if (globally_stopped_ || prob_->calls() >= max_evals_) return;

    // 4) In-run local search (serial, per island, on the island best)
    if (local_rate_ > 0.0 && !local_method_.empty() && prob_->calls() < max_evals_){
        for (auto& I : isl_){
            if (I.stopped) continue;
            if (I.best_x.empty() || !std::isfinite(I.best_f)) continue;

            double u = I.U01(I.rng_local);
            if (u > local_rate_) continue;

            auto [xloc, floc] = localSearch(local_method_, I.best_x);
            if (!xloc.empty() && std::isfinite(floc) && floc < I.best_f){
                I.best_x = std::move(xloc);
                I.best_f = floc;
                if (floc < best_f_){
                    best_f_ = floc;
                    best_x_ = I.best_x;
                }
            }
            if (prob_->calls() >= max_evals_) break;
        }
    }

    // 5) BSS per–island
    parallel_for_islands_([&](int k){
        island_update_bss_(isl_[k]);
    });

    if (islands_stopped_.load() >= stop_after_islands_){
        globally_stopped_ = true;
        max_evals_ = prob_->calls();
        return;
    }

    // 6) Migration
    ++K_;
    if (NR_ > 0 && (K_ % NR_) == 0){
        switch (propagation_){
            case Propagation::Ring1to1: migrate_ring_1to1_topN_();  break;
            case Propagation::OneToN:   migrate_one_to_N_topN_();   break;
            case Propagation::AllToAll: migrate_all_to_all_topN_(); break;
        }
    }

    // 7) Global stop & print
    std::vector<double> allf(final_population_);
    collect_snapshot_(allf);
    updateStop(allf);
    printBest();
}

// -------------------------------------------------
// end — final local search, as in GA/SAO, on the global best
// -------------------------------------------------
void PSAO::end(){
    if (!prob_) return;

    // Use the method end_local_* that came from [global]
    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()){
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (!xloc.empty() && std::isfinite(floc) && floc < best_f_){
            best_x_ = std::move(xloc);
            best_f_ = floc;
        }

        // Replace the worst point with best for consistency
        Island* worstIsl = nullptr;
        int     worstIdx = -1;
        double  fw       = -std::numeric_limits<double>::infinity();

        for (auto& I : isl_){
            for (int i = 0; i < (int)I.fX.size(); ++i){
                if (I.fX[i] > fw){
                    fw       = I.fX[i];
                    worstIsl = &I;
                    worstIdx = i;
                }
            }
        }
        if (worstIsl && worstIdx >= 0){
            worstIsl->X[worstIdx]  = best_x_;
            worstIsl->fX[worstIdx] = best_f_;
        }

        // One last print after polishing
        printBest();
    }

    std::vector<double> allf(final_population_);
    collect_snapshot_(allf);
    updateStop(allf);
}

} // namespace optimsolution
