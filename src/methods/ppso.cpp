#include "ppso.h"

#include <thread>
#include <atomic>
#include <limits>
#include <random>
#include <cstring>
#include <numeric>   // iota

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimsolution {

void PPSO::configure(const MethodConfig& mc){
    // Per-method population: if provided, updates the base IMMEDIATELY (for correct reporting).
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    islands_   = std::max(1, mc.getInt("islands", islands_));
    NR_        = std::max(1, mc.getInt("NR", NR_));
    Np_        = std::max(1, mc.getInt("Np", Np_));
    c1_        = mc.getDbl("c1", c1_);
    c2_        = mc.getDbl("c2", c2_);
    vmax_frac_ = std::max(0.0, mc.getDbl("vmax_frac", vmax_frac_));

    propagation_ = mc.getStr("propagation", propagation_);
    for (auto& ch: propagation_) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    eps_stop_ = mc.getDbl("eps_stop", eps_stop_);
    NM_       = std::max(1, mc.getInt("NM", NM_));

    use_openmp_ = mc.getBool("use_openmp", use_openmp_);
    threads_    = mc.getInt("threads", threads_);

    // GA-style in-run local
    local_rate_        = mc.getDbl("local_rate",   local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;
    local_method_      = mc.getStr("local_method", local_method_);
    inrun_on_improve_  = mc.getBool("inrun_on_improve", inrun_on_improve_);

    // Final local search.
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);

    // Stops when >= K islands have completed.
    stop_after_islands_ = mc.getInt("stop_after_islands", stop_after_islands_);

    final_population_ = -1;
    hard_stop_now_    = false;
}

void PPSO::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j]+U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

double PPSO::inertia_omega_(std::mt19937_64& rng) const {
    // w_iter = 0.5 + r/2, r in [0, 1].
    std::uniform_real_distribution<double> U01(0.0,1.0);
    return 0.5 + 0.5 * U01(rng);
}

void PPSO::init(){
    if (!prob_) return;

    // Final population: per-method or [global].
    const int global_pop = Optimizer::population();
    const int method_pop = (pop_cfg_ > 0 ? pop_cfg_ : -1);
    final_population_    = std::max(4, (method_pop > 0 ? method_pop : global_pop));

    // Updates the base for correct reporting.
    Optimizer::setPopulation(final_population_);

    islands_ = std::max(1, islands_);
    // clamp stop_after_islands
    if (stop_after_islands_ <= 0) stop_after_islands_ = islands_;
    if (stop_after_islands_ > islands_) stop_after_islands_ = islands_;

    // Distribution across islands.
    particles_per_island_ = std::max(1, final_population_ / islands_);
    int remainder = final_population_ - particles_per_island_ * islands_;

    isl_.clear(); isl_.resize(islands_);
    island_done_.assign(islands_, 0);

    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    // Position initialization (same pattern as GA).
    Initializer initSampler;
    initSampler.configure(initopt_);
    std::vector<Vec> allX = initSampler.samplePopulation(*prob_, rng_, final_population_);

    // Distribution plus initial velocities.
    int cursor = 0;
    for (int k=0; k<islands_; ++k){
        Island& S = isl_[k];
        int m_k = particles_per_island_ + (k < remainder ? 1 : 0);

        S.X.assign(m_k, Vec(D,0.0));
        S.V.assign(m_k, Vec(D,0.0));
        S.Pbest.assign(m_k, Vec(D,0.0));
        S.fX.assign(m_k, std::numeric_limits<double>::infinity());
        S.fPbest.assign(m_k, std::numeric_limits<double>::infinity());
        S.gbest_x.assign(D, 0.0);
        S.gbest_f = std::numeric_limits<double>::infinity();
        S.delta_hist.clear();

        // RNG diversification per island.
        S.rng.seed( uint64_t(0x9E3779B97F4A7C15ull) ^ (uint64_t)k ^ (uint64_t)rng_() );

        // vmax
        Vec vmax(D, 0.0);
        for (int j=0;j<D;++j) vmax[j] = vmax_frac_ * (U[j]-L[j]);
        std::uniform_real_distribution<double> Ur(-1.0, 1.0);

        for (int i=0;i<m_k; ++i, ++cursor){
            if (cursor < (int)allX.size()) S.X[i] = allX[cursor];
            else {
                // Uniform fallback.
                for (int j=0;j<D;++j){
                    std::uniform_real_distribution<double> Uj(L[j], U[j]);
                    S.X[i][j] = Uj(S.rng);
                }
            }
            ensureBounds(S.X[i]);
            for (int j=0;j<D;++j) S.V[i][j] = Ur(S.rng) * vmax[j];

            S.fX[i] = eval(S.X[i]);
            S.Pbest[i]  = S.X[i];
            S.fPbest[i] = S.fX[i];
            if (S.fX[i] < S.gbest_f){ S.gbest_f = S.fX[i]; S.gbest_x = S.X[i]; }

            if (prob_->calls() >= max_evals_) break;
        }
        if (prob_->calls() >= max_evals_) break;
    }

    // Global best
    global_best_f_ = std::numeric_limits<double>::infinity();
    global_best_x_.assign(D, 0.0);
    for (const auto& S: isl_){
        if (S.gbest_f < global_best_f_){ global_best_f_ = S.gbest_f; global_best_x_ = S.gbest_x; }
    }

    K_ = 0;
    best_f_ = global_best_f_;
    best_x_ = global_best_x_;
    hard_stop_now_ = false;

    printBest();

    // updateStop with all pbest values across all islands.
    std::vector<double> snapshot;
    snapshot.reserve(final_population_);
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fPbest.begin(), S.fPbest.end());
    updateStop(snapshot);
}

void PPSO::pso_single_iteration_(Island& S){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::uniform_real_distribution<double> U01(0.0,1.0);
    const double omega = inertia_omega_(S.rng);

    // Velocity/position update.
    for (size_t i=0;i<S.X.size(); ++i){
        const Vec& pbi = S.Pbest[i];
        const Vec& gbi = S.gbest_x;

        for (int j=0;j<D;++j){
            double r1 = U01(S.rng), r2 = U01(S.rng);
            double vel = omega * S.V[i][j]
                       + c1_ * r1 * (pbi[j] - S.X[i][j])
                       + c2_ * r2 * (gbi[j] - S.X[i][j]);

            double vmaxj = std::max(1e-16, vmax_frac_ * (U[j]-L[j]));
            if (vel >  vmaxj) vel =  vmaxj;
            if (vel < -vmaxj) vel = -vmaxj;

            S.V[i][j] = vel;
            S.X[i][j] = S.X[i][j] + vel;
        }
        ensureBounds(S.X[i]);

        // Evaluation and pbest/gbest update.
        double f = eval(S.X[i]);
        S.fX[i] = f;
        bool improved = false;
        if (f < S.fPbest[i]){ S.fPbest[i] = f; S.Pbest[i] = S.X[i]; improved = true; }
        if (f < S.gbest_f)  { S.gbest_f = f; S.gbest_x = S.X[i]; }

        // In-run local search (GA-style for logging).
        if (local_rate_ > 0.0 && !local_method_.empty()){
            bool fire = (!inrun_on_improve_) ? (U01(S.rng) < local_rate_) :
                                               (improved && U01(S.rng) < local_rate_);
            if (fire){
                auto [xloc, floc] = localSearch(local_method_, S.X[i]);
                if (floc < S.fX[i]) { S.X[i] = xloc; S.fX[i] = floc; }
                if (floc < S.fPbest[i]) { S.Pbest[i] = xloc; S.fPbest[i] = floc; }
                if (floc < S.gbest_f)   { S.gbest_x = xloc; S.gbest_f   = floc; }
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }
}

std::vector<int> PPSO::best_indices_(const Island& S, int N) const{
    std::vector<int> idx(S.fX.size()); std::iota(idx.begin(), idx.end(), 0);
    N = std::min(N, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin()+N, idx.end(),
        [&](int a,int b){ return S.fX[a] < S.fX[b]; });
    idx.resize(N);
    return idx;
}

int PPSO::worst_index_(const Island& S) const{
    int w = 0; double fw = S.fX[0];
    for (int i=1;i<(int)S.fX.size(); ++i){
        if (S.fX[i] > fw){ fw = S.fX[i]; w = i; }
    }
    return w;
}

void PPSO::send_best_Np_(int src, const std::vector<int>& dst_ids){
    if (src < 0 || src >= (int)isl_.size()) return;
    if (!island_done_.empty() && island_done_[src]) return; // Source finished -> skip.

    Island& A = isl_[src];
    const auto best_idx = best_indices_(A, Np_);
    for (int dst : dst_ids){
        if (dst<0 || dst>=(int)isl_.size() || dst==src) continue;
        if (!island_done_.empty() && island_done_[dst]) continue; // Destination finished -> skip.

        Island& B = isl_[dst];
        for (int bi : best_idx){
            int w = worst_index_(B);
            // Replaces the worst in B with a copy of the best from A.
            B.X[w]      = A.X[bi];
            B.V[w].assign(B.V[w].size(), 0.0); // reset vel
            B.fX[w]     = A.fX[bi];
            B.Pbest[w]  = A.Pbest[bi];
            B.fPbest[w] = A.fPbest[bi];
            if (B.fX[w] < B.gbest_f){ B.gbest_f = B.fX[w]; B.gbest_x = B.X[w]; }
        }
    }
}

void PPSO::propagate_(){
    if (islands_ <= 1) return;

    // 1to1, 2toN, nto1, NtoN - ignores finished islands.
    if (propagation_ == "1to1"){
        std::uniform_int_distribution<int> Uisl(0, islands_-1);
        int src = Uisl(rng_), dst = Uisl(rng_);
        int guard=0;
        while ((dst==src) || island_done_[src] || island_done_[dst]){
            if (++guard > 10*islands_) return;
            src = Uisl(rng_); dst = Uisl(rng_);
        }
        send_best_Np_(src, {dst});
    } else if (propagation_ == "2toN"){
        if (islands_ >= 2){
            std::uniform_int_distribution<int> Uisl(0, islands_-1);
            int s1 = Uisl(rng_), s2 = Uisl(rng_);
            int guard=0;
            while ((s2==s1) || island_done_[s1] || island_done_[s2]){
                if (++guard > 10*islands_) return;
                s1 = Uisl(rng_); s2 = Uisl(rng_);
            }
            std::vector<int> all; all.reserve(islands_);
            for (int k=0;k<islands_; ++k) if (!island_done_[k]) all.push_back(k);
            if (all.size() < 2) return;
            send_best_Np_(s1, all);
            send_best_Np_(s2, all);
        }
    } else if (propagation_ == "nto1"){
        std::uniform_int_distribution<int> Uisl(0, islands_-1);
        int dst = Uisl(rng_);
        int guard=0;
        while (island_done_[dst]){
            if (++guard > 10*islands_) return;
            dst = Uisl(rng_);
        }
        std::vector<int> one{dst};
        for (int s=0;s<islands_; ++s){
            if (s==dst || island_done_[s]) continue;
            send_best_Np_(s, one);
        }
    } else { // NtoN
        for (int s=0;s<islands_; ++s){
            if (island_done_[s]) continue;
            std::vector<int> all; all.reserve(islands_-1);
            for (int d=0; d<islands_; ++d) if (d!=s && !island_done_[d]) all.push_back(d);
            if (all.empty()) continue;
            send_best_Np_(s, all);
        }
    }
}

bool PPSO::stopping_hold_(Island& S, double fmin_k){
    // front = last fmin, rest = recent |delta|.
    if (S.delta_hist.empty()){
        S.delta_hist.push_back(fmin_k); // initialize
        return false;
    }
    double prev_fmin = S.delta_hist.front();
    double delta = std::fabs(fmin_k - prev_fmin);

    S.delta_hist.push_back(delta);
    if ((int)S.delta_hist.size() > (1 + NM_)) S.delta_hist.pop_front();
    S.delta_hist.front() = fmin_k;

    if ((int)S.delta_hist.size() < (1 + NM_)) return false;
    for (int i=1;i<(int)S.delta_hist.size(); ++i){
        if (S.delta_hist[i] > eps_stop_) return false;
    }
    return true;
}

void PPSO::one_iteration(){
    if (!prob_) return;

    // If hard-stop has been requested, freezes everything and feeds a stable snapshot.
    if (hard_stop_now_){
        printBest();
        std::vector<double> snapshot;
        snapshot.reserve(final_population_);
        for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fPbest.begin(), S.fPbest.end());
        updateStop(snapshot); // Same snapshot => the global stopper (e.g., bss) will terminate immediately.
        return;
    }

#ifdef _OPENMP
    int nthreads = threads_>0 ? threads_ : omp_get_max_threads();
    if (!use_openmp_) nthreads = 1;
    #pragma omp parallel for num_threads(nthreads) if(use_openmp_) schedule(static)
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]) {
            pso_single_iteration_(isl_[k]);
        }
    }
#else
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]) {
            pso_single_iteration_(isl_[k]);
        }
    }
#endif

    // Global best update.
    for (const auto& S: isl_){
        if (S.gbest_f < global_best_f_){ global_best_f_ = S.gbest_f; global_best_x_ = S.gbest_x; }
    }
    if (global_best_f_ < best_f_) { best_f_ = global_best_f_; best_x_ = global_best_x_; }

    // Propagation every NR steps (with active islands).
    if (K_ > 0 && (K_ % NR_ == 0)){
        propagate_();
    }

    // Per-island stopping: freezes those that completed.
    int done_now = 0;
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]) {
            if (stopping_hold_(isl_[k], isl_[k].gbest_f)) {
                island_done_[k] = 1;
            }
        }
        if (island_done_[k]) ++done_now;
    }

    // If the target is reached (K islands), freezes everything and requests immediate global stop.
    if (done_now >= stop_after_islands_) {
        hard_stop_now_ = true;
        // Freezes all islands so no additional work is performed.
        std::fill(island_done_.begin(), island_done_.end(), (unsigned char)1);
    }

    ++K_;

    // Reporting/stop history.
    printBest();
    std::vector<double> snapshot;
    snapshot.reserve(final_population_);
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fPbest.begin(), S.fPbest.end());
    updateStop(snapshot);
}

void PPSO::end(){
    // Final local search (optional).
    if (end_local_refine_ && !end_local_method_.empty()){
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (floc < best_f_){ best_f_ = floc; best_x_ = xloc; }
    }

    // Optional injection of the best into the worst individual of the worst island.
    if (!isl_.empty()){
        int worst_island = 0; double fw = isl_[0].gbest_f;
        for (int k=1;k<islands_; ++k){
            if (isl_[k].gbest_f > fw){ fw = isl_[k].gbest_f; worst_island = k; }
        }
        Island& W = isl_[worst_island];
        int wi = worst_index_(W);
        if (wi>=0 && wi<(int)W.X.size()){
            W.X[wi]      = best_x_;
            W.fX[wi]     = best_f_;
            W.Pbest[wi]  = best_x_;
            W.fPbest[wi] = best_f_;
            W.gbest_x    = best_x_;
            W.gbest_f    = best_f_;
        }
    }

    std::vector<double> snapshot;
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fPbest.begin(), S.fPbest.end());
    updateStop(snapshot);
    printBest();
}

} // namespace optimsolution
