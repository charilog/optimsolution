#include "pde.h"

#include <thread>
#include <atomic>
#include <limits>
#include <random>
#include <cstring>
#include <numeric>   // iota
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimsolution {

// ------------------ configure ------------------
void PDE::configure(const MethodConfig& mc){
    // Per-method population: update the base immediately for correct reporting.
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    islands_   = std::max(1, mc.getInt("islands", islands_));
    NR_        = std::max(1, mc.getInt("NR", NR_));
    Np_        = std::max(1, mc.getInt("Np", Np_));
    propagation_ = mc.getStr("propagation", propagation_);
    for (auto& ch: propagation_) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    eps_stop_ = mc.getDbl("eps_stop", eps_stop_);
    NM_       = std::max(1, mc.getInt("NM", NM_));

    use_openmp_ = mc.getBool("use_openmp", use_openmp_);
    threads_    = mc.getInt("threads", threads_);

    // In-run local (GA-style)
    local_rate_        = mc.getDbl("local_rate",   local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;
    local_method_      = mc.getStr("local_method", local_method_);
    inrun_on_improve_  = mc.getBool("inrun_on_improve", inrun_on_improve_);

    // Final local search
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);

    // Global stop by number of islands
    stop_after_islands_ = mc.getInt("stop_after_islands", stop_after_islands_);

    // DE params
    F_        = mc.getDbl("F", F_);
    CR_       = mc.getDbl("CR", CR_);
    strategy_ = mc.getStr("strategy", strategy_);
    for (auto& ch: strategy_) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    p_top_    = mc.getDbl("p_top", p_top_);
    jitter_   = mc.getDbl("jitter", jitter_);
    if (p_top_ <= 0.0) p_top_ = 0.2;
    if (p_top_ > 1.0)  p_top_ = 1.0;
    jitter_ = std::max(0.0, jitter_);

    // NEW: per-island complementary criteria
    island_plateau_iters_ = mc.getInt("island_plateau_iters", island_plateau_iters_);
    island_target_f_      = mc.getDbl("island_target_f", island_target_f_);

    final_population_ = -1;
    hard_stop_now_    = false;
}

// ------------------ utilities ------------------
void PDE::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j]+U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

int PDE::pick_random_excluding_(int n, std::mt19937_64& rng, int ex) const {
    // pick ∈ [0,n-1]\{ex}
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> U(0, n-1);
    int r = U(rng);
    while (r == ex) r = U(rng);
    return r;
}

void PDE::pick_3_distinct_(int n, std::mt19937_64& rng, int i, int& r1, int& r2, int& r3) const {
    // r1,r2,r3 all distinct and != i
    if (n < 4) { r1=0; r2=1; r3=2; if (i==0) r1=3% n; if (i==1) r2=3% n; if (i==2) r3=3% n; return; }
    std::uniform_int_distribution<int> U(0, n-1);
    do { r1 = U(rng); } while (r1==i);
    do { r2 = U(rng); } while (r2==i || r2==r1);
    do { r3 = U(rng); } while (r3==i || r3==r1 || r3==r2);
}

double PDE::F_tuned_(std::mt19937_64& rng) const {
    if (jitter_ <= 0.0) return F_;
    std::uniform_real_distribution<double> Uj(-jitter_, jitter_);
    return F_ * (1.0 + Uj(rng));
}

// ------------------ init ------------------
void PDE::init(){
    if (!prob_) return;

    // Final population
    const int global_pop = Optimizer::population();
    const int method_pop = (pop_cfg_ > 0 ? pop_cfg_ : -1);
    final_population_    = std::max(4, (method_pop > 0 ? method_pop : global_pop));

    // update the base for correct reporting
    Optimizer::setPopulation(final_population_);

    islands_ = std::max(1, islands_);
    if (stop_after_islands_ <= 0) stop_after_islands_ = islands_;
    if (stop_after_islands_ > islands_) stop_after_islands_ = islands_;

    // Distribution across islands
    particles_per_island_ = std::max(1, final_population_ / islands_);
    int remainder = final_population_ - particles_per_island_ * islands_;

    isl_.clear(); isl_.resize(islands_);
    island_done_.assign(islands_, 0);
    last_improve_iter_.assign(islands_, 0);

    const int D = prob_->dimension();

    // Position initialization (as in GA/ppso)
    Initializer initSampler;
    initSampler.configure(initopt_);
    std::vector<Vec> allX = initSampler.samplePopulation(*prob_, rng_, final_population_);

    int cursor = 0;
    for (int k=0; k<islands_; ++k){
        Island& S = isl_[k];
        int m_k = particles_per_island_ + (k < remainder ? 1 : 0);

        S.X.assign(m_k, Vec(D,0.0));
        S.Trial.assign(m_k, Vec(D,0.0));
        S.fX.assign(m_k, std::numeric_limits<double>::infinity());
        S.gbest_x.assign(D, 0.0);
        S.gbest_f = std::numeric_limits<double>::infinity();
        S.delta_hist.clear();

        // differentiate RNG per island
        S.rng.seed( uint64_t(0xA0761D6478BD642FULL) ^ (uint64_t)k ^ (uint64_t)rng_() );

        for (int i=0;i<m_k; ++i, ++cursor){
            if (cursor < (int)allX.size()) S.X[i] = allX[cursor];
            else {
                // safety: if samples are missing for any reason
                const auto& L = prob_->lb(); const auto& U = prob_->ub();
                for (int j=0;j<D;++j){
                    std::uniform_real_distribution<double> Uj(L[j], U[j]);
                    S.X[i][j] = Uj(S.rng);
                }
            }
            ensureBounds(S.X[i]);
            S.fX[i] = eval(S.X[i]);
            if (S.fX[i] < S.gbest_f){ S.gbest_f = S.fX[i]; S.gbest_x = S.X[i]; }

            if (prob_->calls() >= max_evals_) break;
        }
        if (prob_->calls() >= max_evals_) break;

        last_improve_iter_[k] = 0; // from iter 0
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

    // updateStop with all fX
    std::vector<double> snapshot;
    snapshot.reserve(final_population_);
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fX.begin(), S.fX.end());
    updateStop(snapshot);
}

// ------------------ DE core ------------------
void PDE::mutate_rand1_(Island& S, int i, Vec& donor){
    const int n = (int)S.X.size();
    int r1,r2,r3; pick_3_distinct_(n, S.rng, i, r1,r2,r3);
    const double F = F_tuned_(S.rng);
    const int D = (int)donor.size();
    for (int j=0;j<D;++j){
        donor[j] = S.X[r1][j] + F * (S.X[r2][j] - S.X[r3][j]);
    }
}

void PDE::mutate_best1_(Island& S, int i, Vec& donor){
    const int n = (int)S.X.size();
    int r1 = pick_random_excluding_(n, S.rng, i);
    int r2 = pick_random_excluding_(n, S.rng, i);
    while (r2 == r1) r2 = pick_random_excluding_(n, S.rng, i);
    const double F = F_tuned_(S.rng);
    const int D = (int)donor.size();
    for (int j=0;j<D;++j){
        donor[j] = S.gbest_x[j] + F * (S.X[r1][j] - S.X[r2][j]);
    }
}

void PDE::mutate_current2best1_(Island& S, int i, Vec& donor){
    const int n = (int)S.X.size();
    int r1 = pick_random_excluding_(n, S.rng, i);
    int r2 = pick_random_excluding_(n, S.rng, i);
    while (r2 == r1) r2 = pick_random_excluding_(n, S.rng, i);
    const double F = F_tuned_(S.rng);
    const int D = (int)donor.size();
    for (int j=0;j<D;++j){
        donor[j] = S.X[i][j] + F * (S.gbest_x[j] - S.X[i][j]) + F * (S.X[r1][j] - S.X[r2][j]);
    }
}

void PDE::mutate_pbest1_(Island& S, int i, Vec& donor){
    // p-best selection from the top p_top fraction
    const int n = (int)S.X.size();
    int k = std::max(1, (int)std::round(p_top_ * n));
    if (k > n) k = n;

    // indices sorted by fX
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
        [&](int a,int b){ return S.fX[a] < S.fX[b]; });

    std::uniform_int_distribution<int> Up(0, k-1);
    int pbest_idx = idx[ Up(S.rng) ];

    int r1 = pick_random_excluding_(n, S.rng, i);
    int r2 = pick_random_excluding_(n, S.rng, i);
    while (r2 == r1) r2 = pick_random_excluding_(n, S.rng, i);

    const double F = F_tuned_(S.rng);
    const int D = (int)donor.size();
    for (int j=0;j<D;++j){
        donor[j] = S.X[i][j] + F * (S.X[pbest_idx][j] - S.X[i][j]) + F * (S.X[r1][j] - S.X[r2][j]);
    }
}

void PDE::crossover_bin_(Island& S, int i, const Vec& donor, Vec& trial){
    const int D = (int)donor.size();
    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::uniform_int_distribution<int> Uj(0, D-1);
    int jrand = Uj(S.rng);
    for (int j=0;j<D;++j){
        if (U01(S.rng) < CR_ || j == jrand) trial[j] = donor[j]; else trial[j] = S.X[i][j];
    }
}

void PDE::de_iteration_island_(Island& S){
    const int n = (int)S.X.size();
    const int D = prob_->dimension();

    for (int i=0;i<n; ++i){
        Vec donor(D,0.0), trial(D,0.0);

        if      (strategy_ == "best1bin")          mutate_best1_(S, i, donor);
        else if (strategy_ == "current2best1bin")  mutate_current2best1_(S, i, donor);
        else if (strategy_ == "pbest1bin")         mutate_pbest1_(S, i, donor);
        else                                       mutate_rand1_(S, i, donor); // default rand1bin

        crossover_bin_(S, i, donor, trial);
        ensureBounds(trial);
        double ftrial = eval(trial);

        bool improved = false;
        if (ftrial <= S.fX[i]){ // selection
            S.X[i]  = trial;
            S.fX[i] = ftrial;
            improved = true;
            if (ftrial < S.gbest_f){ S.gbest_f = ftrial; S.gbest_x = trial; }
        }

        // in-run local (as in GA/ppso)
        if (local_rate_ > 0.0 && !local_method_.empty()){
            std::uniform_real_distribution<double> U01(0.0,1.0);
            bool fire = (!inrun_on_improve_) ? (U01(S.rng) < local_rate_) :
                                               (improved && U01(S.rng) < local_rate_);
            if (fire){
                auto [xloc, floc] = localSearch(local_method_, S.X[i]);
                if (floc < S.fX[i]) { S.X[i] = xloc; S.fX[i] = floc; }
                if (floc < S.gbest_f)   { S.gbest_x = xloc; S.gbest_f = floc; }
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }
}

// ------------------ migration & stopping ------------------
std::vector<int> PDE::best_indices_(const Island& S, int N) const{
    std::vector<int> idx(S.fX.size()); std::iota(idx.begin(), idx.end(), 0);
    N = std::min(N, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin()+N, idx.end(),
        [&](int a,int b){ return S.fX[a] < S.fX[b]; });
    idx.resize(N);
    return idx;
}

int PDE::worst_index_(const Island& S) const{
    int w = 0; double fw = S.fX[0];
    for (int i=1;i<(int)S.fX.size(); ++i){
        if (S.fX[i] > fw){ fw = S.fX[i]; w = i; }
    }
    return w;
}

void PDE::send_best_Np_(int src, const std::vector<int>& dst_ids){
    if (src < 0 || src >= (int)isl_.size()) return;
    if (!island_done_.empty() && island_done_[src]) return; // source completed -> skip

    Island& A = isl_[src];
    const auto best_idx = best_indices_(A, Np_);
    for (int dst : dst_ids){
        if (dst<0 || dst>=(int)isl_.size() || dst==src) continue;
        if (!island_done_.empty() && island_done_[dst]) continue; // dest completed -> skip

        Island& B = isl_[dst];
        for (int bi : best_idx){
            int w = worst_index_(B);
            B.X[w]  = A.X[bi];
            B.fX[w] = A.fX[bi];
            if (B.fX[w] < B.gbest_f){ B.gbest_f = B.fX[w]; B.gbest_x = B.X[w]; }
        }
    }
}

void PDE::propagate_(){
    if (islands_ <= 1) return;

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

bool PDE::stopping_hold_(Island& S, double fmin_k){
    if (S.delta_hist.empty()){
        S.delta_hist.push_back(fmin_k);
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

// ------------------ one_iteration ------------------
void PDE::one_iteration(){
    if (!prob_) return;

    if (hard_stop_now_){
        printBest();
        std::vector<double> snapshot;
        snapshot.reserve(final_population_);
        for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fX.begin(), S.fX.end());
        updateStop(snapshot);
        return;
    }

    // Keep previous per-island gbest for plateau detection
    std::vector<double> gbest_prev(islands_);
    for (int k=0;k<islands_; ++k) gbest_prev[k] = isl_[k].gbest_f;

#ifdef _OPENMP
    int nthreads = threads_>0 ? threads_ : omp_get_max_threads();
    if (!use_openmp_) nthreads = 1;
    #pragma omp parallel for num_threads(nthreads) if(use_openmp_) schedule(static)
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]) de_iteration_island_(isl_[k]);
    }
#else
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]) de_iteration_island_(isl_[k]);
    }
#endif

    // update global best and update last_improve_iter_ where improvement occurs
    for (int k=0; k<islands_; ++k){
        const auto& S = isl_[k];
        if (S.gbest_f < global_best_f_){ global_best_f_ = S.gbest_f; global_best_x_ = S.gbest_x; }
        if (S.gbest_f < gbest_prev[k]) last_improve_iter_[k] = K_;
    }
    if (global_best_f_ < best_f_) { best_f_ = global_best_f_; best_x_ = global_best_x_; }

    // migration
    if (K_ > 0 && (K_ % NR_ == 0)){
        propagate_();
    }

    // per-island stopping (combination of 3 criteria)
    int done_now = 0;
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]){
            bool done = false;

            // 1) initial stagnation criterion (eps_stop/NM) on the island gbest
            if (stopping_hold_(isl_[k], isl_[k].gbest_f)) {
                done = true;
            }

            // 2) plateau in iterations without improvement
            if (!done && island_plateau_iters_ > 0) {
                if (K_ - last_improve_iter_[k] >= island_plateau_iters_) {
                    done = true;
                }
            }

            // 3) target f (if enabled)
            if (!done && std::isfinite(island_target_f_)) {
                if (isl_[k].gbest_f <= island_target_f_) {
                    done = true;
                }
            }

            if (done) island_done_[k] = 1;
        }
        if (island_done_[k]) ++done_now;
    }

    // If the target is met (K islands), everything is frozen and immediate global stop is requested.
    if (done_now >= stop_after_islands_) {
        hard_stop_now_ = true;
        std::fill(island_done_.begin(), island_done_.end(), (unsigned char)1);
    }

    ++K_;

    // report / stop history
    printBest();
    std::vector<double> snapshot;
    snapshot.reserve(final_population_);
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fX.begin(), S.fX.end());
    updateStop(snapshot);
}

// ------------------ end ------------------
void PDE::end(){
    if (end_local_refine_ && !end_local_method_.empty()){
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (floc < best_f_){ best_f_ = floc; best_x_ = xloc; }
    }

    if (!isl_.empty()){
        int worst_island = 0; double fw = isl_[0].gbest_f;
        for (int k=1;k<islands_; ++k){
            if (isl_[k].gbest_f > fw){ fw = isl_[k].gbest_f; worst_island = k; }
        }
        Island& W = isl_[worst_island];
        int wi = worst_index_(W);
        if (wi>=0 && wi<(int)W.X.size()){
            W.X[wi]  = best_x_;
            W.fX[wi] = best_f_;
            W.gbest_x = best_x_;
            W.gbest_f = best_f_;
        }
    }

    std::vector<double> snapshot;
    for (const auto& S: isl_) snapshot.insert(snapshot.end(), S.fX.begin(), S.fX.end());
    updateStop(snapshot);
    printBest();
}

} // namespace optimsolution
