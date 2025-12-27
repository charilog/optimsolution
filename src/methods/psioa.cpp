#include "psioa.h"

#include <thread>
#include <atomic>
#include <limits>
#include <random>
#include <cstring>
#include <numeric>
#include <cmath>
#include <cctype>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimsolution {

// ------------------ configure ------------------
void PSIOA::configure(const MethodConfig& mc){
    // Population override (reported in summary)
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) Optimizer::setPopulation(pop_cfg_);

    islands_     = std::max(1, mc.getInt("islands", islands_));
    NR_          = std::max(1, mc.getInt("NR", NR_));
    Np_          = std::max(1, mc.getInt("Np", Np_));
    propagation_ = mc.getStr("propagation", propagation_);
    for (auto &ch : propagation_) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
    eps_stop_    = mc.getDbl("eps_stop", eps_stop_);
    NM_          = std::max(1, mc.getInt("NM", NM_));

    use_openmp_  = mc.getBool("use_openmp", use_openmp_);
    threads_     = mc.getInt("threads", threads_);

    // locals
    local_rate_        = mc.getDbl("local_rate", local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;
    local_method_      = mc.getStr("local_method", local_method_);
    inrun_on_improve_  = mc.getBool("inrun_on_improve", inrun_on_improve_);

    end_local_refine_  = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_  = mc.getStr("end_local_method", end_local_method_);

    stop_after_islands_ = mc.getInt("stop_after_islands", stop_after_islands_);

    // PSIOA params
    c1_  = mc.getDbl("c1", c1_);
    c2_  = mc.getDbl("c2", c2_);
    Rmin_ = std::max(1e-12, mc.getDbl("Rmin", Rmin_));
    Rmax_ = std::max(Rmin_, mc.getDbl("Rmax", Rmax_));
    p_spor0_ = std::clamp(mc.getDbl("pspor0", p_spor0_), 0.0, 1.0);
    p_germ0_ = std::clamp(mc.getDbl("pgerm0", p_germ0_), 0.0, 1.0);
    p_zero_  = std::clamp(mc.getDbl("pzero",  p_zero_),  0.0, 1.0);

    adapt_R_kappa_    = std::clamp(mc.getDbl("adapt_R_kappa",    adapt_R_kappa_),    0.0, 1.0);
    adapt_prob_kappa_ = std::clamp(mc.getDbl("adapt_prob_kappa", adapt_prob_kappa_), 0.0, 1.0);

    crowding_metric_  = mc.getStr("crowding_metric", crowding_metric_);
    for (auto &ch : crowding_metric_) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}

    island_plateau_iters_ = mc.getInt("island_plateau_iters", island_plateau_iters_);
    island_target_f_      = mc.getDbl("island_target_f", island_target_f_);

    final_population_ = -1;
    hard_stop_now_    = false;
}

// ------------------ utilities ------------------
void PSIOA::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j]+U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

std::vector<int> PSIOA::best_indices_(const Island& S, int N) const{
    std::vector<int> idx(S.fX.size()); std::iota(idx.begin(), idx.end(), 0);
    N = std::min(N, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin()+N, idx.end(),
        [&](int a,int b){ return S.fX[a] < S.fX[b]; });
    idx.resize(N);
    return idx;
}

int PSIOA::worst_index_(const Island& S) const{
    int w = 0; double fw = S.fX[0];
    for (int i=1;i<(int)S.X.size(); ++i){
        if (S.fX[i] > fw){ fw = S.fX[i]; w = i; }
    }
    return w;
}

void PSIOA::send_best_Np_(int src, const std::vector<int>& dst_ids){
    if (src < 0 || src >= (int)isl_.size()) return;
    if (island_done_[src]) return;
    Island& A = isl_[src];
    auto best = best_indices_(A, Np_);
    for (int d : dst_ids){
        if (d<0 || d>=(int)isl_.size() || d==src) continue;
        if (island_done_[d]) continue;
        Island& B = isl_[d];
        for (int bi : best){
            int w = worst_index_(B);
            B.X[w]  = A.X[bi];
            B.fX[w] = A.fX[bi];
            if (B.fX[w] < B.gbest_f){ B.gbest_f = B.fX[w]; B.gbest_x = B.X[w]; }
        }
    }
}

void PSIOA::propagate_(){
    if (islands_ <= 1) return;

    if (propagation_ == "1to1"){
        std::uniform_int_distribution<int> U(0, islands_-1);
        int s = U(rng_), d = U(rng_);
        int guard=0;
        while ((s==d) || island_done_[s] || island_done_[d]){
            if (++guard > 10*islands_) return;
            s = U(rng_); d = U(rng_);
        }
        send_best_Np_(s, {d});
    } else if (propagation_ == "2toN"){
        if (islands_ >= 2){
            std::uniform_int_distribution<int> U(0, islands_-1);
            int s1 = U(rng_), s2 = U(rng_);
            int guard=0;
            while ((s1==s2) || island_done_[s1] || island_done_[s2]){
                if (++guard > 10*islands_) return;
                s1 = U(rng_); s2 = U(rng_);
            }
            std::vector<int> all; all.reserve(islands_);
            for (int k=0;k<islands_; ++k) if (!island_done_[k]) all.push_back(k);
            if (all.size() < 2) return;
            send_best_Np_(s1, all);
            send_best_Np_(s2, all);
        }
    } else if (propagation_ == "nto1"){
        std::uniform_int_distribution<int> U(0, islands_-1);
        int dst = U(rng_);
        int guard=0;
        while (island_done_[dst]){
            if (++guard > 10*islands_) return;
            dst = U(rng_);
        }
        std::vector<int> one{dst};
        for (int s=0; s<islands_; ++s){
            if (s==dst || island_done_[s]) continue;
            send_best_Np_(s, one);
        }
    } else { // NtoN
        for (int s=0; s<islands_; ++s){
            if (island_done_[s]) continue;
            std::vector<int> all; all.reserve(islands_-1);
            for (int d=0; d<islands_; ++d) if (d!=s && !island_done_[d]) all.push_back(d);
            if (!all.empty()) send_best_Np_(s, all);
        }
    }
}

// ------------------ PSIOA primitives ------------------
// Sporulation + zero-reset + clamp.
Vec PSIOA::make_spore_(Island& S, const Vec& xi) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)xi.size();

    std::uniform_real_distribution<double> U01(0.0,1.0);
    auto Uni   = [&](double a,double b){ return a + (b-a)*U01(S.rng); };

    Vec s = xi; s.assign(D, 0.0);
    for (int j=0;j<D;++j){
        const double u1 = Uni(-1.0, 1.0);
        const double u2 = Uni(-S.R, S.R);
        // si = xi + R*u(-1,1)*c1 + c2*(xbest - xi + u(-R,R))
        s[j] = xi[j] + S.R * u1 * c1_ + c2_ * (S.gbest_x[j] - xi[j] + u2);
        // zero-reset with prob p_zero per dimension
        if (U01(S.rng) < p_zero_) s[j] = 0.0;
        // clamp
        if (s[j] < L[j]) s[j] = L[j];
        if (s[j] > U[j]) s[j] = U[j];
    }
    return s;
}

// Similarity (crowding): bounds-normalized L2 or plain L2
int PSIOA::most_similar_index_(const Island& S, const Vec& s) const {
    const int n = (int)S.X.size();
    const int D = (int)s.size();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    int arg = 0;
    double bestd = std::numeric_limits<double>::infinity();
    for (int i=0;i<n;++i){
        double d2 = 0.0;
        if (crowding_metric_ == "bnorm"){
            for (int j=0;j<D;++j){
                const double span = std::max(1e-32, U[j]-L[j]);
                const double z = (s[j]-S.X[i][j])/span;
                d2 += z*z;
            }
        } else {
            for (int j=0;j<D;++j){
                const double z = (s[j]-S.X[i][j]);
                d2 += z*z;
            }
        }
        if (d2 < bestd){ bestd = d2; arg = i; }
    }
    return arg;
}

// Adapt R, p_spor, p_germ based on mean fitness trend (avg_f vs prev_avg)
void PSIOA::adapt_controls_(Island& S, double avg_f, double prev_avg){
    if (!std::isfinite(prev_avg)) { S.last_avg_f = avg_f; return; }

    // If improving -> decrease R, increase germ, decrease spor.
    // If stagnant/worsening -> increase R, increase spor, decrease germ.
    const bool improving = (avg_f < prev_avg - 1e-12);
    const double sgn = improving ? -1.0 : +1.0;

    S.R      = std::clamp(S.R + sgn * adapt_R_kappa_ * (Rmax_ - Rmin_), Rmin_, Rmax_);
    S.p_spor = std::clamp(S.p_spor + (sgn>0 ? +1 : -1) * adapt_prob_kappa_, 0.0, 1.0);
    S.p_germ = std::clamp(S.p_germ + (sgn>0 ? -1 : +1) * adapt_prob_kappa_, 0.0, 1.0);

    S.last_avg_f = avg_f;
}

// NM-window stationarity
bool PSIOA::stopping_hold_(Island& S, double fmin_k){
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

// ------------------ init ------------------
void PSIOA::init(){
    if (!prob_) return;

    const int global_pop = Optimizer::population();
    final_population_ = std::max(4, (pop_cfg_>0 ? pop_cfg_ : global_pop));
    Optimizer::setPopulation(final_population_);

    islands_ = std::max(1, islands_);
    if (stop_after_islands_ <= 0) stop_after_islands_ = islands_;
    if (stop_after_islands_ > islands_) stop_after_islands_ = islands_;

    per_island_ = std::max(1, final_population_ / islands_);
    int remainder = final_population_ - per_island_*islands_;

    isl_.clear(); isl_.resize(islands_);
    island_done_.assign(islands_, 0);
    last_improve_iter_.assign(islands_, 0);

    const int D = prob_->dimension();

    Initializer initSampler;
    initSampler.configure(initopt_);
    std::vector<Vec> allX = initSampler.samplePopulation(*prob_, rng_, final_population_);

    int cursor = 0;
    for (int k=0;k<islands_; ++k){
        Island& S = isl_[k];
        const int mk = per_island_ + (k < remainder ? 1 : 0);

        S.X.assign(mk, Vec(D,0.0));
        S.fX.assign(mk, std::numeric_limits<double>::infinity());
        S.gbest_x.assign(D, 0.0);
        S.gbest_f = std::numeric_limits<double>::infinity();
        S.delta_hist.clear();
        S.rng.seed( 0x9E3779B97F4A7C15ULL ^ (uint64_t)k ^ (uint64_t)rng_() );

        // self-adaptive starts
        S.R      = 0.5*(Rmin_ + Rmax_);
        S.p_spor = p_spor0_;
        S.p_germ = p_germ0_;
        S.last_avg_f = std::numeric_limits<double>::infinity();

        for (int i=0;i<mk && cursor<(int)allX.size(); ++i, ++cursor){
            S.X[i] = allX[cursor];
            ensureBounds(S.X[i]);
            S.fX[i] = eval(S.X[i]);
            if (S.fX[i] < S.gbest_f){ S.gbest_f = S.fX[i]; S.gbest_x = S.X[i]; }
            if (prob_->calls() >= max_evals_) break;
        }
        if (prob_->calls() >= max_evals_) break;

        last_improve_iter_[k] = 0;
    }

    global_best_f_ = std::numeric_limits<double>::infinity();
    global_best_x_.assign(D, 0.0);
    for (const auto& S: isl_){
        if (S.gbest_f < global_best_f_) { global_best_f_ = S.gbest_f; global_best_x_ = S.gbest_x; }
    }
    best_f_ = global_best_f_; best_x_ = global_best_x_;
    K_ = 0; hard_stop_now_ = false;

    printBest();

    // prime stopper
    std::vector<double> snap; snap.reserve(final_population_);
    for (const auto& S: isl_) snap.insert(snap.end(), S.fX.begin(), S.fX.end());
    updateStop(snap);
}

// ------------------ one_iteration ------------------
void PSIOA::one_iteration(){
    if (!prob_) return;

    if (hard_stop_now_){
        printBest();
        std::vector<double> snap;
        snap.reserve(final_population_);
        for (const auto& S: isl_) snap.insert(snap.end(), S.fX.begin(), S.fX.end());
        updateStop(snap);
        return;
    }

    // Keep previous gbest for plateau detection
    std::vector<double> gbest_prev(islands_);
    for (int k=0;k<islands_; ++k) gbest_prev[k] = isl_[k].gbest_f;

    // --- SPORULATION + GERMINATION per island ---
#ifdef _OPENMP
    int nthreads = threads_>0 ? threads_ : omp_get_max_threads();
    if (!use_openmp_) nthreads = 1;
    #pragma omp parallel for num_threads(nthreads) if(use_openmp_) schedule(static)
    for (int k=0;k<islands_; ++k){
#else
    for (int k=0;k<islands_; ++k){
#endif
        if (island_done_[k]) continue;
        Island& S = isl_[k];
        const int n = (int)S.X.size();

        // Trend: mean f
        double avg_f = 0.0;
        for (double v : S.fX) avg_f += v;
        avg_f /= std::max(1, n);

        // sporulation: for each individual, with prob p_spor -> produce a spore
        std::uniform_real_distribution<double> U01(0.0,1.0);
        for (int i=0;i<n; ++i){
            if (U01(S.rng) < S.p_spor){
                Vec sp = make_spore_(S, S.X[i]);
                double fs = eval(sp);

                // germination: crowding replacement with prob p_germ
                if (U01(S.rng) < S.p_germ){
                    int m = most_similar_index_(S, sp);
                    if (fs < S.fX[m]){
                        bool improved = fs < S.fX[m];
                        S.X[m]  = sp;
                        S.fX[m] = fs;
                        if (fs < S.gbest_f){ S.gbest_f = fs; S.gbest_x = sp; }

                        // in-run local
                        if (local_rate_ > 0.0 && !local_method_.empty()){
                            bool fire = (!inrun_on_improve_) ? (U01(S.rng) < local_rate_) :
                                                           (improved && U01(S.rng) < local_rate_);
                            if (fire){
                                auto [xloc, floc] = localSearch(local_method_, S.X[m]);
                                if (floc < S.fX[m]) { S.X[m] = xloc; S.fX[m] = floc; }
                                if (floc < S.gbest_f) { S.gbest_x = xloc; S.gbest_f = floc; }
                            }
                        }
                    }
                }
                if (prob_->calls() >= max_evals_) break;
            }
            if (prob_->calls() >= max_evals_) break;
        }

        // Adaptation based on avg_f trend
        adapt_controls_(S, avg_f, S.last_avg_f);
    }

    // update globals, plateau timestamps
    for (int k=0;k<islands_; ++k){
        const auto& S = isl_[k];
        if (S.gbest_f < global_best_f_) { global_best_f_ = S.gbest_f; global_best_x_ = S.gbest_x; }
        if (S.gbest_f < gbest_prev[k])   last_improve_iter_[k] = K_;
    }
    if (global_best_f_ < best_f_) { best_f_ = global_best_f_; best_x_ = global_best_x_; }

    // Migration every NR steps
    if (K_>0 && (K_%NR_==0)) propagate_();

    // per-island stop (NM stationarity + plateau + target)
    int done_now = 0;
    for (int k=0;k<islands_; ++k){
        if (!island_done_[k]){
            bool done = false;
            if (stopping_hold_(isl_[k], isl_[k].gbest_f)) done = true;
            if (!done && island_plateau_iters_>0 && (K_ - last_improve_iter_[k] >= island_plateau_iters_)) done = true;
            if (!done && std::isfinite(island_target_f_) && isl_[k].gbest_f <= island_target_f_) done = true;
            if (done) island_done_[k] = 1;
        }
        if (island_done_[k]) ++done_now;
    }

    if (done_now >= stop_after_islands_){
        hard_stop_now_ = true;
        std::fill(island_done_.begin(), island_done_.end(), (unsigned char)1);
    }

    ++K_;

    // report & stopper
    printBest();
    std::vector<double> snap; snap.reserve(final_population_);
    for (const auto& S: isl_) snap.insert(snap.end(), S.fX.begin(), S.fX.end());
    updateStop(snap);
}

// ------------------ end ------------------
void PSIOA::end(){
    if (end_local_refine_ && !end_local_method_.empty()){
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (floc < best_f_) { best_f_ = floc; best_x_ = xloc; }
    }
    // Inject best into the structures (alignment with others)
    if (!isl_.empty()){
        int wi = 0; double fw = isl_[0].gbest_f;
        for (int k=1;k<islands_; ++k){
            if (isl_[k].gbest_f > fw){ fw = isl_[k].gbest_f; wi = k; }
        }
        Island& W = isl_[wi];
        int w = worst_index_(W);
        if (w>=0 && w<(int)W.X.size()){
            W.X[w]  = best_x_; W.fX[w] = best_f_;
            W.gbest_x = best_x_; W.gbest_f = best_f_;
        }
    }

    std::vector<double> snap;
    for (const auto& S: isl_) snap.insert(snap.end(), S.fX.begin(), S.fX.end());
    updateStop(snap);
    printBest();
}

} // namespace optimsolution
