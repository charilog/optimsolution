#include "nlshadelbc.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

std::string NLSHADELBC::toLower(std::string s){
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string NLSHADELBC::trim(std::string s){
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

bool NLSHADELBC::parseBool(const std::string& s, bool fb){
    std::string t = toLower(trim(s));
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    return fb;
}

int NLSHADELBC::parseInt(const std::string& s, int fb){
    std::string t = trim(s);
    if (t.empty()) return fb;
    try {
        size_t pos = 0;
        long v = std::stol(t, &pos);
        if (pos == t.size()) return (int)v;
    } catch (...) {}
    return fb;
}

double NLSHADELBC::parseDouble(const std::string& s, double fb){
    std::string t = trim(s);
    if (t.empty()) return fb;
    try {
        size_t pos = 0;
        double v = std::stod(t, &pos);
        if (pos == t.size() && std::isfinite(v)) return v;
    } catch (...) {}
    return fb;
}

void NLSHADELBC::configure(const MethodConfig& mc){
    pop_override_ = mc.getInt("population", pop_override_);
    pop_override_ = parseInt(mc.getStr("population", std::to_string(pop_override_)), pop_override_);

    use_global_population_ = parseBool(mc.getStr("use_global_population", use_global_population_ ? "1" : "0"), use_global_population_);
    paper_population_default_ = parseBool(mc.getStr("paper_population_default", paper_population_default_ ? "1" : "0"), paper_population_default_);

    H_ = std::max(1, mc.getInt("H", H_));
    H_ = std::max(1, parseInt(mc.getStr("H", std::to_string(H_)), H_));
    min_pop_ = std::max(4, mc.getInt("min_population", min_pop_));
    min_pop_ = std::max(4, parseInt(mc.getStr("min_population", std::to_string(min_pop_)), min_pop_));
    archive_ratio_ = std::max(0.0, parseDouble(mc.getStr("archive_ratio", std::to_string(archive_ratio_)), archive_ratio_));
    archive_use_prob_ = parseDouble(mc.getStr("archive_use_prob", std::to_string(archive_use_prob_)), archive_use_prob_);
    archive_use_prob_ = std::min(1.0, std::max(0.0, archive_use_prob_));
    rank_pressure_k_ = std::max(0.0, parseDouble(mc.getStr("rank_pressure_k", std::to_string(rank_pressure_k_)), rank_pressure_k_));
    max_resamples_ = std::max(1, parseInt(mc.getStr("max_resamples", std::to_string(max_resamples_)), max_resamples_));

    mf_init_ = parseDouble(mc.getStr("mf_init", std::to_string(mf_init_)), mf_init_);
    mcr_init_ = parseDouble(mc.getStr("mcr_init", std::to_string(mcr_init_)), mcr_init_);
    memory_c_ = parseDouble(mc.getStr("memory_c", std::to_string(memory_c_)), memory_c_);
    memory_c_ = std::min(1.0, std::max(0.0, memory_c_));

    pbest_start_ = parseDouble(mc.getStr("pbest_start", std::to_string(pbest_start_)), pbest_start_);
    pbest_end_ = parseDouble(mc.getStr("pbest_end", std::to_string(pbest_end_)), pbest_end_);
    pbest_start_ = std::min(1.0, std::max(0.0, pbest_start_));
    pbest_end_ = std::min(1.0, std::max(0.0, pbest_end_));

    mean_m_ = parseDouble(mc.getStr("mean_m", std::to_string(mean_m_)), mean_m_);
    p_init_f_ = parseDouble(mc.getStr("p_init_f", std::to_string(p_init_f_)), p_init_f_);
    p_final_f_ = parseDouble(mc.getStr("p_final_f", std::to_string(p_final_f_)), p_final_f_);
    p_init_cr_ = parseDouble(mc.getStr("p_init_cr", std::to_string(p_init_cr_)), p_init_cr_);
    p_final_cr_ = parseDouble(mc.getStr("p_final_cr", std::to_string(p_final_cr_)), p_final_cr_);

    debug_nlshadelbc_ = mc.getInt("debug_nlshadelbc", debug_nlshadelbc_);
    debug_nlshadelbc_ = parseInt(mc.getStr("debug_nlshadelbc", std::to_string(debug_nlshadelbc_)), debug_nlshadelbc_);

    const int flg = mc.getInt("end_local_refine", end_local_refine_ ? 1 : 0);
    end_local_refine_ = parseBool(mc.getStr("end_local_refine", std::to_string(flg)), flg != 0);
    end_local_method_ = toLower(trim(mc.getStr("end_local_method", end_local_method_)));
}

bool NLSHADELBC::inBounds(const Vec& x) const {
    const Vec& lb = prob_->lb();
    const Vec& ub = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        double lo = (j < lb.size() ? lb[j] : -1.0);
        double hi = (j < ub.size() ? ub[j] :  1.0);
        if (x[j] < lo || x[j] > hi || !std::isfinite(x[j])) return false;
    }
    return true;
}

void NLSHADELBC::midpointTargetRepair(Vec& u, const Vec& x){
    const Vec& lb = prob_->lb();
    const Vec& ub = prob_->ub();
    for (size_t j = 0; j < u.size(); ++j) {
        const double lo = (j < lb.size() ? lb[j] : -1.0);
        const double hi = (j < ub.size() ? ub[j] :  1.0);
        if (!std::isfinite(u[j])) {
            u[j] = 0.5 * (x[j] + 0.5 * (lo + hi));
        }
        if (u[j] < lo) u[j] = 0.5 * (x[j] + lo);
        else if (u[j] > hi) u[j] = 0.5 * (x[j] + hi);
    }
}

double NLSHADELBC::sampleNormalClamped(double mean, double sd, double lo, double hi){
    std::normal_distribution<double> dist(mean, sd);
    double v = dist(rng_);
    if (!std::isfinite(v)) v = mean;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

double NLSHADELBC::samplePositiveCauchyClamped(double location, double scale, double hi){
    std::cauchy_distribution<double> dist(location, scale);
    double v = location;
    for (int t = 0; t < 100; ++t) {
        v = dist(rng_);
        if (std::isfinite(v) && v > 0.0) {
            if (v > hi) v = hi;
            return v;
        }
    }
    v = std::max(1e-12, std::min(hi, location > 0.0 ? location : 0.5));
    return v;
}

int NLSHADELBC::uniformIndex(int n){
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(rng_);
}

int NLSHADELBC::chooseUniformPopulationIndex(int exclude1, int exclude2, int exclude3){
    const int n = (int)X_.size();
    int r = 0;
    do { r = uniformIndex(n); } while (r == exclude1 || r == exclude2 || r == exclude3);
    return r;
}

int NLSHADELBC::chooseUniformArchiveIndex(){
    return uniformIndex((int)archive_X_.size());
}

int NLSHADELBC::chooseRankBasedPopulationIndex(const std::vector<int>& sorted_idx,
                                               std::discrete_distribution<int>& dist,
                                               int exclude1, int exclude2, int exclude3){
    for (int t = 0; t < 128; ++t) {
        const int pos = dist(rng_);
        const int idx = sorted_idx[pos];
        if (idx != exclude1 && idx != exclude2 && idx != exclude3) return idx;
    }
    return chooseUniformPopulationIndex(exclude1, exclude2, exclude3);
}

int NLSHADELBC::chooseTopPBestIndex(const std::vector<int>& sorted_idx, int pbest_count, int exclude1){
    pbest_count = std::max(1, std::min((int)sorted_idx.size(), pbest_count));
    for (int t = 0; t < 128; ++t) {
        std::uniform_int_distribution<int> dist(0, pbest_count - 1);
        const int idx = sorted_idx[dist(rng_)];
        if (idx != exclude1) return idx;
    }
    for (int pos = 0; pos < pbest_count; ++pos) {
        if (sorted_idx[pos] != exclude1) return sorted_idx[pos];
    }
    return sorted_idx[0];
}

NLSHADELBC::Vec NLSHADELBC::binomialCrossover(const Vec& target, const Vec& donor, double cr){
    const int D = (int)target.size();
    Vec trial = target;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Jrand(0, std::max(0, D - 1));
    const int jrand = Jrand(rng_);
    for (int j = 0; j < D; ++j) {
        if (U01(rng_) < cr || j == jrand) trial[j] = donor[j];
    }
    return trial;
}

double NLSHADELBC::generalizedLehmerMean(const std::vector<double>& S,
                                         const std::vector<double>& weights,
                                         double p, double m) const {
    if (S.empty() || weights.empty() || S.size() != weights.size()) return 0.0;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < S.size(); ++i) {
        const double s = std::max(1e-12, S[i]);
        num += weights[i] * std::pow(s, p);
        den += weights[i] * std::pow(s, p - m);
    }
    if (den <= 0.0 || !std::isfinite(num) || !std::isfinite(den)) return S.back();
    double v = num / den;
    if (!std::isfinite(v)) return S.back();
    return v;
}

double NLSHADELBC::currentPFExponent() const {
    const double nfe = (double)prob_->calls();
    const double nfe_max = (double)std::max<long long>(1, max_evals_);
    const double pg = std::round(((nfe_max - nfe) * (p_init_f_ - p_final_f_) / nfe_max) + p_final_f_);
    return pg;
}

double NLSHADELBC::currentPCRExponent() const {
    const double nfe = (double)prob_->calls();
    const double nfe_max = (double)std::max<long long>(1, max_evals_);
    const double pg = std::round(((nfe_max - nfe) * (p_init_cr_ - p_final_cr_) / nfe_max) + p_final_cr_);
    return pg;
}

int NLSHADELBC::currentPBestCount(int np) const {
    const double ratio = (double)prob_->calls() / (double)std::max<long long>(1, max_evals_);
    const double p = pbest_start_ + (pbest_end_ - pbest_start_) * ratio;
    return std::max(2, std::min(np, (int)std::floor(np * p)));
}

int NLSHADELBC::currentPopulationSizeByNLPSR() const {
    if (initial_population_ < 0) return min_pop_;
    const double ratio = std::min(0.999999, std::max(0.0,
        (double)prob_->calls() / (double)std::max<long long>(1, max_evals_)));
    const double nonlinear = std::pow(ratio, 1.0 - ratio);
    const double next = ((double)min_pop_ - (double)initial_population_) * nonlinear + (double)initial_population_;
    return std::max(min_pop_, (int)std::llround(next));
}

void NLSHADELBC::updateArchive(const Vec& replaced_x, double replaced_f, int target_archive_size){
    if (target_archive_size <= 0) return;
    if ((int)archive_X_.size() < target_archive_size) {
        archive_X_.push_back(replaced_x);
        archive_F_.push_back(replaced_f);
        return;
    }
    const int nA = (int)archive_X_.size();
    for (int steps = 0; steps < nA; ++steps) {
        const int ra = uniformIndex(nA);
        if (replaced_f < archive_F_[ra]) {
            archive_X_[ra] = replaced_x;
            archive_F_[ra] = replaced_f;
            return;
        }
    }
    const int ra = uniformIndex(nA);
    archive_X_[ra] = replaced_x;
    archive_F_[ra] = replaced_f;
}

void NLSHADELBC::shrinkPopulationAndArchive(int next_np, int next_na){
    next_np = std::max(min_pop_, next_np);
    next_np = std::min(next_np, (int)X_.size());
    if ((int)X_.size() > next_np) {
        std::vector<int> order(X_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });
        std::vector<Vec> newX;
        std::vector<double> newF;
        newX.reserve(next_np);
        newF.reserve(next_np);
        for (int i = 0; i < next_np; ++i) {
            newX.push_back(X_[order[i]]);
            newF.push_back(FX_[order[i]]);
        }
        X_.swap(newX);
        FX_.swap(newF);
        this->setPopulation((int)X_.size());
    }
    next_na = std::max(0, next_na);
    while ((int)archive_X_.size() > next_na) {
        const int idx = uniformIndex((int)archive_X_.size());
        archive_X_.erase(archive_X_.begin() + idx);
        archive_F_.erase(archive_F_.begin() + idx);
    }
}

void NLSHADELBC::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    int N = -1;
    if (pop_override_ >= min_pop_) {
        N = pop_override_;
    } else if (paper_population_default_) {
        N = std::max(min_pop_, (int)std::llround(23.0 * D));
    } else if (use_global_population_ && population() >= min_pop_) {
        N = population();
    } else {
        N = std::max(min_pop_, population());
    }

    initial_population_ = N;
    this->setPopulation(N);

    X_.clear(); FX_.clear(); archive_X_.clear(); archive_F_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);
    FX_.assign(N, std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < N; ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    MF_.assign(H_, mf_init_);
    MCR_.assign(H_, mcr_init_);
    memory_index_ = 0;

    if (debug_nlshadelbc_) {
        std::fprintf(stdout,
            "[nlshadelbc] init -> D=%d, N=%d, H=%d, archive_ratio=%.3f, pbest=[%.3f, %.3f], mf_init=%.3f, mcr_init=%.3f\n",
            D, N, H_, archive_ratio_, pbest_start_, pbest_end_, mf_init_, mcr_init_);
        std::fflush(stdout);
    }

    printBest();
}

void NLSHADELBC::one_iteration(){
    if (!prob_ || X_.empty()) return;
    const int D = prob_->dimension();
    const int NP = (int)X_.size();
    if (NP < min_pop_) {
        printBest();
        updateStop(FX_);
        return;
    }

    std::vector<int> sorted_idx(NP);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    std::vector<double> weights_by_rank(NP, 1.0);
    for (int pos = 0; pos < NP; ++pos) {
        weights_by_rank[pos] = std::exp(-rank_pressure_k_ * (double)(pos + 1) / (double)NP);
    }
    std::discrete_distribution<int> rank_dist(weights_by_rank.begin(), weights_by_rank.end());

    std::vector<double> cr_pool(NP);
    for (int i = 0; i < NP; ++i) {
        const int mem_idx = uniformIndex(H_);
        cr_pool[i] = sampleNormalClamped(MCR_[mem_idx], 0.1, 0.0, 1.0);
    }
    std::sort(cr_pool.begin(), cr_pool.end());

    std::vector<double> SF, SCR, improvements;
    SF.reserve(NP); SCR.reserve(NP); improvements.reserve(NP);

    const int pbest_count = currentPBestCount(NP);
    const int target_archive_size = std::max(0, (int)std::llround(archive_ratio_ * (double)NP));
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int rank_pos = 0; rank_pos < NP; ++rank_pos) {
        const int i = sorted_idx[rank_pos];
        const Vec xi = X_[i];
        const double fi = FX_[i];
        const double cr = cr_pool[rank_pos];

        Vec trial = xi;
        double lastF = std::min(1.0, std::max(1e-12, MF_[uniformIndex(H_)]));

        for (int attempt = 0; attempt < max_resamples_; ++attempt) {
            const int mem_idx_f = uniformIndex(H_);
            const double F = samplePositiveCauchyClamped(MF_[mem_idx_f], 0.1, 1.0);
            lastF = F;
            const int pbest_idx = chooseTopPBestIndex(sorted_idx, pbest_count, i);
            const int r1_idx = chooseUniformPopulationIndex(i, pbest_idx, -1);

            bool use_archive = !archive_X_.empty() && (U01(rng_) < archive_use_prob_);
            int r2_pop_idx = -1;
            int r2_archive_idx = -1;
            if (use_archive) {
                r2_archive_idx = chooseUniformArchiveIndex();
            } else {
                r2_pop_idx = chooseRankBasedPopulationIndex(sorted_idx, rank_dist, i, pbest_idx, r1_idx);
            }

            Vec donor(D, 0.0);
            if (use_archive) {
                for (int j = 0; j < D; ++j) {
                    donor[j] = xi[j] + F * (X_[pbest_idx][j] - xi[j]) + F * (X_[r1_idx][j] - archive_X_[r2_archive_idx][j]);
                }
            } else {
                for (int j = 0; j < D; ++j) {
                    donor[j] = xi[j] + F * (X_[pbest_idx][j] - xi[j]) + F * (X_[r1_idx][j] - X_[r2_pop_idx][j]);
                }
            }

            trial = binomialCrossover(xi, donor, cr);
            if (inBounds(trial)) {
                const double fu = eval(trial);
                if (fu < FX_[i]) {
                    updateArchive(xi, fi, target_archive_size);
                    X_[i] = trial;
                    FX_[i] = fu;
                    if (fu < best_f_) {
                        best_f_ = fu;
                        best_x_ = trial;
                    }
                    SF.push_back(F);
                    SCR.push_back(cr);
                    improvements.push_back(std::fabs(fi - fu));
                }
                goto next_individual;
            }
        }

        midpointTargetRepair(trial, xi);
        {
            const double fu = eval(trial);
            if (fu < FX_[i]) {
                updateArchive(xi, fi, target_archive_size);
                X_[i] = trial;
                FX_[i] = fu;
                if (fu < best_f_) {
                    best_f_ = fu;
                    best_x_ = trial;
                }
                SF.push_back(lastF);
                SCR.push_back(cr);
                improvements.push_back(std::fabs(fi - fu));
            }
        }

    next_individual:
        if (prob_->calls() >= max_evals_) break;
    }

    if (!SF.empty() && SF.size() == SCR.size() && SF.size() == improvements.size()) {
        double sum_imp = std::accumulate(improvements.begin(), improvements.end(), 0.0);
        std::vector<double> w(improvements.size(), 0.0);
        if (sum_imp <= 0.0) {
            const double uni = 1.0 / (double)improvements.size();
            std::fill(w.begin(), w.end(), uni);
        } else {
            for (size_t i = 0; i < improvements.size(); ++i) w[i] = improvements[i] / sum_imp;
        }
        const double meanF = generalizedLehmerMean(SF, w, currentPFExponent(), mean_m_);
        const double meanCR = generalizedLehmerMean(SCR, w, currentPCRExponent(), mean_m_);
        MF_[memory_index_] = memory_c_ * MF_[memory_index_] + (1.0 - memory_c_) * std::min(1.0, std::max(1e-12, meanF));
        MCR_[memory_index_] = memory_c_ * MCR_[memory_index_] + (1.0 - memory_c_) * std::min(1.0, std::max(0.0, meanCR));
        memory_index_ = (memory_index_ + 1) % H_;
    }

    const int next_np = currentPopulationSizeByNLPSR();
    const int next_na = std::max(0, (int)std::llround(archive_ratio_ * next_np));
    shrinkPopulationAndArchive(next_np, next_na);

    printBest();
    updateStop(FX_);
}

void NLSHADELBC::end(){
    if (!end_local_refine_ || !prob_ || end_local_method_.empty()) return;
    std::pair<Vec, double> loc = localSearch(end_local_method_, best_x_);
    if (std::isfinite(loc.second) && loc.second < best_f_) {
        best_x_ = loc.first;
        best_f_ = loc.second;
        if (!X_.empty()) {
            size_t worst = 0;
            for (size_t i = 1; i < FX_.size(); ++i) if (FX_[i] > FX_[worst]) worst = i;
            X_[worst] = best_x_;
            FX_[worst] = best_f_;
        }
    }
    printBest();
}

} // namespace optimsolution
