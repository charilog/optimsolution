// =============================================================================
//  L-SRTDE (Stanovov & Semenkin, IEEE CEC 2024 winner) -- see lsrtde.h
//
//  This is a faithful port of Optimizer::MainCycle() from the official
//  L-SRTDE.cpp (github.com/VladimirStanovov/L-SRTDE_CEC-2024). Comments of
//  the form [ref: ...] point to the corresponding construct of that file.
//
//  Deliberate, documented differences from the reference code (none of them
//  changes the algorithm itself):
//   (a) Bounds are taken from the Problem (per-dimension lb/ub) instead of the
//       hard-coded CEC box [-100, 100].
//   (b) The initial population is drawn through the framework Initializer
//       (honours the [init] section; default "uniform" == reference).
//   (c) The evaluation budget is checked inside the generation, so the run
//       never exceeds max_evals (the reference may overshoot by < N evals).
//   (d) After the population shrinks, the round-robin front pointer is
//       wrapped into the new front size (the reference could write one
//       successful trial into an already-removed slot, i.e. lose it).
//   (e) The main population is always sorted before truncation (the
//       reference skips the sort when a generation had zero successes, which
//       only matters if that happens in the very first generation).
//   (f) Uniform integers use std::uniform_int_distribution instead of
//       "rand32768 % n" (removes a tiny modulo bias).
//  Optional switches (defaults reproduce the reference exactly):
//   * delta_from_parent = 0 : the fitness improvement that weights the CR
//     memory is computed AFTER the round-robin overwrite of the front, exactly
//     as in the reference code ("FitDelta = |FitArrFront[TheChosenOne] -
//     TempFit|" placed after "FitArrFront[PFIndex] = TempFit"). When the
//     pointer hits the chosen parent this yields 0. Set to 1 to use the
//     paper formula |f(parent) - f(trial)| instead.
//   * bound_handling = random (reference) | midpoint | clip
// =============================================================================

#include "lsrtde.h"
#include "init.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <limits>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace optimsolution {

namespace {

std::string lsrtdeLower(std::string s)
{
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string lsrtdeTrim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace

// -----------------------------------------------------------------------------
// configuration
// NOTE: Config::load() lower-cases every key, so all keys here are lower-case.
// -----------------------------------------------------------------------------
void LSRTDE::configure(const MethodConfig& mc)
{
    pop_override_ = mc.getInt("population", pop_override_);
    if (pop_override_ < 0) pop_override_ = 0;

    np_init_mult_ = mc.getDbl("np_init_multiplier", np_init_mult_);
    if (!(np_init_mult_ > 0.0)) np_init_mult_ = 20.0;

    np_min_ = mc.getInt("np_min", np_min_);
    if (np_min_ < 4) np_min_ = 4;           // mutation needs >= 3 distinct + target

    memory_size_ = mc.getInt("memory_size", memory_size_);
    if (memory_size_ < 1) memory_size_ = 1;

    mcr_init_ = mc.getDbl("mcr_init", mcr_init_);
    mcr_init_ = std::min(1.0, std::max(0.0, mcr_init_));

    memory_c_ = mc.getDbl("memory_c", memory_c_);
    memory_c_ = std::min(1.0, std::max(0.0, memory_c_));

    sigma_cr_ = mc.getDbl("sigma_cr", sigma_cr_);
    if (sigma_cr_ < 0.0) sigma_cr_ = 0.05;

    f_base_    = mc.getDbl("f_base", f_base_);
    f_amp_     = mc.getDbl("f_amp", f_amp_);
    f_sr_coef_ = mc.getDbl("f_sr_coef", f_sr_coef_);
    sigma_f_   = mc.getDbl("sigma_f", sigma_f_);
    if (sigma_f_ < 0.0) sigma_f_ = 0.02;

    pbest_rate_    = mc.getDbl("pbest_rate", pbest_rate_);
    pbest_rate_    = std::min(1.0, std::max(0.0, pbest_rate_));
    pbest_sr_coef_ = mc.getDbl("pbest_sr_coef", pbest_sr_coef_);
    pbest_min_     = mc.getInt("pbest_min", pbest_min_);
    if (pbest_min_ < 2) pbest_min_ = 2;     // keeps "prand != target" satisfiable

    rank_pressure_ = mc.getDbl("rank_pressure", rank_pressure_);
    if (rank_pressure_ < 0.0) rank_pressure_ = 0.0;

    sr_init_ = mc.getDbl("sr_init", sr_init_);
    sr_init_ = std::min(1.0, std::max(0.0, sr_init_));

    delta_from_parent_ = mc.getBool("delta_from_parent", delta_from_parent_);

    {
        const std::string b = lsrtdeLower(lsrtdeTrim(mc.getStr("bound_handling", "")));
        if (b == "random" || b == "reinit" || b == "uniform") bound_mode_ = BoundMode::Random;
        else if (b == "midpoint" || b == "midpoint_target")   bound_mode_ = BoundMode::Midpoint;
        else if (b == "clip" || b == "clamp")                 bound_mode_ = BoundMode::Clip;
        // empty / unknown -> keep current (default: Random, as in the reference)
    }

    debug_lsrtde_ = mc.getInt("debug_lsrtde", debug_lsrtde_);

    // Per-method end-of-run local refinement overrides the [global] default
    // (setEndLocalFromGlobal() is called before configure(), see main.cpp).
    {
        auto it = mc.kv.find("end_local_refine");
        if (it != mc.kv.end()) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
        auto im = mc.kv.find("end_local_method");
        if (im != mc.kv.end()) end_local_method_ = lsrtdeLower(lsrtdeTrim(im->second));
    }
    {
        const std::string m = lsrtdeLower(lsrtdeTrim(end_local_method_));
        if (m.empty() || m == "none" || m == "off" || m == "0") end_local_refine_ = false;
    }
}

// -----------------------------------------------------------------------------
// small helpers
// -----------------------------------------------------------------------------
double LSRTDE::eval(const Vec& x)
{
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = 1e100;
    return f;
}

int LSRTDE::randInt(int n)
{
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> d(0, n - 1);
    return d(rng_);
}

double LSRTDE::randU01()
{
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng_);
}

// [ref: do F = NormRand(meanF,sigmaF); while(F < 0.0 || F > 1.0);]
double LSRTDE::sampleF(double meanF)
{
    if (!(sigma_f_ > 0.0)) return std::min(1.0, std::max(0.0, meanF));  // sigma 0: deterministic
    std::normal_distribution<double> nd(meanF, sigma_f_);
    for (int t = 0; t < 1000; ++t) {
        const double F = nd(rng_);
        if (F >= 0.0 && F <= 1.0) return F;
    }
    // only reachable with pathological user settings (e.g. f_base far outside [0,1])
    return std::min(1.0, std::max(0.0, meanF));
}

// [ref: Cr = NormRand(MemoryCr[k],0.05); Cr = min(max(Cr,0.0),1.0);]
double LSRTDE::sampleCR(double mcr)
{
    if (!(sigma_cr_ > 0.0)) return std::min(1.0, std::max(0.0, mcr));   // sigma 0: deterministic
    std::normal_distribution<double> nd(mcr, sigma_cr_);
    double cr = nd(rng_);
    if (!std::isfinite(cr)) cr = mcr;
    return std::min(1.0, std::max(0.0, cr));
}

// [ref: if(Trial[j] < Left || Trial[j] > Right) Trial[j] = Random(Left,Right);]
double LSRTDE::repairComponent(int j, double v, double parent_j)
{
    const double lo = prob_->lb()[j];
    const double hi = prob_->ub()[j];
    if (std::isfinite(v) && v >= lo && v <= hi) return v;

    switch (bound_mode_) {
        case BoundMode::Midpoint:
            if (!std::isfinite(v)) return 0.5 * (parent_j + 0.5 * (lo + hi));
            return (v < lo) ? 0.5 * (parent_j + lo) : 0.5 * (parent_j + hi);
        case BoundMode::Clip:
            if (!std::isfinite(v)) return parent_j;
            return (v < lo) ? lo : hi;
        case BoundMode::Random:
        default: {
            std::uniform_real_distribution<double> d(lo, hi);
            return (hi > lo) ? d(rng_) : lo;
        }
    }
}

// [ref: Optimizer::MeanWL] weighted Lehmer mean sum(w*x^2)/sum(w*x).
// The reference returns 1.0 when sum(w*x) ~ 0, and -- because of the
// division 0/0 -> NaN -- also when all weights are zero; reproduced here.
double LSRTDE::weightedLehmerMean(const std::vector<double>& values,
                                  const std::vector<double>& weights) const
{
    const size_t n = std::min(values.size(), weights.size());
    if (n == 0) return 1.0;
    double sumW = 0.0;
    for (size_t i = 0; i < n; ++i) sumW += weights[i];
    if (!(sumW > 0.0) || !std::isfinite(sumW)) return 1.0;

    double sumSq = 0.0, sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = weights[i] / sumW;
        sumSq += w * values[i] * values[i];
        sum   += w * values[i];
    }
    if (std::fabs(sum) > 1e-8 && std::isfinite(sumSq)) return sumSq / sum;
    return 1.0;
}

// [ref: newNIndsFront = int(double(4-NIndsFrontMax)/double(MaxFEval)*NFEval + NIndsFrontMax);]
int LSRTDE::targetPopulationSize() const
{
    const double nfe     = static_cast<double>(prob_->calls());
    const double nfe_max = static_cast<double>(std::max<long long>(1, max_evals_));
    const double v = static_cast<double>(np_min_ - n_init_) / nfe_max * nfe + static_cast<double>(n_init_);
    int n = static_cast<int>(v);                       // truncation, as int() in the reference
    const int cur = static_cast<int>(front_X_.size());
    if (n < np_min_) n = np_min_;
    if (n > cur)     n = cur;
    return n;
}

// [ref: Optimizer::RemoveWorst] repeatedly removes the (first) worst member
// of the front, shifting the rest -> the relative order of the remaining
// members (which the round-robin pointer relies on) is preserved.
void LSRTDE::removeWorstFromFront(int new_size)
{
    while (static_cast<int>(front_X_.size()) > new_size && !front_X_.empty()) {
        size_t worst = 0;
        for (size_t i = 1; i < front_F_.size(); ++i)
            if (front_F_[i] > front_F_[worst]) worst = i;
        front_X_.erase(front_X_.begin() + static_cast<std::ptrdiff_t>(worst));
        front_F_.erase(front_F_.begin() + static_cast<std::ptrdiff_t>(worst));
    }
}

// [ref: end of generation, qSort2int(FitArr,...) + copy of the best NIndsFront]
void LSRTDE::sortAndTruncateMain(int new_size)
{
    std::vector<int> idx;
    sortedOrder(main_F_, static_cast<int>(main_F_.size()), idx);
    const int keep = std::min(new_size, static_cast<int>(idx.size()));

    std::vector<Vec>    nx; nx.reserve(static_cast<size_t>(keep));
    std::vector<double> nf; nf.reserve(static_cast<size_t>(keep));
    for (int i = 0; i < keep; ++i) {
        nx.push_back(std::move(main_X_[static_cast<size_t>(idx[i])]));
        nf.push_back(main_F_[static_cast<size_t>(idx[i])]);
    }
    main_X_.swap(nx);
    main_F_.swap(nf);
}

// indices 0..n-1 ordered by ascending fitness (identity if all equal,
// exactly like the reference's "if(minfit != maxfit) qSort2int(...)")
void LSRTDE::sortedOrder(const std::vector<double>& f, int n, std::vector<int>& idx) const
{
    idx.resize(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return f[static_cast<size_t>(a)] < f[static_cast<size_t>(b)]; });
}

// -----------------------------------------------------------------------------
// init  [ref: Optimizer::Initialize + first part of MainCycle]
// -----------------------------------------------------------------------------
void LSRTDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    // population: explicit override, otherwise the reference default 20*D
    int N = 0;
    if (pop_override_ > 0) N = pop_override_;
    else                   N = static_cast<int>(std::llround(np_init_mult_ * static_cast<double>(D)));
    if (N < np_min_) N = np_min_;

    n_init_ = N;
    setPopulation(N);

    main_X_.clear(); main_F_.clear();
    front_X_.clear(); front_F_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    main_X_ = initSampler.samplePopulation(*prob_, rng_, N);
    main_X_.resize(static_cast<size_t>(N), Vec(static_cast<size_t>(D), 0.0)); // defensive
    main_F_.assign(static_cast<size_t>(N), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(static_cast<size_t>(D), 0.0);

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;
        main_F_[static_cast<size_t>(i)] = eval(main_X_[static_cast<size_t>(i)]);
        if (main_F_[static_cast<size_t>(i)] < best_f_) {
            best_f_ = main_F_[static_cast<size_t>(i)];
            best_x_ = main_X_[static_cast<size_t>(i)];
        }
    }
    if (!std::isfinite(best_f_) && !main_X_.empty()) best_x_ = main_X_[0];

    // [ref: PopulFront = Popul sorted by fitness]
    std::vector<int> idx;
    sortedOrder(main_F_, N, idx);
    front_X_.reserve(static_cast<size_t>(N));
    front_F_.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        front_X_.push_back(main_X_[static_cast<size_t>(idx[i])]);
        front_F_.push_back(main_F_[static_cast<size_t>(idx[i])]);
    }

    MCR_.assign(static_cast<size_t>(memory_size_), mcr_init_);   // [ref: MemoryCr[i] = 1.0]
    memory_iter_  = 0;
    pf_index_     = 0;                                           // [ref: PFIndex = 0]
    success_rate_ = sr_init_;                                    // [ref: SuccessRate = 0.5]
    generation_   = 0;

    if (debug_lsrtde_) {
        std::ostringstream os;   // own stream: printBest() leaves std::fixed on out()
        os << "[lsrtde] init -> D=" << D << ", N=" << N << ", np_min=" << np_min_
              << ", H=" << memory_size_ << ", mcr_init=" << mcr_init_
              << ", bound=" << (bound_mode_ == BoundMode::Random ? "random"
                               : bound_mode_ == BoundMode::Midpoint ? "midpoint" : "clip")
              << ", delta_from_parent=" << (delta_from_parent_ ? 1 : 0)
              << ", max_evals=" << max_evals_ << "\n";
        out() << os.str();
    }

    printBest();
}

// -----------------------------------------------------------------------------
// one generation  [ref: body of "while(NFEval < MaxFEval)" in MainCycle]
// -----------------------------------------------------------------------------
void LSRTDE::one_iteration()
{
    if (!prob_ || front_X_.empty()) return;

    const int D  = prob_->dimension();
    const int NF = static_cast<int>(front_X_.size());        // NIndsFront

    // Both populations always have the same size at the start of a generation.
    if (NF < 4 || static_cast<int>(main_X_.size()) != NF) {
        printBest();
        updateStop(front_F_);
        return;
    }

    // [ref: meanF = 0.4 + tanh(SuccessRate*5)*0.25; sigmaF = 0.02]
    const double meanF = f_base_ + f_amp_ * std::tanh(success_rate_ * f_sr_coef_);

    // [ref: Indices  = argsort(FitArr)      -> main population ranking]
    // [ref: Indices2 = argsort(FitArrFront) -> front ranking]
    std::vector<int> idx_main, idx_front;
    sortedOrder(main_F_,  NF, idx_main);
    sortedOrder(front_F_, NF, idx_front);

    // [ref: FitTemp2[i] = exp(-double(i)/double(NIndsFront)*3)]
    std::vector<double> rank_w(static_cast<size_t>(NF));
    for (int i = 0; i < NF; ++i)
        rank_w[static_cast<size_t>(i)] = std::exp(-static_cast<double>(i) / static_cast<double>(NF) * rank_pressure_);
    std::discrete_distribution<int> rank_dist(rank_w.begin(), rank_w.end());

    // [ref: psizeval = max(2, int(NIndsFront*0.7*exp(-SuccessRate*7)))]
    int psize = static_cast<int>(static_cast<double>(NF) * pbest_rate_ * std::exp(-success_rate_ * pbest_sr_coef_));
    psize = std::max(pbest_min_, psize);
    psize = std::min(psize, NF);

    // capacity for up to NF appended successes: no reallocation inside the loop
    main_X_.reserve(static_cast<size_t>(2 * NF));
    main_F_.reserve(static_cast<size_t>(2 * NF));

    std::vector<double> S_cr;    S_cr.reserve(static_cast<size_t>(NF));     // tempSuccessCr
    std::vector<double> S_delta; S_delta.reserve(static_cast<size_t>(NF));  // FitDelta
    int successes = 0;                                                     // SuccessFilled

    Vec trial(static_cast<size_t>(D));

    for (int it = 0; it < NF; ++it) {
        if (prob_->calls() >= max_evals_) break;

        // [ref: TheChosenOne = IntRandom(NIndsFront); MemoryCurrentIndex = IntRandom(MemorySize)]
        const int c = randInt(NF);
        const int k = randInt(memory_size_);

        // Index exclusions are reproduced exactly as in the reference (they
        // compare indices of the main and of the front population).
        // Termination is guaranteed: psize >= 2, NF >= 4.
        int prand, r1, r2;
        do { prand = idx_main[static_cast<size_t>(randInt(psize))]; } while (prand == c);
        do { r1 = idx_front[static_cast<size_t>(rank_dist(rng_))]; }  while (r1 == prand);
        do { r2 = idx_main[static_cast<size_t>(randInt(NF))]; }        while (r2 == prand || r2 == r1);

        const double F  = sampleF(meanF);
        const double Cr = sampleCR(MCR_[static_cast<size_t>(k)]);

        const Vec& xc = front_X_[static_cast<size_t>(c)];
        const Vec& xp = main_X_[static_cast<size_t>(prand)];
        const Vec& x1 = front_X_[static_cast<size_t>(r1)];
        const Vec& x2 = main_X_[static_cast<size_t>(r2)];

        // [ref: r-new-to-ptop/n/t mutation + binomial crossover, ActualCr]
        const int jrand = randInt(D);
        int crossed = 0;
        for (int j = 0; j < D; ++j) {
            const size_t sj = static_cast<size_t>(j);
            if (randU01() < Cr || j == jrand) {
                const double v = xc[sj] + F * (xp[sj] - xc[sj]) + F * (x1[sj] - x2[sj]);
                trial[sj] = repairComponent(j, v, xc[sj]);
                ++crossed;
            } else {
                trial[sj] = xc[sj];
            }
        }
        const double actualCr = static_cast<double>(crossed) / static_cast<double>(D);

        const double ft = eval(trial);

        // [ref: if(TempFit <= FitArrFront[TheChosenOne])]
        if (ft <= front_F_[static_cast<size_t>(c)]) {
            const double parent_f = front_F_[static_cast<size_t>(c)];

            // successful trial joins the main population ...
            main_X_.push_back(trial);
            main_F_.push_back(ft);
            // ... and overwrites the front in round-robin order (NOT its parent)
            front_X_[static_cast<size_t>(pf_index_)] = trial;
            front_F_[static_cast<size_t>(pf_index_)] = ft;

            if (ft < best_f_) { best_f_ = ft; best_x_ = trial; }

            S_cr.push_back(actualCr);
            // reference: computed after the overwrite above (0 if pf_index_ == c)
            const double delta = delta_from_parent_
                                   ? std::fabs(parent_f - ft)
                                   : std::fabs(front_F_[static_cast<size_t>(c)] - ft);
            S_delta.push_back(delta);
            ++successes;
            pf_index_ = (pf_index_ + 1) % NF;
        }
    }

    // [ref: SuccessRate = double(SuccessFilled)/double(NIndsFront)]
    success_rate_ = static_cast<double>(successes) / static_cast<double>(NF);

    // [ref: LPSR + RemoveWorst on the front]
    const int newN = targetPopulationSize();
    removeWorstFromFront(newN);
    if (pf_index_ >= newN) pf_index_ %= newN;              // difference (d)

    // [ref: UpdateMemoryCr: M = 0.5*(MeanWL(S_CR, delta) + M)]
    if (successes > 0) {
        const double mwl = weightedLehmerMean(S_cr, S_delta);
        double& m = MCR_[static_cast<size_t>(memory_iter_)];
        m = memory_c_ * m + (1.0 - memory_c_) * mwl;
        m = std::min(1.0, std::max(0.0, m));
        memory_iter_ = (memory_iter_ + 1) % memory_size_;
    }

    // [ref: main population = best newN of (old main + successful trials)]
    sortAndTruncateMain(newN);

    setPopulation(static_cast<int>(front_X_.size()));
    ++generation_;

    if (debug_lsrtde_ > 1) {
        const int last = (memory_iter_ + memory_size_ - 1) % memory_size_;
        std::ostringstream os;
        os << "[lsrtde] gen " << generation_ << " N=" << NF << "->" << newN
           << " SR=" << std::setprecision(4) << success_rate_
           << " meanF=" << meanF << " psize=" << psize
           << " MCR[" << last << "]=" << MCR_[static_cast<size_t>(last)] << "\n";
        out() << os.str();
    }

    printBest();
    updateStop(front_F_);
}

// -----------------------------------------------------------------------------
// optional end-of-run local refinement (framework feature, not in the paper)
// -----------------------------------------------------------------------------
void LSRTDE::end()
{
    if (!end_local_refine_ || !prob_ || end_local_method_.empty() || best_x_.empty()) return;
    std::pair<Vec, double> loc = localSearch(end_local_method_, best_x_);
    if (std::isfinite(loc.second) && loc.second < best_f_) {
        best_x_ = loc.first;
        best_f_ = loc.second;
    }
    printBest();
}

} // namespace optimsolution
