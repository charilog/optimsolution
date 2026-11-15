// =============================================================================
//  IMSPSO-DNI -- see imspsodni.h for the reference and the full description.
//  Comments [Eq. n] / [Alg. a, line l] point to the manuscript.
// =============================================================================

#include "imspsodni.h"
#include "init.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cctype>
#include <sstream>

namespace optimsolution {

namespace {
std::string imsLower(std::string s)
{
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string imsTrim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}
} // namespace

// -----------------------------------------------------------------------------
// configuration (keys are lower-case)
// -----------------------------------------------------------------------------
void IMSPSODNI::configure(const MethodConfig& mc)
{
    {
        const int p = mc.getInt("population", 0);
        pop_cfg_ = (p > 0) ? p : 30;                       // Sec. 4.2: swarm size 30
        if (pop_cfg_ < 4) pop_cfg_ = 4;
    }
    w_max_    = mc.getDbl("w_max", w_max_);
    w_min_    = mc.getDbl("w_min", w_min_);
    c1_start_ = mc.getDbl("c1_start", c1_start_);
    c1_end_   = mc.getDbl("c1_end", c1_end_);
    c2_start_ = mc.getDbl("c2_start", c2_start_);
    c2_end_   = mc.getDbl("c2_end", c2_end_);
    phi_      = mc.getDbl("phi", phi_);
    alpha_    = std::min(1.0, std::max(0.0, mc.getDbl("alpha", alpha_)));
    gamma_    = std::min(1.0, std::max(0.0, mc.getDbl("gamma", gamma_)));
    eps_      = std::min(1.0, std::max(0.0, mc.getDbl("epsilon", eps_)));
    gl_       = std::max(1, mc.getInt("gl", gl_));
    T_        = mc.getDbl("t", T_);
    T_        = mc.getDbl("diversity_threshold", T_);
    R_        = mc.getDbl("r", R_);
    vmax_frac_ = mc.getDbl("vmax_frac", vmax_frac_);
    if (!(vmax_frac_ > 0.0)) vmax_frac_ = 0.5;
    gc_reset_on_improve_ = mc.getBool("gc_reset_on_improve", gc_reset_on_improve_);
    accept_equal_        = mc.getBool("accept_equal", accept_equal_);
    {
        const std::string v = imsLower(imsTrim(mc.getStr("v_init", "")));
        if (v == "zero") v_init_random_ = false;
        else if (v == "random") v_init_random_ = true;
    }
    {
        // subgroup number candidates, e.g. "2,4,6,8,10"
        const std::string s = mc.getStr("subgroups", "");
        if (!imsTrim(s).empty()) {
            std::vector<int> v;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok = imsTrim(tok);
                if (tok.empty()) continue;
                try { const int k = std::stoi(tok); if (k >= 2) v.push_back(k); } catch (...) {}
            }
            if (!v.empty()) S_ = v;
        }
    }
    debug_ = mc.getInt("debug_imspsodni", debug_);

    auto it = mc.kv.find("end_local_refine");
    if (it != mc.kv.end()) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    auto im = mc.kv.find("end_local_method");
    if (im != mc.kv.end()) end_local_method_ = imsLower(imsTrim(im->second));
    const std::string m = imsLower(imsTrim(end_local_method_));
    if (m.empty() || m == "none" || m == "off" || m == "0") end_local_refine_ = false;
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------
double IMSPSODNI::eval(const Vec& x)
{
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = 1e100;
    return f;
}

double IMSPSODNI::U01()
{
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng_);
}

int IMSPSODNI::randInt(int n)
{
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> d(0, n - 1);
    return d(rng_);
}

void IMSPSODNI::clampVelocity(Vec& v) const
{
    for (size_t j = 0; j < v.size(); ++j) {
        if (!std::isfinite(v[j])) v[j] = 0.0;
        if (v[j] >  vmax_[j]) v[j] =  vmax_[j];
        if (v[j] < -vmax_[j]) v[j] = -vmax_[j];
    }
}

void IMSPSODNI::clipPosition(Vec& x) const
{
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void IMSPSODNI::noteGbest(const Vec& x, double f)
{
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
}

// [Alg. 1, lines 3-7 / Eq. 7] epsilon-greedy with epsilon = probability of
// EXPLOITING (the manuscript: "if rand < epsilon then Argmax"); ties random.
int IMSPSODNI::chooseAction(int state)
{
    const int n = static_cast<int>(S_.size());
    if (U01() < eps_) {
        double best = -std::numeric_limits<double>::infinity();
        std::vector<int> arg;
        for (int a = 0; a < n; ++a) {
            const double q = Q_[static_cast<size_t>(state)][static_cast<size_t>(a)];
            if (q > best) { best = q; arg.assign(1, a); }
            else if (q == best) arg.push_back(a);
        }
        return arg[static_cast<size_t>(randInt(static_cast<int>(arg.size())))];
    }
    return randInt(n);
}

// [Eq. 6] Q(i,j) = Q(i,j) + alpha*(r_ij + gamma*max(Q(j)) - Q(i,j))
void IMSPSODNI::updateQ(int s, int a, double reward)
{
    double mx = -std::numeric_limits<double>::infinity();
    for (double q : Q_[static_cast<size_t>(a)]) mx = std::max(mx, q);
    double& q = Q_[static_cast<size_t>(s)][static_cast<size_t>(a)];
    q += alpha_ * (reward + gamma_ * mx - q);
}

// [Eq. 11] D(t) = (1/N) sum_i sqrt( sum_d (X_i^d - mean^d)^2 )
double IMSPSODNI::diversity() const
{
    const int N = static_cast<int>(X_.size());
    if (N == 0) return 0.0;
    const size_t D = X_[0].size();
    Vec c(D, 0.0);
    for (const Vec& x : X_) for (size_t j = 0; j < D; ++j) c[j] += x[j];
    for (size_t j = 0; j < D; ++j) c[j] /= N;
    double s = 0.0;
    for (const Vec& x : X_) {
        double d2 = 0.0;
        for (size_t j = 0; j < D; ++j) { const double d = x[j] - c[j]; d2 += d * d; }
        s += std::sqrt(d2);
    }
    return s / N;
}

// -----------------------------------------------------------------------------
// init  [Alg. 4, line 1; Fig. 1 "Initialize parameters and Q-table",
//        "Compute fitness and update pbest, gbest"]
// -----------------------------------------------------------------------------
void IMSPSODNI::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = pop_cfg_;
    setPopulation(N);

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    vmax_.assign(static_cast<size_t>(D), 0.0);
    for (int j = 0; j < D; ++j) {
        double r = U[j] - L[j];
        if (!std::isfinite(r) || r <= 0.0) r = 1.0;
        vmax_[j] = vmax_frac_ * r;
    }

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);
    X_.resize(static_cast<size_t>(N), Vec(static_cast<size_t>(D), 0.0));
    V_.assign(static_cast<size_t>(N), Vec(static_cast<size_t>(D), 0.0));
    if (v_init_random_)
        for (auto& v : V_) for (int j = 0; j < D; ++j) v[j] = (2.0 * U01() - 1.0) * vmax_[j];
    F_.assign(static_cast<size_t>(N), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_ = X_[0];
    for (int i = 0; i < N; ++i) {
        clipPosition(X_[static_cast<size_t>(i)]);
        if (prob_->calls() >= max_evals_) break;
        F_[static_cast<size_t>(i)] = eval(X_[static_cast<size_t>(i)]);
        noteGbest(X_[static_cast<size_t>(i)], F_[static_cast<size_t>(i)]);
    }
    P_  = X_;
    PF_ = F_;

    const size_t nS = S_.size();
    Q_.assign(nS, std::vector<double>(nS, 0.0));          // "All Q-values are initially set to 0"
    state_ = randInt(static_cast<int>(nS));
    gc_ = 0;

    if (debug_) {
        std::ostringstream os;
        os << "[imspsodni] init -> D=" << D << ", N=" << N << ", S={";
        for (size_t k = 0; k < nS; ++k) os << (k ? "," : "") << S_[k];
        os << "}, eps=" << eps_ << ", alpha=" << alpha_ << ", gamma=" << gamma_
           << ", phi=" << phi_ << ", gl=" << gl_ << ", T=" << T_ << ", R=" << R_ << "\n";
        out() << os.str();
    }
    printBest();
}

// -----------------------------------------------------------------------------
// one iteration  [Alg. 4, lines 3-20]
// -----------------------------------------------------------------------------
void IMSPSODNI::one_iteration()
{
    if (!prob_ || X_.empty()) return;
    const int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();

    // [Eq. 5] and Sec. 4.2: linear schedules (iteration fraction = budget fraction)
    const double t  = std::min(1.0, std::max(0.0, progressFraction()));
    const double w  = w_max_ - (w_max_ - w_min_) * t;
    const double c1 = c1_start_ + (c1_end_ - c1_start_) * t;
    const double c2 = c2_start_ + (c2_end_ - c2_start_) * t;

    const double gbest_prev = best_f_;                     // gbest_val_{t-1}

    // [Alg. 4 line 3 / Alg. 1] next action = number of subgroups
    const int action = chooseAction(state_);
    const int K = std::max(2, std::min(S_[static_cast<size_t>(action)], N));

    // [Alg. 4 line 4] sort by pbest fitness and divide into K subgroups
    std::vector<int> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return PF_[static_cast<size_t>(a)] < PF_[static_cast<size_t>(b)]; });
    std::vector<int> start(static_cast<size_t>(K) + 1, 0);  // subgroup g = order[start[g-1] .. start[g]-1]
    {
        const int base = N / K, extra = N % K;              // uneven: larger subgroups first
        for (int g = 0; g < K; ++g) start[static_cast<size_t>(g) + 1] = start[static_cast<size_t>(g)] + base + (g < extra ? 1 : 0);
    }
    auto member = [&](int g, int k) { return order[static_cast<size_t>(start[static_cast<size_t>(g) - 1] + k)]; };
    auto gsize  = [&](int g) { return start[static_cast<size_t>(g)] - start[static_cast<size_t>(g) - 1]; };

    // synchronous update: exemplars/candidates are read from the start of the iteration
    const std::vector<Vec>    Xs = X_;
    const std::vector<double> Fs = F_;

    std::vector<char> moved(static_cast<size_t>(N), 0);

    // [Alg. 4 lines 5-11 / Eq. 10] subgroups g = 3..K
    for (int g = 3; g <= K; ++g) {
        const int pool = start[static_cast<size_t>(g) - 1];          // particles of subgroups 1..g-1
        const int n1   = gsize(1);
        for (int k = 0; k < gsize(g); ++k) {
            const int i = member(g, k);
            // [Alg. 4 lines 7-8] two random candidates from the smaller-index subgroups
            int a = order[static_cast<size_t>(randInt(pool))];
            int b = a;
            if (pool > 1) { do { b = order[static_cast<size_t>(randInt(pool))]; } while (b == a); }
            // a candidate worse than the current particle is replaced by a random particle of subgroup 1
            if (Fs[static_cast<size_t>(a)] > Fs[static_cast<size_t>(i)]) a = member(1, randInt(n1));
            if (Fs[static_cast<size_t>(b)] > Fs[static_cast<size_t>(i)]) b = member(1, randInt(n1));
            const Vec& xe1 = Xs[static_cast<size_t>(a)];
            const Vec& xe2 = Xs[static_cast<size_t>(b)];
            Vec& v = V_[static_cast<size_t>(i)];
            Vec& x = X_[static_cast<size_t>(i)];
            for (int j = 0; j < D; ++j) {
                const double r1 = U01(), r2 = U01(), r3 = U01();
                v[j] = r1 * v[j] + r2 * (xe1[j] - x[j]) + phi_ * r3 * (xe2[j] - x[j]);
            }
            clampVelocity(v);
            for (int j = 0; j < D; ++j) x[j] += v[j];               // [Eq. 2]
            clipPosition(x);
            moved[static_cast<size_t>(i)] = 1;
        }
    }

    // [Alg. 4 lines 12-15 / Eq. 9] subgroup 2
    {
        Vec g1mean(static_cast<size_t>(D), 0.0);                    // mean pbest of subgroup 1
        const int n1 = gsize(1);
        for (int k = 0; k < n1; ++k) {
            const Vec& p = P_[static_cast<size_t>(member(1, k))];
            for (int j = 0; j < D; ++j) g1mean[j] += p[j];
        }
        for (int j = 0; j < D; ++j) g1mean[j] /= n1;
        for (int k = 0; k < gsize(2); ++k) {
            const int i = member(2, k);
            Vec& v = V_[static_cast<size_t>(i)];
            Vec& x = X_[static_cast<size_t>(i)];
            for (int j = 0; j < D; ++j) {
                const double r1 = U01(), r2 = U01();
                v[j] = w * v[j] + c1 * r1 * (g1mean[j] - x[j]) + c2 * r2 * (best_x_[j] - x[j]);
            }
            clampVelocity(v);
            for (int j = 0; j < D; ++j) x[j] += v[j];
            clipPosition(x);
            moved[static_cast<size_t>(i)] = 1;
        }
    }
    // subgroup 1: not moved ("directly proceed to the next iteration")

    // [Alg. 4 line 16] evaluate and update pbest and gbest
    for (int i = 0; i < N; ++i) {
        if (!moved[static_cast<size_t>(i)]) continue;
        if (prob_->calls() >= max_evals_) break;
        const double f = eval(X_[static_cast<size_t>(i)]);
        F_[static_cast<size_t>(i)] = f;
        if (f < PF_[static_cast<size_t>(i)]) { PF_[static_cast<size_t>(i)] = f; P_[static_cast<size_t>(i)] = X_[static_cast<size_t>(i)]; }
        noteGbest(X_[static_cast<size_t>(i)], f);
    }

    // [Alg. 4 line 17 / Alg. 3] gPS-DNI
    if (best_f_ < gbest_prev) {
        if (gc_reset_on_improve_) gc_ = 0;                          // Alg. 3 line 23: g_c unchanged
    } else {
        ++gc_;                                                      // Alg. 3 line 4
        if (gc_ >= gl_ && prob_->calls() < max_evals_) {            // Alg. 3 lines 5-14
            const double Dv = diversity();                          // [Eq. 11]
            const double rad = (Dv > T_) ? Dv : R_ * Dv;            // [Eq. 12] / [Eq. 13]
            Vec gnew = best_x_;
            for (int j = 0; j < D; ++j) gnew[j] += rad * (2.0 * U01() - 1.0);
            clipPosition(gnew);
            const double fn = eval(gnew);
            gc_ = 0;
            // "When a newly generated gbest proves to be no inferior to its
            //  predecessor, the update is adopted" (Alg. 3 line 16 uses '<')
            if (accept_equal_ ? (fn <= best_f_) : (fn < best_f_)) { best_f_ = fn; best_x_ = gnew; }
        }
    }

    // [Alg. 4 lines 18-20 / Eq. 8 / Eq. 6]
    const double reward = (best_f_ < gbest_prev) ? 1.0 : 0.0;
    updateQ(state_, action, reward);
    state_ = action;

    if (debug_ > 1) {
        std::ostringstream os;
        os << "[imspsodni] K=" << K << " r=" << reward << " gc=" << gc_
           << " w=" << w << " c1=" << c1 << " c2=" << c2 << " gbest=" << best_f_ << "\n";
        out() << os.str();
    }

    printBest();
    updateStop(PF_);
}

// optional end-of-run local refinement (framework feature, not in the paper)
void IMSPSODNI::end()
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
