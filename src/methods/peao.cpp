#include "peao.h"
#include "init.h"

#include <limits>
#include <utility>

namespace optimsolution {

namespace {
constexpr double kPi = 3.1415926535897932384626433832795029;
} // namespace

void PEAO::clampToBounds(Vec& x) const {
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const size_t n = std::min(x.size(), L.size());
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        else if (x[j] < L[j]) x[j] = L[j];
        else if (x[j] > U[j]) x[j] = U[j];
    }
}

// Eq. (3): AF(t) = sqrt(t/T). iters_ (base class) counts COMPLETED
// iterations, so "t" for the iteration currently in progress is iters_+1,
// matching the original's `iteration++` at the top of step() before AF(t)
// is computed. Clamped to 1.0 once t>=T (the original has no clamp; this
// only matters if the run continues past maxIterations_ because
// optimsolution's own budget/stop-rule hasn't triggered yet).
double PEAO::adaptiveFactor() const {
    const double t = double(iters_ + 1);
    const double T = double(maxIterations_);
    if (T <= 0.0) return 0.0;
    return std::min(1.0, std::sqrt(t / T));
}

// Eq. (6) auxiliary sigma formula.
double PEAO::levySigma(double beta) const {
    const double numerator = std::tgamma(1.0 + beta) * std::sin(kPi * beta / 2.0);
    const double denominator = std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0);
    return std::pow(numerator / denominator, 1.0 / beta);
}

// Eq. (6). Uses the base class's seeded rng_ (not a fresh
// random_device-seeded generator) so runs are reproducible.
double PEAO::levyScalar(double beta) {
    const double sigma = levySigma(beta);
    std::normal_distribution<double> distMu(0.0, sigma);
    std::normal_distribution<double> distV(0.0, 1.0);

    const double mu = distMu(rng_);
    double v = distV(rng_);
    if (std::fabs(v) < 1e-12) v = (v >= 0.0) ? 1e-12 : -1e-12;

    return mu / std::pow(std::fabs(v), 1.0 / beta);
}

Vec PEAO::levyVector(int dim, double beta) {
    Vec values(dim);
    for (int i = 0; i < dim; ++i) values[i] = levyScalar(beta);
    return values;
}

void PEAO::updateLeadersForSubpop(int subPopIndex) {
    const int n = static_cast<int>(X_.size());
    const int start = subPopIndex * subSize_;
    const int end = (subPopIndex == subPopulation_ - 1) ? n : std::min(start + subSize_, n);
    if (end <= start) return;

    double localBest = std::numeric_limits<double>::max();
    double localWorst = -std::numeric_limits<double>::max();
    int localBestIdx = start, localWorstIdx = start;

    for (int i = start; i < end; ++i) {
        const double f = FX_[i];
        if (f < localBest) { localBest = f; localBestIdx = i; }
        if (f > localWorst) { localWorst = f; localWorstIdx = i; }
    }

    bestF_[subPopIndex] = localBest;
    worstF_[subPopIndex] = localWorst;
    bestIdx_[subPopIndex] = localBestIdx;
    worstIdx_[subPopIndex] = localWorstIdx;
    bestX_[subPopIndex] = X_[localBestIdx];
    worstX_[subPopIndex] = X_[localWorstIdx];
}

void PEAO::updateAllLeaders() {
    for (int k = 0; k < subPopulation_; ++k) updateLeadersForSubpop(k);
    for (int k = 0; k < subPopulation_; ++k) {
        if (bestF_[k] < gBestF_) {
            gBestF_ = bestF_[k];
            gBestX_ = bestX_[k];
        }
    }
}

// Eq. (2). *** BUGFIX vs. the original source ***: the given formula was
// `c[j] = (gbest[j]-current[j]) + ri*sin(AFt*current[j])` -- i.e. it never
// added `current[j]` as a base, unlike createCandidate2A/2B which both
// correctly compute `current[j] + step`. Taken literally, that makes
// candidate1 a raw DIFFERENCE vector rather than a position: for any
// reasonably sized domain, |gbest[j]-current[j]| routinely dominates the
// sine term, so the "candidate" almost always lands far outside [lb, ub]
// and gets clamped to a near-random corner in clampToBounds() -- wasting
// roughly a third of the evaluation budget on a candidate that carries no
// real information about `current` and is essentially never selected.
// Adding the missing `current[j] +` restores the same "current + step"
// structure as 2A/2B (this simplifies to gbest[j] + ri*sin(AFt*current[j]):
// jump toward the global best, perturbed by a sine term that still depends
// on the individual's own current position).
Vec PEAO::createCandidate1(const Vec& current, const Vec& gbest, double AFt) {
    Vec c(dimension_);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double ri = U01(rng_);
    for (int j = 0; j < dimension_; ++j) {
        c[j] = current[j] + (gbest[j] - current[j]) + ri * std::sin(AFt * current[j]);
    }
    return c;
}

// Eq. (5).
Vec PEAO::buildDifferenceVector() {
    Vec diff(dimension_, 0.0);
    const int n = static_cast<int>(X_.size());
    if (n <= 1) return diff;

    std::uniform_int_distribution<int> Un(0, n - 1);
    int p = Un(rng_);
    int q = Un(rng_);
    for (int guard = 0; q == p && guard < 50; ++guard) q = Un(rng_);

    for (int j = 0; j < dimension_; ++j) diff[j] = X_[p][j] - X_[q][j];
    return diff;
}

// Eq. (4): per-dimension Levy step AND per-dimension random enzyme
// coefficient scA.
Vec PEAO::createCandidate2A(const Vec& current, const Vec& gbest, const Vec& diff, double AFt) {
    Vec c(dimension_);
    const double beta = levyBeta();
    const Vec levyStep = levyVector(dimension_, beta);
    std::uniform_real_distribution<double> Usc(enzymeConcentration_, 1.0);

    for (int j = 0; j < dimension_; ++j) {
        const double scA = Usc(rng_);
        c[j] = current[j] + levyStep[j] * diff[j] + AFt * scA * (gbest[j] - current[j]);
    }
    return c;
}

// Eq. (7): a single scalar Levy step AND a single scB, shared across all
// dimensions (unlike candidate 2A's per-dimension values).
Vec PEAO::createCandidate2B(const Vec& current, const Vec& gbest, const Vec& diff, double AFt) {
    Vec c(dimension_);
    const double beta = levyBeta();
    const double levyStep = levyScalar(beta);
    std::uniform_real_distribution<double> Usc(enzymeConcentration_, 1.0);
    const double scB = Usc(rng_);

    for (int j = 0; j < dimension_; ++j) {
        c[j] = current[j] + levyStep * diff[j] + AFt * scB * (gbest[j] - current[j]);
    }
    return c;
}

// Communication strategy 1: Optimal Replacement -- every subpop's worst
// individual is overwritten by the global best.
void PEAO::communicationStrategy1() {
    for (int g = 0; g < subPopulation_; ++g) {
        const int idx = worstIdx_[g];
        if (idx < 0 || idx >= static_cast<int>(X_.size())) continue;
        X_[idx] = gBestX_;
        FX_[idx] = gBestF_;
    }
}

// Communication strategy 2: Optimal Mean Replacement -- every subpop's
// worst individual is overwritten by the (re-evaluated) mean of all
// subpopulations' best positions.
void PEAO::communicationStrategy2() {
    Vec meanBest(dimension_, 0.0);
    for (int g = 0; g < subPopulation_; ++g)
        for (int j = 0; j < dimension_; ++j)
            meanBest[j] += bestX_[g][j];
    if (subPopulation_ > 0) {
        for (int j = 0; j < dimension_; ++j) meanBest[j] /= double(subPopulation_);
    }
    clampToBounds(meanBest);
    const double meanBestFitness = eval(meanBest);

    for (int g = 0; g < subPopulation_; ++g) {
        const int idx = worstIdx_[g];
        if (idx < 0 || idx >= static_cast<int>(X_.size())) continue;
        X_[idx] = meanBest;
        FX_[idx] = meanBestFitness;
    }
}

// Communication strategy 3: Circular Optimal Replacement -- each subpop's
// worst individual is overwritten by the PREVIOUS subpop's best (using the
// best-of-record captured before this generation's swaps, exactly as in
// the original).
void PEAO::communicationStrategy3() {
    if (subPopulation_ <= 1) return;
    const std::vector<Vec> oldBestX = bestX_;
    const std::vector<double> oldBestF = bestF_;

    for (int g = 0; g < subPopulation_; ++g) {
        const int prevGroup = (g - 1 + subPopulation_) % subPopulation_;
        const int idx = worstIdx_[g];
        if (idx < 0 || idx >= static_cast<int>(X_.size())) continue;
        X_[idx] = oldBestX[prevGroup];
        FX_[idx] = oldBestF[prevGroup];
    }
}

void PEAO::communicate() {
    int strategy = communicationStrategyMode_;
    if (strategy == 0) {
        std::uniform_int_distribution<int> U3(1, 3);
        strategy = U3(rng_);
    }
    if (strategy == 1) communicationStrategy1();
    else if (strategy == 2) communicationStrategy2();
    else communicationStrategy3();

    updateAllLeaders();
}

void PEAO::propagate() {
    std::uniform_int_distribution<int> Usub(0, subPopulation_ - 1);

    if (propagationMethod_ == "1to1") {
        const int sender = Usub(rng_);
        int receiver = sender;
        for (int guard = 0; receiver == sender && guard < 50; ++guard) receiver = Usub(rng_);
        propagateBetween(sender, receiver);
    } else if (propagationMethod_ == "1toN") {
        const int sender = Usub(rng_);
        for (int r = 0; r < subPopulation_; ++r) if (r != sender) propagateBetween(sender, r);
    } else if (propagationMethod_ == "Nto1") {
        const int receiver = Usub(rng_);
        for (int s = 0; s < subPopulation_; ++s) if (s != receiver) propagateBetween(s, receiver);
    } else if (propagationMethod_ == "NtoN") {
        for (int s = 0; s < subPopulation_; ++s)
            for (int r = 0; r < subPopulation_; ++r)
                if (s != r) propagateBetween(s, r);
    }
}

// NOTE: the original propagateBetween() computed each group's range with a
// plain std::min(start+sub_size, population) for BOTH sender and receiver,
// without the "last group absorbs the remainder" special-case that
// updateLeadersForSubpop() and the main loop both use elsewhere -- meaning
// it silently excluded any remainder individuals whenever population isn't
// evenly divisible by subPopulation_. Applied consistently here instead,
// so every part of this port agrees on which individuals belong to the
// last subpopulation.
void PEAO::propagateBetween(int sender, int receiver) {
    const int n = static_cast<int>(X_.size());
    const int senderStart = sender * subSize_;
    const int senderEnd = (sender == subPopulation_ - 1) ? n : std::min(senderStart + subSize_, n);
    const int receiverStart = receiver * subSize_;
    const int receiverEnd = (receiver == subPopulation_ - 1) ? n : std::min(receiverStart + subSize_, n);

    std::vector<std::pair<double, int>> senderFI;
    senderFI.reserve(senderEnd - senderStart);
    for (int i = senderStart; i < senderEnd; ++i) senderFI.push_back({FX_[i], i});
    std::sort(senderFI.begin(), senderFI.end());

    std::vector<std::pair<double, int>> receiverFI;
    receiverFI.reserve(receiverEnd - receiverStart);
    for (int i = receiverStart; i < receiverEnd; ++i) receiverFI.push_back({FX_[i], i});
    std::sort(receiverFI.begin(), receiverFI.end(), std::greater<>());

    for (int i = 0; i < propagationNumber_; ++i) {
        if (i >= static_cast<int>(senderFI.size()) || i >= static_cast<int>(receiverFI.size())) break;
        const int bestI = senderFI[i].second;
        const int worstI = receiverFI[i].second;
        X_[worstI] = X_[bestI];
        FX_[worstI] = FX_[bestI];
    }
}

void PEAO::init() {
    if (!prob_) return;
    dimension_ = prob_->dimension();

    if (subPopulation_ < 1) subPopulation_ = 1;
    if (pop_ < subPopulation_) pop_ = subPopulation_;
    subSize_ = pop_ / subPopulation_;
    if (subSize_ < 1) subSize_ = 1;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);
    // BUGFIX: unevaluated individuals must not look like f = 0 (best);
    // and the evaluation budget is respected during initialisation
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < X_.size() && prob_->calls() < max_evals_; ++i) FX_[i] = eval(X_[i]);

    bestF_.assign(subPopulation_, 0.0);
    worstF_.assign(subPopulation_, 0.0);
    bestIdx_.assign(subPopulation_, 0);
    worstIdx_.assign(subPopulation_, 0);
    bestX_.assign(subPopulation_, Vec(dimension_, 0.0));
    worstX_.assign(subPopulation_, Vec(dimension_, 0.0));
    bestFOld_.assign(subPopulation_, 0.0);

    gBestF_ = std::numeric_limits<double>::infinity();
    gBestX_.assign(dimension_, 0.0);

    // Eq. (1): initial sampling (done above via Initializer), then
    // establish the initial leaders.
    updateAllLeaders();

    best_f_ = gBestF_;
    best_x_ = gBestX_;

    updateStop(FX_);
    printBest();
}

void PEAO::one_iteration() {
    if (!prob_) return;

    for (int k = 0; k < subPopulation_; ++k) bestFOld_[k] = bestF_[k];

    const double AFt = adaptiveFactor();
    const int n = static_cast<int>(X_.size());

    for (int g = 0; g < subPopulation_; ++g) {
        const int start = g * subSize_;
        const int end = (g == subPopulation_ - 1) ? n : std::min(start + subSize_, n);

        for (int i = start; i < end; ++i) {
            const Vec currentPosition = X_[i];
            const double currentFitness = FX_[i];

            Vec candidate1 = createCandidate1(currentPosition, gBestX_, AFt);
            Vec diff = buildDifferenceVector();
            Vec candidate2A = createCandidate2A(currentPosition, gBestX_, diff, AFt);
            Vec candidate2B = createCandidate2B(currentPosition, gBestX_, diff, AFt);

            clampToBounds(candidate1);
            clampToBounds(candidate2A);
            clampToBounds(candidate2B);

            const double fitnessCandidate1 = eval(candidate1);
            const double fitnessCandidate2A = eval(candidate2A);
            const double fitnessCandidate2B = eval(candidate2B);

            Vec candidate2;
            double fitnessCandidate2;
            if (fitnessCandidate2A < fitnessCandidate2B) {
                candidate2 = std::move(candidate2A);
                fitnessCandidate2 = fitnessCandidate2A;
            } else {
                candidate2 = std::move(candidate2B);
                fitnessCandidate2 = fitnessCandidate2B;
            }

            Vec updatedCandidate;
            double updatedCandidateFitness;
            if (fitnessCandidate1 < fitnessCandidate2) {
                updatedCandidate = std::move(candidate1);
                updatedCandidateFitness = fitnessCandidate1;
            } else {
                updatedCandidate = std::move(candidate2);
                updatedCandidateFitness = fitnessCandidate2;
            }

            if (updatedCandidateFitness < currentFitness) {
                X_[i] = std::move(updatedCandidate);
                FX_[i] = updatedCandidateFitness;
            }
            if (FX_[i] < gBestF_) {
                gBestF_ = FX_[i];
                gBestX_ = X_[i];
            }

            if (prob_->calls() >= max_evals_) break;
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // *** BUGFIX vs. the original source ***: this was originally the last
    // statement INSIDE the g-loop above, so it fired once PER SUBPOPULATION
    // whenever propagationRate_ was reached -- subPopulation_ times in a
    // row on a single triggering iteration. For propagationMethod=NtoN
    // (already a full all-pairs exchange per call), that means the entire
    // population's elite-for-worst swap repeats subPopulation_ times back
    // to back, rapidly homogenizing every subpopulation and destroying
    // diversity. communicate() below is the closest analogous periodic
    // operator and is correctly called once per triggering iteration;
    // propagate() now matches that pattern.
    if (propEnable_ && ((iters_ + 1) % propagationRate_) == 0 && subPopulation_ > 1) {
        propagate();
    }

    if (communicationThreshold_ > 0 && ((iters_ + 1) % communicationThreshold_) == 0) {
        communicate();
    }

    if (local_rate_ > 0.0 && local_method_ != "none" && !local_method_.empty()) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            if (U01(rng_) < local_rate_) {
                auto [xl, fl] = localSearch(local_method_, X_[i]);
                if (fl < FX_[i]) { X_[i] = std::move(xl); FX_[i] = fl; }
            }
            if (prob_->calls() >= max_evals_) break;
        }
    }

    updateAllLeaders();
    best_f_ = gBestF_;
    best_x_ = gBestX_;

    updateStop(FX_);
    printBest();
}

void PEAO::end() {
    if (end_local_refine_ && !end_local_method_.empty()) {
        auto [xl, fl] = localSearch(end_local_method_, best_x_);
        if (fl < best_f_) {
            best_f_ = fl;
            best_x_ = xl;
        }
    }
    printBest();
}

} // namespace optimsolution
