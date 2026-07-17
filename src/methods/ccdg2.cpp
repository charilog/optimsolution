#include "ccdg2.h"
#include <numeric>
#include <limits>
#include <cstdio>

namespace optimsolution {

void CCDG2::configure(const MethodConfig& mc) {
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    grouping_budget_frac_ = mc.getDbl("grouping_budget_frac", grouping_budget_frac_);
    dg2_max_passes_       = mc.getInt("dg2_max_passes", dg2_max_passes_);
    group_pop_cfg_        = mc.getInt("group_pop", group_pop_cfg_);
    shade_H_              = mc.getInt("shade_H", shade_H_);
    archive_rate_         = mc.getDbl("archive_rate", archive_rate_);
    p_best_min_frac_      = mc.getDbl("p_best_frac", p_best_min_frac_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double CCDG2::safeEval(const Vec& xfull) {
    double f = prob_->evaluate(xfull);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = xfull;
    }
    return f;
}

// ============================================================================
// DG2 interaction test between dims i and j.
//
// p1, p1_pert are two FULL-D vectors supplied by the caller: p1 is a
// baseline point, p1_pert is p1 with ONLY dimension i perturbed (e.g. from
// lb_i to ub_i). f_p1 = f(p1), f_p1_pert = f(p1_pert) -- neither depends on
// j, so the caller computes them ONCE per seed i and reuses them across
// every j tested against that seed (DG2's stated efficiency contribution:
// "reuses the sample points... saves up to half the computational
// resources").
//
// This call performs the two NEW evaluations needed for a specific j:
// f(p2) and f(p2 with x_i perturbed), where p2 = p1 but with x_j moved to
// its opposite bound. If interacting, delta1 = f_p1 - f_p1_pert (effect of
// perturbing i with x_j at its ORIGINAL value) should equal delta2 =
// f(p2) - f(p2_pert) (same perturbation, x_j at its OTHER value) for a
// separable pair; a difference beyond the numerical noise floor indicates
// interaction.
//
// The noise floor (DG2's "epsilon") is derived from a standard
// backward-error-analysis bound for D sequential floating-point
// operations (the conservative, generic assumption DG2's authors use for
// a black-box objective of unknown internal complexity), rather than a
// hand-tuned constant -- this is DG2's key improvement over the original
// (2014) DG, which needed a manually-set threshold.
// ============================================================================
bool CCDG2::dg2Interacts(int i, int j, double f_p1, double f_p1_pert,
                          Vec& p1, Vec& p1_pert, double eps_D) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    // BUG FIX: p2/p2_pert used to reuse the SAME fixed corner value for
    // dimension j (previously U[j], mirroring how the seed's own
    // perturbation used fixed L/U corners throughout). For ANY coupling
    // term that is an even function of the perturbed variable -- e.g.
    // Rosenbrock's classic 100*(x_{k+1}-x_k^2)^2 -- fixed, symmetric
    // bounds (L=-c, U=+c, so L^2=U^2) make delta1 and delta2 CANCEL
    // EXACTLY regardless of true interaction, since the test only ever
    // probes the two endpoints of a parabola that takes the same value at
    // both. This produced systematic false negatives (interactions
    // completely missed) on essentially every benchmark in this framework,
    // since almost all of them use symmetric box bounds. Fixed by drawing
    // j's second probe value RANDOMLY (rather than always U[j]) -- for a
    // continuous random draw, hitting the exact degenerate value has
    // probability zero, so this class of cancellation can no longer occur
    // systematically. (Verified against a hand-built block-Rosenbrock test
    // problem with a known ground-truth grouping -- see delivery notes.)
    std::uniform_real_distribution<double> Uu(0.0, 1.0);
    double frac = 0.1 + 0.8 * Uu(rng_); // avoid extremes too
    const double j_probe2 = L[j] + frac * (U[j] - L[j]);

    Vec p2 = p1;
    p2[j] = j_probe2;
    Vec p2_pert = p1_pert;
    p2_pert[j] = j_probe2;

    const double f_p2      = prob_->evaluate(p2);
    const double f_p2_pert = prob_->evaluate(p2_pert);

    const double delta1 = f_p1 - f_p1_pert;
    const double delta2 = f_p2 - f_p2_pert;

    const double mag = std::fabs(f_p1) + std::fabs(f_p1_pert)
                      + std::fabs(f_p2) + std::fabs(f_p2_pert);
    const double threshold = eps_D * mag;

    return std::fabs(delta1 - delta2) > threshold;
}

void CCDG2::runGrouping() {
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    if (dg2_max_passes_ <= 0) {
        dg2_max_passes_ = std::min(D, 200);
    }

    // Backward-error bound for D sequential floating-point operations
    // (Higham-style gamma_D = D*eps / (1 - D*eps)), the standard
    // conservative "linear operation count" assumption for a generic
    // black-box objective that DG2's authors describe using.
    const double mach_eps = std::numeric_limits<double>::epsilon();
    double gamma_D = (double)D * mach_eps;
    gamma_D = gamma_D / std::max(1e-300, 1.0 - gamma_D);

    const long long grouping_budget =
        (long long)(grouping_budget_frac_ * (double)max_evals_);

    std::vector<int> unassigned(D);
    std::iota(unassigned.begin(), unassigned.end(), 0);

    groups_.clear();

    // BUG FIX: p1_base used to be the all-lower-bound corner, with each
    // seed's own perturbation jumping to the exact opposite (upper-bound)
    // corner. Combined with symmetric box bounds (L=-c, U=+c, the common
    // case for nearly every benchmark in this framework), this made
    // L[k]^2 == U[k]^2 for every dimension, which for any coupling term
    // that is an EVEN function of the perturbed variable -- most notably
    // Rosenbrock's 100*(x_{k+1}-x_k^2)^2 -- caused delta1-delta2 to cancel
    // to EXACTLY ZERO regardless of whether the two dimensions truly
    // interact. This produced systematic false negatives: every dimension
    // was reported separable even for problems with strong, well-known
    // pairwise coupling. Fixed by drawing both the baseline point and the
    // seed's own perturbed value RANDOMLY within the box (never using the
    // literal L/U corners), so the degenerate a1^2==a2^2 configuration has
    // probability zero instead of being hit on every single test.
    // (Found and verified using a hand-built block-Rosenbrock test problem
    // with known ground-truth grouping -- see delivery notes.)
    std::uniform_real_distribution<double> Uinit(0.0, 1.0);
    Vec p1_base(D);
    for (int k = 0; k < D; ++k) {
        p1_base[k] = L[k] + Uinit(rng_) * (U[k] - L[k]);
    }

    while (!unassigned.empty()) {
        if (prob_->calls() >= grouping_budget) {
            // Budget exhausted: treat everything still unassigned as one
            // final (possibly non-separable / "overlap-catch-all") group,
            // the standard safe fallback when a decomposition method runs
            // out of its allotted grouping budget.
            groups_.push_back(Group{});
            groups_.back().dims = unassigned;
            unassigned.clear();
            break;
        }

        const int seed = unassigned.front();
        unassigned.erase(unassigned.begin());

        Group g;
        g.dims.push_back(seed);

        // Frontier of dims whose interactions still need to be propagated
        // (bounded number of passes below avoids full O(D^2) recomputation
        // for large groups while still catching indirect/transitive
        // interactions).
        std::vector<int> frontier = { seed };

        for (int pass = 0; pass < dg2_max_passes_ && !frontier.empty()
                            && !unassigned.empty(); ++pass) {
            std::vector<int> next_frontier;

            for (int fi : frontier) {
                if (unassigned.empty()) break;
                if (prob_->calls() >= grouping_budget) break;

                Vec p1 = p1_base;
                Vec p1_pert = p1_base;
                {
                    // Random perturbed value for the seed dimension, kept
                    // at least half the box range away from p1[fi] so the
                    // interaction signal stays well above the numerical
                    // noise floor.
                    double v;
                    do { v = L[fi] + Uinit(rng_) * (U[fi] - L[fi]); }
                    while (std::fabs(v - p1[fi]) < 0.5 * (U[fi] - L[fi]));
                    p1_pert[fi] = v;
                }
                const double f_p1      = prob_->evaluate(p1);
                const double f_p1_pert = prob_->evaluate(p1_pert);

                std::vector<int> still_unassigned;
                still_unassigned.reserve(unassigned.size());
                for (int j : unassigned) {
                    if (prob_->calls() >= grouping_budget) {
                        still_unassigned.push_back(j);
                        continue;
                    }
                    if (dg2Interacts(fi, j, f_p1, f_p1_pert, p1, p1_pert, gamma_D)) {
                        g.dims.push_back(j);
                        next_frontier.push_back(j);
                    } else {
                        still_unassigned.push_back(j);
                    }
                }
                unassigned.swap(still_unassigned);
            }
            frontier.swap(next_frontier);
        }

        groups_.push_back(std::move(g));
    }

#ifdef CCDG2_DEBUG_GROUPING
    fprintf(stderr, "[CCDG2 DEBUG] %d groups found:\n", (int)groups_.size());
    for (size_t gi = 0; gi < groups_.size(); ++gi) {
        fprintf(stderr, "  group %zu: {", gi);
        for (int d : groups_[gi].dims) fprintf(stderr, "%d,", d);
        fprintf(stderr, "}\n");
    }
#endif
}

CCDG2::Vec CCDG2::buildFull(const Group& g, const Vec& partial) const {
    Vec full = context_;
    for (size_t k = 0; k < g.dims.size(); ++k) full[g.dims[k]] = partial[k];
    return full;
}

void CCDG2::initGroupPopulation(Group& g) {
    const int gd = (int)g.dims.size();
    // BUG FIX: population size used to be hard-capped at 30 regardless of
    // group dimension (max(4, min(30, 4+2*gd))). For SMALL, well-separated
    // groups (the common, desired case after successful DG2 decomposition)
    // that is plenty. But for the few groups that end up large -- most
    // importantly a fully-connected chain like standard Rosenbrock, which
    // DG2 correctly refuses to split (see delivery notes) -- a population
    // of 30 individuals trying to search hundreds of dimensions
    // simultaneously is hopelessly under-powered, and was silently
    // crippling convergence on exactly the hardest, most important cases
    // without any error or warning. Fixed to scale with sqrt(group size)
    // (a standard DE population-sizing convention balancing search power
    // against per-generation cost), with a much higher ceiling so large
    // groups actually get a workable population.
    g.pop = (group_pop_cfg_ > 0)
          ? group_pop_cfg_
          : std::max(18, std::min(80, (int)std::lround(10.0 * std::sqrt((double)gd))));

    g.X.assign(g.pop, Vec(gd, 0.0));
    g.FX.assign(g.pop, std::numeric_limits<double>::infinity());
    g.archive.clear();
    g.M_F.assign(shade_H_, 0.5);
    g.M_CR.assign(shade_H_, 0.5);
    g.mem_pos = 0;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i = 0; i < g.pop; ++i) {
        for (int k = 0; k < gd; ++k) {
            const int d = g.dims[k];
            g.X[i][k] = L[d] + U01(rng_) * (U[d] - L[d]);
        }
        Vec full = buildFull(g, g.X[i]);
        g.FX[i] = safeEval(full);
        if (g.FX[i] < context_f_) {
            context_f_ = g.FX[i];
            context_ = full;
            ++context_version_;
        }
    }
    // Individuals evaluated earlier in this loop may have been evaluated
    // against a slightly older context_ than individuals evaluated later
    // (if an earlier individual improved context_ mid-loop) -- stamping
    // with the version as of THIS moment means refreshStaleFitness() will
    // correctly re-evaluate everything the first time this group's
    // generation runs, picking up any such residual staleness too.
    g.fx_context_version = context_version_;
}

// See class-level comment for the full rationale: if context_ has changed
// since this group's FX[] values were last computed (tracked via
// fx_context_version vs. the global context_version_), the whole
// subpopulation is re-evaluated against the CURRENT context before this
// group's generation proceeds, so acceptance decisions and pbest ranking
// are never made against a stale, no-longer-comparable baseline.
void CCDG2::refreshStaleFitness(Group& g) {
    if (g.fx_context_version == context_version_) return; // already current
    if (g.X.empty()) { g.fx_context_version = context_version_; return; }

    for (int i = 0; i < (int)g.X.size(); ++i) {
        Vec full = buildFull(g, g.X[i]);
        const double f = safeEval(full);
        g.FX[i] = f;
        if (f < context_f_) {
            context_f_ = f;
            context_ = full;
            // NOTE: deliberately do NOT bump context_version_ here -- this
            // re-evaluation is using g.X's EXISTING positions against the
            // context that was already current when this refresh began;
            // any such improvement is folded into context_ for everyone
            // else's future evaluations, but does not itself invalidate
            // the very group we are currently refreshing (its X hasn't
            // changed, only its recorded fitness has, which is exactly
            // what this function is for).
        }
    }
    g.fx_context_version = context_version_;
}

// One current-to-pbest/1/bin DE generation (SHADE-lite parameter
// self-adaptation with a success-history memory + external archive) for a
// single group's subpopulation.
void CCDG2::groupGeneration(Group& g) {
    const int gd = (int)g.dims.size();
    const int NP = g.pop;
    if (NP < 4 || gd < 1) return;

    refreshStaleFitness(g);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::uniform_int_distribution<int> Uh(0, shade_H_ - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Upop(0, NP - 1);

    const int pbest_count = std::max(1, (int)std::round(p_best_min_frac_ * NP));

    std::vector<int> order(NP);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return g.FX[a] < g.FX[b]; });

    std::vector<Vec> newX = g.X;
    std::vector<double> newFX = g.FX;

    std::vector<double> succ_F, succ_CR, succ_w;

    for (int i = 0; i < NP; ++i) {
        const int h = Uh(rng_);
        double CR = std::clamp(0.1 * std::normal_distribution<double>(0.0,1.0)(rng_) + g.M_CR[h], 0.0, 1.0);
        double F;
        {
            std::cauchy_distribution<double> C(g.M_F[h], 0.1);
            do { F = C(rng_); } while (F <= 0.0);
            if (F > 1.0) F = 1.0;
        }

        const int pbest_idx = order[std::uniform_int_distribution<int>(0, pbest_count - 1)(rng_)];

        int r1;
        do { r1 = Upop(rng_); } while (r1 == i);

        // r2 drawn from population U archive.
        Vec r2_vec;
        const int arch_n = (int)g.archive.size();
        const int total = NP + arch_n;
        int r2_idx;
        do { r2_idx = std::uniform_int_distribution<int>(0, total - 1)(rng_); }
        while (r2_idx == i || r2_idx == r1);
        const Vec& xr2 = (r2_idx < NP) ? g.X[r2_idx] : g.archive[r2_idx - NP];

        const Vec& xi = g.X[i];
        const Vec& xpbest = g.X[pbest_idx];
        const Vec& xr1 = g.X[r1];

        Vec v(gd);
        for (int k = 0; k < gd; ++k) {
            v[k] = xi[k] + F * (xpbest[k] - xi[k]) + F * (xr1[k] - xr2[k]);
        }

        Vec u = xi;
        const int jrand = std::uniform_int_distribution<int>(0, gd - 1)(rng_);
        for (int k = 0; k < gd; ++k) {
            if (U01(rng_) <= CR || k == jrand) u[k] = v[k];
        }

        // Bound-repair: reflect into range (standard DE practice), then clamp.
        for (int k = 0; k < gd; ++k) {
            const int d = g.dims[k];
            const double lo = L[d], hi = U[d];
            if (u[k] < lo) u[k] = std::min(hi, 2.0 * lo - u[k]);
            if (u[k] > hi) u[k] = std::max(lo, 2.0 * hi - u[k]);
            if (u[k] < lo) u[k] = lo;
            if (u[k] > hi) u[k] = hi;
        }

        Vec full = buildFull(g, u);
        const double fu = safeEval(full);

        if (fu <= g.FX[i]) {
            if (fu < g.FX[i]) {
                succ_F.push_back(F);
                succ_CR.push_back(CR);
                succ_w.push_back(g.FX[i] - fu);

                // Archive the replaced parent.
                g.archive.push_back(g.X[i]);
            }
            newX[i] = u;
            newFX[i] = fu;

            if (fu < context_f_) {
                context_f_ = fu;
                context_ = full;
                ++context_version_;
            }
        }
    }

    g.X = std::move(newX);
    g.FX = std::move(newFX);

    // Trim archive to capacity.
    const int arch_cap = std::max(1, (int)std::round(archive_rate_ * NP));
    while ((int)g.archive.size() > arch_cap) {
        const int rem = std::uniform_int_distribution<int>(0, (int)g.archive.size() - 1)(rng_);
        g.archive.erase(g.archive.begin() + rem);
    }

    // SHADE memory update (Lehmer mean for F, weighted arithmetic mean for CR).
    if (!succ_F.empty()) {
        double wsum = 0.0;
        for (double w : succ_w) wsum += w;
        if (wsum <= 0.0) wsum = 1.0;

        double num_F = 0.0, den_F = 0.0, mean_CR = 0.0;
        for (size_t k = 0; k < succ_F.size(); ++k) {
            const double w = succ_w[k] / wsum;
            num_F += w * succ_F[k] * succ_F[k];
            den_F += w * succ_F[k];
            mean_CR += w * succ_CR[k];
        }
        if (den_F > 1e-300) {
            g.M_F[g.mem_pos] = num_F / den_F;
        }
        g.M_CR[g.mem_pos] = mean_CR;
        g.mem_pos = (g.mem_pos + 1) % shade_H_;
    }
}

void CCDG2::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();
    context_f_ = std::numeric_limits<double>::infinity();
    context_version_ = 0;

    // Seed the context vector at the box midpoint before grouping/first
    // subpopulation initialisation (only used as a starting reference;
    // every group immediately overwrites its own dims).
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    context_.assign(D, 0.0);
    for (int j = 0; j < D; ++j) context_[j] = 0.5 * (L[j] + U[j]);
    safeEval(context_);
    context_f_ = best_f_;

    runGrouping();

    for (auto& g : groups_) {
        initGroupPopulation(g);
    }
    current_group_ = 0;

    if (context_f_ < best_f_) { best_f_ = context_f_; best_x_ = context_; }

    Vec fx{best_f_};
    updateStop(fx);
    printBest();
}

void CCDG2::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (groups_.empty()) return;

    Group& g = groups_[current_group_];

    // BUG FIX: a plain "one generation per visit" round-robin gives a
    // 500-dimensional group exactly the same attention per cycle as a
    // 1-dimensional one, even though the former needs vastly more search
    // effort. Scale the number of generations given per visit with the
    // group's own dimensionality (bounded, so one large group cannot
    // monopolize the whole budget and starve the others entirely).
    const int gens = std::max(1, std::min(20, (int)g.dims.size() / 10));
    for (int s = 0; s < gens && prob_->calls() < max_evals_; ++s) {
        groupGeneration(g);
    }
    current_group_ = (current_group_ + 1) % (int)groups_.size();

    if (context_f_ < best_f_) {
        best_f_ = context_f_;
        best_x_ = context_;
    }

    // Optional in-run local search after a successful global-best improvement.
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
                context_ = xloc;
                context_f_ = floc;
                ++context_version_;
            }
        }
    }

    Vec fx{best_f_};
    printBest();
    updateStop(fx);
}

void CCDG2::end() {
    if (!end_local_refine_)        return;
    if (!prob_)                    return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }
    printBest();
}

} // namespace optimsolution
