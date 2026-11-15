#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <algorithm>

namespace optimsolution {

using Vec = std::vector<double>;

class Problem {
public:
    virtual ~Problem() = default;

    // ---------------- Core interface ----------------
    virtual void init(int d) {
        dim_ = (d > 0 ? d : 1);
        if ((int)lb_.size() != dim_ || (int)ub_.size() != dim_) {
            lb_.assign(dim_, -5.0);
            ub_.assign(dim_,  5.0);
        }
    }

    virtual int dimension() const { return dim_; }
    virtual const Vec& lb() const { return lb_; }
    virtual const Vec& ub() const { return ub_; }

    // BUGFIX (data race): several methods (ppso, pde, pga, psioa) call
    // evaluate()/gradient() from inside "#pragma omp parallel for". The plain
    // "++calls_" was a data race: increments were lost, so the evaluation
    // budget check (calls() >= max_evals) and every reported evaluation
    // count were wrong (and different from run to run). The increments are
    // now atomic; without OpenMP the pragma is ignored.
    double evaluate(const Vec& x) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++calls_;
        return evaluate_core(x);
    }
    virtual double evaluate_core(const Vec& x) = 0;

    // ---------------- Multi-objective (optional, additive) ----------------
    // Default: 1 objective, forwarding to evaluate_core(). This means every
    // existing problem automatically "works" in multi-objective mode as a
    // trivial 1-objective case, with ZERO change to single-objective
    // behavior. A problem that genuinely has >=2 competing criteria can
    // override BOTH numObjectives() and evaluateMultiCore() while leaving
    // evaluate_core() completely untouched -- so Single/Batch/Sensitivity
    // modes keep using the original scalarized objective exactly as before,
    // and only the dedicated Multi-objective mode uses the split criteria.
    virtual int numObjectives() const { return 1; }

    Vec evaluateMulti(const Vec& x) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++multi_calls_;
        return evaluateMultiCore(x);
    }
    virtual Vec evaluateMultiCore(const Vec& x) { return { evaluate_core(x) }; }

    bool isMultiObjective() const { return numObjectives() >= 2; }
    long long multiCalls() const { return multi_calls_; }

    Vec gradient(const Vec& x) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++grad_calls_;
        Vec g(dimension(), 0.0);
        gradient_core(x, g);
        return g;
    }
    // BUGFIX: the default used to return an all-zero gradient. Every problem
    // without its own gradient_core() therefore made the gradient-based local
    // searches (gd / lbfgs / bfgs -- including the [global] end_local_method
    // = lbfgs refinement used by every method) stop immediately without doing
    // anything. The default is now a central finite-difference gradient
    // (one-sided at the box bounds), same convention as the problems that
    // already implement it themselves (evaluate_core(): gradient work is
    // reported through gradCalls(), not calls()).
    virtual void gradient_core(const Vec& x, Vec& g) {
        finiteDifferenceGradient(x, g);
    }

    long long calls() const      { return calls_; }
    long long gradCalls() const  { return grad_calls_; }

    void setMaxEvaluations(long long m) { max_evals_ = m; }
    long long maxEvaluations() const    { return max_evals_; }

    // ---------------- Metadata: name / type ----------------
    // short machine-friendly name (π.χ. "rastrigin")
    void setName(const std::string& n)         { name_ = n; }
    const std::string& name() const            { return name_; }

    
    void setFullName(const std::string& n)     { full_name_ = n; }
    const std::string& fullName() const        { return full_name_; }

    // modality: "unimodal", "multimodal", "unknown" ...
    void setModality(const std::string& m)     { modality_ = m; }
    const std::string& modality() const        { return modality_; }

    // separability: "separable", "non-separable", ...
    void setSeparability(const std::string& s) { separability_ = s; }
    const std::string& separability() const    { return separability_; }

    // "continuous benchmark", "engineering design", ...
    void setCategory(const std::string& c)     { category_ = c; }
    const std::string& category() const        { return category_; }

    // ---------------- Known global optimum ----------------
    void setKnownGlobalOptimum(double fopt) {
        f_global_opt_  = fopt;
        has_global_opt_ = true;
        x_global_opt_.clear();
    }


    void setKnownGlobalOptimum(double fopt, const Vec& xopt) {
        f_global_opt_   = fopt;
        x_global_opt_   = xopt;
        has_global_opt_ = true;
    }

    bool   hasKnownGlobalOptimum() const { return has_global_opt_; }
    double globalOptimum() const         { return f_global_opt_; }
    const Vec& globalArgOptimum() const  { return x_global_opt_; }

    // ---------------- Bounds helpers ----------------
    bool hasUniformBounds() const {
        if (dim_ <= 0 || lb_.empty() || ub_.empty()) return false;
        double lo0 = lb_[0];
        double hi0 = ub_[0];
        for (int i = 1; i < dim_; ++i) {
            if (lb_[i] != lo0 || ub_[i] != hi0) return false;
        }
        return true;
    }


    double uniformLowerBound() const {
        return (lb_.empty() ? 0.0 : lb_[0]);
    }
    double uniformUpperBound() const {
        return (ub_.empty() ? 0.0 : ub_[0]);
    }

    // ---------------- Warm-start bounds override (additive) ----------------
    // Used by Mode 8 (--portfolio, src/mode_portfolio.cpp) to bias a fresh
    // segment's own Initializer toward the current incumbent without
    // touching any individual method's init() code: every existing sampler
    // in Initializer already draws from lb()/ub(), so temporarily narrowing
    // those is enough to "warm start" any of the 74 registered methods
    // uniformly. Nestable via an internal stack; always paired push/pop.
    // Never changes evaluate_core()/gradient_core() themselves -- only the
    // box the initializer sees -- so a method that ends up wandering outside
    // the shrunk box mid-segment (perfectly normal; most methods clamp to
    // whatever lb()/ub() they read at init() time and never re-read it) is
    // not penalized in any way.
    void pushShrunkBounds(const Vec& center, double radiusFrac) {
        boundsStack_.push_back({lb_, ub_});
        const Vec& baseLo = boundsStack_.back().first;
        const Vec& baseHi = boundsStack_.back().second;
        Vec newLo(dim_), newHi(dim_);
        double rf = radiusFrac;
        if (rf <= 0.0) rf = 1e-6;
        if (rf > 1.0)  rf = 1.0;
        for (int j = 0; j < dim_; ++j) {
            double half = 0.5 * rf * (baseHi[j] - baseLo[j]);
            double c = (j < (int)center.size()) ? center[j] : 0.5 * (baseLo[j] + baseHi[j]);
            double lo = c - half, hi = c + half;
            if (lo < baseLo[j]) lo = baseLo[j];
            if (hi > baseHi[j]) hi = baseHi[j];
            if (lo > hi) std::swap(lo, hi);
            newLo[j] = lo; newHi[j] = hi;
        }
        lb_ = newLo; ub_ = newHi;
    }
    void popBounds() {
        if (boundsStack_.empty()) return;
        lb_ = boundsStack_.back().first;
        ub_ = boundsStack_.back().second;
        boundsStack_.pop_back();
    }

protected:
    // Central finite differences with a relative step h = 6e-6*max(1,|x_j|)
    // (~cbrt(machine eps)); forward/backward difference when x_j +- h would
    // leave [lb_j, ub_j].
    void finiteDifferenceGradient(const Vec& x, Vec& g) {
        const int D = dimension();
        g.assign(D, 0.0);
        Vec xt = x;
        for (int j = 0; j < D && j < (int)x.size(); ++j) {
            const double xj = x[j];
            const double h  = 6.0e-6 * std::max(1.0, std::fabs(xj));
            const bool hasLo = j < (int)lb_.size(), hasHi = j < (int)ub_.size();
            const bool canUp   = !hasHi || xj + h <= ub_[j];
            const bool canDown = !hasLo || xj - h >= lb_[j];
            if (canUp && canDown) {
                xt[j] = xj + h; const double fp = evaluate_core(xt);
                xt[j] = xj - h; const double fm = evaluate_core(xt);
                g[j] = (fp - fm) / (2.0 * h);
            } else if (canUp) {
                const double f0 = evaluate_core(x);
                xt[j] = xj + h; g[j] = (evaluate_core(xt) - f0) / h;
            } else if (canDown) {
                const double f0 = evaluate_core(x);
                xt[j] = xj - h; g[j] = (f0 - evaluate_core(xt)) / h;
            }
            if (!std::isfinite(g[j])) g[j] = 0.0;
            xt[j] = xj;
        }
    }

    void setBounds(double lo, double hi) {
        lb_.assign(dim_, lo);
        ub_.assign(dim_, hi);
    }
    void setBounds(const Vec& lo, const Vec& hi) {
        lb_ = lo; ub_ = hi;
        dim_ = (int)lb_.size();
    }

    int  dim_{0};
    Vec  lb_, ub_;
    std::vector<std::pair<Vec,Vec>> boundsStack_;  // see pushShrunkBounds()/popBounds()
    long long calls_{0};
    long long grad_calls_{0};
    long long multi_calls_{0};
    long long max_evals_{std::numeric_limits<long long>::max()};

    // --- metadata fields ---
    std::string name_{"unknown"};
    std::string full_name_{};
    std::string modality_{"unknown"};
    std::string separability_{"unknown"};
    std::string category_{};

    bool has_global_opt_{false};
    double f_global_opt_{0.0};
    Vec x_global_opt_;
};

} // namespace optimsolution
