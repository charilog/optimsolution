#include "local_methods.h"
#include "localsearch.h"
#include "init.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <utility>

namespace optimsolution {

namespace {

// Forwards every call to the real problem (so its evaluation / gradient
// counters stay authoritative) and records the best-so-far trajectory.
class TrajectoryProblem : public Problem {
public:
    struct Point { long long evals; double f; Vec x; };

    explicit TrajectoryProblem(Problem* inner) : inner_(inner) {}

    int dimension() const override { return inner_->dimension(); }
    const Vec& lb() const override { return inner_->lb(); }
    const Vec& ub() const override { return inner_->ub(); }

    double evaluate_core(const Vec& x) override {
        double f = inner_->evaluate(x);
        if (std::isfinite(f) && f < best_f_) {
            best_f_ = f;
            trace_.push_back({ inner_->calls(), f, x });
        }
        return f;
    }
    void gradient_core(const Vec& x, Vec& g) override { g = inner_->gradient(x); }

    const std::vector<Point>& trace() const { return trace_; }

private:
    Problem* inner_;
    double best_f_{ std::numeric_limits<double>::infinity() };
    std::vector<Point> trace_;
};

std::string lower(std::string s)
{
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

} // namespace

const std::vector<std::string>& LocalMethod::names()
{
    static const std::vector<std::string> n = { "gd", "lbfgs", "bfgs", "nm", "cg", "ncg", "adam" };
    return n;
}

bool LocalMethod::isLocalName(const std::string& name)
{
    const std::string l = lower(name);
    for (const auto& n : names()) if (n == l) return true;
    return false;
}

LocalMethod::LocalMethod(std::string name) : name_(lower(std::move(name))) {}

std::string LocalMethod::methodFullName() const
{
    if (name_ == "gd")    return "Gradient Descent (Local Method)";
    if (name_ == "lbfgs") return "L-BFGS (Local Method)";
    if (name_ == "bfgs")  return "BFGS (Local Method)";
    if (name_ == "nm")    return "Nelder-Mead (Local Method)";
    if (name_ == "cg")    return "Nonlinear Conjugate Gradient (Local Method)";
    if (name_ == "ncg")   return "Truncated Newton / Newton-CG (Local Method)";
    if (name_ == "adam")  return "Adam (Local Method)";
    return name_;
}

void LocalMethod::configure(const MethodConfig& mc)
{
    // Single search point: population is 1 whatever [global] says.
    pop_ = 1;
    // Keep this method's own section values (the settings-file values plus
    // any in-memory changes, e.g. a sensitivity sweep) to overlay onto what
    // the local search reads; keys it does not know are simply ignored.
    overrides_.clear();
    for (const auto& kv : mc.kv) overrides_[lower(kv.first)] = kv.second;
}

void LocalMethod::writeProgressLine(int iter, long long evals, double f, const Vec& x) const
{
    // Same format as Optimizer::printBest(); parsed into the convergence CSV.
    std::ostream& os = out();
    os << "[";
    if (run_index_ >= 0) os << "Run " << (run_index_ + 1) << " | ";
    os << "iter " << iter << " | evals " << evals
       << "] best_f = " << std::setprecision(12) << std::fixed << f << "\n";
    if (landscape_os_ && x.size() == 2) {
        (*landscape_os_) << run_index_ << "," << iter << "," << evals << ","
                         << std::setprecision(15) << f << "," << x[0] << "," << x[1] << "\n";
    }
}

void LocalMethod::init()
{
    max_iters_ = 0;   // the run() loop never executes: the whole search happens here
    pop_ = 1;
    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();
    if (!prob_) return;

    const long long start_calls = prob_->calls();
    const long long budget = max_evals_ - start_calls;
    if (budget <= 0) return;

    // Starting point: one sample from the [init] distribution.
    Initializer initSampler;
    initSampler.configure(initopt_);
    auto pts = initSampler.samplePopulation(*prob_, rng_, 1);
    Vec x0;
    if (!pts.empty()) {
        x0 = pts[0];
    } else {
        const auto& L = prob_->lb();
        const auto& U = prob_->ub();
        x0.assign(prob_->dimension(), 0.0);
        for (size_t d = 0; d < x0.size(); ++d)
            if (std::isfinite(L[d]) && std::isfinite(U[d])) x0[d] = 0.5 * (L[d] + U[d]);
    }

    TrajectoryProblem proxy(prob_);
    std::pair<Vec, double> res;
    profiler_.startIter();
    {
        ScopedLocalSearchOverride scoped(name_, overrides_);
        // The proxy's own counter starts at 0, so the cap is the remaining budget.
        res = runLocalSearch(name_, &proxy, rng_, x0, -1, budget);
    }
    profiler_.endIter(prob_->calls());

    // Best point actually evaluated (all evaluations go through the proxy).
    const auto& tr = proxy.trace();
    if (!tr.empty()) {
        best_x_ = tr.back().x;
        best_f_ = tr.back().f;
    }
    if (std::isfinite(res.second) && res.second < best_f_ && !res.first.empty()) {
        best_x_ = res.first;
        best_f_ = res.second;
    }
    if (best_x_.empty()) best_x_ = x0;

    // Progress lines from the recorded trajectory, thinned to <= ~2000 lines.
    const long long used = std::max<long long>(1, prob_->calls() - start_calls);
    const long long step = std::max<long long>(1, used / 2000);
    long long last_emitted = -1;
    int iter = 0;
    for (size_t k = 0; k < tr.size(); ++k) {
        const bool last = (k + 1 == tr.size());
        if (!last && last_emitted >= 0 && tr[k].evals - last_emitted < step) continue;
        writeProgressLine(iter++, tr[k].evals, tr[k].f, tr[k].x);
        last_emitted = tr[k].evals;
    }
    if (tr.empty()) writeProgressLine(iter++, prob_->calls(), best_f_, best_x_);
    // Final line at the total evaluation count (the search may continue past
    // its last improvement before it stops).
    if (prob_->calls() > last_emitted && !tr.empty())
        writeProgressLine(iter++, prob_->calls(), best_f_, best_x_);
    iters_ = iter;
}

} // namespace optimsolution
