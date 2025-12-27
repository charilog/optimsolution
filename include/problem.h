#pragma once
#include <vector>
#include <limits>
#include <string>

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

    double evaluate(const Vec& x) {
        ++calls_;
        return evaluate_core(x);
    }
    virtual double evaluate_core(const Vec& x) = 0;

    Vec gradient(const Vec& x) {
        ++grad_calls_;
        Vec g(dimension(), 0.0);
        gradient_core(x, g);
        return g;
    }
    virtual void gradient_core(const Vec& x, Vec& g) {
        (void)x;
        g.assign(dimension(), 0.0);
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

protected:
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
    long long calls_{0};
    long long grad_calls_{0};
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
