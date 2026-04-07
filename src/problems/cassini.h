#pragma once
#include "problem.h"
#include <vector>
#include <array>

namespace optimsolution {

/**
 * Cassini interplanetary transfer timing problem.
 *
 * Decision vector x ∈ [0,1]^6 mapped to physical times t_i in mission windows.
 *
 * Objective:
 *   f(t) = w_pref   * Σ_i ((t_i - μ_i)/σ_i)^2
 *        + w_gap    * Σ_{i} max(0, mingap_i - (t_{i+1} - t_i))^2
 *        + w_dur    * duration_penalty(T = t_5 - t_0)
 *        + w_smooth * Σ_i (t_{i+1} - 2 t_i + t_{i-1})^2
 *
 * This is a real-world (non-synthetic) continuous multi-constraint
 * nonlinear timing optimization problem.
 *
 * - Dimension: 6 (forced)
 * - Domain: [0,1]^6 (later mapped to physical windows)
 * - No closed-form known optimum.
 */
class Cassini : public Problem {
public:
    Cassini();

    void init(int dim) override;     // forces dim=6, sets bounds
    void setTimeWindows(const std::vector<double>& tmin,
                        const std::vector<double>& tmax);
    void setPreferredTimes(const std::vector<double>& mu,
                           const std::vector<double>& sigma);
    void setMinGaps(const std::vector<double>& mingap);
    void setDurationLimits(double Tmin, double Tmax);
    void setWeights(double w_pref, double w_gap,
                    double w_dur, double w_smooth);

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    static constexpr int K = 6;

    std::array<double, K> tmin_{};
    std::array<double, K> tmax_{};
    std::array<double, K> mu_{};
    std::array<double, K> sigma_{};
    std::array<double, K - 1> mingap_{};

    double Tmin_{1500.0};
    double Tmax_{3500.0};

    double w_pref_{1.0};
    double w_gap_{100.0};
    double w_dur_{0.1};
    double w_smooth_{0.001};

    inline void map_to_times(const Vec& x,
                             std::array<double, K>& t,
                             std::array<double, K>& dt_dx) const;

    static inline double sqr(double a) { return a * a; }
};

} // namespace optimsolution
