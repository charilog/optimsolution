#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig;

// VkD-CMA-ES -- projection-based restricted covariance matrix adaptation,
// C = D (I + sum_{j<k} s_j v_j v_j^T) D with an online-adapted rank k.
//
// References:
//   Akimoto, Y. and Hansen, N., 2016. "Projection-based restricted covariance
//     matrix adaptation for high dimension." GECCO 2016, pp. 197-204.
//   Akimoto, Y. and Hansen, N., 2016. "Online model selection for restricted
//     covariance matrix adaptation." PPSN XIV, pp. 3-13.
//
// REWRITE: the previous implementation was, by its own description, NOT the
// published algorithm but a self-made "extension" (k independent copies of
// the k=1 VD-CMA update and a heuristic rank growth), and it did not work:
// on a shifted 10-D sphere it stalled at f ~ 1.3e3 after 1,600 of 20,000
// evaluations. This version is a line-by-line port of the authors' official
// algorithm (Y. Akimoto's reference Python code, as distributed in pypop7's
// optimizers/nes/vkdcma.py), which reaches f ~ 1e-16 on the same problem:
//   * sampling z = (N(0,I) + sum_j N(0,1) sqrt(s_j) v_j) .* d, x = m + sigma z
//   * projection of the rank-(k + mu + 1) update matrix onto rank k via the
//     eigen-decomposition of U^T U (or U U^T), learning rates
//     c1 = 2/(n(k+1)+n+2(k+2)+mu_eff), cmu = ..., cc = sqrt(c1)
//   * step-size adaptation by mirrored injection of the last mean shift
//     (two-point / rank-based success, c_s = 0.3, d_s = sqrt(n))
//   * online increase / decrease of k.
// Framework adaptations: initial sigma = sigma0 * (mean box width); initial
// mean sampled through the [init] Initializer (uniform by default, as in the
// reference when no mean is given); sampled points are clipped to the box
// and the update uses the clipped point (Lamarckian repair, as this
// framework's CMA-ES), since the reference code is unconstrained.
class VkDCMAES : public Optimizer {
public:
    VkDCMAES() = default;
    ~VkDCMAES() override = default;

    std::string methodShortName() const override { return "vkdcmaes"; }
    std::string methodFullName()  const override {
        return "VkD-CMA-ES (projection-based restricted CMA, Akimoto & Hansen 2016)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double safeEval(const Vec& x);
    void   clipToBounds(Vec& x) const;
    void   setLearningRates();                 // _get_lr(k)
    double logDet() const;                     // _get_det(d, s)
    double mahalanobisSqOfDx() const;          // _get_m(d_x, d, v, s)
    // symmetric eigen-decomposition (cyclic Jacobi): A (m x m, row-major)
    // -> eigenvalues ev, eigenvectors as COLUMNS of V (row-major m x m)
    static void symEig(std::vector<double> A, int m, Vec& ev, std::vector<double>& V);

private:
    // --- configuration ---
    int    lambda_cfg_{-1};
    double sigma0_{0.3};   // initial step, as a fraction of the average box range

    // --- strategy parameters ---
    int    n_{0}, lambda_{0}, mu_{0};
    Vec    w_;
    double mu_eff_{0.0};
    double c1_{0.0}, cmu_{0.0}, cc_{0.0};
    double cs_{0.3}, ds_{1.0};

    // --- state ---
    Vec    mean_;
    double sigma_{1.0};
    Vec    d_;                    // diagonal scaling, size n
    std::vector<Vec> v_;          // k rows of size n
    Vec    s_;                    // size n (first k_a used)
    Vec    pc_;                   // evolution path, size n
    Vec    dx_;                   // last weighted mean step (in z space)
    double ps_{0.0};
    int    k_{0}, ka_{0}, ki_{0};
    bool   injection_{false};
    double l_s_{0.0};
    Vec    l_d_, l_c_;

    // --- offspring ---
    std::vector<Vec>    Z_;
    std::vector<double> F_;

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
