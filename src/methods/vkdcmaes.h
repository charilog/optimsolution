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

// VkD-CMA-ES -- restricted covariance-matrix-adaptation CMA-ES with a
// diagonal-plus-rank-k model, C = D(I + sum_{j=1}^k v_j v_j^T)D.
//
// References (see class-level notes below for exactly what is, and is
// not, taken verbatim from each):
//   Akimoto, Y.; Auger, A.; Hansen, N. "Comparison-Based Natural Gradient
//     Optimization in High Dimension." GECCO 2014. -- introduces VD-CMA,
//     the k=1 special case of this model, with a full derivation from the
//     Information-Geometric-Optimization / natural-gradient framework and
//     an O(d) closed-form update (their Theorem 3.6). This part is
//     implemented here VERBATIM from the paper's equations.
//   Akimoto, Y.; Hansen, N. "Projection-Based Restricted Covariance
//     Matrix Adaptation for High Dimension." GECCO 2016. -- generalises
//     VD-CMA to rank k > 1 with an online mechanism that grows/shrinks k
//     automatically ("VkD-CMA"). The full derivation of the joint k-vector
//     natural gradient and the exact online model-selection criterion for
//     k were NOT accessible in verifiable form while implementing this
//     (unlike the k=1 GECCO'14 paper, whose complete equations were
//     retrieved and independently numerically verified -- see the
//     sampling-covariance check in the delivery notes). Implementing an
//     unverified reconstruction of that specific mechanism carried a real
//     risk of the same kind of subtle, hard-to-notice error found and
//     fixed elsewhere in this project (e.g. the DG2 grouping bug, the CSO
//     phi instability). Rather than guess, the k>1 behaviour here is a
//     clearly-labelled, self-consistent EXTENSION of the verified k=1
//     mathematics (see below) instead of a claimed reproduction of
//     Akimoto & Hansen's specific unpublished-here mechanism.
//
// What is exact (from the GECCO'14 paper, k=1 case), applied per stored
// vector v_j:
//   - the sampling transform y = z + [(sqrt(1+||v||^2)-1)/||v||^2] <z,v> v
//     (verified numerically to reproduce Cov(y) = I + vv^T -- the paper's
//     printed formula omits a "/||v||^2" that is needed; this is very
//     likely a typesetting/OCR artefact of the source, not a paper error)
//   - the O(d) natural-gradient computation of Theorem 3.6 for updating v
//     and the diagonal D, including the numerically-stabilising alpha/A/b
//     correction of Lemma 3.4 / Proposition 3.5
//   - the cumulation (p_sigma, p_c, h_sigma) and step-size control, shared
//     with standard (mu/mu_w,lambda)-CMA-ES
//   - the VD-CMA-specific learning-rate rescaling c_sigma, c_1, c_mu
//
// What is this implementation's own extension (k>1, "adaptive" k):
//   - k vectors v_1..v_k are each updated via an INDEPENDENT copy of the
//     verified k=1 update above (not a jointly-derived k-vector natural
//     gradient), applied SEQUENTIALLY for sampling -- the same style of
//     approximation this framework's LM-MA-ES uses for its m stored
//     vectors, and which the LM-MA-ES paper itself acknowledges is not
//     exactly measure-preserving for non-orthogonal stored directions.
//   - each v_j gets its OWN exponentially-spaced learning rate (inspired
//     by LM-MA-ES's multi-timescale m_j vectors), so different vectors
//     specialise to different update speeds instead of competing for the
//     same one.
//   - k grows (up to k_max = O(log n)) when the newest vector's norm
//     stays above a threshold for several generations in a row (evidence
//     that the current model still has useful, unexploited anisotropic
//     signal left to capture) and is otherwise held fixed; this is a
//     simple, transparent heuristic capturing the general spirit of
//     online rank selection, not a reproduction of the paper's specific
//     (unverified, here) criterion.
class VkDCMAES : public Optimizer {
public:
    VkDCMAES() = default;
    ~VkDCMAES() override = default;

    std::string methodShortName() const override { return "vkdcmaes"; }
    std::string methodFullName()  const override {
        return "VkD-CMA-ES (diagonal + adaptive rank-k covariance model)";
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
    void   ensureBounds(Vec& x) const;

    // Applies the sampling transform for ONE stored vector v_j to a
    // (partially transformed) vector y in place: y <- y + c_j <y,v_j> v_j.
    void applyVjSample(const Vec& vj, double vj_norm2, Vec& y) const;

    // Theorem 3.6's O(d) natural-gradient computation for ONE vector v_j
    // and the shared diagonal D, given ycoord = D^{-1}(x-m)/sigma.
    // Writes the natural-gradient updates into gs (for theta_D) and gt
    // (for v_j, still needing the ||v_j||^{-1} scale applied by the caller).
    void naturalGradientOneVector(const Vec& ycoord, const Vec& vj, double vj_norm2,
                                   Vec& gs, Vec& gt) const;

private:
    // --- configuration ---
    int    lambda_cfg_{-1};
    int    k_cfg_{-1};     // fixed k if >0; else adaptive starting at k_init_
    int    k_init_{1};
    int    k_max_cfg_{-1}; // <=0 -> auto O(log n)
    double sigma0_{0.3};   // initial step, as a fraction of the average box range
    int    grow_patience_{5}; // generations of sustained signal before k grows

    // --- derived, fixed once D (dimension) is known ---
    int    lambda_{0};
    int    mu_{0};
    double mu_eff_{0.0};
    std::vector<double> w_;
    double c_sigma_{0.0}, d_sigma_{0.0};
    double c_c_{0.0}, c1_{0.0}, cmu_{0.0};
    int    k_max_{0};

    // --- algorithm state ---
    int    t_{0};
    Vec    mean_;
    double sigma_{0.3};
    Vec    Dvec_;                 // diagonal of D, size n
    std::vector<Vec> V_;          // k stored vectors v_j, each size n
    std::vector<double> c_v_;     // per-vector learning rate (LM-MA-ES-style spacing)
    int    k_{1};                 // current active rank
    int    grow_streak_{0};

    Vec    p_sigma_;
    Vec    p_c_;

    // --- offspring buffers ---
    std::vector<Vec>    Z_;   // raw z_i
    std::vector<Vec>    Y_;   // transformed y_i = (I+sum v_j v_j^T)^{1/2}-ish sample
    std::vector<double> F_;

    // Scratch buffers for naturalGradientOneVector(), reused across calls
    // to avoid repeated heap allocation in the hot per-(vector,individual)
    // loop (this call happens k*mu times per generation).
    mutable Vec scratch_s_, scratch_t_, scratch_Ainv_, scratch_Ainv_v_, scratch_Ainv_s_;

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
