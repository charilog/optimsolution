#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * IK6DOF – PUMA 560 inverse kinematics benchmark
 *
 * This problem is modeled as a real-world 6-DOF inverse kinematics task for the
 * Unimation PUMA 560 robot using standard DH parameters and literature joint
 * limits. The native dimensionality is fixed to 6; any requested dimension is
 * ignored in init() so that the benchmark remains faithful to the robotic model.
 *
 * Objective:
 *   f(q) = w_pos * ||p(q) - p_target||^2
 *        + w_ori * angle(R(q), R_target)^2
 *        + k_pen * joint-limit overflow penalty
 *
 * The target pose is generated from a feasible reference joint vector, so the
 * global minimum is 0 for at least one reachable configuration.
 */
class IK6DOF : public Problem {
public:
    IK6DOF();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    static constexpr int kDim = 6;

    // ----- constants / weights -----
    double w_pos_ = 1.0;
    double w_ori_ = 1.0;
    double k_pen_ = 1e3;

    // ----- standard DH parameters for the PUMA 560 -----
    double a_[kDim];
    double alpha_[kDim];
    double d_[kDim];
    double theta0_[kDim];

    // ----- literature joint limits -----
    double qlo_[kDim];
    double qhi_[kDim];

    // ----- reachable reference configuration used to define the target pose -----
    double q_target_[kDim];

    // ----- target pose -----
    double tp_[3];
    double tq_[4];

    // ----- helpers -----
    struct Mat4 { double m[16]; };
    static inline double sqr(double v) { return v * v; }

    Mat4 matMul(const Mat4& A, const Mat4& B) const;
    Mat4 dh(double a, double alpha, double d, double theta) const;
    Mat4 forwardKinematics6(const double q6[kDim]) const;
    void setTargetFromReference();

    void   rotToQuat(const Mat4& T, double& qw, double& qx, double& qy, double& qz) const;
    double quatAngleError(double qw1, double qx1, double qy1, double qz1,
                          double qw2, double qx2, double qy2, double qz2) const;
};

} // namespace optimsolution
