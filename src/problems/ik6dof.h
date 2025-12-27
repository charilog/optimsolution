#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * IK6DOF (6R serial arm) – literature-faithful inverse kinematics objective
 *
 * Decision vector: q ∈ R^n, with n chosen at init(). Only the first 6 affect FK.
 * Bounds: default [-pi, pi] for all n.
 *
 * Objective:
 *   f(q) = w_pos * ||p(q) - p_target||^2
 *        + w_ori * (angle( qR(q), qR_target ))^2
 *        + k_pen * soft joint-limit penalties (for first min(n,6) joints)
 *        + k_extra * sum_{i>6} q_i^2  (if n>6)
 *
 * FK: Denavit–Hartenberg chain with (a_i, alpha_i, d_i, theta0_i + q_i).
 */
class IK6DOF : public Problem {
public:
    IK6DOF();
    void init(int dim) override;                  // sets dimension & bounds

protected:
    double evaluate_core(const Vec& x) override;  // objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diff

private:
    // ----- constants / weights -----
    double w_pos_   = 1.0;
    double w_ori_   = 1.0;
    double k_pen_   = 1e-3;
    double k_extra_ = 1e-3;

    // ----- DH parameters for a generic 6R arm -----
    double a_[6], alpha_[6], d_[6], theta0_[6];

    // ----- target pose -----
    double tp_[3];     // target position (px, py, pz)
    double tq_[4];     // target quaternion (w, x, y, z), normalized

    // ----- helpers  -----
    struct Mat4 { double m[16]; }; // row-major 4x4
    static inline double sqr(double v) { return v*v; }

    Mat4 matMul(const Mat4& A, const Mat4& B) const;
    Mat4 dh(double a, double alpha, double d, double theta) const;
    Mat4 forwardKinematics6(const double q6[6]) const;

    void   rotToQuat(const Mat4& T, double& qw, double& qx, double& qy, double& qz) const;
    double quatAngleError(double qw1,double qx1,double qy1,double qz1,
                          double qw2,double qx2,double qy2,double qz2) const;
};

} // namespace optimsolution
