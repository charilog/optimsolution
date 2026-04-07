#include "ik6dof.h"
#include <algorithm>
#include <cmath>

namespace optimsolution {

namespace {
constexpr double PI  = 3.141592653589793238462643383279502884;
constexpr double DEG = PI / 180.0;
constexpr double INCH = 0.0254;
}

IK6DOF::IK6DOF() {
    setName("ik6dof");
    setFullName("PUMA 560 6-DOF Inverse Kinematics");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("real-world robotics benchmark");

    // Standard-DH PUMA 560 parameters commonly used in the robotics literature.
    const double a_def[kDim]     = {0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0};
    const double alpha_def[kDim] = {PI / 2.0, 0.0, -PI / 2.0, PI / 2.0, -PI / 2.0, 0.0};
    const double d_def[kDim]     = {26.45 * INCH, 0.0, 0.15005, 0.4318, 0.0, 0.0};
    const double th0_def[kDim]   = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < kDim; ++i) {
        a_[i] = a_def[i];
        alpha_[i] = alpha_def[i];
        d_[i] = d_def[i];
        theta0_[i] = th0_def[i];
    }

    // Joint limits widely used for the PUMA 560.
    qlo_[0] = -160.0 * DEG; qhi_[0] =  160.0 * DEG;
    qlo_[1] = -110.0 * DEG; qhi_[1] =  110.0 * DEG;
    qlo_[2] = -135.0 * DEG; qhi_[2] =  135.0 * DEG;
    qlo_[3] = -266.0 * DEG; qhi_[3] =  266.0 * DEG;
    qlo_[4] = -100.0 * DEG; qhi_[4] =  100.0 * DEG;
    qlo_[5] = -266.0 * DEG; qhi_[5] =  266.0 * DEG;

    // Feasible non-singular reference configuration used to define a reachable target pose.
    const double q_target_def[kDim] = {
        0.0,
        45.0 * DEG,
        -90.0 * DEG,
        0.0,
        45.0 * DEG,
        0.0
    };
    for (int i = 0; i < kDim; ++i) {
        q_target_[i] = q_target_def[i];
    }

    setTargetFromReference();
}

void IK6DOF::init(int /*dim*/) {
    Problem::init(kDim);

    Vec lo(kDim), hi(kDim);
    for (int i = 0; i < kDim; ++i) {
        lo[i] = qlo_[i];
        hi[i] = qhi_[i];
    }
    setBounds(lo, hi);

    setTargetFromReference();
}

double IK6DOF::evaluate_core(const Vec& x) {
    double q6[kDim] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int k = std::min(kDim, static_cast<int>(x.size()));
    for (int i = 0; i < k; ++i) {
        q6[i] = x[i];
    }

    const Mat4 T = forwardKinematics6(q6);

    const double px = T.m[3];
    const double py = T.m[7];
    const double pz = T.m[11];
    const double ex = px - tp_[0];
    const double ey = py - tp_[1];
    const double ez = pz - tp_[2];
    const double pos_err2 = ex * ex + ey * ey + ez * ez;

    double qw, qx, qy, qz;
    rotToQuat(T, qw, qx, qy, qz);
    const double ang = quatAngleError(qw, qx, qy, qz, tq_[0], tq_[1], tq_[2], tq_[3]);
    const double ori_err2 = ang * ang;

    double overflow_pen = 0.0;
    for (int i = 0; i < k; ++i) {
        if (x[i] < qlo_[i]) {
            overflow_pen += sqr(qlo_[i] - x[i]);
        } else if (x[i] > qhi_[i]) {
            overflow_pen += sqr(x[i] - qhi_[i]);
        }
    }

    double f = w_pos_ * pos_err2 + w_ori_ * ori_err2 + k_pen_ * overflow_pen;
    if (std::isnan(f) || std::isinf(f)) {
        f = 1e12;
    }
    return f;
}

void IK6DOF::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

IK6DOF::Mat4 IK6DOF::matMul(const Mat4& A, const Mat4& B) const {
    Mat4 C{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) {
                s += A.m[4 * r + k] * B.m[4 * k + c];
            }
            C.m[4 * r + c] = s;
        }
    }
    return C;
}

IK6DOF::Mat4 IK6DOF::dh(double a, double alpha, double d, double theta) const {
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    return Mat4{
        ct,   -st * ca,  st * sa,  a * ct,
        st,    ct * ca, -ct * sa,  a * st,
        0.0,        sa,       ca,       d,
        0.0,       0.0,      0.0,     1.0
    };
}

IK6DOF::Mat4 IK6DOF::forwardKinematics6(const double q6[kDim]) const {
    Mat4 T{1.0, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 1.0};

    for (int i = 0; i < kDim; ++i) {
        const Mat4 Ti = dh(a_[i], alpha_[i], d_[i], theta0_[i] + q6[i]);
        T = matMul(T, Ti);
    }
    return T;
}

void IK6DOF::setTargetFromReference() {
    const Mat4 T = forwardKinematics6(q_target_);
    tp_[0] = T.m[3];
    tp_[1] = T.m[7];
    tp_[2] = T.m[11];
    rotToQuat(T, tq_[0], tq_[1], tq_[2], tq_[3]);
}

void IK6DOF::rotToQuat(const Mat4& T, double& qw, double& qx, double& qy, double& qz) const {
    const double r00 = T.m[0],  r01 = T.m[1],  r02 = T.m[2];
    const double r10 = T.m[4],  r11 = T.m[5],  r12 = T.m[6];
    const double r20 = T.m[8],  r21 = T.m[9],  r22 = T.m[10];
    const double tr = r00 + r11 + r22;

    if (tr > 0.0) {
        const double S = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * S;
        qx = (r21 - r12) / S;
        qy = (r02 - r20) / S;
        qz = (r10 - r01) / S;
    } else if (r00 > r11 && r00 > r22) {
        const double S = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
        qw = (r21 - r12) / S;
        qx = 0.25 * S;
        qy = (r01 + r10) / S;
        qz = (r02 + r20) / S;
    } else if (r11 > r22) {
        const double S = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
        qw = (r02 - r20) / S;
        qx = (r01 + r10) / S;
        qy = 0.25 * S;
        qz = (r12 + r21) / S;
    } else {
        const double S = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
        qw = (r10 - r01) / S;
        qx = (r02 + r20) / S;
        qy = (r12 + r21) / S;
        qz = 0.25 * S;
    }

    const double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (n > 0.0) {
        qw /= n;
        qx /= n;
        qy /= n;
        qz /= n;
    } else {
        qw = 1.0;
        qx = 0.0;
        qy = 0.0;
        qz = 0.0;
    }
}

double IK6DOF::quatAngleError(double qw1, double qx1, double qy1, double qz1,
                              double qw2, double qx2, double qy2, double qz2) const {
    double dot = qw1 * qw2 + qx1 * qx2 + qy1 * qy2 + qz1 * qz2;
    dot = std::abs(dot);
    dot = std::max(0.0, std::min(1.0, dot));
    return 2.0 * std::acos(dot);
}

} // namespace optimsolution
