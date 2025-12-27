#include "ik6dof.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.141592653589793238462643383279502884; }

IK6DOF::IK6DOF() {
    
    setName("ik6dof");
    setFullName("6-DOF Inverse Kinematics (DH serial chain)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("inverse kinematics benchmark");

    // Generic 6R DH (industrial-like)
    const double a_def[6]     = {0.0, 0.30, 0.20, 0.0, 0.0, 0.0};
    const double alpha_def[6] = {PI/2, 0.0,  0.0,  PI/2, -PI/2, 0.0};
    const double d_def[6]     = {0.30, 0.0,  0.0,  0.25,  0.0,   0.10};
    const double th0_def[6]   = {0,0,0,0,0,0};
    for (int i=0;i<6;++i){ a_[i]=a_def[i]; alpha_[i]=alpha_def[i]; d_[i]=d_def[i]; theta0_[i]=th0_def[i]; }

    // Default target pose
    tp_[0]=0.40; tp_[1]=0.20; tp_[2]=0.30;
    tq_[0]=1.0;  tq_[1]=0.0;  tq_[2]=0.0;  tq_[3]=0.0; // identity orientation
}

void IK6DOF::init(int dim) {
    if (dim < 1) dim = 1;
    Problem::init(dim);

    // Bounds: [-pi, pi] for all dims
    Vec lo(dim, -PI), hi(dim, PI);
    setBounds(lo, hi);

    // Normalize target quaternion (defensive)
    double n = std::sqrt(tq_[0]*tq_[0]+tq_[1]*tq_[1]+tq_[2]*tq_[2]+tq_[3]*tq_[3]);
    if (n > 0) { for (int i=0;i<4;++i) tq_[i] /= n; } else { tq_[0]=1; tq_[1]=tq_[2]=tq_[3]=0; }
}

double IK6DOF::evaluate_core(const Vec& x) {
    // Build first 6 joint values (zero-fill if dim<6)
    double q6[6] = {0,0,0,0,0,0};
    const int k = std::min(6, (int)x.size());
    for (int i=0;i<k; ++i) q6[i] = x[i];

    // FK
    Mat4 T = forwardKinematics6(q6);

    // EE position
    const double px = T.m[3],  py = T.m[7],  pz = T.m[11];
    const double ex = px - tp_[0];
    const double ey = py - tp_[1];
    const double ez = pz - tp_[2];
    const double pos_err2 = ex*ex + ey*ey + ez*ez;

    // EE orientation quaternion
    double qw, qx_, qy_, qz_;
    rotToQuat(T, qw, qx_, qy_, qz_);
    const double ang = quatAngleError(qw,qx_,qy_,qz_, tq_[0],tq_[1],tq_[2],tq_[3]);
    const double ori_err2 = ang * ang;

    // Soft penalties:
    // (a) joint limits near-bound/overflow for first min(n,6) joints
    double pen = 0.0;
    // local bounds used at init()
    const double L = -PI, R = PI, buf = 0.05*(R-L); // 5% smooth buffer
    for (int i=0;i<k; ++i) {
        const double qi = x[i];
        if (qi < L) pen += sqr(L-qi);
        else if (qi < L+buf) { const double t=(L+buf-qi)/buf; pen += 0.1*sqr(t); }
        if (qi > R) pen += sqr(qi-R);
        else if (qi > R-buf) { const double t=(qi-(R-buf))/buf; pen += 0.1*sqr(t); }
    }
    // (b) extra dims softly to 0
    for (int i=6; i<(int)x.size(); ++i) pen += k_extra_ * sqr(x[i]);

    double f = w_pos_*pos_err2 + w_ori_*ori_err2 + k_pen_*pen;
    if (std::isnan(f) || std::isinf(f)) f = 1e12;
    return f;
}

void IK6DOF::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int i=0;i<(int)x.size(); ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

// ---------- helpers ----------
IK6DOF::Mat4 IK6DOF::matMul(const Mat4& A, const Mat4& B) const {
    Mat4 C{};
    for (int r=0;r<4;++r)
        for (int c=0;c<4;++c) {
            double s = 0.0;
            for (int k=0;k<4;++k) s += A.m[4*r+k]*B.m[4*k+c];
            C.m[4*r+c] = s;
        }
    return C;
}

IK6DOF::Mat4 IK6DOF::dh(double a, double alpha, double d, double theta) const {
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double ct = std::cos(theta), st = std::sin(theta);
    return Mat4{
        ct,   -st*ca,  st*sa,  a*ct,
        st,    ct*ca, -ct*sa,  a*st,
        0.0,     sa,     ca,     d,
        0.0,    0.0,    0.0,   1.0
    };
}

IK6DOF::Mat4 IK6DOF::forwardKinematics6(const double q6[6]) const {
    Mat4 T{ 1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1 };
    for (int i=0;i<6;++i) {
        Mat4 Ti = dh(a_[i], alpha_[i], d_[i], theta0_[i] + q6[i]);
        T = matMul(T, Ti);
    }
    return T;
}

void IK6DOF::rotToQuat(const Mat4& T, double& qw, double& qx, double& qy, double& qz) const {
    const double r00=T.m[0], r01=T.m[1], r02=T.m[2];
    const double r10=T.m[4], r11=T.m[5], r12=T.m[6];
    const double r20=T.m[8], r21=T.m[9], r22=T.m[10];
    const double tr = r00 + r11 + r22;

    if (tr > 0) {
        const double S = std::sqrt(tr + 1.0)*2.0;
        qw = 0.25*S;
        qx = (r21 - r12)/S;
        qy = (r02 - r20)/S;
        qz = (r10 - r01)/S;
    } else if (r00 > r11 && r00 > r22) {
        const double S = std::sqrt(1.0 + r00 - r11 - r22)*2.0;
        qw = (r21 - r12)/S; qx = 0.25*S;
        qy = (r01 + r10)/S; qz = (r02 + r20)/S;
    } else if (r11 > r22) {
        const double S = std::sqrt(1.0 + r11 - r00 - r22)*2.0;
        qw = (r02 - r20)/S;
        qx = (r01 + r10)/S; qy = 0.25*S; qz = (r12 + r21)/S;
    } else {
        const double S = std::sqrt(1.0 + r22 - r00 - r11)*2.0;
        qw = (r10 - r01)/S;
        qx = (r02 + r20)/S; qy = (r12 + r21)/S; qz = 0.25*S;
    }
    const double n = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (n>0){ qw/=n; qx/=n; qy/=n; qz/=n; }
}

double IK6DOF::quatAngleError(double qw1,double qx1,double qy1,double qz1,
                              double qw2,double qx2,double qy2,double qz2) const {
    double dot = qw1*qw2 + qx1*qx2 + qy1*qy2 + qz1*qz2;
    dot = std::max(-1.0, std::min(1.0, dot));
    dot = std::abs(dot); // q ~ -q
    return 2.0 * std::acos(dot);
}

} // namespace optimsolution
