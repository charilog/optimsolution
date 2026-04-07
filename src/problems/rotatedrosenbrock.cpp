#include "rotatedrosenbrock.h"
#include <cmath>
#include <cassert>

namespace optimsolution {

RotatedRosenbrock::RotatedRosenbrock()
{
    setName("rotatedrosenbrock");
    setFullName("Rotated Rosenbrock function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");


    setKnownGlobalOptimum(0.0);
}

void RotatedRosenbrock::init(int dim) {
    if (dim < 2) dim = 2;        
    Problem::init(dim);

 
    Vec lo(dim, -5.0), hi(dim, 10.0);
    setBounds(lo, hi);

    // Default: identity rotation
    R_.clear();
    R_.shrink_to_fit();
}

void RotatedRosenbrock::setRotation(const std::vector<Vec>& R) {
    const int D = dimension();
    if ((int)R.size() != D) {
        
        R_.clear();
        return;
    }
    for (int i = 0; i < D; ++i) {
        if ((int)R[i].size() != D) {
            R_.clear();
            return;
        }
    }
    R_ = R;
}

void RotatedRosenbrock::applyRotation(const Vec& x, Vec& y) const {
    const int D = (int)x.size();
    y.assign(D, 0.0);

    if (R_.empty()) {
        y = x;
        return;
    }
    for (int r = 0; r < D; ++r) {
        double acc = 0.0;
        for (int c = 0; c < D; ++c) acc += R_[r][c] * x[c];
        y[r] = acc;
    }
}

void RotatedRosenbrock::applyRotationT(const Vec& gy, Vec& gx) const {
    const int D = (int)gy.size();
    gx.assign(D, 0.0);

    if (R_.empty()) {
        gx = gy;
        return;
    }
    for (int c = 0; c < D; ++c) {
        double acc = 0.0;
        for (int r = 0; r < D; ++r) acc += R_[r][c] * gy[r]; // (R^T)_{c,r} = R_{r,c}
        gx[c] = acc;
    }
}

// Rosenbrock σε y = R x:
// f(y) = Σ_{i=1}^{D-1} [100 (y_{i+1} - y_i^2)^2 + (1 - y_i)^2]
double RotatedRosenbrock::evaluate_core(const Vec& x) {
    const int D = dimension();
    Vec y;
    applyRotation(x, y);

    double f = 0.0;
    for (int i = 0; i < D - 1; ++i) {
        const double yi  = y[i];
        const double yip = y[i + 1];
        const double t1  = yip - yi * yi;
        const double t2  = 1.0 - yi;
        f += 100.0 * t1 * t1 + t2 * t2;
    }
    return f;
}

// Gradient:
// for i=1..D-1, t1 = y_{i+1} - y_i^2, t2 = 1 - y_i
// ∂f/∂y_i     = -400 y_i t1 - 2 t2
// ∂f/∂y_{i+1} =  200 t1
// Τέλος g_x = R^T g_y
void RotatedRosenbrock::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();

    Vec y;
    applyRotation(x, y);

    Vec gy(D, 0.0);
    for (int i = 0; i < D - 1; ++i) {
        const double yi  = y[i];
        const double yip = y[i + 1];
        const double t1  = yip - yi * yi;
        const double t2  = 1.0 - yi;

        gy[i]     += (-400.0 * yi * t1) - (2.0 * t2);
        gy[i + 1] += (200.0 * t1);
    }

    applyRotationT(gy, g);
}

} // namespace optimsolution
