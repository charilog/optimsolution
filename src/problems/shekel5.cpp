#include "shekel5.h"
#include <cmath>

namespace optimsolution {

Shekel5::Shekel5()
{
    setName("shekel5");
    setFullName("Shekel function (m = 5, D = 4)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Shekel5::init(int /*dim*/) {

    Problem::init(4);

    Vec lo(4, 0.0), hi(4, 10.0);
    setBounds(lo, hi);

    // global minimizer x* = (4,4,4,4), f* ≈ -10.1532
    Vec xopt(4, 4.0);
    setKnownGlobalOptimum(-10.1532, xopt);
}

double Shekel5::evaluate_core(const Vec& x) {
    const double A[5][4] = {
        {4.0, 4.0, 4.0, 4.0},
        {1.0, 1.0, 1.0, 1.0},
        {8.0, 8.0, 8.0, 8.0},
        {6.0, 6.0, 6.0, 6.0},
        {3.0, 7.0, 3.0, 7.0}
    };
    const double c[5] = {0.1, 0.2, 0.2, 0.4, 0.4};

    double f = 0.0;
    for (int i = 0; i < 5; ++i) {
        double d = 0.0;
        for (int j = 0; j < 4; ++j) {
            const double diff = x[j] - A[i][j];
            d += diff * diff;
        }
        f += -1.0 / (d + c[i]);
    }
    return f;
}

void Shekel5::gradient_core(const Vec& x, Vec& g) {
    g.assign(4, 0.0);

    const double A[5][4] = {
        {4.0, 4.0, 4.0, 4.0},
        {1.0, 1.0, 1.0, 1.0},
        {8.0, 8.0, 8.0, 8.0},
        {6.0, 6.0, 6.0, 6.0},
        {3.0, 7.0, 3.0, 7.0}
    };
    const double c[5] = {0.1, 0.2, 0.2, 0.4, 0.4};

    for (int i = 0; i < 5; ++i) {
        double d = 0.0;
        double diff[4];

        for (int j = 0; j < 4; ++j) {
            diff[j] = x[j] - A[i][j];
            d += diff[j] * diff[j];
        }

        const double denom = d + c[i];
        const double scale = 2.0 / (denom * denom); // dt/dx = 2*diff/(d+c)^2

        for (int k = 0; k < 4; ++k)
            g[k] += scale * diff[k];
    }
}

} // namespace optimsolution
