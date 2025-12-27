#include "sinusoidal.h"

namespace optimsolution {

Sinusoidal::Sinusoidal() {
    setName("sinusoidal");
    setFullName("Multidimensional sinusoidal test function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Sinusoidal::init(int dim) {
    int D = (dim > 0 ? dim : 1);
    Problem::init(D);

    const double PI = 3.1415926535897932384626433832795;
    Vec lo(D, 0.0), hi(D, PI);
    setBounds(lo, hi);
}

double Sinusoidal::evaluate_core(const Vec& x) {
    const int D = dimension();
    const double PI = 3.1415926535897932384626433832795;
    const double z = PI / 6.0;

    double p1 = 1.0;
    double p2 = 1.0;

    for (int i = 0; i < D; ++i) {
        const double t = x[i] - z;
        p1 *= std::sin(t);
        p2 *= std::sin(5.0 * t);
    }

    return -2.5 * p1 - p2;
}

void Sinusoidal::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    const double PI = 3.1415926535897932384626433832795;
    const double z = PI / 6.0;

    g.assign(D, 0.0);


    for (int i = 0; i < D; ++i) {
        double p1 = 1.0;
        double p2 = 1.0;

        // product over j ≠ i
        for (int j = 0; j < D; ++j) {
            if (j == i) continue;
            const double tj = x[j] - z;
            p1 *= std::sin(tj);
            p2 *= std::sin(5.0 * tj);
        }

        const double ti = x[i] - z;
        const double s1 = std::cos(ti);         // d/dx_i sin(x_i - z)
        const double s2 = std::cos(5.0 * ti);   // d/dx_i sin(5(x_i - z)) = 5 cos(5(...))

        // f = -2.5 * Π sin(x_k-z) - Π sin(5(x_k-z))
        // ∂f/∂x_i = -2.5 * cos(x_i-z)*Π_{j≠i} sin(x_j-z)
        //           - 5 * cos(5(x_i-z))*Π_{j≠i} sin(5(x_j-z))
        g[i] = -2.5 * s1 * p1 - 5.0 * p2 * s2;
    }
}

} // namespace optimsolution
