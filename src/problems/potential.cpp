#include "potential.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Potential::Potential()
{
    setName("potential");
    setFullName("Lennard-Jones Pairwise Potential");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("molecular modelling / cluster optimization");
}

void Potential::init(int dim) {

    int nAtoms = dim;
    if (nAtoms <= 0)
        nAtoms = 1;

    const int D = 3 * nAtoms;   
    Problem::init(D);


    Vec l(D, -2.0), u(D,  2.0);
    setBounds(l, u);
}

void Potential::initByAtoms(int nAtoms) {
    if (nAtoms <= 0)
        throw std::invalid_argument("Potential::initByAtoms: nAtoms must be > 0");

    init(nAtoms);
}

double Potential::evaluate_core(const Vec& x) {
    const int D = dimension();
    const int N = D / 3;
    double value = 0.0;

    for (int i = 0; i < N; ++i) {
        const double xi = x[3*i + 0];
        const double yi = x[3*i + 1];
        const double zi = x[3*i + 2];

        for (int j = i + 1; j < N; ++j) {
            const double xj = x[3*j + 0];
            const double yj = x[3*j + 1];
            const double zj = x[3*j + 2];

            const double dx = xi - xj;
            const double dy = yi - yj;
            const double dz = zi - zj;
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double r  = std::sqrt(std::max(r2, 1e-18)); // avoid r=0

            value += lj_energy(r);
        }
    }
    return value;
}

// Gradient: ∂E/∂r_i = Σ_{j≠i} (dE/dr_ij) * ( (r_i - r_j)/r_ij ), pairwise antisymmetric
void Potential::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    const int N = D / 3;
    g.assign(D, 0.0);

    for (int i = 0; i < N; ++i) {
        const double xi = x[3*i + 0];
        const double yi = x[3*i + 1];
        const double zi = x[3*i + 2];

        for (int j = i + 1; j < N; ++j) {
            const double xj = x[3*j + 0];
            const double yj = x[3*j + 1];
            const double zj = x[3*j + 2];

            const double dx = xi - xj;
            const double dy = yi - yj;
            const double dz = zi - zj;
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double r  = std::sqrt(std::max(r2, 1e-18));

            const double scale = dE_dr(r) / r; // dE/dr * (1/r)

            const double gx = scale * dx;
            const double gy = scale * dy;
            const double gz = scale * dz;

            g[3*i + 0] += gx;
            g[3*i + 1] += gy;
            g[3*i + 2] += gz;

            g[3*j + 0] -= gx;
            g[3*j + 1] -= gy;
            g[3*j + 2] -= gz;
        }
    }
}

} // namespace optimsolution
