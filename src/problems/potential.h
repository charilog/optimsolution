#pragma once
#include "problem.h"
#include <stdexcept>

namespace optimsolution {

/**
 * Lennard–Jones pairwise potential (ε = σ = 1).
 *
 * Decision vector x ∈ R^{3N}:
 *   x = [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
 *
 *
 *   E(x) = Σ_{i<j} 4[(1/r_{ij})^{12} − (1/r_{ij})^{6}]
 *
 * with r_{ij} = ||r_i − r_j||_2.
 */
class Potential : public Problem {
public:
    Potential();

    // Initialize by number of atoms (dim = N, NOT 3N).
    void init(int dim) override;

    // Convenience: initialize explicitly by number of atoms.
    void initByAtoms(int nAtoms);

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    static inline double lj_energy(double r) {
        // 4 * ( (1/r)^12 - (1/r)^6 )
        const double rinv   = 1.0 / r;
        const double rinv2  = rinv * rinv;
        const double rinv6  = rinv2 * rinv2 * rinv2;
        const double rinv12 = rinv6 * rinv6;
        return 4.0 * (rinv12 - rinv6);
    }

    // d/dr [ 4*(r^-12 - r^-6) ] = 4*(-12 r^-13 + 6 r^-7)
    static inline double dE_dr(double r) {
        const double rinv   = 1.0 / r;
        const double rinv2  = rinv * rinv;
        const double rinv7  = rinv2 * rinv2 * rinv2 * rinv;      // r^-7
        const double rinv13 = rinv7 * rinv2 * rinv2 * rinv2;     // r^-13
        return 4.0 * (-12.0 * rinv13 + 6.0 * rinv7);
    }
};

} // namespace optimsolution
