#pragma once
#include "problem.h"

namespace optimsolution {

/**
 * Welded Beam Design (classic constrained engineering benchmark).
 *
 * Decision vector x (D=4): h=x0 (weld thickness), l=x1 (weld length),
 * t=x2 (bar height), b=x3 (bar thickness).
 *
 * f(x) = 1.10471*x0^2*x1 + 0.04811*x2*x3*(14.0+x1)
 *
 * Constraints g_i(x) <= 0:
 *   g1 = tau(x)  - tau_max
 *   g2 = sigma(x) - sigma_max
 *   g3 = x0 - x3
 *   g4 = 0.10471*x0^2 + 0.04811*x2*x3*(14.0+x1) - 5.0
 *   g5 = 0.125 - x0
 *   g6 = delta(x) - delta_max
 *   g7 = P - Pc(x)
 *
 * with (P=6000 lb, L=14 in, E=30e6 psi, G=12e6 psi fixed constants):
 *   tau'  = P / (sqrt(2)*x0*x1)
 *   M     = P*(L + x1/2)
 *   R     = sqrt(x1^2/4 + ((x0+x2)/2)^2)
 *   J     = 2*sqrt(2)*x0*x1*(x1^2/4 + ((x0+x2)/2)^2)
 *   tau'' = M*R/J
 *   tau(x)  = sqrt(tau'^2 + 2*tau'*tau''*x1/(2R) + tau''^2)
 *   sigma(x)= 6*P*L / (x3*x2^2)
 *   delta(x)= 4*P*L^3 / (E*x3*x2^3)
 *   Pc(x)   = (4.013*E*sqrt(x2^2*x3^6/36) / L^2) * (1 - (x2/(2L))*sqrt(E/(4G)))
 *
 * Domain: 0.1<=x0<=2, 0.1<=x1<=10, 0.1<=x2<=10, 0.1<=x3<=2.
 * Known best (literature): f* ~= 1.72485 at x* ~= (0.2057, 3.4705, 9.0366, 0.2057).
 *
 * NOTE: this is the same problem as the "welded beam design" example discussed
 * in Peng et al., "A Modified Sand Cat Swarm Optimization Algorithm..."
 * (Mathematics 2024, 12, 2153), Eq. (17). That paper's own Table 5/6 pairing
 * of "Reducer" and "Welded beam" objective values appears swapped (Table 5's
 * "1.6702" for the reducer and Table 6's "2994.4245" for the welded beam do
 * not match evaluating either formula at its own reported optimal point --
 * doing so instead reproduces the long-established literature optima for
 * each problem with the OTHER table's number). The formula and constants
 * here follow the paper's Eq. (17) exactly (with a couple of OCR-garbled
 * coefficients corrected to their standard literature form -- see
 * weldedbeam.cpp for details).
 */
class WeldedBeam : public Problem {
public:
    WeldedBeam();
    void init(int dim) override;                  // force D=4, set bounds

protected:
    double evaluate_core(const Vec& x) override;  // penalized objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    double P_, L_, E_, G_;
    double tau_max_, sigma_max_, delta_max_;
    double w_pen_;

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
