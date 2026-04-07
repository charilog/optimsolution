#pragma once
#include "problem.h"
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * WirelessCoverage — antenna placement & power optimization (2D plane)
 *
 * Decision vector x groups M antennas as triplets: [x0,y0,p0_dBm, x1,y1,p1_dBm, ...]
 * Dimension D = 3*M (M inferred from init(dim): if dim<3 -> M=2 => D=6).
 *
 * Objective (minimize):
 *   w_uncovered * (1 - coverage_ratio)
 * + w_sinr      * mean_sinr_penalty
 * + w_power     * normalized_mean_tx_power
 *
 * Path-loss: L(d)[dB] = L0_dB + 10 * nPath * log10(d), d>=1 m.
 * SINR[dB]  = S_dBm - (10*log10( sum_j≠best Pr_j_lin + N0_lin ))
 */
class WirelessCoverage : public Problem {
public:
    WirelessCoverage();
    void   init(int dim) override;                   // set D=3*M, bounds

protected:
    double evaluate_core(const Vec& x) override;     // objective
    void   gradient_core(const Vec& x, Vec& g) override; // numeric gradient

private:
    // geometry
    double W_, H_;                // area [m]
    // tx power bounds [dBm]
    double pmin_dBm_, pmax_dBm_;
    // path-loss model
    double L0_dB_, nPath_;
    // thresholds [dBm or dB]
    double N0_dBm_, Pmin_dBm_, SINRmin_dB_, Imax_dBm_;
    // user grid
    int Nx_, Ny_;
    std::vector<std::pair<double,double>> users_; // (ux,uy)
    // weights
    double w_uncovered_, w_sinr_, w_power_;

    // dimensioning
    int M_;     // antennas
    int Dv_;    // = 3*M_

    // helpers
    void build_grid();
    inline double clamp(double v, double lo, double hi) const {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    inline double lin_from_dBm(double dBm) const {
        return std::pow(10.0, dBm / 10.0);
    }
    inline double dBm_from_lin(double lin) const {
        return 10.0 * std::log10(std::max(lin, 1e-30));
    }
};

} // namespace optimsolution
