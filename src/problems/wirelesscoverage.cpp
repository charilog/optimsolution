#include "wirelesscoverage.h"

namespace optimsolution {

WirelessCoverage::WirelessCoverage() {

    setName("wirelesscoverage");
    setFullName("Wireless coverage planning (antenna placement & power)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("communication networks / coverage optimization");


    // default geometry (square km)
    W_ = 1000.0; H_ = 1000.0;

    // tx power bounds [dBm]
    pmin_dBm_ = 10.0;
    pmax_dBm_ = 30.0;

    // log-distance path-loss (typical urban-ish)
    // L(d)[dB] = L0 + 10*n*log10(d)
    L0_dB_ = 40.0;
    nPath_ = 3.0;

    // thresholds
    N0_dBm_     = -95.0; // noise floor
    Pmin_dBm_   = -80.0; // minimum RSRP for service
    SINRmin_dB_ = 6.0;   // minimum SINR
    Imax_dBm_   = -85.0; // soft cap on interference+noise

    // user grid
    Nx_ = 25; Ny_ = 25;
    build_grid();

    // weights
    w_uncovered_ = 1.0;
    w_sinr_      = 0.2;
    w_power_     = 0.02;

    // default antennas: M=2 → D=6
    M_  = 2;
    Dv_ = 3 * M_;
}

void WirelessCoverage::build_grid() {
    users_.clear();
    users_.reserve(Nx_ * Ny_);
    const double mx = W_ / (Nx_ + 1.0);
    const double my = H_ / (Ny_ + 1.0);
    for (int iy = 1; iy <= Ny_; ++iy)
        for (int ix = 1; ix <= Nx_; ++ix)
            users_.emplace_back(ix * mx, iy * my);
}

void WirelessCoverage::init(int dim) {
    // infer M from requested dim (closest multiple of 3, min 2 antennas)
    if (dim >= 3) M_ = std::max(2, dim / 3);
    else          M_ = 2;
    Dv_ = 3 * M_;

    Problem::init(Dv_);

    // bounds: (x,y) in [0,W_]x[0,H_], power in [pmin_dBm_, pmax_dBm_]
    Vec lo(Dv_, 0.0), hi(Dv_, 0.0);
    for (int i = 0; i < M_; ++i) {
        lo[3*i+0] = 0.0;       hi[3*i+0] = W_;
        lo[3*i+1] = 0.0;       hi[3*i+1] = H_;
        lo[3*i+2] = pmin_dBm_; hi[3*i+2] = pmax_dBm_;
    }
    setBounds(lo, hi);
}

double WirelessCoverage::evaluate_core(const Vec& x) {
    // clamp inside bounds for safety
    std::vector<double> xv(x);
    for (int i = 0; i < M_; ++i) {
        xv[3*i+0] = clamp(xv[3*i+0], 0.0,       W_);
        xv[3*i+1] = clamp(xv[3*i+1], 0.0,       H_);
        xv[3*i+2] = clamp(xv[3*i+2], pmin_dBm_, pmax_dBm_);
    }

    // precompute tx linear powers
    std::vector<double> tx_lin(M_);
    for (int i = 0; i < M_; ++i)
        tx_lin[i] = lin_from_dBm(xv[3*i+2]);

    const double noise_lin = lin_from_dBm(N0_dBm_);

    // coverage metrics
    int covered = 0;
    double sinr_pen_sum = 0.0;

    // iterate users
    for (const auto& u : users_) {
        // received power from each AP
        std::vector<double> pr_lin(M_);
        double best_pr_lin = 0.0;
        int    best_idx    = -1;

        for (int i = 0; i < M_; ++i) {
            const double dx = u.first  - xv[3*i+0];
            const double dy = u.second - xv[3*i+1];
            double d = std::sqrt(dx*dx + dy*dy);
            if (d < 1.0) d = 1.0; // avoid singularity, >= 1 m

            // path loss in dB, then received power in dBm
            const double LdB    = L0_dB_ + 10.0 * nPath_ * std::log10(d);
            const double pr_dBm = xv[3*i+2] - LdB;
            pr_lin[i] = lin_from_dBm(pr_dBm);

            if (pr_lin[i] > best_pr_lin) { best_pr_lin = pr_lin[i]; best_idx = i; }
        }

        // SINR for the strongest AP
        double interf_lin = noise_lin;
        for (int j = 0; j < M_; ++j)
            if (j != best_idx) interf_lin += pr_lin[j];

        const double S_dBm   = dBm_from_lin(best_pr_lin);
        const double I_dBm   = dBm_from_lin(interf_lin);
        const double SINR_dB = S_dBm - I_dBm; // SINR = S/(I+N) ⇒ dB difference

        // coverage test
        const bool ok = (S_dBm >= Pmin_dBm_) && (SINR_dB >= SINRmin_dB_);
        if (ok) covered++;
        else {
            double miss = 0.0;
            if (S_dBm   < Pmin_dBm_)   miss += (Pmin_dBm_   - S_dBm);
            if (SINR_dB < SINRmin_dB_) miss += (SINRmin_dB_ - SINR_dB);
            sinr_pen_sum += (miss * miss) * 0.02; // small per-user scale
        }

        // extra soft penalty when interference exceeds soft cap
        if (I_dBm > Imax_dBm_) {
            const double over = I_dBm - Imax_dBm_;
            sinr_pen_sum += 0.005 * over * over;
        }
    }

    const double uncovered_frac = 1.0 - (double)covered / (double)users_.size();

    // mean transmit power in dBm (normalized to [0,1] over [pmin,pmax])
    double mean_p_dBm = 0.0;
    for (int i = 0; i < M_; ++i) mean_p_dBm += xv[3*i+2];
    mean_p_dBm /= (double)M_;
    const double power_norm =
        (mean_p_dBm - pmin_dBm_) / std::max(1.0, pmax_dBm_ - pmin_dBm_);

    // final objective
    double cost = w_uncovered_ * uncovered_frac
                + w_sinr_      * (sinr_pen_sum / (double)users_.size())
                + w_power_     * power_norm;

    if (!(cost >= 0.0) || std::isnan(cost) || std::isinf(cost))
        cost = 1e6;
    return cost;
}

void WirelessCoverage::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        const double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
