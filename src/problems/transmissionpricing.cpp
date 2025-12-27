#include "transmissionpricing.h"

namespace optimsolution {

TransmissionPricing::TransmissionPricing() {

    setName("transmissionpricing");
    setFullName("Transmission pricing via PTDF-based DC power flow");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("power systems / pricing benchmark");


    // Lines: (a,b,x,Fmax,alpha)
    lines_ = {
        {0,1, 0.10, 100.0, 1.0},
        {1,2, 0.08,  90.0, 1.2},
        {2,3, 0.07,  80.0, 1.5},
        {3,4, 0.09,  85.0, 1.1},
        {4,5, 0.11,  70.0, 1.6},
        {5,0, 0.10,  75.0, 1.3},
        {1,4, 0.12,  60.0, 1.8},
        {2,5, 0.10,  65.0, 1.7}
    };
    NL_ = (int)lines_.size();

    // Generators (3)
    gen_bus_ = {0, 1, 2};
    Pg_min_  = {10.0,  0.0,  0.0};
    Pg_max_  = {120.0, 90.0, 80.0};
    NG_ = (int)gen_bus_.size();

    // Base load buses & base demands (3)
    load_bus_base_ = {3, 4, 5};
    Pd_base_       = {70.0, 60.0, 50.0}; // MW
}

void TransmissionPricing::init(int dim) {
    // Decide ND from requested dim: if valid multiple of NG_, use it; else default ND = base size (3)
    if (dim >= NG_ && dim % NG_ == 0) ND_ = dim / NG_;
    else ND_ = (int)load_bus_base_.size();

    Dv_ = NG_ * ND_;
    Problem::init(Dv_);

    expand_loads_by_ND();

    // Bounds: GD_{i,j} in [0, max_ij], where max_ij ~ (sum Pg_max) / ND_
    double sumPgMax = 0.0;
    for (double v : Pg_max_) sumPgMax += v;
    const double max_ij = std::max(1.0, sumPgMax / std::max(1, ND_));

    Vec lo(Dv_, 0.0), hi(Dv_, max_ij);
    setBounds(lo, hi);
}

void TransmissionPricing::build_Bbus(std::vector<std::vector<double>>& Bbus) const {
    Bbus.assign(NB_, std::vector<double>(NB_, 0.0));
    for (const auto& L : lines_) {
        const double b = 1.0 / std::max(1e-9, L.x);
        Bbus[L.a][L.a] += b;
        Bbus[L.b][L.b] += b;
        Bbus[L.a][L.b] -= b;
        Bbus[L.b][L.a] -= b;
    }
}

bool TransmissionPricing::invert_dense(std::vector<std::vector<double>> A,
                                       std::vector<std::vector<double>>& Ainv) const {
    const int n = (int)A.size();
    Ainv.assign(n, std::vector<double>(n, 0.0));
    for (int i=0;i<n;++i) Ainv[i][i] = 1.0;

    for (int i=0;i<n;++i) {
        // pivot
        int piv = i; double p = std::fabs(A[i][i]);
        for (int r=i+1;r<n;++r) {
            if (std::fabs(A[r][i]) > p) { p = std::fabs(A[r][i]); piv = r; }
        }
        if (p < 1e-12) return false;
        if (piv != i) { std::swap(A[i], A[piv]); std::swap(Ainv[i], Ainv[piv]); }

        const double invp = 1.0 / A[i][i];
        for (int c=0;c<n;++c) { A[i][c] *= invp; Ainv[i][c] *= invp; }

        for (int r=0;r<n;++r) if (r!=i) {
            const double f = A[r][i];
            if (std::fabs(f) < 1e-18) continue;
            for (int c=0;c<n;++c) {
                A[r][c]    -= f * A[i][c];
                Ainv[r][c] -= f * Ainv[i][c];
            }
        }
    }
    return true;
}

bool TransmissionPricing::reduced_inverse(const std::vector<std::vector<double>>& M,
                                          int skip_row_col,
                                          std::vector<std::vector<double>>& Minv_red) const {
    const int n = (int)M.size();
    const int rN = n - 1;
    std::vector<std::vector<double>> R(rN, std::vector<double>(rN, 0.0));
    int rr=0;
    for (int i=0;i<n;++i) {
        if (i==skip_row_col) continue;
        int cc=0;
        for (int j=0;j<n;++j) {
            if (j==skip_row_col) continue;
            R[rr][cc++] = M[i][j];
        }
        ++rr;
    }
    return invert_dense(R, Minv_red);
}

void TransmissionPricing::ptdf_for_pair(int s_bus, int r_bus,
                                        const std::vector<std::vector<double>>& Binv_red,
                                        std::vector<double>& ptdf_lr) const {
    // e = e_s - e_r (NB_), reduce removing slack_ to e_red (NB_-1)
    std::vector<double> e_red(NB_-1, 0.0);
    int idx=0;
    for (int i=0;i<NB_;++i) {
        if (i==slack_) continue;
        if (i==s_bus) e_red[idx] += 1.0;
        if (i==r_bus) e_red[idx] -= 1.0;
        ++idx;
    }

    // theta_red = Binv_red * e_red
    std::vector<double> theta_red(NB_-1, 0.0);
    for (int i=0;i<NB_-1;++i) {
        double s=0.0;
        for (int j=0;j<NB_-1;++j) s += Binv_red[i][j]*e_red[j];
        theta_red[i]=s;
    }

    // reconstruct theta (slack=0)
    std::vector<double> theta(NB_, 0.0);
    idx=0;
    for (int i=0;i<NB_;++i) {
        if (i==slack_) continue;
        theta[i]=theta_red[idx++];
    }

    // PTDF on each line k: b_k * (theta[a]-theta[b])
    ptdf_lr.assign(NL_, 0.0);
    for (int k=0;k<NL_;++k) {
        const auto& L = lines_[k];
        const double bk = 1.0 / std::max(1e-9, L.x);
        ptdf_lr[k] = bk * (theta[L.a] - theta[L.b]);
    }
}

void TransmissionPricing::expand_loads_by_ND() {
    // Total base demand
    double Pd_total = 0.0;
    for (double v : Pd_base_) Pd_total += v;

    load_bus_.clear(); load_bus_.reserve(ND_);
    Pd_.clear();      Pd_.reserve(ND_);

    for (int j=0;j<ND_; ++j) {
        load_bus_.push_back(load_bus_base_[ j % (int)load_bus_base_.size() ]);
        Pd_.push_back(Pd_total / std::max(1, ND_));
    }
}

double TransmissionPricing::evaluate_core(const Vec& x) {
    // Build B and reduced inverse
    std::vector<std::vector<double>> Bbus; build_Bbus(Bbus);
    std::vector<std::vector<double>> Binv_red;
    if (!reduced_inverse(Bbus, slack_, Binv_red)) return 1e9;

    // Accumulators
    std::vector<double> flows(NL_, 0.0);
    std::vector<double> Gsum(NG_, 0.0);
    std::vector<double> Lsum(ND_, 0.0);

    // Sum flows using PTDF per (i,j)
    int idx = 0;
    for (int i=0;i<NG_; ++i) {
        const int s_bus = gen_bus_[i];
        for (int j=0;j<ND_; ++j) {
            const int r_bus = load_bus_[j];
            const double GDij = std::max(0.0, x[idx++]); // enforce ≥0
            if (GDij <= 0.0) continue;

            std::vector<double> ptdf;
            ptdf_for_pair(s_bus, r_bus, Binv_red, ptdf);

            for (int k=0;k<NL_; ++k) flows[k] += ptdf[k] * GDij;

            Gsum[i] += GDij;
            Lsum[j] += GDij;
        }
    }

    // 1) Usage-based line cost: sum alpha_k * |F_k| / Fmax_k
    double cost_usage = 0.0;
    for (int k=0;k<NL_; ++k) {
        const auto& L = lines_[k];
        const double norm = std::fabs(flows[k]) / std::max(1e-6, L.Fmax);
        cost_usage += L.alpha * norm;
    }

    // 2) Congestion penalty: quadratic on (|F|-Fmax)_+
    double pen_cong = 0.0;
    for (int k=0;k<NL_; ++k) {
        const auto& L = lines_[k];
        const double over = std::fabs(flows[k]) - L.Fmax;
        if (over > 0.0) {
            const double r = over / std::max(1.0, L.Fmax);
            pen_cong += W_cong_ * (r*r);
        }
    }

    // 3) Generator bounds penalty
    double pen_gen = 0.0;
    for (int i=0;i<NG_; ++i) {
        if (Gsum[i] < Pg_min_[i]) {
            const double r = (Pg_min_[i] - Gsum[i]) / std::max(1.0, Pg_min_[i]);
            pen_gen += W_gen_ * r * r;
        }
        if (Gsum[i] > Pg_max_[i]) {
            const double r = (Gsum[i] - Pg_max_[i]) / std::max(1.0, Pg_max_[i]);
            pen_gen += W_gen_ * r * r;
        }
    }

    // 4) Load service penalty (meet each Pd_j)
    double pen_load = 0.0;
    for (int j=0;j<ND_; ++j) {
        const double diff = Lsum[j] - Pd_[j];
        const double r = diff / std::max(1.0, Pd_[j]);
        pen_load += W_load_ * r * r;
    }

    // 5) System balance penalty
    double sumG=0.0, sumL=0.0;
    for (double v : Gsum) sumG += v;
    for (double v : Lsum) sumL += v;
    const double pen_sys = W_sys_ * std::pow((sumG - sumL) / std::max(1.0, sumL), 2.0);

    // 6) Light regularizer
    double reg = 0.0;
    for (double v : x) reg += reg_w_ * std::max(0.0, v);

    double f = cost_usage + pen_cong + pen_gen + pen_load + pen_sys + reg;
    if (!std::isfinite(f)) f = 1e9;
    return f;
}

void TransmissionPricing::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i=0;i<(int)x.size(); ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
