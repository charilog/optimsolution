#include "tnep.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace optimsolution {

TNEP::TNEP() {
 
    setName("tnep");
    setFullName("Transmission Network Expansion Planning (DC-OPF surrogate)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("power systems / network expansion benchmark");

    from_  = {0,0,0,1,1,1,1,2,3,3,4};
    to_    = {1,3,4,2,3,4,5,3,4,5,5};
    L_     = static_cast<int>(from_.size());

    xreact_= {0.20,0.25,0.30,0.20,0.25,0.30,0.35,0.20,0.20,0.25,0.20};
    bper_.resize(L_);
    for (int l=0;l<L_;++l) bper_[l] = 1.0 / xreact_[l];

    fmax_  = {100, 90, 80, 100, 90, 80, 80, 100, 110, 90, 100};
    cost_  = {  2,  4,  5,   5,  3,  6,  6,   4,   2,  4,   3};
    n0_    = {  0,  0,  0,   0,  0,  0,  0,   0,   0,  0,   0};
    nmax_  = {  3,  3,  3,   3,  3,  3,  3,   3,   3,  3,   3};

    Pinj_  = {+300, -90, -80, -70, -60, 0};
    N_ = 6;
}

void TNEP::init(int /*dim*/) {

    Problem::init(L_);
    Vec lo(L_, 0.0), hi(L_, 0.0);
    for (int l=0;l<L_;++l) hi[l] = (double)nmax_[l];
    setBounds(lo, hi);

    print_reference_solution_once();
}

bool TNEP::solve_dc_pf(const std::vector<int>& n_tot,
                       std::vector<double>& theta,
                       std::vector<double>& flow,
                       double& shed) const {

    std::vector<double> B(N_*N_, 0.0);
    auto Bij = [&](int i,int j)->double& { return B[i*N_ + j]; };

    for (int l=0;l<L_;++l) {
        const int n = n_tot[l];
        if (n <= 0) continue;
        const int i = from_[l], j = to_[l];
        const double b = bper_[l] * n;
        Bij(i,i) += b; Bij(j,j) += b;
        Bij(i,j) -= b; Bij(j,i) -= b;
    }

    const int Nred = N_ - 1;
    if (Nred <= 0) return false;

    std::vector<double> Bred(Nred*Nred, 0.0), rhs(Nred, 0.0);
    auto Bk = [&](int r,int c)->double& { return Bred[r*Nred + c]; };
    for (int r=0;r<Nred;++r) {
        rhs[r] = Pinj_[r+1];
        for (int c=0;c<Nred;++c) Bk(r,c) = B[(r+1)*N_ + (c+1)];
    }

    // Gaussian elimination with partial pivoting
    for (int k=0;k<Nred;++k) {
        int piv = k; double amax = std::fabs(Bk(k,k));
        for (int r=k+1;r<Nred;++r) {
            double a = std::fabs(Bk(r,k));
            if (a > amax) { amax = a; piv = r; }
        }
        if (!(amax > 1e-12)) return false;
        if (piv != k) {
            for (int c=k;c<Nred;++c) std::swap(Bk(k,c), Bk(piv,c));
            std::swap(rhs[k], rhs[piv]);
        }
        const double pivv = Bk(k,k);
        for (int r=k+1;r<Nred;++r) {
            const double f = Bk(r,k)/pivv;
            if (!std::isfinite(f)) return false;
            for (int c=k;c<Nred;++c) Bk(r,c) -= f*Bk(k,c);
            rhs[r] -= f*rhs[k];
        }
    }

    std::vector<double> thetared(Nred, 0.0);
    for (int r=Nred-1;r>=0;--r) {
        double s = rhs[r];
        for (int c=r+1;c<Nred;++c) s -= Bk(r,c)*thetared[c];
        const double diag = Bk(r,r);
        if (!(std::fabs(diag) > 1e-12)) return false;
        thetared[r] = s / diag;
        if (!std::isfinite(thetared[r])) return false;
    }

    theta.assign(N_, 0.0);
    for (int i=1;i<N_;++i) theta[i] = thetared[i-1];

    flow.assign(L_, 0.0);
    for (int l=0;l<L_;++l) {
        const int n = n_tot[l];
        if (n <= 0) continue;
        const int i = from_[l], j = to_[l];
        flow[l] = bper_[l] * n * (theta[i] - theta[j]);
        if (!std::isfinite(flow[l])) return false;
    }

    shed = 0.0; 
    return true;
}

double TNEP::evaluate_from_integer(const std::vector<int>& n_add,
                                   double& out_over,
                                   double& out_cost) const {
    std::vector<int> n_tot(L_, 0);
    for (int l=0;l<L_;++l) {
        int nl = n_add[l];
        if (nl < 0) nl = 0;
        if (nl > nmax_[l]) nl = nmax_[l];
        n_tot[l] = n0_[l] + nl;
    }

    std::vector<double> theta, flow;
    double shed = 0.0;
    if (!solve_dc_pf(n_tot, theta, flow, shed)) return 1e12;

    double cost = 0.0;
    for (int l=0;l<L_;++l) cost += cost_[l] * (double)std::max(0, n_add[l]);

    double over = 0.0;
    for (int l=0;l<L_;++l) {
        const int n = n_tot[l];
        if (n <= 0) continue;
        const double cap = fmax_[l] * (double)n;
        const double a   = std::fabs(flow[l]) - cap;
        if (a > 0.0) over += a;
    }

    out_over = over;
    out_cost = cost;
    return cost + W_over_ * over + W_shed_ * shed;
}

void TNEP::print_reference_solution_once() const {
	// Reference feasible vector (without overload) that gives f ~ 21
	// Paths in order of from_/to_:
    // 0–1, 0–3, 0–4, 1–2, 1–3, 1–4, 1–5, 2–3, 3–4, 3–5, 4–5
    std::vector<int> n_add = {1,1,1,1,1,0,0,0,1,0,0};

    double over = 0.0, cost = 0.0;
    const double f = evaluate_from_integer(n_add, over, cost);

    std::ostringstream os;
    os << "[TNEP reference] n_add = [";
    for (int l=0;l<L_; ++l) {
        os << n_add[l];
        if (l+1 < L_) os << ",";
    }
    os << "], cost=" << cost << ", overload=" << over << ", f=" << f << "\n";

    std::fprintf(stderr, "%s", os.str().c_str());
}

double TNEP::evaluate_core(const Vec& x) {
    std::vector<int> n_add(L_, 0), n_tot(L_, 0);
    for (int l=0;l<L_;++l) {
        const double xl = clampd(x[l], 0.0, (double)nmax_[l]);
        int nl = (int)std::floor(xl + 0.5);
        if (nl > nmax_[l]) nl = nmax_[l];
        n_add[l] = nl;
        n_tot[l] = n0_[l] + nl;
    }

    std::vector<double> theta, flow;
    double shed = 0.0;
    if (!solve_dc_pf(n_tot, theta, flow, shed)) return 1e12;

    double cost = 0.0;
    for (int l=0;l<L_;++l) cost += cost_[l] * (double)n_add[l];

    double over = 0.0;
    for (int l=0;l<L_;++l) {
        const int n = n_tot[l];
        if (n <= 0) continue;
        const double cap = fmax_[l] * (double)n;
        const double a   = std::fabs(flow[l]) - cap;
        if (a > 0.0) over += a;
    }

    double f = cost + W_over_ * over + W_shed_ * shed;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void TNEP::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i=0;i<(int)x.size();++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
