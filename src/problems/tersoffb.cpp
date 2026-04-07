#include "tersoffb.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.141592653589793238462643383279502884; }

TersoffB::TersoffB()
: num_atoms_(10),
  D_(3 * num_atoms_ - 6),
  // ---- Si(B) parameters (from your reference) ----
  A_(1830.8),
  B_(471.18),
  lambda1_(2.4799),
  lambda2_(1.7322),
  beta_(1.1e-6),
  n_(0.78734),
  c_(100390.0),
  d_(16.217),
  h_(-0.59825),
  R_(2.85),          // (R'+S')/2 so that R-D=2.70 and R+D=3.00
  Dcut_(0.15),
  lambda3_(0.0),     // for Si(B) -> exp term = 1
  m_(3)
{
   
    setName("tersoffb");
    setFullName("TersoffB Si(B) Cluster Potential (CEC2011)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("molecular modelling / cluster optimization");

 
}

void TersoffB::init(int dim) {
    // Accept user-requested D if it's 3N-6; else default N=10 (D=24).
    if (dim >= 3 && (dim + 6) % 3 == 0) {
        int N = (dim + 6) / 3;
        if (N >= 3) {
            num_atoms_ = N;
            D_ = dim;
        }
    }
    Problem::init(D_);

    // Bounds
    Vec lo(D_, -4.25), hi(D_, 4.25);
    if (D_ >= 1) { lo[0] = 0.0;  hi[0] = 4.0;  } // x0
    if (D_ >= 2) { lo[1] = 0.0;  hi[1] = 4.0;  } // x1
    if (D_ >= 3) { lo[2] = 0.0;  hi[2] = PI;   } // x2
    setBounds(lo, hi);
}

double TersoffB::distance(const std::vector<double>& a, const std::vector<double>& b) {
    const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double TersoffB::angle_abc(const std::vector<double>& a,
                           const std::vector<double>& b,
                           const std::vector<double>& c) {
    // angle at b, between ba and bc
    double ux = a[0]-b[0], uy = a[1]-b[1], uz = a[2]-b[2];
    double vx = c[0]-b[0], vy = c[1]-b[1], vz = c[2]-b[2];
    const double nu = std::sqrt(ux*ux+uy*uy+uz*uz);
    const double nv = std::sqrt(vx*vx+vy*vy+vz*vz);
    if (nu == 0.0 || nv == 0.0) return 0.0;
    const double dot = (ux*vx + uy*vy + uz*vz) / (nu*nv);
    const double cth = std::max(-1.0, std::min(1.0, dot));
    return std::acos(cth);
}

std::vector<std::vector<double>>
TersoffB::reconstruct_positions(const Vec& x) const {
    std::vector<std::vector<double>> pos(num_atoms_, std::vector<double>(3, 0.0));

    // Atom 0 at origin
    // Atom 1 on +x at distance x0
    if (D_ >= 1) {
        pos[1][0] = x[0];
        pos[1][1] = 0.0;
        pos[1][2] = 0.0;
    }

    // Atom 2 in xy-plane with polar (r=x1, phi=x2)
    if (D_ >= 3) {
        const double r = x[1];
        const double s = std::sin(x[2]);
        const double c = std::cos(x[2]);
        pos[2][0] = r * c;
        pos[2][1] = r * s;
        pos[2][2] = 0.0;
    }

    // Atoms 3..N-1: Cartesian triples from x[3..]
    for (int i = 3; i < num_atoms_; ++i) {
        const int idx = 3 + 3*(i-3);
        if (idx + 2 < D_) {
            pos[i][0] = x[idx + 0];
            pos[i][1] = x[idx + 1];
            pos[i][2] = x[idx + 2];
        }
    }
    return pos;
}

double TersoffB::fc(double r) const {
    if (r <= (R_ - Dcut_)) return 1.0;
    if (r >= (R_ + Dcut_)) return 0.0;
    return 0.5 + 0.5 * std::cos(PI * (r - R_) / Dcut_);
}

double TersoffB::VR(double r) const { return A_ * std::exp(-lambda1_ * r); }
double TersoffB::VA(double r) const { return B_ * std::exp(-lambda2_ * r); }

double TersoffB::Bij(int i, int j, const std::vector<std::vector<double>>& pos) const {
    const double rij = distance(pos[i], pos[j]);
    double zeta = 0.0;

    for (int k = 0; k < num_atoms_; ++k) {
        if (k == i || k == j) continue;

        const double rik   = distance(pos[i], pos[k]);
        const double theta = angle_abc(pos[j], pos[i], pos[k]); // θ_ijk (vertex at i)

        // g(θ) = 1 + c^2/d^2 - c^2/(d^2 + (h - cosθ)^2)
        const double c2 = c_ * c_;
        const double d2 = d_ * d_;
        const double x  = h_ - std::cos(theta);
        const double g  = 1.0 + c2/d2 - c2/(d2 + x*x);

        double expo = 1.0;
        if (lambda3_ != 0.0) {
            const double diff = rij - rik;
            expo = std::exp(std::pow(lambda3_, m_) * std::pow(diff, m_));
        }
        zeta += fc(rik) * g * expo;
    }

    // B_ij = (1 + (β ζ)^n)^(-1/(2n))
    const double t = std::pow(beta_ * zeta, n_);
    return std::pow(1.0 + t, -1.0 / (2.0 * n_));
}

double TersoffB::evaluate_core(const Vec& x) {
    const auto pos = reconstruct_positions(x);

    double E = 0.0;
    for (int i = 0; i < num_atoms_; ++i) {
        double Ei = 0.0;
        for (int j = 0; j < num_atoms_; ++j) {
            if (i == j) continue;
            const double rij = distance(pos[i], pos[j]);
            const double fij = fc(rij);
            if (fij == 0.0) continue;

            const double vr  = VR(rij);
            const double va  = VA(rij);
            const double bij = Bij(i, j, pos);

            Ei += fij * (vr - bij * va);
        }
        E += 0.5 * Ei; // avoid double counting
    }

    if (std::isnan(E) || std::isinf(E)) E = 1e12;
    return E;
}

void TersoffB::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
