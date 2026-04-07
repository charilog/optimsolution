#include "init.h"
#include "utils.h"   
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace optimsolution;


static constexpr double kPI = 3.141592653589793238462643383279502884L;

std::vector<Vec> Initializer::samplePopulation(const Problem& prob, std::mt19937_64& rng, int NP){
    std::vector<Vec> X;
    const int D = prob.dimension();
    if (NP <= 0 || D <= 0) return X;
    X.reserve(NP);

    const std::string t = toLower(opt_.type);

    if (t == "uniform") {
        sample_uniform(prob, rng, NP, X);

    } else if (t == "normal" || t == "gaussian") {
        double mu = opt_.getDbl("mu", 0.0);
        double sigma = std::max(1e-12, opt_.getDbl("sigma", 1.0));
        sample_normal(prob, rng, NP, X, mu, sigma);

    } else if (t == "cauchy") {
        double gamma = std::max(1e-12, opt_.getDbl("gamma", 1.0));
        sample_cauchy(prob, rng, NP, X, gamma);

    } else if (t == "laplace") {
        double mu = opt_.getDbl("mu", 0.0);
        double b  = std::max(1e-12, opt_.getDbl("b", 1.0));
        sample_laplace(prob, rng, NP, X, mu, b);

    } else if (t == "lognormal" || t == "lognorm") {
        double mu = opt_.getDbl("mu", 0.0);
        double sigma = std::max(1e-12, opt_.getDbl("sigma", 1.0));
        sample_lognorm(prob, rng, NP, X, mu, sigma);

    } else if (t == "exponential" || t == "exp") {
        double lambda = std::max(1e-12, opt_.getDbl("lambda", 1.0));
        sample_expon(prob, rng, NP, X, lambda);

    } else if (t == "beta") {
        double a = std::max(1e-12, opt_.getDbl("alpha", 2.0));
        double b = std::max(1e-12, opt_.getDbl("beta",  5.0));
        sample_beta(prob, rng, NP, X, a, b);

    } else if (t == "levy") {
        double c = std::max(1e-12, opt_.getDbl("c", 1.0));
        sample_levy(prob, rng, NP, X, c);

    } else if (t == "lhs") {
        sample_lhs(prob, rng, NP, X);

    } else if (t == "halton") {
        sample_halton(prob, NP, X);

    } else if (t == "oppositional") {
        sample_oppositional(prob, rng, NP, X);

    } else {
        // default
        sample_uniform(prob, rng, NP, X);
    }

    return X;
}

// ------------------ concrete samplers ------------------

void Initializer::sample_uniform(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    std::uniform_real_distribution<double> U01(0.0,1.0);
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            v[j] = L[j] + U01(rng)*(U[j]-L[j]);
        }
        X.push_back(std::move(v));
    }
}

double Initializer::box_muller(std::mt19937_64& rng){
    std::uniform_real_distribution<double> U(0.0,1.0);
    double u1 = std::max(1e-12, U(rng));
    double u2 = U(rng);
    return std::sqrt(-2.0*std::log(u1))*std::cos(2.0*kPI*u2);
}
void Initializer::sample_normal(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double sigma){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double z = mu + sigma * box_muller(rng);
            
            double m = 0.5*(L[j]+U[j]);
            double s = 0.25*(U[j]-L[j]); 
            double x = m + s * z;
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

double Initializer::cauchy(std::mt19937_64& rng, double gamma){
    std::uniform_real_distribution<double> U01(0.0,1.0);
    double u = U01(rng)-0.5;
    return std::tan(kPI * u) * gamma;
}
void Initializer::sample_cauchy(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double gamma){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double m = 0.5*(L[j]+U[j]);
            double s = 0.25*(U[j]-L[j]);
            double x = m + s * cauchy(rng, gamma);
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

double Initializer::laplace(std::mt19937_64& rng, double mu, double b){
    std::uniform_real_distribution<double> U01(0.0,1.0);
    double u = U01(rng) - 0.5;
    return mu - b * ((u<0)? 1.0 : -1.0) * std::log(1.0 - 2.0*std::abs(u));
}
void Initializer::sample_laplace(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double b){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double m = 0.5*(L[j]+U[j]);
            double s = 0.25*(U[j]-L[j]);
            double x = m + s * laplace(rng, mu, b);
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

void Initializer::sample_lognorm(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double sigma){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double z = std::exp(mu + sigma*box_muller(rng));
            double m = 0.5*(L[j]+U[j]);
            double s = 0.25*(U[j]-L[j]);
            double x = m + s*(z-1.0);
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

void Initializer::sample_expon(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double lambda){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    std::exponential_distribution<double> E(lambda);
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double m = 0.5*(L[j]+U[j]);
            double s = 0.25*(U[j]-L[j]);
            double x = m + s*(E(rng) - 1.0);
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

void Initializer::sample_beta(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double a, double b){
    const int D = prob.dimension();
    auto gamma_sample = [&](double k) {
        std::gamma_distribution<double> G(k, 1.0);
        return G(rng);
    };
    std::vector<Vec> X01; X01.reserve(NP);
    for(int i=0;i<NP;++i){
        Vec u(D,0.0);
        for(int j=0;j<D;++j){
            double x = gamma_sample(a);
            double y = gamma_sample(b);
            double t = x / (x+y + 1e-300);
            u[j] = t; // [0,1]
        }
        X01.push_back(std::move(u));
    }
    scale01_to_bounds(prob, X01, X);
}

double Initializer::levy_mantegna(std::mt19937_64& rng, double c){
    std::normal_distribution<double> N(0.0, 1.0);
    double u = N(rng) * std::pow(c, 1.0/1.5);
    double v = std::abs(N(rng));
    return u / std::pow(v, 1.0/1.5);
}
void Initializer::sample_levy(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double c){
    const int D = prob.dimension();
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for(int i=0;i<NP;++i){
        Vec v(D,0.0);
        for(int j=0;j<D;++j){
            double m = 0.5*(L[j]+U[j]);
            double s = 0.1*(U[j]-L[j]);
            double x = m + s * levy_mantegna(rng, c);
            v[j] = clamp(x, L[j], U[j]);
        }
        X.push_back(std::move(v));
    }
}

void Initializer::sample_lhs(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X){
    const int D = prob.dimension();
    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::vector<std::vector<double>> strata(D, std::vector<double>(NP));
    for (int j=0;j<D;++j){
        for (int i=0;i<NP;++i){
            strata[j][i] = (i + U01(rng)) / (double)NP;
        }
        std::shuffle(strata[j].begin(), strata[j].end(), rng);
    }
    std::vector<Vec> X01; X01.reserve(NP);
    for (int i=0;i<NP;++i){
        Vec u(D,0.0);
        for (int j=0;j<D;++j) u[j] = strata[j][i];
        X01.push_back(std::move(u));
    }
    scale01_to_bounds(prob, X01, X);
}

double Initializer::halton_index(int index, int base){
    double f = 1.0, r = 0.0; int i = index;
    while (i > 0) { f /= base; r += f * (i % base); i /= base; }
    return r;
}
std::vector<int> Initializer::first_primes(int k){
    std::vector<int> primes; primes.reserve(k);
    auto isPrime = [](int n){
        if (n<2) return false; if (n%2==0) return n==2;
        for (int d=3; d*d<=n; d+=2) if (n%d==0) return false;
        return true;
    };
    int x=2; while((int)primes.size()<k){ if(isPrime(x)) primes.push_back(x); ++x; }
    return primes;
}
void Initializer::sample_halton(const Problem& prob, int NP, std::vector<Vec>& X){
    const int D = prob.dimension();
    std::vector<int> bases = first_primes(D);
    std::vector<Vec> X01; X01.reserve(NP);
    for (int i=1;i<=NP;++i){
        Vec u(D,0.0);
        for (int j=0;j<D;++j){
            u[j] = halton_index(i, bases[j]);
        }
        X01.push_back(std::move(u));
    }
    scale01_to_bounds(prob, X01, X);
}

void Initializer::sample_oppositional(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X){
    int half = NP/2;
    std::vector<Vec> base;
    sample_uniform(prob, rng, std::max(1,half), base);
    X.reserve(NP);
    for (auto& v : base) X.push_back(v);
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    for (int i=0;i<NP - (int)base.size(); ++i){
        const Vec& a = base[i % base.size()];
        Vec b(a.size(),0.0);
        for (size_t j=0;j<a.size();++j) b[j] = L[j] + U[j] - a[j];
        for (size_t j=0;j<a.size();++j) b[j] = clamp(b[j], L[j], U[j]);
        X.push_back(std::move(b));
    }
}

void Initializer::scale01_to_bounds(const Problem& prob, std::vector<Vec>& X01, std::vector<Vec>& Xout){
    const Vec& L = prob.lb(); const Vec& U = prob.ub();
    const int D = prob.dimension();
    for (auto& u : X01){
        Vec v(D,0.0);
        for (int j=0;j<D;++j){
            double t = std::min(1.0, std::max(0.0, u[j]));
            v[j] = L[j] + t*(U[j]-L[j]);
        }
        Xout.push_back(std::move(v));
    }
}
