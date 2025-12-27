#pragma once
#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include "problem.h"
#include "options.h"

namespace optimsolution {


class Initializer {
public:
    void configure(const InitOptions& opt) { opt_ = opt; }

    // sampler
    std::vector<Vec> samplePopulation(const Problem& prob, std::mt19937_64& rng, int NP);

private:
    InitOptions opt_;

    // helpers 
    void sample_uniform (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X);
    void sample_normal  (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double sigma);
    void sample_cauchy  (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double gamma);
    void sample_laplace (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double b);
    void sample_lognorm (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double mu, double sigma);
    void sample_expon   (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double lambda);
    void sample_beta    (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double a, double b);
    void sample_levy    (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X, double c);
    void sample_lhs     (const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X);
    void sample_halton  (const Problem& prob, int NP, std::vector<Vec>& X);

    void sample_oppositional(const Problem& prob, std::mt19937_64& rng, int NP, std::vector<Vec>& X);

    // utilities
    static double clamp(double v, double lo, double hi) { return v<lo?lo:(v>hi?hi:v); }
    static double box_muller(std::mt19937_64& rng);
    static double cauchy(std::mt19937_64& rng, double gamma);
    static double laplace(std::mt19937_64& rng, double mu, double b);
    static double levy_mantegna(std::mt19937_64& rng, double c); // scale c

    static void scale01_to_bounds(const Problem& prob, std::vector<Vec>& X01, std::vector<Vec>& Xout);

    // Halton helpers
    static double halton_index(int index, int base);
    static std::vector<int> first_primes(int k);
};

} // namespace optimsolution
