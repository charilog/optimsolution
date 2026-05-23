#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>

namespace optimsolution {

class NLSHADELBC : public Optimizer {
public:
    NLSHADELBC() = default;
    ~NLSHADELBC() override = default;

    std::string methodShortName() const override { return "NL-SHADE-LBC"; }
    std::string methodFullName()  const override { return "NL-SHADE-LBC (Non-Linear SHADE with Linear Bias Change)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    inline double eval(const Vec& x) {
        double f = prob_->evaluate(x);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    static std::string toLower(std::string s);
    static std::string trim(std::string s);
    static bool parseBool(const std::string& s, bool fb);
    static int parseInt(const std::string& s, int fb);
    static double parseDouble(const std::string& s, double fb);

    bool inBounds(const Vec& x) const;
    void midpointTargetRepair(Vec& u, const Vec& x);

    double sampleNormalClamped(double mean, double sd, double lo, double hi);
    double samplePositiveCauchyClamped(double location, double scale, double hi);
    int uniformIndex(int n);
    int chooseUniformPopulationIndex(int exclude1=-1, int exclude2=-1, int exclude3=-1);
    int chooseUniformArchiveIndex();
    int chooseRankBasedPopulationIndex(const std::vector<int>& sorted_idx,
                                       std::discrete_distribution<int>& dist,
                                       int exclude1=-1, int exclude2=-1, int exclude3=-1);
    int chooseTopPBestIndex(const std::vector<int>& sorted_idx, int pbest_count, int exclude1=-1);
    Vec binomialCrossover(const Vec& target, const Vec& donor, double cr);

    double generalizedLehmerMean(const std::vector<double>& S,
                                 const std::vector<double>& weights,
                                 double p, double m) const;
    double currentPFExponent() const;
    double currentPCRExponent() const;
    int currentPBestCount(int np) const;
    int currentPopulationSizeByNLPSR() const;
    void updateArchive(const Vec& replaced_x, double replaced_f, int target_archive_size);
    void shrinkPopulationAndArchive(int next_np, int next_na);

private:
    std::vector<Vec> X_;
    std::vector<double> FX_;

    std::vector<Vec> archive_X_;
    std::vector<double> archive_F_;

    std::vector<double> MF_;
    std::vector<double> MCR_;
    int memory_index_ = 0;
    int initial_population_ = -1;

    int pop_override_ = -1;
    bool use_global_population_ = false;
    bool paper_population_default_ = true;

    int H_ = 20;
    int min_pop_ = 4;
    double archive_ratio_ = 1.0;
    double archive_use_prob_ = 0.5;
    double rank_pressure_k_ = 4.0;
    int max_resamples_ = 100;

    double mf_init_ = 0.5;
    double mcr_init_ = 0.9;
    double memory_c_ = 0.5;

    double pbest_start_ = 0.2;
    double pbest_end_ = 0.3;

    double mean_m_ = 1.5;
    double p_init_f_ = 3.5;
    double p_final_f_ = 1.5;
    double p_init_cr_ = 1.0;
    double p_final_cr_ = 1.5;

    bool end_local_refine_ = false;
    std::string end_local_method_;

    int debug_nlshadelbc_ = 0;
};

} // namespace optimsolution
