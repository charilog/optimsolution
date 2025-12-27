#ifndef OPTIMSOLUTION_FUSE_H
#define OPTIMSOLUTION_FUSE_H

#include "optimizer.h"
#include <vector>

namespace optimsolution {

class FUSE : public Optimizer
{
public:
    FUSE()  = default;
    ~FUSE() override = default;

    std::string methodShortName() const override { return "fuse"; }
    std::string methodFullName()  const override {
        return "Fusion Search Ensemble (FUSE)";
    }

    void configure(const MethodConfig &mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // Population
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;

    int pop_override_{-1};

    // LSHADE-style parameters
    double muF_{0.5};
    double muCR_{0.8};
    double sh_c_{0.1};     // learning rate for memory

    double F_lo_{0.1};
    double F_hi_{0.9};
    double CR_lo_{0.0};
    double CR_hi_{1.0};

    double pbest_frac_{0.2};
    double archive_rate_{2.0}; // × population size

    // Stagnation / micro-restart
    int    stagnation_trigger_{20};
    int    stagn_iters_{0};
    double restart_frac_{0.2};   // fraction of worst individuals to be restarted
    double restart_sigma_{0.30}; // Gaussian scale around the best

    // Helpers
    static inline double clamp_(double v, double lo, double hi)
    {
        return (v < lo ? lo : (v > hi ? hi : v));
    }

    double eval(const Vec &x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec &v);

    // RNG helpers (not const because they use rng_)
    int  pickDistinct_(int n, int a=-1, int b=-1, int c=-1, int d=-1);
    int  pickPbestIndex_(const std::vector<int> &sorted_idx);

    void pushArchive_(const Vec &x);
    void microRestart_();

    // Operator selection: 0 = LSHADE, 1 = BEST2
    int  selectOperator_(double eval_ratio, double rank_q);

    // Operators
    void opLSHADE_(int i,
                   const std::vector<int> &sorted_idx,
                   double F, double CR,
                   Vec &trial);

    void opBEST2_(int i, double F, double CR, Vec &trial);
};

} // namespace optimsolution

#endif // OPTIMSOLUTION_FUSE_H
