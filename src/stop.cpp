#include "stop.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace optimsolution;

void StopController::configure(const TerminationOptions& to, int population){
    opt_ = to; NP_ = population; reset();
}
void StopController::reset(){
    iter_ = 0; stableCount_ = 0;
    last_best_ = last_worst_ = last_sumTop_ = last_sumBot_ = last_range_ = 0.0;
    last_impr_best_ = last_impr_worst_ = 0.0;
    last_var_ = last_var_at_improvement_ = 0.0;
    global_best_ = std::numeric_limits<double>::infinity();
}
double StopController::variance_(const std::vector<double>& fx) const{
    if (fx.size()<2) return 0.0;
    double m = std::accumulate(fx.begin(), fx.end(), 0.0)/fx.size();
    double acc=0.0; for(double v:fx){ double d=v-m; acc+=d*d; }
    return acc/(fx.size()-1);
}
bool StopController::checkBSS(double cur_best){
    double d = std::abs(cur_best - last_best_);
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_best_=cur_best;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkWSS(double cur_worst){
    double d = std::abs(cur_worst - last_worst_);
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_worst_=cur_worst;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkTSS(const std::vector<double>& sorted){
    int K = std::max(1, (int)std::ceil(NP_*opt_.sumRate));
    double s=0.0; for(int i=0;i<K;++i) s += sorted[i]; // οι καλύτεροι K (sorted asc)
    double d = std::abs(s - last_sumTop_);
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_sumTop_ = s;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkBOSS(const std::vector<double>& sorted){
    int K = std::max(1, (int)std::ceil(NP_*opt_.sumRate));
    double s=0.0; for(int i=0;i<K;++i) s += sorted[(int)sorted.size()-1 - i]; // χειρότεροι
    double d = std::abs(s - last_sumBot_);
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_sumBot_ = s;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkSRS(double cur_range){
    double d = std::abs(cur_range - last_range_);
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_range_=cur_range;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkIRS(double impr_best, double impr_worst){
    double cur = std::abs( (impr_worst) - (impr_best) );
    double d = std::abs(cur - (last_impr_best_ - last_impr_worst_));
    if (iter_>0 && d <= opt_.eps) ++stableCount_; else stableCount_=1;
    last_impr_best_=impr_best; last_impr_worst_=impr_worst;
    return (stableCount_>=opt_.sim);
}
bool StopController::checkDoublebox(const std::vector<double>& fx){
    double v = variance_(fx);
    bool improved = (fx.size()>0 && *std::min_element(fx.begin(),fx.end()) < global_best_);
    if (improved){ global_best_ = *std::min_element(fx.begin(),fx.end()); last_var_at_improvement_ = v; }
    bool stop = (v <= 0.5 * last_var_at_improvement_);
    return stop;
}
bool StopController::updateAndCheck(const std::vector<double>& fx){
    ++iter_;
    if (opt_.rule==StopRule::NONE) return false;
    std::vector<double> sorted = fx; std::sort(sorted.begin(), sorted.end());
    double best = sorted.front();
    double worst= sorted.back();
    double range= worst - best;

    auto one = [&](StopRule r){
        switch(r){
            case StopRule::BSS: return checkBSS(best);
            case StopRule::WSS: return checkWSS(worst);
            case StopRule::TSS: return checkTSS(sorted);
            case StopRule::BOSS:return checkBOSS(sorted);
            case StopRule::SRS: return checkSRS(range);
            case StopRule::IRS: {
                double impr_best  = (iter_>1? last_best_  - best  : 0.0);
                double impr_worst = (iter_>1? last_worst_ - worst : 0.0);
                return checkIRS(impr_best, impr_worst);
            }
            case StopRule::DOUBLEBOX: return checkDoublebox(fx);
            case StopRule::MAXEVALS:  return false;
            default: return false;
        }
    };

    if (opt_.rule==StopRule::ALL){
        StopRule rules[] = {StopRule::BSS,StopRule::WSS,StopRule::TSS,StopRule::BOSS,StopRule::SRS,StopRule::IRS,StopRule::DOUBLEBOX};
        for (auto r: rules){ if (one(r)) return true; }
        return false;
    } else {
        return one(opt_.rule);
    }
}
