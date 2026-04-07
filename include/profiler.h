#pragma once
#include <vector>
#include <chrono>
#include <cstddef>
#include <algorithm>
#include <cmath> 

namespace optimsolution {

class Profiler {
public:
    using Clock = std::chrono::steady_clock;

    void reset() {
        iter_times_ms_.clear();
        run_start_ = run_end_ = Clock::time_point{};
        last_iter_start_ = Clock::time_point{};
        start_evals_ = end_evals_ = last_iter_evals_ = 0;
        peak_rss_kb_start_ = peak_rss_kb_end_ = 0;
    }

    void startRun(long long evals_so_far) {
        reset();
        run_start_ = Clock::now();
        start_evals_ = evals_so_far;
        peak_rss_kb_start_ = getPeakRSSKB();
    }
    void endRun(long long evals_so_far) {
        run_end_ = Clock::now();
        end_evals_ = evals_so_far;
        peak_rss_kb_end_ = getPeakRSSKB();
    }

    void startIter() {
        last_iter_start_ = Clock::now();
        last_iter_evals_ = end_evals_;
    }
    void endIter(long long evals_so_far) {
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - last_iter_start_).count();
        iter_times_ms_.push_back(ms);
        end_evals_ = evals_so_far;
    }

    // ---- accessors ----
    double runSeconds() const {
        if (run_start_.time_since_epoch().count()==0 || run_end_.time_since_epoch().count()==0) return 0.0;
        return std::chrono::duration<double>(run_end_ - run_start_).count();
    }
    std::size_t iters() const { return iter_times_ms_.size(); }
    long long evals() const { return std::max(0LL, end_evals_ - start_evals_); }

    double evalsPerSec() const {
        double s = runSeconds();
        if (s <= 0.0) return 0.0;
        return (double)evals() / s;
    }

    double meanIterMS() const {
        if (iter_times_ms_.empty()) return 0.0;
        double s=0.0; for (double x: iter_times_ms_) s+=x; return s / (double)iter_times_ms_.size();
    }
    double p50IterMS() const {
        if (iter_times_ms_.empty()) return 0.0;
        std::vector<double> v = iter_times_ms_;
        std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
        return v[v.size()/2];
    }
    double p95IterMS() const {
        if (iter_times_ms_.empty()) return 0.0;
        std::vector<double> v = iter_times_ms_;
        std::sort(v.begin(), v.end());
        std::size_t idx = (std::size_t)std::ceil(0.95*(v.size()-1));
        return v[idx];
    }

    long long peakRSSKBStart() const { return peak_rss_kb_start_; }
    long long peakRSSKBEnd()   const { return peak_rss_kb_end_; }
    long long peakRSSKB()      const { return std::max(peak_rss_kb_start_, peak_rss_kb_end_); }

    static long long getPeakRSSKB(); // platform-specific implementation

private:
    std::vector<double> iter_times_ms_;

    Clock::time_point run_start_{};
    Clock::time_point run_end_{};
    Clock::time_point last_iter_start_{};

    long long start_evals_{0};
    long long end_evals_{0};
    long long last_iter_evals_{0};

    long long peak_rss_kb_start_{0};
    long long peak_rss_kb_end_{0};
};

} // namespace optimsolution
