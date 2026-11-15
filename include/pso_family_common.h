#pragma once
// Shared helpers for the PSO-family methods spso2011, hclpso, olpso, glpso,
// apso and dmspso: velocity limits and box handling, so that all six treat
// the search box in the same, configurable way ("bound_mode" key).
#include <vector>
#include <string>
#include <cmath>
#include <cctype>
#include <limits>
#include <algorithm>

namespace optimsolution {
namespace psofamily {

// How a particle that leaves the box after x += v is treated.
//   Skip    : not moved back and NOT evaluated; its pbest is not updated
//             (CLPSO / HCLPSO reference MATLAB code, OLPSO paper). Particles
//             are drawn back by their attractors, which lie inside the box.
//   Clamp   : clipped to the box and evaluated; velocity unchanged.
//   Reflect : clipped to the box, the offending velocity components become
//             -0.5 * v_d, then evaluated (SPSO-2011 confinement, GL-PSO
//             reference implementation).
enum class BoundMode { Skip, Clamp, Reflect };

inline BoundMode parseBoundMode(std::string s, BoundMode def)
{
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "skip" || s == "noeval" || s == "none")  return BoundMode::Skip;
    if (s == "clamp" || s == "clip")                   return BoundMode::Clamp;
    if (s == "reflect" || s == "confine" || s == "bounce") return BoundMode::Reflect;
    return def;
}

inline const char* boundModeName(BoundMode m)
{
    switch (m) {
        case BoundMode::Skip:    return "skip";
        case BoundMode::Clamp:   return "clamp";
        case BoundMode::Reflect: return "reflect";
    }
    return "?";
}

// Vmax_d = frac * (ub_d - lb_d). A zero-width dimension gets Vmax = 0 (the
// coordinate can never move, so it can never leave the box); an unbounded
// dimension gets Vmax = +inf (no velocity clamping).
inline std::vector<double> velocityLimits(const std::vector<double>& L,
                                          const std::vector<double>& U,
                                          double frac)
{
    const size_t D = L.size();
    std::vector<double> vmax(D, 0.0);
    for (size_t d = 0; d < D; ++d) {
        const double span = U[d] - L[d];
        if (!std::isfinite(span)) vmax[d] = std::numeric_limits<double>::infinity();
        else if (span <= 0.0)     vmax[d] = 0.0;
        else                      vmax[d] = frac * span;
    }
    return vmax;
}

inline void clampVelocity(std::vector<double>& v, const std::vector<double>& vmax)
{
    for (size_t d = 0; d < v.size(); ++d) {
        if (v[d] >  vmax[d]) v[d] =  vmax[d];
        if (v[d] < -vmax[d]) v[d] = -vmax[d];
    }
}

// Applies the bound mode to x (and v) after the move x += v. Non-finite
// coordinates are always repaired (centre of the box, zero velocity).
// Returns true if the particle must be evaluated.
inline bool applyBounds(std::vector<double>& x, std::vector<double>& v,
                        const std::vector<double>& L, const std::vector<double>& U,
                        BoundMode mode)
{
    bool inside = true;
    for (size_t d = 0; d < x.size(); ++d) {
        if (!std::isfinite(x[d])) {
            x[d] = (std::isfinite(L[d]) && std::isfinite(U[d])) ? 0.5 * (L[d] + U[d]) : 0.0;
            v[d] = 0.0;
        }
        if (x[d] >= L[d] && x[d] <= U[d]) continue;
        switch (mode) {
            case BoundMode::Skip:
                inside = false;
                break;
            case BoundMode::Clamp:
                x[d] = std::min(U[d], std::max(L[d], x[d]));
                break;
            case BoundMode::Reflect:
                x[d] = std::min(U[d], std::max(L[d], x[d]));
                v[d] = -0.5 * v[d];
                break;
        }
    }
    return inside;
}

// Safety net for BoundMode::Skip: after this many consecutive generations
// in which NO particle could be evaluated, the out-of-range particles of the
// next generation are clipped and evaluated instead (prevents a run from
// spinning without evaluations in pathological cases).
constexpr int kMaxGenerationsWithoutEval = 1000;

} // namespace psofamily
} // namespace optimsolution
