#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

namespace optimsolution {

/// Return the set of dimensions a problem's init() actually accepts, or an
/// empty vector if the problem is either genuinely scalable (any D) or
/// already covered by getFixedDimOrZero() (exactly ONE valid D -- see
/// fixed_dims.h). This is deliberately a SEPARATE table rather than folded
/// into getFixedDimOrZero(), because that function's single-int return
/// value/semantics ("0 = flexible, >0 = the one valid D") is relied on by
/// many call sites across the GUI that only ever needed the "is this
/// problem fixed to one specific D" question -- this table instead answers
/// "which SET of D values does this problem's init() accept", for problems
/// that support MORE THAN ONE but not ALL dimensions (e.g. CEC2017's
/// D in {10,30,50}, CEC2022's D in {2,10,20} per the official technical
/// report). Names are matched case-insensitively.
inline std::vector<int> getAllowedDimsOrEmpty(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    // CEC2017 (F1, F3-F30): every implementation in this codebase only
    // embeds shift/rotation/shuffle data for D = 10, 30, 50 (see e.g.
    // cec2017_f1.h) and throws std::invalid_argument for anything else.
    static const std::vector<int> kCec2017Dims = {10, 30, 50};
    if (name.rfind("cec2017f", 0) == 0) return kCec2017Dims;

    // CEC2022 (all 12 functions in this codebase): D = 10, 20 only.
    // (Note: the official CEC2022 technical report's generic convention
    // also allows D=2, but this codebase's actual CEC2022 implementations
    // only support D=10/20 -- confirmed directly against the running
    // application's behavior, which is authoritative over the report's
    // generic convention here.)
    static const std::vector<int> kCec2022Dims = {10, 20};
    if (name.rfind("cec2022", 0) == 0) return kCec2022Dims;

    return {};
}

} // namespace optimsolution
