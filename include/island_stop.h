#pragma once
// Shared helpers for the island-model methods (pde, pga, ppso, psioa, psao).
//
// 1) Island early stopping vs. the [stop] rule.
//    These methods have their own per-island termination test (eps_stop/NM,
//    plateau, target) plus "stop when stop_after_islands islands are done".
//    That test used to be applied ALWAYS -- even with "[stop] rule =
//    maxevals", i.e. when the user asked the run to stop only at the
//    evaluation budget -- so these methods often used a few percent of the
//    budget (e.g. ppso 729 of 20000 evaluations) and could not be compared
//    fairly with the other methods. The per-method key
//        island_stop = auto | 1 | 0
//    now controls it: auto (default) = active only when [stop] rule is a
//    real stopping rule (not maxevals / none); 1 = always; 0 = never.
//
// 2) Minimum island size.
//    DE/GA/PSO-style operators need at least 4 members per island (target +
//    3 distinct others). With e.g. D = 1 (population 10) and 10 islands, each
//    island had ONE member: "while (r2 == r1)" loops never ended (hang) and
//    rand/1 indexed outside the island. The number of islands is reduced so
//    that every island has >= 4 members.

#include <string>
#include <algorithm>
#include <cctype>
#include "options.h"

namespace optimsolution {

// -1 = auto, 0 = off, 1 = on
inline int parseIslandStopMode(const MethodConfig& mc, int current = -1,
                               const char* key = "island_stop")
{
    std::string v = mc.getStr(key, "");
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back();
    while (!v.empty() && std::isspace(static_cast<unsigned char>(v.front()))) v.erase(v.begin());
    if (v.empty()) return current;
    if (v == "auto") return -1;
    if (v == "1" || v == "on" || v == "yes" || v == "true") return 1;
    if (v == "0" || v == "off" || v == "no" || v == "false") return 0;
    return current;
}

inline bool islandStopEnabled(int mode, const TerminationOptions& t)
{
    if (mode == 1) return true;
    if (mode == 0) return false;
    return !(t.rule == StopRule::MAXEVALS || t.rule == StopRule::NONE);
}

inline int effectiveIslands(int islands, int population, int minPerIsland = 4)
{
    const int maxIslands = std::max(1, population / std::max(1, minPerIsland));
    return std::max(1, std::min(islands, maxIslands));
}

} // namespace optimsolution
