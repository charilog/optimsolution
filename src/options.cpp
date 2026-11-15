#include "options.h"
#include <algorithm>
#include <cctype>

using namespace optimsolution;

// -----------------------------------------------------------------------------
// BUGFIX (key case): Config::load() lower-cases every key it reads from
// optimsolution.cfg, but many methods query mixed-case keys ("H", "F", "CR",
// "cauchy_scale_F", ...). Those lookups never matched, so the corresponding
// cfg values were silently ignored and the compiled defaults were used.
// Lookup order: exact key -> lower-cased key -> case-insensitive scan, so
// every existing caller (including maps filled with mixed-case keys, e.g. by
// the sensitivity sweep) keeps working.
// -----------------------------------------------------------------------------
namespace {
using KV = std::unordered_map<std::string, std::string>;

std::string lowerKey(const std::string& s) {
    std::string t = s;
    for (auto& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return t;
}

KV::const_iterator findKey(const KV& kv, const std::string& k) {
    auto it = kv.find(k);
    if (it != kv.end()) return it;
    const std::string lk = lowerKey(k);
    if (lk != k) {
        it = kv.find(lk);
        if (it != kv.end()) return it;
    }
    for (auto jt = kv.begin(); jt != kv.end(); ++jt)
        if (lowerKey(jt->first) == lk) return jt;
    return kv.end();
}
} // namespace

int MethodConfig::getInt(const std::string& k, int def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
double MethodConfig::getDbl(const std::string& k, double def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}
bool MethodConfig::getBool(const std::string& k, bool def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    std::string v = it->second; std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v=="1" || v=="true" || v=="yes" || v=="on") return true;
    if (v=="0" || v=="false"|| v=="no"  || v=="off") return false;
    return def;
}
std::string MethodConfig::getStr(const std::string& k, const std::string& def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    return it->second;
}

// InitOptions
int InitOptions::getInt(const std::string& k, int def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
double InitOptions::getDbl(const std::string& k, double def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}
bool InitOptions::getBool(const std::string& k, bool def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    std::string v = it->second; std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v=="1" || v=="true" || v=="yes" || v=="on") return true;
    if (v=="0" || v=="false"|| v=="no"  || v=="off") return false;
    return def;
}
std::string InitOptions::getStr(const std::string& k, const std::string& def) const {
    auto it = findKey(kv, k); if (it==kv.end()) return def;
    return it->second;
}
