#include "options.h"
#include <algorithm>

using namespace optimsolution;

int MethodConfig::getInt(const std::string& k, int def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
double MethodConfig::getDbl(const std::string& k, double def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}
bool MethodConfig::getBool(const std::string& k, bool def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    std::string v = it->second; std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v=="1" || v=="true" || v=="yes" || v=="on") return true;
    if (v=="0" || v=="false"|| v=="no"  || v=="off") return false;
    return def;
}
std::string MethodConfig::getStr(const std::string& k, const std::string& def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    return it->second;
}

// InitOptions
int InitOptions::getInt(const std::string& k, int def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
double InitOptions::getDbl(const std::string& k, double def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}
bool InitOptions::getBool(const std::string& k, bool def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    std::string v = it->second; std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v=="1" || v=="true" || v=="yes" || v=="on") return true;
    if (v=="0" || v=="false"|| v=="no"  || v=="off") return false;
    return def;
}
std::string InitOptions::getStr(const std::string& k, const std::string& def) const {
    auto it = kv.find(k); if (it==kv.end()) return def;
    return it->second;
}
