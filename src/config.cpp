#include "config.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>

using namespace optimsolution;

static std::string ltrim(std::string s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a]))
        ++a;
    return s.substr(a);
}
static std::string rtrim(std::string s)
{
    size_t b = s.size();
    while (b > 0 && std::isspace((unsigned char)s[b - 1]))
        --b;
    s.resize(b);
    return s;
}
static std::string trim(const std::string &s) { return rtrim(ltrim(s)); }

static std::string strip_inline_comment(std::string v)
{
    auto cut2 = v.find("//");
    if (cut2 != std::string::npos)
        v = v.substr(0, cut2);
    auto cut1 = v.find(';');
    if (cut1 != std::string::npos)
        v = v.substr(0, cut1);
    auto cut3 = v.find('#');
    if (cut3 != std::string::npos)
        v = v.substr(0, cut3);
    return trim(v);
}

static StopRule parseStopRule(std::string v)
{
    v = toLower(trim(v));
    if (v == "none" || v.empty()) return StopRule::NONE;
    if (v == "bss")               return StopRule::BSS;
    if (v == "wss")               return StopRule::WSS;
    if (v == "tss")               return StopRule::TSS;
    if (v == "boss")              return StopRule::BOSS;
    if (v == "srs")               return StopRule::SRS;
    if (v == "irs")               return StopRule::IRS;
    if (v == "doublebox")         return StopRule::DOUBLEBOX;
    if (v == "maxevals" || v == "max_evals" || v == "maxeval") return StopRule::MAXEVALS;
    if (v == "all")               return StopRule::ALL;
    return StopRule::NONE;
}

static std::string to_lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

Config Config::load(const std::string &path, const std::string &methodName)
{
    Config c;
    std::ifstream in(path);
    if (!in) return c;

    std::string line, section;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> S;

    bool firstLine = true;
    while (std::getline(in, line))
    {
        if (firstLine && line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
        { line.erase(0, 3); }
        firstLine = false;

        std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';' || (t.size() > 1 && t[0] == '/' && t[1] == '/'))
            continue;

        if (t.front() == '[' && t.back() == ']')
        { section = toLower(trim(t.substr(1, t.size() - 2))); continue; }

        auto pos = t.find('=');
        if (pos == std::string::npos) continue;

        std::string k = toLower(trim(t.substr(0, pos)));
        std::string v = strip_inline_comment(t.substr(pos + 1));
        S[section][k] = v;
    }

    auto getInt = [&](const std::string &sec, const std::string &k, int def)
    { try { return std::stoi(S[sec].at(k)); } catch (...) { return def; } };
    auto getDbl = [&](const std::string &sec, const std::string &k, double def)
    { try { return std::stod(S[sec].at(k)); } catch (...) { return def; } };
    auto getStr = [&](const std::string &sec, const std::string &k, std::string def)
    { auto it = S[sec].find(k); return it == S[sec].end() ? def : trim(it->second); };

    // [global]
    c.g.population  = getInt("global", "population",  c.g.population);
    c.g.max_iters   = getInt("global", "max_iters",   c.g.max_iters);
    c.g.max_evals   = (long long)getInt("global", "max_evals", (int)c.g.max_evals);
    try { c.g.seed_base = (unsigned long long)std::stoull(getStr("global", "seed_base", std::to_string(c.g.seed_base))); } catch (...) {}
    c.g.runs        = getInt("global", "runs",        c.g.runs);

    // final local search (in end())
    {
        int elr = getInt("global", "end_local_refine", 0);
        if (elr == 0) { elr = getInt("global", "local_refine", 0); } // back-compat
        c.g.end_local_refine = (elr != 0);
        c.g.end_local_method = toLower(getStr("global", "end_local_method", getStr("global", "local_method", "")));
    }

    // CSV 
    {
        c.g.csv_enable      = getInt("global", "csv_enable",      c.g.csv_enable ? 1 : 0) != 0;
        c.g.csv_convergence = getInt("global", "csv_convergence", c.g.csv_convergence ? 1 : 0) != 0;
        c.g.csv_summary     = getInt("global", "csv_summary",     c.g.csv_summary ? 1 : 0) != 0;
        c.g.csv_prefix      = getStr("global", "csv_prefix",      c.g.csv_prefix);
    }

    // [stop]
    c.t.rule    = parseStopRule(getStr("stop", "rule", "none"));
    c.t.eps     = getDbl("stop", "eps", c.t.eps);
    c.t.sim     = getInt("stop", "sim", c.t.sim);
    c.t.sumRate = getDbl("stop", "sumrate", c.t.sumRate);

    // [init]
    c.init.type = toLower(getStr("init", "type", "uniform"));
    if (S.count("init")){
        for (auto &kv : S["init"]){
            if (kv.first == "type") continue;
            c.init.kv[kv.first] = kv.second;
        }
    }

    // [methodName]
    std::string msec = toLower(methodName);
    if (S.count(msec)){
        for (auto &kv : S[msec])
            c.methodKV.kv[kv.first] = kv.second;
    }

    // [global] defaults for in-run local search (used when the method section does not override them)
    {
        auto git = S.find("global");
        if (git != S.end()){
            const auto &G = git->second;
            auto putIfMissing = [&](const std::string &key){
                auto itg = G.find(key);
                if (itg != G.end() && c.methodKV.kv.find(key) == c.methodKV.kv.end())
                    c.methodKV.kv[key] = trim(itg->second);
            };
            putIfMissing("local_refine");
            putIfMissing("local_method");
            putIfMissing("local_rate");
        }
    }

    // [sensitivity] (ό,τι έχεις ήδη)
    {
        auto sit = S.find("sensitivity");
        if (sit != S.end()){
            const auto &M = sit->second;
            auto getIntS = [&](const std::string &k, int def)
            { try { return std::stoi(M.at(k)); } catch(...) { return def; } };
            auto getStrS = [&](const std::string &k, const std::string &def)
            { auto it=M.find(k); return it==M.end()?def:trim(it->second); };

            c.sens.enabled    = getIntS("enabled", c.sens.enabled ? 1 : 0) != 0;
            c.sens.mode       = getStrS("mode", c.sens.mode);
            c.sens.output_csv = getStrS("output", c.sens.output_csv);

            std::string pl = getStrS("params", "");
            if (!pl.empty()){
                std::stringstream ss(pl); std::string tok;
                while (std::getline(ss, tok, ',')){
                    tok = trim(tok);
                    if (!tok.empty()) c.sens.params.push_back(to_lower_copy(tok));
                }
                std::sort(c.sens.params.begin(), c.sens.params.end());
                c.sens.params.erase(std::unique(c.sens.params.begin(), c.sens.params.end()), c.sens.params.end());
            }
            for (const auto &kv : M){
                std::string k = trim(kv.first);
                if (k.rfind("values.", 0) == 0){
                    std::string pname = to_lower_copy(trim(k.substr(7)));
                    std::string vlist = trim(kv.second);
                    std::vector<double> vals;
                    std::stringstream ss(vlist); std::string tok;
                    while (std::getline(ss, tok, ',')){
                        tok = trim(tok);
                        if (!tok.empty()){ try { vals.push_back(std::stod(tok)); } catch (...) {} }
                    }
                    if (!pname.empty() && !vals.empty()){
                        c.sens.values[pname] = std::move(vals);
                        if (std::find(c.sens.params.begin(), c.sens.params.end(), pname) == c.sens.params.end())
                            c.sens.params.push_back(pname);
                    }
                }
            }
            std::sort(c.sens.params.begin(), c.sens.params.end());
            c.sens.params.erase(std::unique(c.sens.params.begin(), c.sens.params.end()), c.sens.params.end());
        }
    }

    return c;
}
