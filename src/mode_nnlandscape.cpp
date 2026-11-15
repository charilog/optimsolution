// ---------------------------------------------------------------------
// Mode 7: neural-network training + local landscape sampling ("--nnlandscape").
//
// Trains a small feed-forward network (NNProblem, src/problems/nn_problem.h)
// with any optimizer already registered in makeMethod() -- gradient-based
// (lbfgs, bfgs, gd) or population-based (de, pso, cmaes, jso, ...) alike,
// since NNProblem is just another optimsolution::Problem and every existing
// optimizer already knows how to minimize one. After training, it samples
// the training-loss landscape locally around the found minimum at several
// radius fractions of the weight box and writes:
//   - one pflacco-ready landscape-sample CSV per radius, in the SAME column
//     layout Mode 5 (--features) already writes, so the existing ELA
//     extraction step needs no changes to consume either;
//   - one row appended to a run-summary CSV recording the generalization
//     gap (validation loss minus training loss) and cheap perturbation-
//     sharpness statistics computed from the same local samples.
//
// Self-contained: does not touch main(), makeProblem(), makeMethod(), or
// any other mode's code. main.cpp only needs a two-line dispatch hook (see
// the accompanying integration notes) to route "--nnlandscape" here.
// ---------------------------------------------------------------------
#include "problem.h"
#include "optimizer.h"
#include "factory.h"
#include "localsearch.h"
#include "problems/nn_problem.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace optimsolution {

namespace {

struct NNLandscapeArgs {
    std::string dataset;              // path to "<name>.train"; ".test" is auto-derived
    std::string hidden      = "16";
    std::string activation  = "relu";
    std::string optimizer   = "de";
    int    iterations       = 200;
    int    population       = 40;
    double wBound           = 5.0;
    unsigned seed            = 42;
    std::string radii        = "0.02,0.05,0.1";
    int    localSamples      = 0;      // 0 => auto = max(50, 30*dim)
    std::string outPrefix;             // required
    std::string outSummary;            // required
    std::string label;
    // OFF by default: the *_features.csv sampled-point coordinates (x1..xD,
    // one column per weight) are the dominant cost of this mode's disk
    // footprint -- for a D-dimensional network with the default "Auto"
    // sample count (30*D per radius), that's O(D^2) values per radius per
    // run, which reaches into the hundreds of MB to low GB *per run* for
    // the larger networks in a real batch (D in the hundreds), and was
    // confirmed to be the actual cause of a 1.5+ TB optimsolution_gui_run
    // folder partway through a real 30-dataset x 6-optimizer x 30-seed
    // batch. Nothing in this mode's own analysis pipeline
    // (py_scripts/train_gap_regressor.py, and the GUI's Results/Mean/
    // Wilcoxon/Rankings/LODO tables, all driven by nn_summary.csv) ever
    // reads the x-columns -- only the scalar loss value "f" is used, to
    // compute mean/std/min/max/range/iqr/skew/kurtosis/cv per radius. The
    // one real consumer of the x-columns is the separate, optional
    // py_scripts/extract_ela_features.py (genuine pflacco ELA features,
    // which do need the actual sampled point positions, not just their
    // loss) -- pass --save-x-columns to keep writing them for that use.
    bool   saveXColumns      = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage:\n  %s --nnlandscape --dataset <path.train> --out-prefix <prefix> --out-summary <summary.csv>\n"
        "     [--hidden 16] [--activation relu|tanh|sigmoid] [--optimizer de]\n"
        "     [--iterations 200] [--population 40] [--wbound 5.0] [--seed 42]\n"
        "     [--radii 0.02,0.05,0.1] [--local-samples 0] [--label pilot]\n"
        "     [--save-x-columns]\n"
        "\noptimizer: any name registered in makeMethod() (de, pso, cmaes, jso, sa, lbfgs, bfgs, gd, nm, cg, ncg, adam, ...)\n"
        "--save-x-columns: also write the x1..xD sampled-point coordinates to each\n"
        "  *_features.csv (needed only for py_scripts/extract_ela_features.py's full\n"
        "  pflacco ELA extraction; OFF by default since it can add hundreds of MB to\n"
        "  low GB per run for larger networks, and nothing in this mode's own\n"
        "  analysis pipeline reads those columns).\n",
        argv0);
    std::exit(1);
}

std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t a = item.find_first_not_of(" \t");
        size_t b = item.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

NNLandscapeArgs parseArgs(int argc, char** argv) {
    NNLandscapeArgs a;
    auto need = [&](int i) -> std::string {
        if (i + 1 >= argc) usage(argv[0]);
        return argv[i + 1];
    };
    // argv[1] == "--nnlandscape" itself; scan the rest.
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--dataset")       { a.dataset = need(i); ++i; }
        else if (k == "--hidden")        { a.hidden = need(i); ++i; }
        else if (k == "--activation")    { a.activation = need(i); ++i; }
        else if (k == "--optimizer")     { a.optimizer = need(i); ++i; }
        else if (k == "--iterations")    { a.iterations = std::stoi(need(i)); ++i; }
        else if (k == "--population")    { a.population = std::stoi(need(i)); ++i; }
        else if (k == "--wbound")        { a.wBound = std::stod(need(i)); ++i; }
        else if (k == "--seed")          { a.seed = (unsigned)std::stoul(need(i)); ++i; }
        else if (k == "--radii")         { a.radii = need(i); ++i; }
        else if (k == "--local-samples") { a.localSamples = std::stoi(need(i)); ++i; }
        else if (k == "--out-prefix")    { a.outPrefix = need(i); ++i; }
        else if (k == "--out-summary")   { a.outSummary = need(i); ++i; }
        else if (k == "--label")         { a.label = need(i); ++i; }
        else if (k == "--save-x-columns") { a.saveXColumns = true; }
        else if (k == "-h" || k == "--help") usage(argv[0]);
        else { std::fprintf(stderr, "Unknown argument: %s\n", k.c_str()); usage(argv[0]); }
    }
    if (a.dataset.empty() || a.outPrefix.empty() || a.outSummary.empty()) usage(argv[0]);
    return a;
}

NNProblem::Activation parseActivation(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = (char)std::tolower((unsigned char)c);
    if (l == "tanh")    return NNProblem::Activation::Tanh;
    if (l == "sigmoid") return NNProblem::Activation::Sigmoid;
    return NNProblem::Activation::ReLU;
}

std::vector<int> parseHidden(const std::string& s) {
    std::vector<int> out;
    for (const auto& tok : splitCsv(s)) if (!tok.empty()) out.push_back(std::max(1, std::stoi(tok)));
    if (out.empty()) out.push_back(16);
    return out;
}

std::string toLowerCopy(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = (char)std::tolower((unsigned char)c);
    return l;
}

// The local methods gd/lbfgs/bfgs/nm/cg/ncg/adam are trained here through
// runLocalSearch() directly rather than makeMethod()+run(): a network needs
// a small random starting point around 0 (symmetry breaking, see below)
// rather than a sample from the weight box, and --iterations is passed as
// their iteration limit (maxItersOverride) under the same evaluation budget
// as every other optimizer of this mode. --population does not apply to them.
// Their remaining settings come from the [gd]/[lbfgs]/... sections.
bool isLocalMethod(const std::string& m) {
    return m == "gd" || m == "lbfgs" || m == "bfgs" || m == "nm" ||
           m == "cg" || m == "ncg"   || m == "adam";
}

// CSV-escapes a field that may contain a raw comma (e.g. multi-layer
// --hidden "8,8" ends up inside the run_key/arch columns). Learned the hard
// way in an earlier iteration of this pipeline: writing these unescaped
// silently misaligns every column after them once a comma-bearing value
// appears. Every string field written to either output CSV in this file
// goes through this first.
std::string csvField(const std::string& s) {
    if (s.find(',') == std::string::npos && s.find('"') == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    out += "\"";
    return out;
}

std::string deriveTestPath(const std::string& trainPath) {
    const std::string suffix = ".train";
    if (trainPath.size() >= suffix.size() &&
        trainPath.compare(trainPath.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return trainPath.substr(0, trainPath.size() - suffix.size()) + ".test";
    }
    return {};   // unrecognized naming convention: no auto-derived validation file
}

std::string baseNameNoExt(const std::string& path) {
    namespace fs = std::filesystem;
    return fs::path(path).stem().string();
}

} // namespace

int runNNLandscapeCli(int argc, char** argv) {
    NNLandscapeArgs args = parseArgs(argc, argv);
    const std::string optLower = toLowerCopy(args.optimizer);

    std::vector<std::string> radiusToks = splitCsv(args.radii);
    if (radiusToks.empty()) radiusToks = {"0.05"};

    // ---- Fail fast on a radii/summary-header mismatch, BEFORE any training
    // happens (same lesson learned building the NeuralStudio-side version of
    // this tool: checking only at append time still wastes a full run and
    // leaves an orphaned features file behind before erroring out). ----
    namespace fs = std::filesystem;
    if (fs::exists(args.outSummary) && fs::file_size(args.outSummary) > 0) {
        std::ifstream existing(args.outSummary);
        std::string headerLine;
        std::getline(existing, headerLine);
        existing.close();

        std::vector<std::string> existingRadii;
        std::stringstream hs(headerLine);
        std::string col;
        while (std::getline(hs, col, ',')) {
            const std::string prefix = "sharp_mean_r";
            if (col.rfind(prefix, 0) == 0) existingRadii.push_back(col.substr(prefix.size()));
        }
        if (existingRadii != radiusToks) {
            std::fprintf(stderr,
                "[nnlandscape] ERROR: --radii (%s) does not match the radii already baked "
                "into the header of %s. Use the same --radii for every run sharing this "
                "summary file, or pass a different --out-summary path.\n",
                args.radii.c_str(), args.outSummary.c_str());
            return 1;
        }
    }

    try {
        const std::string testPath = deriveTestPath(args.dataset);
        NNProblem prob(args.dataset, testPath, parseHidden(args.hidden),
                       parseActivation(args.activation), args.wBound);
        const int dim = prob.dimension();

        std::fprintf(stderr,
            "[nnlandscape] dataset=%s dim(weights)=%d hidden=%s act=%s optimizer=%s "
            "iters=%d pop=%d wBound=%.4g seed=%u task=%s val=%s\n",
            args.dataset.c_str(), dim, args.hidden.c_str(), args.activation.c_str(),
            args.optimizer.c_str(), args.iterations, args.population, args.wBound, args.seed,
            prob.task() == NNProblem::Task::Regression ? "regression" :
            prob.task() == NNProblem::Task::BinaryClassification ? "binary" : "multiclass",
            prob.hasValidation() ? "yes" : "no (no .test file found)");

        // ---- Train ----
        GeneralOptions g;
        g.population = std::max(4, args.population);
        g.max_iters  = args.iterations;
        g.max_evals  = (long long)args.iterations * (long long)g.population * 4LL + 10000LL;
        g.seed_base  = args.seed;
        prob.setMaxEvaluations(g.max_evals);

        Vec xbest_result;
        double fbest_result;
        long long evalsUsed;

        if (isLocalMethod(optLower)) {
            std::mt19937_64 rng(args.seed);
            // NOT the zero vector, which is a textbook symmetry-breaking failure
            // for a neural network -- confirmed by testing: with all weights
            // and biases at 0, every hidden unit's pre-activation is 0 for
            // every sample, making the gradient w.r.t. every weight EXCEPT the
            // output-layer biases exactly zero, so LBFGS/BFGS/GD "converge" in
            // a handful of evaluations to log(numClasses) (a uniform
            // prediction) without ever actually training. A small random
            // point breaks the symmetry, exactly like standard NN weight
            // initialization practice.
            std::uniform_real_distribution<double> initDist(-0.5, 0.5);
            Vec x0(dim);
            for (double& v : x0) v = initDist(rng);
            // GUI/CLI --iterations and the same evaluation budget every other
            // optimizer in this mode is bound by (g.max_evals, already set
            // above from --iterations/--population) are passed straight
            // through here instead of letting gd/lbfgs/bfgs/nm fall back to
            // their own independent [gd]/[lbfgs]/[bfgs]/[nm] cfg-section
            // defaults (max_iters=500 there) -- so every optimizer in a mode
            // 7 batch runs under the same nominal budget and is directly
            // comparable, with nothing left for the paper to have to caveat.
            auto res = runLocalSearch(optLower, &prob, rng, x0, args.iterations, g.max_evals);
            xbest_result = std::move(res.first);
            fbest_result = res.second;
            evalsUsed = prob.calls();
        } else {
            auto opt = makeMethod(optLower);
            if (!opt) {
                std::fprintf(stderr, "[nnlandscape] ERROR: unknown optimizer '%s'\n", args.optimizer.c_str());
                return 1;
            }
            opt->setProblem(&prob);
            opt->setSeed(args.seed);
            opt->setPopulation(g.population);
            opt->setGeneralOptions(g);
            // TerminationOptions left at its default (StopRule::NONE): only the
            // plain max_iters/max_evals bounds set above apply, matching the
            // predictable, per-optimizer-comparable budget this study needs.
            Result res = opt->run();
            xbest_result = std::move(res.xbest);
            fbest_result = res.fbest;
            evalsUsed = res.evals;
        }

        if (xbest_result.empty()) {
            std::fprintf(stderr, "[nnlandscape] ERROR: optimizer produced no solution.\n");
            return 1;
        }
        prob.setWeights(xbest_result);

        const double trainLoss = prob.trainLoss();
        const double valLoss   = prob.valLoss();
        const double trainAcc  = prob.trainAcc();
        const double valAcc    = prob.valAcc();
        const bool hasVal = prob.hasValidation();
        const double genGapLoss = hasVal ? (valLoss - trainLoss) : std::numeric_limits<double>::quiet_NaN();
        const double genGapErr  = (hasVal && trainAcc >= 0.0)
                                 ? ((1.0 - valAcc) - (1.0 - trainAcc))
                                 : std::numeric_limits<double>::quiet_NaN();

        std::fprintf(stderr,
            "[nnlandscape] trained: evals=%lld trainLoss=%.6f valLoss=%.6f trainAcc=%.4f "
            "valAcc=%.4f genGap(loss)=%.6f\n",
            (long long)evalsUsed, trainLoss, valLoss, trainAcc, valAcc, genGapLoss);

        // ---- Local sampling per radius ----
        const Vec& lb = prob.lb();
        const Vec& ub = prob.ub();
        const Vec  xbest = xbest_result;

        const std::string datasetName = baseNameNoExt(args.dataset);
        const std::string archTag = args.hidden + "_" + args.activation;
        std::ostringstream keyss;
        keyss << datasetName << "__" << archTag << "__" << optLower << "__seed" << args.seed;
        if (!args.label.empty()) keyss << "__" << args.label;
        const std::string runKey = keyss.str();
        const std::string runKeyCsv  = csvField(runKey);
        const std::string dsNameCsv  = csvField(datasetName);
        const std::string archTagCsv = csvField(archTag);

        std::vector<std::pair<std::string,double>> sharpMean, sharpMax, sharpStd;
        std::mt19937_64 sampleRng(args.seed ^ 0x9E3779B97F4A7C15ULL);

        for (const auto& radStr : radiusToks) {
            const double alpha = std::stod(radStr);
            if (alpha <= 0.0) continue;

            const int nSamples = (args.localSamples > 0) ? args.localSamples : std::max(50, 30 * dim);

            std::vector<double> loLoc(dim), hiLoc(dim);
            for (int d = 0; d < dim; ++d) {
                const double half = alpha * (ub[d] - lb[d]);
                loLoc[d] = std::max(lb[d], xbest[d] - half);
                hiLoc[d] = std::min(ub[d], xbest[d] + half);
                if (hiLoc[d] < loLoc[d]) std::swap(loLoc[d], hiLoc[d]);
            }

            const std::string outPath = args.outPrefix + "_r" + radStr + "_features.csv";
            // Write to a temp file, then atomically rename into place once
            // fully written, instead of writing the final filename directly.
            // This is the one place a run that gets interrupted mid-flight
            // (Stop in the GUI, Ctrl+C, a killed process, a power loss) could
            // previously leave a partially-written file behind under its
            // real, final name -- for a large network (dim in the
            // thousands, e.g. the "iran" family of datasets) a single
            // radius's sampling loop can run for a long time, so this is a
            // real, not just theoretical, window. nn_summary.csv itself
            // does NOT have this risk: its one-line append only happens
            // after every radius here has already finished and been
            // successfully renamed into place, so an interrupted run simply
            // never gets a row there at all -- cleanly as if it never ran,
            // nothing to clean up, nothing malformed to guard against when
            // reading it back later.
            const std::string tmpPath = outPath + ".tmp";
            std::ofstream out(tmpPath, std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("Could not open output file: " + tmpPath);

            // Same column layout Mode 5 (--features) already writes, so the
            // existing pflacco extraction step needs no changes for either --
            // EXCEPT the x1..xD coordinate columns, which are skipped unless
            // --save-x-columns is given (see NNLandscapeArgs::saveXColumns
            // for why: they are the dominant cost of this mode's disk
            // footprint and nothing in this mode's own analysis pipeline
            // reads them).
            out << "problem,dim,n_samples,init_type,modality,separability,"
                   "has_known_global_optimum,known_global_optimum,";
            if (args.saveXColumns)
                for (int d = 0; d < dim; ++d) out << "x" << (d + 1) << ",";
            out << "f,run_key,dataset,arch,optimizer,seed,radius_frac\n";

            std::vector<std::uniform_real_distribution<double>> distr;
            distr.reserve(dim);
            for (int d = 0; d < dim; ++d) distr.emplace_back(loLoc[d], hiLoc[d]);

            double sumF = 0.0, sumF2 = 0.0, maxF = -std::numeric_limits<double>::infinity();
            Vec x(dim);
            for (int s = 0; s < nSamples; ++s) {
                for (int d = 0; d < dim; ++d) x[d] = distr[d](sampleRng);
                const double f = prob.evaluate(x);   // real training-loss evaluation

                out << runKeyCsv << "," << dim << "," << nSamples << ",local_box_uniform,"
                    << "unknown,unknown,0,0,";
                if (args.saveXColumns)
                    for (int d = 0; d < dim; ++d) out << std::setprecision(15) << x[d] << ",";
                out << std::setprecision(15) << f << ","
                    << runKeyCsv << "," << dsNameCsv << "," << archTagCsv << "," << optLower << ","
                    << args.seed << "," << alpha << "\n";

                sumF += f; sumF2 += f * f; maxF = std::max(maxF, f);
            }
            out.close();
            if (out.fail()) throw std::runtime_error("Failed while writing: " + tmpPath);
            {
                std::error_code renameEc;
                fs::rename(tmpPath, outPath, renameEc);
                if (renameEc) {
                    // Cross-filesystem temp dirs etc. can make rename() fail
                    // even though the write itself succeeded -- fall back to
                    // copy+remove rather than losing the completed data.
                    fs::copy_file(tmpPath, outPath, fs::copy_options::overwrite_existing, renameEc);
                    if (!renameEc) fs::remove(tmpPath, renameEc);
                    else throw std::runtime_error("Could not finalize output file: " + outPath);
                }
            }

            prob.setWeights(xbest);   // restore before computing sharpness relative to trainLoss

            const double meanF = sumF / nSamples;
            const double varF  = std::max(0.0, sumF2 / nSamples - meanF * meanF);
            const double denom = 1.0 + std::abs(trainLoss);
            sharpMean.push_back({radStr, (meanF - trainLoss) / denom});
            sharpMax .push_back({radStr, (maxF  - trainLoss) / denom});
            sharpStd .push_back({radStr, std::sqrt(varF) / denom});

            std::fprintf(stderr, "[nnlandscape] radius=%s dim=%d n_samples=%d -> %s "
                                  "(sharp_mean=%.6f sharp_max=%.6f)\n",
                          radStr.c_str(), dim, nSamples, outPath.c_str(),
                          (meanF - trainLoss) / denom, (maxF - trainLoss) / denom);
        }
        prob.setWeights(xbest);

        // ---- Append summary row ----
        const bool writeHeader = !fs::exists(args.outSummary) || fs::file_size(args.outSummary) == 0;
        std::ofstream summary(args.outSummary, std::ios::app);
        if (!summary.is_open()) throw std::runtime_error("Could not open summary file: " + args.outSummary);

        if (writeHeader) {
            summary << "run_key,dataset,arch,optimizer,seed,dim,iterations,population,evals,wbound,"
                       "train_loss,val_loss,train_acc,val_acc,gen_gap_loss,gen_gap_err";
            for (auto& kv : sharpMean) summary << ",sharp_mean_r" << kv.first;
            for (auto& kv : sharpMax)  summary << ",sharp_max_r"  << kv.first;
            for (auto& kv : sharpStd)  summary << ",sharp_std_r"  << kv.first;
            summary << ",label\n";
        }
        summary << runKeyCsv << "," << dsNameCsv << "," << archTagCsv << "," << optLower << ","
                << args.seed << "," << dim << "," << args.iterations << ","
                // population: 1 for the local methods (a single search point);
                // evals: the TRAINING evaluations only -- prob.calls() at this
                // point also counts the landscape-sampling evaluations above.
                << (isLocalMethod(optLower) ? 1 : g.population) << ","
                << evalsUsed << "," << args.wBound << ","
                << std::setprecision(15) << trainLoss << "," << std::setprecision(15) << valLoss << ","
                << std::setprecision(15) << trainAcc  << "," << std::setprecision(15) << valAcc  << ","
                << std::setprecision(15) << genGapLoss << "," << std::setprecision(15) << genGapErr;
        for (auto& kv : sharpMean) summary << "," << std::setprecision(15) << kv.second;
        for (auto& kv : sharpMax)  summary << "," << std::setprecision(15) << kv.second;
        for (auto& kv : sharpStd)  summary << "," << std::setprecision(15) << kv.second;
        summary << "," << csvField(args.label) << "\n";
        summary.close();

        std::fprintf(stderr, "[nnlandscape] done. summary row appended to %s\n", args.outSummary.c_str());
        return 0;

    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[nnlandscape] ERROR: %s\n", ex.what());
        return 1;
    }
}

} // namespace optimsolution
