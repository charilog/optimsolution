#include "nn_problem.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace optimsolution {

// ------------------------------------------------------------------------
// Dataset loading: "<count>\n<n_samples>\n" header, then n_samples rows of
// whitespace-separated numbers. First inputCount_ columns are inputs, the
// remaining columns are outputs. inputCount_ is fixed by the FIRST file
// loaded (the training file) and reused (unchanged) for the test file.
// ------------------------------------------------------------------------
void NNProblem::loadDataset(const std::string& path,
                             std::vector<std::vector<double>>& X,
                             std::vector<std::vector<double>>& Y) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("NNProblem: could not open dataset file: " + path);

    int declaredInputs = 0;
    long long nSamples = 0;
    if (!(in >> declaredInputs)) throw std::runtime_error("NNProblem: bad header (inputs) in " + path);
    if (!(in >> nSamples))       throw std::runtime_error("NNProblem: bad header (n_samples) in " + path);

    if (inputCount_ == 0) inputCount_ = declaredInputs;
    else if (declaredInputs != inputCount_)
        throw std::runtime_error("NNProblem: input-count mismatch between train/test files in " + path);

    X.clear(); Y.clear();
    X.reserve(nSamples); Y.reserve(nSamples);

    // Read the first row to learn how many output columns this file has,
    // then keep reading until EOF (more robust than trusting nSamples
    // exactly, in case of trailing blank lines).
    std::string line;
    int outCols = -1;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::vector<double> vals;
        double v;
        while (ls >> v) vals.push_back(v);
        if (vals.empty()) continue;   // skip blank lines
        if ((int)vals.size() < inputCount_ + 1)
            throw std::runtime_error("NNProblem: row has fewer columns than expected in " + path);
        if (outCols < 0) outCols = (int)vals.size() - inputCount_;

        std::vector<double> xs(vals.begin(), vals.begin() + inputCount_);
        std::vector<double> ys(vals.begin() + inputCount_, vals.begin() + inputCount_ + outCols);
        X.push_back(std::move(xs));
        Y.push_back(std::move(ys));
    }
    if (X.empty())
        throw std::runtime_error("NNProblem: no data rows read from " + path);

    if (outputCount_ == 0) outputCount_ = outCols;
    else if (outCols != outputCount_)
        throw std::runtime_error("NNProblem: output-column-count mismatch between train/test files in " + path);
}

// ------------------------------------------------------------------------
// Task/architecture detection: classification when there's exactly one
// output column and it takes few, integer-like distinct values in the
// training data (mirrors the convention already used elsewhere in this
// project's tooling); regression otherwise. Output layer size follows:
// binary classification -> 1 unit (sigmoid, BCE), multi-class -> one unit
// per class (softmax, cross-entropy), regression -> outputCount_ units
// (linear, MSE).
// ------------------------------------------------------------------------
void NNProblem::detectTaskAndBuildArchitecture(const std::vector<int>& hiddenSizes) {
    task_ = Task::Regression;
    classValues_.clear();
    oneHotLabels_ = false;

    if (outputCount_ == 1) {
        std::set<double> uniq;
        bool allIntegerLike = true;
        for (const auto& row : trainY_) {
            const double v = row[0];
            uniq.insert(v);
            if (std::abs(v - std::round(v)) > 1e-6) allIntegerLike = false;
        }
        if (allIntegerLike && uniq.size() >= 2 && uniq.size() <= 20) {
            classValues_.assign(uniq.begin(), uniq.end());
            task_ = (classValues_.size() == 2) ? Task::BinaryClassification
                                                : Task::MultiClassClassification;
        }
    } else if (outputCount_ >= 2) {
        // Common alternative encoding: one row-vector per sample that is
        // already one-hot (all entries in {0,1}, summing to exactly 1).
        // Without this check, a dataset stored this way (observed in this
        // project's own dataset folder, e.g. "card") would silently be
        // treated as outputCount_-dimensional regression with MSE loss
        // instead of classification -- the gradient computed for THAT
        // (wrong) interpretation is still correct, but the loss/accuracy
        // reported would not mean what a caller expects.
        bool looksOneHot = true;
        for (const auto& row : trainY_) {
            double sum = 0.0;
            for (double v : row) {
                if (std::abs(v) > 1e-9 && std::abs(v - 1.0) > 1e-9) { looksOneHot = false; break; }
                sum += v;
            }
            if (!looksOneHot || std::abs(sum - 1.0) > 1e-9) { looksOneHot = false; break; }
        }
        if (looksOneHot) {
            oneHotLabels_ = true;
            // Always softmax+CE over outputCount_ units, even when outputCount_==2:
            // simpler and unambiguous vs. trying to map one-hot columns onto the
            // single-unit sigmoid convention used by the single-column-label path.
            task_ = Task::MultiClassClassification;
            classValues_.resize(outputCount_);
            for (int k = 0; k < outputCount_; ++k) classValues_[k] = (double)k;   // index-as-label
        }
    }

    outputLayerSize_ = (task_ == Task::MultiClassClassification)
                      ? (int)classValues_.size()
                      : (task_ == Task::BinaryClassification ? 1 : outputCount_);

    layerSizes_.clear();
    layerSizes_.push_back(inputCount_);
    for (int h : hiddenSizes) layerSizes_.push_back(std::max(1, h));
    layerSizes_.push_back(outputLayerSize_);
}

NNProblem::NNProblem(const std::string& trainPath,
                      const std::string& testPath,
                      const std::vector<int>& hiddenSizes,
                      Activation hiddenActivation,
                      double wBound)
    : hiddenAct_(hiddenActivation)
{
    loadDataset(trainPath, trainX_, trainY_);
    if (!testPath.empty()) loadDataset(testPath, testX_, testY_);

    // z-score normalize inputs using TRAINING-set statistics only (test set
    // uses the same transform, never its own stats -- avoids leaking test
    // distribution info into what "normalized" means).
    inputMean_.assign(inputCount_, 0.0);
    inputStd_.assign(inputCount_, 1.0);
    for (const auto& row : trainX_)
        for (int d = 0; d < inputCount_; ++d) inputMean_[d] += row[d];
    for (double& m : inputMean_) m /= (double)trainX_.size();
    for (const auto& row : trainX_)
        for (int d = 0; d < inputCount_; ++d) { double diff = row[d]-inputMean_[d]; inputStd_[d] += diff*diff; }
    for (int d = 0; d < inputCount_; ++d) {
        inputStd_[d] = std::sqrt(inputStd_[d] / (double)trainX_.size());
        if (inputStd_[d] < 1e-8) inputStd_[d] = 1.0;   // constant feature: leave untouched (already ~0 after centering)
    }
    auto applyNorm = [&](std::vector<std::vector<double>>& X) {
        for (auto& row : X)
            for (int d = 0; d < inputCount_; ++d) row[d] = (row[d] - inputMean_[d]) / inputStd_[d];
    };
    applyNorm(trainX_);
    if (!testX_.empty()) applyNorm(testX_);

    detectTaskAndBuildArchitecture(hiddenSizes);

    int totalWeights = 0;
    for (size_t i = 1; i < layerSizes_.size(); ++i)
        totalWeights += layerSizes_[i-1] * layerSizes_[i] + layerSizes_[i]; // weights + biases

    setName("nn_training");
    setFullName("Feed-forward network training loss (weight-space optimization)");
    setModality("unknown");
    setSeparability("unknown");
    setCategory("neural network training");

    Problem::init(totalWeights);
    Vec lo(totalWeights, -wBound), hi(totalWeights, wBound);
    setBounds(lo, hi);
}

void NNProblem::init(int /*dim*/) {
    // Architecture (and therefore dimension) is fixed by the constructor;
    // any externally-requested dimension is intentionally ignored, exactly
    // like every other fixed-dimension Problem in this framework.
}

// ------------------------------------------------------------------------
// Forward pass for one sample. cache.a[0] = input; cache.z[i]/cache.a[i]
// (i=1..L) = pre/post activation of layer i, where layer L is the output
// layer (linear pre-activation always kept in cache.z.back(); cache.a.back()
// holds the task-appropriate output: linear for regression, sigmoid for
// binary classification, softmax for multi-class -- gradient_core knows
// which and differentiates accordingly).
// ------------------------------------------------------------------------
namespace {
    inline double actForward(NNProblem::Activation a, double z) {
        switch (a) {
            case NNProblem::Activation::ReLU:    return z > 0.0 ? z : 0.0;
            case NNProblem::Activation::Tanh:    return std::tanh(z);
            case NNProblem::Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-z));
        }
        return z;
    }
    inline double actBackward(NNProblem::Activation a, double z, double postAct) {
        switch (a) {
            case NNProblem::Activation::ReLU:    return z > 0.0 ? 1.0 : 0.0;
            case NNProblem::Activation::Tanh:    return 1.0 - postAct * postAct;
            case NNProblem::Activation::Sigmoid: return postAct * (1.0 - postAct);
        }
        return 1.0;
    }
}

void NNProblem::forward(const Vec& weights, const std::vector<double>& xSample, ForwardCache& cache) const {
    const size_t L = layerSizes_.size() - 1;   // number of weight layers
    cache.a.assign(L + 1, {});
    cache.z.assign(L + 1, {});
    cache.a[0] = xSample;

    size_t off = 0;
    for (size_t l = 1; l <= L; ++l) {
        const int inSz  = layerSizes_[l-1];
        const int outSz = layerSizes_[l];
        cache.z[l].assign(outSz, 0.0);
        cache.a[l].assign(outSz, 0.0);
        const bool isOutputLayer = (l == L);

        for (int o = 0; o < outSz; ++o) {
            double s = weights[off + inSz * outSz + o];   // bias, laid out after the weight block
            const size_t wBase = off + (size_t)o;          // column-major-ish: w[i*outSz + o]
            for (int i = 0; i < inSz; ++i)
                s += weights[wBase + (size_t)i * outSz] * cache.a[l-1][i];
            cache.z[l][o] = s;
            cache.a[l][o] = isOutputLayer ? s /* apply task activation later */
                                          : actForward(hiddenAct_, s);
        }
        off += (size_t)inSz * outSz + outSz;
    }

    // Task-specific output activation, applied in place on the last layer.
    Vec& out = cache.a[L];
    if (task_ == Task::BinaryClassification) {
        out[0] = 1.0 / (1.0 + std::exp(-out[0]));
    } else if (task_ == Task::MultiClassClassification) {
        double mx = *std::max_element(out.begin(), out.end());
        double sum = 0.0;
        for (double& v : out) { v = std::exp(v - mx); sum += v; }
        for (double& v : out) v /= sum;
    }
    // Regression: leave linear.
}

int NNProblem::trueClassIndex(const std::vector<double>& ySample) const {
    if (oneHotLabels_)
        return (int)std::distance(ySample.begin(), std::max_element(ySample.begin(), ySample.end()));
    for (size_t k = 0; k < classValues_.size(); ++k)
        if (classValues_[k] == ySample[0]) return (int)k;
    return 0;
}

double NNProblem::sampleLoss(const ForwardCache& cache, const std::vector<double>& ySample) const {
    const Vec& out = cache.a.back();
    if (task_ == Task::Regression) {
        double s = 0.0;
        for (size_t k = 0; k < out.size(); ++k) { double d = out[k] - ySample[k]; s += d * d; }
        return s / out.size();
    }
    if (task_ == Task::BinaryClassification) {
        const double eps = 1e-12;
        double t = 0.0;
        for (double cv : classValues_) { if (cv == ySample[0]) break; t += 1.0; }
        // classValues_ sorted ascending; positive class = classValues_[1]
        const double y = (ySample[0] == classValues_[1]) ? 1.0 : 0.0;
        const double p = std::min(1.0 - eps, std::max(eps, out[0]));
        return -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
    // MultiClassClassification: cross-entropy against the true class index.
    const int trueIdx = trueClassIndex(ySample);
    const double eps = 1e-12;
    return -std::log(std::max(eps, out[trueIdx]));
}

bool NNProblem::samplePredictedClassCorrect(const ForwardCache& cache, const std::vector<double>& ySample) const {
    const Vec& out = cache.a.back();
    if (task_ == Task::BinaryClassification) {
        const int pred = out[0] >= 0.5 ? 1 : 0;
        const int truth = (ySample[0] == classValues_[1]) ? 1 : 0;
        return pred == truth;
    }
    if (task_ == Task::MultiClassClassification) {
        int predIdx = (int)std::distance(out.begin(), std::max_element(out.begin(), out.end()));
        return predIdx == trueClassIndex(ySample);
    }
    return false;   // not meaningful for regression
}

std::pair<double,double> NNProblem::evaluateOn(const Vec& weights,
                                                 const std::vector<std::vector<double>>& X,
                                                 const std::vector<std::vector<double>>& Y,
                                                 bool wantAcc) const {
    if (X.empty()) return {std::numeric_limits<double>::quiet_NaN(), -1.0};
    double sumLoss = 0.0;
    long long correct = 0;
    ForwardCache cache;
    for (size_t n = 0; n < X.size(); ++n) {
        forward(weights, X[n], cache);
        sumLoss += sampleLoss(cache, Y[n]);
        if (wantAcc && task_ != Task::Regression && samplePredictedClassCorrect(cache, Y[n])) ++correct;
    }
    const double meanLoss = sumLoss / X.size();
    const double acc = (wantAcc && task_ != Task::Regression) ? (double)correct / X.size() : -1.0;
    return {meanLoss, acc};
}

double NNProblem::evaluate_core(const Vec& x) {
    current_x_ = x;
    return evaluateOn(x, trainX_, trainY_, /*wantAcc*/false).first;
}

double NNProblem::valLoss() const {
    if (testX_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return evaluateOn(current_x_, testX_, testY_, false).first;
}
double NNProblem::trainAcc() const {
    if (task_ == Task::Regression) return -1.0;
    return evaluateOn(current_x_, trainX_, trainY_, true).second;
}
double NNProblem::valAcc() const {
    if (task_ == Task::Regression || testX_.empty()) return -1.0;
    return evaluateOn(current_x_, testX_, testY_, true).second;
}

// ------------------------------------------------------------------------
// Analytic backpropagation over the full training set (batch gradient --
// consistent with evaluate_core() computing the full-batch training loss,
// so gradient-based optimizers descend the exact function they're told
// they're minimizing).
// ------------------------------------------------------------------------
void NNProblem::gradient_core(const Vec& x, Vec& g) {
    current_x_ = x;
    g.assign(x.size(), 0.0);
    const size_t L = layerSizes_.size() - 1;

    ForwardCache cache;
    std::vector<Vec> delta(L + 1);   // delta[l] = dLoss/dz[l], l=1..L

    for (size_t n = 0; n < trainX_.size(); ++n) {
        forward(x, trainX_[n], cache);
        const Vec& out = cache.a[L];
        const std::vector<double>& ySample = trainY_[n];

        // ---- output-layer error dLoss/dz[L] ----
        delta[L].assign(out.size(), 0.0);
        if (task_ == Task::Regression) {
            for (size_t k = 0; k < out.size(); ++k)
                delta[L][k] = 2.0 * (out[k] - ySample[k]) / out.size();
        } else if (task_ == Task::BinaryClassification) {
            const double y = (ySample[0] == classValues_[1]) ? 1.0 : 0.0;
            delta[L][0] = out[0] - y;   // sigmoid+BCE combined gradient
        } else { // MultiClassClassification: softmax+CE combined gradient
            const int trueIdx = trueClassIndex(ySample);
            for (size_t k = 0; k < out.size(); ++k)
                delta[L][k] = out[k] - (k == (size_t)trueIdx ? 1.0 : 0.0);
        }

        // ---- backprop through hidden layers ----
        for (size_t l = L; l >= 1; --l) {
            const int inSz  = layerSizes_[l-1];
            const int outSz = layerSizes_[l];

            // locate this layer's weight block offset (same layout as forward())
            size_t off = 0;
            for (size_t k = 1; k < l; ++k)
                off += (size_t)layerSizes_[k-1] * layerSizes_[k] + layerSizes_[k];

            // accumulate gradient contributions for this sample
            for (int o = 0; o < outSz; ++o) {
                const double d = delta[l][o];
                g[off + (size_t)inSz * outSz + o] += d;   // bias
                const size_t wBase = off + (size_t)o;
                for (int i = 0; i < inSz; ++i)
                    g[wBase + (size_t)i * outSz] += d * cache.a[l-1][i];
            }

            if (l == 1) break;   // no earlier layer to propagate into

            // propagate to delta[l-1] via hidden activation derivative
            delta[l-1].assign(inSz, 0.0);
            for (int i = 0; i < inSz; ++i) {
                double s = 0.0;
                const size_t wBase = off + (size_t)i * outSz;
                for (int o = 0; o < outSz; ++o) s += x[wBase + (size_t)o] * delta[l][o];
                delta[l-1][i] = s * actBackward(hiddenAct_, cache.z[l-1][i], cache.a[l-1][i]);
            }
        }
    }

    for (double& v : g) v /= (double)trainX_.size();
}

} // namespace optimsolution
