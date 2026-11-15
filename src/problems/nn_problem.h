#pragma once
#include "problem.h"
#include <string>
#include <vector>

namespace optimsolution {

// -----------------------------------------------------------------------
// NNProblem: trains a small feed-forward network by treating its flattened
// weight vector as the point being optimized -- x IS the network's weights,
// evaluate_core(x) unflattens x, runs a forward pass over the TRAINING
// split, and returns the training loss. This lets any existing OptimSolution
// optimizer (gradient-based or population-based) train the network exactly
// as it would minimize any other Problem, with zero changes anywhere else
// in the framework.
//
// Self-contained on purpose: no Qt, no NeuralStudio dependency. Reads the
// same plain-text dataset format already used elsewhere in this project
// (line 1 = input feature count, line 2 = sample count, then that many
// whitespace-separated rows of "input_1 ... input_d output_1 ... output_k"),
// directly from disk via <fstream>.
//
// Unlike NSProblem (NeuralStudio), train/validation are two SEPARATE files
// on disk (dataset.train / dataset.test) rather than one file re-split by
// ratio -- this respects whatever split the dataset's author intended
// instead of re-deriving one, and is what makes valLoss()/valAcc() below
// meaningful as a genuine generalization-gap measurement.
// -----------------------------------------------------------------------
class NNProblem : public Problem {
public:
    enum class Task { Regression, BinaryClassification, MultiClassClassification };
    enum class Activation { ReLU, Tanh, Sigmoid };

    // trainPath/testPath: paths to "<name>.train" / "<name>.test" (or any
    // file in the same format). testPath may be empty, in which case
    // valLoss()/valAcc() report NaN/-1 (no generalization-gap signal
    // available, but training still works).
    // hiddenSizes: e.g. {8} or {8,8}. wBound: symmetric weight-box half-width.
    NNProblem(const std::string& trainPath,
              const std::string& testPath,
              const std::vector<int>& hiddenSizes,
              Activation hiddenActivation,
              double wBound);

    void init(int dim) override;   // dim is ignored: architecture fixes it

    // ---- Problem metadata / introspection, used by the CLI mode ----
    Task task() const { return task_; }
    int  inputCount() const { return inputCount_; }
    int  outputLayerSize() const { return outputLayerSize_; }
    const std::vector<int>& layerSizes() const { return layerSizes_; }  // [in, h1, ..., hk, out]

    // Explicitly set the "current" weights (e.g. to restore x_best after
    // optimization finishes) so the metrics below reflect that point rather
    // than whatever evaluate_core()/gradient_core() last touched.
    void setWeights(const Vec& x) { current_x_ = x; }
    const Vec& currentWeights() const { return current_x_; }

    // Metrics computed at current_x_ (call setWeights() first if you need
    // them at a specific point, e.g. the optimizer's returned x_best).
    double trainLoss() const { return evaluateOn(current_x_, trainX_, trainY_, /*wantAcc*/false).first; }
    double valLoss()   const;   // NaN if no test file was provided
    double trainAcc()  const;   // -1 if task_ == Regression
    double valAcc()    const;   // -1 if Regression or no test file

    bool hasValidation() const { return !testX_.empty(); }

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // ---- dataset ----
    // Inputs are z-score normalized (using TRAINING-set mean/std only, then
    // the same transform applied to the test set) before being stored here.
    // This is standard practice for training stability regardless, and
    // incidentally also matters for correctness verification: gradient-
    // checking a ReLU network against finite differences on raw, unnormalized
    // features was observed to produce large spurious mismatches (many
    // samples' pre-activations clustering near the same kink simultaneously
    // under the full-batch loss), which normalization resolves -- confirmed
    // by direct testing, see the accompanying integration notes.
    std::vector<double> inputMean_, inputStd_;
    int inputCount_{0};
    int outputCount_{0};                 // raw output columns in the file
    std::vector<std::vector<double>> trainX_, testX_;
    std::vector<std::vector<double>> trainY_, testY_;   // raw output columns as read

    // ---- task / architecture ----
    Task task_{Task::Regression};
    Activation hiddenAct_{Activation::ReLU};
    std::vector<double> classValues_;    // sorted unique labels, single-column-label classification only
    bool oneHotLabels_{false};           // true when the raw file already stores a one-hot vector per row
    std::vector<int> layerSizes_;        // [inputCount_, hidden..., outputLayerSize_]
    int outputLayerSize_{1};

    Vec current_x_;                      // last weights set via evaluate_core()/setWeights()

    // ---- helpers ----
    void loadDataset(const std::string& path, std::vector<std::vector<double>>& X,
                      std::vector<std::vector<double>>& Y);
    void detectTaskAndBuildArchitecture(const std::vector<int>& hiddenSizes);

    // Forward pass for one sample, returning per-layer pre/post activations
    // (needed by gradient_core's backward pass); also usable loss-only via
    // evaluateOn() for evaluate_core()/trainLoss()/valLoss().
    struct ForwardCache {
        std::vector<Vec> a;   // a[0] = input, a[i] = post-activation of layer i
        std::vector<Vec> z;   // z[i]  = pre-activation of layer i (i=1..L)
    };
    // Class index of the true label for one sample: argmax(ySample) when the
    // file already stores a one-hot vector (oneHotLabels_), otherwise a
    // lookup of the single raw label column against classValues_.
    int trueClassIndex(const std::vector<double>& ySample) const;

    void forward(const Vec& weights, const std::vector<double>& xSample, ForwardCache& cache) const;
    double sampleLoss(const ForwardCache& cache, const std::vector<double>& ySample) const;
    bool samplePredictedClassCorrect(const ForwardCache& cache, const std::vector<double>& ySample) const;

    std::pair<double,double> evaluateOn(const Vec& weights,
                                         const std::vector<std::vector<double>>& X,
                                         const std::vector<std::vector<double>>& Y,
                                         bool wantAcc) const;   // returns {loss, accuracy(-1 if !wantAcc)}
};

} // namespace optimsolution
