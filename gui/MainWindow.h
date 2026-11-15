#pragma once
#include <QMainWindow>
#include <QTextEdit>
#include <QColor>
#include <QProcess>
#include <QVector>
#include <QMap>
#include <QSet>
#include <QFutureWatcher>
#include <QElapsedTimer>
#include <QDateTime>
#include <QThread>
#include <QPointer>
#include <QString>
#include <functional>
#include <memory>
#include <vector>
#include <utility>
#include <limits>

#include "uscore.h"
#include "ModeAwareTabWidget.h"

QT_BEGIN_NAMESPACE
class QComboBox;
class QListWidget;
class QSpinBox;
class QDoubleSpinBox;
class QFormLayout;
class QTableWidget;
class QPushButton;
class QTabWidget;
class QLabel;
class QCheckBox;
class QRadioButton;
class QProgressBar;
class QSplitter;
class QGroupBox;
class QSlider;
class QLineEdit;
class QFile;
class QVBoxLayout;
class QEvent;
QT_END_NAMESPACE

class QTimer;
class QThread;

class BatchLogWriter;

namespace optimsolution_gui {

class ConfigFile;
class BusySpinner;

class MainWindow final : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;

private:
  // Forward declarations for batch types used in method prototypes.
  enum class BatchMetricMode : int;
  struct BatchCellData;
  struct BatchJob;
  struct BatchPostResult;

  struct OutputRunTab {
    QString title;              // "<method>+<problem>" (short names)
    QString methodShort;
    QString problemShort;
    int problemDim = 0;
    bool isMethodParamSensitivity = false; // true for "Sensitivity analysis of method parameters" run mode
    QWidget* page = nullptr;
    QTextEdit* log = nullptr;
    QLabel* convergenceInfo = nullptr;
    QWidget* convergencePlot = nullptr; // ConvergencePlotWidget
    QCheckBox* overlayMethodChk = nullptr; // overlay all problems for this method
    QCheckBox* overlayProblemChk = nullptr; // overlay all methods for this problem
    QRadioButton* xAxisIterRadio = nullptr; // x-axis mode: iterations
    QRadioButton* xAxisEvalRadio = nullptr; // x-axis mode: function evaluations

    // Convergence display options
    QComboBox* themeCombo = nullptr;
    QComboBox* gridDensityCombo = nullptr; // 0: sparse, 1: medium, 2: dense
    // Distribution plot appearance
    QRadioButton* distThemeLightRadio = nullptr;
    QRadioButton* distThemeDarkRadio  = nullptr;
    QComboBox*    distThemeCombo      = nullptr;
    QComboBox*    distGridDensityCombo = nullptr;
    // Sensitivity plot appearance
    QRadioButton* sensThemeLightRadio = nullptr;
    QRadioButton* sensThemeDarkRadio  = nullptr;
    QComboBox*    sensGridDensityCombo = nullptr;
    QPushButton*  exportSensitivityPngBtn = nullptr;

    QPushButton* exportConvergencePngBtn = nullptr;
    QCheckBox* convBandChk = nullptr; // show median + IQR band (runs) on convergence
    QLabel* distributionInfo = nullptr;
    QWidget* distributionPlot = nullptr; // DistributionPlotWidget
    QPushButton* exportDistributionPngBtn = nullptr;
    QVector<double> convIterX;
    QVector<double> convEvalX;
    QVector<double> convY;
    bool convHasEvalX = false;
    bool convLoaded = false;
    QString convLoadedCsvPath;   // exact CSV path that was last successfully loaded for this tab
    QString runtimeWorkingDir;
    QString runtimeCfgPath;
    qint64 runStartMsecsUtc = 0; // ms-since-epoch when the CLI process started for this tab

    // Sensitivity (runtime results)
    QTabWidget* innerTabs = nullptr;
    QComboBox* sensitivityParamCombo = nullptr; // parameter selector (metric is fixed)
    QWidget* sensitivityPlot = nullptr;         // SensitivityBarWidget
    QTableWidget* sensitivitySummaryTable = nullptr; // value -> mean(best_f)
    QTextEdit* sensitivityLog = nullptr;
    QString sensitivityCsvPath;
    bool sensLoaded = false;
    QPushButton* generateSensReportBtn = nullptr;

    // Landscape (2D contour + best-so-far trajectory animation)
    QLabel* landscapeInfo = nullptr;
    QWidget* landscapePlot = nullptr;      // LandscapeWidget
    QPushButton* landscapePlayBtn = nullptr;
    QSlider* landscapeSlider = nullptr;
    QComboBox* landscapeThemeCombo = nullptr;
    QPushButton* exportLandscapePngBtn = nullptr;
    bool landscapeLoaded = false;
    QString landscapeLoadedCsvPath;
    bool landscapeGridReady = false;

    // Multi-objective (Pareto front) results -- reuses this same struct (and
    // the outputRuns_/outputTabs_ index-paired bookkeeping already in place
    // for tab closing) instead of a parallel data structure, so closing a
    // tab keeps behaving correctly regardless of run mode.
    bool isMultiObjective = false;
    QWidget* moParetoPlot = nullptr;       // ParetoFrontWidget
    QTableWidget* moParetoTable = nullptr;
    QLabel* moInfoLbl = nullptr;
    QPushButton* moExportPngBtn = nullptr;
    QPushButton* moExportXlsxBtn = nullptr;

    // Portfolio (Mode 8: mid-run dynamic algorithm switching) results.
    // Reuses this same struct/bookkeeping rather than a parallel data
    // structure, same rationale as isMultiObjective above.
    bool isPortfolioRun = false;
    bool portfolioMultiProblem = false;
    QString portfolioOutPrefix;   // "<out-prefix>" passed to --portfolio; CSVs are "<prefix>_<name>.csv"
    QTableWidget* portfolioSegmentsTable = nullptr;
    QTableWidget* portfolioSummaryTable = nullptr;
    QTableWidget* portfolioRankingTable = nullptr;
    QTableWidget* portfolioFriedmanTable = nullptr;
    QTableWidget* portfolioWilcoxonTable = nullptr;
    QTableWidget* portfolioUsageTable = nullptr;
    QTableWidget* portfolioAggregateTable = nullptr;
    QWidget* portfolioWilcoxonPlot = nullptr;   // WilcoxonBoxPlotWidget*
    QWidget* portfolioFriedmanPlot = nullptr;   // RankPlotWidget*
    QComboBox* portfolioMetricCombo = nullptr;  // synced Wilcoxon<->Friedman (final_f / evals_used / num_switches)
    QComboBox* portfolioMetricCombo2 = nullptr;
    QLabel* portfolioInfoLbl = nullptr;
    QLabel* portfolioFriedmanLbl = nullptr;
    QLabel* portfolioStatsHintLbl = nullptr;
    bool portfolioLoaded = false;
  };

  // Region focus inside a run-mode tab (Selection / Output). The Settings
  // area is no longer one of the maximizable regions: it has its own
  // top-level "Settings" tab (see buildUi()/assembleMainTabs()).
  enum class FocusArea : int { None = 0, Selection = 1, Output = 3 };

  // Top-level tab layout of the main window: one tab per run mode (indices
  // 0..7 map 1:1 onto runModeBox_'s item data, i.e. onto the 8 run modes),
  // followed by the Code Wizard and Settings tabs.
  enum MainTabIndex : int {
    kModeTabCount   = 8,
    kCodeWizardTab  = 8,
    kSettingsTab    = 9,
    kHelpTab        = 10
  };

private slots:
  void onRefreshFactory();
  void onProblemChanged(const QString& problem);
  void onRunClicked();
  void onSelectSettings();
  void onReloadSettings();
  void onSaveSettings();
  void onMethodChanged(const QString& method);

  void onSensitivityEnableToggled(bool checked);
  void onSensitivityCellChanged(int row, int column);

  void onAddGlobalParam();
  void onRemoveGlobalParam();

  void onAddStopParam();
  void onRemoveStopParam();

  void onAddInitParam();
  void onRemoveInitParam();

  void onAddMethodParam();
  void onRemoveMethodParam();

  void onAddSensitivityParam();
  void onRemoveSensitivityParam();

  void onProcessReadyReadStdout();
  void onProcessReadyReadStderr();
  void onProcessFinished(int exitCode, QProcess::ExitStatus status);

  void onReloadConvergence();
  void onClearCsvFiles();

  void onExportConvergencePng();

  void onOutputTabCloseRequested(int index);
  void onOverlayMethodToggled(bool checked);
  void onOverlayProblemToggled(bool checked);
  void onConvergenceXAxisToggled(bool checked);
  void onConvergenceThemeToggled(bool checked);
  void onConvergenceGridChanged(int index);

  void onConvergenceBandToggled(bool checked);
  void onExportDistributionPng();

  void onOutputTabChanged(int index);
  void onOutputSensitivityParamChanged(int index);

  void onSettingItemChanged(int row, int column);
    void onRunModeChanged(int index);
    void onNewMethod();
    void onNewProblem();
    void onDeleteMethod();
    void onDeleteProblem();
    void updateRunBtn(bool running);
    void updateStatsTabsEnabled();
    void onProgressPollTick();
    void onExportStats();
    void onExportSensitivityPng();
    void onStatsTestChanged(int index);
    void onExportRankPlotPng();
    void onExportWilcoxonPlotPng();
    void onExportCdPlotPng();
    void onGenerateReport();
    void onRunMultiObjective();
    void onMultiObjectiveProcessFinished(int exitCode, QProcess::ExitStatus status);
    // Unified "Meta-dataset pipeline" mode (batch+rank, landscape sampling,
    // and the two Python stages all live together here, driven by their own
    // buttons rather than the shared Run/Stop button).
    void onRunPipelineBatch();
    void onPipelineBatchProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onPipelineRankProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onRunPipelineFeatures();
    void onPipelineFeaturesProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onRunPipelineExtractEla();
    // [pipeline] extract_ela_jobs from optimsolution.cfg -- passed to
    // extract_ela_features.py as --jobs. See onRunPipelineExtractEla().
    int  pipelineExtractElaJobsFromSettings() const;
    void onRunPipelineTrainModel();
    void onRunPipelineAblation(); // Reviewer 2, comment 4: feature-group ablation (Python, --ablation)
    void onRunPipelineCompareClassifiers(); // Reviewer 2, comment 4: classifier comparison (Python, --compare-classifiers)
    void onPipelinePyProcessFinished(int exitCode, QProcess::ExitStatus status);
    // Re-derives the 4/5 pipeline stage-done flags from whatever output
    // files already exist in optimsolution_gui_run/ (rather than assuming
    // "not done" just because this is a fresh GUI session), and populates
    // each stage's table from them. Called on every (re-)entry into
    // pipeline mode and at the end of "Load previous experiment", so
    // resuming earlier work never requires blindly re-running stages whose
    // output is already sitting on disk.
    void detectPipelineExistingOutputs();
    // Shared by onPipelinePyProcessFinished() (a run that just finished)
    // and detectPipelineExistingOutputs() (output found already on disk):
    // reads meta_learning_results/ (and, if includeAblation /
    // includeCompare, its ablation/ / classifier_comparison/ subfolders)
    // into the "Meta-learning results" / "Feature ablation" / "Classifier
    // comparison" tables. jumpToTab switches the inner tab widget to show
    // whichever of the three is most relevant; pass false for a silent
    // background (re-)detection.
    void populatePipelineMetaLearningResults(bool includeAblation, bool includeCompare, bool jumpToTab);
    // Mode 7 (--nnlandscape): trains NNProblem with the chosen optimizer,
    // then samples the loss landscape locally around the found minimum.
    // Same "own dedicated buttons instead of the shared Run/Stop button"
    // convention as mode 6 (Meta-dataset pipeline) -- two independent
    // stages: (1) the C++ CLI itself, (2) the Python analysis script.
    void onRunNNLandscapeTrain();
    void onNNLandscapeTrainProcessFinished(int exitCode, QProcess::ExitStatus status);
    // Pooled-mode counterpart of the slot above, used only when
    // nnParallelJobs_ > 1 (see nnActiveProcs_). Identifies which QProcess
    // finished via sender(), since several may be in flight at once.
    void onNNLandscapeJobSlotFinished(int exitCode, QProcess::ExitStatus status);
    void onRunNNLandscapeAnalyze();
    void onNNLandscapeAnalyzeProcessFinished(int exitCode, QProcess::ExitStatus status);
    // Stage 3 (optional): bootstrap confidence intervals on the LODO R2/MAE/
    // Spearman numbers stage 2 already wrote, by cluster-bootstrapping over
    // datasets (py_scripts/bootstrap_lodo_ci.py, reads stage 2's own
    // merged_dataset.csv -- no re-run of stage 1 or 2 needed). Addresses the
    // "how stable is this R2 with only a couple dozen datasets" question
    // directly, without a new training experiment.
    void onRunNNLandscapeBootstrap();
    void onNNLandscapeBootstrapProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onNNLandscapeBrowseDataset();
    void onRunNNLandscapeLoadExperiment();
    void onSaveNNBatchSelection();
    void onLoadNNBatchSelection();
    void onNNStatsUiChanged();
    void onExportParetoPng();
    void onExportParetoTable();
    void onLandscapePlayToggled(bool checked);
    void onLandscapeSliderMoved(int value);
    void onLandscapeTimerTick();
    void onExportLandscapePng();

protected:
  // Mirrors enable/disable state changes onto the mode tabs and the Code
  // Wizard tab (see assembleMainTabs()). Never consumes an event.
  bool eventFilter(QObject* watched, QEvent* event) override;
  void changeEvent(QEvent* e) override;   // re-derive palette-based styling on a light/dark switch

private:
  void buildUi();
  void checkAndAutoRebuild();   // auto-build & restart if .rebuild_pending exists
  void applyRegionStyling();
  void toggleRegionFocus(FocusArea area);
  void restoreRegionLayout();
  void updateRegionFocusStyling();

  // ── Tabbed main window (run modes as tabs) ───────────────────────────
  // The 8 run modes used to be chosen from the "Run mode" drop-down in the
  // Selection area. They are now top-level tabs. runModeBox_ is still the
  // single source of truth that all existing mode logic reads (it is just no
  // longer visible); selecting a tab drives it, and a programmatic change of
  // it (e.g. "Load batch selection...") moves the tab, so no mode logic had
  // to change.
  void assembleMainTabs(QWidget* central, QVBoxLayout* rootLay, QWidget* codeWizardPage);
  void onMainTabChanged(int tabIndex);
  void onRunModeComboChangedBefore(int mode);   // runs BEFORE onRunModeChanged()
  void onRunModeComboChangedAfter(int mode);    // runs AFTER onRunModeChanged()
  void placeWorkspaceForMode(int mode);         // re-parents Selection+Output into that mode's tab
  void applyModeSpecificSelectionUi(int mode);  // hides rows that do not apply to the mode
  void applyModeWorkspaceSizes(int mode);       // Selection/Output split for the mode
  void filterOutputTabsForMode(int mode);       // each mode shows only its own run tabs
  void updateModeTabsLock();                    // grey out other modes while a run is active
  void keepWindowWidthAcrossSwitch(int widthBefore);
  QWidget* buildHelpPage();                     // content of the "Help" tab
  QString  helpUserGuidePath() const;            // docs/optimsolution_manual_en.pdf, or empty if not found
  QString  helpSystemInfoText() const;           // plain-text "about / environment" block
  bool isRunBusy() const;                       // true while any run/stage is in flight
  // "New experiment in another mode" guard (see guardedStart()).
  bool guardedStart(int mode, const std::function<void()>& starter);
  QStringList runOutputFolders() const;          // existing optimsolution_gui_run folders of this session
  int  countFilesIn(const QStringList& folders) const;
  int  currentModeIndex() const;                // runModeBox_ item data (0 if unavailable)
  void loadFactoryLists();
  void loadSettings();
  void populateSettingsTables();

  void installSettingsValueWidgets(QTableWidget* table, const QString& section);
  void installSettingsValueWidgetRow(QTableWidget* table, const QString& section, int row);
  void populateSensitivityTableForMethod(const QString& methodSection);
  void populateSensitivityTableForProblem(const QString& problemSection);
  void syncSensitivityEnableUiFromConfig();
  void applySensitivityRowFlags(int row);
  void writeSensitivityConfigFromUi();
  void cleanupSensitivityLegacyKeys();

  void tryLoadSensitivityForTab(int index);
  void renderSensitivityForParam(int tabIndex, const QString& paramName);
  // Populates the Results tab's table with ALL swept parameters at once
  // (one "Parameter" column identifies each row's sweep), independent of
  // whatever parameter is currently selected in the Parameter Sensitivity
  // chart tab's combo box. See MainWindow_Tabs.cpp for why this exists as
  // a separate function from renderSensitivityForParam().
  void populateSensitivitySummaryAllParams(int tabIndex, const QString& csvPath, const QStringList& paramCols);
  void updateDimUiForProblem(const QString& problem);
  int  fixedDimForProblem(const QString& problem) const;
  // Multi-value dimension restriction (e.g. CEC2017: {10,30,50}), separate
  // from the single-value fixedDimForProblem() mechanism above -- see
  // include/allowed_dims.h for why these are kept as two distinct concepts.
  std::vector<int> allowedDimsForProblem(const QString& problem) const;
  // Text shown for a problem in the problem dropdown, the Batch/Sensitivity
  // problem lists and their Dims tables: the short name, followed by the
  // supported dimensions when the problem only accepts some of them, e.g.
  // "cec2017f1  (D: 10, 30, 50)". Display only -- every list keeps the bare
  // short name as its item data, and that is what the code reads.
  QString problemDisplayName(const QString& problem) const;
  // Inverse of problemDisplayName() for TEXT (the problem combo box is
  // editable): strips a trailing "  (D: ...)" and returns the short name.
  // Text without that suffix (e.g. a name typed by hand) is returned as is.
  static QString problemNameFromDisplay(const QString& text);
  // For problems with a restricted dimension set (CEC2017 / CEC2022): keeps
  // only the supported entries of `dims` (sorted); if none is left -- or
  // `dims` is empty -- returns ALL supported dimensions. For every other
  // problem `dims` is returned unchanged.
  QVector<int> restrictToAllowedDims(const QString& problem, const QVector<int>& dims) const;
  void snapDimSpinToAllowed();

  int createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim);
  OutputRunTab* outputTab(int index);
  const OutputRunTab* outputTab(int index) const;
  OutputRunTab* activeOutputTab();
  const OutputRunTab* activeOutputTab() const;
  void tryLoadConvergenceForTab(int index);
  void tryLoadPortfolioForTab(int index);
  void rebuildPortfolioPlotsForTab(int index);  void updateConvergencePlotForTab(int index);
  QString buildBatchReportHtml(bool includeConvergence) const;
  QString buildSensitivityReportHtml(int tabIndex) const;
  bool writeReportDocument(const QString& html, const QString& filePath) const;
  int createMultiObjectiveOutputTab(const QString& method, const QString& problem, int dim);
  int createPipelineOutputTab(); // lazily created once, reused by all 4 pipeline stages
  int createNNLandscapeOutputTab();   // mode 7's own output tab (Log + Wilcoxon + Rankings)
  void populateNNDatasetList();       // scans datasetsDir() for *.train, fills nnDatasetsList_
  void populateNNOptimizerList();     // fills nnOptimizersList_ from listMethodNames() -- ALL of them
  void refreshNNButtonStates();       // enables "2. Analyze" once nnStage1Done_ is true
  void refreshNNSeedInfoLabel();      // updates "Seeds" to preview what a Train click would use
  void flushPipelineLog();
  // Generic 300-DPI PNG export for the pipeline's own plot widgets
  // (RankPlotWidget/WilcoxonBoxPlotWidget instances in the Rankings/
  // Wilcoxon/Friedman tabs) -- same render()-at-scale technique as
  // exportRankPlotPng()/exportWilcoxonPlotPng(), just widget-generic since
  // those two are tied to mode 1's own rankPlot_/wilcoxonPlot_ members.
  bool exportPipelinePlotPng(QWidget* widget, const QString& outPath, bool isTransparent) const;
  // Reads batch_method_summary.csv / batch_ranked.csv / batch_wilcoxon.csv /
  // batch_friedman.txt from optimsolution_gui_run/ and fills in the
  // Rankings/Wilcoxon/Friedman tabs. Called once stage 1 (batch + rank)
  // succeeds.
  void populatePipelineStatsTabs();
  // Stage 2 raw-data tab (Landscape files) -- lists every generated
  // *_features.csv with its basic metadata (problem, dim, n_samples,
  // modality, separability), parsed from each file's own first data row.
  void populatePipelineLandscapeFilesTable();
  // Stage 3 raw-data tab (ELA features) -- the full ela_features.csv, so a
  // researcher can inspect the actual feature values stage 4 will train on.
  void populatePipelineElaFeaturesTable();
  // Session-scoped completion tracking for the 4 pipeline stages -- these
  // are NOT derived from file existence (leftover files from an earlier
  // session must not silently mark a stage as "done" in a fresh one).
  // Reset to false (all checkmarks hidden, stage 3/4 disabled) on entering
  // pipeline mode and on "Delete output files"; set true only by this
  // session's own successful stage completions.
  bool pipelineStage1Done_ = false;
  bool pipelineStage2Done_ = false;
  bool pipelineStage3Done_ = false;
  bool pipelineStage4Done_ = false;
  bool pipelineAblationDone_ = false; // Reviewer 2, comment 4: feature-group ablation
  bool pipelineCompareDone_ = false; // Reviewer 2, comment 4: classifier comparison
  void resetPipelineButtonStates();
  // Narrower version used specifically by onLoadPipelineExperiment() --
  // resets only the 4 stage checkmarks, without touching
  // pipelineExperimentLoaded_/pipelineLoadedTripleKeys_ (see definition).
  void resetPipelineStageFlags();
  void refreshPipelineButtonStates();
  // "Load previous experiment" (add/remove workflow): when a previous
  // batch_master.csv has been loaded, pipelineExperimentLoaded_ is true and
  // pipelineLoadedTripleKeys_ holds every (method,problem,dim) triple it
  // already contains, as "method|problem|dim" strings. Stage 1 then runs
  // ONLY the combinations newly selected but not yet on disk, and deletes
  // the summary CSVs of any combination that's no longer selected --
  // unlike Batch run (mode 1)'s own CSV loader, which can only ever add
  // more runs on top of what's already there. Reset (back to today's
  // full-recompute default) on entering the pipeline mode or on
  // "Delete output files".
  bool pipelineExperimentLoaded_ = false;
  QSet<QString> pipelineLoadedTripleKeys_;
  QSet<QString> pipelinePendingDesiredKeys_; // computed at Stage-1 launch, applied to pipelineLoadedTripleKeys_ on success
  void onLoadPipelineExperiment();
  // Diagnostic transparency: logs exactly which (problem,dim) groups
  // became newly complete/incomplete/brand-new vs. the loaded baseline
  // after a Stage-1 rerun -- see definition for why this matters.
  void logPipelineGroupChanges();
  // Disables/enables Selection, Settings, "Delete output files" and "Load
  // previous experiment" while any of the 4 pipeline stages is running --
  // "Open output folder" and "Maximize" are deliberately left alone. Called
  // at the start of each stage's launch and unconditionally (success or
  // failure) when that stage's process finishes. Mirrors the exact same
  // pattern the main Run button already uses for every other mode (see
  // MainWindow_ProcessSettings.cpp).
  void setPipelineUiEnabled(bool enabled);
  // "Metric" dropdown (Rankings/Wilcoxon/Friedman tabs): which
  // batch_master.csv column --rank ranks by (defaults to "mean_f").
  // Persisted here so it survives across Stage-1 re-runs instead of
  // silently reverting to the default every time.
  QString pipelineMetricColumn_ = "mean_f";
  QProcess* pipelineMetricRerankProc_ = nullptr;
  void onPipelineMetricChanged(const QString& column);

  // Per-stage wall-clock timing (mode 6 / pipeline). Persisted to
  // "pipeline_timings.csv" in optimsolution_gui_run/ so that reloading a
  // previous experiment (see onLoadPipelineExperiment()) can still show how
  // long each stage took last time, not just within the current session.
  QElapsedTimer pipelineStage1Timer_;
  QElapsedTimer pipelineStage2Timer_;
  QElapsedTimer pipelineStage3Timer_;
  QElapsedTimer pipelineStage4Timer_;
  QElapsedTimer pipelineAblationTimer_;
  QElapsedTimer pipelineCompareTimer_;
  // Reads/writes exactly one row per stage key ("batch_ranking",
  // "landscape_sampling", "extract_ela", "train_model") -- read-modify-write
  // so saving one stage's timing never clobbers the others already on disk.
  void savePipelineStageTiming(const QString& stageKey, double seconds);
  QMap<QString, double> loadPipelineStageTimings();
  // Updates the per-stage time labels + the cumulative total label (shown
  // only once all 4 stages have a recorded time) from whatever is
  // currently on disk. Called after any stage finishes, after loading a
  // previous experiment, and cleared on resetPipelineButtonStates().
  void refreshPipelineTimingLabels();
  // Drives the shared outputProgressBar_/progressWidget_ for pipeline
  // stages (same widget Batch run etc. use, via MainWindow_ProcessSettings.cpp
  // -- the pipeline's own separate QProcess objects need their own calls
  // into it since they don't go through that shared code path).
  void startPipelineProgress(const QString& label);
  void finishPipelineProgress(bool success);
  // Finds a Python interpreter that actually has `requiredModules` importable
  // -- not just "some python that runs" (a bare "py" on Windows can resolve
  // to a different/newer install than the one the required packages were
  // installed into). Tries several common version selectors. Returns a
  // "<program> [args...]" spec (split on spaces) or empty if none qualify.
  QString findPythonForModules(const QStringList& requiredModules) const;
  QString pyScriptsDir() const;          // <projectRoot_>/py_scripts
  QString datasetsDir() const;           // <projectRoot_>/datasets -- used by mode 7 (--nnlandscape)
  void loadParetoCsvIntoTab(int index, const QString& csvPath);
  bool exportParetoPng(int index, const QString& filePath) const;
  void tryLoadLandscapeForTab(int index);
  void ensureLandscapeGridForTab(int index);
  bool exportLandscapePng(const QString& filePath);

  bool exportConvergencePlotPng(int outputTabIndex, const QString& filePath);

  void updateDistributionPlotForTab(int index);

  bool exportDistributionPlotPng(int outputTabIndex, const QString& filePath);

  void appendLog(const QString& text);
  void appendLogStyled(const QString& line, const QColor& color, bool bold=false);
  void appendFinishSummaryColored(int exitCode);
  void startCliProcess(const QStringList& args);
  void requestStop();
  // Batch responsiveness helpers (do not block GUI thread during batch runs)
  void startBatchLogWriter();
  void stopBatchLogWriter();
  void batchHandleProcessOutput(const QString& cleaned);
  void flushBatchUiLog();
  // Single-run responsiveness helpers (avoid blocking GUI thread during verbose stdout).
  void singleHandleProcessOutput(const QString& cleaned);
  void flushSingleUiLog();
  void startSensCsvPoll(const QString& csvPath, int expectedPoints, int runsPerPoint);
  void stopSensCsvPoll();
  void onSensCsvPollTick();
  void updateSingleBatchProgress();
  void startNextSensJob();

  // Nested types used by sensitivity helper methods — must be declared before use.
  struct SensPointResult {
    QMap<QString,QString> paramValues;
    double meanF = 0, stdevF = 0, minF = 0, maxF = 0;
    double meanEvals = 0, stdevEvals = 0, successRate = 0;
    int runs = 0;
    bool valid = false;
  };
  struct SensGroup {
    QString method, problem;
    int dim = 2;
    QStringList sweepParams;
    QVector<SensPointResult> results;
  };

  void parseSummaryCsvForSensPoint(const QString& csvPath, SensPointResult& out);
  void writeSensGroupCsv(const SensGroup& group, const QString& wd);
  QString cliPath() const;
  QString currentMethodShort() const;
    void updateBatchPanelVisibility();
    void populateBatchLists();
    void syncSensProblemDimsTable();
    void syncBatchProblemDimsTable();
    void refreshBatchSelectionView();
    void bindBatchSummaryUiForPage(QWidget* page);
    QString batchBaseProblemShort(const QString& problemKey) const;
    QString batchProblemDisplayKey(const QString& problemShort, int dim) const;
    QVector<int> batchDimsForProblem(const QString& problemShort, int defaultDim) const;
    bool batchHasCachedCell(const QString& problemShort, const QString& methodShort, int dim) const;
    void invalidateBatchProblemCache(const QString& problemShort);
    int  batchDimForProblem(const QString& problemShort, int defaultDim) const;
    void startBatch();
    void startNextBatchJob();
    void startBatchPostProcessAsync(const BatchJob& job, int exitCode, const QString& runtimeWorkingDir);
    void finishBatchPostProcess(int exitCode, const BatchPostResult& result);
    void finalizeBatch();
    void rebuildStatsTable();
    void initStatsTableForBatch(const QStringList& methods, const QStringList& problems);
    void updateStatsTableCell(const QString& problemShort, const QString& methodShort);
    void finalizeStatsTableAfterBatch();
    void rebuildStatsComparisons();

  // Helpers for batch metric labeling and statistics plots.
  QString batchMetricTitle() const;
  QString batchMetricUnitSuffix() const;
  void rebuildWilcoxonPlot();
    double extractBatchMetric(const QString& text, bool* ok) const;
    BatchMetricMode currentBatchMetricMode() const;
    QString batchMetricAxisLabel() const;
    void onBatchMetricUiChanged();
    int globalRunsFromSettings() const;

    QStringList selectedBatchMethodShortNames() const;
    QStringList selectedBatchProblemShortNames() const;
    QStringList effectiveStatsMethodShortNames() const;
    QStringList effectiveStatsProblemShortNames() const;

    QString findBestConvergenceCsvForJob(const QString& workingDir,
                                        const QString& methodShort,
                                        const QString& problemShort,
                                        int dim,
                                        qint64 newerThanMsecsUtc,
                                        QString* err) const;

    bool loadBatchCellFromConvergence(const QString& csvPath, BatchCellData& out, QString* err) const;

    // --- U-score ("trial-based dominance") batch report ---------------
    // See include/uscore.h for the ranking algorithm itself (Qt-free, unit
    // tested in tests/test_uscore.cpp). These two methods are the only
    // GUI-side additions needed to run it automatically at the end of a
    // batch, using the same convergence CSVs already produced by the CLI.
    //
    // Reads one method/problem's *_convergence.csv and returns, per run,
    // the full (evals, best_f) trace sorted by ascending evals. Mirrors
    // loadBatchCellFromConvergence()'s CSV parsing but keeps the whole
    // trace instead of collapsing it into a BatchCellData.
    bool loadRunTracesFromConvergenceCsv(const QString& csvPath,
                                          std::vector<std::vector<optimsolution::uscore::EvalPoint>>& outRuns,
                                          QString* err) const;

    // Computes the U-score ranking for every problem that has >=2 methods
    // in the just-finished batch (using g_liveBatchCsvPaths), aggregates
    // it into an overall ranking, and writes a "uscore_report_<timestamp>.xlsx"
    // workbook next to the batch's own convergence CSVs -- with an "About"
    // sheet, a "Ranking" sheet (mirrors the CEC-2025-slides pp. 19-20
    // Algorithm/Ranking tables) and a "Per-Problem U-scores" sheet (mirrors
    // pp. 22-23's Prob\Algo tables, incl. the trailing Sum/RS row) -- then
    // logs its path. Called once, additively, at the very end of
    // finalizeBatch(); does nothing (and never throws) if fewer than two
    // methods were compared.
    void exportUScoreReport();
    double aggregateValues(const QVector<double>& vals, const QString& agg) const;
    double stdevValues(const QVector<double>& vals) const;
    bool exportStatsToXlsxNative(const QString& outPath) const;
    bool exportStatsToXlsxViaPython(const QString& outPath, const QString& csvPath);
    // Generic "any QTableWidget -> xlsx (falls back to csv)" export, reusing
    // the same xlsx machinery as the batch derived-table export buttons
    // above (temporarily points statsTable_ at the given table). Shows the
    // save dialog itself; returns without doing anything if the table is
    // empty. Used by Mode 8's Segments/Summary tables.
    void exportTableWidgetToXlsxOrCsv(QTableWidget* table, const QString& dialogTitle, const QString& defaultPath);
    // Generic "any QWidget -> 300 DPI PNG" export, matching the exact DPI
    // math already used by exportWilcoxonPlotPng()/exportRankPlotPng()/
    // exportCdPlotPng() above (96 -> 300 DPI scale, embedded DPI metadata,
    // antialiased QPainter::render()). Used by Mode 8's Wilcoxon/Friedman
    // plots, which are separate widget instances from the Batch-run ones
    // above and so need a caller-supplied widget rather than a hardcoded
    // member pointer.
    bool exportWidgetToPng300Dpi(QWidget* widget, const QString& filePath) const;
    bool exportTableWidgetToXlsxNative(QTableWidget* table, const QString& outPath) const;
    bool exportSensitivityPlotPng(int outputTabIndex, const QString& filePath);
    bool exportRankPlotPng(const QString& filePath);
    bool exportWilcoxonPlotPng(const QString& filePath);
    bool exportCdPlotPng(const QString& filePath);
    void updateOutputInnerTabsForRunMode();
    QString batchMetricYAxisLabel() const;
  QString currentProblemShort() const;
  bool problemHasConfigSection(const QString& problemShort) const;
  bool isFixedDimensionProblem(const QString& problemShort) const;
  bool detectCliSupportsConfigArg(const QString& cli);

  void updateOutputProgressFromText(const QString& text);

  void setTablesEditable(bool editable);

  // Guards against signal re-entrancy while the UI is programmatically updated.
  bool internalUpdate_ = false;

  // Region focus state (Selection / Output inside a run-mode tab)
  FocusArea focusedArea_ = FocusArea::None;

  // Region widgets and controls (for focus/restore UI only)
  QGroupBox* selectionBox_ = nullptr;
  QGroupBox* settingsBox_ = nullptr;
  QGroupBox* outputBox_ = nullptr;

  // Tabbed main window. mainTabs_ holds one tab per run mode (0..7), then the
  // Code Wizard tab and the Settings tab. The Selection + Output areas are a
  // single shared "workspace" (workspace_) that is re-parented into whichever
  // mode tab is active, so every mode keeps using the very same widgets and
  // handlers as before. The Settings area (settingsBox_, with its own inner
  // tabs_) lives in the Settings tab; the Code Wizard has its own tab.
  QTabWidget* mainTabs_ = nullptr;
  QWidget* workspace_ = nullptr;
  QSplitter* workspaceSplitter_ = nullptr;   // vertical: Selection above, Output below
  QWidget* codeWizardPage_ = nullptr;
  QVector<QWidget*> modePages_;              // one page per run mode (placeholder for workspace_)
  int  lastMainTab_ = 0;                     // previously shown top-level tab (for lock revert)
  bool mainTabRevertGuard_ = false;          // suppresses onMainTabChanged() during programmatic tab moves
  int  shownMode_ = 0;                       // run mode whose workspace/output tabs are currently displayed
  bool outputFilterGuard_ = false;           // re-entrancy guard for filterOutputTabsForMode()
  QVector<QVector<int>> modeSplitterSizes_;  // per-mode Selection/Output split (user-adjusted)
  QPointer<QWidget> modeLastOutputPage_[kModeTabCount];  // last active output page, per mode
  QString modeStatusText_[kModeTabCount];                // Output status line ("Completed. ...") per mode
  bool    modeHasResults_[kModeTabCount] = {};           // a run of this mode produced output in this session
  QPushButton* selectionMaxBtn_ = nullptr;
  QPushButton* outputMaxBtn_ = nullptr;

  // Cached capability probe for the CLI.
  // -1: unknown, 0: no, 1: yes
  int cliSupportsConfigArg_ = -1;

  // Widgets
  QComboBox* methodBox_ = nullptr;
  QComboBox* problemBox_ = nullptr;
  QSpinBox*  dimSpin_ = nullptr;


  QFormLayout* selectionForm_ = nullptr; // Selection form layout (used to hide label rows in batch mode)
  QPushButton* runBtn_ = nullptr;
    QComboBox*  runModeBox_ = nullptr;
    QGroupBox*  moPanel_ = nullptr;
    QComboBox*  moProblemBox_ = nullptr;
    QComboBox*  moMethodBox_ = nullptr;
    QSpinBox*   moDimSpin_ = nullptr;
    // Multi-objective runs use their own QProcess, entirely separate from
    // proc_ (used by Single/Batch/Sensitivity), so nothing about how those
    // modes launch/monitor/parse their CLI process is touched.
    QProcess*   moProc_ = nullptr;
    // Unified "Meta-dataset pipeline" mode (run mode 5): replaces the former
    // separate Headless-batch-export and Landscape-features modes. Reuses
    // batchPanel_ (methods+problems selection, including per-problem
    // dimension overrides) for input; its own panel below only adds the
    // 4 action buttons plus a sample-count spinbox. Each stage writes into
    // optimsolution_gui_run/, feeding the next: batch -> rank -> features ->
    // (Python) ELA extraction -> (Python) meta-learner training. Stages are
    // triggered by their own dedicated buttons, not the shared Run/Stop
    // button (which is hidden for this mode -- see updateBatchPanelVisibility()).
    QGroupBox*   pipelinePanel_          = nullptr;
    QPushButton* pipelineRunBatchBtn_    = nullptr;
    QPushButton* pipelineRunFeaturesBtn_ = nullptr;
    QPushButton* pipelineExtractElaBtn_  = nullptr;
    QPushButton* pipelineTrainModelBtn_  = nullptr;
    // Reviewer 2, comment 4 ("feature-group ablations ... would help
    // explain why and when landscape information is useful"): an optional
    // 5th stage, re-running the same fold-safe LOPO protocol as stage 4
    // once per ELA feature family (excluded, then in isolation). Shares
    // stage 4's prerequisites (stage 1 + stage 3 done) since it re-derives
    // everything stage 4 does internally, via train_meta_learner.py's
    // --ablation flag -- it does not depend on stage 4 having been run.
    QPushButton* pipelineAblationBtn_    = nullptr;
    // Reviewer 2, comment 4 ("comparisons with alternative selection
    // models ... would help explain why and when landscape information is
    // useful"): another optional stage, re-running the same fold-safe LOPO
    // protocol with SVM, Gradient Boosting, and Logistic Regression
    // alongside this study's own Random Forest. Shares stage 4's own
    // prerequisites, exactly like the ablation stage.
    QPushButton* pipelineCompareBtn_     = nullptr;
    QSpinBox*    pipelineSamplesSpin_    = nullptr;
    // One spinner per stage (right next to its button), same widget class
    // as the shared busySpinner_ used elsewhere.
    BusySpinner* pipelineBatchSpinner_    = nullptr;
    BusySpinner* pipelineFeaturesSpinner_ = nullptr;
    BusySpinner* pipelineExtractSpinner_  = nullptr;
    BusySpinner* pipelineTrainSpinner_    = nullptr;
    BusySpinner* pipelineAblationSpinner_ = nullptr;
    BusySpinner* pipelineCompareSpinner_  = nullptr;

    QProcess*    pipelineBatchProc_    = nullptr; // stage 1a: --batch
    QProcess*    pipelineRankProc_     = nullptr; // stage 1b: --rank (auto-chained after 1a)
    QProcess*    pipelineFeaturesProc_ = nullptr; // stage 2: --features (looped, one pair at a time)
    QProcess*    pipelinePyProc_       = nullptr; // stage 3/4: python (reused sequentially, never concurrent)
    // Which of the two Python stages pipelinePyProc_ currently belongs to,
    // if any -- so that clicking one stage's button while the OTHER stage
    // is running can warn correctly ("training is still running") instead
    // of silently killing the wrong process with a misleading "stopping X"
    // message.
    enum class PipelinePyStage { None, Extract, Train, Ablation, CompareClassifiers };
    PipelinePyStage pipelinePyRunningStage_ = PipelinePyStage::None;

    // Stage-1 (batch) output can run to tens of thousands of lines -- same
    // throttled-logging technique used before: raw output streams straight
    // to disk, only a filtered/capped subset is periodically flushed into
    // the visible tab (see flushPipelineLog()).
    QString      pipelineLogBuffer_;
    QFile*       pipelineLogFile_        = nullptr;
    QTimer*      pipelineFlushTimer_     = nullptr;

    // Stage-2 (features) processes one (problem, dim) pair at a time from
    // this queue, advancing on each QProcess::finished -- avoids spawning
    // dozens of processes at once and keeps a clean, readable progress log.
    std::vector<std::pair<QString,int>> pipelineFeatureQueue_;
    int          pipelineFeatureTotal_   = 0;

    int          pipelineOutputTabIndex_ = -1; // single tab reused by all 4 stages

    // ── Mode 7 (--nnlandscape): NN local landscape / generalization gap ────
    // Selection-area settings panel + Output-area action panel, same split
    // as the Meta-dataset pipeline mode above (settings live in Selection,
    // actions that produce output live in Output). Batch-capable: multiple
    // datasets x multiple architectures x multiple optimizers x "Runs" seeds
    // (seed_base from optimsolution.cfg, same convention every other mode
    // already uses -- see globalRunsFromSettings()/cfg_).
    QGroupBox*      nnLandscapePanel_    = nullptr;  // Selection area
    QListWidget*    nnDatasetsList_      = nullptr;  // multi-select, populated from datasetsDir()
    QPushButton*    nnDatasetBrowseBtn_  = nullptr;  // adds an extra checked entry outside datasetsDir()
    QLineEdit*      nnHiddenEdit_        = nullptr;  // ';'-separated architectures, e.g. "8;16;8,8"
    QComboBox*      nnActivationBox_     = nullptr;  // relu|tanh|sigmoid (applies to every arch in the batch)
    QListWidget*    nnOptimizersList_    = nullptr;  // multi-select, ALL of listMethodNames()
    QSpinBox*       nnIterationsSpin_    = nullptr;
    QSpinBox*       nnPopulationSpin_    = nullptr;
    QDoubleSpinBox* nnWBoundSpin_        = nullptr;
    QLineEdit*      nnRadiiEdit_         = nullptr;  // e.g. "0.02,0.05,0.1"
    QSpinBox*       nnLocalSamplesSpin_  = nullptr;  // 0 = auto (30*dim)
    QLineEdit*      nnLabelEdit_         = nullptr;  // optional run label

    QGroupBox*      nnActionPanel_       = nullptr;  // Output area
    QPushButton*    nnSaveSelectionBtn_  = nullptr;  // Selection area: "Save batch selection..." (own .batchsel dialect)
    QPushButton*    nnLoadSelectionBtn_  = nullptr;  // Selection area: "Load batch selection..."
    QPushButton*    nnLoadExperimentBtn_ = nullptr;  // "Load previous experiment..." (replaces "Refresh plots" for this mode)
    BusySpinner*    nnLoadExperimentSpinner_ = nullptr;
    QString         nnLoadedRunDir_;      // empty = use the default runtime dir; set by "Load previous experiment..."
    // Set of "dataset|archTag|optimizer|seed" keys already present in a
    // loaded experiment's nn_summary.csv -- mirrors pipelineLoadedTripleKeys_.
    // Consulted when building a new batch queue so re-running after a load
    // only adds genuinely new combinations instead of duplicating rows.
    QSet<QString>   nnLoadedTripleKeys_;
    QPushButton*    nnRunTrainBtn_       = nullptr;  // stage 1: optimsolution --nnlandscape (looped over the batch queue)
    BusySpinner*    nnTrainSpinner_      = nullptr;
    QPushButton*    nnRunAnalyzeBtn_     = nullptr;  // stage 2: python train_gap_regressor_logtransform.py
    BusySpinner*    nnAnalyzeSpinner_    = nullptr;
    // Stage 2 modifier: applies a signed-log transform, slog(x) = sign(x) *
    // log1p(|x|), to every loss-scale quantity (local-sample f values,
    // train_loss, and the gen_gap_loss target) before any statistic or
    // model is fit, to test whether a small number of extreme-scale
    // datasets (e.g. very high-dimensional regression problems) are
    // distorting the LODO result -- see the article's Discussion and
    // Limitations section, "scale explanation". OFF by default; leaving it
    // unchecked reproduces the original train_gap_regressor.py behaviour
    // exactly (train_gap_regressor_logtransform.py is a strict superset).
    QCheckBox*      nnLogTransformCheck_ = nullptr;
    QPushButton*    nnRunBootstrapBtn_   = nullptr;  // stage 3 (optional): python bootstrap_lodo_ci.py
    BusySpinner*    nnBootstrapSpinner_  = nullptr;

    QProcess*       nnTrainProc_         = nullptr;
    QProcess*       nnAnalyzeProc_       = nullptr;
    QProcess*       nnBootstrapProc_     = nullptr;
    // Opt-in process-pool parallelism for stage 1 (see optimsolution.cfg's
    // [nnlandscape] parallel_jobs, read by nnLandscapeParallelJobsFromSettings()).
    // Left at 1 (and nnActiveProcs_ empty), the ORIGINAL serial code path
    // above (nnTrainProc_, one at a time) is used completely unchanged.
    // Only when parallel_jobs > 1 does startNNLandscapeJobSlots() launch
    // more than one --nnlandscape OS process at once, tracked here; see
    // its definition in MainWindow_NNLandscape.cpp for the full safety
    // reasoning (each job writes its own private summary row instead of
    // appending to the shared nn_summary.csv directly).
    int             nnParallelJobs_      = 1;
    QVector<QProcess*>    nnActiveProcs_;
    QMap<QProcess*,QString> nnPendingRowPath_;
    int             nnLandscapeParallelJobsFromSettings() const;
    // [nnlandscape] bootstrap_jobs from optimsolution.cfg -- passed to
    // bootstrap_lodo_ci.py as --jobs. See onRunNNLandscapeBootstrap().
    int             nnLandscapeBootstrapJobsFromSettings() const;
    void            startNNLandscapeJobSlots();
    // Session-scoped: true once at least one training run has appended a
    // row to nn_summary.csv, which is what unlocks "2. Analyze". Deliberately
    // NOT re-derived from disk on entering the mode (mirrors
    // pipelineStage1Done_'s own comment: a stale nn_summary.csv from an
    // earlier session shouldn't silently unlock stage 2 without the user
    // having done anything in THIS session).
    bool            nnStage1Done_        = false;
    // Mirrors nnStage1Done_ for stage 2: true once "2. Analyze" has
    // completed successfully at least once this session, which is what
    // unlocks the optional "3. Bootstrap CI" stage (it reads stage 2's
    // merged_dataset.csv, so it cannot run before stage 2 has written one).
    bool            nnStage2Done_        = false;
    int             nnOutputTabIndex_    = -1; // single tab reused by both stages
    QElapsedTimer   nnStage1Timer_;    // whole-batch elapsed (all queued runs)
    QElapsedTimer   nnStage2Timer_;
    QElapsedTimer   nnStage3Timer_;

    // Batch queue: each entry is {datasetPath, archSpec, optimizerName, seed}.
    // Built once when "1. Train + sample landscape" is pressed (cartesian
    // product of selected datasets x parsed architectures x selected
    // optimizers x range(runs), seed = seed_base + run_index), then drained
    // one at a time on nnTrainProc_'s finished signal -- same queue-advance
    // pattern as onPipelineFeaturesProcessFinished().
    QVector<QStringList> nnQueue_;
    int             nnQueueTotal_        = 0;
    int             nnQueueOk_           = 0;
    int             nnQueueFailed_       = 0;

    // Wilcoxon/Rankings tabs (inside the same output tab as Log), rebuilt
    // from nn_summary.csv after every batch completes. "Problem" here means
    // a (dataset, architecture) pair; "Method" means an optimizer -- the
    // same problem-as-subject pairing convention the framework's own
    // Wilcoxon/Friedman tabs use for algorithm comparison, just computed
    // fresh here rather than reusing the batch-benchmark-specific
    // rebuildStatsComparisons() pipeline (that one parses raw per-run CLI
    // text output; nn_summary.csv already has clean numeric columns).
    QComboBox*      nnWilcoxonMetricCombo_ = nullptr;
    QLabel*         nnWilcoxonSummaryLbl_  = nullptr;
    QWidget*        nnWilcoxonPlot_        = nullptr; // WilcoxonBoxPlotWidget
    QTableWidget*   nnRankingsTable_       = nullptr;
    // "The most important information" (raw + aggregated results tables,
    // exportable to XLSX -- the numbers that go directly into an article's
    // own results tables, not just a plot). Populated by rebuildNNStats()
    // alongside Wilcoxon/Rankings; the LODO/importance tables are populated
    // separately by onNNLandscapeAnalyzeProcessFinished() once stage 2
    // (Python) has written its own lodo_results.csv/feature_importance.csv.
    QTableWidget*   nnResultsTable_        = nullptr; // one row per run, every nn_summary.csv column
    QTableWidget*   nnMeanTable_           = nullptr; // one row per (dataset,arch,optimizer): mean+std+N
    QTableWidget*   nnLodoTable_           = nullptr; // from nn_analysis/lodo_results.csv (stage 2)
    QTableWidget*   nnImportanceTable_     = nullptr; // from nn_analysis/feature_importance.csv (stage 2)
    // from nn_analysis/merged_dataset_bootstrap_ci.csv (stage 3, optional):
    // median + 95% CI per (feature set, metric), cluster-bootstrapped over
    // datasets. Populated by onNNLandscapeBootstrapProcessFinished().
    QTableWidget*   nnBootstrapTable_      = nullptr;
    void rebuildNNStats(); // reads nn_summary.csv, refreshes Results/Mean/Wilcoxon/Rankings tabs
    void loadNNAnalysisTables(const QString& outDir); // reads lodo_results.csv/feature_importance.csv
    void loadNNBootstrapTable(const QString& outDir); // reads merged_dataset_bootstrap_ci.csv (stage 3)
    QString nnActiveRunDir() const; // nnLoadedRunDir_ if set (via "Load previous experiment..."), else the default runtime dir
    void setNNLandscapeUiEnabled(bool enabled); // mirrors setPipelineUiEnabled() for mode 7

    QWidget*     batchPanel_        = nullptr;
    QWidget*     sensProblemPanel_  = nullptr;
    QListWidget* sensProblemsList_  = nullptr;
    QTableWidget* sensProblemDimsTable_ = nullptr;

    // Sensitivity job queue and group tracking (GUI-driven sensitivity analysis).
    struct SensQueueJob {
      QString problem;
      int dim = 2;
      QMap<QString,QString> injectedParams;
      int groupId  = 0;
      int pointIdx = 0;
    };
    QVector<SensQueueJob> sensJobQueue_;
    int                   sensJobIndex_ = 0;
    QDateTime             sensJobStartTime_;
    QVector<SensGroup>    sensGroups_;
    QMap<QString,QString> sensOrigMethodParams_;
    QMap<QString,QString> sensOrigProblemParams_;
    bool                  sensIsProblemMode_ = false;  // true when mode==3 (problem sensitivity)
    QMap<int,int>         sensGroupTabIdx_; // groupId → output tab index
    QListWidget* batchMethodsList_ = nullptr;
    QListWidget* batchProblemsList_ = nullptr;
    QTableWidget* batchProblemDimsTable_ = nullptr; // Batch-only: per-problem dimension override table
    QSpinBox*   batchRunsSpin_ = nullptr; // Mirrors [global].runs (read-only).
    QComboBox*  batchAggCombo_ = nullptr;  // Aggregation for time-to-best metrics.
    QComboBox*  batchMetricCombo_ = nullptr; // Metric selector (Value definition).
    QLabel*     batchStatusLbl_ = nullptr;
    QProgressBar* batchProgress_ = nullptr;
    QCheckBox*  batchShowRateChk_ = nullptr;
    QCheckBox*  batchShowSdChk_   = nullptr;
    QCheckBox*  batchShowTimeChk_ = nullptr;

    // Batch-only: dimension override per problem (variable-dimension problems only).
    // Each entry stores one or more dimensions selected for the corresponding problem.
    // If a problem is fixed-dimension, the fixed dimension is always used.
    QMap<QString, QStringList> batchProblemDimOverride_;
// Batch execution state
    enum class BatchMetricMode : int {
        BestFinalBestF = 0,   // Value = min(final best_f) across runs
        MeanFinalBestF = 1,   // Value = mean(final best_f) across runs
        IterationAtBest = 2,  // Value = aggregation of iteration index where the (global) best is first reached
        EvalsAtBest = 3       // Value = aggregation of function evaluations where the (global) best is first reached
    };

    struct BatchCellData {
        QVector<double> finalBestF;   // per-run final best_f (last iteration)
        QVector<double> hitIter;      // per-run iteration where the global best is first reached (NaN if not reached)
        QVector<double> hitEvals;     // per-run evals where the global best is first reached (NaN if not reached)

        double bestOverall = std::numeric_limits<double>::quiet_NaN();
        double tol = 0.0;

        int nRuns = 0;
        int nSuccess = 0;

        // Convenience: computed at table rebuild time (depending on mode/aggregation).
        double value = std::numeric_limits<double>::quiet_NaN();
        double sd = std::numeric_limits<double>::quiet_NaN();
        double ratePct = 0.0;

        // Mean wall-clock time per run in seconds (from CLI summary output).
        double meanTimePerRun = std::numeric_limits<double>::quiet_NaN();
    };

    struct BatchJob {
        QString methodShort;
        QString problemShort;   // base problem short name used for the CLI
        QString problemKey;     // unique batch/stats key (e.g. "ackley [d=30]")
        int     dim = 0;
        qint64  startMsecsUtc = 0; // ms since epoch (UTC), used to locate the newest convergence CSV for this job
        double  timePerRunSecs = std::numeric_limits<double>::quiet_NaN(); // from CLI "Time per run (mean):" output
    };

    struct BatchPostResult {
        QString methodShort;
        QString problemShort;   // base problem short name used for the CLI
        QString problemKey;     // unique batch/stats key (e.g. "ackley [d=30]")
        int     dim = 0;
        qint64   startMsecsUtc = 0;
        QString  csvPath;
        BatchCellData cell;
        bool     ok = false;
        QString  message;
    };

    QVector<BatchJob> batchQueue_;
    int      batchJobIndex_ = 0;
    int      batchTotalJobs_ = 0;
    bool     batchActive_ = false;
    QThread* batchLogThread_ = nullptr;
    BatchLogWriter* batchLogWriter_ = nullptr;
    QString  batchLogFilePath_;
    QString  batchUiPendingLog_;
    QTimer*  batchUiFlushTimer_ = nullptr;
    QElapsedTimer batchProgressThrottle_;
    bool     batchStopRequested_ = false;

    // Progress tracking: poll convergence CSV for real-time run number.
    int       currentRunNumber_      = 1;
    int       singleRunsPerJob_      = 1;
    QTimer*   progressPollTimer_     = nullptr;
    QString   progressPollDir_;
    QDateTime progressPollStartTime_;
    // Single-run UI throttling (single run can be very verbose; avoid freezing the GUI).
    QString  singleUiPendingLog_;
    QTimer*  singleUiFlushTimer_ = nullptr;
    QElapsedTimer singleProgressThrottle_;

    // Sensitivity run CSV-based progress polling.
    // Polls the growing sensitivity_results.csv to show X/Y row progress when
    // the CLI does not emit parsable progress tokens to stdout.
    QTimer*  sensCsvPollTimer_        = nullptr;
    QString  sensCsvPollPath_;           // path of sensitivity output CSV (for final load)
    int      sensCsvExpectedRows_   = 0; // total sensitivity points (grid combinations)
    int      sensCsvLastRowCount_   = 0; // last observed completed point row count
    int      sensRunsPerPoint_      = 0; // runs per sensitivity point (from config)
    int      sensCurrentRunInPoint_ = 0; // current run index within active point (from stdout)
    int      sensRunsSeenTotal_     = 0; // cumulative "Run X/Y" occurrences seen in stdout
    int      sensTotalExpectedRuns_ = 0; // totalPoints × runsPerPoint
    bool     sensFirstChunkLogged_  = false; // debug: log first stdout line once per run
    bool     sensPollPathLogged_    = false; // debug: log CSV poll path once per run
    int      sensMaxEvalsPerRun_    = 0;     // max_evals from config (per single run)
    int      sensCompletedRuns_     = 0;     // runs fully completed (evals reset detected)
    int      sensCurrentEvals_      = 0;     // evals in the current active run
    int      sensPrevEvals_         = 0;     // previous evals value (to detect run reset)
    QElapsedTimer sensElapsed_;              // wall-clock timer started when poll starts
    qint64   sensFirstRunMs_        = -1;    // ms elapsed when first run completed (calibration)

    // Batch post-processing (CSV scanning/parsing) is performed off the GUI thread to keep the UI responsive
    // when many problems/methods are selected.
    QFutureWatcher<BatchPostResult>* batchPostWatcher_ = nullptr;
    bool     batchPostInFlight_ = false;
    QString  batchCurMethodShort_;
    QString  batchCurProblemShort_;
    int      batchCurRunIndex_ = 0;
    quint32  batchCurSeed_ = 0;

    // Legacy container (kept for backwards compatibility with older v37 code paths).
    QMap<QString, QMap<QString, QVector<double>>> batchMetrics_;

    // New: raw per-cell data derived from convergence CSV.
    QMap<QString, QMap<QString, BatchCellData>> batchCells_;
    // Loaded batch-summary tabs keep isolated per-page cell caches so switching between
    // multiple loaded experiments does not leak state through the global batchCells_.
    QMap<QWidget*, QMap<QString, QMap<QString, BatchCellData>>> batchSummaryCellsByPage_;
    QMap<QString, int> batchCachedProblemDims_;
    // Statistics UI
    QPushButton*  exportStatsBtn_ = nullptr;
    QTableWidget* statsTable_ = nullptr;
    QLabel*       statsNoteLbl_ = nullptr;
    QPushButton*  generateReportBtn_ = nullptr;
    QComboBox*    statsAlphaCombo_ = nullptr;
    QTabWidget*   statsTabs_ = nullptr;
    // Wilcoxon tab
    QComboBox*    wilcoxonPairsCombo_ = nullptr;
    QPushButton*  exportWilcoxonPlotBtn_ = nullptr;
    QWidget*      wilcoxonPlot_ = nullptr;      // WilcoxonBoxPlotWidget
    QLabel*       wilcoxonSummaryLbl_ = nullptr;
    // Friedman tab
    QPushButton*  exportRankPlotBtn_ = nullptr;
    QWidget*      rankPlot_ = nullptr;          // RankPlotWidget
    QTableWidget* statsPairwiseTable_ = nullptr;
    QLabel*       statsSummaryLbl_ = nullptr;
    // Critical Difference (Nemenyi) tab
    QPushButton*  exportCdPlotBtn_ = nullptr;
    QWidget*      cdPlot_ = nullptr;            // CDDiagramWidget
    QLabel*       cdSummaryLbl_ = nullptr;
    // Landscape animation (single shared timer drives whichever tab started playback)
    QTimer*       landscapeTimer_ = nullptr;
    int           landscapePlayingTabIndex_ = -1;
    // v40: incremental batch statistics table updates (avoid rebuilding the full table per job).
    bool statsBatchTableActive_ = false;
    QStringList statsBatchMethods_;
    QStringList statsBatchProblems_;
    QMap<QString, int> statsRowByProblem_;
    QMap<QString, int> statsColBaseByMethod_;
    int statsPerMethodCols_ = 1;
    bool statsShowRate_ = true;
    bool statsShowSd_ = true;
    bool statsShowTime_ = true;
    BatchMetricMode statsMode_ = BatchMetricMode::BestFinalBestF;
    QString statsAgg_;
  BusySpinner* busySpinner_ = nullptr;
  QPushButton* refreshBtn_ = nullptr;
  QPushButton* selectCfgBtn_ = nullptr;
  QPushButton* reloadCfgBtn_ = nullptr;
  QPushButton* saveCfgBtn_   = nullptr;
  QPushButton* saveBatchSelectionBtn_ = nullptr;
  QPushButton* loadBatchSelectionBtn_ = nullptr;
  QPushButton* loadExperimentCsvBtn_  = nullptr;
  QWidget*     progressWidget_        = nullptr;
  // Live "time elapsed" readout shown ABOVE-RIGHT of outputProgressBar_
  // (inside progressWidget_ -- see its construction in MainWindow_Ui.cpp),
  // ticking once a second WHILE a stage actually runs. Wired up only via
  // startPipelineProgress()/finishPipelineProgress(), which only the
  // Meta-dataset pipeline (mode 6) and NN landscape (mode 7) stage
  // functions call -- so it only ever appears for those two modes. Fully
  // separate from outputElapsedLbl_/runElapsedTimer_ below (single-run/
  // batch mode's own elapsed readout) and from each mode's own per-stage
  // "(Xs, N/M ok)" labels (pipelineBatchTimeLbl, nnTrainTimeLbl, ...),
  // neither of which this change touches.
  QLabel*      pipelineElapsedLbl_    = nullptr;
  QTimer*      pipelineElapsedTimer_  = nullptr;
  QElapsedTimer pipelineElapsedClock_;
  void         updatePipelineElapsedLabel();
  QLabel*      outputStatusLbl_       = nullptr;
  QLabel*      outputElapsedLbl_      = nullptr;
  QTimer*      runElapsedTimer_       = nullptr;
  QElapsedTimer runElapsed_;

  QPushButton* addGlobalParamBtn_ = nullptr;
  QPushButton* removeGlobalParamBtn_ = nullptr;

  QPushButton* addStopParamBtn_ = nullptr;
  QPushButton* removeStopParamBtn_ = nullptr;

  QPushButton* addInitParamBtn_ = nullptr;
  QPushButton* removeInitParamBtn_ = nullptr;

  QPushButton* addMethodParamBtn_ = nullptr;
  QPushButton* removeMethodParamBtn_ = nullptr;

  QPushButton* addSensitivityParamBtn_ = nullptr;
  QPushButton* removeSensitivityParamBtn_ = nullptr;

  QTabWidget* tabs_ = nullptr;
  QTableWidget* runTable_ = nullptr;
  QTableWidget* stopTable_ = nullptr;
  QTableWidget* initTable_ = nullptr;
  QTableWidget* methodTable_ = nullptr;
  QTableWidget* sensitivityTable_ = nullptr;
  QCheckBox* sensitivityEnableChk_ = nullptr;

  ModeAwareTabWidget* outputTabs_ = nullptr;   // run tabs of all modes; each mode tab shows only its own (see ModeAwareTabWidget.h)
  QPushButton* reloadConvergenceBtn_ = nullptr;
  QPushButton* clearCsvBtn_ = nullptr;
  QPushButton* openOutputFolderBtn_ = nullptr;
  // "Load previous experiment" (Meta-dataset pipeline mode only) -- lives
  // in the main Output toolbar, right after "Delete output files", visible
  // only for that mode (see updateBatchPanelVisibility()).
  QPushButton* pipelineLoadExperimentBtn_ = nullptr;
  BusySpinner* pipelineLoadExperimentSpinner_ = nullptr;
  QCheckBox* forceConvergenceCsvChk_ = nullptr;

  // Output progress (separate from the busy spinner near Run/Stop).
  QProgressBar* outputProgressBar_ = nullptr;
  bool outputProgressDeterminate_ = false;
  int outputProgressValue_ = 0;
  int outputProgressMaximum_ = 0;

  std::vector<OutputRunTab> outputRuns_;
  int activeOutputRunIndex_ = -1;

  // State
  QString projectRoot_;
  QString settingsPath_;
  int currentFixedDim_ = 0;
  // Non-empty only for multi-value-restricted problems (e.g. CEC2017's
  // D in {10,30,50}); see allowedDimsForProblem()/snapDimSpinToAllowed().
  std::vector<int> currentAllowedDims_;

  std::unique_ptr<ConfigFile> cfg_;


  // Last-run state (for auto-retry behavior and Run/Stop UX)
  QStringList lastArgs_;
  bool lastHadDimension_ = false;
  bool lastAutoRetriedFixedDim_ = false;
  QString processTextBuffer_;

  QString lastRuntimeWorkingDir_;
  QString lastRuntimeCfgPath_;

  QProcess* proc_ = nullptr;
  QString ansiCarryStdout_;
  QString ansiCarryStderr_;
};

} // namespace optimsolution_gui
