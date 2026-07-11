#pragma once
#include <QMainWindow>
#include <QTextEdit>
#include <QColor>
#include <QProcess>
#include <QVector>
#include <QMap>
#include <QFutureWatcher>
#include <QElapsedTimer>
#include <QDateTime>
#include <QThread>
#include <memory>
#include <vector>
#include <limits>

QT_BEGIN_NAMESPACE
class QComboBox;
class QListWidget;
class QSpinBox;
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
  };

  // Region focus (Selection / Settings / Output)
  enum class FocusArea : int { None = 0, Selection = 1, Settings = 2, Output = 3 };

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

private:
  void buildUi();
  void checkAndAutoRebuild();   // auto-build & restart if .rebuild_pending exists
  void applyRegionStyling();
  void toggleRegionFocus(FocusArea area);
  void restoreRegionLayout();
  void updateRegionFocusStyling();
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
  void updateDimUiForProblem(const QString& problem);
  int  fixedDimForProblem(const QString& problem) const;

  int createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim);
  OutputRunTab* outputTab(int index);
  const OutputRunTab* outputTab(int index) const;
  OutputRunTab* activeOutputTab();
  const OutputRunTab* activeOutputTab() const;
  void tryLoadConvergenceForTab(int index);
  void updateConvergencePlotForTab(int index);

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
    double aggregateValues(const QVector<double>& vals, const QString& agg) const;
    double stdevValues(const QVector<double>& vals) const;
    bool exportStatsToXlsxNative(const QString& outPath) const;
    bool exportStatsToXlsxViaPython(const QString& outPath, const QString& csvPath);
    bool exportSensitivityPlotPng(int outputTabIndex, const QString& filePath);
    bool exportRankPlotPng(const QString& filePath);
    bool exportWilcoxonPlotPng(const QString& filePath);
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

  // Region focus state (Selection / Settings / Output)
  FocusArea focusedArea_ = FocusArea::None;
  QVector<int> selectionSettingsSplitterSizes_;

  // Region widgets and controls (for focus/restore UI only)
  QGroupBox* selectionBox_ = nullptr;
  QGroupBox* settingsBox_ = nullptr;
  QGroupBox* outputBox_ = nullptr;
  QGroupBox* wizardBox_ = nullptr;
  QSplitter* selectionSettingsSplitter_ = nullptr;
  QSplitter* bottomSplit_ = nullptr;
  QPushButton* selectionMaxBtn_ = nullptr;
  QPushButton* settingsMaxBtn_ = nullptr;
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

  QTabWidget* outputTabs_ = nullptr;
  QPushButton* reloadConvergenceBtn_ = nullptr;
  QPushButton* clearCsvBtn_ = nullptr;
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
