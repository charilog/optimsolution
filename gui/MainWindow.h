#pragma once
#include <QMainWindow>
#include <QTextEdit>
#include <QColor>
#include <QProcess>
#include <QVector>
#include <memory>
#include <vector>

QT_BEGIN_NAMESPACE
class QComboBox;
class QSpinBox;
class QTableWidget;
class QPushButton;
class QTabWidget;
class QLabel;
class QCheckBox;
QT_END_NAMESPACE

namespace optimsolution_gui {

class ConfigFile;

class MainWindow final : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;

private:
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
    QPushButton* exportConvergencePngBtn = nullptr;
    QVector<double> convX;
    QVector<double> convY;
    bool convLoaded = false;
    QString runtimeWorkingDir;
    QString runtimeCfgPath;
  };

private slots:
  void onRefreshFactory();
  void onProblemChanged(const QString& problem);
  void onRunClicked();
  void onSelectSettings();
  void onReloadSettings();
  void onSaveSettings();
  void onMethodChanged(const QString& method);

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

  void onExportConvergencePng();

  void onOutputTabCloseRequested(int index);
  void onOverlayMethodToggled(bool checked);

  void onSettingItemChanged(int row, int column);

private:
  void buildUi();
  void loadFactoryLists();
  void loadSettings();
  void populateSettingsTables();
  void updateDimUiForProblem(const QString& problem);
  int  fixedDimForProblem(const QString& problem) const;

  int createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim);
  OutputRunTab* outputTab(int index);
  const OutputRunTab* outputTab(int index) const;
  OutputRunTab* activeOutputTab();
  void tryLoadConvergenceForTab(int index);
  void updateConvergencePlotForTab(int index);

  bool exportConvergencePlotPng(int outputTabIndex, const QString& filePath);

  void appendLog(const QString& text);
  void appendLogStyled(const QString& line, const QColor& color, bool bold=false);
  void appendFinishSummaryColored(int exitCode);
  void startCliProcess(const QStringList& args);
  void requestStop();
  QString cliPath() const;
  QString currentMethodShort() const;
  bool detectCliSupportsConfigArg(const QString& cli);

  void setTablesEditable(bool editable);

  // Guards against signal re-entrancy while the UI is programmatically updated.
  bool internalUpdate_ = false;

  // Cached capability probe for the CLI.
  // -1: unknown, 0: no, 1: yes
  int cliSupportsConfigArg_ = -1;

  // Widgets
  QComboBox* methodBox_ = nullptr;
  QComboBox* problemBox_ = nullptr;
  QSpinBox*  dimSpin_ = nullptr;

  QPushButton* runBtn_ = nullptr;
  QPushButton* refreshBtn_ = nullptr;
  QPushButton* selectCfgBtn_ = nullptr;
  QPushButton* reloadCfgBtn_ = nullptr;
  QPushButton* saveCfgBtn_ = nullptr;

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

  QTabWidget* outputTabs_ = nullptr;
  QPushButton* reloadConvergenceBtn_ = nullptr;
  QCheckBox* forceConvergenceCsvChk_ = nullptr;

  std::vector<OutputRunTab> outputRuns_;
  int activeOutputRunIndex_ = -1;

  // State
  QString projectRoot_;
  QString settingsPath_;
  QString factoryPath_;
  QString fixedDimsPath_;
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
