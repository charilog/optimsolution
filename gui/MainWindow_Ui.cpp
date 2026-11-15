// MainWindow_Ui.cpp
// MainWindow construction: constructor/destructor, auto-rebuild check, buildUi(), region styling & focus.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"
#include "MainTabWidget.h"
#include "WrapToolbarLayout.h"
#include <functional>
#include <memory>
#include <QAbstractItemView>
#include <QAbstractItemModel>
#include <QEvent>
#include <QSysInfo>
#include <QUrl>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDesktopServices>
#include <QGuiApplication>
#include <QClipboard>
#include <QGridLayout>
#include <QPointer>
#include <QProgressDialog>
#include <QThreadPool>
#include <QEventLoop>
#include <atomic>
#include <QScrollArea>

namespace optimsolution_gui {

namespace {
constexpr const char* kGuiVersion = "56";

// Some entries of the method list are very long (a whole description), and a
// combo box by default is as wide as its widest entry -- which made the
// Selection area, and with it the window, more than 1400 px wide in the modes
// that show the method/problem boxes. The boxes now keep a modest width and
// simply fill the space they are given; the drop-down list itself is still as
// wide as its longest entry, so nothing becomes unreadable.
void makeComboCompact(QComboBox* cb) {
  if (!cb) return;
  cb->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
  cb->setMinimumContentsLength(18);

  auto pending = std::make_shared<bool>(false);
  const QPointer<QComboBox> guard(cb);
  auto schedule = [guard, pending]() {
    if (!guard || *pending) return;
    *pending = true;
    QTimer::singleShot(0, guard.data(), [guard, pending]() {
      *pending = false;
      if (!guard || !guard->view()) return;
      const QFontMetrics fm(guard->view()->font());
      int widest = 0;
      for (int i = 0; i < guard->count(); ++i)
        widest = std::max(widest, fm.horizontalAdvance(guard->itemText(i)));
      guard->view()->setMinimumWidth(widest + 48);   // + scroll bar / margins
    });
  };
  if (QAbstractItemModel* m = cb->model()) {
    QObject::connect(m, &QAbstractItemModel::rowsInserted, cb, schedule);
    QObject::connect(m, &QAbstractItemModel::rowsRemoved,  cb, schedule);
    QObject::connect(m, &QAbstractItemModel::modelReset,   cb, schedule);
    QObject::connect(m, &QAbstractItemModel::dataChanged,  cb, schedule);
  }
  schedule();
}
}  // namespace

void MainWindow::tryLoadSensitivityForTab(int index) {
  if (index < 0 || index >= static_cast<int>(outputRuns_.size())) return;
  OutputRunTab& tab = outputRuns_[index];
  if (tab.isPortfolioRun) return;   // Portfolio tabs have no sensitivity_results.csv

  if (!tab.sensitivityLog || !tab.sensitivityParamCombo || !tab.sensitivityPlot || !tab.sensitivitySummaryTable) return;

  auto* plot = dynamic_cast<SensitivityBarWidget*>(tab.sensitivityPlot);
  if (plot) plot->clear();

  tab.sensitivitySummaryTable->setRowCount(0);
  tab.sensLoaded = false;

  // Sensitivity metric is fixed: Mean best_f (stored as mean_f in the CSV).
  if (tab.runtimeWorkingDir.isEmpty()) {
    tab.sensitivityLog->setPlainText("Sensitivity results will appear here after the run.");
    return;
  }

  // If the tab already has a pinned CSV path (set by multi-job sensitivity queue),
  // use it directly instead of re-deriving from config.
  QString csvPath;
  if (!tab.sensitivityCsvPath.isEmpty() && QFileInfo::exists(tab.sensitivityCsvPath)) {
    csvPath = tab.sensitivityCsvPath;
  } else {
    QString csvName = "sensitivity_results.csv";
    const QString configured = cfg_ ? cfg_->value("sensitivity", "output") : QString();
    if (!configured.trimmed().isEmpty()) {
      // Strip inline comments (e.g. "file.csv ; comment") — take only text before ';' or '#'.
      QString clean = configured;
      const int semi = clean.indexOf(';');
      if (semi >= 0) clean = clean.left(semi);
      const int hash = clean.indexOf('#');
      if (hash >= 0) clean = clean.left(hash);
      clean = clean.trimmed();
      if (!clean.isEmpty()) csvName = clean;
    }

    const QDir wd(tab.runtimeWorkingDir);
    csvPath = wd.filePath(csvName);

    // If configured file does not exist, try to locate a likely sensitivity CSV in the working directory.
    if (!QFileInfo::exists(csvPath)) {
      QString best;
      QDirIterator it(tab.runtimeWorkingDir, {"*.csv"}, QDir::Files, QDirIterator::Subdirectories);
      while (it.hasNext()) {
        const QString p = it.next();
        const QString base = QFileInfo(p).fileName().toLower();
        if (base.contains("sensitivity")) {
          best = p;
          if (base == "sensitivity_results.csv") break;
        }
      }
      if (!best.isEmpty()) csvPath = best;
    }

    tab.sensitivityCsvPath = csvPath;
  }

  if (!QFileInfo::exists(csvPath)) {
    tab.sensitivityLog->setPlainText(QString("Sensitivity: no results file found (%1).").arg(csvPath));
    return;
  }

  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    tab.sensitivityLog->setPlainText(QString("Sensitivity: failed to open %1").arg(csvPath));
    return;
  }

  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);

  const QString headerLine = ts.readLine().trimmed();
  if (headerLine.isEmpty()) {
    tab.sensitivityLog->setPlainText(QString("Sensitivity: empty CSV file: %1").arg(csvPath));
    return;
  }

  QStringList headers = headerLine.split(',', Qt::KeepEmptyParts);
  for (auto& h : headers) h = h.trimmed();

  // Identify parameter columns by excluding known metadata/outcome columns.
  QSet<QString> reserved;
  const QStringList reservedList = {
    "method","problem","dim","runs",
    "mean_best_f","mean_f","stdev_best_f","stdev_f","min_f","max_f","best_f",
    "mean_evals","stdev_evals","min_evals","max_evals",
    "mean_time","stdev_time","time",
    "success_rate","success","seed"
  };
  for (const auto& r : reservedList) reserved.insert(r);

  QStringList paramCols;
  for (const auto& hname : headers) {
    const QString k = hname.trimmed().toLower();
    if (k.isEmpty()) continue;
    if (reserved.contains(k)) continue;
    if (k.startsWith("mean_") || k.startsWith("stdev_") || k.startsWith("min_") || k.startsWith("max_")) continue;
    paramCols << hname;
  }
  paramCols.removeDuplicates();

  // Filter parameters by the run-time [sensitivity].params list, so the output dropdown
  // shows only the parameters that were actually swept for this run.
  {
    QStringList cfgParams;
    if (!tab.runtimeCfgPath.isEmpty()) {
      ConfigFile runCfg;
      if (runCfg.load(tab.runtimeCfgPath)) {
        cfgParams = runCfg.value(QStringLiteral("sensitivity"), QStringLiteral("params"), QString()).split(",", Qt::SkipEmptyParts);
      }
    }

    QStringList normalizedCfgParams;
    normalizedCfgParams.reserve(cfgParams.size());
    for (const QString& p : cfgParams) {
      QString pp = p.trimmed();
      if (pp.startsWith(QStringLiteral("values."))) {
        pp = pp.mid(QStringLiteral("values.").size());
      }
      if (!pp.isEmpty()) {
        normalizedCfgParams.push_back(pp);
      }
    }
    normalizedCfgParams.removeDuplicates();

    if (!normalizedCfgParams.isEmpty()) {
      QStringList filtered;
      filtered.reserve(normalizedCfgParams.size());
      for (const QString& p : normalizedCfgParams) {
        if (paramCols.contains(p)) {
          filtered.push_back(p);
        }
      }
      if (!filtered.isEmpty()) {
        paramCols = filtered;
      }
    }
  }


  tab.sensitivityParamCombo->blockSignals(true);
  const QString current = tab.sensitivityParamCombo->currentText().trimmed();
  tab.sensitivityParamCombo->clear();
  tab.sensitivityParamCombo->addItems(paramCols);
  tab.sensitivityParamCombo->blockSignals(false);

  if (paramCols.isEmpty()) {
    tab.sensitivityLog->setPlainText(QString("Sensitivity: no parameter columns found in %1").arg(csvPath));
    return;
  }

  QString toSelect = current;
  if (toSelect.isEmpty() || !paramCols.contains(toSelect)) toSelect = paramCols.front();
  const int selIndex = tab.sensitivityParamCombo->findText(toSelect);
  if (selIndex >= 0) tab.sensitivityParamCombo->setCurrentIndex(selIndex);

  populateSensitivitySummaryAllParams(index, csvPath, paramCols);
  renderSensitivityForParam(index, toSelect);
}



MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  cfg_ = std::make_unique<ConfigFile>();

  projectRoot_ = PathUtils::findProjectRootFrom(QDir::currentPath());
  if (projectRoot_.isEmpty()) projectRoot_ = QDir::currentPath();
  settingsPath_ = PathUtils::findExistingFileUpwards(QDir::currentPath(), "optimsolution.cfg");

  CrashLog::append("MainWindow: ctor begin.");
  CrashLog::append(QString("MainWindow: root=%1").arg(projectRoot_));
  CrashLog::append(QString("MainWindow: settings=%1").arg(settingsPath_));
  buildUi();
  CrashLog::append("MainWindow: UI built.");
  // If a method/problem was added or deleted since last build, auto-rebuild.
  // This must run after buildUi() but before loadFactoryLists().
  checkAndAutoRebuild();
  loadFactoryLists();
  CrashLog::append("MainWindow: factory lists loaded.");
  loadSettings();
  CrashLog::append("MainWindow: settings loaded.");
  populateSettingsTables();
  CrashLog::append("MainWindow: settings tables populated.");

  setWindowTitle(QString("OptimSolution v%1").arg(kGuiVersion));
}

// ──────────────────────────────────────────────────────────────────────
// checkAndAutoRebuild
//
// Called once on startup.  If .rebuild_pending exists (written by
// CodeGenDialog when a method/problem is created or deleted), we ask
// the user whether to build & restart now.
//
// KEY design decision to avoid an infinite-ask loop:
//   - The flag is deleted BEFORE the build script is launched.
//   - If cmake fails, the script re-creates the flag so the next
//     startup will ask again — but we never loop unconditionally.
//
// Why a script instead of running cmake directly?
//   On Windows the linker cannot replace the running .exe (LNK1104).
//   The script waits ~2 s for this process to fully exit, then builds.
// ──────────────────────────────────────────────────────────────────────
void MainWindow::checkAndAutoRebuild()
{
  const QString flagPath   = QDir(projectRoot_).filePath(".rebuild_pending");
  const QString errorsPath = QDir(projectRoot_).filePath(".rebuild_errors.txt");

  if (!QFileInfo::exists(flagPath)) return;

  // ── Previous build failed: show errors and open without rebuilding ──
  if (QFileInfo::exists(errorsPath)) {
    QString errorText;
    QFile ef(errorsPath);
    if (ef.open(QIODevice::ReadOnly | QIODevice::Text)) {
      errorText = QTextStream(&ef).readAll().trimmed();
      ef.close();  // must close before remove — on Windows open files cannot be deleted
    }

    // Read the source file paths saved at generation time.
    QString hPath, cppPath;
    const QString sourcesPath = QDir(projectRoot_).filePath(".rebuild_sources.txt");
    QFile sf(sourcesPath);
    if (sf.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream ts(&sf);
      hPath   = ts.readLine().trimmed();
      cppPath = ts.readLine().trimmed();
      sf.close();  // close before remove
    }

    // Clear all error/flag/sources files so we don't loop on next startup.
    QFile::remove(errorsPath);
    QFile::remove(sourcesPath);
    QFile::remove(flagPath);

    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Build errors — manual fix required");
    msgBox.setIcon(QMessageBox::Warning);
    msgBox.setText(
        "The previous rebuild attempt failed.\n\n"
        "OptimSolution will open without rebuilding.\n\n"
        "Click \"Open in Code Editor\" to fix the errors shown below,\n"
        "then click  Save & Restart  to rebuild.");
    if (!errorText.isEmpty())
      msgBox.setDetailedText(errorText);

    QPushButton* openBtn = nullptr;
    const bool filesExist = !hPath.isEmpty() && !cppPath.isEmpty()
                            && QFileInfo::exists(hPath)
                            && QFileInfo::exists(cppPath);
    if (filesExist)
      openBtn = msgBox.addButton("Open in Code Editor", QMessageBox::ActionRole);
    msgBox.addButton(QMessageBox::Ok);
    msgBox.exec();

    if (openBtn && msgBox.clickedButton() == openBtn) {
      auto* editor = new optimsolution::CodeEditorDialog(
                          hPath, cppPath, projectRoot_, this);
      editor->setAttribute(Qt::WA_DeleteOnClose);
      editor->show();
    }
    return;  // open normally — user fixes errors in the editor
  }

  // ── Normal case: ask whether to build now ───────────────────────────
  const auto ans = QMessageBox::question(
      this,
      "Rebuild required",
      "A method or problem was added or removed since the last build.\n\n"
      "The application needs to rebuild and restart to reflect these changes.\n\n"
      "Build & Restart now?",
      QMessageBox::Yes | QMessageBox::No,
      QMessageBox::Yes);

  if (ans != QMessageBox::Yes) return;

  // ── Delete the flag NOW before we do anything else ─────────────────
  // This prevents an infinite-ask loop: even if cmake fails, the loop
  // won't repeat unless the script explicitly re-creates the flag.
  QFile::remove(flagPath);

  const QString appExe  = QCoreApplication::applicationFilePath();
  const QStringList appArgs = QCoreApplication::arguments().mid(1);
  const QString appArgsStr  = appArgs.isEmpty() ? QString() : " " + appArgs.join(" ");

#ifdef Q_OS_WIN
  // ── Windows: use PowerShell with hidden window — no console popup ──
  // On failure: saves compiler output to .rebuild_errors.txt and
  // re-creates .rebuild_pending so the next startup shows the errors.
  const QString nRoot   = QDir::toNativeSeparators(projectRoot_).replace("'", "''");
  const QString nFlag   = QDir::toNativeSeparators(flagPath).replace("'", "''");
  const QString nErrors = QDir::toNativeSeparators(errorsPath).replace("'", "''");
  const QString nExe    = QDir::toNativeSeparators(appExe).replace("'", "''");

  const QString psCmd = QString(
      "Start-Sleep -Seconds 2; "
      "Set-Location '%1'; "
      "$out = cmake --build build --config Release 2>&1 | Out-String; "
      "if ($LASTEXITCODE -ne 0) { "
          "$out | Out-File -Encoding utf8 '%2'; "
          "'Rebuild needed' | Out-File '%3' "
      "}; "
      "Start-Process '%4'%5")
      .arg(nRoot, nErrors, nFlag, nExe,
           appArgsStr.isEmpty()
               ? QString()
               : QString(" -ArgumentList '%1'").arg(appArgsStr.trimmed()));

  QProcess::startDetached("powershell.exe",
      {"-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
       "-Command", psCmd});
#else
  // ── Linux / macOS: write a shell script ───────────────────────────
  // On failure: saves compiler output to .rebuild_errors.txt and
  // re-creates .rebuild_pending so the next startup shows the errors.
  const QString scriptPath = QDir(projectRoot_).filePath("_rebuild_restart.sh");
  {
    QFile s(scriptPath);
    if (s.open(QIODevice::WriteOnly | QIODevice::Text)) {
      QTextStream t(&s);
      t << "#!/bin/bash\n";
      t << "sleep 2\n";
      t << "cd \"" << projectRoot_ << "\"\n";
      t << "cmake_out=$(cmake --build build --config Release 2>&1)\n";
      t << "if [ $? -ne 0 ]; then\n";
      t << "    echo \"$cmake_out\" > \"" << errorsPath << "\"\n";
      t << "    echo 'Rebuild needed' > \"" << flagPath << "\"\n";
      t << "fi\n";
      t << "\"" << appExe << "\"" << appArgsStr << " &\n";
    }
  }
  QProcess::execute("chmod", {"+x", scriptPath});
  QProcess::startDetached("/bin/bash", {scriptPath});
#endif

  // Schedule quit for the next event-loop tick so the process starts
  // cleanly.  The script waits 2 s, so by then the exe lock is released.
  QTimer::singleShot(0, qApp, &QApplication::quit);
}

MainWindow::~MainWindow() {
  stopBatchLogWriter();
}

void MainWindow::buildUi() {
  auto* central = new QWidget(this);
  auto* rootLay = new QVBoxLayout(central);

  selectionBox_ = new QGroupBox("Selection", central);
  selectionBox_->setObjectName("selectionBox");
  selectionBox_->setProperty("focused", false);
  selectionBox_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  auto* topBox = selectionBox_;
  auto* topBoxOuterLay = new QVBoxLayout(topBox);
  topBoxOuterLay->setContentsMargins(6, 6, 6, 6);
  topBoxOuterLay->setSpacing(0);
  auto* formContainer = new QWidget(topBox);
  auto* form = new QFormLayout(formContainer);
  form->setContentsMargins(0, 0, 0, 0);
  selectionForm_ = form;
  topBoxOuterLay->addWidget(formContainer, 0);

  methodBox_ = new QComboBox(topBox);
  methodBox_->setEditable(true);
  methodBox_->setInsertPolicy(QComboBox::NoInsert);
  methodBox_->setSizeAdjustPolicy(QComboBox::AdjustToContents);

  problemBox_ = new QComboBox(topBox);
  problemBox_->setEditable(true);
  problemBox_->setInsertPolicy(QComboBox::NoInsert);
  problemBox_->setSizeAdjustPolicy(QComboBox::AdjustToContents);

  dimSpin_ = new QSpinBox(topBox);
  dimSpin_->setRange(2, 100000);
  dimSpin_->setValue(2);
  // For multi-value-restricted problems (e.g. CEC2017: D in {10,30,50}),
  // arrow-clicks/wheel already land exactly on allowed values via the
  // singleStep configured in updateDimUiForProblem(); this only catches a
  // manually TYPED unsupported value once editing finishes (Enter/blur),
  // so it doesn't fight the user mid-keystroke -- see
  // snapDimSpinToAllowed().
  connect(dimSpin_, &QAbstractSpinBox::editingFinished, this, [this]() {
    snapDimSpinToAllowed();
  });

  // Buttons row — constructed here so it can be added to form ABOVE Run mode.
  refreshBtn_ = new QPushButton("Refresh lists", topBox);
  refreshBtn_->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
  refreshBtn_->setIconSize(QSize(18, 18));

  saveBatchSelectionBtn_ = new QPushButton("Save batch selection...", topBox);
  saveBatchSelectionBtn_->setObjectName("saveBatchSelectionBtn");
  saveBatchSelectionBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  saveBatchSelectionBtn_->setIconSize(QSize(18, 18));
  saveBatchSelectionBtn_->setToolTip("Save the currently selected batch methods and problems.");
  saveBatchSelectionBtn_->setVisible(false);

  loadBatchSelectionBtn_ = new QPushButton("Load batch selection...", topBox);
  loadBatchSelectionBtn_->setObjectName("loadBatchSelectionBtn");
  loadBatchSelectionBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
  loadBatchSelectionBtn_->setIconSize(QSize(18, 18));
  loadBatchSelectionBtn_->setToolTip("Load a previously saved batch selection of methods and problems.");
  loadBatchSelectionBtn_->setVisible(false);

  // Mode 7's own "Save/Load batch selection..." -- same .batchsel-style
  // plain-text file convention as the batch/pipeline ones above, but for
  // this mode's own selection shape (Datasets + Optimizers lists, Hidden
  // layers, Activation) instead of batchMethodsList_/batchProblemsList_, so
  // they're separate dedicated widgets/handlers (see
  // onSaveNNBatchSelection()/onLoadNNBatchSelection() in
  // MainWindow_NNLandscape.cpp) rather than reusing the buttons above.
  nnSaveSelectionBtn_ = new QPushButton("Save batch selection...", topBox);
  nnSaveSelectionBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  nnSaveSelectionBtn_->setIconSize(QSize(18, 18));
  nnSaveSelectionBtn_->setToolTip("Save the currently selected datasets, optimizers, and architectures (Mode 7).");
  nnSaveSelectionBtn_->setVisible(false);
  connect(nnSaveSelectionBtn_, &QPushButton::clicked, this, &MainWindow::onSaveNNBatchSelection);

  nnLoadSelectionBtn_ = new QPushButton("Load batch selection...", topBox);
  nnLoadSelectionBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
  nnLoadSelectionBtn_->setIconSize(QSize(18, 18));
  nnLoadSelectionBtn_->setToolTip("Load a previously saved datasets/optimizers/architectures selection (Mode 7).");
  nnLoadSelectionBtn_->setVisible(false);
  connect(nnLoadSelectionBtn_, &QPushButton::clicked, this, &MainWindow::onLoadNNBatchSelection);

  runBtn_ = new QPushButton("Run", topBox);
  runBtn_->setIconSize(QSize(20, 20));
  runBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_MediaPlay), QColor(0, 160, 0), QSize(20, 20), 2.0));
  runBtn_->setStyleSheet("QPushButton { color: #00a000; font-weight: bold; }");

  busySpinner_ = new BusySpinner(topBox);
  busySpinner_->setFixedSize(18, 18);
  busySpinner_->stop();

  selectionMaxBtn_ = new QPushButton("Maximize", topBox);
  selectionMaxBtn_->setProperty("role", "regionMax");
  selectionMaxBtn_->setToolTip("Expand the Selection area and hide Output. Click again to restore both areas.");
  selectionMaxBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_TitleBarMaxButton), QColor(0, 120, 215), QSize(18, 18), 2.0));
  selectionMaxBtn_->setIconSize(QSize(18, 18));
  selectionMaxBtn_->setFlat(true);
  connect(selectionMaxBtn_, &QPushButton::clicked, this, [this]() { toggleRegionFocus(FocusArea::Selection); });

  {
    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(refreshBtn_);
    btnRow->addWidget(saveBatchSelectionBtn_);
    btnRow->addWidget(loadBatchSelectionBtn_);
    btnRow->addWidget(nnSaveSelectionBtn_);
    btnRow->addWidget(nnLoadSelectionBtn_);
    btnRow->addStretch(1);
    btnRow->addSpacing(8);
    btnRow->addWidget(selectionMaxBtn_);
    form->addRow(btnRow);
  }

  runModeBox_ = new QComboBox(topBox);
  runModeBox_->addItem("Single run", 0);
  runModeBox_->addItem("Batch run (selected methods/problems)", 1);
  runModeBox_->addItem("Sensitivity analysis of method parameters", 2);
  runModeBox_->addItem("Sensitivity analysis of problem parameters", 3);
  runModeBox_->addItem("Multi-objective optimization (Pareto front)", 4);
  runModeBox_->addItem("Meta-dataset pipeline (batch + ELA + ML)", 5);
  runModeBox_->addItem("Neural-network landscape / generalization gap", 6);
  runModeBox_->addItem("Portfolio / dynamic algorithm switching (mid-run)", 7);
  form->addRow("Run mode", runModeBox_);
  // The run modes are now top-level tabs of the main window (see
  // assembleMainTabs()). The combo box is kept, unchanged, as the single source
  // of truth that all the existing mode logic reads -- it is just no longer
  // shown. Selecting a tab drives it; changing it programmatically moves the tab.
  runModeBox_->setVisible(false);
  if (auto* runModeLbl = form->labelForField(runModeBox_)) runModeLbl->setVisible(false);

  // Only relevant to the pipeline mode's stage 2 (landscape sampling) --
  // hidden for every other mode, see updateBatchPanelVisibility(). Lives
  // here (Selection) rather than in the pipeline panel itself (Output) per
  // request, directly below Run mode and above Runs.
  pipelineSamplesSpin_ = new QSpinBox(topBox);
  pipelineSamplesSpin_->setRange(0, 1000000);
  pipelineSamplesSpin_->setSingleStep(50);
  pipelineSamplesSpin_->setSpecialValueText("Auto (100 x dim)");
  pipelineSamplesSpin_->setValue(0);
  pipelineSamplesSpin_->setToolTip("Number of landscape samples per (problem, dimension) pair used by "
                                    "pipeline stage 2 (Landscape sampling).");
  form->addRow("Samples per landscape", pipelineSamplesSpin_);

  form->addRow("Optimization method", methodBox_);
  form->addRow("Problem", problemBox_);
  form->addRow("Dimension", dimSpin_);

  // Read-only runs display — always visible in all modes.
  // The value comes from [global].runs in the settings file.
  // Edit it in Settings → Global tab.
  auto* selectionRunsLbl = new QLabel(topBox);
  selectionRunsLbl->setObjectName("selectionRunsLbl");
  selectionRunsLbl->setToolTip("Number of independent runs per job.\nEdit in Settings → Global → runs.");
  selectionRunsLbl->setTextInteractionFlags(Qt::NoTextInteraction);
  form->addRow("Runs", selectionRunsLbl);

// Batch selection panel (shown only when Run mode is Batch).
batchPanel_ = new QWidget(topBox);
batchPanel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
auto* batchLay = new QVBoxLayout(batchPanel_);
batchLay->setContentsMargins(0, 0, 0, 0);
batchLay->setSpacing(6);

// Top row: Methods (left) + Problems (right) side by side.
auto* batchListsRow = new QHBoxLayout();
batchListsRow->setSpacing(8);

auto* batchMethodsBox = new QGroupBox("Methods", batchPanel_);
auto* batchMethodsLay = new QVBoxLayout(batchMethodsBox);
batchMethodsLay->setContentsMargins(8, 8, 8, 8);
batchMethodsList_ = new QListWidget(batchMethodsBox);
batchMethodsList_->setSelectionMode(QAbstractItemView::ExtendedSelection);
batchMethodsLay->addWidget(batchMethodsList_);

auto refreshBatchMethodSettingsSelector = [this]() {
  if (!settingsBox_) return;

  auto* row = settingsBox_->findChild<QWidget*>("batchMethodSettingsRow");
  auto* combo = settingsBox_->findChild<QComboBox*>("batchMethodSettingsCombo");
  if (!combo) return;

  const bool isBatch = (runModeBox_ && runModeBox_->currentData().toInt() == 1);
  if (row) row->setVisible(isBatch);

  QString preferredShort = currentMethodShort();
  if (batchMethodsList_ && batchMethodsList_->currentItem()) {
    const QString curShort = batchMethodsList_->currentItem()->data(Qt::UserRole).toString().trimmed();
    if (!curShort.isEmpty()) preferredShort = curShort;
  }

  QStringList selectedShorts;
  if (batchMethodsList_) {
    for (int i = 0; i < batchMethodsList_->count(); ++i) {
      if (auto* it = batchMethodsList_->item(i); it && it->isSelected()) {
        const QString shortName = it->data(Qt::UserRole).toString().trimmed();
        if (!shortName.isEmpty() && !selectedShorts.contains(shortName)) {
          selectedShorts << shortName;
        }
      }
    }
  }

  {
    QSignalBlocker blocker(combo);
    combo->clear();
    for (const QString& shortName : selectedShorts) {
      QString display = shortName;
      if (methodBox_) {
        int idx = methodBox_->findData(shortName);
        if (idx < 0) idx = methodBox_->findText(shortName);
        if (idx >= 0) display = methodBox_->itemText(idx);
      }
      combo->addItem(display, shortName);
    }
    combo->setEnabled(isBatch && combo->count() > 0);

    int targetIndex = -1;
    for (int i = 0; i < combo->count(); ++i) {
      if (combo->itemData(i).toString().trimmed() == preferredShort) {
        targetIndex = i;
        break;
      }
    }
    if (targetIndex < 0 && combo->count() > 0) targetIndex = 0;
    if (targetIndex >= 0) combo->setCurrentIndex(targetIndex);
  }

  if (!isBatch || combo->count() <= 0 || !methodBox_) return;

  const QString chosenShort = combo->currentData().toString().trimmed();
  if (chosenShort.isEmpty()) return;

  int methodIndex = methodBox_->findData(chosenShort);
  if (methodIndex < 0) methodIndex = methodBox_->findText(chosenShort);
  if (methodIndex < 0) return;

  if (methodBox_->currentIndex() != methodIndex) methodBox_->setCurrentIndex(methodIndex);
  else onMethodChanged(methodBox_->currentText());
};

connect(batchMethodsList_, &QListWidget::itemSelectionChanged, this, [this, refreshBatchMethodSettingsSelector]() {
  refreshBatchMethodSettingsSelector();
  refreshBatchSelectionView();
  updateStatsTabsEnabled();
});
connect(batchMethodsList_, &QListWidget::currentItemChanged, this,
        [refreshBatchMethodSettingsSelector](QListWidgetItem*, QListWidgetItem*) {
          refreshBatchMethodSettingsSelector();
          // No refreshBatchSelectionView() here: the Output tabs depend only on
          // WHICH methods are selected, and every selection change already
          // refreshes them via itemSelectionChanged above. Refreshing here as
          // well rebuilt all Output tabs twice per click.
        });

batchListsRow->addWidget(batchMethodsBox, 1);

auto* batchProblemsBox = new QGroupBox("Problems", batchPanel_);
auto* batchProblemsLay = new QVBoxLayout(batchProblemsBox);
batchProblemsLay->setContentsMargins(8, 8, 8, 8);
batchProblemsList_ = new QListWidget(batchProblemsBox);
batchProblemsList_->setSelectionMode(QAbstractItemView::ExtendedSelection);
batchProblemsLay->addWidget(batchProblemsList_);

// Batch-only: per-problem dimension overrides for variable-dimension problems.
// Fixed-dimension problems are shown read-only with their fixed dimension.
batchProblemDimsTable_ = new QTableWidget(batchProblemsBox);
batchProblemDimsTable_->setColumnCount(2);
batchProblemDimsTable_->setHorizontalHeaderLabels({"Problem", "Dims"});
batchProblemDimsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
batchProblemDimsTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
batchProblemDimsTable_->setColumnWidth(1, 160);
batchProblemDimsTable_->verticalHeader()->setVisible(false);
batchProblemDimsTable_->setSelectionMode(QAbstractItemView::NoSelection);
batchProblemDimsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
batchProblemDimsTable_->setAlternatingRowColors(true);
batchProblemDimsTable_->setMaximumHeight(140);
batchProblemsLay->addWidget(batchProblemDimsTable_);

// Keep the per-problem dimension UI in sync with selection and default dimension.
connect(batchProblemsList_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
        [this](const QItemSelection&, const QItemSelection&) {
          syncBatchProblemDimsTable();
          refreshBatchSelectionView();
        });
connect(dimSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int newDefaultDim) {
  if (runModeBox_ && runModeBox_->currentData().toInt() == 1) {
    // NOTE: the default dimension changed. Results are stored per dimension
    // ("sphere [d=30]"), so nothing has to be deleted here: the results of
    // the previous default dimension stay cached and are reused if it comes
    // back (the former invalidateBatchProblemCache() call here deleted them,
    // which made "Run batch" re-run finished jobs).
    (void)newDefaultDim;
    syncBatchProblemDimsTable();
    refreshBatchSelectionView();
  }
});

batchListsRow->addWidget(batchProblemsBox, 1);
batchLay->addLayout(batchListsRow, 1);

auto* batchSettingsBox = new QGroupBox("Batch settings", batchPanel_);
batchSettingsBox->setObjectName("batchSettingsBox"); // hidden for the pipeline mode; see updateBatchPanelVisibility()
auto* batchSettingsLay = new QFormLayout(batchSettingsBox);
batchSettingsLay->setContentsMargins(8, 8, 8, 8);

// Runs are shown as read-only info in the Selection area (selectionRunsLbl).
// No spinner needed here.

// Metric and Aggregate combos are created inside the Wilcoxon / Friedman
// controls panels of the Batch Summary tab (see createOutputRunTab).
// batchMetricCombo_ and batchAggCombo_ are bound there via bindBatchSummaryUiForPage.

auto* batchShowMeanChk = new QCheckBox("Mean", batchSettingsBox);
batchShowMeanChk->setObjectName("batchShowMeanChk");
batchShowMeanChk->setChecked(true);
batchShowRateChk_ = new QCheckBox("Rate (%)", batchSettingsBox);
batchShowRateChk_->setChecked(true);
batchShowSdChk_ = new QCheckBox("SD", batchSettingsBox);
batchShowSdChk_->setChecked(true);
batchShowTimeChk_ = new QCheckBox("Time (s)", batchSettingsBox);
batchShowTimeChk_->setChecked(true);

auto* colOpts = new QWidget(batchSettingsBox);
auto* colOptsLay = new QHBoxLayout(colOpts);
colOptsLay->setContentsMargins(0, 0, 0, 0);
colOptsLay->addWidget(batchShowMeanChk);
colOptsLay->addWidget(batchShowRateChk_);
colOptsLay->addWidget(batchShowSdChk_);
colOptsLay->addWidget(batchShowTimeChk_);
colOptsLay->addStretch(1);

batchSettingsLay->addRow("Columns", colOpts);

auto* batchStatsNoteLbl = new QLabel("Value/Mean/Rate/SD are computed from the convergence CSV of each batch job.", batchSettingsBox);
batchStatsNoteLbl->setWordWrap(true);
batchSettingsLay->addRow(batchStatsNoteLbl);

  // Status label moved to output area.

  // Progress bar removed from batch settings — the output area already has one.

batchLay->addWidget(batchSettingsBox, 0);

// batchPanel_ goes into the outer expanding layout (not the form) so it
// fills all available vertical space when the Selection area is maximized.
topBoxOuterLay->addWidget(batchPanel_, 1);

// ── Multi-objective panel ──────────────────────────────────────────────
// Shown only in Multi-objective run mode. The dropdowns are filtered to
// only the problems/methods that actually expose >=2 objectives, queried
// directly from the (already-linked) optimcore library -- no changes to
// makeProblem()/makeMethod() or any single-objective list are involved.
moPanel_ = new QGroupBox("Multi-objective settings", topBox);
moPanel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
{
  auto* moLay = new QFormLayout(moPanel_);
  moLay->setContentsMargins(8, 8, 8, 8);

  moProblemBox_ = new QComboBox(moPanel_);
  moMethodBox_  = new QComboBox(moPanel_);
  for (const auto& name : optimsolution::listMultiObjectiveProblemNames()) {
    moProblemBox_->addItem(QString::fromStdString(name));
  }
  for (const auto& name : optimsolution::listMultiObjectiveMethodNames()) {
    moMethodBox_->addItem(QString::fromStdString(name));
  }
  if (moProblemBox_->count() == 0) {
    moProblemBox_->addItem("(none yet)");
    moProblemBox_->setEnabled(false);
  }

  moDimSpin_ = new QSpinBox(moPanel_);
  moDimSpin_->setRange(2, 200);
  moDimSpin_->setValue(2);

  moLay->addRow("Method", moMethodBox_);
  moLay->addRow("Problem", moProblemBox_);
  moLay->addRow("Dimension", moDimSpin_);

  auto* moHint = new QLabel(
    "Only problems/methods with >=2 objectives appear here. Population and generations "
    "are read from optimsolution.cfg (per-method section, e.g. [nsga2]), not set here. "
    "Running produces its own \"Pareto front\" output tab (scatter of objective 1 vs "
    "objective 2 + a results table); Single/Batch/Sensitivity modes are unaffected.",
    moPanel_);
  moHint->setWordWrap(true);
  moHint->setStyleSheet("opacity: 0.75;");
  moLay->addRow(moHint);
}
topBoxOuterLay->addWidget(moPanel_, 1);

  // ── Neural-network landscape panel (run mode 7) ─────────────────────────
  // Trains a small feed-forward network as an optimsolution::Problem
  // (NNProblem) and samples the training-loss landscape locally around the
  // found minimum -- see src/problems/nn_problem.h and
  // src/mode_nnlandscape.cpp ("--nnlandscape" CLI mode) for the underlying
  // mechanism. Batch-capable, same spirit as Batch run's own Methods/Problems
  // lists: multi-select Datasets and Optimizers, ';'-separated Architectures,
  // and "Runs" independent seeds per combination (seed_base read from
  // optimsolution.cfg at run time -- no per-run seed field here, matching
  // every other mode's convention of NOT exposing seed directly in the UI).
  nnLandscapePanel_ = new QGroupBox("Neural-network landscape settings", topBox);
  nnLandscapePanel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  {
    auto* nnOuterLay = new QVBoxLayout(nnLandscapePanel_);
    nnOuterLay->setContentsMargins(8, 8, 8, 8);
    nnOuterLay->setSpacing(6);

    // Datasets + Optimizers side by side, same layout convention as
    // Batch run's Methods/Problems lists.
    auto* nnListsRow = new QHBoxLayout();
    nnListsRow->setSpacing(8);

    auto* nnDatasetsBox = new QGroupBox("Datasets (.train)", nnLandscapePanel_);
    auto* nnDatasetsLay = new QVBoxLayout(nnDatasetsBox);
    nnDatasetsLay->setContentsMargins(8, 8, 8, 8);
    nnDatasetsList_ = new QListWidget(nnDatasetsBox);
    nnDatasetsList_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    nnDatasetsList_->setToolTip("Select one or more datasets (Ctrl/Shift-click). "
                                 "Populated from datasetsDir(); use Browse... to add one from elsewhere.");
    nnDatasetsLay->addWidget(nnDatasetsList_, 1);
    nnDatasetBrowseBtn_ = new QPushButton("Browse...", nnDatasetsBox);
    connect(nnDatasetBrowseBtn_, &QPushButton::clicked, this, &MainWindow::onNNLandscapeBrowseDataset);
    nnDatasetsLay->addWidget(nnDatasetBrowseBtn_, 0);
    nnListsRow->addWidget(nnDatasetsBox, 1);

    auto* nnOptimizersBox = new QGroupBox("Optimizers", nnLandscapePanel_);
    auto* nnOptimizersLay = new QVBoxLayout(nnOptimizersBox);
    nnOptimizersLay->setContentsMargins(8, 8, 8, 8);
    nnOptimizersList_ = new QListWidget(nnOptimizersBox);
    nnOptimizersList_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    nnOptimizersList_->setToolTip("Every method the framework knows (same roster as Single run/Batch run). "
                                   "For the local methods gd/lbfgs/bfgs/nm/cg/ncg/adam, Iterations is their "
                                   "iteration limit and Population is ignored; their other settings come "
                                   "from optimsolution.cfg's [gd]/[lbfgs]/... sections.");
    nnOptimizersLay->addWidget(nnOptimizersList_, 1);
    nnListsRow->addWidget(nnOptimizersBox, 1);

    nnOuterLay->addLayout(nnListsRow, 1);

    auto* formContainerNN = new QWidget(nnLandscapePanel_);
    auto* nnLay = new QFormLayout(formContainerNN);
    nnLay->setContentsMargins(0, 4, 0, 0);

    nnHiddenEdit_ = new QLineEdit("8", formContainerNN);
    nnHiddenEdit_->setToolTip("';'-separated architectures, each comma-separated hidden-layer sizes. "
                               "E.g. \"16;8,8\" runs both a single 16-unit hidden layer AND a two-layer "
                               "8+8 network, for every selected dataset x optimizer x seed.");
    nnLay->addRow("Hidden layers", nnHiddenEdit_);

    nnActivationBox_ = new QComboBox(formContainerNN);
    nnActivationBox_->addItems({"relu", "tanh", "sigmoid"});
    nnLay->addRow("Activation", nnActivationBox_);

    nnIterationsSpin_ = new QSpinBox(formContainerNN);
    nnIterationsSpin_->setRange(1, 1000000);
    nnIterationsSpin_->setValue(200);
    nnIterationsSpin_->setToolTip("Iterations of every optimizer; for the local methods "
                                   "gd/lbfgs/bfgs/nm/cg/ncg/adam, their iteration limit.");
    nnLay->addRow("Iterations", nnIterationsSpin_);

    nnPopulationSpin_ = new QSpinBox(formContainerNN);
    nnPopulationSpin_->setRange(4, 100000);
    nnPopulationSpin_->setValue(40);
    nnLay->addRow("Population", nnPopulationSpin_);

    nnWBoundSpin_ = new QDoubleSpinBox(formContainerNN);
    nnWBoundSpin_->setRange(0.1, 1000.0);
    nnWBoundSpin_->setDecimals(2);
    nnWBoundSpin_->setValue(5.0);
    nnWBoundSpin_->setToolTip("Half-width of the symmetric [-w,w]^d weight box the network is trained in.");
    nnLay->addRow("Weight bound", nnWBoundSpin_);

    nnRadiiEdit_ = new QLineEdit("0.02,0.05,0.1", formContainerNN);
    nnRadiiEdit_->setToolTip("Local-sampling radius fractions of the weight box, comma-separated. "
                              "Must stay the same across every run appended to the same "
                              "nn_summary.csv -- the CLI itself rejects a mismatch rather than "
                              "silently misaligning columns.");
    nnLay->addRow("Radii", nnRadiiEdit_);

    nnLocalSamplesSpin_ = new QSpinBox(formContainerNN);
    nnLocalSamplesSpin_->setRange(0, 1000000);
    nnLocalSamplesSpin_->setSingleStep(50);
    nnLocalSamplesSpin_->setSpecialValueText("Auto (30 x dim)");
    nnLocalSamplesSpin_->setValue(0);
    nnLay->addRow("Local samples/radius", nnLocalSamplesSpin_);

    nnLabelEdit_ = new QLineEdit(formContainerNN);
    nnLabelEdit_->setPlaceholderText("optional run label");
    nnLay->addRow("Label", nnLabelEdit_);

    auto* nnSeedInfoLbl = new QLabel(formContainerNN);
    nnSeedInfoLbl->setObjectName("nnSeedInfoLbl");
    nnSeedInfoLbl->setStyleSheet("opacity: 0.75;");
    nnSeedInfoLbl->setWordWrap(true);
    nnSeedInfoLbl->setToolTip("Seeds are NOT set here: each combination is repeated \"Runs\" times "
                               "(Settings -> Global -> runs) with seed = seed_base + run index "
                               "(Settings -> Global -> seed_base), exactly like every other run mode.");
    nnLay->addRow("Seeds", nnSeedInfoLbl);

    nnOuterLay->addWidget(formContainerNN, 0);
  }
  topBoxOuterLay->addWidget(nnLandscapePanel_, 1);

  // NOTE: the "Meta-dataset pipeline" panel (4 action buttons) now lives in
  // the OUTPUT area instead of here -- see outLay, right after outTop --
  // since these are actions that produce output, not selection controls.
  // The method/problem selection lists it uses (batchPanel_, above) still
  // live here in Selection, same as Batch run.

  // ── Sensitivity problem panel ──────────────────────────────────────────
  // Shown only in Sensitivity run mode. Lets the user pick multiple
  // problems (and per-problem dimensions) while the method stays fixed.
  sensProblemPanel_ = new QWidget(topBox);
  sensProblemPanel_->setVisible(false);
  auto* sensProbLay = new QHBoxLayout(sensProblemPanel_);
  sensProbLay->setContentsMargins(0, 0, 0, 0);
  sensProbLay->setSpacing(8);

  auto* sensProblemsBox = new QGroupBox("Problems", sensProblemPanel_);
  auto* sensProblemsLay = new QVBoxLayout(sensProblemsBox);
  sensProblemsLay->setContentsMargins(8, 8, 8, 8);
  sensProblemsList_ = new QListWidget(sensProblemsBox);
  sensProblemsList_->setSelectionMode(QAbstractItemView::ExtendedSelection);
  sensProblemsList_->setToolTip("Select one or more problems to sweep. The sensitivity\n"
                                 "analysis will run once per selected problem/dimension.");
  sensProblemsLay->addWidget(sensProblemsList_);

  sensProblemDimsTable_ = new QTableWidget(sensProblemsBox);
  sensProblemDimsTable_->setColumnCount(2);
  sensProblemDimsTable_->setHorizontalHeaderLabels({"Problem", "Dims"});
  sensProblemDimsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
  sensProblemDimsTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
  sensProblemDimsTable_->setColumnWidth(1, 160);
  sensProblemDimsTable_->verticalHeader()->setVisible(false);
  sensProblemDimsTable_->setSelectionMode(QAbstractItemView::NoSelection);
  sensProblemDimsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  sensProblemDimsTable_->setAlternatingRowColors(true);
  sensProblemDimsTable_->setMaximumHeight(140);
  sensProblemsLay->addWidget(sensProblemDimsTable_);

  // Sync dims table when selection changes — reuse the batch helper.
  connect(sensProblemsList_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
          [this](const QItemSelection&, const QItemSelection&) {
            syncSensProblemDimsTable();
          });

  sensProbLay->addWidget(sensProblemsBox, 1);
  form->addRow(sensProblemPanel_);

  // btnRow was constructed and added to form above (before Run mode).
  auto saveCurrentBatchSelection = [this]() -> bool {
    if (!batchMethodsList_ || !batchProblemsList_) {
      return false;
    }

    const QStringList methods = selectedBatchMethodShortNames();
    const QStringList problemKeys = selectedBatchProblemShortNames();
    QStringList problems;
    for (const QString& problemKey : problemKeys) {
      const QString baseProblem = batchBaseProblemShort(problemKey);
      if (!baseProblem.isEmpty() && !problems.contains(baseProblem)) {
        problems << baseProblem;
      }
    }
    if (methods.isEmpty() && problems.isEmpty()) {
      QMessageBox::warning(this, "Save batch selection", "There are no selected batch methods or problems to save.");
      return false;
    }

    QString suggestedDir = !projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath();
    if (!lastRuntimeWorkingDir_.trimmed().isEmpty()) {
      suggestedDir = lastRuntimeWorkingDir_.trimmed();
    }
    const QString suggestedPath = QDir(suggestedDir).filePath("batch_selection.batchsel");

    const QString path = QFileDialog::getSaveFileName(
      this,
      "Save batch selection",
      suggestedPath,
      "OptimSolution batch selection (*.batchsel);;All files (*.*)"
    );
    if (path.isEmpty()) return false;

    QSaveFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
      QMessageBox::warning(this, "Save batch selection", QString("Cannot write file:\n%1").arg(path));
      return false;
    }

    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);
    ts << "# OptimSolution batch selection\n";
    ts << "version=1\n";
    ts << "methods=" << methods.join(",") << "\n";
    ts << "problems=" << problems.join(",") << "\n";
    ts << "default_dim=" << (dimSpin_ ? dimSpin_->value() : 2) << "\n";

    const int defaultDim = dimSpin_ ? std::max(2, dimSpin_->value()) : 2;
    for (const QString& prob : problems) {
      const QString p = prob.trimmed();
      if (p.isEmpty()) continue;
      const int fixed = fixedDimForProblem(p);
      if (fixed > 0) continue;

      const QVector<int> dims = batchDimsForProblem(p, defaultDim);
      if (dims.isEmpty()) continue;

      if (dims.size() == 1) {
        ts << "problem_dim." << p << '=' << dims.front() << "\n";
      } else {
        ts << "problem_dims." << p << '=' << batchDimensionListToText(dims) << "\n";
      }
    }

    if (!f.commit()) {
      QMessageBox::warning(this, "Save batch selection", QString("Failed to finalize file:\n%1").arg(path));
      return false;
    }

    statusBar()->showMessage(QString("Batch selection saved to %1").arg(QFileInfo(path).fileName()), 5000);
    appendLog(QString("Saved batch selection: %1").arg(path));
    return true;
  };

  auto loadSavedBatchSelection = [this]() -> bool {
    if (!batchMethodsList_ || !batchProblemsList_) {
      return false;
    }

    QString startDir = !projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath();
    if (!lastRuntimeWorkingDir_.trimmed().isEmpty()) {
      startDir = lastRuntimeWorkingDir_.trimmed();
    }

    const QString path = QFileDialog::getOpenFileName(
      this,
      "Load batch selection",
      startDir,
      "OptimSolution batch selection (*.batchsel);;All files (*.*)"
    );
    if (path.isEmpty()) return false;

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QMessageBox::warning(this, "Load batch selection", QString("Cannot open file:\n%1").arg(path));
      return false;
    }

    QStringList methods;
    QStringList problems;
    QMap<QString, QStringList> loadedDims;
    int loadedDefaultDim = -1;

    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);
    while (!ts.atEnd()) {
      const QString rawLine = ts.readLine().trimmed();
      if (rawLine.isEmpty() || rawLine.startsWith('#')) continue;
      const int eq = rawLine.indexOf('=');
      if (eq <= 0) continue;

      const QString key = rawLine.left(eq).trimmed();
      const QString value = rawLine.mid(eq + 1).trimmed();
      if (key.compare("methods", Qt::CaseInsensitive) == 0) {
        const QStringList parts = value.split(',', Qt::SkipEmptyParts);
        for (const QString& part : parts) {
          const QString v = part.trimmed();
          if (!v.isEmpty() && !methods.contains(v)) methods << v;
        }
      } else if (key.compare("problems", Qt::CaseInsensitive) == 0) {
        const QStringList parts = value.split(',', Qt::SkipEmptyParts);
        for (const QString& part : parts) {
          const QString v = part.trimmed();
          if (!v.isEmpty() && !problems.contains(v)) problems << v;
        }
      } else if (key.compare("default_dim", Qt::CaseInsensitive) == 0) {
        bool ok = false;
        const int v = value.toInt(&ok);
        if (ok && v > 1) loadedDefaultDim = v;
      } else if (key.startsWith("problem_dim.", Qt::CaseInsensitive)) {
        const QString prob = key.mid(QString("problem_dim.").size()).trimmed();
        bool ok = false;
        const int v = value.toInt(&ok);
        if (!prob.isEmpty() && ok && v > 1) {
          loadedDims.insert(prob, QStringList{QString::number(v)});
        }
      } else if (key.startsWith("problem_dims.", Qt::CaseInsensitive)) {
        const QString prob = key.mid(QString("problem_dims.").size()).trimmed();
        if (!prob.isEmpty()) {
          const QVector<int> dims = parseBatchDimensionList(value, (loadedDefaultDim > 1 ? loadedDefaultDim : 2), 2,
                                                            dimSpin_ ? dimSpin_->maximum() : std::numeric_limits<int>::max());
          QStringList parts;
          for (int d : dims) parts << QString::number(d);
          if (!parts.isEmpty()) loadedDims.insert(prob, parts);
        }
      }
    }

    if (methods.isEmpty() && problems.isEmpty()) {
      QMessageBox::warning(this, "Load batch selection", "The selected file does not contain any batch methods or problems.");
      return false;
    }

    if (runModeBox_) {
      // Only force-switch to Batch run if we're somewhere that doesn't
      // already show the batch selection lists. Both Batch run (1) and the
      // unified pipeline mode (5) use these same lists -- loading a
      // selection while already in either one should just populate the
      // lists in place, not yank the user out of the pipeline mode.
      const int currentMode = runModeBox_->currentData().toInt();
      const bool alreadyUsesBatchLists = (currentMode == 1 || currentMode == 5);
      if (!alreadyUsesBatchLists) {
        const int batchIndex = runModeBox_->findData(1);
        if (batchIndex >= 0) runModeBox_->setCurrentIndex(batchIndex);
      }
    }

    populateBatchLists();


    if (batchMethodsList_) {
      batchMethodsList_->clearSelection();
      QSet<QString> wanted;
      for (const QString& m : methods) wanted.insert(m);
      for (int i = 0; i < batchMethodsList_->count(); ++i) {
        if (auto* it = batchMethodsList_->item(i)) {
          QString key = it->data(Qt::UserRole).toString().trimmed();
          if (key.contains(" - ")) key = key.section(" - ", 0, 0).trimmed();
          if (key.isEmpty()) key = it->text().section(" - ", 0, 0).trimmed();
          it->setSelected(wanted.contains(key));
        }
      }
      for (int i = 0; i < batchMethodsList_->count(); ++i) {
        if (auto* it = batchMethodsList_->item(i); it && it->isSelected()) {
          batchMethodsList_->setCurrentItem(it);
          break;
        }
      }
    }

    if (batchProblemsList_) {
      batchProblemsList_->clearSelection();
      QSet<QString> wanted;
      for (const QString& p : problems) wanted.insert(batchBaseProblemShort(p));
      for (int i = 0; i < batchProblemsList_->count(); ++i) {
        if (auto* it = batchProblemsList_->item(i)) {
          QString key = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
          if (key.isEmpty()) key = batchBaseProblemShort(it->text().trimmed());
          it->setSelected(wanted.contains(key));
        }
      }
    }

    if (dimSpin_ && loadedDefaultDim > 1) {
      dimSpin_->setValue(loadedDefaultDim);
    }

    batchProblemDimOverride_.clear();
    for (auto it = loadedDims.begin(); it != loadedDims.end(); ++it) {
      if (fixedDimForProblem(it.key()) <= 0 && !it.value().isEmpty()) {
        batchProblemDimOverride_.insert(it.key(), it.value());
      }
    }
    syncBatchProblemDimsTable();

    if (settingsBox_) {
      if (auto* combo = settingsBox_->findChild<QComboBox*>("batchMethodSettingsCombo")) {
        combo->setEnabled((runModeBox_ && runModeBox_->currentData().toInt() == 1) && combo->count() > 0);
        if (combo->count() > 0) {
          QString preferredShort;
          if (batchMethodsList_ && batchMethodsList_->currentItem()) {
            preferredShort = batchMethodsList_->currentItem()->data(Qt::UserRole).toString().trimmed();
          }
          int targetIndex = -1;
          for (int i = 0; i < combo->count(); ++i) {
            if (combo->itemData(i).toString().trimmed() == preferredShort) {
              targetIndex = i;
              break;
            }
          }
          if (targetIndex >= 0) combo->setCurrentIndex(targetIndex);
        }
      }
    }

    appendLog(QString("Loaded batch selection: %1").arg(path));
    statusBar()->showMessage(QString("Batch selection loaded from %1").arg(QFileInfo(path).fileName()), 5000);
    refreshBatchSelectionView();
    return true;
  };

  connect(saveBatchSelectionBtn_, &QPushButton::clicked, this, [saveCurrentBatchSelection]() {
    saveCurrentBatchSelection();
  });
  connect(loadBatchSelectionBtn_, &QPushButton::clicked, this, [loadSavedBatchSelection]() {
    loadSavedBatchSelection();
  });


  // Settings controls (no file/path displayed)
  settingsBox_ = new QGroupBox("Settings", central);
  settingsBox_->setObjectName("settingsBox");
  settingsBox_->setProperty("focused", false);
  auto* cfgBox = settingsBox_;
  auto* cfgLay = new QVBoxLayout(cfgBox);
  auto* cfgBtnRow = new QHBoxLayout();
  selectCfgBtn_ = new QPushButton("Select settings...", cfgBox);
  selectCfgBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
  selectCfgBtn_->setIconSize(QSize(18, 18));
  reloadCfgBtn_ = new QPushButton("Reload", cfgBox);
  reloadCfgBtn_->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
  reloadCfgBtn_->setIconSize(QSize(18, 18));
  saveCfgBtn_   = new QPushButton("Save", cfgBox);
  saveCfgBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  saveCfgBtn_->setIconSize(QSize(18, 18));
  cfgBtnRow->addWidget(selectCfgBtn_);
  cfgBtnRow->addStretch(1);
  cfgBtnRow->addWidget(reloadCfgBtn_);
  cfgBtnRow->addWidget(saveCfgBtn_);
  // (The Settings area is now a top-level tab of its own, so it no longer
  // needs a "Maximize" button.)
  cfgLay->addLayout(cfgBtnRow);

  auto* batchMethodSettingsRow = new QWidget(cfgBox);
  batchMethodSettingsRow->setObjectName("batchMethodSettingsRow");
  auto* batchMethodSettingsLay = new QHBoxLayout(batchMethodSettingsRow);
  batchMethodSettingsLay->setContentsMargins(0, 0, 0, 0);
  auto* batchMethodSettingsLbl = new QLabel("Edit settings for batch method", batchMethodSettingsRow);
  batchMethodSettingsLbl->setObjectName("batchMethodSettingsLabel");
  auto* batchMethodSettingsCombo = new QComboBox(batchMethodSettingsRow);
  batchMethodSettingsCombo->setObjectName("batchMethodSettingsCombo");
  batchMethodSettingsCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
  batchMethodSettingsCombo->setEnabled(false);
  batchMethodSettingsLay->addWidget(batchMethodSettingsLbl);
  batchMethodSettingsLay->addWidget(batchMethodSettingsCombo, 1);
  batchMethodSettingsRow->setVisible(false);
  cfgLay->addWidget(batchMethodSettingsRow);

  connect(batchMethodSettingsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this, batchMethodSettingsCombo](int index) {
            if (index < 0 || !methodBox_) return;

            const QString shortName = batchMethodSettingsCombo->itemData(index).toString().trimmed();
            if (shortName.isEmpty()) return;

            int methodIndex = methodBox_->findData(shortName);
            if (methodIndex < 0) methodIndex = methodBox_->findText(shortName);
            if (methodIndex >= 0) {
              if (methodBox_->currentIndex() != methodIndex) methodBox_->setCurrentIndex(methodIndex);
              else onMethodChanged(methodBox_->currentText());
            }

            if (batchMethodsList_) {
              for (int i = 0; i < batchMethodsList_->count(); ++i) {
                if (auto* it = batchMethodsList_->item(i)) {
                  if (it->data(Qt::UserRole).toString().trimmed() == shortName) {
                    QSignalBlocker blocker(batchMethodsList_);
                    batchMethodsList_->setCurrentItem(it, QItemSelectionModel::NoUpdate);
                    break;
                  }
                }
              }
            }
          });

  tabs_ = new QTabWidget(cfgBox);
  runTable_ = new QTableWidget(tabs_);
  stopTable_ = new QTableWidget(tabs_);
  initTable_ = new QTableWidget(tabs_);
  methodTable_ = new QTableWidget(tabs_);
  sensitivityTable_ = new QTableWidget(tabs_);

  /* Global tab wrapper: allows adding/removing/renaming arbitrary global parameters. */
  auto* globalTab = new QWidget(tabs_);
  auto* globalLay = new QVBoxLayout(globalTab);
  globalLay->setContentsMargins(0, 0, 0, 0);
  auto* globalBtnRow = new QHBoxLayout();
  addGlobalParamBtn_ = new QPushButton("Add parameter", globalTab);
  addGlobalParamBtn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  addGlobalParamBtn_->setIconSize(QSize(18, 18));
  removeGlobalParamBtn_ = new QPushButton("Remove parameter", globalTab);
  removeGlobalParamBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  removeGlobalParamBtn_->setIconSize(QSize(18, 18));
  globalBtnRow->addWidget(addGlobalParamBtn_);
  globalBtnRow->addWidget(removeGlobalParamBtn_);
  globalBtnRow->addStretch(1);
  globalLay->addLayout(globalBtnRow);
  globalLay->addWidget(runTable_);

  /* Termination rule tab wrapper: allows adding/removing/renaming arbitrary stop parameters. */
  auto* stopTab = new QWidget(tabs_);
  auto* stopLay = new QVBoxLayout(stopTab);
  stopLay->setContentsMargins(0, 0, 0, 0);
  auto* stopBtnRow = new QHBoxLayout();
  addStopParamBtn_ = new QPushButton("Add parameter", stopTab);
  addStopParamBtn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  addStopParamBtn_->setIconSize(QSize(18, 18));
  removeStopParamBtn_ = new QPushButton("Remove parameter", stopTab);
  removeStopParamBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  removeStopParamBtn_->setIconSize(QSize(18, 18));
  stopBtnRow->addWidget(addStopParamBtn_);
  stopBtnRow->addWidget(removeStopParamBtn_);
  stopBtnRow->addStretch(1);
  stopLay->addLayout(stopBtnRow);
  stopLay->addWidget(stopTable_);

  /* Initialization tab wrapper: allows adding/removing/renaming arbitrary init parameters. */
  auto* initTab = new QWidget(tabs_);
  auto* initLay = new QVBoxLayout(initTab);
  initLay->setContentsMargins(0, 0, 0, 0);
  auto* initBtnRow = new QHBoxLayout();
  addInitParamBtn_ = new QPushButton("Add parameter", initTab);
  addInitParamBtn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  addInitParamBtn_->setIconSize(QSize(18, 18));
  removeInitParamBtn_ = new QPushButton("Remove parameter", initTab);
  removeInitParamBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  removeInitParamBtn_->setIconSize(QSize(18, 18));
  initBtnRow->addWidget(addInitParamBtn_);
  initBtnRow->addWidget(removeInitParamBtn_);
  initBtnRow->addStretch(1);
  initLay->addLayout(initBtnRow);
  initLay->addWidget(initTable_);

  /* Method tab wrapper: allows adding/removing arbitrary method parameters (names/count vary per method). */
  auto* methodTab = new QWidget(tabs_);
  auto* methodLay = new QVBoxLayout(methodTab);
  methodLay->setContentsMargins(0, 0, 0, 0);
  auto* methodBtnRow = new QHBoxLayout();
  addMethodParamBtn_ = new QPushButton("Add parameter", methodTab);
  addMethodParamBtn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  addMethodParamBtn_->setIconSize(QSize(18, 18));
  removeMethodParamBtn_ = new QPushButton("Remove parameter", methodTab);
  removeMethodParamBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  removeMethodParamBtn_->setIconSize(QSize(18, 18));
  methodBtnRow->addWidget(addMethodParamBtn_);
  methodBtnRow->addWidget(removeMethodParamBtn_);
  methodBtnRow->addStretch(1);
  methodLay->addLayout(methodBtnRow);
  methodLay->addWidget(methodTable_);

  /* Sensitivity tab: parameter sweep for the selected method (excluding global and local-search keys). */
  auto* sensitivityTab = new QWidget(tabs_);
  auto* sensitivityLay = new QVBoxLayout(sensitivityTab);
  sensitivityLay->setContentsMargins(0, 0, 0, 0);

  sensitivityEnableChk_ = new QCheckBox("Enable sensitivity analysis", sensitivityTab);
  sensitivityEnableChk_->setChecked(false);
  connect(sensitivityEnableChk_, &QCheckBox::toggled, this, &MainWindow::onSensitivityEnableToggled);
  sensitivityLay->addWidget(sensitivityEnableChk_);

  // Table columns: Parameter | Analyze | Values (comma-separated list)
  sensitivityTable_->setColumnCount(3);
  sensitivityTable_->setHorizontalHeaderLabels({"Parameter", "Analyze", "Values"});
  sensitivityTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  sensitivityTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
  sensitivityTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
  sensitivityTable_->verticalHeader()->setVisible(false);
  sensitivityTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
  sensitivityTable_->setSelectionMode(QAbstractItemView::SingleSelection);
  sensitivityLay->addWidget(sensitivityTable_);

  setupTable(runTable_, {"Key", "Value", "Info", "Source"});
  setupTable(stopTable_, {"Key", "Value", "Info"});
  setupTable(initTable_, {"Key", "Value", "Info"});
  setupTable(methodTable_, {"Key", "Value"});

  // Settings value editors (visible for all parameters) - only for the Settings tabs.
  runTable_->setItemDelegateForColumn(1, new SettingsValueDelegate(runTable_));
  stopTable_->setItemDelegateForColumn(1, new SettingsValueDelegate(stopTable_));
  initTable_->setItemDelegateForColumn(1, new SettingsValueDelegate(initTable_));

  tabs_->addTab(globalTab, "Global");
  tabs_->addTab(stopTab, "Termination rule");
  tabs_->addTab(initTab, "Initialization");
  tabs_->addTab(methodTab, "Optimization method");
  tabs_->addTab(sensitivityTab, "Sensitivity");

  cfgLay->addWidget(tabs_);
  // Selection and Settings used to sit side by side here. They are now placed
  // by assembleMainTabs(): Selection (+ Output) inside each run-mode tab,
  // Settings in its own tab.

  outputBox_ = new QGroupBox("Output", central);
  outputBox_->setObjectName("outputBox");
  outputBox_->setProperty("focused", false);
  auto* outBox = outputBox_;
  auto* outLay = new QVBoxLayout(outBox);
  // Global output controls (apply to next run, and to the currently selected output tab when reloading).
  auto* outTop = new WrapToolbarLayout();   // wraps instead of forcing a ~900 px minimum width
  forceConvergenceCsvChk_ = nullptr;
  loadExperimentCsvBtn_ = new QPushButton("Load experiment CSV...", outBox);
  loadExperimentCsvBtn_->setIcon(selectCfgBtn_ ? selectCfgBtn_->icon() : style()->standardIcon(QStyle::SP_DialogOpenButton));
  loadExperimentCsvBtn_->setIconSize(QSize(18, 18));
  loadExperimentCsvBtn_->setToolTip("Load a previously saved experiment CSV. In Batch mode, if the selected folder contains batch summary CSV files, the whole batch is reconstructed. In either Sensitivity mode, if the folder contains sensitivity_*.csv files, one tab per problem is reconstructed.");
  reloadConvergenceBtn_ = new QPushButton("Refresh plots", outBox);
  reloadConvergenceBtn_->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
  reloadConvergenceBtn_->setIconSize(QSize(18, 18));
  clearCsvBtn_ = new QPushButton("Delete output files", outBox);
  clearCsvBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  clearCsvBtn_->setIconSize(QSize(18, 18));
  clearCsvBtn_->setToolTip("Delete EVERYTHING (all files and subfolders) under the optimsolution_gui_run runtime working folder(s) -- current and previous runs.");
  openOutputFolderBtn_ = new QPushButton("Open output folder", outBox);
  openOutputFolderBtn_->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
  openOutputFolderBtn_->setIconSize(QSize(18, 18));
  openOutputFolderBtn_->setToolTip("Open the optimsolution_gui_run runtime folder in the system file browser.");
  connect(openOutputFolderBtn_, &QPushButton::clicked, this, [this]() {
    QString baseWd = QCoreApplication::applicationDirPath();
    if (!settingsPath_.isEmpty()) {
      QFileInfo fi(settingsPath_);
      if (fi.exists()) baseWd = fi.absolutePath();
    }
    const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
    QDir().mkpath(runtimeWd);
    QDesktopServices::openUrl(QUrl::fromLocalFile(runtimeWd));
  });

  outTop->addWidget(runBtn_);
  outTop->addWidget(busySpinner_);
  outTop->addSpacing(8);
  outTop->addStretch(1);
  outTop->addWidget(loadExperimentCsvBtn_);
  outTop->addWidget(reloadConvergenceBtn_);
  outTop->addWidget(clearCsvBtn_);
  pipelineLoadExperimentBtn_ = new QPushButton("Load previous experiment...", outBox);
  pipelineLoadExperimentBtn_->setObjectName("pipelineLoadExperimentBtn");
  pipelineLoadExperimentBtn_->setIcon(selectCfgBtn_ ? selectCfgBtn_->icon() : style()->standardIcon(QStyle::SP_DialogOpenButton));
  pipelineLoadExperimentBtn_->setIconSize(QSize(18, 18));
  pipelineLoadExperimentBtn_->setToolTip(
    "Load an earlier batch_master.csv and restore its method/problem/dimension "
    "selection in the Selection area. Then check more items to add them, or "
    "uncheck items to remove them, before running \"1. Batch + ranking\" again -- "
    "only the newly added combinations actually run, and removed ones are "
    "dropped from the results. (Batch run's own CSV loader can only ever add, "
    "never remove.)");
  connect(pipelineLoadExperimentBtn_, &QPushButton::clicked, this, &MainWindow::onLoadPipelineExperiment);
  outTop->addWidget(pipelineLoadExperimentBtn_);
  pipelineLoadExperimentSpinner_ = new BusySpinner(outBox);
  pipelineLoadExperimentSpinner_->setFixedSize(18, 18);
  pipelineLoadExperimentSpinner_->stop();
  outTop->addWidget(pipelineLoadExperimentSpinner_);

  // Mode 7's own "Load previous experiment..." (replaces "Refresh plots" for
  // this mode, same as the pipeline mode's own button does) -- points the
  // Results/Mean/Wilcoxon/Rankings/LODO/Feature-Importances tabs at an
  // earlier run's nn_summary.csv instead of the default runtime folder.
  nnLoadExperimentBtn_ = new QPushButton("Load previous experiment...", outBox);
  nnLoadExperimentBtn_->setObjectName("nnLoadExperimentBtn");
  nnLoadExperimentBtn_->setIcon(selectCfgBtn_ ? selectCfgBtn_->icon() : style()->standardIcon(QStyle::SP_DialogOpenButton));
  nnLoadExperimentBtn_->setIconSize(QSize(18, 18));
  nnLoadExperimentBtn_->setToolTip(
    "Load an earlier nn_summary.csv (and its nn_analysis/ folder, if present) and display it in the "
    "Results/Mean/Wilcoxon/Rankings/LODO Results/Feature Importances tabs. Does not affect where a new "
    "\"1. Train + sample landscape\" run writes to.");
  connect(nnLoadExperimentBtn_, &QPushButton::clicked, this, &MainWindow::onRunNNLandscapeLoadExperiment);
  outTop->addWidget(nnLoadExperimentBtn_);
  nnLoadExperimentSpinner_ = new BusySpinner(outBox);
  nnLoadExperimentSpinner_->setFixedSize(18, 18);
  nnLoadExperimentSpinner_->stop();
  outTop->addWidget(nnLoadExperimentSpinner_);
  outTop->addWidget(openOutputFolderBtn_);
  outputMaxBtn_ = new QPushButton("Maximize", outBox);
  outputMaxBtn_->setProperty("role", "regionMax");
  outputMaxBtn_->setToolTip("Expand the Output area and hide Selection. Click again to restore both areas.");
  outputMaxBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_TitleBarMaxButton), QColor(0, 120, 215), QSize(18, 18), 2.0));
  outputMaxBtn_->setIconSize(QSize(18, 18));
outputMaxBtn_->setFlat(true);
  connect(outputMaxBtn_, &QPushButton::clicked, this, [this]() { toggleRegionFocus(FocusArea::Output); });
  outTop->addSpacing(6);
  outTop->addWidget(outputMaxBtn_);

  // General progress indicator for the active run (separate from the spinner near Run/Stop).
  // - Starts as indeterminate while the CLI runs.
  // - Becomes determinate if the GUI can parse iteration/evaluation progress from the log.
  auto* progRow = new QHBoxLayout();
  auto* progLbl = new QLabel("Progress", outBox);
  outputProgressBar_ = new QProgressBar(outBox);
  outputProgressBar_->setTextVisible(true);
  outputProgressBar_->setMinimumWidth(220);
  outputProgressBar_->setFixedHeight(25);
  outputProgressBar_->setStyleSheet(
    "QProgressBar {"
    "  min-height: 25px;"
    "  max-height: 25px;"
    "  border: 1px solid #555;"
    "  border-radius: 4px;"
    "  background: #222;"
    "  text-align: center;"
    "  color: white;"
    "  font-weight: bold;"
    "}"
    "QProgressBar::chunk {"
    "  background: #2196F3;"
    "  border-radius: 3px;"
    "}");
  outputProgressBar_->setRange(0, 100);
  outputProgressBar_->setValue(0);
  outputProgressBar_->setFormat("Idle");
  progRow->addWidget(progLbl);
  progRow->addWidget(outputProgressBar_, 1);

  // Live elapsed-time readout, above-right of the progress bar. Hidden by
  // default -- startPipelineProgress()/finishPipelineProgress() are the
  // only places that show/tick or hide it, and those are only called by
  // the Meta-dataset pipeline (mode 6) and NN landscape (mode 7) stage
  // functions, so this never appears for Single run / Batch / Sensitivity
  // / Multi-objective. See pipelineElapsedLbl_'s comment in MainWindow.h.
  pipelineElapsedLbl_ = new QLabel(outBox);
  pipelineElapsedLbl_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  pipelineElapsedLbl_->setStyleSheet("color: #aaa;");
  {
    QFont ef = pipelineElapsedLbl_->font();
    ef.setFamily("Monospace");
    ef.setPointSize(ef.pointSize() > 1 ? ef.pointSize() - 1 : ef.pointSize());
    pipelineElapsedLbl_->setFont(ef);
  }
  pipelineElapsedLbl_->setVisible(false);
  auto* elapsedRow = new QHBoxLayout();
  elapsedRow->addStretch(1);
  elapsedRow->addWidget(pipelineElapsedLbl_);

  auto* progCol = new QVBoxLayout();
  progCol->setContentsMargins(0, 0, 0, 0);
  progCol->setSpacing(2);
  progCol->addLayout(elapsedRow);
  progCol->addLayout(progRow);

  progressWidget_ = new QWidget(outBox);
  progressWidget_->setLayout(progCol);
  progressWidget_->setVisible(false);

  pipelineElapsedTimer_ = new QTimer(this);
  pipelineElapsedTimer_->setInterval(1000);
  connect(pipelineElapsedTimer_, &QTimer::timeout, this, &MainWindow::updatePipelineElapsedLabel);

  // Right column: the existing toolbar row, with the progress bar stacked
  // directly underneath it -- same progress bar every other mode (e.g.
  // Batch run) already uses.
  auto* rightColumn = new QVBoxLayout();
  rightColumn->addLayout(outTop);
  rightColumn->addWidget(progressWidget_);

  // ── Unified pipeline panel (run mode 5) ────────────────────────────────
  // Lives in the OUTPUT area, to the LEFT of the existing toolbar/progress
  // column above -- these are actions that produce output, not selection
  // controls. The method/problem lists it uses (batchPanel_) still live in
  // Selection, same as Batch run. Buttons are deliberately narrow (~1/4 of
  // the panel width, via stretch factors below) with their own spinner
  // widget next to whichever one is currently running, matching the
  // Run/Stop + busySpinner_ convention used everywhere else. No inline
  // explanatory text -- full descriptions are tooltips instead.
  pipelinePanel_ = new QGroupBox("Meta-dataset pipeline", outBox);
  // Sized to exactly fit its content (widest button + spinner + margins),
  // no more -- Preferred (not Expanding) so it doesn't claim extra space
  // from the topRow HBox beyond what its buttons actually need.
  // Sized to exactly fit its content (widest button + spinner + margins)
  // and never more -- Maximum (not Preferred) means Qt will actively shrink
  // it to its sizeHint rather than letting it grow into leftover space in
  // topRow, which is what made it balloon out to half the window before.
  pipelinePanel_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
  {
    auto* pLay = new QVBoxLayout(pipelinePanel_);
    pLay->setContentsMargins(8, 8, 8, 8);

    auto makeStageRow = [&](QPushButton*& btn, BusySpinner*& spinner, const QString& text,
                            const QString& tip, const QString& checkmarkObjName,
                            const QString& timeLblObjName) {
      auto* row = new QWidget(pipelinePanel_);
      auto* rowLay = new QHBoxLayout(row);
      rowLay->setContentsMargins(0, 0, 0, 0);
      btn = new QPushButton(text, row);
      btn->setToolTip(tip);
      spinner = new BusySpinner(row);
      spinner->setFixedSize(18, 18);
      spinner->stop();
      // Green checkmark shown once this stage has completed successfully --
      // stays visible so the user can see at a glance which stages are done,
      // as opposed to the spinner which only shows while actively running.
      auto* checkmark = new QLabel(QString::fromUtf8("\xE2\x9C\x93"), row); // "✓"
      checkmark->setObjectName(checkmarkObjName);
      checkmark->setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 14px;");
      checkmark->setVisible(false);
      checkmark->setToolTip("This stage has completed successfully.");
      // How long this stage took the last time it completed successfully --
      // persisted to pipeline_timings.csv, so it's still shown after
      // reloading a previous experiment, not just within this session.
      auto* timeLbl = new QLabel(QString(), row);
      timeLbl->setObjectName(timeLblObjName);
      timeLbl->setStyleSheet("color: #666666;");
      rowLay->addWidget(spinner, 0);
      rowLay->addWidget(btn, 0);
      rowLay->addWidget(checkmark, 0);
      rowLay->addWidget(timeLbl, 0);
      rowLay->addStretch(1);
      pLay->addWidget(row);
      return row;
    };

    makeStageRow(pipelineRunBatchBtn_, pipelineBatchSpinner_,
      "1. Batch + ranking",
      "Runs \"optimsolution --batch ... .\" then automatically \"--rank\" on "
      "the result. Uses the methods/problems selected in the Selection area. "
      "Writes batch_master.csv, batch_ranked.csv, batch_method_summary.csv, "
      "batch_friedman.txt, batch_wilcoxon.csv into optimsolution_gui_run/.",
      "pipelineBatchCheckmark",
      "pipelineBatchTimeLbl");
    connect(pipelineRunBatchBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineBatch(); }); });

    // "Samples per landscape" now lives in Selection (below Run mode) --
    // see form->addRow("Samples per landscape", pipelineSamplesSpin_) above.

    makeStageRow(pipelineRunFeaturesBtn_, pipelineFeaturesSpinner_,
      "2. Landscape sampling",
      "Runs \"optimsolution --features <problem> <dim> <samples>\" once for "
      "every (problem, dimension) pair currently selected in the Selection "
      "area -- no method/algorithm involved. Writes one "
      "<problem>_d<dim>_features.csv per pair into optimsolution_gui_run/.",
      "pipelineFeaturesCheckmark",
      "pipelineFeaturesTimeLbl");
    connect(pipelineRunFeaturesBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineFeatures(); }); });

    makeStageRow(pipelineExtractElaBtn_, pipelineExtractSpinner_,
      "3. Extract ELA (Python)",
      "Runs py_scripts/extract_ela_features.py over every *_features.csv in "
      "optimsolution_gui_run/, computing Exploratory Landscape Analysis "
      "features with pflacco. Writes ela_features.csv. See "
      "py_scripts/README_python_ela.md for one-time dependency setup. "
      "Disabled until stage 2 has produced at least one *_features.csv.",
      "pipelineExtractCheckmark",
      "pipelineExtractTimeLbl");
    connect(pipelineExtractElaBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineExtractEla(); }); });
    pipelineExtractElaBtn_->setEnabled(false);

    makeStageRow(pipelineTrainModelBtn_, pipelineTrainSpinner_,
      "4. Train model (Python)",
      "Runs py_scripts/train_meta_learner.py on ela_features.csv + "
      "batch_ranked.csv, training a landscape-aware algorithm-selection "
      "model with leave-one-problem-out cross-validation. Results appear in "
      "the \"Meta-learning results\" tab. Disabled until stage 1 and stage 3 "
      "have both completed.",
      "pipelineTrainCheckmark",
      "pipelineTrainTimeLbl");
    connect(pipelineTrainModelBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineTrainModel(); }); });
    pipelineTrainModelBtn_->setEnabled(false);

    // Reviewer 2, comment 4: optional 5th stage. Re-runs train_meta_learner.py
    // with --ablation, which does the same run as stage 4 AND additionally
    // repeats the fold-safe LOPO evaluation once per ELA feature family
    // (excluded, then in isolation), writing a comparison table so a reader
    // can see which families the accuracy actually depends on. Shares stage
    // 4's own prerequisites, not stage 4's completion -- it is self-contained.
    makeStageRow(pipelineAblationBtn_, pipelineAblationSpinner_,
      "5. Feature-group ablation (Python)",
      "Runs py_scripts/train_meta_learner.py --ablation on ela_features.csv + "
      "batch_ranked.csv: the same leave-one-problem-out evaluation as stage 4, "
      "repeated once per ELA feature family (with that family excluded, then "
      "with only that family), to show which families the model's accuracy "
      "actually depends on. Results appear in the \"Feature ablation\" tab. "
      "Takes noticeably longer than stage 4 (13 full LOPO passes instead of "
      "1). Disabled until stage 1 and stage 3 have both completed.",
      "pipelineAblationCheckmark",
      "pipelineAblationTimeLbl");
    connect(pipelineAblationBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineAblation(); }); });
    pipelineAblationBtn_->setEnabled(false);

    // Reviewer 2, comment 4: another optional stage, alongside ablation.
    // Re-runs train_meta_learner.py with --compare-classifiers, which does
    // the same run as stage 4 AND additionally repeats the fold-safe LOPO
    // evaluation with SVM, Gradient Boosting, and Logistic Regression, so
    // a reader can see whether this study's Random Forest is actually the
    // best available choice on the same features. Shares stage 4's own
    // prerequisites, not stage 4's completion -- it is self-contained,
    // exactly like the ablation stage.
    makeStageRow(pipelineCompareBtn_, pipelineCompareSpinner_,
      "6. Classifier comparison (Python)",
      "Runs py_scripts/train_meta_learner.py --compare-classifiers on "
      "ela_features.csv + batch_ranked.csv: the same leave-one-problem-out "
      "evaluation as stage 4, repeated with three alternative classifiers "
      "(SVM, Gradient Boosting, Logistic Regression) on the same features, "
      "to show whether this study's Random Forest is actually the strongest "
      "choice. Results appear in the \"Classifier comparison\" tab. Takes "
      "longer than stage 4 (4 full LOPO passes instead of 1). Disabled "
      "until stage 1 and stage 3 have both completed.",
      "pipelineCompareCheckmark",
      "pipelineCompareTimeLbl");
    connect(pipelineCompareBtn_, &QPushButton::clicked, this, [this]() { guardedStart(5, [this]() { onRunPipelineCompareClassifiers(); }); });
    pipelineCompareBtn_->setEnabled(false);

    // Cumulative time across all 4 stages -- only shown once every stage
    // has a recorded time (see refreshPipelineTimingLabels()), since a
    // partial sum would be misleading.
    auto* totalTimeLbl = new QLabel(QString(), pipelinePanel_);
    totalTimeLbl->setObjectName("pipelineTotalTimeLbl");
    totalTimeLbl->setStyleSheet("color: #444444; font-weight: bold;");
    pLay->addWidget(totalTimeLbl);
  }

  auto* topRow = new QHBoxLayout();
  topRow->addWidget(pipelinePanel_, 0);

  // ── Neural-network landscape action panel (run mode 7) ──────────────────
  // Two independent stages, own dedicated buttons, checkmark + time label
  // per stage -- same makeStageRow convention as the pipeline panel above
  // (item 3 of the mode-7 batch upgrade: progress bar + timings, mirroring
  // mode 6). Progress itself is driven by the shared outputProgressBar_ via
  // startPipelineProgress()/finishPipelineProgress() (see
  // MainWindow_NNLandscape.cpp), same widgets every other batch-like mode
  // already uses -- no new progress bar was added.
  nnActionPanel_ = new QGroupBox("NN landscape / generalization gap", outBox);
  nnActionPanel_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
  {
    auto* nActLay = new QVBoxLayout(nnActionPanel_);
    nActLay->setContentsMargins(8, 8, 8, 8);

    auto makeNNRow = [&](QPushButton*& btn, BusySpinner*& spinner, const QString& text, const QString& tip,
                          const QString& checkmarkObjName, const QString& timeLblObjName) {
      auto* row = new QWidget(nnActionPanel_);
      auto* rowLay = new QHBoxLayout(row);
      rowLay->setContentsMargins(0, 0, 0, 0);
      btn = new QPushButton(text, row);
      btn->setToolTip(tip);
      spinner = new BusySpinner(row);
      spinner->setFixedSize(18, 18);
      spinner->stop();
      auto* checkmark = new QLabel(QString::fromUtf8("\xE2\x9C\x93"), row); // "✓"
      checkmark->setObjectName(checkmarkObjName);
      checkmark->setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 14px;");
      checkmark->setVisible(false);
      checkmark->setToolTip("This stage has completed successfully at least once this session.");
      auto* timeLbl = new QLabel(QString(), row);
      timeLbl->setObjectName(timeLblObjName);
      timeLbl->setStyleSheet("color: #666666;");
      rowLay->addWidget(spinner, 0);
      rowLay->addWidget(btn, 0);
      rowLay->addWidget(checkmark, 0);
      rowLay->addWidget(timeLbl, 0);
      rowLay->addStretch(1);
      nActLay->addWidget(row);
    };

    makeNNRow(nnRunTrainBtn_, nnTrainSpinner_,
      "1. Train + sample landscape",
      "Runs \"optimsolution --nnlandscape ...\" once per (dataset, architecture, optimizer, "
      "seed) combination in the batch built from the panel to the left -- every selected "
      "dataset x every ';'-separated architecture x every selected optimizer x \"Runs\" seeds "
      "(seed_base from optimsolution.cfg). Appends one row per combination to "
      "optimsolution_gui_run/nn_summary.csv and writes one *_features.csv per radius per combination.",
      "nnTrainCheckmark", "nnTrainTimeLbl");
    connect(nnRunTrainBtn_, &QPushButton::clicked, this, [this]() { guardedStart(6, [this]() { onRunNNLandscapeTrain(); }); });

    makeNNRow(nnRunAnalyzeBtn_, nnAnalyzeSpinner_,
      "2. Analyze (Python)",
      "Runs py_scripts/train_gap_regressor_logtransform.py over every run recorded so far in "
      "optimsolution_gui_run/nn_summary.csv, merging in the local-landscape samples "
      "and reporting a Leave-One-Dataset-Out comparison (meta-only vs "
      "meta+landscape features) in the Log tab below. Disabled until at least one "
      "training run has completed in this session. Behaves identically to the "
      "original train_gap_regressor.py unless \"Log-transform\" below is checked.",
      "nnAnalyzeCheckmark", "nnAnalyzeTimeLbl");
    connect(nnRunAnalyzeBtn_, &QPushButton::clicked, this, [this]() { guardedStart(6, [this]() { onRunNNLandscapeAnalyze(); }); });
    nnRunAnalyzeBtn_->setEnabled(false);

    // Stage 2 modifier, not its own row: ticking this changes what "2.
    // Analyze" computes (adds --log-transform to the same script call) but
    // does not add a new pipeline stage of its own.
    nnLogTransformCheck_ = new QCheckBox(
        "Log-transform (test whether extreme-scale datasets distort the LODO result)",
        nnActionPanel_);
    nnLogTransformCheck_->setToolTip(
        "Applies slog(x) = sign(x) * log1p(|x|) to every loss-scale quantity "
        "(local-sample f values, train_loss, and the gen_gap_loss target) before "
        "computing any statistic or fitting any model in stage 2. Directly tests "
        "whether a handful of very high-dimensional datasets are distorting the "
        "Leave-One-Dataset-Out result, without changing the sign or discarding any "
        "run. Off by default; re-run \"2. Analyze\" after toggling to see the effect.");
    nActLay->addWidget(nnLogTransformCheck_);

    // Checking this box changes what the NEXT "2. Analyze" run writes to
    // optimsolution_gui_run/nn_analysis/ (lodo_results.csv,
    // feature_importance.csv, merged_dataset.csv) -- same filenames as a
    // previous, non-log-transformed run, so those files would be silently
    // overwritten on disk (not just in the GUI tables) the moment Analyze
    // runs again. Warn once, here, only if there is something to lose, and
    // let the user back out (unchecking again) instead of exporting first.
    connect(nnLogTransformCheck_, &QCheckBox::toggled, this, [this](bool checked) {
      if (!checked) return; // nothing to warn about when turning it back off
      const QString outDir = QDir(nnActiveRunDir()).filePath("nn_analysis");
      const bool hasExistingResults =
          QFileInfo::exists(QDir(outDir).filePath("lodo_results.csv")) ||
          QFileInfo::exists(QDir(outDir).filePath("merged_dataset.csv"));
      if (!hasExistingResults) return; // no previous "2. Analyze" run yet -- nothing to protect

      const auto ans = QMessageBox::question(
          this,
          "Log-transform will overwrite previous results",
          "Running \"2. Analyze\" with Log-transform checked writes over the same "
          "files your last \"2. Analyze\" run already wrote to:\n\n"
          "    " + outDir + "\n"
          "    (lodo_results.csv, feature_importance.csv, merged_dataset.csv)\n\n"
          "If you want to keep those earlier (non-log-transformed) results, go to "
          "the LODO Results / Feature Importances tabs now and use \"Export "
          "XLSX...\" to save them somewhere else first -- once \"2. Analyze\" runs "
          "again, the files on disk are replaced and cannot be recovered from "
          "inside the app.\n\n"
          "Continue with Log-transform checked?",
          QMessageBox::Yes | QMessageBox::No,
          QMessageBox::No);

      if (ans != QMessageBox::Yes) {
        QSignalBlocker blocker(nnLogTransformCheck_); // avoid re-entering this lambda
        nnLogTransformCheck_->setChecked(false);
      }
    });

    makeNNRow(nnRunBootstrapBtn_, nnBootstrapSpinner_,
      "3. Bootstrap CI (optional)",
      "Runs py_scripts/bootstrap_lodo_ci.py on stage 2's own merged_dataset.csv, "
      "cluster-bootstrapping over datasets to put a 95% confidence interval on the "
      "R2 / MAE / Spearman numbers already shown in the LODO Results tab -- directly "
      "addresses \"how stable is this with only a couple dozen datasets?\" without a "
      "new training experiment. Disabled until \"2. Analyze\" has completed at least "
      "once this session.",
      "nnBootstrapCheckmark", "nnBootstrapTimeLbl");
    connect(nnRunBootstrapBtn_, &QPushButton::clicked, this, [this]() { guardedStart(6, [this]() { onRunNNLandscapeBootstrap(); }); });
    nnRunBootstrapBtn_->setEnabled(false);
  }
  topRow->addWidget(nnActionPanel_, 0);
  // rightColumn (toolbar buttons + progress bar) stretches to fill the rest
  // of the Output area's width -- buttons end up pushed to the top-right
  // (via outTop's own internal stretch), and the progress bar spans exactly
  // from where the pipeline panel ends to the Output area's right edge.
  topRow->addLayout(rightColumn, 1);
  outLay->addLayout(topRow);

  // Status + elapsed time row (all modes).
  auto* statusRow = new QHBoxLayout();
  outputStatusLbl_ = new QLabel("Idle.", outBox);
  outputStatusLbl_->setWordWrap(false);
  outputStatusLbl_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  outputElapsedLbl_ = new QLabel("", outBox);
  outputElapsedLbl_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  outputElapsedLbl_->setMinimumWidth(80);
  {
    QFont ef = outputElapsedLbl_->font();
    ef.setFamily("Monospace");
    outputElapsedLbl_->setFont(ef);
  }
  statusRow->addWidget(outputStatusLbl_, 1);
  statusRow->addWidget(outputElapsedLbl_);
  outLay->addLayout(statusRow);

  outputTabs_ = new ModeAwareTabWidget(outBox);
  outputTabs_->setDocumentMode(true);
  // Every run tab belongs to the run mode that was active when it was created;
  // each mode tab of the main window shows only its own run tabs.
  outputTabs_->onTabInserted = [this](int index) {
    if (!outputTabs_ || index < 0 || index >= outputTabs_->count()) return;
    QWidget* page = outputTabs_->widget(index);
    if (!page) return;
    page->setProperty("optimMode", currentModeIndex());
    outputTabs_->setTabVisible(index, true);
    // A QTabWidget that has no (visible) current tab of this mode behaves like
    // an empty one: the first tab added becomes the current tab.
    const int cur = outputTabs_->currentIndex();
    const bool curUsable = cur >= 0 && cur < outputTabs_->count() && cur != index &&
                           outputTabs_->isTabVisible(cur);
    if (!curUsable) outputTabs_->setCurrentIndex(index);
  };

  outputTabs_->setTabsClosable(true);
  connect(outputTabs_, &QTabWidget::tabCloseRequested, this, &MainWindow::onOutputTabCloseRequested);
  connect(outputTabs_, &QTabWidget::currentChanged, this, &MainWindow::onOutputTabChanged);

  // Do not create a batch summary tab eagerly.
  // It is created only when a batch run starts or when external batch results are loaded.
  activeOutputRunIndex_ = -1;


  outLay->addWidget(outputTabs_);
  // ── Code Wizard tab ──────────────────────────────────────────────────
  // The Code Wizard used to be the last tab of the Settings area. It is now a
  // top-level tab of its own (see assembleMainTabs()); the five actions and
  // their handlers are exactly the same as before.
  auto* wizardPage = new QWidget(central);
  auto* wizPageLay = new QVBoxLayout(wizardPage);
  wizPageLay->setContentsMargins(6, 6, 6, 6);

  auto* wizardBox = new QGroupBox("Code Wizard", wizardPage);
  wizardBox->setObjectName("codeWizardBox");
  wizardBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  auto* wizLay = new QVBoxLayout(wizardBox);
  wizLay->setContentsMargins(12, 12, 12, 12);
  wizLay->setSpacing(10);

  auto* wizDescLbl = new QLabel(
    "Generate skeleton .h/.cpp files for new methods or problems.\n"
    "Files are placed in src/methods/ or src/problems/ of your project.", wizardBox);
  wizDescLbl->setWordWrap(true);
  wizDescLbl->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  wizLay->addWidget(wizDescLbl);

  auto* newMethodBtn = new QPushButton("New method…", wizardBox);
  newMethodBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  newMethodBtn->setIconSize(QSize(18, 18));
  newMethodBtn->setToolTip("Open the Method Wizard to generate a new optimizer .h/.cpp skeleton");

  auto* newProblemBtn = new QPushButton("New problem…", wizardBox);
  newProblemBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  newProblemBtn->setIconSize(QSize(18, 18));
  newProblemBtn->setToolTip("Open the Problem Wizard to generate a new benchmark problem .h/.cpp skeleton");

  connect(newMethodBtn,  &QPushButton::clicked, this, &MainWindow::onNewMethod);
  connect(newProblemBtn, &QPushButton::clicked, this, &MainWindow::onNewProblem);

  auto* delMethodBtn = new QPushButton("Delete method…", wizardBox);
  delMethodBtn->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  delMethodBtn->setIconSize(QSize(18, 18));
  delMethodBtn->setToolTip("Remove a method: deletes .h/.cpp, factory.cpp entry, CMakeLists.txt entry");

  auto* delProblemBtn = new QPushButton("Delete problem…", wizardBox);
  delProblemBtn->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  delProblemBtn->setIconSize(QSize(18, 18));
  delProblemBtn->setToolTip("Remove a problem: deletes .h/.cpp, factory.cpp entry, CMakeLists.txt entry");

  connect(delMethodBtn,  &QPushButton::clicked, this, &MainWindow::onDeleteMethod);
  connect(delProblemBtn, &QPushButton::clicked, this, &MainWindow::onDeleteProblem);

  auto* openFileBtn = new QPushButton("Open file…", wizardBox);
  openFileBtn->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
  openFileBtn->setIconSize(QSize(18, 18));
  openFileBtn->setToolTip("Open any .h/.cpp file pair for editing and building");
  connect(openFileBtn, &QPushButton::clicked, this, [this]() {
    const QString h = QFileDialog::getOpenFileName(this, "Open header (.h)",
        projectRoot_, "C++ Header (*.h)");
    if (h.isEmpty()) return;
    const QString cpp = QFileDialog::getOpenFileName(this, "Open source (.cpp)",
        QFileInfo(h).absolutePath(), "C++ Source (*.cpp)");
    if (cpp.isEmpty()) return;
    auto* ed = new optimsolution::CodeEditorDialog(h, cpp, projectRoot_, this);
    ed->setAttribute(Qt::WA_DeleteOnClose);
    ed->show();
  });

  // Same buttons, laid out as a small grid: Methods / Problems / Files.
  auto* wizGrid = new QGridLayout();
  wizGrid->setHorizontalSpacing(10);
  wizGrid->setVerticalSpacing(8);
  auto addWizRow = [&](int row, const QString& caption, QPushButton* a, QPushButton* b) {
    auto* lbl = new QLabel(caption, wizardBox);
    wizGrid->addWidget(lbl, row, 0);
    for (QPushButton* btn : {a, b}) {
      if (btn) btn->setMinimumWidth(170);
    }
    wizGrid->addWidget(a, row, 1);
    if (b) wizGrid->addWidget(b, row, 2);
  };
  addWizRow(0, "Methods",  newMethodBtn,  delMethodBtn);
  addWizRow(1, "Problems", newProblemBtn, delProblemBtn);
  addWizRow(2, "Files",    openFileBtn,   nullptr);
  wizGrid->setColumnStretch(3, 1);
  wizLay->addLayout(wizGrid);

  wizPageLay->addWidget(wizardBox, 0, Qt::AlignTop);
  wizPageLay->addStretch(1);
  codeWizardPage_ = wizardPage;

  // Assemble the tabbed main window: one tab per run mode (Selection + Output),
  // then Code Wizard, then Settings.
  assembleMainTabs(central, rootLay, wizardPage);

  setCentralWidget(central);

  applyRegionStyling();
  updateRegionFocusStyling();

  statusBar()->showMessage("OptimSolution by OptimTeam");

  // connections
  connect(refreshBtn_, &QPushButton::clicked, this, &MainWindow::onRefreshFactory);
  // The displayed text may carry the supported dimensions
  // ("cec2017f1  (D: 10, 30, 50)"); the handler gets the bare short name.
  // Still driven by the TEXT, as before: the combo box is editable, so a
  // name typed by hand must keep working exactly as it did.
  connect(problemBox_, &QComboBox::currentTextChanged, this,
          [this](const QString& t) { onProblemChanged(problemNameFromDisplay(t)); });
  connect(methodBox_, &QComboBox::currentTextChanged, this, &MainWindow::onMethodChanged);
  connect(runBtn_, &QPushButton::clicked, this, [this]() {
    guardedStart(currentModeIndex(), [this]() { onRunClicked(); });
  });
  // Tabbed main window: move the workspace / filter the output tabs BEFORE the
  // existing onRunModeChanged() runs (so it already sees the new mode's own
  // output tab as the active one), and refine the Selection area AFTER it has
  // applied its own per-mode visibility. onRunModeChanged() itself is unchanged.
  connect(runModeBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
    onRunModeComboChangedBefore(currentModeIndex());
  });
  connect(runModeBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onRunModeChanged);
  connect(runModeBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
    onRunModeComboChangedAfter(currentModeIndex());
  });
  // Defensive: if Qt ever makes a hidden (other-mode) output tab current -- e.g.
  // after closing a tab -- bring the view back to this mode's own tabs.
  connect(outputTabs_, &QTabWidget::currentChanged, this, [this](int index) {
    if (!outputTabs_ || outputFilterGuard_ || index < 0) return;
    if (index < outputTabs_->count() && !outputTabs_->isTabVisible(index)) {
      filterOutputTabsForMode(currentModeIndex());
    }
  });
  // Batch metric UI — batchMetricCombo_ and batchAggCombo_ are connected at
  // system-tab creation time (createOutputRunTab → Wilcoxon/Friedman panels).
  if (auto* batchShowMeanChk = batchSettingsBox->findChild<QCheckBox*>("batchShowMeanChk")) {
    connect(batchShowMeanChk, &QCheckBox::toggled, this, &MainWindow::onBatchMetricUiChanged);
  }
  if (batchShowRateChk_) {
    connect(batchShowRateChk_, &QCheckBox::toggled, this, &MainWindow::onBatchMetricUiChanged);
  }
  if (batchShowSdChk_) {
    connect(batchShowSdChk_, &QCheckBox::toggled, this, &MainWindow::onBatchMetricUiChanged);
  }
  if (batchShowTimeChk_) {
    connect(batchShowTimeChk_, &QCheckBox::toggled, this, &MainWindow::onBatchMetricUiChanged);
  }
  updateBatchPanelVisibility();
  if (selectionBox_) {
    const auto combos = selectionBox_->findChildren<QComboBox*>();
    for (QComboBox* cb : combos) makeComboCompact(cb);
  }
  applyModeSpecificSelectionUi(currentModeIndex());
  // Initialise the Runs info label in the Selection area.
  if (auto* lbl = findChild<QLabel*>("selectionRunsLbl"))
    lbl->setText(QString::number(globalRunsFromSettings()));

  connect(selectCfgBtn_, &QPushButton::clicked, this, &MainWindow::onSelectSettings);
  connect(reloadCfgBtn_, &QPushButton::clicked, this, &MainWindow::onReloadSettings);
  connect(saveCfgBtn_, &QPushButton::clicked, this, &MainWindow::onSaveSettings);

  connect(addGlobalParamBtn_, &QPushButton::clicked, this, &MainWindow::onAddGlobalParam);
  connect(removeGlobalParamBtn_, &QPushButton::clicked, this, &MainWindow::onRemoveGlobalParam);
  connect(addStopParamBtn_, &QPushButton::clicked, this, &MainWindow::onAddStopParam);
  connect(removeStopParamBtn_, &QPushButton::clicked, this, &MainWindow::onRemoveStopParam);
  connect(addInitParamBtn_, &QPushButton::clicked, this, &MainWindow::onAddInitParam);
  connect(removeInitParamBtn_, &QPushButton::clicked, this, &MainWindow::onRemoveInitParam);

  connect(addMethodParamBtn_, &QPushButton::clicked, this, &MainWindow::onAddMethodParam);
  connect(removeMethodParamBtn_, &QPushButton::clicked, this, &MainWindow::onRemoveMethodParam);

  auto loadExternalBatchFromDirectory = [this](const QString& dirPath, const QString& selectedCsvPath) -> bool {
    QDir dir(dirPath);
    if (!dir.exists()) return false;

    auto collectKnownKeysFromList = [](QListWidget* list) -> QStringList {
      QStringList out;
      if (!list) return out;
      for (int i = 0; i < list->count(); ++i) {
        if (auto* it = list->item(i)) {
          QString key = it->data(Qt::UserRole).toString().trimmed();
          if (key.contains(' ')) key = key.section(' ', 0, 0).trimmed();
          if (key.isEmpty()) key = it->text().section(' ', 0, 0).trimmed();
          if (!key.isEmpty() && !out.contains(key, Qt::CaseInsensitive)) out << key;
        }
      }
      return out;
    };

    auto appendUniqueComboItems = [](QStringList& out, const QComboBox* box) {
      if (!box) return;
      for (int i = 0; i < box->count(); ++i) {
        QString key = box->itemData(i).toString().trimmed();
        if (key.contains(' ')) key = key.section(' ', 0, 0).trimmed();
        if (key.isEmpty()) key = box->itemText(i).section(' ', 0, 0).trimmed();
        if (!key.isEmpty() && !out.contains(key, Qt::CaseInsensitive)) out << key;
      }
    };

    QStringList knownMethods = collectKnownKeysFromList(batchMethodsList_);
    appendUniqueComboItems(knownMethods, methodBox_);
    QStringList knownProblems = collectKnownKeysFromList(batchProblemsList_);
    appendUniqueComboItems(knownProblems, problemBox_);

    std::sort(knownMethods.begin(), knownMethods.end(), [](const QString& a, const QString& b) {
      if (a.size() != b.size()) return a.size() > b.size();
      return a.toLower() < b.toLower();
    });
    std::sort(knownProblems.begin(), knownProblems.end(), [](const QString& a, const QString& b) {
      if (a.size() != b.size()) return a.size() > b.size();
      return a.toLower() < b.toLower();
    });

    auto canonicalizeKnownKey = [](const QStringList& known, const QString& value) -> QString {
      for (const QString& k : known) {
        if (k.compare(value, Qt::CaseInsensitive) == 0) return k;
      }
      return value;
    };

    auto parseConvergenceCsvMetadataFromFilename = [&](const QFileInfo& cfi,
                                                       QString& outMethod,
                                                       QString& outProblem,
                                                       int& outDim,
                                                       QString& outError) -> bool {
      outMethod.clear();
      outProblem.clear();
      outDim = 0;
      outError.clear();

      const QString fileName = cfi.fileName().trimmed();
      static const QRegularExpression re(
        QStringLiteral(R"(^(.+)_d(\d+)_\d{8}-\d{6}_convergence\.csv$)"),
        QRegularExpression::CaseInsensitiveOption);
      const QRegularExpressionMatch m = re.match(fileName);
      if (!m.hasMatch()) {
        outError = QString("filename does not match the expected convergence pattern");
        return false;
      }

      bool okDim = false;
      outDim = m.captured(2).toInt(&okDim);
      if (!okDim || outDim <= 0) {
        outError = QString("failed to parse dimension from filename");
        return false;
      }

      const QString stem = m.captured(1).trimmed();
      QString method;
      QString problem;

      for (const QString& knownMethod : knownMethods) {
        const QString prefix = knownMethod + "_";
        if (stem.compare(knownMethod, Qt::CaseInsensitive) == 0) {
          method = knownMethod;
          problem.clear();
          break;
        }
        if (stem.startsWith(prefix, Qt::CaseInsensitive)) {
          method = knownMethod;
          problem = stem.mid(prefix.size()).trimmed();
          break;
        }
      }

      if (method.isEmpty()) {
        int splitPos = -1;
        for (int i = 0; i < stem.size(); ++i) {
          if (stem.at(i) == QChar('_')) {
            splitPos = i;
            break;
          }
        }
        if (splitPos <= 0 || splitPos >= stem.size() - 1) {
          outError = QString("failed to infer method/problem from filename stem '%1'").arg(stem);
          return false;
        }
        method = stem.left(splitPos).trimmed();
        problem = stem.mid(splitPos + 1).trimmed();
      }

      if (problem.isEmpty()) {
        outError = QString("failed to infer problem name from filename stem '%1'").arg(stem);
        return false;
      }

      outMethod = canonicalizeKnownKey(knownMethods, method);
      outProblem = canonicalizeKnownKey(knownProblems, problem);
      return true;
    };

    // BUGFIX: an experiment is defined by its manifest (*.osexp, written by
    // every batch run, see writeBatchManifest()), NOT by "every CSV in the
    // folder": single runs and other batches write into the same folder and
    // used to be merged into the loaded experiment (e.g. a Single-run
    // "weierstrass D=60" appeared as an extra problem row and was then also
    // re-run by the next batch). The manifest is either the selected file
    // itself or the newest manifest of the folder that lists the selected CSV.
    QString manifestPath;
    QList<BatchManifestEntry> manifestEntries;
    {
      const QFileInfo selFi(selectedCsvPath);
      if (selFi.suffix().compare(batchManifestFileSuffix(), Qt::CaseInsensitive) == 0) {
        manifestPath = selFi.absoluteFilePath();
      } else if (!selectedCsvPath.trimmed().isEmpty()) {
        manifestPath = findBatchManifestForCsv(dir.absolutePath(), selFi.absoluteFilePath());
      }
      if (!manifestPath.isEmpty()) {
        QString merr;
        if (!readBatchManifest(manifestPath, manifestEntries, &merr)) {
          QMessageBox::warning(this, "Load experiment CSV", merr);
          return true;   // handled (do not fall back to a single-CSV tab)
        }
      }
    }

    QFileInfoList summaryFiles;
    if (manifestPath.isEmpty()) {
      QDirIterator it(dir.absolutePath(), QStringList() << "*_summary.csv", QDir::Files, QDirIterator::Subdirectories);
      while (it.hasNext()) {
        it.next();
        summaryFiles.push_back(it.fileInfo());
      }
      std::sort(summaryFiles.begin(), summaryFiles.end(), [](const QFileInfo& a, const QFileInfo& b) {
        return a.absoluteFilePath().toLower() < b.absoluteFilePath().toLower();
      });
    }

    QFileInfoList convergenceFiles;
    auto scanConvergenceFiles = [&]() {
      convergenceFiles.clear();
      QDirIterator it(dir.absolutePath(), QStringList() << "*_convergence.csv", QDir::Files, QDirIterator::Subdirectories);
      while (it.hasNext()) {
        it.next();
        convergenceFiles.push_back(it.fileInfo());
      }
      std::sort(convergenceFiles.begin(), convergenceFiles.end(), [](const QFileInfo& a, const QFileInfo& b) {
        return a.absoluteFilePath().toLower() < b.absoluteFilePath().toLower();
      });
    };

    const QString selectedName = QFileInfo(selectedCsvPath).fileName().toLower();
    bool useSummaryFiles = !summaryFiles.isEmpty();
    if (!manifestPath.isEmpty()) {
      useSummaryFiles = false;
    } else if (useSummaryFiles && summaryFiles.size() < 2 && !selectedName.endsWith("_summary.csv")) {
      scanConvergenceFiles();
      if (convergenceFiles.size() >= 2) {
        useSummaryFiles = false;
      } else {
        return false;
      }
    } else if (!useSummaryFiles) {
      scanConvergenceFiles();
      if (convergenceFiles.size() < 2) return false;
    }

    QMap<QString, QMap<QString, BatchCellData>> loadedCells;
    QMap<QString, QMap<QString, QString>> loadedCsvPaths;
    QMap<QString, int> loadedDims;
    QMap<QString, QSet<int>> loadedProblemDimsByBase;
    QSet<QString> methodSet;
    QSet<QString> problemSet;
    QStringList loadedLines;
    QStringList skippedLines;
    int loadedCount = 0;

    // Step 1 (fast, UI thread): decide which convergence CSV belongs to
    // which (method, problem, dim) cell -- no CSV content is read yet.
    struct PendingCell {
      QString method;
      QString problem;
      int dim = 0;
      QString csvPath;
    };
    QVector<PendingCell> pending;

    if (!manifestPath.isEmpty()) {
      for (const BatchManifestEntry& e : manifestEntries) {
        if (!QFileInfo::exists(e.csvPath)) {
          skippedLines << QString("Skipped %1 + %2 (D=%3): file not found: %4")
                            .arg(e.method, e.problem).arg(e.dim).arg(e.csvPath);
          continue;
        }
        PendingCell pc;
        pc.method  = canonicalizeKnownKey(knownMethods, e.method);
        pc.problem = canonicalizeKnownKey(knownProblems, e.problem);
        pc.dim     = e.dim;
        pc.csvPath = e.csvPath;
        pending.push_back(pc);
      }
    } else if (useSummaryFiles) {
      for (const QFileInfo& sfi : summaryFiles) {
        QString method;
        QString problem;
        int dim = 0;
        QString metaErr;
        if (!parseBatchSummaryCsvMetadata(sfi.absoluteFilePath(), method, problem, dim, metaErr)) {
          skippedLines << QString("Skipped %1: %2").arg(sfi.fileName(), metaErr);
          continue;
        }

        const QString convPath = siblingConvergenceCsvForSummary(sfi.absoluteFilePath());
        if (convPath.isEmpty() || !QFileInfo::exists(convPath)) {
          skippedLines << QString("Skipped %1: sibling convergence CSV was not found.").arg(sfi.fileName());
          continue;
        }
        pending.push_back(PendingCell{method, problem, dim, convPath});
      }
    } else {
      for (const QFileInfo& cfi : convergenceFiles) {
        QString method;
        QString problem;
        int dim = 0;
        QString metaErr;
        if (!parseConvergenceCsvMetadataFromFilename(cfi, method, problem, dim, metaErr)) {
          skippedLines << QString("Skipped %1: %2").arg(cfi.fileName(), metaErr);
          continue;
        }
        pending.push_back(PendingCell{method, problem, dim, cfi.absoluteFilePath()});
      }
    }

    // Step 2 (slow part): parse all convergence CSVs in parallel on worker
    // threads while the GUI keeps running and shows the progress. This used
    // to be done one file after the other on the GUI thread, which froze the
    // whole window for minutes on large experiments (hundreds of multi-MB
    // CSVs). loadBatchCellFromConvergence() only touches its own arguments,
    // so it is safe to call concurrently (the batch post-processing already
    // calls it from a worker thread).
    struct ParsedCell {
      bool ok = false;
      BatchCellData cell;
      QString err;
    };
    QVector<ParsedCell> parsed(pending.size());
    bool loadCancelled = false;
    if (!pending.isEmpty()) {
      ParsedCell* const out = parsed.data();          // fixed storage, one slot per file
      const PendingCell* const in = pending.constData();
      const int total = int(pending.size());

      // A private pool (QtCore only -- no extra Qt module needed), so that
      // waitForDone() below waits for exactly these jobs.
      QThreadPool pool;
      pool.setMaxThreadCount(std::max(1, QThread::idealThreadCount()));
      std::atomic<int> done{0};
      std::atomic<bool> cancelRequested{false};
      for (int i = 0; i < total; ++i) {
        pool.start([this, out, in, i, &done, &cancelRequested]() {
          if (!cancelRequested.load()) {
            out[i].ok = loadBatchCellFromConvergence(in[i].csvPath, out[i].cell, &out[i].err);
          }
          done.fetch_add(1);
        });
      }

      QProgressDialog progress(QString("Loading experiment results (%1 files)...").arg(total),
                               "Cancel", 0, total, this);
      progress.setWindowTitle("Load experiment CSV");
      progress.setWindowModality(Qt::WindowModal);
      progress.setMinimumDuration(300);
      progress.setAutoClose(false);
      progress.setAutoReset(false);
      progress.setValue(0);
      connect(&progress, &QProgressDialog::canceled, &progress, [&cancelRequested]() {
        cancelRequested.store(true);
      });

      // Keep the GUI alive (repaints, spinner, Cancel button) while the
      // workers parse; poll their progress a few times per second.
      QEventLoop waitLoop;
      QTimer poll;
      poll.setInterval(50);
      connect(&poll, &QTimer::timeout, &waitLoop, [&]() {
        if (progress.wasCanceled()) cancelRequested.store(true);
        const int d = done.load();
        if (!cancelRequested.load()) {
          progress.setValue(d);
          progress.setLabelText(QString("Loading experiment results: %1 / %2 files...").arg(d).arg(total));
        }
        if (d >= total) waitLoop.quit();
      });
      poll.start();
      if (done.load() < total) waitLoop.exec();
      poll.stop();
      pool.waitForDone();          // every job has returned; 'parsed' is complete
      loadCancelled = cancelRequested.load() || progress.wasCanceled();
      progress.close();
    }
    if (loadCancelled) {
      appendLog("Load experiment CSV: cancelled by the user; nothing was loaded.");
      statusBar()->showMessage("Loading the experiment was cancelled.", 6000);
      return true;   // handled: do not fall back to a single-CSV tab
    }

    // Step 3 (fast, UI thread): build the experiment in the original order.
    for (int i = 0; i < pending.size(); ++i) {
      const PendingCell& pc = pending[i];
      const ParsedCell& pr = parsed[i];
      if (!pr.ok) {
        skippedLines << QString("Skipped %1: %2").arg(QFileInfo(pc.csvPath).fileName(), pr.err);
        continue;
      }
      const QString problemKey = batchProblemDisplayKey(pc.problem, pc.dim);
      loadedCells[problemKey][pc.method] = pr.cell;
      loadedCsvPaths[problemKey][pc.method] = pc.csvPath;
      loadedDims[problemKey] = pc.dim;
      methodSet.insert(pc.method);
      problemSet.insert(problemKey);
      loadedProblemDimsByBase[batchBaseProblemShort(problemKey)].insert(pc.dim);
      loadedLines << QString("Loaded batch cell: %1 + %2").arg(pc.method, problemKey);
      ++loadedCount;
    }

    if (loadedCells.isEmpty()) {
      QMessageBox::warning(
        this,
        "Load experiment CSV",
        "No batch results could be reconstructed from the selected folder."
      );
      return false;
    }

    // Legacy folders (results written before experiment manifests existed):
    // the folder may mix several experiments and single runs. A problem row
    // that does not have results for every method (typically a single run,
    // e.g. "weierstrass D=60" run with one or two methods) is very likely NOT
    // part of the experiment -- let the user decide; such rows start
    // unchecked, complete rows start checked.
    if (manifestPath.isEmpty() && problemSet.size() > 1) {
      QStringList probs = problemSet.values();
      std::sort(probs.begin(), probs.end(), [](const QString& a, const QString& b) { return a.toLower() < b.toLower(); });
      const int nMeth = methodSet.size();
      bool anyIncomplete = false;
      for (const QString& pk : probs) if (loadedCells.value(pk).size() < nMeth) { anyIncomplete = true; break; }
      if (anyIncomplete) {
        QDialog dlg(this);
        dlg.setWindowTitle("Load experiment CSV");
        auto* lay = new QVBoxLayout(&dlg);
        auto* info = new QLabel(QString(
          "This folder has no experiment manifest (results written by an older version), so the "
          "experiment is reconstructed from all CSV files of the folder. It can also contain single "
          "runs or other batches.\n\nProblems that do not have results for all %1 methods are "
          "unchecked. Check the problems that belong to this experiment:").arg(nMeth), &dlg);
        info->setWordWrap(true);
        lay->addWidget(info);
        auto* lst = new QListWidget(&dlg);
        for (const QString& pk : probs) {
          const QMap<QString, BatchCellData> row = loadedCells.value(pk);
          QStringList ms = row.keys();
          std::sort(ms.begin(), ms.end(), [](const QString& a, const QString& b) { return a.toLower() < b.toLower(); });
          const bool complete = (row.size() >= nMeth);
          auto* item = new QListWidgetItem(
            complete ? QString("%1   (%2/%3 methods)").arg(pk).arg(row.size()).arg(nMeth)
                     : QString("%1   (%2/%3 methods: %4)").arg(pk).arg(row.size()).arg(nMeth).arg(ms.join(", ")),
            lst);
          item->setData(Qt::UserRole, pk);
          item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
          item->setCheckState(complete ? Qt::Checked : Qt::Unchecked);
        }
        lay->addWidget(lst);
        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
        connect(bb, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
        lay->addWidget(bb);
        dlg.resize(560, 420);
        if (dlg.exec() != QDialog::Accepted) return true;   // cancelled: nothing loaded

        QSet<QString> keep;
        for (int i = 0; i < lst->count(); ++i)
          if (lst->item(i)->checkState() == Qt::Checked) keep.insert(lst->item(i)->data(Qt::UserRole).toString());
        if (keep.isEmpty()) return true;

        for (const QString& pk : probs) {
          if (keep.contains(pk)) continue;
          for (const QString& m : loadedCells.value(pk).keys())
            loadedLines.removeAll(QString("Loaded batch cell: %1 + %2").arg(m, pk));
          loadedCount -= loadedCells.value(pk).size();
          loadedCells.remove(pk);
          loadedCsvPaths.remove(pk);
          loadedDims.remove(pk);
          problemSet.remove(pk);
          skippedLines << QString("Excluded by the user: %1").arg(pk);
        }
        // rebuild the dependent sets from what is left
        methodSet.clear();
        loadedProblemDimsByBase.clear();
        for (auto it = loadedCells.cbegin(); it != loadedCells.cend(); ++it) {
          for (auto jt = it.value().cbegin(); jt != it.value().cend(); ++jt) methodSet.insert(jt.key());
          loadedProblemDimsByBase[batchBaseProblemShort(it.key())].insert(loadedDims.value(it.key()));
        }
      }
    }

    // FIX 1: Persist the current experiment's snapshot BEFORE overwriting the global
    // state with the new experiment's data.  Without this, switching back to a
    // previously loaded batch tab after loading a second one would find stale (or
    // empty) batchCells_ / csvPaths because the old values were never committed.
    if (g_activeBatchSummaryPage && g_batchSummarySnapshots.contains(g_activeBatchSummaryPage)) {
      BatchSummarySnapshot prevSnap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
      prevSnap.csvPaths             = g_liveBatchCsvPaths;
      prevSnap.cachedProblemDims    = batchCachedProblemDims_;
      g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, prevSnap);
      batchSummaryCellsByPage_[g_activeBatchSummaryPage] = batchCells_;
    }

    batchCells_ = loadedCells;
    batchCachedProblemDims_ = loadedDims;
    g_liveBatchCsvPaths = loadedCsvPaths;
    batchProblemDimOverride_.clear();
    for (auto it = loadedProblemDimsByBase.begin(); it != loadedProblemDimsByBase.end(); ++it) {
      if (fixedDimForProblem(it.key()) > 0) continue;
      QList<int> dims = it.value().values();
      std::sort(dims.begin(), dims.end());
      QStringList parts;
      for (int d : dims) {
        if (d > 0) parts << QString::number(d);
      }
      if (!parts.isEmpty()) {
        batchProblemDimOverride_[it.key()] = parts;
      }
    }

    // If all reconstructed variable-dimension problems agree on one dimension,
    // reflect that in the Selection-area spin box as well. This prevents the UI
    // from falling back to the stale default (commonly 2) after loading an
    // external batch whose actual dimension is, for example, 20.
    if (dimSpin_) {
      QSet<int> variableDims;
      for (auto it = loadedProblemDimsByBase.begin(); it != loadedProblemDimsByBase.end(); ++it) {
        if (fixedDimForProblem(it.key()) > 0) continue;
        for (int d : it.value()) {
          if (d > 0) variableDims.insert(d);
        }
      }
      if (variableDims.size() == 1) {
        const int loadedDefaultDim = *variableDims.constBegin();
        QSignalBlocker blocker(dimSpin_);
        dimSpin_->setValue(std::clamp(loadedDefaultDim, dimSpin_->minimum(), dimSpin_->maximum()));
      }
    }

    QStringList methods = methodSet.values();
    QStringList problems = problemSet.values();
    std::sort(methods.begin(), methods.end(), [](const QString& a, const QString& b){ return a.toLower() < b.toLower(); });
    std::sort(problems.begin(), problems.end(), [this, &loadedDims](const QString& a, const QString& b) {
      const QString baseA = batchBaseProblemShort(a).toLower();
      const QString baseB = batchBaseProblemShort(b).toLower();
      if (baseA != baseB) return baseA < baseB;
      return loadedDims.value(a, 0) < loadedDims.value(b, 0);
    });

    auto selectListItems = [this](QListWidget* list, const QStringList& wanted) {
      if (!list) return;
      QSet<QString> wantedExact;
      QSet<QString> wantedBase;
      for (const QString& w : wanted) {
        const QString exact = w.trimmed().toLower();
        if (!exact.isEmpty()) wantedExact.insert(exact);
        const QString base = batchBaseProblemShort(w).trimmed().toLower();
        if (!base.isEmpty()) wantedBase.insert(base);
      }

      QSignalBlocker blocker(list);
      // FIX 8a: Also block the QAbstractItemModel / QItemSelectionModel that the
      // QListWidget owns.  QSignalBlocker(list) only suppresses signals emitted by
      // the QListWidget itself (e.g. itemSelectionChanged), but the underlying
      // QItemSelectionModel emits its own selectionChanged signal independently.
      // That signal is connected to refreshBatchSelectionView() for batchProblemsList_,
      // so without blocking it, every setSelected() call during loading triggers
      // refreshBatchSelectionView(), which overwrites batchCells_ with the previous
      // experiment's cached data and runs initStatsTableForBatch() on the wrong tab.
      const QSignalBlocker smBlocker(list->selectionModel());
      list->clearSelection();
      for (int i = 0; i < list->count(); ++i) {
        if (auto* it = list->item(i)) {
          QString key = it->data(Qt::UserRole).toString().trimmed();
          if (key.isEmpty()) key = it->text().trimmed();
          const QString exact = key.toLower();
          const QString base = batchBaseProblemShort(key).trimmed().toLower();
          const bool sel = wantedExact.contains(exact) || (!base.isEmpty() && wantedBase.contains(base));
          it->setSelected(sel);
        }
      }
    };

    selectListItems(batchMethodsList_, methods);
    selectListItems(batchProblemsList_, problems);

    // The list selections above are applied under QSignalBlocker, so the batch-method
    // settings combo in the Settings area does not refresh automatically while the UI
    // is already in Batch mode. Refresh it explicitly here without touching any other
    // code path.
    if (batchMethodsList_) {
      QListWidgetItem* firstSelectedMethod = nullptr;
      for (int i = 0; i < batchMethodsList_->count(); ++i) {
        if (auto* it = batchMethodsList_->item(i); it && it->isSelected()) {
          firstSelectedMethod = it;
          break;
        }
      }
      if (firstSelectedMethod) {
        // FIX 8b: setCurrentItem fires currentItemChanged (connected to
        // refreshBatchSelectionView) even when the item is already current.
        // Block the list and its selection model to keep the signal silent.
        QSignalBlocker b1(batchMethodsList_);
        if (auto* sm = batchMethodsList_->selectionModel()) {
          QSignalBlocker b2(sm);
          batchMethodsList_->setCurrentItem(firstSelectedMethod);
        } else {
          batchMethodsList_->setCurrentItem(firstSelectedMethod);
        }
      }
    }
    if (settingsBox_) {
      auto* combo = settingsBox_->findChild<QComboBox*>("batchMethodSettingsCombo");
      if (combo) {
        QString preferredShort = currentMethodShort();
        if (batchMethodsList_ && batchMethodsList_->currentItem()) {
          const QString curSelected = batchMethodsList_->currentItem()->data(Qt::UserRole).toString().trimmed();
          if (!curSelected.isEmpty()) preferredShort = curSelected;
        }

        QStringList selectedShorts;
        if (batchMethodsList_) {
          for (int i = 0; i < batchMethodsList_->count(); ++i) {
            if (auto* it = batchMethodsList_->item(i); it && it->isSelected()) {
              const QString shortName = it->data(Qt::UserRole).toString().trimmed();
              if (!shortName.isEmpty() && !selectedShorts.contains(shortName)) {
                selectedShorts << shortName;
              }
            }
          }
        }

        {
          QSignalBlocker blocker(combo);
          combo->clear();
          for (const QString& shortName : selectedShorts) {
            QString display = shortName;
            if (methodBox_) {
              int idx = methodBox_->findData(shortName);
              if (idx < 0) idx = methodBox_->findText(shortName);
              if (idx >= 0) display = methodBox_->itemText(idx);
            }
            combo->addItem(display, shortName);
          }
          combo->setEnabled((runModeBox_ && runModeBox_->currentData().toInt() == 1) && combo->count() > 0);

          int targetIndex = -1;
          for (int i = 0; i < combo->count(); ++i) {
            if (combo->itemData(i).toString().trimmed() == preferredShort) {
              targetIndex = i;
              break;
            }
          }
          if (targetIndex < 0 && combo->count() > 0) targetIndex = 0;
          if (targetIndex >= 0) combo->setCurrentIndex(targetIndex);
        }
      }
    }

    // Re-apply reconstructed dimensions after the selection widgets are updated,
    // then refresh both the per-problem batch dimension table and the main
    // Selection-area dimension control.
    batchProblemDimOverride_.clear();
    for (auto it = loadedProblemDimsByBase.begin(); it != loadedProblemDimsByBase.end(); ++it) {
      if (fixedDimForProblem(it.key()) > 0) continue;
      QList<int> dims = it.value().values();
      std::sort(dims.begin(), dims.end());
      QStringList parts;
      for (int d : dims) {
        if (d > 0) parts << QString::number(d);
      }
      if (!parts.isEmpty()) {
        batchProblemDimOverride_[it.key()] = parts;
      }
    }
    syncBatchProblemDimsTable();
    updateDimUiForProblem(currentProblemShort());

    const int summaryIdx = createOutputRunTab("System", QString(), 0);
    QWidget* summaryPage = nullptr;
    if (summaryIdx >= 0 && outputTabs_) {
      outputTabs_->setCurrentIndex(summaryIdx);
      activeOutputRunIndex_ = summaryIdx;
      if (const auto* out = outputTab(summaryIdx)) {
        summaryPage = out->page;
      }
    }

    // FIX 8c: Despite the signal-blocker fixes above there are other signal paths
    // (e.g. onOutputTabChanged calling refreshBatchSelectionView via dimSpin or
    // list widgets being restored) that may have overwritten batchCells_ with
    // data from a previously loaded experiment.  Re-apply the just-loaded values
    // and explicitly re-bind the member pointers to the new page's UI bundle
    // so that initStatsTableForBatch writes to the correct physical widget.
    batchCells_ = loadedCells;
    batchCachedProblemDims_ = loadedDims;
    g_liveBatchCsvPaths = loadedCsvPaths;
    if (summaryPage) {
      bindBatchSummaryUiForPage(summaryPage);
    }

    initStatsTableForBatch(methods, problems);

    // Make the DIM column reflect the dimensions reconstructed from the loaded
    // experiment files, even before any later table rebuilds.
    if (statsTable_) {
      for (const QString& prob : problems) {
        if (!loadedDims.contains(prob)) continue;
        const int row = statsRowByProblem_.value(prob, -1);
        if (row < 0) continue;
        if (auto* dimItem = statsTable_->item(row, 1)) {
          dimItem->setText(QString::number(loadedDims.value(prob)));
        } else {
          auto* newDimItem = new QTableWidgetItem(QString::number(loadedDims.value(prob)));
          newDimItem->setFlags(newDimItem->flags() & ~Qt::ItemIsEditable);
          statsTable_->setItem(row, 1, newDimItem);
        }
      }
    }

    for (const QString& prob : problems) {
      for (const QString& meth : methods) {
        updateStatsTableCell(prob, meth);
      }
    }
    finalizeStatsTableAfterBatch();
    rebuildStatsComparisons();

    if (statsTabs_) statsTabs_->setCurrentIndex(0);

    if (summaryPage) {
      BatchSummarySnapshot snap;
      snap.methods = methods;
      snap.problems = problems;
      // The loaded experiment in full (deselecting entries later only hides them).
      snap.allMethods = methods;
      snap.allProblems = problems;
      snap.csvPaths = loadedCsvPaths;
      batchSummaryCellsByPage_[summaryPage] = loadedCells;
      snap.cachedProblemDims = loadedDims;
      snap.problemDimOverrides = batchProblemDimOverride_;
      snap.metricComboIndex = (batchMetricCombo_ ? batchMetricCombo_->currentIndex() : -1);
      snap.aggText = (batchAggCombo_ ? batchAggCombo_->currentText() : QString("Mean"));
      snap.showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                        ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                        : true);
      snap.showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
      snap.showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
      snap.showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
      snap.manifestPath = manifestPath;   // empty for legacy folders: the next batch writes a new one
      g_batchSummarySnapshots.insert(summaryPage, snap);
      g_activeBatchSummaryPage = summaryPage;
    }

    lastRuntimeWorkingDir_ = dir.absolutePath();

    QStringList logLines;
    logLines << QString("Loaded external batch results from: %1").arg(dir.absolutePath());
    if (!manifestPath.isEmpty())
      logLines << QString("Experiment manifest: %1").arg(QFileInfo(manifestPath).fileName());
    logLines << QString("Reconstructed %1 batch cell(s).").arg(loadedCount);
    if (!loadedLines.isEmpty()) logLines << loadedLines;
    if (!skippedLines.isEmpty()) {
      logLines << "Warnings:";
      logLines << skippedLines;
    }

    appendLog(logLines.join("\n"));
    statusBar()->showMessage(QString("Loaded batch results from %1 file(s).").arg(loadedCount), 6000);
    return true;
  };

  // Sensitivity-mode counterpart to loadExternalBatchFromDirectory above.
  // Sensitivity result files are named "sensitivity_<problem>_D<dim>.csv"
  // (grid mode: all swept params combined into one file per problem) or
  // "sensitivity_<param>_<problem>_D<dim>.csv" (OAT mode: one file per
  // param+problem) -- see writeSensGroupCsv()'s caller in
  // MainWindow_ProcessSettings.cpp for the exact naming this mirrors.
  // Unlike batch's convergence CSVs, the method name is never part of the
  // filename (one sensitivity run always uses a single, fixed method), so
  // it is taken from the Selection area's current method instead. The
  // actual list of swept parameters doesn't need to be known here either
  // -- it's discovered from each CSV's own header columns by
  // tryLoadSensitivityForTab(), reused unchanged from the normal live-run
  // path (including today's fix so the Results table shows every swept
  // parameter, not just one).
  auto loadExternalSensitivityFromDirectory = [this](const QString& dirPath, const QString& /*selectedCsvPath*/) -> bool {
    QDir dir(dirPath);
    if (!dir.exists()) return false;

    const QStringList files = dir.entryList(QStringList() << "sensitivity_*_D*.csv", QDir::Files, QDir::Name);
    if (files.isEmpty()) return false;

    static const QRegularExpression re(QStringLiteral(R"(^sensitivity_(.+)_D(\d+)\.csv$)"),
                                        QRegularExpression::CaseInsensitiveOption);

    const QString method = currentMethodShort().trimmed();
    if (method.isEmpty()) {
      QMessageBox::warning(
        this,
        "Load experiment CSV",
        "Select a method in the Selection area first. Sensitivity result files "
        "don't include the method name (a sensitivity run always uses one fixed "
        "method), so the loader needs to know which method these results belong to."
      );
      return true; // handled (as an error) -- don't fall through to batch/single loading
    }

    int loadedCount = 0;
    QStringList logLines;
    for (const QString& fname : files) {
      const auto m = re.match(fname);
      if (!m.hasMatch()) continue;
      const QString problem = m.captured(1);
      const int dim = m.captured(2).toInt();
      const QString csvPath = dir.filePath(fname);

      const int outIdx = createOutputRunTab(method, problem, dim);
      if (outIdx < 0 || outIdx >= static_cast<int>(outputRuns_.size())) continue;

      outputRuns_[outIdx].sensitivityCsvPath  = csvPath;
      outputRuns_[outIdx].runtimeWorkingDir   = dirPath;
      tryLoadSensitivityForTab(outIdx);

      logLines << QString("Loaded sensitivity results: %1 + %2 (D=%3)").arg(method, problem).arg(dim);
      ++loadedCount;
    }

    if (loadedCount == 0) {
      QMessageBox::warning(
        this,
        "Load experiment CSV",
        "No sensitivity result files could be reconstructed from the selected folder."
      );
      return true;
    }

    if (outputTabs_ && outputTabs_->count() > 0) outputTabs_->setCurrentIndex(outputTabs_->count() - 1);
    appendLog(logLines.join("\n"));
    statusBar()->showMessage(QString("Loaded sensitivity results for %1 problem(s).").arg(loadedCount), 6000);
    return true;
  };

  connect(loadExperimentCsvBtn_, &QPushButton::clicked, this, [this, loadExternalBatchFromDirectory, loadExternalSensitivityFromDirectory]() {
    const QString startDir = !lastRuntimeWorkingDir_.trimmed().isEmpty()
      ? lastRuntimeWorkingDir_
      : (!projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath());

    const QString csvPath = QFileDialog::getOpenFileName(
      this,
      "Load experiment CSV",
      startDir,
      "Experiment files (*.csv *.osexp);;CSV files (*.csv);;Experiment manifests (*.osexp);;All files (*.*)"
    );
    if (csvPath.isEmpty()) return;

    // Reconstructing a batch from a folder with many experiment CSVs can
    // take a while (it runs synchronously on the UI thread, one file at a
    // time). Without visible feedback the window looks frozen even though
    // it is working correctly. Show the spinner and disable every control
    // for the duration; a small RAII guard makes the re-enable unconditional
    // -- this handler has several early "return" points below, and a guard
    // is the only way to guarantee they all restore the UI the same way a
    // single "finally" block would. (The guard itself is a plain
    // std::function holder with no MainWindow-specific logic: a LOCAL
    // struct/class defined here would NOT get implicit access to
    // MainWindow's private members just by being lexically nested inside
    // one of its member functions -- only the [this]-capturing lambdas
    // below have that access, same as any other lambda in this file.)
    struct ScopeExit {
      std::function<void()> fn;
      ~ScopeExit() { if (fn) fn(); }
    };

    auto setUiEnabled = [this](bool enabled) {
      if (busySpinner_) { if (enabled) busySpinner_->stop(); else busySpinner_->start(); }
      if (selectionBox_) selectionBox_->setEnabled(enabled);
      if (settingsBox_) settingsBox_->setEnabled(enabled);
      if (runBtn_) runBtn_->setEnabled(enabled);
      if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(enabled);
      if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(enabled);
      if (clearCsvBtn_) clearCsvBtn_->setEnabled(enabled);
    };
    setUiEnabled(false);
    QApplication::processEvents(); // let the disabled state / spinner start actually paint now
    ScopeExit busyGuard{ [setUiEnabled]() { setUiEnabled(true); } };

    const QFileInfo fi(csvPath);

    const int curMode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
    if (curMode == 2 || curMode == 3) {
      if (loadExternalSensitivityFromDirectory(fi.absolutePath(), fi.absoluteFilePath())) {
        return;
      }
    } else if (loadExternalBatchFromDirectory(fi.absolutePath(), fi.absoluteFilePath())) {
      return;
    }

    // FIX 4: The loadExternalBatchFromDirectory call above returned false, which
    // means we are about to create a plain single-convergence tab.  Before we do,
    // flush the current batch summary state to its per-page cache so it survives
    // the tab switch that follows.  Without this, switching back to an already
    // loaded batch summary tab after loading a plain CSV would find batchCells_
    // already overwritten by the new tab's (empty) state.
    if (g_activeBatchSummaryPage) {
      batchSummaryCellsByPage_[g_activeBatchSummaryPage] = batchCells_;
      if (g_batchSummarySnapshots.contains(g_activeBatchSummaryPage)) {
        BatchSummarySnapshot curSnap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
        curSnap.csvPaths          = g_liveBatchCsvPaths;
        curSnap.cachedProblemDims = batchCachedProblemDims_;
        g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, curSnap);
      }
    }

    QVector<double> iterX;
    QVector<double> evalX;
    QVector<double> y;
    QString info;
    bool hasEvalX = false;
    if (!parseConvergenceCsvFile(csvPath, iterX, evalX, y, info, &hasEvalX)) {
      QMessageBox::warning(
        this,
        "Load experiment CSV",
        QString("The selected file could not be interpreted as a convergence CSV.\n\n%1").arg(info)
      );
      return;
    }

    QString tabName = fi.completeBaseName().trimmed();
    if (tabName.isEmpty()) tabName = "Loaded CSV";

    const int tabIndex = createOutputRunTab(tabName, QString(), 0);
    if (tabIndex < 0) return;

    if (auto* tab = outputTab(tabIndex)) {
      tab->runtimeWorkingDir = fi.absolutePath();
      if (tab->page) {
        tab->page->setProperty("explicitConvergenceCsvPath", fi.absoluteFilePath());
      }
      if (tab->log) {
        tab->log->append(QString("Loaded external experiment CSV:\n%1").arg(fi.absoluteFilePath()));
        if (!info.trimmed().isEmpty()) tab->log->append(info);
      }
    }

    lastRuntimeWorkingDir_ = fi.absolutePath();
    outputTabs_->setCurrentIndex(tabIndex);
    activeOutputRunIndex_ = tabIndex;
    tryLoadConvergenceForTab(tabIndex);
    tryLoadLandscapeForTab(tabIndex);
    tryLoadSensitivityForTab(tabIndex);
    statusBar()->showMessage(QString("Loaded experiment CSV: %1").arg(fi.fileName()), 5000);
  });

  connect(reloadConvergenceBtn_, &QPushButton::clicked, this, &MainWindow::onReloadConvergence);
  connect(clearCsvBtn_, &QPushButton::clicked, this, &MainWindow::onClearCsvFiles);
  // After the user has emptied the output folder there is nothing left that
  // could be "lost" by starting another mode.
  connect(clearCsvBtn_, &QPushButton::clicked, this, [this]() {
    if (countFilesIn(runOutputFolders()) == 0) {
      for (int m = 0; m < kModeTabCount; ++m) modeHasResults_[m] = false;
    }
  });

  // item change hooks (editable settings)
  connect(runTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(stopTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(initTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(methodTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(sensitivityTable_, &QTableWidget::cellChanged, this, &MainWindow::onSensitivityCellChanged);
}


void MainWindow::applyRegionStyling() {
  // Theme-aware, scoped styling for the three main regions and their maximize buttons.
  // The palette is used so the UI remains consistent in both light and dark themes.
  auto blend = [](const QColor& a, const QColor& b, double t) {
    const int r = static_cast<int>(a.red()   + (b.red()   - a.red())   * t);
    const int g = static_cast<int>(a.green() + (b.green() - a.green()) * t);
    const int bl = static_cast<int>(a.blue() + (b.blue()  - a.blue())  * t);
    return QColor(r, g, bl);
  };

  auto rotateHue = [](QColor c, int deg) {
    QColor hsv = c.toHsv();
    int h = hsv.hue();
    if (h < 0) h = 210; // fallback for grayscale colors
    h = (h + deg) % 360;
    hsv.setHsv(h, hsv.saturation(), hsv.value());
    return hsv.toRgb();
  };

  auto clampSaturation = [](QColor c, int maxSat) {
    QColor hsv = c.toHsv();
    int h = hsv.hue();
    if (h < 0) h = 210;
    int s = hsv.saturation();
    if (s > maxSat) s = maxSat;
    hsv.setHsv(h, s, hsv.value());
    return hsv.toRgb();
  };

  auto rgba = [](const QColor& c, int a) {
    return QString("rgba(%1,%2,%3,%4)").arg(c.red()).arg(c.green()).arg(c.blue()).arg(a);
  };

  const QColor hi = palette().color(QPalette::Highlight);
  const QColor win = palette().color(QPalette::Window);

  // Slightly stronger accents (still discreet) so they remain visible.
  QColor baseAccent = blend(win, hi, 0.28);
  baseAccent = clampSaturation(baseAccent, 120);

  const QColor selAccent = clampSaturation(rotateHue(baseAccent,   0), 120);
  const QColor setAccent = clampSaturation(rotateHue(baseAccent, 120), 120);
  const QColor outAccent = clampSaturation(rotateHue(baseAccent, 240), 120);

  const QColor focusBorder = blend(win, hi, 0.38);

  const QString qss = QString(
R"(
QGroupBox#selectionBox { border: 1px solid palette(mid); border-left: 5px solid %1; border-radius: 8px; margin-top: 10px; background-color: %5; }
QGroupBox#settingsBox  { border: 1px solid palette(mid); border-left: 5px solid %2; border-radius: 8px; margin-top: 10px; background-color: %6; }
QGroupBox#outputBox    { border: 1px solid palette(mid); border-left: 5px solid %3; border-radius: 8px; margin-top: 10px; background-color: %7; }
QGroupBox#helpBox, QGroupBox#codeWizardBox { border: 1px solid palette(mid); border-left: 5px solid %2; border-radius: 8px; margin-top: 10px; background-color: %6; }

/* Main tab strip: drawn from the palette only, so pane, tabs and the group boxes inside
   always agree with the OS light/dark scheme (no style-specific tab colours). Scoped with
   the child combinator so the inner tab widgets (Settings, output tabs) keep their look. */
QTabWidget#mainTabs::pane { border: 1px solid palette(mid); border-radius: 4px; background-color: palette(window); top: -1px; }
QTabWidget#mainTabs > QTabBar::tab { background-color: palette(button); color: palette(window-text); border: 1px solid palette(mid); border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 5px 12px; margin-right: 1px; }
QTabWidget#mainTabs > QTabBar::tab:selected { background-color: palette(window); }
QTabWidget#mainTabs > QTabBar::tab:!selected:hover { background-color: palette(alternate-base); }
QTabWidget#mainTabs > QTabBar::tab:disabled { color: palette(mid); }

QSplitter::handle:vertical { background-color: palette(mid); height: 3px; margin: 1px 14px; border-radius: 1px; }

QGroupBox#selectionBox::title, QGroupBox#settingsBox::title, QGroupBox#outputBox::title, QGroupBox#helpBox::title, QGroupBox#codeWizardBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; font-weight: 600; }

QGroupBox#selectionBox[focused="true"] { border: 2px solid %4; border-left: 7px solid %1; background-color: palette(alternateBase); }
QGroupBox#settingsBox[focused="true"]  { border: 2px solid %4; border-left: 7px solid %2; background-color: palette(alternateBase); }
QGroupBox#outputBox[focused="true"]    { border: 2px solid %4; border-left: 7px solid %3; background-color: palette(alternateBase); }

/* Unify all buttons to the same visual language as the Maximize buttons. */
QPushButton { padding: 4px 10px; border: 1px solid palette(dark); border-radius: 6px; background-color: palette(button); }
QPushButton:hover { background-color: palette(alternateBase); }
QPushButton:pressed { background-color: palette(base); }
QPushButton:disabled { color: palette(mid); border: 1px solid palette(mid); background-color: palette(window); }
QPushButton:default { border: 1px solid %4; }
QPushButton[restore="true"] { border: 1px solid %4; font-weight: 600; }
)"
  ).arg(rgba(selAccent, 230),
        rgba(setAccent, 230),
        rgba(outAccent, 230),
        rgba(focusBorder, 230),
        rgba(selAccent, 34),
        rgba(setAccent, 34),
        rgba(outAccent, 34));

  if (auto* c = centralWidget()) {
    c->setStyleSheet(qss);
  }
}


void MainWindow::toggleRegionFocus(FocusArea area) {
  if (area == FocusArea::None) {
    restoreRegionLayout();
    return;
  }
  if (focusedArea_ == area) {
    restoreRegionLayout();
    return;
  }
  if (focusedArea_ != FocusArea::None) {
    restoreRegionLayout();
  }

  // Remember the user's Selection/Output split of the active mode so
  // "Restore layout" brings back exactly what was on screen.
  if (workspaceSplitter_ && selectionBox_ && outputBox_) {
    const int mode = currentModeIndex();
    if (mode >= 0 && mode < modeSplitterSizes_.size()) {
      modeSplitterSizes_[mode] = workspaceSplitter_->sizes();
    }
  }

  focusedArea_ = area;

  // Inside a run-mode tab there are only two regions: Selection and Output.
  if (area == FocusArea::Selection) {
    if (selectionBox_) selectionBox_->setVisible(true);
    if (outputBox_)    outputBox_->setVisible(false);
  } else if (area == FocusArea::Output) {
    if (selectionBox_) selectionBox_->setVisible(false);
    if (outputBox_)    outputBox_->setVisible(true);
  }

  updateRegionFocusStyling();
}

void MainWindow::restoreRegionLayout() {
  focusedArea_ = FocusArea::None;

  if (selectionBox_) selectionBox_->setVisible(true);
  if (outputBox_)    outputBox_->setVisible(true);

  const int mode = currentModeIndex();
  if (workspaceSplitter_ && mode >= 0 && mode < modeSplitterSizes_.size() &&
      modeSplitterSizes_[mode].size() >= 2) {
    workspaceSplitter_->setSizes(modeSplitterSizes_[mode]);
  }

  updateRegionFocusStyling();
}

void MainWindow::updateRegionFocusStyling() {
  const bool hasFocus = (focusedArea_ != FocusArea::None);

  if (selectionBox_) selectionBox_->setProperty("focused", focusedArea_ == FocusArea::Selection);
  if (outputBox_)    outputBox_->setProperty("focused", focusedArea_ == FocusArea::Output);

  auto refreshWidgetStyle = [](QWidget* w) {
    if (!w) return;
    if (auto* s = w->style()) {
      s->unpolish(w);
      s->polish(w);
    }
    w->update();
  };

  auto updateBtn = [&](QPushButton* b, bool activeArea) {
    if (!b) return;
    const bool restoring = hasFocus && activeArea;
    b->setProperty("restore", restoring);
    b->setText(restoring ? "Restore layout" : "Maximize");
    const QSize isz(18, 18);
    const QIcon base = style()->standardIcon(restoring ? QStyle::SP_TitleBarNormalButton : QStyle::SP_TitleBarMaxButton);
    b->setIcon(makeTintedIcon(base, QColor(0, 120, 215), isz, 2.0));
    b->setIconSize(isz);
    b->setToolTip(restoring
                    ? "Restore both areas (Selection and Output)."
                    : "Expand this area and hide the other one.");
    refreshWidgetStyle(b);
  };

  updateBtn(selectionMaxBtn_, focusedArea_ == FocusArea::Selection);
  updateBtn(outputMaxBtn_, focusedArea_ == FocusArea::Output);

  refreshWidgetStyle(selectionBox_);
  refreshWidgetStyle(outputBox_);
}

// ──────────────────────────────────────────────────────────────────────
// Tabbed main window: one tab per run mode, plus Code Wizard and Settings.
//
// Design (so that NO mode/computation logic had to change):
//  * runModeBox_ is still the single source of truth for the active run mode
//    -- every existing handler (onRunModeChanged(), updateBatchPanelVisibility(),
//    onRunClicked(), ...) keeps reading it. It is only hidden. Clicking a mode
//    tab sets it; a programmatic change of it (e.g. "Load batch selection...")
//    moves the tab.
//  * Selection + Output are ONE shared "workspace" widget that is re-parented
//    into the page of the active mode tab. The existing per-mode visibility
//    logic therefore makes each tab show exactly the Selection components that
//    belong to its mode, together with the Output area.
//  * The Output area's run tabs are tagged with the mode that created them and
//    each mode tab shows only its own runs.
//  * While a run/stage is active the other mode tabs are locked, exactly like
//    the (then disabled) "Run mode" drop-down used to be.
// ──────────────────────────────────────────────────────────────────────

int MainWindow::currentModeIndex() const {
  if (!runModeBox_) return 0;
  const int m = runModeBox_->currentData().toInt();
  return (m >= 0 && m < kModeTabCount) ? m : 0;
}

void MainWindow::assembleMainTabs(QWidget* central, QVBoxLayout* rootLay, QWidget* codeWizardPage) {
  // 1) Shared workspace: Selection above, Output below.
  // The workspace sits in a scroll area. The modes need different minimum
  // widths (e.g. the sensitivity panels are much wider than the single-run
  // form); without the scroll area, switching to a wide mode would enlarge
  // the main window. Now the window keeps its size and a scroll bar appears
  // only if the window really is narrower than the mode needs.
  auto* wsScroll = new QScrollArea(central);
  wsScroll->setWidgetResizable(true);
  wsScroll->setFrameShape(QFrame::NoFrame);
  wsScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  wsScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  wsScroll->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  wsScroll->setMinimumSize(0, 0);
  workspace_ = wsScroll;
  auto* wsInner = new QWidget();
  wsInner->setObjectName("workspaceInner");
  wsInner->setStyleSheet("QWidget#workspaceInner { background: transparent; }");
  wsScroll->setWidget(wsInner);
  auto* wsLay = new QVBoxLayout(wsInner);
  wsLay->setContentsMargins(0, 0, 0, 0);
  workspaceSplitter_ = new QSplitter(Qt::Vertical, wsInner);
  workspaceSplitter_->setChildrenCollapsible(false);
  workspaceSplitter_->addWidget(selectionBox_);
  workspaceSplitter_->addWidget(outputBox_);
  workspaceSplitter_->setStretchFactor(0, 0);   // extra window height goes to Output
  workspaceSplitter_->setStretchFactor(1, 1);
  wsLay->addWidget(workspaceSplitter_, 1);
  modeSplitterSizes_.resize(kModeTabCount);

  // 2) Main tabs.
  auto* mainTabWidget = new MainTabWidget(central);
  mainTabs_ = mainTabWidget;
  mainTabWidget->mainTabBar()->setFirstToolTab(kCodeWizardTab);
  mainTabs_->setObjectName("mainTabs");
  mainTabs_->setUsesScrollButtons(true);
  mainTabs_->setElideMode(Qt::ElideNone);
  mainTabs_->tabBar()->setExpanding(false);

  // Short tab captions; the tooltip carries the full mode name (the original
  // drop-down text), so there is still one place that defines the names.
  static const char* const kTabCaptions[kModeTabCount] = {
    "Single run",
    "Batch run",
    "Method sensitivity",
    "Problem sensitivity",
    "Multi-objective",
    "Meta-dataset pipeline",
    "NN landscape",
    "Portfolio"
  };
  modePages_.clear();
  for (int mode = 0; mode < kModeTabCount; ++mode) {
    auto* page = new QWidget(mainTabs_);
    auto* pageLay = new QVBoxLayout(page);
    pageLay->setContentsMargins(6, 6, 6, 6);
    modePages_.push_back(page);
    const int tab = mainTabs_->addTab(page, QString::fromLatin1(kTabCaptions[mode]));
    const int comboIdx = runModeBox_ ? runModeBox_->findData(mode) : -1;
    if (comboIdx >= 0) mainTabs_->setTabToolTip(tab, runModeBox_->itemText(comboIdx));
  }

  mainTabs_->addTab(codeWizardPage, "Code Wizard");
  mainTabs_->setTabToolTip(kCodeWizardTab, "Generate, delete and open method/problem source files");

  // The Settings area (with all of its own inner tabs) lives in the Settings tab.
  auto* settingsPage = new QWidget(mainTabs_);
  auto* settingsPageLay = new QVBoxLayout(settingsPage);
  settingsPageLay->setContentsMargins(6, 6, 6, 6);
  settingsPageLay->addWidget(settingsBox_);
  mainTabs_->addTab(settingsPage, "Settings");
  mainTabs_->setTabToolTip(kSettingsTab, "Global, termination, initialization, method and sensitivity settings");

  mainTabs_->addTab(buildHelpPage(), "Help");
  mainTabs_->setTabToolTip(kHelpTab, "About OptimSolution, user guide, project page and environment information");

  rootLay->addWidget(mainTabs_, 1);

  // 3) Initial state: the workspace sits in the tab of the current run mode.
  shownMode_ = currentModeIndex();
  placeWorkspaceForMode(shownMode_);
  mainTabs_->setCurrentIndex(shownMode_);
  lastMainTab_ = shownMode_;
  connect(mainTabs_, &QTabWidget::currentChanged, this, &MainWindow::onMainTabChanged);

  // 4) Mirror enable/disable state (see eventFilter()).
  if (runModeBox_)   runModeBox_->installEventFilter(this);
  if (selectionBox_) selectionBox_->installEventFilter(this);
  if (settingsBox_)  settingsBox_->installEventFilter(this);

  // 5) Not every mode signals its start/end through those widgets (e.g. a
  // Multi-objective run disables nothing; some stages only start a process),
  // so also re-check the main-tab lock a few times per second. It is cheap:
  // isRunBusy() only reads a few flags, and tabs are only touched on change.
  auto* tabLockTimer = new QTimer(this);
  tabLockTimer->setInterval(250);
  connect(tabLockTimer, &QTimer::timeout, this, &MainWindow::updateModeTabsLock);
  tabLockTimer->start();
}

// ──────────────────────────────────────────────────────────────────────
// "Help" tab: author, project page, user guide, short orientation and the
// environment details that are useful in a bug report. Presentation only.
// ──────────────────────────────────────────────────────────────────────
QString MainWindow::helpUserGuidePath() const {
  const QString rel = "docs/optimsolution_manual_en.pdf";
  QStringList starts;
  if (!projectRoot_.isEmpty()) starts << projectRoot_;
  starts << QCoreApplication::applicationDirPath() << QDir::currentPath();
  if (!settingsPath_.isEmpty()) starts << QFileInfo(settingsPath_).absolutePath();
  for (const QString& st : starts) {
    const QString hit = PathUtils::findExistingFileUpwards(st, rel);
    if (!hit.isEmpty() && QFileInfo::exists(hit)) return hit;
  }
  return QString();
}

QString MainWindow::helpSystemInfoText() const {
  QString runDir;
  {
    QString baseWd = QCoreApplication::applicationDirPath();
    if (!settingsPath_.isEmpty()) {
      const QFileInfo fi(settingsPath_);
      if (fi.exists()) baseWd = fi.absolutePath();
    }
    runDir = QDir(baseWd).filePath("optimsolution_gui_run");
  }
  const QString guide = helpUserGuidePath();
  QStringList l;
  l << QString("OptimSolution GUI: v%1").arg(kGuiVersion);
  l << QString("Qt (runtime): %1").arg(QString::fromLatin1(qVersion()));
  l << QString("Operating system: %1").arg(QSysInfo::prettyProductName());
  l << QString("CPU architecture: %1").arg(QSysInfo::currentCpuArchitecture());
  l << QString("Application folder: %1").arg(QDir::toNativeSeparators(QCoreApplication::applicationDirPath()));
  l << QString("Project folder: %1").arg(QDir::toNativeSeparators(projectRoot_));
  l << QString("Settings file: %1").arg(settingsPath_.isEmpty() ? QString("(none found)") : QDir::toNativeSeparators(settingsPath_));
  l << QString("Output folder: %1").arg(QDir::toNativeSeparators(runDir));
  l << QString("User guide: %1").arg(guide.isEmpty() ? QString("docs/optimsolution_manual_en.pdf (not found)") : QDir::toNativeSeparators(guide));
  return l.join("\n");
}

QWidget* MainWindow::buildHelpPage() {
  auto* scroll = new QScrollArea(mainTabs_);
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  auto* inner = new QWidget();
  inner->setObjectName("helpInner");
  inner->setStyleSheet("QWidget#helpInner { background: transparent; }");
  scroll->setWidget(inner);
  auto* lay = new QVBoxLayout(inner);
  lay->setContentsMargins(6, 6, 6, 6);
  lay->setSpacing(10);

  auto makeBox = [&](const QString& title) {
    auto* box = new QGroupBox(title, inner);
    box->setObjectName("helpBox");
    auto* v = new QVBoxLayout(box);
    v->setContentsMargins(14, 18, 14, 12);
    v->setSpacing(6);
    lay->addWidget(box);
    return v;
  };
  auto makeLabel = [](const QString& text, QWidget* parent, bool rich = false) {
    auto* lbl = new QLabel(text, parent);
    lbl->setWordWrap(true);
    if (rich) {
      lbl->setTextFormat(Qt::RichText);
      lbl->setTextInteractionFlags(Qt::TextBrowserInteraction);
      lbl->setOpenExternalLinks(true);
    } else {
      lbl->setTextInteractionFlags(Qt::TextSelectableByMouse);
    }
    return lbl;
  };

  // ── About ────────────────────────────────────────────────────────────
  {
    auto* v = makeBox("About");
    auto* title = new QLabel(QString("<span style='font-size:16pt; font-weight:600;'>OptimSolution</span>"
                                     "&nbsp;&nbsp;<span style='color:gray;'>GUI v%1</span>").arg(kGuiVersion), inner);
    title->setTextFormat(Qt::RichText);
    v->addWidget(title);
    v->addWidget(makeLabel("A framework for continuous optimization: run and compare optimization methods on benchmark "
                           "problems, analyse parameter sensitivity, and study algorithm behaviour.", inner));
    auto* grid = new QGridLayout();
    grid->setHorizontalSpacing(14);
    grid->setVerticalSpacing(4);
    grid->setColumnStretch(1, 1);
    grid->addWidget(new QLabel("<b>Author</b>", inner), 0, 0, Qt::AlignTop);
    grid->addWidget(makeLabel("Vasileios Charilogis", inner), 0, 1);
    grid->addWidget(new QLabel("<b>Project page</b>", inner), 1, 0, Qt::AlignTop);
    grid->addWidget(makeLabel("<a href=\"https://github.com/charilog/optimsolution\">https://github.com/charilog/optimsolution</a>", inner, true), 1, 1);
    grid->addWidget(new QLabel("<b>Report a problem</b>", inner), 2, 0, Qt::AlignTop);
    grid->addWidget(makeLabel("<a href=\"https://github.com/charilog/optimsolution/issues\">https://github.com/charilog/optimsolution/issues</a>", inner, true), 2, 1);
    v->addLayout(grid);
  }

  // ── User guide ───────────────────────────────────────────────────────
  {
    auto* v = makeBox("User guide");
    v->addWidget(makeLabel("The user guide is the PDF file <b>optimsolution_manual_en.pdf</b> in the <b>docs</b> folder of the project.", inner, true));
    auto* row = new QHBoxLayout();
    auto* openBtn = new QPushButton("Open user guide (optimsolution_manual_en.pdf)", inner);
    openBtn->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
    openBtn->setToolTip("Open docs/optimsolution_manual_en.pdf in the default PDF viewer.");
    auto* docsBtn = new QPushButton("Open docs folder", inner);
    docsBtn->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    docsBtn->setToolTip("Open the docs folder (manuals, installation notes, examples) in the file browser.");
    row->addWidget(openBtn);
    row->addWidget(docsBtn);
    row->addStretch(1);
    v->addLayout(row);

    connect(openBtn, &QPushButton::clicked, this, [this]() {
      const QString pdf = helpUserGuidePath();
      if (pdf.isEmpty()) {
        QMessageBox::warning(this, "User guide not found",
          "The file docs/optimsolution_manual_en.pdf was not found.\n\n"
          "It is expected in the docs folder of the project (next to the src and gui folders).");
        return;
      }
      if (!QDesktopServices::openUrl(QUrl::fromLocalFile(pdf))) {
        QMessageBox::warning(this, "Cannot open the user guide",
          QString("The file exists but could not be opened. Is a PDF viewer installed?\n\n%1")
            .arg(QDir::toNativeSeparators(pdf)));
      }
    });
    connect(docsBtn, &QPushButton::clicked, this, [this]() {
      QString dir;
      const QString pdf = helpUserGuidePath();
      if (!pdf.isEmpty()) dir = QFileInfo(pdf).absolutePath();
      else {
        const QString hit = PathUtils::findExistingFileUpwards(projectRoot_, "docs");
        if (!hit.isEmpty() && QFileInfo(hit).isDir()) dir = hit;
        else if (QFileInfo(QDir(projectRoot_).filePath("docs")).isDir()) dir = QDir(projectRoot_).filePath("docs");
      }
      if (dir.isEmpty()) {
        QMessageBox::warning(this, "Docs folder not found", "The docs folder of the project could not be found.");
        return;
      }
      QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    });
  }

  // ── Orientation ──────────────────────────────────────────────────────
  {
    auto* v = makeBox("What the tabs do");
    v->addWidget(makeLabel(
      "<table cellspacing='0' cellpadding='2'>"
      "<tr><td><b>Single run</b></td><td>&nbsp;&nbsp;One method on one problem, with convergence, distribution and landscape plots.</td></tr>"
      "<tr><td><b>Batch run</b></td><td>&nbsp;&nbsp;Several methods on several problems, with summary tables and statistical comparisons.</td></tr>"
      "<tr><td><b>Method sensitivity</b></td><td>&nbsp;&nbsp;How the parameters of a method influence its results.</td></tr>"
      "<tr><td><b>Problem sensitivity</b></td><td>&nbsp;&nbsp;How the parameters of a problem influence the results.</td></tr>"
      "<tr><td><b>Multi-objective</b></td><td>&nbsp;&nbsp;Multi-objective optimization and the resulting Pareto front.</td></tr>"
      "<tr><td><b>Meta-dataset pipeline</b></td><td>&nbsp;&nbsp;Batch runs, landscape (ELA) features and machine-learning models in stages.</td></tr>"
      "<tr><td><b>NN landscape</b></td><td>&nbsp;&nbsp;Neural-network loss landscape and generalization-gap experiments.</td></tr>"
      "<tr><td><b>Portfolio</b></td><td>&nbsp;&nbsp;Dynamic switching between several methods during a run.</td></tr>"
      "<tr><td><b>Code Wizard</b></td><td>&nbsp;&nbsp;Create, delete and open the source files of methods and problems.</td></tr>"
      "<tr><td><b>Settings</b></td><td>&nbsp;&nbsp;Global, termination, initialization, method and sensitivity settings (the .cfg file).</td></tr>"
      "</table>", inner, true));
  }

  // ── Good to know ─────────────────────────────────────────────────────
  {
    auto* v = makeBox("Good to know");
    v->addWidget(makeLabel(
      "<ul style='margin-left:-18px;'>"
      "<li>While a run is active the other mode tabs are locked; stop the run or wait until it finishes to switch.</li>"
      "<li>Each mode tab shows only its own results. All modes write to the same output folder, so starting an "
      "experiment in a different mode asks for confirmation and then <b>deletes the earlier results</b> "
      "(output tabs and files).</li>"
      "<li><b>Delete output files</b> empties the whole output folder; <b>Open output folder</b> shows it in the file browser.</li>"
      "<li>The <b>Maximize</b> buttons expand Selection or Output; press again to restore both.</li>"
      "<li>Changes in the Settings tab are used by the next run; use <b>Save</b> there to keep them in the .cfg file.</li>"
      "</ul>", inner, true));
  }

  // ── Environment ──────────────────────────────────────────────────────
  {
    auto* v = makeBox("Environment");
    auto* info = makeLabel(helpSystemInfoText(), inner);
    info->setObjectName("helpSystemInfo");
    {
      QFont f = info->font();
      f.setFamily("Monospace");
      f.setStyleHint(QFont::Monospace);
      info->setFont(f);
    }
    v->addWidget(info);
    auto* row = new QHBoxLayout();
    auto* copyBtn = new QPushButton("Copy to clipboard", inner);
    copyBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    copyBtn->setToolTip("Copy this information, e.g. to paste it into a bug report.");
    row->addWidget(copyBtn);
    row->addStretch(1);
    v->addLayout(row);
    connect(copyBtn, &QPushButton::clicked, this, [this]() {
      QGuiApplication::clipboard()->setText(helpSystemInfoText());
      statusBar()->showMessage("Environment information copied to the clipboard.", 3000);
    });
    // Paths can change while the program runs (other settings file chosen, ...).
    connect(mainTabs_, &QTabWidget::currentChanged, this, [this, info](int i) {
      if (i == kHelpTab && info) info->setText(helpSystemInfoText());
    });
  }

  lay->addStretch(1);
  return scroll;
}


// Switching tabs must never change the width of the window. The layout is
// built so that no mode forces a wider window (the workspace sits in a scroll
// area); this is the safety net: if anything -- a style, a font, a late
// layout pass -- still widens the window as a side effect of a tab switch, it
// is put back to the width it had before. A user's own resizing is never
// touched (only a *growth* within a short time after the switch is undone),
// and maximized / full-screen windows are left alone.
void MainWindow::keepWindowWidthAcrossSwitch(int widthBefore) {
  if (isMaximized() || isFullScreen() || widthBefore <= 0) return;
  const QPointer<MainWindow> self(this);
  for (const int delay : {0, 60, 250}) {
    QTimer::singleShot(delay, this, [self, widthBefore]() {
      if (!self || self->isMaximized() || self->isFullScreen()) return;
      if (self->width() > widthBefore) {
        const int target = std::max(widthBefore, self->minimumSizeHint().width());
        if (target < self->width()) self->resize(target, self->height());
      }
    });
  }
}

void MainWindow::onMainTabChanged(int tabIndex) {
  if (mainTabRevertGuard_ || tabIndex < 0) return;
  keepWindowWidthAcrossSwitch(width());

  if (tabIndex < kModeTabCount) {
    const int curMode = currentModeIndex();
    // Same rule the (then disabled) drop-down enforced: no mode switch while a
    // run or pipeline stage is active.
    if (tabIndex != curMode && isRunBusy()) {
      mainTabRevertGuard_ = true;
      mainTabs_->setCurrentIndex(lastMainTab_);
      mainTabRevertGuard_ = false;
      statusBar()->showMessage(
        "The run mode cannot be changed while a run is active. Stop it or wait until it finishes.", 5000);
      return;
    }
    // Move the workspace / output tabs first, then flip the (hidden) run-mode
    // combo, which fires the unchanged onRunModeChanged() logic.
    onRunModeComboChangedBefore(tabIndex);
    if (runModeBox_ && curMode != tabIndex) {
      const int comboIdx = runModeBox_->findData(tabIndex);
      if (comboIdx >= 0) runModeBox_->setCurrentIndex(comboIdx);
    }
  }
  lastMainTab_ = tabIndex;
  updateModeTabsLock();
}

void MainWindow::onRunModeComboChangedBefore(int mode) {
  if (mode < 0 || mode >= kModeTabCount || mode == shownMode_) return;
  keepWindowWidthAcrossSwitch(width());

  // Leaving shownMode_: remember its Selection/Output split and which of its
  // own output tabs was showing.
  const int leaving = shownMode_;
  if (workspaceSplitter_ && focusedArea_ == FocusArea::None &&
      leaving >= 0 && leaving < modeSplitterSizes_.size()) {
    modeSplitterSizes_[leaving] = workspaceSplitter_->sizes();
  }
  if (outputTabs_) {
    if (QWidget* cur = outputTabs_->currentWidget()) {
      const QVariant tag = cur->property("optimMode");
      if (tag.isValid() && tag.toInt() == leaving) modeLastOutputPage_[leaving] = cur;
    }
  }

  // A maximized Selection/Output belongs to the mode being left.
  if (focusedArea_ != FocusArea::None) {
    focusedArea_ = FocusArea::None;
    if (selectionBox_) selectionBox_->setVisible(true);
    if (outputBox_)    outputBox_->setVisible(true);
    updateRegionFocusStyling();
  }

  filterOutputTabsForMode(mode);
  placeWorkspaceForMode(mode);

  // The status line ("Completed. Total time: ...") and the progress bar are
  // shared widgets, but the result they describe belongs to the mode that
  // produced it: remember it for the mode being left and show the new mode's
  // own state (idle if it has not run yet). Never touched while a run is in
  // flight, so live progress is not disturbed.
  if (!isRunBusy()) {
    if (outputStatusLbl_) {
      if (leaving >= 0 && leaving < kModeTabCount) modeStatusText_[leaving] = outputStatusLbl_->text();
      outputStatusLbl_->setText(modeStatusText_[mode].isEmpty() ? QString("Idle.") : modeStatusText_[mode]);
    }
    if (outputElapsedLbl_) outputElapsedLbl_->setVisible(false);
    if (progressWidget_) progressWidget_->setVisible(false);
    if (outputProgressBar_) {
      outputProgressBar_->setRange(0, 100);
      outputProgressBar_->setValue(0);
      outputProgressBar_->setFormat("Idle");
    }
  }
  shownMode_ = mode;
}

void MainWindow::onRunModeComboChangedAfter(int mode) {
  if (mode < 0 || mode >= kModeTabCount) return;
  applyModeSpecificSelectionUi(mode);

  // Programmatic mode change (e.g. "Load batch selection..."): bring the
  // visible tab along, silently.
  if (mainTabs_ && mainTabs_->currentIndex() < kModeTabCount && mainTabs_->currentIndex() != mode) {
    mainTabRevertGuard_ = true;
    mainTabs_->setCurrentIndex(mode);
    mainTabRevertGuard_ = false;
    lastMainTab_ = mode;
  }
  applyModeWorkspaceSizes(mode);
  updateModeTabsLock();
}

void MainWindow::placeWorkspaceForMode(int mode) {
  if (!workspace_ || mode < 0 || mode >= modePages_.size()) return;
  QWidget* page = modePages_[mode];
  if (!page || !page->layout()) return;
  if (workspace_->parentWidget() == page) return;
  if (QWidget* oldParent = workspace_->parentWidget()) {
    if (QLayout* oldLay = oldParent->layout()) oldLay->removeWidget(workspace_);
  }
  page->layout()->addWidget(workspace_);   // re-parents
  workspace_->show();
}

// The Selection area is shared by all modes. updateBatchPanelVisibility()
// (unchanged) already shows only the components of the active mode; this adds
// the few remaining rows that do not apply to a mode.
void MainWindow::applyModeSpecificSelectionUi(int mode) {
  // The run modes are tabs now -- the combo box must never reappear.
  if (runModeBox_) runModeBox_->setVisible(false);
  if (selectionForm_) {
    if (auto* lbl = selectionForm_->labelForField(runModeBox_)) lbl->setVisible(false);
  }

  const bool isMultiObjective = (mode == 4);   // own panel; ignores [global].runs
  const bool isNNLandscape    = (mode == 6);   // own dataset/optimizer lists

  // "Runs" (read-only copy of [global].runs): not used by Multi-objective runs.
  if (auto* runsLbl = findChild<QLabel*>("selectionRunsLbl")) {
    runsLbl->setVisible(!isMultiObjective);
    if (selectionForm_) {
      if (auto* lbl = selectionForm_->labelForField(runsLbl)) lbl->setVisible(!isMultiObjective);
    }
  }
  // "Refresh lists" reloads the factory method/problem lists, which the
  // Multi-objective and NN-landscape panels do not use.
  if (refreshBtn_) refreshBtn_->setVisible(!isMultiObjective && !isNNLandscape);
}

// Default Selection/Output proportions per mode (until the user drags the
// splitter): compact Selection for the form-only modes, a taller one where the
// Selection area holds lists.
void MainWindow::applyModeWorkspaceSizes(int mode) {
  if (!workspaceSplitter_ || mode < 0 || mode >= kModeTabCount) return;
  QTimer::singleShot(0, this, [this, mode]() {
    if (!workspaceSplitter_ || !selectionBox_ || !outputBox_) return;
    if (currentModeIndex() != mode || focusedArea_ != FocusArea::None) return;
    const int total = workspaceSplitter_->height() - workspaceSplitter_->handleWidth();
    if (total <= 0) return;

    int sel = 0;
    const QVector<int> saved = (mode < modeSplitterSizes_.size()) ? modeSplitterSizes_[mode] : QVector<int>();
    if (saved.size() >= 2 && (saved[0] + saved[1]) > 0) {
      sel = int(double(total) * double(saved[0]) / double(saved[0] + saved[1]));
    } else {
      // Form-only modes (single run, problem sensitivity, multi-objective):
      // exactly as tall as their content. List modes (batch, method
      // sensitivity, pipeline, portfolio): ~40% so the Output keeps most of
      // the room; the NN-landscape form is tall, so it gets ~60%.
      const int hint = selectionBox_->sizeHint().height();
      const bool listMode = (mode == 1 || mode == 2 || mode == 5 || mode == 7);
      if (mode == 6)       sel = std::min(hint, total * 60 / 100);
      else if (listMode)   sel = std::min(hint, total * 40 / 100);
      else                 sel = hint;
    }
    const int minSel = selectionBox_->minimumSizeHint().height();
    const int minOut = outputBox_->minimumSizeHint().height();
    sel = std::max(sel, minSel);
    sel = std::min(sel, std::max(minSel, total - minOut));
    workspaceSplitter_->setSizes({sel, std::max(1, total - sel)});
  });
}

// Each run-mode tab shows only the run tabs that its own mode created. Tabs of
// other modes are hidden (never removed, so every tab index -- and with it the
// index-paired outputRuns_ bookkeeping -- stays exactly as it was).
void MainWindow::filterOutputTabsForMode(int mode) {
  if (!outputTabs_ || outputFilterGuard_) return;
  QScopedValueRollback<bool> guard(outputFilterGuard_, true);

  const int n = outputTabs_->count();
  auto belongs = [&](int i) {
    QWidget* w = outputTabs_->widget(i);
    const QVariant tag = w ? w->property("optimMode") : QVariant();
    return !tag.isValid() || tag.toInt() == mode;   // untagged => always shown
  };

  // 1) Reveal this mode's tabs.
  for (int i = 0; i < n; ++i) {
    if (belongs(i) && !outputTabs_->isTabVisible(i)) outputTabs_->setTabVisible(i, true);
  }

  // 2) Make sure the current tab is one of this mode's (the one that was
  //    showing when the mode was left, else its newest; none if it has none).
  const int cur = outputTabs_->currentIndex();
  if (!(cur >= 0 && cur < n && belongs(cur))) {
    int target = -1;
    if (QWidget* last = modeLastOutputPage_[mode]) {
      const int i = outputTabs_->indexOf(last);
      if (i >= 0 && belongs(i)) target = i;
    }
    if (target < 0) {
      for (int i = n - 1; i >= 0; --i) {
        if (belongs(i)) { target = i; break; }
      }
    }
    outputTabs_->setCurrentIndex(target);
  }

  // 3) Hide the other modes' tabs.
  for (int i = 0; i < n; ++i) {
    if (!belongs(i) && outputTabs_->isTabVisible(i)) outputTabs_->setTabVisible(i, false);
  }

  // 4) Keep "the active run tab" consistent with what is now shown: none for a
  //    mode that has no run tab yet (exactly the original empty state), else the
  //    visible current one.
  activeOutputRunIndex_ = outputTabs_->currentIndex();   // -1 when nothing of this mode is current
}

// ---------------------------------------------------------------------------
// "A new experiment in another mode discards the earlier ones"
//
// All modes write into the same optimsolution_gui_run folder, so results of
// different modes cannot be kept side by side. Before a run/stage is started
// in mode M while results of another mode exist, the user is warned; if they
// agree, the earlier results (output tabs + files) are removed and the new
// experiment starts. The existing run code is not modified: this wraps the
// start buttons only.
//
// The old files are first MOVED aside (not deleted) so that the new run does
// not see them (e.g. as "cached" batch jobs); they are deleted for good only
// once the new run has really started, and are moved back if it did not start
// (e.g. the run was refused because nothing is selected).
// ---------------------------------------------------------------------------
QStringList MainWindow::runOutputFolders() const {
  QSet<QString> roots;
  auto addRoot = [&roots](const QString& path) {
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()) return;
    const QFileInfo fi(trimmed);
    const QString dirPath = fi.exists() ? (fi.isDir() ? fi.absoluteFilePath() : fi.absolutePath()) : trimmed;
    const QDir dir(dirPath);
    if (dir.dirName().compare("optimsolution_gui_run", Qt::CaseInsensitive) != 0) return;
    if (!dir.exists()) return;
    roots.insert(QDir::cleanPath(dir.absolutePath()));
  };
  addRoot(lastRuntimeWorkingDir_);
  for (const auto& r : outputRuns_) {
    if (r.runStartMsecsUtc > 0) addRoot(r.runtimeWorkingDir);   // started by this GUI, not a loaded folder
  }
  addRoot(QDir(QCoreApplication::applicationDirPath()).filePath("optimsolution_gui_run"));
  if (!settingsPath_.trimmed().isEmpty()) {
    const QFileInfo fi(settingsPath_);
    if (fi.exists()) addRoot(QDir(fi.absolutePath()).filePath("optimsolution_gui_run"));
  }
  QStringList list = roots.values();
  std::sort(list.begin(), list.end(), [](const QString& a, const QString& b) { return a.toLower() < b.toLower(); });
  return list;
}

int MainWindow::countFilesIn(const QStringList& folders) const {
  int n = 0;
  for (const QString& f : folders) {
    QDirIterator it(f, QDir::Files | QDir::Hidden | QDir::System, QDirIterator::Subdirectories);
    while (it.hasNext()) { it.next(); ++n; }
  }
  return n;
}

bool MainWindow::guardedStart(int mode, const std::function<void()>& starter) {
  // While something is running the same button means "Stop" (or is refused by
  // the run code): never prompt, never touch files.
  if (isRunBusy() || mode < 0 || mode >= kModeTabCount || !outputTabs_) {
    starter();
    return true;
  }

  // Which other modes still have results?
  QMap<int, int> tabsPerMode;
  QVector<QPointer<QWidget>> oldPages;
  for (int i = 0; i < outputTabs_->count(); ++i) {
    QWidget* pg = outputTabs_->widget(i);
    if (!pg) continue;
    const QVariant tag = pg->property("optimMode");
    if (!tag.isValid()) continue;
    const int m = tag.toInt();
    if (m < 0 || m >= kModeTabCount || m == mode) continue;
    tabsPerMode[m] += 1;
    oldPages.push_back(pg);
  }
  const QStringList folders = runOutputFolders();
  const bool filesExist = countFilesIn(folders) > 0;
  QSet<int> otherModes;
  for (auto it = tabsPerMode.cbegin(); it != tabsPerMode.cend(); ++it) otherModes.insert(it.key());
  if (filesExist) {
    for (int m = 0; m < kModeTabCount; ++m) {
      if (m != mode && modeHasResults_[m]) otherModes.insert(m);
    }
  }

  auto startAndRemember = [&]() {
    const int tabsBefore = int(outputRuns_.size());
    starter();
    if (isRunBusy() || int(outputRuns_.size()) != tabsBefore) modeHasResults_[mode] = true;
  };

  if (otherModes.isEmpty()) {
    startAndRemember();
    return true;
  }

  // ---- Warn --------------------------------------------------------------
  QStringList modeNames;
  QList<int> sorted = otherModes.values();
  std::sort(sorted.begin(), sorted.end());
  for (int m : sorted) {
    const int n = tabsPerMode.value(m, 0);
    modeNames << QString::fromUtf8("  • %1%2")
                   .arg(mainTabs_ ? mainTabs_->tabText(m) : QString::number(m))
                   .arg(n > 0 ? QString(" (%1 output tab(s))").arg(n) : QString(" (output files)"));
  }
  QString text = QString("Experiments from another mode are still in the Output area:\n\n%1\n\n"
                         "If you continue with the new experiment, these earlier experiments will be LOST: "
                         "their output tabs will be closed and their output files will be permanently deleted")
                   .arg(modeNames.join("\n"));
  if (!folders.isEmpty()) text += QString(" from:\n%1").arg(folders.join("\n"));
  text += ".\n\nThe new experiment is then started.";

  QMessageBox box(QMessageBox::Warning, "Experiments from another mode exist", text, QMessageBox::NoButton, this);
  QPushButton* goBtn = box.addButton("Delete old results and run", QMessageBox::DestructiveRole);
  QPushButton* cancelBtn = box.addButton("Cancel", QMessageBox::RejectRole);
  box.setDefaultButton(cancelBtn);
  box.setEscapeButton(cancelBtn);
  box.exec();
  if (box.clickedButton() != goBtn) return false;

  // ---- Move the old files aside -----------------------------------------
  struct Stash { QString root; QString dir; QStringList names; };
  QVector<Stash> stashes;
  auto rollback = [&stashes]() {
    for (int i = stashes.size() - 1; i >= 0; --i) {
      const Stash& st = stashes[i];
      for (const QString& n : st.names)
        QDir().rename(QDir(st.dir).filePath(n), QDir(st.root).filePath(n));
      QDir().rmdir(st.dir);
    }
  };
  bool moveFailed = false;
  for (const QString& root : folders) {
    const QStringList entries = QDir(root).entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System);
    if (entries.isEmpty()) continue;
    QString stashDir = root + "_discard_tmp";
    if (QFileInfo::exists(stashDir)) stashDir += "_" + QString::number(QDateTime::currentMSecsSinceEpoch());
    if (!QDir().mkpath(stashDir)) { moveFailed = true; break; }
    stashes.push_back({root, stashDir, {}});
    for (const QString& n : entries) {
      if (!QDir().rename(QDir(root).filePath(n), QDir(stashDir).filePath(n))) { moveFailed = true; break; }
      stashes.last().names << n;
    }
    if (moveFailed) break;
  }
  if (moveFailed) {
    rollback();
    QMessageBox::warning(this, "Cannot remove the earlier results",
                         "The output files of the earlier experiments could not be removed (they may be open in "
                         "another program, e.g. a spreadsheet or a file explorer window).\n\n"
                         "Nothing was deleted and the new experiment was not started.");
    return false;
  }

  // In-memory batch caches describe the files that were just moved away; the
  // new run must not treat them as results that still exist.
  batchCells_.clear();
  batchCachedProblemDims_.clear();
  g_liveBatchCsvPaths.clear();

  // ---- Start the new experiment -----------------------------------------
  const int tabsBefore = int(outputRuns_.size());
  starter();
  const bool started = isRunBusy() || int(outputRuns_.size()) != tabsBefore ||
                       countFilesIn(runOutputFolders()) > 0;

  if (!started) {
    // The run was refused (nothing selected, invalid input, ...): nothing was
    // lost, put everything back.
    rollback();
    return false;
  }

  // ---- The new experiment is running: drop the old one for good ----------
  bool leftovers = false;
  for (const Stash& st : stashes) {
    if (!QDir(st.dir).removeRecursively()) leftovers = true;
  }
  for (const QPointer<QWidget>& pg : oldPages) {
    if (!pg) continue;
    const int idx = outputTabs_->indexOf(pg);
    if (idx >= 0) onOutputTabCloseRequested(idx);
  }
  for (int m = 0; m < kModeTabCount; ++m) modeHasResults_[m] = false;
  for (int m = 0; m < kModeTabCount; ++m) if (otherModes.contains(m)) modeStatusText_[m].clear();
  modeHasResults_[mode] = true;

  const QString msg = leftovers
    ? QString("Earlier experiments removed; some old files could not be deleted (see the *_discard_tmp folder next to optimsolution_gui_run).")
    : QString("Earlier experiments from other mode(s) were deleted; the new experiment is running.");
  statusBar()->showMessage(msg, 8000);
  return true;
}

bool MainWindow::isRunBusy() const {
  // Single/Batch/Sensitivity/Portfolio runs and the pipeline / NN-landscape
  // stages all disable the Selection area (and Batch/NN also the run-mode
  // combo) for as long as they are active.
  if (selectionBox_ && !selectionBox_->isEnabled()) return true;
  if (runModeBox_ && !runModeBox_->isEnabled()) return true;
  // Multi-objective runs do not disable anything; check their process directly.
  if (moProc_ && moProc_->state() != QProcess::NotRunning) return true;
  if (proc_ && proc_->state() != QProcess::NotRunning) return true;
  if (batchActive_) return true;
  return false;
}

void MainWindow::updateModeTabsLock() {
  if (!mainTabs_) return;
  // While an experiment of ANY mode is running, every main tab is disabled
  // except
  //   * the tab of the running mode (it holds the Output area with the
  //     progress, the log and the Stop button), and
  //   * Help (read-only).
  // Code Wizard and Settings are disabled too: editing settings or generating
  // / deleting method or problem sources in the middle of a run would change
  // what the running experiment reads.
  const bool busy = isRunBusy();
  const int cur = currentModeIndex();
  for (int tab = 0; tab < mainTabs_->count(); ++tab) {
    const bool enabled = !busy || tab == cur || tab == kHelpTab;
    if (mainTabs_->isTabEnabled(tab) != enabled) mainTabs_->setTabEnabled(tab, enabled);
  }
}

void MainWindow::changeEvent(QEvent* e) {
  QMainWindow::changeEvent(e);
  // The region accents are derived from the palette once; redo them when the
  // OS switches between light and dark mode while the program is running.
  if (e && (e->type() == QEvent::ApplicationPaletteChange || e->type() == QEvent::PaletteChange)) {
    static bool restyling = false;
    if (!restyling && centralWidget()) {
      restyling = true;
      applyRegionStyling();
      restyling = false;
    }
  }
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
  if (event && event->type() == QEvent::EnabledChange) {
    if (watched == runModeBox_ || watched == selectionBox_) {
      updateModeTabsLock();
    } else if (watched == settingsBox_ && codeWizardPage_ && settingsBox_) {
      // The Code Wizard used to be a child of the Settings area and was
      // therefore disabled together with it while a run is active.
      codeWizardPage_->setEnabled(settingsBox_->isEnabled());
    }
  }
  return QMainWindow::eventFilter(watched, event);
}

} // namespace optimsolution_gui
