// MainWindow_Tabs.cpp
// Factory lists, method/problem selection, parameter tables (method/global/stop/init/sensitivity), output run tabs and convergence loading.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"
#include "stats.h"

namespace optimsolution_gui {
void MainWindow::onRefreshFactory() {
  loadFactoryLists();
}

void MainWindow::loadFactoryLists() {
  // Portable discovery: ask the core factory for available short names.
  // This allows the GUI to run from any directory (or from a deployed folder)
  // without requiring access to the source tree.
  QStringList methods;
  QStringList problems;
  try {
    for (const auto& s : optimsolution::listMethodNames()) {
      methods << QString::fromStdString(s);
    }
    for (const auto& s : optimsolution::listProblemNames()) {
      problems << QString::fromStdString(s);
    }
  } catch (...) {
    // Fall back to empty lists.
  }

  // Keep a stable, user-friendly order.
  methods.removeDuplicates();
  problems.removeDuplicates();
  methods.sort(Qt::CaseInsensitive);
  problems.sort(Qt::CaseInsensitive);
  methodBox_->blockSignals(true);
  problemBox_->blockSignals(true);

  methodBox_->clear();
  problemBox_->clear();

  // Methods: display "Full name (short)" while storing the short name in itemData.
  for (const QString& shortName : methods) {
    QString display = shortName;
    try {
      auto opt = optimsolution::makeMethod(shortName.toStdString());
      if (opt) {
        const std::string full = opt->methodFullName();
        if (!full.empty() && full != shortName.toStdString()) {
          display = QString::fromStdString(full) + " (" + shortName.toLower() + ")";
        }
      }
    } catch (...) {
      // If a method cannot be instantiated here, keep the short name.
    }
    methodBox_->addItem(display, shortName);
  }

  // Problems: short name as itemData; the display adds the supported
  // dimensions for problems that accept only some (CEC2017 / CEC2022).
  for (const QString& p : problems) {
    problemBox_->addItem(problemDisplayName(p), p);
  }

  methodBox_->blockSignals(false);
  problemBox_->blockSignals(false);

  // try to keep current selections
  if (methodBox_->count() > 0 && methodBox_->currentText().isEmpty()) methodBox_->setCurrentIndex(0);
  if (problemBox_->count() > 0 && problemBox_->currentText().isEmpty()) problemBox_->setCurrentIndex(0);

  updateDimUiForProblem(problemNameFromDisplay(problemBox_->currentText()));

  // Batch lists mirror the factory lists and must be rebuilt after refresh.
  populateBatchLists();

  appendLog("Factory lists loaded.");
}

void MainWindow::onProblemChanged(const QString& problem) {
  updateDimUiForProblem(problem);
  // In Problem Sensitivity mode, refresh the sensitivity table when the problem changes.
  const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
  if (mode == 3 && !problem.isEmpty()) {
    populateSensitivityTableForProblem(problem.trimmed());
  }
}

void MainWindow::onMethodChanged(const QString&) {
  populateSettingsTables();
  // Rename method tab to include selected method
  QString m = methodBox_->currentText().trimmed();
  if (m.isEmpty()) m = "Method";
  tabs_->setTabText(3, QString("Optimization method (%1)").arg(m));
}


void MainWindow::onAddMethodParam() {
  if (!methodTable_) return;

  // Insert a new empty row. Parameter names/count vary per method, so the key cell is editable.
  const int row = methodTable_->rowCount();
  methodTable_->insertRow(row);

  auto* keyItem = new QTableWidgetItem(QString());
  keyItem->setData(Qt::UserRole, QString()); // old key (empty => new entry)

  auto* valItem = new QTableWidgetItem(QString());

  methodTable_->setItem(row, 0, keyItem);
  methodTable_->setItem(row, 1, valItem);

  setTablesEditable(true);
  methodTable_->setCurrentCell(row, 0);
  methodTable_->editItem(keyItem);
}

void MainWindow::onRemoveMethodParam() {
  if (!methodTable_) return;
  const int row = methodTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = methodTable_->item(row, 0);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();
  const QString method = currentMethodShort();

  if (cfg_ && cfg_->isLoaded() && !method.isEmpty() && !key.isEmpty()) {
    cfg_->removeKey(method, key);
  }

  methodTable_->removeRow(row);
  populateSettingsTables();
}

void MainWindow::onAddSensitivityParam() {
  if (!sensitivityTable_) return;

  const int row = sensitivityTable_->rowCount();
  sensitivityTable_->insertRow(row);

  auto* keyItem = new QTableWidgetItem(QString());
  keyItem->setData(Qt::UserRole, QString()); // old key (empty => new entry)
  auto* valItem = new QTableWidgetItem(QString());

  sensitivityTable_->setItem(row, 0, keyItem);
  sensitivityTable_->setItem(row, 1, valItem);

  setTablesEditable(true);
  sensitivityTable_->setCurrentCell(row, 0);
  sensitivityTable_->editItem(keyItem);
}

void MainWindow::onRemoveSensitivityParam() {
  if (!sensitivityTable_) return;
  const int row = sensitivityTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = sensitivityTable_->item(row, 0);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();

  if (cfg_ && cfg_->isLoaded() && !key.isEmpty()) {
    cfg_->removeKey("sensitivity", key);
  }

  sensitivityTable_->removeRow(row);
  populateSettingsTables();
}

void MainWindow::onAddGlobalParam() {
  if (!runTable_) return;

  const int row = runTable_->rowCount();
  runTable_->insertRow(row);

  auto* keyItem = new QTableWidgetItem(QString());
  keyItem->setData(Qt::UserRole, QString()); // old key (empty => new entry)

  auto* valItem = new QTableWidgetItem(QString());

  // Default new entries belong to [global].
  auto* srcItem = new QTableWidgetItem("global");

  runTable_->setItem(row, 0, keyItem);
  runTable_->setItem(row, 1, valItem);
  installSettingsValueWidgetRow(runTable_, "global", row);
  runTable_->setItem(row, 3, srcItem);

  setTablesEditable(true);
  runTable_->setCurrentCell(row, 0);
  runTable_->editItem(keyItem);
}

void MainWindow::onRemoveGlobalParam() {
  if (!runTable_) return;
  const int row = runTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = runTable_->item(row, 0);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();

  if (cfg_ && cfg_->isLoaded() && !key.isEmpty()) {
    cfg_->removeKey("global", key);
  }

  runTable_->removeRow(row);
  populateSettingsTables();
}

void MainWindow::onAddStopParam() {
  if (!stopTable_) return;

  const int row = stopTable_->rowCount();
  stopTable_->insertRow(row);

  auto* keyItem = new QTableWidgetItem(QString());
  keyItem->setData(Qt::UserRole, QString()); // old key (empty => new entry)

  auto* valItem = new QTableWidgetItem(QString());

  stopTable_->setItem(row, 0, keyItem);
  stopTable_->setItem(row, 1, valItem);
  installSettingsValueWidgetRow(stopTable_, "stop", row);

  setTablesEditable(true);
  stopTable_->setCurrentCell(row, 0);
  stopTable_->editItem(keyItem);
}

void MainWindow::onRemoveStopParam() {
  if (!stopTable_) return;
  const int row = stopTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = stopTable_->item(row, 0);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();

  if (cfg_ && cfg_->isLoaded() && !key.isEmpty()) {
    cfg_->removeKey("stop", key);
  }

  stopTable_->removeRow(row);
  populateSettingsTables();
}

void MainWindow::onAddInitParam() {
  if (!initTable_) return;

  const int row = initTable_->rowCount();
  initTable_->insertRow(row);

  auto* keyItem = new QTableWidgetItem(QString());
  keyItem->setData(Qt::UserRole, QString()); // old key (empty => new entry)

  auto* valItem = new QTableWidgetItem(QString());

  initTable_->setItem(row, 0, keyItem);
  initTable_->setItem(row, 1, valItem);
  installSettingsValueWidgetRow(initTable_, "init", row);

  setTablesEditable(true);
  initTable_->setCurrentCell(row, 0);
  initTable_->editItem(keyItem);
}

void MainWindow::onRemoveInitParam() {
  if (!initTable_) return;
  const int row = initTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = initTable_->item(row, 0);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();

  if (cfg_ && cfg_->isLoaded() && !key.isEmpty()) {
    cfg_->removeKey("init", key);
  }

  initTable_->removeRow(row);
  populateSettingsTables();
}


int MainWindow::fixedDimForProblem(const QString& problem) const {
  return optimsolution::getFixedDimOrZero(problem.toStdString());
}

std::vector<int> MainWindow::allowedDimsForProblem(const QString& problem) const {
  return optimsolution::getAllowedDimsOrEmpty(problem.toStdString());
}

QString MainWindow::problemDisplayName(const QString& problem) const {
  const std::vector<int> dims = allowedDimsForProblem(problem);
  if (dims.empty()) return problem;
  QStringList parts;
  for (int d : dims) parts << QString::number(d);
  return QString("%1  (D: %2)").arg(problem, parts.join(", "));
}

QVector<int> MainWindow::restrictToAllowedDims(const QString& problem, const QVector<int>& dims) const {
  const std::vector<int> allowed = allowedDimsForProblem(problem);
  if (allowed.empty()) return dims;
  QVector<int> ok;
  for (int d : dims) {
    if (std::find(allowed.begin(), allowed.end(), d) != allowed.end() && !ok.contains(d)) ok.push_back(d);
  }
  if (ok.isEmpty()) {
    for (int d : allowed) ok.push_back(d);
  }
  std::sort(ok.begin(), ok.end());
  return ok;
}

QString MainWindow::problemNameFromDisplay(const QString& text) {
  static const QRegularExpression re(QStringLiteral(R"(\s*\(D:[^)]*\)\s*$)"));
  QString t = text.trimmed();
  t.remove(re);
  return t.trimmed();
}

void MainWindow::snapDimSpinToAllowed() {
  if (!dimSpin_ || currentAllowedDims_.empty()) return;
  const int v = dimSpin_->value();
  if (std::find(currentAllowedDims_.begin(), currentAllowedDims_.end(), v) != currentAllowedDims_.end()) {
    return; // already a supported dimension
  }
  int best = currentAllowedDims_.front();
  int bestDist = std::abs(v - best);
  for (int cand : currentAllowedDims_) {
    const int d = std::abs(v - cand);
    if (d < bestDist) { best = cand; bestDist = d; }
  }
  const QSignalBlocker blocker(dimSpin_);
  dimSpin_->setValue(best);
}

void MainWindow::updateDimUiForProblem(const QString& problem) {
  // Batch run: the Single-run Problem / Dimension controls are hidden, and
  // the dimension spin box holds the DEFAULT dimension of the batch (used by
  // every problem without its own entry in the Dims table). The problem
  // passed here is the hidden Single-run dropdown's problem, which has
  // nothing to do with the batch. Letting it move the spin box (e.g. to 6
  // when that dropdown sits on the fixed-dimension "eld1", or to 10..50 for a
  // CEC problem) silently changed every batch problem key from
  // "sphere [d=30]" to "sphere [d=6]": all results vanished from the Output
  // tabs and "Run batch" re-ran the whole experiment. In Batch mode this
  // function therefore leaves the spin box alone; onRunModeChanged() re-syncs
  // it when the user switches back to a mode that shows it.
  if (runModeBox_ && runModeBox_->currentData().toInt() == 1) {
    return;
  }

  currentFixedDim_ = fixedDimForProblem(problem);

  if (currentFixedDim_ > 0) {
    currentAllowedDims_.clear();
    dimSpin_->setSingleStep(1);
    dimSpin_->setEnabled(false);
    // The spinner's general floor is 2 (scalable problems start at D=2),
    // but some FIXED-dimension problems have a true dimension below that
    // (e.g. equalmaxima, bifunctionalcatalyst: D=1). Without widening the
    // minimum first, Qt's QSpinBox::setValue() silently clamps to the
    // existing minimum (2), so the box would display "2" for a D=1
    // problem instead of the correct "1".
    if (currentFixedDim_ < dimSpin_->minimum()) {
      dimSpin_->setMinimum(currentFixedDim_);
    }
    dimSpin_->setValue(currentFixedDim_);
  } else {
    currentAllowedDims_ = allowedDimsForProblem(problem);

    if (!currentAllowedDims_.empty()) {
      // Multi-value restriction (e.g. CEC2017: D in {10,30,50}): the user
      // still picks among the allowed set, so the spinbox stays enabled.
      // Sort defensively, then configure range + step so the up/down
      // arrows and mouse wheel land EXACTLY on allowed values -- e.g. for
      // {10,30,50} that means range [10,50] with step 20, so 10 -> 30 -> 50
      // via a single arrow click each, never landing on an unsupported
      // in-between value in the first place. (An earlier version of this
      // snapped to the nearest allowed value on every valueChanged, which
      // left singleStep at its default of 1 -- so a single arrow click
      // from 10 produced 11, which is closer to 10 than to 30 and got
      // immediately snapped back to 10, making the spinbox appear stuck.)
      std::sort(currentAllowedDims_.begin(), currentAllowedDims_.end());
      const int lo = currentAllowedDims_.front();
      const int hi = currentAllowedDims_.back();
      dimSpin_->setRange(lo, hi);
      int step = 1;
      bool uniform = currentAllowedDims_.size() >= 2;
      if (uniform) {
        step = currentAllowedDims_[1] - currentAllowedDims_[0];
        for (size_t i = 2; i < currentAllowedDims_.size(); ++i) {
          if (currentAllowedDims_[i] - currentAllowedDims_[i - 1] != step) { uniform = false; break; }
        }
      }
      dimSpin_->setSingleStep(uniform ? step : 1);
      snapDimSpinToAllowed(); // backstop: correct if the current value isn't in the set
    } else {
      // Restore the standard range/step for fully scalable problems -- a
      // prior multi-value-restricted problem (e.g. CEC2017's max of 50)
      // must not linger and cap what should now be an unrestricted D.
      dimSpin_->setSingleStep(1);
      dimSpin_->setRange(2, 100000);
    }
    dimSpin_->setEnabled(true);
    // (The former "batch mode: align the spin box with the per-problem
    // override" block is gone: in Batch mode this function now returns at
    // the top, see the comment there.)
  }
}

QString MainWindow::cliPath() const {
  // Prefer CLI next to GUI (post-build copy), else in the same build tree.
  QString dir = QDir(QCoreApplication::applicationDirPath()).absolutePath();
  QString candidate = QDir(dir).filePath("optimsolution.exe");
  if (QFileInfo::exists(candidate)) return candidate;

  // fallback: try build folder relative
  QString root = projectRoot_;
  QString cand2 = QDir(root).filePath("build/Debug/optimsolution.exe");
  if (QFileInfo::exists(cand2)) return cand2;
  return QString();
}

QString MainWindow::currentMethodShort() const {
  const QVariant d = methodBox_->currentData();
  QString s = d.isValid() ? d.toString().trimmed() : QString();
  if (s.isEmpty()) {
    // Fallback: in case items were added without userData.
    const QString t = methodBox_->currentText().trimmed();
    const QRegularExpression re(R"(\(([A-Za-z0-9_]+)\)\s*$)");
    const auto mm = re.match(t);
    if (mm.hasMatch()) s = mm.captured(1);
    else s = t;
  }
  return s;
}

QString MainWindow::currentProblemShort() const {
  const QVariant v = problemBox_ ? problemBox_->currentData() : QVariant();
  if (v.isValid()) {
    const QString s = v.toString().trimmed();
    if (!s.isEmpty()) return s;
  }
  const QString txt = problemBox_ ? problemNameFromDisplay(problemBox_->currentText()) : QString();
  const int l = txt.lastIndexOf('(');
  const int r = txt.lastIndexOf(')');
  if (l >= 0 && r > l) {
    const QString inside = txt.mid(l + 1, r - l - 1).trimmed();
    if (!inside.isEmpty()) return inside;
  }
  return txt;
}

bool MainWindow::isFixedDimensionProblem(const QString& problemShort) const {
  return fixedDimForProblem(problemShort.trimmed()) > 0;
}




bool MainWindow::detectCliSupportsConfigArg(const QString& cli) {
  // Best-effort probe: parse --help output and look for "--config".
  QProcess p;
  p.setProgram(cli);
  p.setProcessChannelMode(QProcess::MergedChannels);

  auto probe = [&](const QStringList& a) -> QString {
    p.setArguments(a);
    p.start();
    if (!p.waitForFinished(1500)) {
      p.kill();
      p.waitForFinished(500);
      return QString();
    }
    return QString::fromUtf8(p.readAll());
  };

  QString out = probe(QStringList() << "--help");
  if (out.isEmpty()) out = probe(QStringList() << "-h");
  if (out.isEmpty()) out = probe(QStringList() << "/?");

  return out.contains("--config", Qt::CaseInsensitive);
}

// Sanitizes a name for use in a tab title / exported file name: any run of
// characters that isn't a letter/digit is collapsed to a single underscore
// "_", and leading/trailing underscores are trimmed. Used so that names built
// from problem/parameter names never contain spaces or other symbols.
static QString sanitizeUnderscoreName(const QString& s) {
  QString out = s.trimmed();
  out.replace(QRegularExpression("[^A-Za-z0-9]+"), "_");
  while (out.startsWith('_')) out.remove(0, 1);
  while (out.endsWith('_')) out.chop(1);
  return out;
}

int MainWindow::createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim) {
  if (!outputTabs_) return -1;

  const QString m = methodShort.trimmed();
  const QString p = problemShort.trimmed();
  const bool isSystemTab = (m.compare("System", Qt::CaseInsensitive) == 0);
  const int runMode = (runModeBox_ ? runModeBox_->currentData().toInt() : 0);
  const bool isBatchUi = (runMode == 1);
  const bool isPortfolioUi = (runMode == 7);
  const bool isSensitivityUi = (runMode == 2 || runMode == 3);
  // "Sensitivity analysis of method parameters" (Run mode combo entry #3,
  // data value 2): tabs/exports here are named "<problem>_<dimension>".
  const bool isMethodParamSensitivity = (runMode == 2);

  auto makeFriendlyOutputTitle = [&](const QString& methodName, const QString& problemName, int dim) {
    if (isSystemTab) {
      return QString("Batch Summary");
    }

    if (problemName.isEmpty()) {
      return methodName;
    }

    const QString dimSuffix = (dim > 0) ? QString(" | D=%1").arg(dim) : QString();
    if (isBatchUi) {
      return QString("Job Log: %1 | %2%3").arg(methodName, problemName, dimSuffix);
    }
    if (isMethodParamSensitivity) {
      // e.g. "rastrigin_50"
      const QString pName = sanitizeUnderscoreName(problemName);
      return (dim > 0) ? QString("%1_%2").arg(pName).arg(dim) : pName;
    }
    if (isSensitivityUi) {
      const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 2;
      const QString prefix = (mode == 3) ? "Problem Sens" : "Sensitivity";
      return QString("%1: %2 | %3%4").arg(prefix, methodName, problemName, dimSuffix);
    }
    return QString("%1 | %2%3").arg(methodName, problemName, dimSuffix);
  };

  const QString baseTitle = makeFriendlyOutputTitle(m, p, problemDim);
  QString title = baseTitle;

  auto titleExists = [&](const QString& t) {
    for (const auto& r : outputRuns_) {
      if (r.title.compare(t, Qt::CaseInsensitive) == 0) return true;
    }
    return false;
  };

  int suffix = 2;
  while (titleExists(title)) {
    title = QString("%1 (%2)").arg(baseTitle).arg(suffix++);
  }

  auto* page = new QWidget(outputTabs_);
  auto* pageLay = new QVBoxLayout(page);
  pageLay->setContentsMargins(0, 0, 0, 0);

  // Inner tabs keep output features grouped per run while the outer tabs remain per-run/per-job.
  auto* inner = new QTabWidget(page);
  inner->setDocumentMode(true);
  pageLay->addWidget(inner);

  // Log
  auto* logTab = new QWidget(inner);
  auto* logLay = new QVBoxLayout(logTab);
  logLay->setContentsMargins(0, 0, 0, 0);
  auto* log = new QTextEdit(logTab);
  log->setReadOnly(true);
  log->setLineWrapMode(QTextEdit::NoWrap);
  log->document()->setMaximumBlockCount(20000);
  logLay->addWidget(log);

  // Per-batch-summary derived tables must remain visible later in this function
  // when the newly created System tab is registered into g_batchSummaryUi.
  QPushButton* exportBestTableBtn = nullptr;
  QTableWidget* bestTable = nullptr;
  QPushButton* exportMeanTableBtn = nullptr;
  QTableWidget* meanTable = nullptr;
  QPushButton* exportBestRankingBtn = nullptr;
  QTableWidget* bestRankingTable = nullptr;
  QPushButton* exportMeanRankingBtn = nullptr;
  QTableWidget* meanRankingTable = nullptr;
  QPushButton* exportFinalRankingBtn = nullptr;
  QTableWidget* finalRankingTable = nullptr;
  // Metric/Aggregate combos — created inside the System tab stats block,
  // referenced in the bundle assignment block below.
  QComboBox* wMetricCombo = nullptr;
  QComboBox* wAggCombo    = nullptr;
  QComboBox* fMetricCombo = nullptr;
  QComboBox* fAggCombo    = nullptr;

// Statistics (batch) are attached to the System tab only.
  if (m.compare("System", Qt::CaseInsensitive) == 0) {
    auto* statsTab = new QWidget(inner);
    auto* statsLay = new QVBoxLayout(statsTab);
    statsLay->setContentsMargins(8, 8, 8, 8);

    
    statsTabs_ = new QTabWidget(statsTab);
    statsTabs_->setDocumentMode(true);
    statsLay->addWidget(statsTabs_, 1);

    // Results tab.
    auto* resTab = new QWidget(statsTabs_);
    auto* resLay = new QVBoxLayout(resTab);
    resLay->setContentsMargins(0, 0, 0, 0);

    auto* resTop = new QHBoxLayout();
    exportStatsBtn_ = new QPushButton("Export table...", resTab);
    exportStatsBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportStatsBtn_->setIconSize(QSize(18, 18));
    exportStatsBtn_->setEnabled(false);
    connect(exportStatsBtn_, &QPushButton::clicked, this, &MainWindow::onExportStats);
    resTop->addWidget(exportStatsBtn_);

    generateReportBtn_ = new QPushButton("Generate report (PDF/Word)...", resTab);
    generateReportBtn_->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    generateReportBtn_->setIconSize(QSize(18, 18));
    generateReportBtn_->setEnabled(false);
    generateReportBtn_->setToolTip(
      "Collects the Results table, Rankings, Wilcoxon/Friedman/Critical-Difference plots, and (optionally) "
      "per-run convergence curves into a single PDF or Word-compatible (.odt) document.");
    connect(generateReportBtn_, &QPushButton::clicked, this, &MainWindow::onGenerateReport);
    resTop->addWidget(generateReportBtn_);

    resTop->addStretch(1);
    resLay->addLayout(resTop);

    statsTable_ = new QTableWidget(resTab);
    statsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    statsTable_->setSelectionBehavior(QAbstractItemView::SelectItems);
    statsTable_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    statsTable_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    statsTable_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    statsTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    statsTable_->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    resLay->addWidget(statsTable_, 1);

    statsNoteLbl_ = new QLabel(
      "Notes: lower Value is better; tests use one aggregated value per (problem, method) cell; p-values are Holm-adjusted.",
      resTab
    );
    statsNoteLbl_->setWordWrap(true);
    resLay->addWidget(statsNoteLbl_, 0);

    statsTabs_->addTab(resTab, "Results Table");

    const QString exportStem = sanitizeExportFileStem(title);
    auto exportDerivedTable = [this](QTableWidget* table, const QString& dialogTitle, const QString& defaultPath) {
      if (!table || table->rowCount() == 0) return;

      const QString out = QFileDialog::getSaveFileName(
        this,
        dialogTitle,
        defaultPath,
        "Excel Workbook (*.xlsx);;CSV (*.csv)"
      );
      if (out.isEmpty()) return;

      QTableWidget* savedStatsTable = statsTable_;
      statsTable_ = table;

      const QString lower = out.toLower();
      if (lower.endsWith(".xlsx")) {
        if (exportStatsToXlsxNative(out)) {
          statsTable_ = savedStatsTable;
          return;
        }
      }

      QTemporaryFile tmp(QDir::tempPath() + "/optimsolution_stats_XXXXXX.csv");
      tmp.setAutoRemove(true);
      if (!tmp.open()) {
        statsTable_ = savedStatsTable;
        QMessageBox::warning(this, "Export", "Failed to create a temporary file.");
        return;
      }

      QTextStream ts(&tmp);
      ts.setEncoding(QStringConverter::Utf8);

      QStringList header;
      for (int c = 0; c < table->columnCount(); ++c) {
        header << (table->horizontalHeaderItem(c) ? table->horizontalHeaderItem(c)->text() : QString());
      }
      ts << header.join(",") << "\n";

      for (int r = 0; r < table->rowCount(); ++r) {
        QStringList row;
        for (int c = 0; c < table->columnCount(); ++c) {
          QTableWidgetItem* it = table->item(r, c);
          QString cell = it ? it->text() : QString();
          if (cell.contains(',') || cell.contains('"') || cell.contains('\n') || cell.contains('\r')) {
            cell.replace("\"", "\"\"");
            cell = "\"" + cell + "\"";
          }
          row << cell;
        }
        ts << row.join(",") << "\n";
      }
      ts.flush();
      tmp.flush();

      if (lower.endsWith(".xlsx")) {
        if (!exportStatsToXlsxViaPython(out, tmp.fileName())) {
          const QString csvOut = out.left(out.size() - 5) + ".csv";
          QFile::remove(csvOut);
          QFile::copy(tmp.fileName(), csvOut);
          QMessageBox::information(this,
                                   "Export",
                                   "XLSX export requires Python + openpyxl on the system.\n"
                                   "A CSV file was written instead:\n" + csvOut);
        }
      } else {
        QFile::remove(out);
        QFile::copy(tmp.fileName(), out);
      }

      statsTable_ = savedStatsTable;
    };

    auto makeDerivedTab = [&](const QString& tabTitle,
                              const QString& defaultSuffix,
                              QPushButton*& outBtn,
                              QTableWidget*& outTable) {
      auto* tab = new QWidget(statsTabs_);
      auto* lay = new QVBoxLayout(tab);
      lay->setContentsMargins(0, 0, 0, 0);
      auto* top = new QHBoxLayout();
      outBtn = new QPushButton("Export table...", tab);
      outBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
      outBtn->setIconSize(QSize(18, 18));
      outBtn->setEnabled(false);
      top->addWidget(outBtn);
      top->addStretch(1);
      lay->addLayout(top);
      outTable = new QTableWidget(tab);
      setupDerivedAnalysisTable(outTable);
      lay->addWidget(outTable, 1);
      connect(outBtn, &QPushButton::clicked, this, [this, outTable, tabTitle, defaultSuffix, exportStem, exportDerivedTable]() {
        exportDerivedTable(outTable,
                           QString("Export %1").arg(tabTitle.toLower()),
                           QDir::homePath() + "/" + exportStem + defaultSuffix);
      });
      statsTabs_->addTab(tab, tabTitle);
    };

    makeDerivedTab("Best Table", "_best_table.xlsx", exportBestTableBtn, bestTable);
    makeDerivedTab("Mean Table", "_mean_table.xlsx", exportMeanTableBtn, meanTable);
    makeDerivedTab("Best Ranking", "_best_ranking.xlsx", exportBestRankingBtn, bestRankingTable);
    makeDerivedTab("Mean Ranking", "_mean_ranking.xlsx", exportMeanRankingBtn, meanRankingTable);
    makeDerivedTab("Final Ranking", "_final_ranking.xlsx", exportFinalRankingBtn, finalRankingTable);

    // Wilcoxon tab (paired; annotations are drawn inside the plot).
    auto* wTab = new QWidget(statsTabs_);
    auto* wLay = new QVBoxLayout(wTab);
    wLay->setContentsMargins(0, 0, 0, 0);

    auto* wTop = new QHBoxLayout();
    wTop->addWidget(new QLabel("Pairs:", wTab));
    wilcoxonPairsCombo_ = new QComboBox(wTab);
    wilcoxonPairsCombo_->addItem("All pairs", QVariant(0));
    wilcoxonPairsCombo_->addItem("Best vs others", QVariant(1));
    wilcoxonPairsCombo_->setCurrentIndex(0);
    connect(wilcoxonPairsCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onStatsTestChanged);
    wTop->addWidget(wilcoxonPairsCombo_);

    wTop->addSpacing(12);
    wTop->addWidget(new QLabel("alpha:", wTab));
    statsAlphaCombo_ = new QComboBox(wTab);
    statsAlphaCombo_->addItem("0.05", QVariant(0.05));
    statsAlphaCombo_->addItem("0.01", QVariant(0.01));
    statsAlphaCombo_->setCurrentIndex(0);
    connect(statsAlphaCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onStatsTestChanged);
    wTop->addWidget(statsAlphaCombo_);

    wTop->addSpacing(12);
    exportWilcoxonPlotBtn_ = new QPushButton("Export Wilcoxon plot (PNG)...", wTab);
    exportWilcoxonPlotBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportWilcoxonPlotBtn_->setIconSize(QSize(18, 18));
    exportWilcoxonPlotBtn_->setEnabled(false);
    connect(exportWilcoxonPlotBtn_, &QPushButton::clicked, this, &MainWindow::onExportWilcoxonPlotPng);
    wTop->addWidget(exportWilcoxonPlotBtn_);

    wTop->addStretch(1);
    wLay->addLayout(wTop);

    wilcoxonPlot_ = new WilcoxonBoxPlotWidget(wTab);

    // Controls panel to the RIGHT of the plot (mirrors single-run convergence layout).
    auto* wControlsPanel = new QWidget(wTab);
    auto* wControlsLay   = new QVBoxLayout(wControlsPanel);
    wControlsLay->setContentsMargins(8, 0, 0, 0);
    wControlsLay->setSpacing(10);
    wControlsPanel->setFixedWidth(150);

    // Metric combo (PRIMARY — bound to batchMetricCombo_ via bundle).
    auto* wMetricLbl = new QLabel("Metric", wControlsPanel);
    wMetricLbl->setAlignment(Qt::AlignLeft);
    wMetricCombo = new QComboBox(wControlsPanel);
    wMetricCombo->addItem("best_f (best of runs)",         int(BatchMetricMode::BestFinalBestF));
    wMetricCombo->addItem("best_f (mean of runs)",         int(BatchMetricMode::MeanFinalBestF));
    wMetricCombo->addItem("Iteration of best_f",           int(BatchMetricMode::IterationAtBest));
    wMetricCombo->addItem("Function evals of best_f",      int(BatchMetricMode::EvalsAtBest));
    wMetricCombo->setCurrentIndex(0);

    // Aggregate combo (PRIMARY — bound to batchAggCombo_ via bundle).
    auto* wAggLbl = new QLabel("Aggregate", wControlsPanel);
    wAggLbl->setAlignment(Qt::AlignLeft);
    wAggCombo = new QComboBox(wControlsPanel);
    wAggCombo->addItems(QStringList() << "Mean" << "Median" << "Min" << "Max");
    wAggCombo->setEnabled(false); // only active for iteration/eval metrics

    auto* wThemeLbl = new QLabel("Theme", wControlsPanel);
    wThemeLbl->setAlignment(Qt::AlignLeft);
    auto* wThemeCombo = new QComboBox(wControlsPanel);
    wThemeCombo->addItem("Dark");
    wThemeCombo->addItem("Light");
    wThemeCombo->addItem("Transparent");
    wThemeCombo->setCurrentIndex(0);

    auto* wGridLbl = new QLabel("Grid", wControlsPanel);
    auto* wGridCombo = new QComboBox(wControlsPanel);
    wGridCombo->addItem("Sparse");
    wGridCombo->addItem("Medium");
    wGridCombo->addItem("Dense");
    wGridCombo->setCurrentIndex(1);

    connect(wThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        WilcoxonBoxPlotWidget::ThemeMode tm = WilcoxonBoxPlotWidget::ThemeMode::Dark;
        if (index == 1) tm = WilcoxonBoxPlotWidget::ThemeMode::Light;
        else if (index == 2) tm = WilcoxonBoxPlotWidget::ThemeMode::Transparent;
        wp->setThemeMode(tm);
      }
    });
    connect(wGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        WilcoxonBoxPlotWidget::GridDensity gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
        if (index == 0) gd = WilcoxonBoxPlotWidget::GridDensity::Sparse;
        else if (index == 2) gd = WilcoxonBoxPlotWidget::GridDensity::Dense;
        wp->setGridDensity(gd);
      }
    });
    // Metric and Agg connects — update both the member pointer and the Friedman panel
    // (fMetricCombo / fAggCombo are captured at Friedman panel creation below).
    // The connection to onBatchMetricUiChanged is done below once fMetricCombo is known.
    // For now, store wMetricCombo/wAggCombo; connections are finalised after Friedman panel.

    wControlsLay->addWidget(wMetricLbl);
    wControlsLay->addWidget(wMetricCombo);
    wControlsLay->addSpacing(2);
    wControlsLay->addWidget(wAggLbl);
    wControlsLay->addWidget(wAggCombo);
    wControlsLay->addSpacing(6);
    wControlsLay->addWidget(wThemeLbl);
    wControlsLay->addWidget(wThemeCombo);
    wControlsLay->addSpacing(6);
    wControlsLay->addWidget(wGridLbl);
    wControlsLay->addWidget(wGridCombo);
    wControlsLay->addStretch(1);

    auto* wPlotRow = new QHBoxLayout();
    wPlotRow->addWidget(wilcoxonPlot_, 1);
    wPlotRow->addWidget(wControlsPanel, 0);
    wLay->addLayout(wPlotRow, 1);


    wilcoxonSummaryLbl_ = new QLabel("No Wilcoxon comparisons yet.", wTab);
    wilcoxonSummaryLbl_->setWordWrap(true);
    wLay->addWidget(wilcoxonSummaryLbl_, 0);

    statsTabs_->addTab(wTab, "Pairwise Wilcoxon");

    // Friedman tab (average ranks + pairwise post-hoc).
    auto* fTab = new QWidget(statsTabs_);
    auto* fLay = new QVBoxLayout(fTab);
    fLay->setContentsMargins(0, 0, 0, 0);

    auto* fTop = new QHBoxLayout();
    fTop->addWidget(new QLabel("alpha:", fTab));
    auto* friedmanAlphaCombo = new QComboBox(fTab);
    friedmanAlphaCombo->addItem("0.05", QVariant(0.05));
    friedmanAlphaCombo->addItem("0.01", QVariant(0.01));
    friedmanAlphaCombo->setCurrentIndex(0);
    fTop->addWidget(friedmanAlphaCombo);

    fTop->addSpacing(12);
    exportRankPlotBtn_ = new QPushButton("Export Friedman plot (PNG)...", fTab);
    exportRankPlotBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportRankPlotBtn_->setIconSize(QSize(18, 18));
    exportRankPlotBtn_->setEnabled(false);
    connect(exportRankPlotBtn_, &QPushButton::clicked, this, &MainWindow::onExportRankPlotPng);
    fTop->addWidget(exportRankPlotBtn_);
    fTop->addStretch(1);
    fLay->addLayout(fTop);

    // Keep Friedman alpha control in sync with the canonical Statistics alpha.
    if (statsAlphaCombo_) {
      connect(statsAlphaCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [friedmanAlphaCombo](int idx){
        if (!friedmanAlphaCombo) return;
        if (friedmanAlphaCombo->currentIndex() == idx) return;
        QSignalBlocker b(friedmanAlphaCombo);
        friedmanAlphaCombo->setCurrentIndex(idx);
      });
      connect(friedmanAlphaCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, friedmanAlphaCombo](int idx){
        if (!statsAlphaCombo_) return;
        if (statsAlphaCombo_->currentIndex() == idx) { onStatsTestChanged(idx); return; }
        QSignalBlocker b(statsAlphaCombo_);
        statsAlphaCombo_->setCurrentIndex(idx);
        onStatsTestChanged(idx);
      });
    } else {
      connect(friedmanAlphaCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onStatsTestChanged);
    }

    rankPlot_ = new RankPlotWidget(fTab);

    // Controls panel to the RIGHT of the plot (mirrors single-run convergence layout).
    auto* fControlsPanel = new QWidget(fTab);
    auto* fControlsLay   = new QVBoxLayout(fControlsPanel);
    fControlsLay->setContentsMargins(8, 0, 0, 0);
    fControlsLay->setSpacing(10);
    fControlsPanel->setFixedWidth(150);

    // Metric combo (SECONDARY — synced with wMetricCombo).
    auto* fMetricLbl = new QLabel("Metric", fControlsPanel);
    fMetricLbl->setAlignment(Qt::AlignLeft);
    fMetricCombo = new QComboBox(fControlsPanel);
    fMetricCombo->addItem("best_f (best of runs)",         int(BatchMetricMode::BestFinalBestF));
    fMetricCombo->addItem("best_f (mean of runs)",         int(BatchMetricMode::MeanFinalBestF));
    fMetricCombo->addItem("Iteration of best_f",           int(BatchMetricMode::IterationAtBest));
    fMetricCombo->addItem("Function evals of best_f",      int(BatchMetricMode::EvalsAtBest));
    fMetricCombo->setCurrentIndex(0);

    // Aggregate combo (SECONDARY — synced with wAggCombo).
    auto* fAggLbl = new QLabel("Aggregate", fControlsPanel);
    fAggLbl->setAlignment(Qt::AlignLeft);
    fAggCombo = new QComboBox(fControlsPanel);
    fAggCombo->addItems(QStringList() << "Mean" << "Median" << "Min" << "Max");
    fAggCombo->setEnabled(false);

    // ---------- bidirectional sync: Wilcoxon ↔ Friedman metric/agg ----------
    // Wilcoxon → Friedman + trigger update
    connect(wMetricCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, fMetricCombo, fAggCombo, wAggCombo](int idx) {
      { QSignalBlocker b(fMetricCombo); fMetricCombo->setCurrentIndex(idx); }
      const bool aggEn = (idx == 2 || idx == 3);
      fAggCombo->setEnabled(aggEn);
      wAggCombo->setEnabled(aggEn);
      onBatchMetricUiChanged();
    });
    connect(wAggCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, fAggCombo](int idx) {
      { QSignalBlocker b(fAggCombo); fAggCombo->setCurrentIndex(idx); }
      onBatchMetricUiChanged();
    });
    // Friedman → Wilcoxon + trigger update
    connect(fMetricCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, wMetricCombo, fAggCombo, wAggCombo](int idx) {
      { QSignalBlocker b(wMetricCombo); wMetricCombo->setCurrentIndex(idx); }
      const bool aggEn = (idx == 2 || idx == 3);
      fAggCombo->setEnabled(aggEn);
      wAggCombo->setEnabled(aggEn);
      onBatchMetricUiChanged();
    });
    connect(fAggCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, wAggCombo](int idx) {
      { QSignalBlocker b(wAggCombo); wAggCombo->setCurrentIndex(idx); }
      onBatchMetricUiChanged();
    });
    // ---------- end sync ----------

    auto* fThemeLbl = new QLabel("Theme", fControlsPanel);
    fThemeLbl->setAlignment(Qt::AlignLeft);
    auto* fThemeCombo = new QComboBox(fControlsPanel);
    fThemeCombo->addItem("Dark");
    fThemeCombo->addItem("Light");
    fThemeCombo->addItem("Transparent");
    fThemeCombo->setCurrentIndex(0);

    auto* fGridLbl = new QLabel("Grid", fControlsPanel);
    auto* fGridCombo = new QComboBox(fControlsPanel);
    fGridCombo->addItem("Sparse");
    fGridCombo->addItem("Medium");
    fGridCombo->addItem("Dense");
    fGridCombo->setCurrentIndex(1);

    connect(fThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* rp = dynamic_cast<RankPlotWidget*>(rankPlot_)) {
        RankPlotWidget::ThemeMode tm = RankPlotWidget::ThemeMode::Dark;
        if (index == 1) tm = RankPlotWidget::ThemeMode::Light;
        else if (index == 2) tm = RankPlotWidget::ThemeMode::Transparent;
        rp->setThemeMode(tm);
      }
    });
    connect(fGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* rp = dynamic_cast<RankPlotWidget*>(rankPlot_)) {
        RankPlotWidget::GridDensity gd = RankPlotWidget::GridDensity::Medium;
        if (index == 0) gd = RankPlotWidget::GridDensity::Sparse;
        else if (index == 2) gd = RankPlotWidget::GridDensity::Dense;
        rp->setGridDensity(gd);
      }
    });

    fControlsLay->addWidget(fMetricLbl);
    fControlsLay->addWidget(fMetricCombo);
    fControlsLay->addSpacing(2);
    fControlsLay->addWidget(fAggLbl);
    fControlsLay->addWidget(fAggCombo);
    fControlsLay->addSpacing(6);
    fControlsLay->addWidget(fThemeLbl);
    fControlsLay->addWidget(fThemeCombo);
    fControlsLay->addSpacing(6);
    fControlsLay->addWidget(fGridLbl);
    fControlsLay->addWidget(fGridCombo);
    fControlsLay->addStretch(1);

    auto* fPlotRow = new QHBoxLayout();
    fPlotRow->addWidget(rankPlot_, 1);
    fPlotRow->addWidget(fControlsPanel, 0);
    fLay->addLayout(fPlotRow, 1);


    // Pairwise comparison details are rendered inside the plot widget.
    // The widgets below are kept (hidden) for export/compatibility.
    statsSummaryLbl_ = new QLabel("No Friedman comparisons yet.", fTab);
    statsSummaryLbl_->setWordWrap(true);
    statsSummaryLbl_->setVisible(false);

    statsPairwiseTable_ = new QTableWidget(fTab);
    statsPairwiseTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    statsPairwiseTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    statsPairwiseTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    statsPairwiseTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    statsPairwiseTable_->horizontalHeader()->setStretchLastSection(true);
    statsPairwiseTable_->verticalHeader()->setVisible(false);
    statsPairwiseTable_->setVisible(false);

    statsTabs_->addTab(fTab, "Friedman Ranking");

    // Critical Difference (Nemenyi) tab — post-hoc CD diagram companion to the Friedman test above.
    auto* cdTab = new QWidget(statsTabs_);
    auto* cdLay = new QVBoxLayout(cdTab);
    cdLay->setContentsMargins(0, 0, 0, 0);

    auto* cdTop = new QHBoxLayout();
    cdTop->addWidget(new QLabel("Uses the same alpha and metric as the Friedman tab.", cdTab));
    cdTop->addSpacing(12);
    exportCdPlotBtn_ = new QPushButton("Export CD diagram (PNG)...", cdTab);
    exportCdPlotBtn_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportCdPlotBtn_->setIconSize(QSize(18, 18));
    exportCdPlotBtn_->setEnabled(false);
    connect(exportCdPlotBtn_, &QPushButton::clicked, this, &MainWindow::onExportCdPlotPng);
    cdTop->addWidget(exportCdPlotBtn_);
    cdTop->addStretch(1);
    cdLay->addLayout(cdTop);

    cdPlot_ = new CDDiagramWidget(cdTab);

    auto* cdControlsPanel = new QWidget(cdTab);
    auto* cdControlsLay   = new QVBoxLayout(cdControlsPanel);
    cdControlsLay->setContentsMargins(8, 0, 0, 0);
    cdControlsLay->setSpacing(10);
    cdControlsPanel->setFixedWidth(150);

    auto* cdThemeLbl = new QLabel("Theme", cdControlsPanel);
    cdThemeLbl->setAlignment(Qt::AlignLeft);
    auto* cdThemeCombo = new QComboBox(cdControlsPanel);
    cdThemeCombo->addItem("Dark");
    cdThemeCombo->addItem("Light");
    cdThemeCombo->addItem("Transparent");
    cdThemeCombo->setCurrentIndex(0);
    connect(cdThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* cp = dynamic_cast<CDDiagramWidget*>(cdPlot_)) {
        CDDiagramWidget::ThemeMode tm = CDDiagramWidget::ThemeMode::Dark;
        if (index == 1) tm = CDDiagramWidget::ThemeMode::Light;
        else if (index == 2) tm = CDDiagramWidget::ThemeMode::Transparent;
        cp->setThemeMode(tm);
      }
    });

    auto* cdGridLbl = new QLabel("Grid", cdControlsPanel);
    auto* cdGridCombo = new QComboBox(cdControlsPanel);
    cdGridCombo->addItem("Sparse");
    cdGridCombo->addItem("Medium");
    cdGridCombo->addItem("Dense");
    cdGridCombo->setCurrentIndex(1);
    connect(cdGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* cp = dynamic_cast<CDDiagramWidget*>(cdPlot_)) {
        CDDiagramWidget::GridDensity gd = CDDiagramWidget::GridDensity::Medium;
        if (index == 0) gd = CDDiagramWidget::GridDensity::Sparse;
        else if (index == 2) gd = CDDiagramWidget::GridDensity::Dense;
        cp->setGridDensity(gd);
      }
    });

    cdControlsLay->addWidget(cdThemeLbl);
    cdControlsLay->addWidget(cdThemeCombo);
    cdControlsLay->addSpacing(6);
    cdControlsLay->addWidget(cdGridLbl);
    cdControlsLay->addWidget(cdGridCombo);
    cdControlsLay->addStretch(1);

    auto* cdPlotRow = new QHBoxLayout();
    cdPlotRow->addWidget(cdPlot_, 1);
    cdPlotRow->addWidget(cdControlsPanel, 0);
    cdLay->addLayout(cdPlotRow, 1);

    cdSummaryLbl_ = new QLabel("No Nemenyi comparisons yet.", cdTab);
    cdSummaryLbl_->setWordWrap(true);
    cdSummaryLbl_->setVisible(false);
    cdLay->addWidget(cdSummaryLbl_);

    statsTabs_->addTab(cdTab, "Critical Difference (Nemenyi)");

    // Tabs 1+ start disabled until batch completes with ≥2 methods.
    updateStatsTabsEnabled();

    inner->addTab(statsTab, "Batch Analysis");
    inner->addTab(logTab, "Batch Log");
  }


  // Batch mode per-job tabs fall through to the full tab creation code below,
  // so they get Convergence + Final Distribution tabs with theme/grid/x-axis controls.

  // The Batch Summary tab does not show Convergence/Distribution/Sensitivity.
  if (isSystemTab) {
    const int tabIndex = outputTabs_->addTab(page, title);

    OutputRunTab run;
    run.title = title;
    run.methodShort = methodShort;
    run.problemShort = problemShort;
    run.problemDim = problemDim;
    run.page = page;
    run.log = log;
    run.innerTabs = inner;
    outputRuns_.push_back(run);

    BatchSummaryUiBundle ui;
    ui.statsTabs = statsTabs_;
    ui.exportStatsBtn = exportStatsBtn_;
    ui.statsTable = statsTable_;
    ui.statsNoteLbl = statsNoteLbl_;
    ui.generateReportBtn = generateReportBtn_;
    ui.exportBestTableBtn = exportBestTableBtn;
    ui.bestTable = bestTable;
    ui.exportMeanTableBtn = exportMeanTableBtn;
    ui.meanTable = meanTable;
    ui.exportBestRankingBtn = exportBestRankingBtn;
    ui.bestRankingTable = bestRankingTable;
    ui.exportMeanRankingBtn = exportMeanRankingBtn;
    ui.meanRankingTable = meanRankingTable;
    ui.exportFinalRankingBtn = exportFinalRankingBtn;
    ui.finalRankingTable = finalRankingTable;
    ui.wilcoxonPairsCombo = wilcoxonPairsCombo_;
    ui.statsAlphaCombo = statsAlphaCombo_;
    ui.exportWilcoxonPlotBtn = exportWilcoxonPlotBtn_;
    ui.wilcoxonPlot = wilcoxonPlot_;
    ui.wilcoxonSummaryLbl = wilcoxonSummaryLbl_;
    ui.exportRankPlotBtn = exportRankPlotBtn_;
    ui.rankPlot = rankPlot_;
    ui.statsSummaryLbl = statsSummaryLbl_;
    ui.statsPairwiseTable = statsPairwiseTable_;
    ui.exportCdPlotBtn = exportCdPlotBtn_;
    ui.cdPlot = cdPlot_;
    ui.cdSummaryLbl = cdSummaryLbl_;
    ui.metricCombo  = wMetricCombo;   // primary (Wilcoxon)
    ui.aggCombo     = wAggCombo;      // primary (Wilcoxon)
    ui.metricCombo2 = fMetricCombo;   // secondary (Friedman)
    ui.aggCombo2    = fAggCombo;      // secondary (Friedman)
    g_batchSummaryUi.insert(page, ui);

    outputTabs_->setTabToolTip(tabIndex, title);
    outputTabs_->setCurrentIndex(tabIndex);
    return tabIndex;
  }

  // Pre-declare all convergence/distribution/sensitivity variables as nullptr.
  // For Sensitivity run mode, conv/dist widgets are NOT created (avoids orphan
  // child widgets appearing inside the QTabWidget content area).
  QLabel*       convInfo          = nullptr;
  QWidget*      convPlot          = nullptr;
  QCheckBox*    overlayMethodChk  = nullptr;
  QCheckBox*    overlayProblemChk = nullptr;
  QRadioButton* xIterRadio        = nullptr;
  QRadioButton* xEvalRadio        = nullptr;
  QComboBox*    themeCombo        = nullptr;
  QComboBox*    gridCombo         = nullptr;
  QRadioButton* distThemeLightRadio = nullptr;
  QRadioButton* distThemeDarkRadio  = nullptr;
  QComboBox*    distThemeCombo      = nullptr;
  QComboBox*    distGridCombo       = nullptr;
  QPushButton*  exportPngBtn        = nullptr;
  QCheckBox*    bandChk             = nullptr;
  QLabel*       distInfo            = nullptr;
  QWidget*      distPlot            = nullptr;
  QPushButton*  exportDistBtn       = nullptr;
  QWidget*      convTab             = nullptr;
  QWidget*      distTab             = nullptr;
  QWidget*      landscapeTab        = nullptr;
  QLabel*       landscapeInfo       = nullptr;
  QWidget*      landscapePlot       = nullptr;
  QPushButton*  landscapePlayBtn    = nullptr;
  QSlider*      landscapeSlider     = nullptr;
  QComboBox*    landscapeThemeCombo = nullptr;
  QPushButton*  exportLandscapeBtn  = nullptr;

  if (!isSensitivityUi) {
  // Convergence
  convTab = new QWidget(inner);
  auto* convLay = new QVBoxLayout(convTab);
  convLay->setContentsMargins(8, 8, 8, 8);

  auto* convTop = new QHBoxLayout();
  overlayMethodChk = new QCheckBox("Overlay all problems for this method", convTab);
  overlayProblemChk = new QCheckBox("Overlay all methods for this problem", convTab);
  overlayProblemChk->setChecked(false);
  connect(overlayProblemChk, &QCheckBox::toggled, this, &MainWindow::onOverlayProblemToggled);
  overlayMethodChk->setChecked(false);
  connect(overlayMethodChk, &QCheckBox::toggled, this, &MainWindow::onOverlayMethodToggled);
  convTop->addWidget(overlayMethodChk);
  convTop->addWidget(overlayProblemChk);

  bandChk = new QCheckBox("Median + IQR (runs)", convTab);
  bandChk->setToolTip("Show a summary convergence curve: median line with an interquartile range (25–75%) band across runs.\n"
                      "Requires multiple independent runs in the convergence CSV.");
  bandChk->setChecked(false);
  connect(bandChk, &QCheckBox::toggled, this, &MainWindow::onConvergenceBandToggled);
  convTop->addWidget(bandChk);

  exportPngBtn = new QPushButton("Export PNG...", convTab);
  exportPngBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  exportPngBtn->setIconSize(QSize(18, 18));
    exportPngBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
connect(exportPngBtn, &QPushButton::clicked, this, &MainWindow::onExportConvergencePng);
  convTop->addWidget(exportPngBtn);

  convTop->addStretch(1);
  convLay->addLayout(convTop);

  const QString dimInfo = (problemDim > 0) ? QString(" | D=%1").arg(problemDim) : QString();
  convInfo = new QLabel(QString("Method: %1 | Problem: %2%3\nNo convergence data loaded.")
                               .arg(m.isEmpty() ? "-" : m)
                               .arg(p.isEmpty() ? "-" : p)
                               .arg(dimInfo),
                             convTab);
  convInfo->setWordWrap(true);
  convLay->addWidget(convInfo);
  convPlot = new ConvergencePlotWidget(convTab);

  // Controls panel to the right of the plot.
  auto* convControlsPanel = new QWidget(convTab);
  auto* convControlsLay   = new QVBoxLayout(convControlsPanel);
  convControlsLay->setContentsMargins(8, 0, 0, 0);
  convControlsLay->setSpacing(10);
  convControlsPanel->setFixedWidth(130);

  auto* themeLbl = new QLabel("Theme", convControlsPanel);
  themeLbl->setAlignment(Qt::AlignLeft);
  themeCombo = new QComboBox(convControlsPanel);
  themeCombo->addItem("Dark");
  themeCombo->addItem("Light");
  themeCombo->addItem("Transparent");
  themeCombo->setCurrentIndex(0);
  connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &MainWindow::onConvergenceThemeToggled);

  auto* gridLbl = new QLabel("Grid", convControlsPanel);
  gridCombo = new QComboBox(convControlsPanel);
  gridCombo->addItem("Sparse");
  gridCombo->addItem("Medium");
  gridCombo->addItem("Dense");
  gridCombo->setCurrentIndex(1);
  connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &MainWindow::onConvergenceGridChanged);

  convControlsLay->addWidget(themeLbl);
  convControlsLay->addWidget(themeCombo);
  convControlsLay->addSpacing(6);
  convControlsLay->addWidget(gridLbl);
  convControlsLay->addWidget(gridCombo);
  convControlsLay->addStretch(1);

  // Plot + controls side by side.
  auto* convPlotRow = new QHBoxLayout();
  convPlotRow->addWidget(convPlot, 1);
  convPlotRow->addWidget(convControlsPanel, 0);
  convLay->addLayout(convPlotRow, 1);

  // X-axis mode selector (below the plot).
  auto* xAxisRow = new QHBoxLayout();
  xAxisRow->addStretch(1);

  xEvalRadio = new QRadioButton("Function Evaluations", convTab);
  xIterRadio = new QRadioButton("Iterations", convTab);
  xEvalRadio->setChecked(true);
  xIterRadio->setChecked(false);

  auto* xAxisGroup = new QButtonGroup(convTab);
  xAxisGroup->setExclusive(true);
  xAxisGroup->addButton(xEvalRadio);
  xAxisGroup->addButton(xIterRadio);

  connect(xIterRadio, &QRadioButton::toggled, this, &MainWindow::onConvergenceXAxisToggled);
  connect(xEvalRadio, &QRadioButton::toggled, this, &MainWindow::onConvergenceXAxisToggled);

  xAxisRow->addWidget(xEvalRadio);
  xAxisRow->addSpacing(12);
  xAxisRow->addWidget(xIterRadio);
  xAxisRow->addStretch(1);
  convLay->addLayout(xAxisRow);

  // Distribution (Boxplot of final best_f across runs)
  distTab = new QWidget(inner);
  auto* distLay = new QVBoxLayout(distTab);
  distLay->setContentsMargins(8, 8, 8, 8);

  auto* distTop = new QHBoxLayout();
  distInfo = new QLabel("Distribution: no data loaded.", distTab);
  distInfo->setWordWrap(true);
  distInfo->hide();
  distTop->addWidget(distInfo, 1);

  exportDistBtn = new QPushButton("Export PNG...", distTab);
  exportDistBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  exportDistBtn->setIconSize(QSize(18, 18));
    exportDistBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
exportDistBtn->setToolTip("Export the distribution plot (PNG, 300 DPI).");
  connect(exportDistBtn, &QPushButton::clicked, this, &MainWindow::onExportDistributionPng);
  distTop->addWidget(exportDistBtn);
  distLay->addLayout(distTop);

  // Plot row: diagram on the left, Theme/Grid controls panel on the right.
  auto* distPlotRow = new QHBoxLayout();

  distPlot = new DistributionPlotWidget(distTab);
  distPlotRow->addWidget(distPlot, 1);

  // Right-side controls panel (Theme + Grid).
  auto* distThemeLbl = new QLabel("Theme", distTab);
  distThemeLbl->setAlignment(Qt::AlignLeft);
  distThemeCombo = new QComboBox(distTab);
  distThemeCombo->addItem("Dark");
  distThemeCombo->addItem("Light");
  distThemeCombo->addItem("Transparent");
  distThemeCombo->setCurrentIndex(0);
  distThemeCombo->setMaximumWidth(110);

  auto* distGridLbl = new QLabel("Grid", distTab);
  distGridCombo = new QComboBox(distTab);
  distGridCombo->addItem("Sparse");
  distGridCombo->addItem("Medium");
  distGridCombo->addItem("Dense");
  distGridCombo->setCurrentIndex(1); // default: Medium
  distGridCombo->setMaximumWidth(110);

  auto* distRightPanel = new QVBoxLayout();
  distRightPanel->setContentsMargins(10, 4, 4, 4);
  distRightPanel->setSpacing(4);
  distRightPanel->addWidget(distThemeLbl);
  distRightPanel->addWidget(distThemeCombo);
  distRightPanel->addSpacing(12);
  distRightPanel->addWidget(distGridLbl);
  distRightPanel->addWidget(distGridCombo);
  distRightPanel->addStretch(1);

  distPlotRow->addLayout(distRightPanel);
  distLay->addLayout(distPlotRow, 1);

  // distThemeCombo change: sync convergence themeCombo (fires onConvergenceThemeToggled which updates all plots).
  connect(distThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [themeCombo](int idx) {
    if (themeCombo) themeCombo->setCurrentIndex(idx); // NOT blocked: lets onConvergenceThemeToggled fire
  });
  // convergence themeCombo change: sync distThemeCombo visually only (plot update handled by onConvergenceThemeToggled).
  connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [distThemeCombo](int idx) {
    QSignalBlocker b(distThemeCombo); distThemeCombo->setCurrentIndex(idx);
  });
  // distGridCombo change: sync convergence gridCombo (fires onConvergenceGridChanged which updates all plots).
  connect(distGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [gridCombo](int idx) {
    if (gridCombo) gridCombo->setCurrentIndex(idx); // NOT blocked: lets onConvergenceGridChanged fire
  });
  // convergence gridCombo change: sync distGridCombo visually only.
  connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [distGridCombo](int idx) {
    if (!distGridCombo) return;
    QSignalBlocker b(distGridCombo);
    distGridCombo->setCurrentIndex(idx);
  });


  auto* distHint = new QLabel("Boxplot: whiskers = 1.5×IQR, dots = outliers. Values are final best_f per run from the convergence CSV.", distTab);
  distHint->setWordWrap(true);
  distHint->setStyleSheet("opacity: 0.75;");
  distHint->hide();
  distLay->addWidget(distHint);

  } // end if (!isSensitivityUi)

  // Landscape (2D contour + best-so-far trajectory animation).
  // Only built for non-sensitivity runs on exactly 2D problems — the contour is
  // sampled directly from the in-process Problem instance, and the animated
  // trajectory is read from the CLI's "<prefix>_landscape.csv" (written only
  // when dim==2; see src/main.cpp).
  if (!isSensitivityUi && problemDim == 2) {
    landscapeTab = new QWidget(inner);
    auto* lsLay = new QVBoxLayout(landscapeTab);
    lsLay->setContentsMargins(8, 8, 8, 8);

    auto* lsTop = new QHBoxLayout();
    landscapeInfo = new QLabel("Landscape: no data loaded.", landscapeTab);
    landscapeInfo->setWordWrap(true);
    lsTop->addWidget(landscapeInfo, 1);

    landscapePlayBtn = new QPushButton(QString::fromUtf8("\xE2\x96\xB6 Play"), landscapeTab);
    landscapePlayBtn->setCheckable(true);
    landscapePlayBtn->setEnabled(false);
    connect(landscapePlayBtn, &QPushButton::toggled, this, &MainWindow::onLandscapePlayToggled);
    lsTop->addWidget(landscapePlayBtn);

    exportLandscapeBtn = new QPushButton("Export PNG...", landscapeTab);
    exportLandscapeBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportLandscapeBtn->setIconSize(QSize(18, 18));
    exportLandscapeBtn->setEnabled(false);
    exportLandscapeBtn->setToolTip("Export the landscape plot (PNG, 300 DPI).");
    connect(exportLandscapeBtn, &QPushButton::clicked, this, &MainWindow::onExportLandscapePng);
    lsTop->addWidget(exportLandscapeBtn);
    lsLay->addLayout(lsTop);

    auto* lsPlotRow = new QHBoxLayout();
    landscapePlot = new LandscapeWidget(landscapeTab);
    lsPlotRow->addWidget(landscapePlot, 1);

    auto* lsControlsPanel = new QWidget(landscapeTab);
    auto* lsControlsLay = new QVBoxLayout(lsControlsPanel);
    lsControlsLay->setContentsMargins(8, 0, 0, 0);
    lsControlsLay->setSpacing(10);
    lsControlsPanel->setFixedWidth(120);

    auto* lsThemeLbl = new QLabel("Theme", lsControlsPanel);
    lsThemeLbl->setAlignment(Qt::AlignLeft);
    landscapeThemeCombo = new QComboBox(lsControlsPanel);
    landscapeThemeCombo->addItem("Dark");
    landscapeThemeCombo->addItem("Light");
    landscapeThemeCombo->addItem("Transparent");
    landscapeThemeCombo->setCurrentIndex(0);
    connect(landscapeThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [landscapePlot](int index) {
      if (auto* lw = dynamic_cast<LandscapeWidget*>(landscapePlot)) {
        LandscapeWidget::ThemeMode tm = LandscapeWidget::ThemeMode::Dark;
        if (index == 1) tm = LandscapeWidget::ThemeMode::Light;
        else if (index == 2) tm = LandscapeWidget::ThemeMode::Transparent;
        lw->setThemeMode(tm);
      }
    });

    lsControlsLay->addWidget(lsThemeLbl);
    lsControlsLay->addWidget(landscapeThemeCombo);
    lsControlsLay->addStretch(1);

    lsPlotRow->addWidget(lsControlsPanel, 0);
    lsLay->addLayout(lsPlotRow, 1);

    landscapeSlider = new QSlider(Qt::Horizontal, landscapeTab);
    landscapeSlider->setEnabled(false);
    landscapeSlider->setRange(0, 0);
    connect(landscapeSlider, &QSlider::valueChanged, this, &MainWindow::onLandscapeSliderMoved);
    lsLay->addWidget(landscapeSlider);

    auto* lsHint = new QLabel(
      "Contour = objective landscape (blue = better, red = worse). Marker + trail = the best-so-far "
      "point across iterations for this run — illustrates exploration (early, spread out) vs. "
      "exploitation (later, converging). For the local methods (gd/lbfgs/bfgs/nm/cg/ncg/adam) it is the "
      "path of the single local search from its random start.",
      landscapeTab);
    lsHint->setWordWrap(true);
    lsHint->setStyleSheet("opacity: 0.75;");
    lsLay->addWidget(lsHint);
  }


  // -----------------------------------------------------------------------
  // Sensitivity widgets and inner-tab assembly — layout depends on run mode.
  // -----------------------------------------------------------------------

  // Declare all sensitivity pointers here; populated only when needed.
  QComboBox*    sensParamCombo = nullptr;
  QWidget*      sensPlot       = nullptr;
  QTableWidget* sensTable      = nullptr;
  QTextEdit*    sensText        = nullptr;
  QRadioButton* sensThemeLightRadio = nullptr;
  QRadioButton* sensThemeDarkRadio  = nullptr;
  QComboBox*    sensGridCombo   = nullptr;
  QPushButton*  exportSensBtn   = nullptr;
  QPushButton*  generateSensReportBtn = nullptr;

  // Declared here for the same reason as the sensitivity pointers above;
  // populated only when isPortfolioUi (see the if/else-if/else block below).
  QLabel*       portfolioInfoLbl       = nullptr;
  QLabel*       portfolioFriedmanLbl   = nullptr;
  QLabel*       portfolioStatsHintLbl  = nullptr;
  QTableWidget* portfolioSegmentsTable = nullptr;
  QTableWidget* portfolioSummaryTable  = nullptr;
  QTableWidget* portfolioRankingTable  = nullptr;
  QTableWidget* portfolioFriedmanTable = nullptr;
  QTableWidget* portfolioWilcoxonTable = nullptr;
  QTableWidget* portfolioUsageTable    = nullptr;
  QWidget*      portfolioWilcoxonPlot  = nullptr;
  QWidget*      portfolioFriedmanPlot  = nullptr;
  QComboBox*    portfolioMetricCombo   = nullptr;
  QComboBox*    portfolioMetricCombo2  = nullptr;
  QTableWidget* portfolioAggregateTable = nullptr;

  if (isSensitivityUi) {
    // ---- Tab 1: "Results table" -----------------------------------------
    auto* resTab = new QWidget(inner);
    auto* resLay = new QVBoxLayout(resTab);
    resLay->setContentsMargins(4, 4, 4, 4);

    // Top bar: metric label + Export table button (right-aligned)
    auto* resTop = new QHBoxLayout();
    auto* sensMetricLbl = new QLabel("Metric: Mean best_f", resTab);
    auto* exportTableBtn = new QPushButton("Export table...", resTab);
    exportTableBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportTableBtn->setIconSize(QSize(18, 18));
    exportTableBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    exportTableBtn->setToolTip("Export the results table to a CSV file.");
    connect(exportTableBtn, &QPushButton::clicked, this, [this, isMethodParamSensitivity]() {
      auto* tab = activeOutputTab();
      if (!tab || !tab->sensitivitySummaryTable) return;

      QString defName;
      if (isMethodParamSensitivity) {
        // Same name as the tab itself, e.g. "rastrigin_50.xlsx".
        const QString pName = sanitizeUnderscoreName(tab->problemShort.isEmpty() ? "problem" : tab->problemShort);
        defName = (tab->problemDim > 0) ? QString("%1_%2.xlsx").arg(pName).arg(tab->problemDim)
                                         : QString("%1.xlsx").arg(pName);
      } else {
        // Build default filename: sensitivity_<param>_<problem>.xlsx
        const QString param   = (tab->sensitivityParamCombo && !tab->sensitivityParamCombo->currentText().isEmpty())
                                  ? tab->sensitivityParamCombo->currentText().trimmed()
                                  : "param";
        const QString problem = tab->problemShort.isEmpty() ? "problem" : tab->problemShort;
        defName = QString("sensitivity_%1_%2.xlsx").arg(param, problem);
      }

      const QString out = QFileDialog::getSaveFileName(
        this, "Export Results Table",
        QDir(tab->runtimeWorkingDir.isEmpty() ? QDir::homePath() : tab->runtimeWorkingDir).filePath(defName),
        "Excel Workbook (*.xlsx);;All files (*)");
      if (out.isEmpty()) return;

      auto* tbl = tab->sensitivitySummaryTable;

      // Try native xlsx writer first.
      QTableWidget* saved = statsTable_;
      statsTable_ = tbl;
      const bool nativeOk = exportStatsToXlsxNative(out);
      statsTable_ = saved;
      if (nativeOk) return;

      // Fall back: write CSV temp file, then convert via Python/openpyxl.
      QTemporaryFile tmp(QDir::tempPath() + "/optimsolution_sens_XXXXXX.csv");
      tmp.setAutoRemove(true);
      if (!tmp.open()) {
        QMessageBox::warning(this, "Export", "Failed to create a temporary file.");
        return;
      }
      QTextStream ts(&tmp);
      ts.setEncoding(QStringConverter::Utf8);
      QStringList hdr;
      for (int c = 0; c < tbl->columnCount(); ++c)
        hdr << (tbl->horizontalHeaderItem(c) ? tbl->horizontalHeaderItem(c)->text() : QString());
      ts << hdr.join(",") << "\n";
      for (int r = 0; r < tbl->rowCount(); ++r) {
        QStringList row;
        for (int c = 0; c < tbl->columnCount(); ++c) {
          QString cell = tbl->item(r, c) ? tbl->item(r, c)->text() : QString();
          if (cell.contains(',') || cell.contains('"') || cell.contains('\n')) {
            cell.replace("\"", "\"\"");
            cell = "\"" + cell + "\"";
          }
          row << cell;
        }
        ts << row.join(",") << "\n";
      }
      ts.flush(); tmp.flush();

      if (!exportStatsToXlsxViaPython(out, tmp.fileName())) {
        const QString csvOut = out.left(out.size() - 5) + ".csv";
        QFile::remove(csvOut);
        QFile::copy(tmp.fileName(), csvOut);
        QMessageBox::information(this, "Export",
          "XLSX export requires Python + openpyxl.\n"
          "A CSV file was saved instead:\n" + csvOut);
      }
    });

    generateSensReportBtn = new QPushButton("Generate report (PDF/Word)...", resTab);
    generateSensReportBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    generateSensReportBtn->setIconSize(QSize(18, 18));
    generateSensReportBtn->setToolTip(
      "Collects the sensitivity results table, the parameter chart, and the summary log into a single "
      "PDF or Word-compatible (.odt) document.");
    connect(generateSensReportBtn, &QPushButton::clicked, this, [this, page]() {
      int idx = -1;
      for (int i = 0; i < outputRuns_.size(); ++i) {
        if (outputRuns_[i].page == page) { idx = i; break; }
      }
      if (idx < 0) return;

      const QString html = buildSensitivityReportHtml(idx);
      if (html.isEmpty()) {
        QMessageBox::warning(this, "Generate report", "No sensitivity data is available yet for this run.");
        return;
      }
      const QString out = QFileDialog::getSaveFileName(
        this, "Generate sensitivity report",
        QDir::homePath() + "/sensitivity_report.pdf",
        "PDF Document (*.pdf);;Word-compatible Document (*.odt)");
      if (out.isEmpty()) return;
      if (!writeReportDocument(html, out)) {
        QMessageBox::warning(this, "Generate report", "Failed to write the report file.");
      }
    });

    resTop->addWidget(sensMetricLbl);
    resTop->addStretch(1);
    resTop->addWidget(generateSensReportBtn);
    resTop->addWidget(exportTableBtn);
    resLay->addLayout(resTop);

    sensTable = new QTableWidget(resTab);
    sensTable->setColumnCount(8);
    sensTable->setHorizontalHeaderLabels(QStringList()
      << "Value"
      << "Mean best_f"
      << "SD"
      << "Min best_f"
      << "Max best_f"
      << "Main effect"
      << "Rank"
      << "N (runs)");
    sensTable->verticalHeader()->setVisible(false);
    sensTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    sensTable->setSelectionMode(QAbstractItemView::SingleSelection);
    sensTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    sensTable->setAlternatingRowColors(true);
    sensTable->horizontalHeader()->setStretchLastSection(false);
    sensTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    resLay->addWidget(sensTable, 2);

    sensText = new QTextEdit(resTab);
    sensText->setReadOnly(true);
    sensText->setLineWrapMode(QTextEdit::NoWrap);
    sensText->setMaximumHeight(120);
    sensText->setPlainText("Sensitivity results will appear here after the run.");
    resLay->addWidget(sensText, 1);

    // ---- Tab 2: "Parameter Sensitivity" (bar chart) ----------------------
    auto* sensChartTab = new QWidget(inner);
    auto* sensChartLay = new QVBoxLayout(sensChartTab);
    sensChartLay->setContentsMargins(4, 4, 4, 4);

    // Top bar: Parameter selector + Export PNG (belong here with the chart)
    auto* chartTop = new QHBoxLayout();
    auto* sensParamLbl = new QLabel("Parameter:", sensChartTab);
    sensParamCombo = new QComboBox(sensChartTab);
    exportSensBtn  = new QPushButton("Export PNG...", sensChartTab);
    exportSensBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    exportSensBtn->setIconSize(QSize(18, 18));
    exportSensBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    exportSensBtn->setToolTip("Export the sensitivity plot (PNG, 300 DPI).");
    connect(exportSensBtn, &QPushButton::clicked, this, &MainWindow::onExportSensitivityPng);

    chartTop->addWidget(sensParamLbl);
    chartTop->addWidget(sensParamCombo);
    chartTop->addStretch(1);
    chartTop->addWidget(exportSensBtn);
    sensChartLay->addLayout(chartTop);

    connect(sensParamCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onOutputSensitivityParamChanged);

    sensPlot = new SensitivityBarWidget(sensChartTab);
    sensChartLay->addWidget(sensPlot, 3);

    // Display options row (Theme + Grid)
    auto* sensDispRow = new QHBoxLayout();
    sensDispRow->addStretch(1);
    auto* sensThemeLbl = new QLabel("Theme", sensChartTab);
    sensThemeLbl->setAlignment(Qt::AlignLeft);
    auto* sensThemeCombo = new QComboBox(sensChartTab);
    sensThemeCombo->addItem("Dark");
    sensThemeCombo->addItem("Light");
    sensThemeCombo->addItem("Transparent");
    sensThemeCombo->setCurrentIndex(0);
    auto* sensGridLbl = new QLabel("Grid", sensChartTab);
    sensGridCombo = new QComboBox(sensChartTab);
    sensGridCombo->addItem("Sparse");
    sensGridCombo->addItem("Medium");
    sensGridCombo->addItem("Dense");
    sensGridCombo->setCurrentIndex(1);

    connect(sensThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [sp = sensPlot](int index) {
              SensitivityBarWidget::ThemeMode tm = SensitivityBarWidget::ThemeMode::Dark;
              if (index == 1) tm = SensitivityBarWidget::ThemeMode::Light;
              else if (index == 2) tm = SensitivityBarWidget::ThemeMode::Transparent;
              if (auto* w = dynamic_cast<SensitivityBarWidget*>(sp))
                w->setThemeMode(tm);
            });
    connect(sensGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [sp = sensPlot](int idx) {
              SensitivityBarWidget::GridDensity gd = SensitivityBarWidget::GridDensity::Medium;
              if (idx == 0) gd = SensitivityBarWidget::GridDensity::Sparse;
              else if (idx == 2) gd = SensitivityBarWidget::GridDensity::Dense;
              if (auto* w = dynamic_cast<SensitivityBarWidget*>(sp))
                w->setGridDensity(gd);
            });

    sensDispRow->addWidget(sensThemeLbl);
    sensDispRow->addSpacing(8);
    sensDispRow->addWidget(sensThemeCombo);
    sensDispRow->addSpacing(22);
    sensDispRow->addWidget(sensGridLbl);
    sensDispRow->addSpacing(8);
    sensDispRow->addWidget(sensGridCombo);
    sensDispRow->addStretch(1);
    sensChartLay->addLayout(sensDispRow);

    // Sensitivity run: Results + Chart + Log only (Convergence/Distribution not meaningful here)
    inner->addTab(resTab,       "Results table");
    inner->addTab(sensChartTab, "Parameter Sensitivity");
    inner->addTab(logTab,       "Run Log");

  } else if (isPortfolioUi) {
    // Convergence/Distribution/Landscape were already constructed above
    // (unconditionally, before this if/else-if/else) but are deliberately
    // NOT added to `inner` here -- they assume the single-(method,problem)
    // *_convergence.csv/_landscape.csv schema, which --portfolio does not
    // write. A widget that is constructed with a parent but never added to
    // that parent's layout/tab-stack can still end up painted (Qt does not
    // guarantee it stays hidden just because no layout manages it), so they
    // are explicitly hidden here to prevent them bleeding through on top of
    // the tabs actually shown below.
    convTab->hide();
    distTab->hide();
    if (landscapeTab) landscapeTab->hide();

    // Portfolio run (Mode 8): Convergence/Distribution/Landscape assume the
    // single-(method,problem) *_convergence.csv/_landscape.csv schema that
    // --portfolio does not write -- showing those tabs here would just be
    // empty/misleading. Six dedicated views instead, all filled in by
    // tryLoadPortfolioForTab() once the process finishes.
    const QString exportStem = sanitizeExportFileStem(title);

    // Builds one "header label (optional) + table + Export button" tab.
    // Every portfolio view has the same shape, so they share one builder
    // rather than six near-identical copies.
    auto makePortfolioTableTab = [&](const QString& tabTitle,
                                     const QString& exportSuffix,
                                     QLabel** outLabel,
                                     const QString& labelText) -> QTableWidget* {
      auto* tab = new QWidget(inner);
      auto* lay = new QVBoxLayout(tab);
      lay->setContentsMargins(6, 6, 6, 6);
      if (outLabel) {
        *outLabel = new QLabel(labelText, tab);
        (*outLabel)->setWordWrap(true);
        lay->addWidget(*outLabel);
      }
      auto* table = new QTableWidget(tab);
      table->setEditTriggers(QAbstractItemView::NoEditTriggers);
      table->setSelectionBehavior(QAbstractItemView::SelectRows);
      table->setAlternatingRowColors(true);
      lay->addWidget(table, 1);

      auto* row = new QHBoxLayout();
      auto* btn = new QPushButton("Export table...", tab);
      btn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
      btn->setIconSize(QSize(18, 18));
      row->addWidget(btn);
      row->addStretch(1);
      lay->addLayout(row);
      connect(btn, &QPushButton::clicked, this, [this, table, tabTitle, exportStem, exportSuffix]() {
        exportTableWidgetToXlsxOrCsv(table, QString("Export %1").arg(tabTitle.toLower()),
                                      QDir::homePath() + "/" + exportStem + exportSuffix);
      });

      inner->addTab(tab, tabTitle);
      return table;
    };

    // Order is deliberate: the comparison verdict first (Ranking ->
    // Friedman -> Wilcoxon), then the descriptive detail behind it
    // (per-run Summary, per-segment trace, method usage), then the log.
    portfolioRankingTable = makePortfolioTableTab(
      "Ranking", "_ranking.xlsx", &portfolioStatsHintLbl,
      "Comparison of every switching variant against its baselines, over the same per-run seeds. "
      "Requires 'with_baselines' in the [portfolio] section of the settings file; with a single "
      "variant there is nothing to rank and these three tabs stay empty.");

    // Friedman and Wilcoxon each get a plot (RankPlotWidget / WilcoxonBoxPlotWidget --
    // the same widget classes Batch run's own Friedman/Wilcoxon tabs use, see the
    // "System" tab construction above) with a Metric/Aggregate/Theme/Grid control
    // panel to the right and a 300 DPI PNG export button, mirroring that layout
    // exactly, plus the underlying data table below it. Metric is synced between
    // the two tabs, same as Batch run's wMetricCombo<->fMetricCombo.
    QComboBox* pfMetricComboF = nullptr; // Friedman's combo (declared first: captured by Wilcoxon's sync lambda)
    QComboBox* pfMetricComboW = nullptr;

    auto makePortfolioPlotTab = [&](const QString& tabTitle, const QString& exportPngSuffix,
                                     const QString& exportPngDialogTitle, const QString& exportXlsxSuffix,
                                     QWidget* plotWidget, QComboBox*& outMetricCombo,
                                     QLabel** outSummaryLbl, const QString& summaryText) -> QTableWidget* {
      auto* tab = new QWidget(inner);
      auto* lay = new QVBoxLayout(tab);
      lay->setContentsMargins(6, 6, 6, 6);

      if (outSummaryLbl) {
        *outSummaryLbl = new QLabel(summaryText, tab);
        (*outSummaryLbl)->setWordWrap(true);
        lay->addWidget(*outSummaryLbl);
      }

      auto* topRow = new QHBoxLayout();
      auto* exportPngBtn = new QPushButton("Export plot (PNG, 300 DPI)...", tab);
      exportPngBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
      exportPngBtn->setIconSize(QSize(18, 18));
      const QString exportStem = sanitizeExportFileStem(title);
      connect(exportPngBtn, &QPushButton::clicked, this, [this, plotWidget, exportPngDialogTitle, exportStem, exportPngSuffix]() {
        const QString out = QFileDialog::getSaveFileName(
          this, exportPngDialogTitle, QDir::homePath() + "/" + exportStem + exportPngSuffix, "PNG Image (*.png)");
        if (out.isEmpty()) return;
        if (!exportWidgetToPng300Dpi(plotWidget, out)) {
          QMessageBox::warning(this, "Export", "Failed to export the plot.");
        }
      });
      topRow->addWidget(exportPngBtn);
      topRow->addStretch(1);
      lay->addLayout(topRow);

      auto* controlsPanel = new QWidget(tab);
      auto* controlsLay = new QVBoxLayout(controlsPanel);
      controlsLay->setContentsMargins(8, 0, 0, 0);
      controlsLay->setSpacing(10);
      controlsPanel->setFixedWidth(150);

      auto* metricLbl = new QLabel("Metric", controlsPanel);
      auto* metricCombo = new QComboBox(controlsPanel);
      metricCombo->addItem("final_f (lower better)", QVariant("final_f"));
      metricCombo->addItem("evals_used", QVariant("evals_used"));
      metricCombo->addItem("num_switches", QVariant("num_switches"));
      metricCombo->setCurrentIndex(0);
      outMetricCombo = metricCombo;

      // "Aggregate" is part of every other mode's plot control panel (see
      // Batch run's own Wilcoxon/Friedman tabs above), so it is kept here
      // for a consistent layout -- but every Mode-8 metric is already a
      // single scalar per run (unlike Batch's "iteration/evals of best_f",
      // which need a within-run aggregate), so there is nothing for it to
      // control here and it stays disabled.
      auto* aggLbl = new QLabel("Aggregate", controlsPanel);
      auto* aggCombo = new QComboBox(controlsPanel);
      aggCombo->addItems(QStringList() << "Mean" << "Median" << "Min" << "Max");
      aggCombo->setEnabled(false);
      aggCombo->setToolTip("Not applicable: every Portfolio metric is already one value per run.");

      auto* themeLbl = new QLabel("Theme", controlsPanel);
      auto* themeCombo = new QComboBox(controlsPanel);
      themeCombo->addItem("Dark"); themeCombo->addItem("Light"); themeCombo->addItem("Transparent");
      themeCombo->setCurrentIndex(0);

      auto* gridLbl = new QLabel("Grid", controlsPanel);
      auto* gridCombo = new QComboBox(controlsPanel);
      gridCombo->addItem("Sparse"); gridCombo->addItem("Medium"); gridCombo->addItem("Dense");
      gridCombo->setCurrentIndex(1);

      connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plotWidget](int index) {
        if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(plotWidget)) {
          auto tm = WilcoxonBoxPlotWidget::ThemeMode::Dark;
          if (index == 1) tm = WilcoxonBoxPlotWidget::ThemeMode::Light;
          else if (index == 2) tm = WilcoxonBoxPlotWidget::ThemeMode::Transparent;
          wp->setThemeMode(tm);
        } else if (auto* rp = dynamic_cast<RankPlotWidget*>(plotWidget)) {
          auto tm = RankPlotWidget::ThemeMode::Dark;
          if (index == 1) tm = RankPlotWidget::ThemeMode::Light;
          else if (index == 2) tm = RankPlotWidget::ThemeMode::Transparent;
          rp->setThemeMode(tm);
        }
      });
      connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plotWidget](int index) {
        if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(plotWidget)) {
          auto gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
          if (index == 0) gd = WilcoxonBoxPlotWidget::GridDensity::Sparse;
          else if (index == 2) gd = WilcoxonBoxPlotWidget::GridDensity::Dense;
          wp->setGridDensity(gd);
        } else if (auto* rp = dynamic_cast<RankPlotWidget*>(plotWidget)) {
          auto gd = RankPlotWidget::GridDensity::Medium;
          if (index == 0) gd = RankPlotWidget::GridDensity::Sparse;
          else if (index == 2) gd = RankPlotWidget::GridDensity::Dense;
          rp->setGridDensity(gd);
        }
      });

      controlsLay->addWidget(metricLbl);
      controlsLay->addWidget(metricCombo);
      controlsLay->addSpacing(2);
      controlsLay->addWidget(aggLbl);
      controlsLay->addWidget(aggCombo);
      controlsLay->addSpacing(6);
      controlsLay->addWidget(themeLbl);
      controlsLay->addWidget(themeCombo);
      controlsLay->addSpacing(6);
      controlsLay->addWidget(gridLbl);
      controlsLay->addWidget(gridCombo);
      controlsLay->addStretch(1);

      auto* plotRow = new QHBoxLayout();
      plotRow->addWidget(plotWidget, 1);
      plotRow->addWidget(controlsPanel, 0);
      lay->addLayout(plotRow, 2);

      auto* table = new QTableWidget(tab);
      table->setEditTriggers(QAbstractItemView::NoEditTriggers);
      table->setSelectionBehavior(QAbstractItemView::SelectRows);
      table->setAlternatingRowColors(true);
      table->setMaximumHeight(220);
      lay->addWidget(table, 1);

      auto* exportTableBtn = new QPushButton("Export table...", tab);
      exportTableBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
      exportTableBtn->setIconSize(QSize(18, 18));
      connect(exportTableBtn, &QPushButton::clicked, this, [this, table, exportStem, exportXlsxSuffix]() {
        exportTableWidgetToXlsxOrCsv(table, "Export table",
                                      QDir::homePath() + "/" + exportStem + exportXlsxSuffix);
      });
      auto* tblBtnRow = new QHBoxLayout();
      tblBtnRow->addWidget(exportTableBtn);
      tblBtnRow->addStretch(1);
      lay->addLayout(tblBtnRow);

      inner->addTab(tab, tabTitle);
      return table;
    };

    portfolioFriedmanPlot = new RankPlotWidget(inner);
    portfolioFriedmanTable = makePortfolioPlotTab(
      "Friedman", "_friedman_plot.png", "Export Friedman plot", "_friedman.xlsx",
      portfolioFriedmanPlot, pfMetricComboF, &portfolioFriedmanLbl,
      "Omnibus test: are the variants all equivalent? A small p-value rejects that, which is what "
      "licenses reading the pairwise Wilcoxon tab.");

    portfolioWilcoxonPlot = new WilcoxonBoxPlotWidget(inner);
    portfolioWilcoxonTable = makePortfolioPlotTab(
      "Wilcoxon (pairwise)", "_wilcoxon_plot.png", "Export Wilcoxon plot", "_wilcoxon.xlsx",
      portfolioWilcoxonPlot, pfMetricComboW, nullptr, QString());

    portfolioMetricCombo = pfMetricComboW;
    portfolioMetricCombo2 = pfMetricComboF;

    // Sync the two Metric combos (Wilcoxon <-> Friedman), same convention
    // as Batch run's wMetricCombo<->fMetricCombo above, and rebuild both
    // plots whenever either changes.
    connect(pfMetricComboW, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, pfMetricComboF, index = int(outputRuns_.size())](int idx) {
      if (pfMetricComboF) { QSignalBlocker b(pfMetricComboF); pfMetricComboF->setCurrentIndex(idx); }
      rebuildPortfolioPlotsForTab(index);
    });
    connect(pfMetricComboF, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, pfMetricComboW, index = int(outputRuns_.size())](int idx) {
      if (pfMetricComboW) { QSignalBlocker b(pfMetricComboW); pfMetricComboW->setCurrentIndex(idx); }
      rebuildPortfolioPlotsForTab(index);
    });

    portfolioSummaryTable = makePortfolioTableTab(
      "Summary (per run)", "_summary.xlsx", &portfolioInfoLbl,
      "Results will appear here once the run finishes.");
    portfolioSegmentsTable = makePortfolioTableTab(
      "Segments (per switch)", "_segments.xlsx", nullptr, QString());
    portfolioUsageTable = makePortfolioTableTab(
      "Method usage", "_method_usage.xlsx", nullptr, QString());
    portfolioAggregateTable = makePortfolioTableTab(
      "Aggregate (all problems)", "_aggregate.xlsx", nullptr,
      "One row per (problem, variant): mean/median/sd of the final result, mean evals and switches, "
      "and its rank among variants on that problem specifically. Populated only when multiple problems "
      "were selected -- with a single problem this is identical to the Summary tab and stays empty.");

    inner->addTab(logTab, "Run Log");

  } else {
    // Single run / Batch job: Convergence + Final Distribution + [Landscape] + log
    inner->addTab(convTab,  "Convergence");
    inner->addTab(distTab,  "Final Distribution");
    if (landscapeTab) inner->addTab(landscapeTab, "Landscape (2D)");
    inner->addTab(logTab,   isBatchUi ? "Job Log" : "Run Log");
  }

  const int tabIndex = outputTabs_->addTab(page, title);
  if (tabIndex != int(outputRuns_.size())) {
    // This should never happen unless tabs are inserted/reordered externally.
    CrashLog::append(QString("Output tab index mismatch: tabIndex=%1 runs=%2").arg(tabIndex).arg(outputRuns_.size()));
  }

  OutputRunTab run;
  run.title = title;
  run.methodShort = m;
  run.problemShort = p;
  run.problemDim = problemDim;
  run.isMethodParamSensitivity = isMethodParamSensitivity;
  run.page = page;
  run.log = log;
  run.convergenceInfo = convInfo;
  run.convergencePlot = convPlot;
  run.innerTabs = inner;
  run.sensitivityParamCombo = sensParamCombo;
  run.sensitivityPlot = sensPlot;
  run.sensitivitySummaryTable = sensTable;
  run.sensitivityLog = sensText;
  run.overlayMethodChk = overlayMethodChk;
  run.overlayProblemChk = overlayProblemChk;
  run.xAxisIterRadio = xIterRadio;
  run.xAxisEvalRadio = xEvalRadio;
  run.themeCombo      = themeCombo;
  run.gridDensityCombo = gridCombo;
  run.distThemeLightRadio = distThemeLightRadio;
  run.distThemeDarkRadio = distThemeDarkRadio;
  run.distThemeCombo = distThemeCombo;
  run.distGridDensityCombo = distGridCombo;
  run.sensThemeLightRadio = sensThemeLightRadio;
  run.sensThemeDarkRadio = sensThemeDarkRadio;
  run.sensGridDensityCombo = sensGridCombo;
  run.exportConvergencePngBtn = exportPngBtn;
  run.convBandChk = bandChk;
  run.distributionInfo = distInfo;
  run.distributionPlot = distPlot;
  run.exportDistributionPngBtn = exportDistBtn;
  run.exportSensitivityPngBtn = exportSensBtn;
  run.generateSensReportBtn = generateSensReportBtn;
  run.landscapeInfo = landscapeInfo;
  run.landscapePlot = landscapePlot;
  run.landscapePlayBtn = landscapePlayBtn;
  run.landscapeSlider = landscapeSlider;
  run.landscapeThemeCombo = landscapeThemeCombo;
  run.exportLandscapePngBtn = exportLandscapeBtn;
  run.landscapeLoaded = false;
  run.landscapeGridReady = false;
  run.convLoaded = false;
  run.isPortfolioRun = isPortfolioUi;
  run.portfolioInfoLbl = portfolioInfoLbl;
  run.portfolioFriedmanLbl = portfolioFriedmanLbl;
  run.portfolioStatsHintLbl = portfolioStatsHintLbl;
  run.portfolioSegmentsTable = portfolioSegmentsTable;
  run.portfolioSummaryTable = portfolioSummaryTable;
  run.portfolioRankingTable = portfolioRankingTable;
  run.portfolioFriedmanTable = portfolioFriedmanTable;
  run.portfolioWilcoxonTable = portfolioWilcoxonTable;
  run.portfolioUsageTable = portfolioUsageTable;
  run.portfolioAggregateTable = portfolioAggregateTable;
  run.portfolioWilcoxonPlot = portfolioWilcoxonPlot;
  run.portfolioFriedmanPlot = portfolioFriedmanPlot;
  run.portfolioMetricCombo = portfolioMetricCombo;
  run.portfolioMetricCombo2 = portfolioMetricCombo2;
  run.portfolioLoaded = false;
  outputRuns_.push_back(std::move(run));

  return tabIndex;
}

MainWindow::OutputRunTab* MainWindow::outputTab(int index) {
  if (index < 0 || index >= int(outputRuns_.size())) return nullptr;
  return &outputRuns_[index];
}


// Builds the Results tab's summary table with ALL swept parameters at
// once (a leading "Parameter" column identifies each row's sweep), so the
// table doesn't depend on -- and isn't silently overwritten by -- whatever
// single parameter the Parameter Sensitivity chart tab's combo box
// currently has selected. Called once when a tab's sensitivity results
// are (re)loaded (see tryLoadSensitivityForTab()); the chart itself still
// shows one parameter at a time via renderSensitivityForParam(), which is
// the natural/correct way to display a bar chart.
//
// Re-reads the CSV once per parameter (paramCols is typically 2-6 entries
// and these sweep files are small, so this is simpler and safer than
// trying to share a single parse pass across differently-keyed groups).
// Rank and "Main effect" are computed WITHIN each parameter's own group of
// swept values, not mixed across different parameters' value scales.
void MainWindow::populateSensitivitySummaryAllParams(int tabIndex, const QString& csvPath,
                                                       const QStringList& paramCols) {
  if (tabIndex < 0 || tabIndex >= static_cast<int>(outputRuns_.size())) return;
  OutputRunTab& tab = outputRuns_[tabIndex];
  if (!tab.sensitivitySummaryTable) return;

  tab.sensitivitySummaryTable->setRowCount(0);
  if (csvPath.isEmpty() || !QFileInfo::exists(csvPath) || paramCols.isEmpty()) return;

  struct Agg {
    long long n = 0;
    double mean = 0.0;
    double m2   = 0.0;
    double minV =  std::numeric_limits<double>::quiet_NaN();
    double maxV = -std::numeric_limits<double>::quiet_NaN();

    void addBatch(long long nn, double m, double sd, double vmin, double vmax) {
      if (nn <= 0) nn = 1;
      const double var = (nn > 1 && sd > 0.0) ? (sd * sd) : 0.0;
      const double m2b = (nn > 1) ? (var * double(nn - 1)) : 0.0;

      if (n == 0) {
        n = nn; mean = m; m2 = m2b; minV = vmin; maxV = vmax;
        return;
      }
      const double delta = m - mean;
      const long long newN = n + nn;
      mean += delta * (double(nn) / double(newN));
      m2   += m2b + delta * delta * (double(n) * double(nn) / double(newN));
      n    = newN;
      if (!std::isnan(vmin)) minV = std::isnan(minV) ? vmin : std::min(minV, vmin);
      if (!std::isnan(vmax)) maxV = std::isnan(maxV) ? vmax : std::max(maxV, vmax);
    }
    double sd() const { return (n <= 1) ? 0.0 : std::sqrt(m2 / double(n - 1)); }
  };

  struct Row { QString param; double x; double mean; double sd; double minV; double maxV; int n; };
  QVector<Row> allRows;

  for (const QString& paramName : paramCols) {
    QFile f(csvPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);

    const QString headerLine = ts.readLine().trimmed();
    QStringList headers = headerLine.split(',', Qt::KeepEmptyParts);
    for (auto& h : headers) h = h.trimmed();

    auto findCol = [&](const QStringList& candidates) -> int {
      for (const auto& c : candidates)
        for (int i = 0; i < headers.size(); ++i)
          if (headers[i].trimmed().toLower() == c.trimmed().toLower()) return i;
      return -1;
    };

    const int paramIdx = findCol({paramName});
    const int meanIdx  = findCol({"mean_best_f", "mean_f", "meanbestf", "mean"});
    const int sdIdx    = findCol({"stdev_best_f", "stdev_f", "sd_best_f", "sd_f",
                                   "std_best_f", "std_f", "stdev", "sd", "std"});
    const int minIdx   = findCol({"min_best_f", "min_f", "min_best", "min"});
    const int maxIdx   = findCol({"max_best_f", "max_f", "max_best", "max"});
    const int runsIdx  = findCol({"runs", "num_runs", "n_runs", "n", "samples", "repeats", "replications"});
    if (paramIdx < 0 || meanIdx < 0) continue;

    QMap<double, Agg> byValue;
    while (!ts.atEnd()) {
      const QString line = ts.readLine().trimmed();
      if (line.isEmpty()) continue;
      QStringList parts = line.split(',', Qt::KeepEmptyParts);
      for (auto& s : parts) s = s.trimmed();
      if (parts.size() < headers.size()) continue;

      bool okX = false, okM = false;
      const double x = parts[paramIdx].toDouble(&okX);
      const double m = parts[meanIdx].toDouble(&okM);
      if (!okX || !okM) continue;

      double s = 0.0;
      if (sdIdx >= 0) { bool okS = false; const double sv = parts[sdIdx].toDouble(&okS); if (okS) s = sv; }

      double vmin = std::numeric_limits<double>::quiet_NaN();
      double vmaxOfMin = std::numeric_limits<double>::quiet_NaN();
      if (minIdx >= 0 && minIdx < parts.size()) {
        bool okMn = false; const double mv = parts[minIdx].toDouble(&okMn);
        if (okMn) { vmin = mv; vmaxOfMin = mv; }
      }
      double vmax = std::numeric_limits<double>::quiet_NaN();
      if (maxIdx >= 0 && maxIdx < parts.size()) {
        bool okMx = false; const double mv = parts[maxIdx].toDouble(&okMx);
        if (okMx) vmax = mv;
      }
      if (std::isnan(vmax) && !std::isnan(vmaxOfMin)) vmax = vmaxOfMin;

      long long nn = 1;
      if (runsIdx >= 0) {
        bool okN = false; const long long nv = parts[runsIdx].toLongLong(&okN);
        if (okN && nv > 0) nn = nv;
      }
      byValue[x].addBatch(nn, m, s, vmin, vmax);
    }

    for (auto it = byValue.begin(); it != byValue.end(); ++it) {
      Row r;
      r.param = paramName;
      r.x     = it.key();
      r.mean  = it.value().mean;
      r.sd    = it.value().sd();
      r.minV  = it.value().minV;
      r.maxV  = it.value().maxV;
      r.n     = int(std::min<long long>(it.value().n, std::numeric_limits<int>::max()));
      allRows.push_back(r);
    }
  }

  if (allRows.isEmpty()) return;

  const bool hasSd  = std::any_of(allRows.begin(), allRows.end(), [](const auto& r){ return r.sd > 0.0; });
  const bool hasMin = std::any_of(allRows.begin(), allRows.end(), [](const auto& r){ return !std::isnan(r.minV); });
  const bool hasMax = std::any_of(allRows.begin(), allRows.end(), [](const auto& r){ return !std::isnan(r.maxV); });

  struct ColDef { QString header; };
  QVector<ColDef> cols;
  cols.push_back({"Parameter"});
  cols.push_back({"Value"});
  cols.push_back({"Mean best_f"});
  if (hasSd)  cols.push_back({"SD"});
  if (hasMin) cols.push_back({"Min best_f"});
  if (hasMax) cols.push_back({"Max best_f"});
  cols.push_back({"Main effect"});
  cols.push_back({"Rank"});
  cols.push_back({"N (runs)"});

  auto colIdx = [&](const QString& name) -> int {
    for (int i = 0; i < cols.size(); ++i) if (cols[i].header == name) return i;
    return -1;
  };
  const int cParam  = colIdx("Parameter");
  const int cValue  = colIdx("Value");
  const int cMean   = colIdx("Mean best_f");
  const int cSd     = colIdx("SD");
  const int cMin    = colIdx("Min best_f");
  const int cMax    = colIdx("Max best_f");
  const int cEffect = colIdx("Main effect");
  const int cRank   = colIdx("Rank");
  const int cN      = colIdx("N (runs)");

  // Rank + main-effect are computed per-parameter-group, not mixed across
  // different parameters' value scales.
  QVector<int> rank(allRows.size(), 0);
  QVector<double> effect(allRows.size(), 0.0);
  for (const QString& paramName : paramCols) {
    QVector<int> idxInGroup;
    for (int i = 0; i < allRows.size(); ++i) if (allRows[i].param == paramName) idxInGroup.push_back(i);
    if (idxInGroup.isEmpty()) continue;

    double gMin =  std::numeric_limits<double>::max();
    double gMax = -std::numeric_limits<double>::max();
    for (int i : idxInGroup) { gMin = std::min(gMin, allRows[i].mean); gMax = std::max(gMax, allRows[i].mean); }
    const double eff = (idxInGroup.size() > 1) ? (gMax - gMin) : 0.0;

    QVector<int> order = idxInGroup;
    std::sort(order.begin(), order.end(), [&](int a, int b){ return allRows[a].mean < allRows[b].mean; });
    for (int r = 0; r < order.size(); ++r) { rank[order[r]] = r + 1; effect[order[r]] = eff; }
  }

  auto fmtVal = [](double v) -> QString { return QString::number(v, 'g', 8); };

  tab.sensitivitySummaryTable->setColumnCount(cols.size());
  QStringList colHeaders;
  for (const auto& c : cols) colHeaders << c.header;
  tab.sensitivitySummaryTable->setHorizontalHeaderLabels(colHeaders);
  tab.sensitivitySummaryTable->setRowCount(allRows.size());

  for (int i = 0; i < allRows.size(); ++i) {
    const auto& r = allRows[i];
    const bool isBest = (rank[i] == 1);

    auto makeItem = [&](const QString& text, bool bold, bool leftAlign = false) {
      auto* item = new QTableWidgetItem(text);
      item->setTextAlignment((leftAlign ? Qt::AlignLeft : Qt::AlignRight) | Qt::AlignVCenter);
      if (bold) { QFont f = item->font(); f.setBold(true); item->setFont(f); }
      return item;
    };
    auto setCol = [&](int c, const QString& text, bool leftAlign = false) {
      if (c >= 0) tab.sensitivitySummaryTable->setItem(i, c, makeItem(text, isBest, leftAlign));
    };

    setCol(cParam,  r.param, true);
    setCol(cValue,  fmtVal(r.x));
    setCol(cMean,   fmtVal(r.mean));
    if (cSd  >= 0) setCol(cSd,  fmtVal(r.sd));
    if (cMin >= 0) setCol(cMin, fmtVal(r.minV));
    if (cMax >= 0) setCol(cMax, fmtVal(r.maxV));
    setCol(cEffect, fmtVal(effect[i]));
    setCol(cRank,   QString::number(rank[i]));
    setCol(cN,      QString::number(r.n));
  }

  tab.sensitivitySummaryTable->resizeColumnsToContents();
}

void MainWindow::renderSensitivityForParam(int tabIndex, const QString& paramName) {
  if (tabIndex < 0 || tabIndex >= static_cast<int>(outputRuns_.size())) return;
  OutputRunTab& tab = outputRuns_[tabIndex];

  if (!tab.sensitivityLog || !tab.sensitivityParamCombo || !tab.sensitivityPlot || !tab.sensitivitySummaryTable) return;

  auto* plot = dynamic_cast<SensitivityBarWidget*>(tab.sensitivityPlot);
  if (plot) plot->clear();

  const QString csvPath = tab.sensitivityCsvPath.trimmed();
  if (csvPath.isEmpty() || !QFileInfo::exists(csvPath)) {
    tab.sensitivityLog->setPlainText("Sensitivity: no results file available.");
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
  QStringList headers = headerLine.split(',', Qt::KeepEmptyParts);
  for (auto& h : headers) h = h.trimmed();

  auto findCol = [&](const QStringList& candidates) -> int {
    for (const auto& c : candidates) {
      for (int i = 0; i < headers.size(); ++i) {
        if (headers[i].trimmed().toLower() == c.trimmed().toLower()) return i;
      }
    }
    return -1;
  };

    const int paramIdx  = findCol({paramName});
    const int meanIdx   = findCol({"mean_best_f", "mean_f", "meanbestf", "mean"});
    const int sdIdx     = findCol({"stdev_best_f", "stdev_f", "sd_best_f", "sd_f",
                                   "std_best_f", "std_f", "stdev", "sd", "std"});
    const int minIdx    = findCol({"min_best_f", "min_f", "min_best", "min"});
    const int maxIdx    = findCol({"max_best_f", "max_f", "max_best", "max"});
    const int runsIdx   = findCol({"runs", "num_runs", "n_runs", "n", "samples", "repeats", "replications"});
    if (paramIdx < 0 || meanIdx < 0) {
      tab.sensitivityLog->setPlainText(QString("Sensitivity: expec...s not found in %1\nParameter: %2\nMetric: mean_f (Mean best_f)")
                                          .arg(csvPath, paramName));
      return;
    }

    // Aggregate per-parameter value across the sensitivity grid.
    // If the CSV provides per-row standard deviations (stdev_*), combine them using a pooled-variance update.
    struct Agg {
      long long n = 0;
      double mean = 0.0;
      double m2   = 0.0;
      double minV =  std::numeric_limits<double>::quiet_NaN();
      double maxV = -std::numeric_limits<double>::quiet_NaN();

      void addBatch(long long nn, double m, double sd, double vmin, double vmax) {
        if (nn <= 0) nn = 1;
        const double var = (nn > 1 && sd > 0.0) ? (sd * sd) : 0.0;
        const double m2b = (nn > 1) ? (var * double(nn - 1)) : 0.0;

        if (n == 0) {
          n    = nn;
          mean = m;
          m2   = m2b;
          minV = vmin;
          maxV = vmax;
          return;
        }

        const double delta = m - mean;
        const long long newN = n + nn;
        mean += delta * (double(nn) / double(newN));
        m2   += m2b + delta * delta * (double(n) * double(nn) / double(newN));
        n    = newN;
        if (!std::isnan(vmin)) minV = std::isnan(minV) ? vmin : std::min(minV, vmin);
        if (!std::isnan(vmax)) maxV = std::isnan(maxV) ? vmax : std::max(maxV, vmax);
      }

      double sd() const {
        if (n <= 1) return 0.0;
        return std::sqrt(m2 / double(n - 1));
      }
    };

    QMap<double, Agg> byValue;

  int rows = 0;
  int used = 0;
  while (!ts.atEnd()) {
    const QString line = ts.readLine().trimmed();
    if (line.isEmpty()) continue;
    QStringList parts = line.split(',', Qt::KeepEmptyParts);
    for (auto& s : parts) s = s.trimmed();
    if (parts.size() < headers.size()) continue;

    bool okX = false, okM = false;
    const double x = parts[paramIdx].toDouble(&okX);
    const double m = parts[meanIdx].toDouble(&okM);

    double s = 0.0;
    if (sdIdx >= 0) {
      bool okS = false;
      const double sv = parts[sdIdx].toDouble(&okS);
      if (okS) s = sv;
    }

    double vmin = std::numeric_limits<double>::quiet_NaN();
    double vmaxOfMin = std::numeric_limits<double>::quiet_NaN(); // max of min_f = worst best result
    if (minIdx >= 0 && minIdx < parts.size()) {
      bool okMn = false;
      const double mv = parts[minIdx].toDouble(&okMn);
      if (okMn) { vmin = mv; vmaxOfMin = mv; }
    }

    double vmax = std::numeric_limits<double>::quiet_NaN();
    if (maxIdx >= 0 && maxIdx < parts.size()) {
      bool okMx = false;
      const double mv = parts[maxIdx].toDouble(&okMx);
      if (okMx) vmax = mv;
    }
    // If no dedicated max_f column, derive Max best_f as max-of-min_f (worst best result).
    if (std::isnan(vmax) && !std::isnan(vmaxOfMin)) vmax = vmaxOfMin;

    long long nn = 1;
    if (runsIdx >= 0) {
      bool okN = false;
      const long long nv = parts[runsIdx].toLongLong(&okN);
      if (okN && nv > 0) nn = nv;
    }
    ++rows;
    if (!okX || !okM) continue;

    byValue[x].addBatch(nn, m, s, vmin, vmax);
    ++used;
  }

  QVector<SensitivityBarWidget::Point> pts;
  pts.reserve(byValue.size());
  for (auto it = byValue.begin(); it != byValue.end(); ++it) {
    SensitivityBarWidget::Point p;
    p.x    = it.key();
    p.mean = it.value().mean;
    p.sd   = it.value().sd();
    p.n    = int(std::min<long long>(it.value().n, std::numeric_limits<int>::max()));
    p.minV = it.value().minV;
    p.maxV = it.value().maxV;
    pts.push_back(p);
  }

  if (plot) plot->setData(paramName, pts);

  // NOTE: the Results tab's table used to be rebuilt here too (for just
  // this one parameter), which meant switching this combo box silently
  // overwrote the table with only the currently-selected parameter's rows
  // -- since the Results tab has no parameter selector of its own, that
  // made the other swept parameters' results appear to vanish. The table
  // is now populated once, comprehensively (all swept parameters at once,
  // via the "Parameter" column), by populateSensitivitySummaryAllParams()
  // -- called from tryLoadSensitivityForTab() -- and is left untouched
  // here so it stays stable regardless of which parameter the chart shows.

  // Pre-compute global stats (still needed for the log text below).
  double globalMinMean =  std::numeric_limits<double>::max();
  double globalMaxMean = -std::numeric_limits<double>::max();
  for (const auto& p : pts) {
    globalMinMean = std::min(globalMinMean, p.mean);
    globalMaxMean = std::max(globalMaxMean, p.mean);
  }
  const double mainEffectRange = (pts.size() > 1) ? (globalMaxMean - globalMinMean) : 0.0;

  tab.sensLoaded = !pts.isEmpty();

  QString log;
  log.reserve(2048);
  log += QString("File: %1\n").arg(csvPath);
  log += QString("Parameter: %1  (%2 distinct values)\n").arg(paramName).arg(pts.size());
  log += "Metric: Mean best_f (CSV column: mean_f)\n";
  log += QString("Rows parsed: %1, rows used: %2\n").arg(rows).arg(used);
  if (pts.size() > 1) {
    int bestIdx = 0;
    int worstIdx = 0;
    for (int i = 1; i < pts.size(); ++i) {
      if (pts[i].mean < pts[bestIdx].mean)  bestIdx  = i;
      if (pts[i].mean > pts[worstIdx].mean) worstIdx = i;
    }
    log += QString("Best  (rank 1): value=%1  mean=%2\n")
             .arg(QString::number(pts[bestIdx].x,    'g', 8),
                  QString::number(pts[bestIdx].mean, 'g', 8));
    log += QString("Worst (rank %3): value=%1  mean=%2\n")
             .arg(QString::number(pts[worstIdx].x,    'g', 8),
                  QString::number(pts[worstIdx].mean, 'g', 8))
             .arg(pts.size());
    log += QString("Main effect range: %1  (max_mean - min_mean)\n")
             .arg(QString::number(mainEffectRange, 'g', 8));
  }
  log += "\nNote: When multiple parameters are swept (grid mode), "
         "the plotted mean for each value is averaged over the other swept parameters.\n";
  tab.sensitivityLog->setPlainText(log);
}

void MainWindow::onOutputSensitivityParamChanged(int) {
  auto* combo = qobject_cast<QComboBox*>(sender());
  if (!combo) return;

  for (int i = 0; i < static_cast<int>(outputRuns_.size()); ++i) {
    if (outputRuns_[i].sensitivityParamCombo == combo) {
      renderSensitivityForParam(i, combo->currentText().trimmed());
      return;
    }
  }
}

void MainWindow::onOutputTabChanged(int index) {
  activeOutputRunIndex_ = index;

  if (auto* tab = outputTab(index)) {
    if (tab->methodShort.compare("System", Qt::CaseInsensitive) == 0 && tab->page) {
      const QWidget* page = tab->page;

      if (!(batchActive_ && g_activeBatchSummaryPage && page != g_activeBatchSummaryPage)) {
        bindBatchSummaryUiForPage(tab->page);

        if (!batchActive_ && g_batchSummarySnapshots.contains(tab->page)) {
          QScopedValueRollback<bool> restoreGuard(internalUpdate_, true);
          const BatchSummarySnapshot snap = g_batchSummarySnapshots.value(tab->page);

          if (batchMetricCombo_ && snap.metricComboIndex >= 0 && batchMetricCombo_->currentIndex() != snap.metricComboIndex) {
            QSignalBlocker blocker(batchMetricCombo_);
            batchMetricCombo_->setCurrentIndex(snap.metricComboIndex);
            // sync secondary
            if (auto* mc2 = g_batchSummaryUi.value(tab->page).metricCombo2) {
              QSignalBlocker b2(mc2); mc2->setCurrentIndex(snap.metricComboIndex);
            }
          }
          if (batchAggCombo_) {
            const int aggIdx = batchAggCombo_->findText(snap.aggText);
            if (aggIdx >= 0 && batchAggCombo_->currentIndex() != aggIdx) {
              QSignalBlocker blocker(batchAggCombo_);
              batchAggCombo_->setCurrentIndex(aggIdx);
              // sync secondary
              if (auto* ac2 = g_batchSummaryUi.value(tab->page).aggCombo2) {
                QSignalBlocker b2(ac2); ac2->setCurrentIndex(aggIdx);
              }
            }
          }
          if (batchShowRateChk_ && batchShowRateChk_->isChecked() != snap.showRate) {
            QSignalBlocker blocker(batchShowRateChk_);
            batchShowRateChk_->setChecked(snap.showRate);
          }
          if (batchShowSdChk_ && batchShowSdChk_->isChecked() != snap.showSd) {
            QSignalBlocker blocker(batchShowSdChk_);
            batchShowSdChk_->setChecked(snap.showSd);
          }
          if (batchShowTimeChk_ && batchShowTimeChk_->isChecked() != snap.showTime) {
            QSignalBlocker blocker(batchShowTimeChk_);
            batchShowTimeChk_->setChecked(snap.showTime);
          }
          if (batchPanel_) {
            if (auto* showMeanChk = batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")) {
              if (showMeanChk->isChecked() != snap.showMean) {
                QSignalBlocker blocker(showMeanChk);
                showMeanChk->setChecked(snap.showMean);
              }
            }
          }

          g_activeBatchSummaryPage = tab->page;
          batchCells_ = batchSummaryCellsByPage_.value(tab->page);
          batchCachedProblemDims_ = snap.cachedProblemDims;
          batchProblemDimOverride_ = snap.problemDimOverrides;
          g_liveBatchCsvPaths = snap.csvPaths;

          auto restoreBatchListSelection = [this](QListWidget* list, const QStringList& wanted) {
            if (!list) return;
            QSet<QString> wantedExact;
            QSet<QString> wantedBase;
            for (const QString& w : wanted) {
              const QString exact = w.trimmed().toLower();
              if (!exact.isEmpty()) wantedExact.insert(exact);
              const QString base = batchBaseProblemShort(w).trimmed().toLower();
              if (!base.isEmpty()) wantedBase.insert(base);
            }

            QListWidgetItem* firstSelected = nullptr;
            QSignalBlocker blocker(list);
            list->clearSelection();
            for (int i = 0; i < list->count(); ++i) {
              if (auto* it = list->item(i)) {
                QString key = it->data(Qt::UserRole).toString().trimmed();
                if (key.isEmpty()) key = it->text().trimmed();
                const QString exact = key.toLower();
                const QString base = batchBaseProblemShort(key).trimmed().toLower();
                const bool sel = wantedExact.contains(exact) || (!base.isEmpty() && wantedBase.contains(base));
                it->setSelected(sel);
                if (sel && !firstSelected) firstSelected = it;
              }
            }
            if (firstSelected) list->setCurrentItem(firstSelected);
          };

          if (!snap.methods.isEmpty()) restoreBatchListSelection(batchMethodsList_, snap.methods);
          if (!snap.problems.isEmpty()) restoreBatchListSelection(batchProblemsList_, snap.problems);

          syncBatchProblemDimsTable();
          updateDimUiForProblem(currentProblemShort());

          if (settingsBox_) {
            if (auto* combo = settingsBox_->findChild<QComboBox*>("batchMethodSettingsCombo")) {
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

              if (combo->count() > 0 && methodBox_) {
                const QString chosenShort = combo->currentData().toString().trimmed();
                int methodIndex = methodBox_->findData(chosenShort);
                if (methodIndex < 0) methodIndex = methodBox_->findText(chosenShort);
                if (methodIndex >= 0) {
                  if (methodBox_->currentIndex() != methodIndex) methodBox_->setCurrentIndex(methodIndex);
                  else onMethodChanged(methodBox_->currentText());
                }
              }
            }
          }

          if (batchCells_.isEmpty() && !snap.csvPaths.isEmpty()) {
            QMap<QString, QSet<int>> restoredDimsByBase;
            for (auto pit = snap.csvPaths.constBegin(); pit != snap.csvPaths.constEnd(); ++pit) {
              for (auto mit = pit.value().constBegin(); mit != pit.value().constEnd(); ++mit) {
                const QString csvPath = mit.value().trimmed();
                if (csvPath.isEmpty() || !QFileInfo::exists(csvPath)) continue;
                BatchCellData cell;
                QString err;
                if (loadBatchCellFromConvergence(csvPath, cell, &err)) {
                  batchCells_[pit.key()][mit.key()] = cell;
                  const int cachedDim = parseDimFromConvergenceCsvFilename(csvPath);
                  if (cachedDim > 0) {
                    batchCachedProblemDims_[pit.key()] = cachedDim;
                    restoredDimsByBase[batchBaseProblemShort(pit.key())].insert(cachedDim);
                  }
                }
              }
            }
            if (batchProblemDimOverride_.isEmpty()) {
              for (auto it = restoredDimsByBase.constBegin(); it != restoredDimsByBase.constEnd(); ++it) {
                if (fixedDimForProblem(it.key()) > 0) continue;
                QList<int> dims = it.value().values();
                std::sort(dims.begin(), dims.end());
                QStringList parts;
                for (int d : dims) {
                  if (d > 0) parts << QString::number(d);
                }
                if (!parts.isEmpty()) batchProblemDimOverride_[it.key()] = parts;
              }
            }
          }

          {
            BatchSummarySnapshot snapUpdated = g_batchSummarySnapshots.value(tab->page);
            snapUpdated.csvPaths = g_liveBatchCsvPaths;
            snapUpdated.cachedProblemDims = batchCachedProblemDims_;
            snapUpdated.problemDimOverrides = batchProblemDimOverride_;
            g_batchSummarySnapshots.insert(tab->page, snapUpdated);
            batchSummaryCellsByPage_[tab->page] = batchCells_;
          }

          // FIX 2: The stats-table rebuild must run with internalUpdate_==false
          // so that refreshBatchSelectionView() is allowed to persist the final
          // snapshot.  We therefore close the QScopedValueRollback scope here
          // and do the rebuild afterwards, reading back from g_batchSummarySnapshots.
        } // ← QScopedValueRollback destroyed here; internalUpdate_ is false again

        // FIX 2 (corrected): Rebuild the stats table only when NOT in an active
        // batch run.  The previous version of this block ran unconditionally inside
        // the outer "if (!(batchActive_ && ...))" check but outside the inner
        // "if (!batchActive_)" guard.  During a live batch that caused
        // finalizeStatsTableAfterBatch() to set statsBatchTableActive_=false,
        // making every subsequent updateStatsTableCell() return early and leaving
        // the Results Table completely blank for the whole run.
        if (!batchActive_) {
          if (g_batchSummarySnapshots.contains(tab->page)) {
            const BatchSummarySnapshot& snapRef = g_batchSummarySnapshots.value(tab->page);
            const QStringList restoredMethods  = snapRef.methods;
            const QStringList restoredProblems = snapRef.problems;
            if (!restoredMethods.isEmpty() && !restoredProblems.isEmpty()) {
              initStatsTableForBatch(restoredMethods, restoredProblems);
              for (const QString& prob : restoredProblems) {
                for (const QString& meth : restoredMethods) {
                  updateStatsTableCell(prob, meth);
                }
              }
              finalizeStatsTableAfterBatch();
              rebuildStatsComparisons();
            } else {
              rebuildStatsComparisons();
            }
          }
        }
      }
    }
  }

  // FIX 5: System (batch summary) tabs have no convergencePlot / sensitivityPlot
  // widgets.  Calling these functions on them is a no-op guarded by a null-check
  // inside each function, but it still scans the filesystem unnecessarily and can
  // write a misleading "No previous run information available" message to a label
  // that doesn't exist on System tabs.  Skip both calls for System tabs.
  if (const auto* t = outputTab(index)) {
    if (t->methodShort.compare("System", Qt::CaseInsensitive) != 0) {
      tryLoadConvergenceForTab(index);
      tryLoadLandscapeForTab(index);
      tryLoadSensitivityForTab(index);
    }
  }
}

const MainWindow::OutputRunTab* MainWindow::outputTab(int index) const {
  if (index < 0 || index >= int(outputRuns_.size())) return nullptr;
  return &outputRuns_[index];
}

MainWindow::OutputRunTab* MainWindow::activeOutputTab() {
  if (auto* t = outputTab(activeOutputRunIndex_)) return t;
  if (outputTabs_) return outputTab(outputTabs_->currentIndex());
  return nullptr;
}

const MainWindow::OutputRunTab* MainWindow::activeOutputTab() const {
  if (const auto* t = outputTab(activeOutputRunIndex_)) return t;
  if (outputTabs_) return outputTab(outputTabs_->currentIndex());
  return nullptr;
}

// Splits one CSV line, honouring double-quoted fields (Mode 8 quotes any
// field that can contain a comma or semicolon -- variant labels, method
// histograms, the Wilcoxon "better" verdict). A naive split on ',' would
// tear those apart and shift every subsequent column, which is exactly the
// kind of silent misalignment that makes a results table look plausible
// and be wrong.
static QStringList splitCsvLineQuoted(const QString& line) {
  QStringList out;
  QString cur;
  bool inQuotes = false;
  for (int i = 0; i < line.size(); ++i) {
    const QChar c = line[i];
    if (inQuotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }  // escaped ""
        else inQuotes = false;
      } else {
        cur += c;
      }
    } else {
      if (c == '"')      inQuotes = true;
      else if (c == ',') { out << cur; cur.clear(); }
      else               cur += c;
    }
  }
  out << cur;
  return out;
}

// Loads a CSV file into a QTableWidget: first row becomes the horizontal
// header, everything else becomes rows. Returns the number of data rows
// loaded, or -1 if the file could not be read.
static int loadCsvIntoTableWidget(QTableWidget* table, const QString& csvPath) {
  if (!table) return -1;
  table->setRowCount(0);
  table->setColumnCount(0);

  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return -1;
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);

  const QString headerLine = ts.readLine();
  if (headerLine.isNull()) return -1;
  const QStringList headers = splitCsvLineQuoted(headerLine);
  table->setColumnCount(headers.size());
  for (int c = 0; c < headers.size(); ++c) {
    table->setHorizontalHeaderItem(c, new QTableWidgetItem(headers[c].trimmed()));
  }

  int n = 0;
  while (!ts.atEnd()) {
    const QString line = ts.readLine();
    if (line.trimmed().isEmpty()) continue;
    const QStringList cells = splitCsvLineQuoted(line);
    const int row = table->rowCount();
    table->insertRow(row);
    for (int c = 0; c < cells.size() && c < table->columnCount(); ++c) {
      table->setItem(row, c, new QTableWidgetItem(cells[c].trimmed()));
    }
    ++n;
  }
  table->resizeColumnsToContents();
  return n;
}

void MainWindow::tryLoadPortfolioForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || !tab->isPortfolioRun) return;

  tab->portfolioLoaded = false;

  if (tab->runtimeWorkingDir.isEmpty() || tab->portfolioOutPrefix.isEmpty()) {
    if (tab->portfolioInfoLbl) tab->portfolioInfoLbl->setText("Results will appear here once the run finishes.");
    return;
  }

  const QDir wd(tab->runtimeWorkingDir);
  auto pathFor = [&](const char* suffix) {
    return wd.filePath(tab->portfolioOutPrefix + QString::fromLatin1(suffix));
  };

  const int nSummary  = loadCsvIntoTableWidget(tab->portfolioSummaryTable,  pathFor("_summary.csv"));
  const int nSegments = loadCsvIntoTableWidget(tab->portfolioSegmentsTable, pathFor("_segments.csv"));
  loadCsvIntoTableWidget(tab->portfolioUsageTable,    pathFor("_method_usage.csv"));
  const int nAggregate = loadCsvIntoTableWidget(tab->portfolioAggregateTable, pathFor("_aggregate.csv"));
  tab->portfolioMultiProblem = (nAggregate >= 0);

  // Multi-problem runs write BOTH a cross-problem "_overall_*" comparison
  // AND, per problem, the classic run-blocked "__<problem>_*" one (see
  // src/mode_portfolio.cpp). The Ranking/Friedman/Wilcoxon tabs show the
  // cross-problem headline when there is one; the per-problem detail files
  // are still on disk (visible via "Open output folder") for anyone who
  // wants to drill into a specific problem.
  int nRanking = loadCsvIntoTableWidget(tab->portfolioRankingTable, pathFor("_overall_ranking.csv"));
  if (nRanking < 0) nRanking = loadCsvIntoTableWidget(tab->portfolioRankingTable, pathFor("_ranking.csv"));
  if (loadCsvIntoTableWidget(tab->portfolioWilcoxonTable, pathFor("_overall_wilcoxon.csv")) < 0)
    loadCsvIntoTableWidget(tab->portfolioWilcoxonTable, pathFor("_wilcoxon.csv"));
  if (loadCsvIntoTableWidget(tab->portfolioFriedmanTable, pathFor("_overall_friedman.csv")) < 0)
    loadCsvIntoTableWidget(tab->portfolioFriedmanTable, pathFor("_friedman.csv"));

  if (nSummary < 0 && nSegments < 0) {
    if (tab->portfolioInfoLbl) {
      tab->portfolioInfoLbl->setText(
        QString("No results file found yet.\nExpected files starting with:\n  %1\n"
                "(the run may have exited early -- check the Run Log tab for errors)")
          .arg(wd.filePath(tab->portfolioOutPrefix)));
    }
    return;
  }

  if (tab->portfolioInfoLbl) {
    tab->portfolioInfoLbl->setText(
      tab->portfolioMultiProblem
        ? QString("%1 run row(s) across multiple problems, %2 segment row(s) loaded. See the Aggregate tab for "
                   "one row per (problem, variant), and Ranking/Friedman/Wilcoxon for the cross-problem comparison "
                   "(blocked by problem, using each problem's mean of runs).")
            .arg(qMax(nSummary, 0)).arg(qMax(nSegments, 0))
        : QString("%1 run row(s) and %2 segment row(s) loaded. One row per (variant, run): the final best f, "
                   "the evaluations actually spent (probes included) and how many times the relay switched method.")
            .arg(qMax(nSummary, 0)).arg(qMax(nSegments, 0)));
  }

  // Turn the Friedman key/value table into a one-line verdict, so the
  // headline answer is readable without decoding a p-value column.
  if (tab->portfolioFriedmanLbl) {
    if (nRanking < 0) {
      tab->portfolioFriedmanLbl->setText(
        "No omnibus test was run: only one variant was executed. Enable 'with_baselines' in the "
        "[portfolio] section of the settings file to compare the policy against round_robin and "
        "each single-method baseline.");
    } else {
      QString pStr, sigStr;
      QTableWidget* t = tab->portfolioFriedmanTable;
      for (int r = 0; t && r < t->rowCount(); ++r) {
        const QString key = t->item(r, 0) ? t->item(r, 0)->text() : QString();
        const QString val = t->item(r, 1) ? t->item(r, 1)->text() : QString();
        if (key == "p_value") pStr = val;
        if (key == "significant_at_0.05") sigStr = val;
      }
      const bool sig = (sigStr == "1");
      const QString blockNoun = tab->portfolioMultiProblem ? "problem" : "run";
      tab->portfolioFriedmanLbl->setText(
        sig ? QString("Friedman p = %1 (blocked by %2): the variants are NOT all equivalent. The pairwise "
                       "Wilcoxon tab shows which specific pairs differ once corrected for multiple comparisons.")
                  .arg(pStr, blockNoun)
            : QString("Friedman p = %1 (blocked by %2): no statistically significant difference between the "
                       "variants at alpha = 0.05. Any ordering in the Ranking tab should be treated as noise, "
                       "and the pairwise table read with that in mind.").arg(pStr, blockNoun));
    }
  }

  if (tab->portfolioStatsHintLbl && nRanking < 0) {
    tab->portfolioStatsHintLbl->setText(
      "Empty: only one variant was run, so there is nothing to rank. Set 'with_baselines = 1' in the "
      "[portfolio] section of the settings file to additionally run round_robin and one fixed:<method> "
      "baseline per candidate on the same seeds, which is what these three tabs compare.");
  }

  tab->portfolioLoaded = true;
  rebuildPortfolioPlotsForTab(index);
}

// Rebuilds the Friedman rank-bar plot and the Wilcoxon box plot for one
// Portfolio output tab from whatever is currently in its (already loaded)
// Summary and Wilcoxon tables, for whichever metric the tab's Metric combo
// currently selects. Called once after a run finishes and again every time
// that combo changes. Deliberately reads back from the QTableWidgets rather
// than re-parsing CSV: the tables are the loaded, on-screen truth, and
// re-deriving from them (instead of keeping a second, parallel copy of the
// same numbers in memory) means the plot can never show something the
// table itself doesn't.
void MainWindow::rebuildPortfolioPlotsForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || !tab->isPortfolioRun || !tab->portfolioLoaded) return;
  auto* summary = tab->portfolioSummaryTable;
  auto* wilcoxonTab = tab->portfolioWilcoxonTable;
  auto* friedmanPlot = dynamic_cast<RankPlotWidget*>(tab->portfolioFriedmanPlot);
  auto* wilcoxonPlot = dynamic_cast<WilcoxonBoxPlotWidget*>(tab->portfolioWilcoxonPlot);
  if (!summary || summary->rowCount() == 0) {
    if (friedmanPlot) { friedmanPlot->clear(); }
    if (wilcoxonPlot) { wilcoxonPlot->setEmptyMessage("No results yet"); wilcoxonPlot->clear(); }
    return;
  }

  const QString metricKey = tab->portfolioMetricCombo
    ? tab->portfolioMetricCombo->currentData().toString() : QString("final_f");
  const QString metricLabel = tab->portfolioMetricCombo
    ? tab->portfolioMetricCombo->currentText() : QString("final_f");

  auto colIndex = [&](const QString& name) {
    for (int c = 0; c < summary->columnCount(); ++c) {
      if (summary->horizontalHeaderItem(c) && summary->horizontalHeaderItem(c)->text() == name) return c;
    }
    return -1;
  };
  const int cVariant = colIndex("variant");
  const int cRun = colIndex("run");
  const int cProblem = colIndex("problem");
  const int cMetric = colIndex(metricKey);
  if (cVariant < 0 || cRun < 0 || cMetric < 0) return;

  // variantOrder: first-seen order, so the plots and the Ranking table
  // agree on which bar/box is which without re-sorting behind the user.
  QStringList variantOrder;
  auto variantIdx = [&](const QString& v, QVector<QVector<double>>* rawByVariant) {
    int i = variantOrder.indexOf(v);
    if (i < 0) { variantOrder.push_back(v); if (rawByVariant) rawByVariant->push_back({}); i = variantOrder.size() - 1; }
    return i;
  };

  // Distinct problems seen -- decides which blocking this rebuild uses.
  // A raw per-run value only means what it says WITHIN one problem; mixing
  // rows from several problems into the same rank-by-run block would pair
  // run 3 of sphere against run 3 of ackley, which share nothing (not even
  // the same scale). With multiple problems the blocks become problems
  // instead, using each (problem,variant)'s mean as one observation --
  // exactly what src/mode_portfolio.cpp's own "_overall_*" files already
  // do, so the interactive plot and the CLI's own numbers agree.
  QSet<QString> problemsSeen;
  if (cProblem >= 0) {
    for (int r = 0; r < summary->rowCount(); ++r) {
      if (auto* it = summary->item(r, cProblem)) problemsSeen.insert(it->text());
    }
  }
  const bool blockByProblem = (problemsSeen.size() > 1);

  QVector<QVector<double>> rawByVariant; // fed to the Wilcoxon box plot
  std::map<QString, std::vector<double>> valuesByBlock; // block label -> per-variant value (blockByProblem)
  std::map<int, std::vector<double>> valuesByRun;         // run index -> per-variant value (!blockByProblem)

  if (blockByProblem) {
    // First pass: raw run values grouped by (problem, variant).
    std::map<QString, std::map<int, std::vector<double>>> raw; // problem -> variantIdx -> runs
    for (int r = 0; r < summary->rowCount(); ++r) {
      const QString variant = summary->item(r, cVariant) ? summary->item(r, cVariant)->text() : QString();
      const QString problem = summary->item(r, cProblem) ? summary->item(r, cProblem)->text() : QString();
      bool ok = false;
      const double val = summary->item(r, cMetric) ? summary->item(r, cMetric)->text().toDouble(&ok) : 0.0;
      if (variant.isEmpty() || problem.isEmpty() || !ok) continue;
      const int vi = variantIdx(variant, nullptr);
      raw[problem][vi].push_back(val);
    }
    const int k0 = variantOrder.size();
    rawByVariant.resize(k0);
    for (const auto& pkv : raw) {
      auto& row = valuesByBlock[pkv.first];
      row.assign((size_t)k0, std::numeric_limits<double>::quiet_NaN());
      for (const auto& vkv : pkv.second) {
        double sum = 0.0; for (double x : vkv.second) sum += x;
        const double mean = vkv.second.empty() ? std::numeric_limits<double>::quiet_NaN() : sum / vkv.second.size();
        row[(size_t)vkv.first] = mean;
        if (vkv.first < rawByVariant.size()) rawByVariant[vkv.first].push_back(mean);
      }
    }
  } else {
    for (int r = 0; r < summary->rowCount(); ++r) {
      const QString variant = summary->item(r, cVariant) ? summary->item(r, cVariant)->text() : QString();
      const int run = summary->item(r, cRun) ? summary->item(r, cRun)->text().toInt() : -1;
      bool ok = false;
      const double val = summary->item(r, cMetric) ? summary->item(r, cMetric)->text().toDouble(&ok) : 0.0;
      if (variant.isEmpty() || run < 0 || !ok) continue;
      const int vi = variantIdx(variant, &rawByVariant);
      rawByVariant[vi].push_back(val);
      auto& row = valuesByRun[run];
      if ((int)row.size() <= vi) row.resize((size_t)variantOrder.size(), std::numeric_limits<double>::quiet_NaN());
      row[(size_t)vi] = val;
    }
    for (auto& kv : valuesByRun) {
      if ((int)kv.second.size() < variantOrder.size())
        kv.second.resize((size_t)variantOrder.size(), std::numeric_limits<double>::quiet_NaN());
    }
  }

  const int k = variantOrder.size();
  const int numBlocks = blockByProblem ? (int)valuesByBlock.size() : (int)valuesByRun.size();
  const QString blockNoun = blockByProblem ? "problem" : "run";

  // -- Wilcoxon box plot: raw per-variant values + significance annotations --
  if (wilcoxonPlot) {
    if (k < 1) {
      wilcoxonPlot->setEmptyMessage("No results yet");
      wilcoxonPlot->clear();
    } else {
      QVector<WilcoxonBoxPlotWidget::Annotation> ann;
      // Annotations only carry meaning for final_f: that's the only metric
      // the CLI's Wilcoxon test was actually computed on (see
      // src/mode_portfolio.cpp). Showing them for evals_used/num_switches
      // would silently mislabel significance that was never tested for
      // that metric, so they're limited to the metric they came from.
      if (metricKey == "final_f" && wilcoxonTab) {
        for (int r = 0; r < wilcoxonTab->rowCount(); ++r) {
          auto cellText = [&](const char* col) -> QString {
            const int c = [&]{ for (int cc = 0; cc < wilcoxonTab->columnCount(); ++cc)
                                    if (wilcoxonTab->horizontalHeaderItem(cc) &&
                                        wilcoxonTab->horizontalHeaderItem(cc)->text() == QString::fromLatin1(col)) return cc;
                                  return -1; }();
            return (c >= 0 && wilcoxonTab->item(r, c)) ? wilcoxonTab->item(r, c)->text() : QString();
          };
          const QString a = cellText("variant_a"), b = cellText("variant_b");
          const bool sig = cellText("significant_at_0.05") == "1";
          const int ia = variantOrder.indexOf(a), ib = variantOrder.indexOf(b);
          if (ia < 0 || ib < 0) continue;
          WilcoxonBoxPlotWidget::Annotation an;
          an.i = ia; an.j = ib; an.significant = sig;
          an.text = sig ? ("p=" + cellText("p_holm").left(6)) : "n.s.";
          ann.push_back(an);
        }
      }
      QStringList labels = variantOrder;
      wilcoxonPlot->setData(labels, rawByVariant,
                             QString("Portfolio variants -- %1").arg(metricLabel),
                             blockByProblem
                               ? QString("%1 problem(s), mean of runs per problem").arg(numBlocks)
                               : QString("%1 run(s) per variant, paired by seed").arg(numBlocks),
                             ann);
    }
  }

  // -- Friedman rank plot: ranks across variants, blocked by %1 above,     --
  // -- same algorithm as src/mode_portfolio.cpp's own Friedman computation --
  if (friedmanPlot) {
    if (k < 2 || numBlocks < 2) {
      friedmanPlot->clear();
    } else {
      QVector<QVector<double>> ranksByVariant(k);
      if (blockByProblem) {
        for (const auto& kv : valuesByBlock) {
          const std::vector<double> ranks = optimsolution::stats::averageRank(kv.second);
          for (int vi = 0; vi < k; ++vi) ranksByVariant[vi].push_back(ranks[(size_t)vi]);
        }
      } else {
        for (const auto& kv : valuesByRun) {
          const std::vector<double> ranks = optimsolution::stats::averageRank(kv.second);
          for (int vi = 0; vi < k; ++vi) ranksByVariant[vi].push_back(ranks[(size_t)vi]);
        }
      }
      friedmanPlot->setData(variantOrder, ranksByVariant,
                             QString("Portfolio variants -- rank by %1").arg(metricLabel),
                             QString("Average rank across %1 %2(s) (1 = best)").arg(numBlocks).arg(blockNoun));
    }
  }
}

void MainWindow::tryLoadConvergenceForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || tab->isPortfolioRun) return;   // Portfolio tabs have no *_convergence.csv; see tryLoadPortfolioForTab()
  if (!tab || !tab->convergenceInfo) return;

  auto setConvInfo = [&](const QString& msg) {
    const QString m = tab->methodShort.isEmpty() ? "-" : tab->methodShort;
    const QString p = tab->problemShort.isEmpty() ? "-" : tab->problemShort;
    QString header = QString("Method: %1 | Problem: %2").arg(m).arg(p);
    if (tab->problemDim > 0) header += QString(" | D=%1").arg(tab->problemDim);
    tab->convergenceInfo->setText(header + "\n" + msg);
  };

  // Helper: load a single CSV path into the tab and update plots.
  // Returns true on success. Does NOT fall through to the directory scan.
  auto loadFromExplicitPath = [&](const QString& csvPath) -> bool {
    QVector<double> iterX, evalX, y;
    QString info;
    bool hasEvalX = false;
    if (!parseConvergenceCsvFile(csvPath, iterX, evalX, y, info, &hasEvalX)) {
      tab->convLoaded = false;
      tab->convLoadedCsvPath.clear();
      setConvInfo(QString("Could not load: %1\n%2").arg(info, csvPath));
      if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
      updateDistributionPlotForTab(index);
      return false;
    }
    tab->convIterX    = iterX;
    tab->convEvalX    = evalX;
    tab->convY        = y;
    tab->convHasEvalX = hasEvalX;
    tab->convLoaded   = true;
    tab->convLoadedCsvPath = csvPath;
    if (tab->xAxisEvalRadio) {
      tab->xAxisEvalRadio->setEnabled(hasEvalX);
      if (!hasEvalX && tab->xAxisEvalRadio->isChecked() && tab->xAxisIterRadio) {
        QSignalBlocker b(tab->xAxisIterRadio);
        tab->xAxisIterRadio->setChecked(true);
      }
    }
    setConvInfo(QString("%1\n%2").arg(info, csvPath));
    updateConvergencePlotForTab(index);
    updateDistributionPlotForTab(index);
    return true;
  };

  // ── Fast path A: an explicit CSV was pinned to this tab (e.g. via "Load experiment CSV").
  //    Always honour it — never let the directory scan overwrite it with a different file.
  if (tab->page) {
    const QString explicitPath = tab->page->property("explicitConvergenceCsvPath").toString().trimmed();
    if (!explicitPath.isEmpty()) {
      if (QFileInfo::exists(explicitPath)) {
        loadFromExplicitPath(explicitPath);
      } else {
        tab->convLoaded = false;
        tab->convLoadedCsvPath.clear();
        setConvInfo(QString("Pinned CSV no longer exists:\n%1").arg(explicitPath));
        if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
        updateDistributionPlotForTab(index);
      }
      // Regardless of success, do NOT fall through to the directory scan.
      return;
    }
  }

  // ── Fast path B: data already loaded and the run for this tab has finished.
  //    Re-use the cached data instead of scanning the directory again.
  const bool runningThisTab = (proc_ && proc_->state() != QProcess::NotRunning
                               && activeOutputRunIndex_ == index);
  if (tab->convLoaded && !tab->convLoadedCsvPath.isEmpty()
      && QFileInfo::exists(tab->convLoadedCsvPath) && !runningThisTab) {
    // Data is current — just refresh the plots without touching convIterX/Y.
    updateConvergencePlotForTab(index);
    updateDistributionPlotForTab(index);
    return;
  }

  // ── Full scan path: reset and search the working directory for a suitable CSV.
  tab->convLoaded = false;
  tab->convLoadedCsvPath.clear();
  tab->convIterX.clear();
  tab->convEvalX.clear();
  tab->convY.clear();
  tab->convHasEvalX = false;

  if (tab->runtimeWorkingDir.isEmpty()) {
    setConvInfo("No previous run information available.");
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    updateDistributionPlotForTab(index);
    return;
  }

  const QString wd = tab->runtimeWorkingDir;

  // Search for a suitable convergence CSV near the runtime working directory.
  const QStringList searchDirs = {
    wd,
    QDir(wd).filePath("csv"),
    QDir(wd).filePath("output"),
    QDir(wd).filePath("results"),
    QDir(wd).filePath("logs")
  };

  struct Cand { QString path; QDateTime mtime; int score = 0; };
  QVector<Cand> candidates;
  QSet<QString> seen;

  auto scoreName = [](const QString& fn) {
    const QString n = fn.toLower();
    int s = 0;
    if (n.contains("convergence")) s += 5;
    if (n.contains("conv"))        s += 3;
    if (n.contains("trace") || n.contains("progress") || n.contains("history")) s += 1;
    return s;
  };

  for (const QString& d : searchDirs) {
    QDir dir(d);
    if (!dir.exists()) continue;

    const QFileInfoList files = dir.entryInfoList(QStringList() << "*.csv", QDir::Files, QDir::Time);
    for (const QFileInfo& fi : files) {
      const QString ap = fi.absoluteFilePath();
      if (seen.contains(ap)) continue;
      seen.insert(ap);

      Cand c;
      c.path = ap;
      c.mtime = fi.lastModified();
      c.score = scoreName(fi.fileName());
      candidates.push_back(c);
    }
  }

  if (candidates.isEmpty()) {
    setConvInfo(
      "No CSV files were found near the run folder. "
      "Ensure [global] csv_enable=1 and csv_convergence=1 (or keep 'Force CSV convergence' enabled)."
    );
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    updateDistributionPlotForTab(index);
    return;
  }

  // ── Timestamp filter: when a start time is recorded for this tab, keep only
  //    files that were last modified at or after that moment.  This prevents
  //    an older run's CSV (sitting in the same folder) from being picked up for
  //    a newer tab, and vice-versa.
  if (tab->runStartMsecsUtc > 0) {
    const QDateTime startTime = QDateTime::fromMSecsSinceEpoch(tab->runStartMsecsUtc);
    candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
        [&startTime](const Cand& c) { return c.mtime < startTime; }),
      candidates.end());
    // If filtering removed everything, fall back to the full set so the user
    // still sees *something* (the old behaviour) rather than a blank plot.
    if (candidates.isEmpty()) {
      for (const QString& d : searchDirs) {
        QDir dir(d);
        if (!dir.exists()) continue;
        const QFileInfoList files = dir.entryInfoList(QStringList() << "*.csv", QDir::Files, QDir::Time);
        for (const QFileInfo& fi : files) {
          const QString ap = fi.absoluteFilePath();
          if (seen.contains(ap)) continue;
          seen.insert(ap);
          Cand c;
          c.path  = ap;
          c.mtime = fi.lastModified();
          c.score = scoreName(fi.fileName());
          candidates.push_back(c);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.mtime > b.mtime;
  });

  QVector<double> iterX;
  QVector<double> evalX;
  QVector<double> y;
  QString info;
  QString loadedPath;
  bool hasEvalX = false;

  // Try the best-scoring recent candidates first, but keep a hard cap on attempts.
  const int maxAttempts = std::min(40, int(candidates.size()));
  for (int i = 0; i < maxAttempts; ++i) {
    iterX.clear(); evalX.clear(); y.clear(); info.clear();
    hasEvalX = false;
    if (parseConvergenceCsvFile(candidates[i].path, iterX, evalX, y, info, &hasEvalX)) {
      loadedPath = candidates[i].path;
      break;
    }
  }

  if (loadedPath.isEmpty()) {
    setConvInfo(
      "CSV files were found, but none looked like a convergence trace. "
      "Enable convergence CSV output in the CLI, or adjust its filename/pattern."
    );
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    updateDistributionPlotForTab(index);
    return;
  }

  tab->convIterX = iterX;
  tab->convEvalX = evalX;
  tab->convY = y;
  tab->convHasEvalX = hasEvalX;
  tab->convLoaded = true;
  tab->convLoadedCsvPath = loadedPath;

  // Enable/disable x-axis mode depending on what the CSV contains.
  if (tab->xAxisEvalRadio) {
    tab->xAxisEvalRadio->setEnabled(tab->convHasEvalX);
    if (!tab->convHasEvalX && tab->xAxisEvalRadio->isChecked() && tab->xAxisIterRadio) {
      QSignalBlocker b(tab->xAxisIterRadio);
      tab->xAxisIterRadio->setChecked(true);
    }
  }

  setConvInfo(QString("%1\n%2").arg(info).arg(loadedPath));
  updateConvergencePlotForTab(index);
  updateDistributionPlotForTab(index);

  // If other tabs for the same method are in overlay mode, refresh them as well.
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (!outputRuns_[i].overlayMethodChk) continue;
    if (!outputRuns_[i].overlayMethodChk->isChecked()) continue;
    if (outputRuns_[i].methodShort.compare(tab->methodShort, Qt::CaseInsensitive) != 0) continue;
    updateConvergencePlotForTab(i);
  }

  // If other tabs for the same problem are in overlay mode, refresh them as well.
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (!outputRuns_[i].overlayProblemChk) continue;
    if (!outputRuns_[i].overlayProblemChk->isChecked()) continue;
    if (outputRuns_[i].problemShort.compare(tab->problemShort, Qt::CaseInsensitive) != 0) continue;
    if (tab->problemDim > 0 && outputRuns_[i].problemDim > 0 && outputRuns_[i].problemDim != tab->problemDim) continue;
    updateConvergencePlotForTab(i);
  }
}

} // namespace optimsolution_gui
