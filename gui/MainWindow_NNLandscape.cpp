// MainWindow_NNLandscape.cpp
// Mode 7 (run mode data=6): "Neural-network landscape / generalization gap".
// Batch-capable: trains NNProblem (src/problems/nn_problem.h) with every
// selected optimizer, on every selected dataset, for every ';'-separated
// architecture, repeated "Runs" times (seed_base from optimsolution.cfg) --
// exactly the same batch-of-independent-runs convention every other mode
// already uses -- via the "optimsolution --nnlandscape" CLI mode
// (src/mode_nnlandscape.cpp). A second stage optionally runs
// py_scripts/train_gap_regressor.py for the ELA/landscape-feature analysis.
// Progress bar, per-stage timing, and the Wilcoxon/Rankings tabs mirror the
// Meta-dataset pipeline mode (mode 6)'s own conventions as closely as
// possible; see the accompanying integration notes for exactly what's
// reused vs newly written.
#include "MainWindowShared.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace optimsolution_gui {

namespace {

// Minimal CSV line parser respecting double-quoted fields (the CLI quotes
// any field that contains a raw comma, e.g. multi-layer architectures like
// "16,8" inside the run_key/arch columns -- a naive split(',') would
// misalign those rows).
QStringList parseCsvLine(const QString& line) {
  QStringList out;
  QString cur;
  bool inQuotes = false;
  for (int i = 0; i < line.size(); ++i) {
    const QChar c = line[i];
    if (inQuotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
        else inQuotes = false;
      } else cur += c;
    } else {
      if (c == '"') inQuotes = true;
      else if (c == ',') { out << cur; cur.clear(); }
      else cur += c;
    }
  }
  out << cur;
  return out;
}

// Appends the one data row from a per-job "<prefix>_row_summary.csv" (written
// by a pooled --nnlandscape process given a private --out-summary path -- see
// startNNLandscapeJobSlots()) into the shared nn_summary.csv, writing the
// header first if the shared file is new/empty. Called only from
// MainWindow::onNNLandscapeJobSlotFinished(), which -- regardless of how many
// --nnlandscape OS processes are training concurrently -- always runs on the
// GUI's single main thread in response to a QProcess::finished signal. Qt
// delivers those signals one at a time on that one thread, so merges into
// nn_summary.csv here are always fully serialized and never interleave, even
// though the training processes writing their own row files are genuinely
// concurrent. This is the piece that makes parallel_jobs > 1 safe without any
// OS-level file locking: the only shared file is touched by exactly one
// thread, one merge at a time.
bool mergeNNSummaryRow(const QString& masterPath, const QString& rowPath, QString* errorOut) {
  QFile rowFile(rowPath);
  if (!rowFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
    if (errorOut) *errorOut = "row file missing or unreadable: " + rowPath;
    return false;
  }
  QTextStream rin(&rowFile);
  const QString header = rin.readLine();
  const QString dataLine = rin.readLine();
  rowFile.close();
  if (dataLine.isNull() || dataLine.isEmpty()) {
    if (errorOut) *errorOut = "row file has no data line: " + rowPath;
    return false;
  }

  const bool needHeader = !QFileInfo::exists(masterPath) || QFileInfo(masterPath).size() == 0;
  QFile master(masterPath);
  if (!master.open(QIODevice::Append | QIODevice::Text)) {
    if (errorOut) *errorOut = "could not open " + masterPath + " for append";
    return false;
  }
  QTextStream out(&master);
  if (needHeader) out << header << "\n";
  out << dataLine << "\n";
  master.close();

  QFile::remove(rowPath);
  return true;
}

// Whether a lower value of this nn_summary.csv metric column is "better" --
// used both for Rankings (ascending vs descending sort) and, cosmetically,
// nothing else (Wilcoxon itself is direction-agnostic).
bool nnMetricLowerIsBetter(const QString& col) {
  return col != "val_acc" && col != "train_acc";
}

// Proper, capitalized display title + bracketed unit for each metric column,
// used for the Wilcoxon plot's Y-axis title instead of passing the raw
// nn_summary.csv column name straight through. Ratio-valued metrics
// (val_acc, gen_gap_err) are shown as percentages -- see
// nnMetricUnitScale() below, applied to every value read from that column
// so the numbers on the plot actually match the "[%]" unit rather than
// showing e.g. "0.84" under a "%" label.
QString nnMetricAxisLabel(const QString& col) {
  if (col == "gen_gap_loss") return "Generalization gap";
  if (col == "val_loss")     return "Validation loss";
  if (col == "val_acc")      return "Validation accuracy [%]";
  if (col == "train_loss")   return "Training loss";
  if (col == "gen_gap_err")  return "Generalization gap [%]";
  return col;
}

double nnMetricUnitScale(const QString& col) {
  return (col == "val_acc" || col == "gen_gap_err") ? 100.0 : 1.0;
}

// Paired Wilcoxon signed-rank test, normal approximation, two-sided p-value.
// Self-contained on purpose (mirrors, rather than reuses, the framework's
// own batch-benchmark Wilcoxon machinery in MainWindow_BatchStats.cpp,
// which is tightly coupled to that mode's raw-CLI-text parsing pipeline --
// nn_summary.csv already has clean numeric columns, so a small independent
// implementation here is simpler than adapting that one). Returns NaN if
// fewer than 6 non-zero paired differences remain (too small to be
// meaningful either way).
double wilcoxonSignedRankP(const QVector<double>& a, const QVector<double>& b) {
  const int n0 = std::min(a.size(), b.size());
  QVector<double> absd;
  QVector<int> sign;
  for (int i = 0; i < n0; ++i) {
    const double d = a[i] - b[i];
    if (std::abs(d) < 1e-12) continue;
    absd.push_back(std::abs(d));
    sign.push_back(d > 0 ? 1 : -1);
  }
  const int n = absd.size();
  if (n < 6) return std::numeric_limits<double>::quiet_NaN();

  QVector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int x, int y) { return absd[x] < absd[y]; });

  QVector<double> rank(n);
  int i = 0;
  while (i < n) {
    int j = i;
    while (j + 1 < n && absd[idx[j + 1]] == absd[idx[i]]) ++j;
    const double avgRank = (i + j) / 2.0 + 1.0;
    for (int k = i; k <= j; ++k) rank[idx[k]] = avgRank;
    i = j + 1;
  }

  double Wp = 0.0, Wn = 0.0;
  for (int k = 0; k < n; ++k) { if (sign[k] > 0) Wp += rank[k]; else Wn += rank[k]; }
  const double W = std::min(Wp, Wn);
  const double mean = n * (n + 1) / 4.0;
  const double sd = std::sqrt(n * (n + 1) * (2 * n + 1) / 24.0);
  if (sd < 1e-9) return 1.0;
  const double z = (W - mean) / sd;
  return std::erfc(std::abs(z) / std::sqrt(2.0));
}

QString nnRuntimeWorkingDir(const QString& projectRoot, const QString& settingsPath) {
  QString baseWd = projectRoot.isEmpty() ? QCoreApplication::applicationDirPath() : projectRoot;
  if (!settingsPath.isEmpty()) {
    QFileInfo fi(settingsPath);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString wd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(wd);
  return wd;
}

// Same h/m/s formatting as the pipeline mode's own formatPipelineDuration()
// (MainWindow_RunControl.cpp) -- duplicated here rather than shared since
// that one is file-local (static) to its translation unit, and this file
// otherwise avoids touching pipeline-specific code.
QString formatNNDuration(double seconds) {
  if (seconds < 0) return QString();
  const qint64 totalSec = qRound64(seconds);
  const qint64 h = totalSec / 3600;
  const qint64 m = (totalSec % 3600) / 60;
  const qint64 s = totalSec % 60;
  if (h > 0) return QString("%1h %2m %3s").arg(h).arg(m).arg(s);
  if (m > 0) return QString("%1m %2s").arg(m).arg(s);
  return QString("%1s").arg(s);
}

// Reads [global] seed_base directly from the cfg file on disk, mirroring
// globalRunsFromSettings()'s own "freshest value, bypass any cached cfg_"
// approach -- confirmed by testing that relying on cfg_->isLoaded() here
// instead produced the wrong seed (cfg_ was not reliably loaded at the
// point this runs, even though the Settings table on screen already shows
// values read from the same file by a different code path).
// Lightweight task-type detection for the Datasets list -- mirrors
// NNProblem::detectTaskAndBuildArchitecture()'s exact heuristic (single
// integer-valued column with 2-20 distinct values -> classification;
// multi-column exact one-hot rows -> classification; anything else ->
// regression) but reads only the output columns of the .train file itself,
// without constructing a full NNProblem (parsing, normalization, network
// setup) just to answer this one question. File format: "<inputs>\n
// <n_samples>\n" header, then whitespace-separated rows (see
// NNProblem::loadDataset() for the authoritative version this mirrors).
QString nnDetectTaskType(const QString& path) {
  std::ifstream in(path.toStdString());
  if (!in.is_open()) return QString();

  int inputCount = 0;
  long long nSamples = 0;
  if (!(in >> inputCount)) return QString();
  if (!(in >> nSamples)) return QString();
  if (inputCount < 0) return QString();

  std::string line;
  std::getline(in, line); // consume rest of the header line
  int outCols = -1;
  QVector<QVector<double>> Y;
  Y.reserve(4096);
  int rowsRead = 0;
  while (std::getline(in, line) && rowsRead < 20000) {
    std::istringstream ls(line);
    QVector<double> vals;
    double v;
    while (ls >> v) vals.push_back(v);
    if (vals.isEmpty()) continue;
    if (vals.size() < inputCount + 1) continue;
    if (outCols < 0) outCols = vals.size() - inputCount;
    if (vals.size() < inputCount + outCols) continue;
    QVector<double> ys;
    ys.reserve(outCols);
    for (int k = 0; k < outCols; ++k) ys.push_back(vals[inputCount + k]);
    Y.push_back(ys);
    ++rowsRead;
  }
  if (Y.isEmpty() || outCols <= 0) return QString();

  if (outCols == 1) {
    QSet<double> uniq;
    bool allIntegerLike = true;
    for (const auto& row : Y) {
      const double v = row[0];
      uniq.insert(v);
      if (std::abs(v - std::round(v)) > 1e-6) allIntegerLike = false;
    }
    if (allIntegerLike && uniq.size() >= 2 && uniq.size() <= 20) {
      return uniq.size() == 2 ? QString("binary") : QString("%1-class").arg(uniq.size());
    }
    return QString("regression");
  }

  bool looksOneHot = true;
  for (const auto& row : Y) {
    double sum = 0.0;
    for (double v : row) {
      if (std::abs(v) > 1e-9 && std::abs(v - 1.0) > 1e-9) { looksOneHot = false; break; }
      sum += v;
    }
    if (!looksOneHot || std::abs(sum - 1.0) > 1e-9) { looksOneHot = false; break; }
  }
  if (looksOneHot) return QString("%1-class").arg(outCols);
  return QString("regression (%1-dim)").arg(outCols);
}

// Reads [global] seed_base directly from the cfg file on disk, mirroring
// globalRunsFromSettings()'s own "freshest value, bypass any cached cfg_"
// approach -- confirmed by testing that relying on cfg_->isLoaded() here
// instead produced the wrong seed (cfg_ was not reliably loaded at the
// point this runs, even though the Settings table on screen already shows
// values read from the same file by a different code path).
unsigned long long nnSeedBaseFromSettings(const QString& settingsPath) {
  const QString path = settingsPath.isEmpty()
                      ? PathUtils::findExistingFileUpwards(QDir::currentPath(), "optimsolution.cfg")
                      : settingsPath;
  if (!path.isEmpty()) {
    QFile f(path);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      bool inGlobal = false;
      QTextStream ts(&f);
      while (!ts.atEnd()) {
        const QString line = ts.readLine().trimmed();
        if (line.startsWith('[')) { inGlobal = (line.toLower() == "[global]"); continue; }
        if (!inGlobal) continue;
        if (line.startsWith(';') || line.startsWith('#')) continue;
        const int eq = line.indexOf('=');
        if (eq < 0) continue;
        const QString key = line.left(eq).trimmed().toLower();
        if (key == "seed_base") {
          const QString val = line.mid(eq + 1).section(';', 0, 0).trimmed();
          bool ok = false;
          const unsigned long long v = val.toULongLong(&ok);
          if (ok) return v;
        }
      }
    }
  }
  return 1337ULL;
}

} // namespace

// ── Dataset list (Selection panel) ──────────────────────────────────────────
void MainWindow::populateNNDatasetList() {
  if (!nnDatasetsList_) return;

  QStringList prevSelected;
  for (int i = 0; i < nnDatasetsList_->count(); ++i)
    if (nnDatasetsList_->item(i)->isSelected())
      prevSelected << nnDatasetsList_->item(i)->data(Qt::UserRole).toString();

  nnDatasetsList_->clear();

  const QDir dir(datasetsDir());
  if (dir.exists()) {
    const QStringList files = dir.entryList(QStringList{"*.train"}, QDir::Files, QDir::Name);
    for (const QString& f : files) {
      const QString absPath = dir.filePath(f);
      // Task type shown right in the list, so mixing e.g. a regression file
      // with classification files is visible BEFORE selecting anything --
      // this is exactly the ambiguity that produced a near-meaningless (very
      // negative R^2/Spearman) LODO result when a same-named regression and
      // classification version of the same underlying dataset were mixed in
      // one batch. Detection is a lightweight read of the .train file's own
      // output column(s) (see nnDetectTaskType()), not a full NNProblem
      // construction, so this stays cheap even for the full dataset folder.
      const QString taskType = nnDetectTaskType(absPath);
      const QString display = taskType.isEmpty() ? f : QString("%1  (%2)").arg(f, taskType);
      auto* it = new QListWidgetItem(display, nnDatasetsList_);
      it->setData(Qt::UserRole, absPath);      // absolute path -- authoritative identity
      it->setData(Qt::UserRole + 1, taskType); // cached task label, for the mixed-type warning below
      if (!taskType.isEmpty()) it->setToolTip(QString("Detected task: %1").arg(taskType));
      if (prevSelected.contains(absPath)) it->setSelected(true);
    }
  }
  if (nnDatasetsList_->count() == 0) {
    auto* it = new QListWidgetItem(QString("(no *.train files found in %1 -- use Browse...)").arg(datasetsDir()),
                                    nnDatasetsList_);
    it->setFlags(it->flags() & ~Qt::ItemIsSelectable);
  }
}

void MainWindow::populateNNOptimizerList() {
  if (!nnOptimizersList_) return;

  QStringList prevSelected;
  for (int i = 0; i < nnOptimizersList_->count(); ++i)
    if (nnOptimizersList_->item(i)->isSelected())
      prevSelected << nnOptimizersList_->item(i)->text();

  nnOptimizersList_->clear();

  QStringList names;
  try {
    for (const auto& s : optimsolution::listMethodNames()) names << QString::fromStdString(s);
  } catch (...) {}
  names.removeDuplicates();
  names.sort(Qt::CaseInsensitive);

  for (const QString& name : names) {
    auto* it = new QListWidgetItem(name, nnOptimizersList_);
    if (prevSelected.contains(name)) it->setSelected(true);
  }
}

void MainWindow::onNNLandscapeBrowseDataset() {
  const QString start = QFileInfo(datasetsDir()).exists() ? datasetsDir() : QDir::homePath();
  const QStringList paths = QFileDialog::getOpenFileNames(this, "Select dataset(s) (.train)", start,
                                                            "Training set (*.train)");
  if (paths.isEmpty() || !nnDatasetsList_) return;

  for (const QString& path : paths) {
    bool found = false;
    for (int i = 0; i < nnDatasetsList_->count(); ++i) {
      if (nnDatasetsList_->item(i)->data(Qt::UserRole).toString() == path) {
        nnDatasetsList_->item(i)->setSelected(true);
        found = true;
        break;
      }
    }
    if (found) continue;
    const QString taskType = nnDetectTaskType(path);
    const QString display = taskType.isEmpty() ? QFileInfo(path).fileName()
                                                : QString("%1  (%2)").arg(QFileInfo(path).fileName(), taskType);
    auto* it = new QListWidgetItem(display, nnDatasetsList_);
    it->setData(Qt::UserRole, path);
    it->setData(Qt::UserRole + 1, taskType);
    if (!taskType.isEmpty()) it->setToolTip(QString("Detected task: %1").arg(taskType));
    it->setSelected(true);
  }
}

// "Save batch selection..." for mode 7 -- same .batchsel plain-text
// convention as the Batch/Pipeline modes' own Save/Load batch selection
// (key=value lines, "#" comments), but for this mode's own selection shape
// (Datasets + Optimizers lists, Hidden layers, Activation) rather than
// batchMethodsList_/batchProblemsList_, hence a separate, dedicated
// implementation instead of reusing that one.
void MainWindow::onSaveNNBatchSelection() {
  if (!nnDatasetsList_ || !nnOptimizersList_) return;

  QStringList datasets, optimizers;
  for (auto* it : nnDatasetsList_->selectedItems()) {
    const QString path = it->data(Qt::UserRole).toString();
    if (!path.isEmpty()) datasets << QFileInfo(path).fileName();
  }
  for (auto* it : nnOptimizersList_->selectedItems()) optimizers << it->text();

  if (datasets.isEmpty() && optimizers.isEmpty()) {
    QMessageBox::warning(this, "Save batch selection", "There are no selected datasets or optimizers to save.");
    return;
  }

  QString suggestedDir = !projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath();
  if (!lastRuntimeWorkingDir_.trimmed().isEmpty()) suggestedDir = lastRuntimeWorkingDir_.trimmed();
  const QString suggestedPath = QDir(suggestedDir).filePath("nn_batch_selection.batchsel");

  const QString path = QFileDialog::getSaveFileName(this, "Save batch selection", suggestedPath,
                                                      "OptimSolution batch selection (*.batchsel);;All files (*.*)");
  if (path.isEmpty()) return;

  QSaveFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, "Save batch selection", QString("Cannot write file:\n%1").arg(path));
    return;
  }
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);
  ts << "# OptimSolution NN landscape (mode 7) batch selection\n";
  ts << "version=1\n";
  ts << "datasets=" << datasets.join(",") << "\n";
  ts << "optimizers=" << optimizers.join(",") << "\n";
  ts << "architectures=" << (nnHiddenEdit_ ? nnHiddenEdit_->text().trimmed() : QString()) << "\n";
  ts << "activation=" << (nnActivationBox_ ? nnActivationBox_->currentText() : QString()) << "\n";

  if (!f.commit()) {
    QMessageBox::warning(this, "Save batch selection", QString("Failed to finalize file:\n%1").arg(path));
    return;
  }
  statusBar()->showMessage(QString("NN batch selection saved to %1").arg(QFileInfo(path).fileName()), 5000);
  appendLog(QString("Saved NN batch selection: %1").arg(path));
}

void MainWindow::onLoadNNBatchSelection() {
  if (!nnDatasetsList_ || !nnOptimizersList_) return;

  QString startDir = !projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath();
  if (!lastRuntimeWorkingDir_.trimmed().isEmpty()) startDir = lastRuntimeWorkingDir_.trimmed();

  const QString path = QFileDialog::getOpenFileName(this, "Load batch selection", startDir,
                                                      "OptimSolution batch selection (*.batchsel);;All files (*.*)");
  if (path.isEmpty()) return;

  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::warning(this, "Load batch selection", QString("Cannot open file:\n%1").arg(path));
    return;
  }

  QStringList datasets, optimizers;
  QString architectures, activation;
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);
  while (!ts.atEnd()) {
    const QString rawLine = ts.readLine().trimmed();
    if (rawLine.isEmpty() || rawLine.startsWith('#')) continue;
    const int eq = rawLine.indexOf('=');
    if (eq <= 0) continue;
    const QString key = rawLine.left(eq).trimmed();
    const QString value = rawLine.mid(eq + 1).trimmed();
    if (key.compare("datasets", Qt::CaseInsensitive) == 0) {
      for (const QString& part : value.split(',', Qt::SkipEmptyParts)) {
        const QString v = part.trimmed();
        if (!v.isEmpty() && !datasets.contains(v)) datasets << v;
      }
    } else if (key.compare("optimizers", Qt::CaseInsensitive) == 0) {
      for (const QString& part : value.split(',', Qt::SkipEmptyParts)) {
        const QString v = part.trimmed();
        if (!v.isEmpty() && !optimizers.contains(v)) optimizers << v;
      }
    } else if (key.compare("architectures", Qt::CaseInsensitive) == 0) {
      architectures = value;
    } else if (key.compare("activation", Qt::CaseInsensitive) == 0) {
      activation = value;
    }
  }

  if (datasets.isEmpty() && optimizers.isEmpty()) {
    QMessageBox::warning(this, "Load batch selection", "The selected file does not contain any datasets or optimizers.");
    return;
  }

  // Only force-switch to mode 7 if we're not already there.
  if (runModeBox_) {
    const int currentMode = runModeBox_->currentData().toInt();
    if (currentMode != 6) {
      const int idx = runModeBox_->findData(6);
      if (idx >= 0) runModeBox_->setCurrentIndex(idx);
    }
  }

  populateNNDatasetList();
  populateNNOptimizerList();

  const QSet<QString> wantedDatasets(datasets.begin(), datasets.end());
  for (int i = 0; i < nnDatasetsList_->count(); ++i) {
    if (auto* it = nnDatasetsList_->item(i)) {
      const QString fname = QFileInfo(it->data(Qt::UserRole).toString()).fileName();
      it->setSelected(wantedDatasets.contains(fname));
    }
  }
  const QSet<QString> wantedOptimizers(optimizers.begin(), optimizers.end());
  for (int i = 0; i < nnOptimizersList_->count(); ++i) {
    if (auto* it = nnOptimizersList_->item(i)) it->setSelected(wantedOptimizers.contains(it->text()));
  }

  if (!architectures.isEmpty() && nnHiddenEdit_) nnHiddenEdit_->setText(architectures);
  if (!activation.isEmpty() && nnActivationBox_) {
    const int idx = nnActivationBox_->findText(activation);
    if (idx >= 0) nnActivationBox_->setCurrentIndex(idx);
  }

  appendLog(QString("Loaded NN batch selection: %1").arg(path));
  statusBar()->showMessage(QString("NN batch selection loaded from %1").arg(QFileInfo(path).fileName()), 5000);
}

void MainWindow::refreshNNButtonStates() {
  if (nnRunAnalyzeBtn_) nnRunAnalyzeBtn_->setEnabled(nnStage1Done_);
}

// Shows what a click on "1. Train + sample landscape" would actually use for
// seeds, without requiring a click first -- reads the same
// globalRunsFromSettings()/nnSeedBaseFromSettings() this mode's batch queue
// itself uses, so the label is always an accurate preview. Called both when
// entering mode 7 (so "Seeds" isn't blank until the first run) and again
// right before building the queue (in case optimsolution.cfg changed since).
void MainWindow::refreshNNSeedInfoLabel() {
  if (!nnLandscapePanel_) return;
  auto* lbl = nnLandscapePanel_->findChild<QLabel*>("nnSeedInfoLbl");
  if (!lbl) return;
  const int runs = std::max(1, globalRunsFromSettings());
  const unsigned long long seedBase = nnSeedBaseFromSettings(settingsPath_);
  lbl->setText(QString("%1 run(s) per combination, seed = %2 .. %3 (Settings -> Global)")
                   .arg(runs).arg(seedBase).arg(seedBase + (unsigned long long)runs - 1));
}

QString MainWindow::nnActiveRunDir() const {
  return nnLoadedRunDir_.isEmpty() ? nnRuntimeWorkingDir(projectRoot_, settingsPath_) : nnLoadedRunDir_;
}

// Mirrors setPipelineUiEnabled() exactly, for mode 7: disable Selection and
// Settings, plus the two buttons that would otherwise let the user delete
// or swap out the very files a running batch/analysis is reading or
// writing. Deliberately does NOT touch the "1. Train"/"2. Analyze" buttons
// themselves (same scope setPipelineUiEnabled() has for the pipeline mode
// -- those already toggle their own "Stop" behavior independently).
void MainWindow::setNNLandscapeUiEnabled(bool enabled) {
  if (selectionBox_) selectionBox_->setEnabled(enabled);
  if (settingsBox_) settingsBox_->setEnabled(enabled);
  if (clearCsvBtn_) clearCsvBtn_->setEnabled(enabled);
  if (nnLoadExperimentBtn_) nnLoadExperimentBtn_->setEnabled(enabled);
}

// Points the Results/Mean/Wilcoxon/Rankings/LODO Results/Feature Importances
// tabs at an earlier run's folder instead of the default runtime directory.
// Purely a read/display operation: does not change where a subsequent
// "1. Train + sample landscape" run writes to (that always resets
// nnLoadedRunDir_ and goes back to the default location).
void MainWindow::onRunNNLandscapeLoadExperiment() {
  QString startDir = nnRuntimeWorkingDir(projectRoot_, settingsPath_);
  if (!QDir(startDir).exists()) {
    startDir = !lastRuntimeWorkingDir_.trimmed().isEmpty()
      ? lastRuntimeWorkingDir_
      : (!projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath());
  }

  const QString dir = QFileDialog::getExistingDirectory(
      this, "Load previous experiment (folder containing nn_summary.csv)", startDir);
  if (dir.isEmpty()) return;

  const QString summaryPath = QDir(dir).filePath("nn_summary.csv");
  if (!QFileInfo::exists(summaryPath)) {
    QMessageBox::warning(this, "Load previous experiment",
      "No nn_summary.csv found in:\n" + dir);
    return;
  }

  if (nnLoadExperimentSpinner_) nnLoadExperimentSpinner_->start();
  nnLoadExperimentBtn_->setEnabled(false);
  QApplication::processEvents();

  // ---- Parse nn_summary.csv and restore the Datasets/Optimizers/Hidden
  // layers/Activation selection that produced it, exactly like
  // onLoadPipelineExperiment() restores batchMethodsList_/batchProblemsList_
  // from batch_master.csv -- nn_summary.csv already IS this mode's "master"
  // file (one row per run, written incrementally by stage 1), so no
  // separate manifest file is needed. ----
  QStringList datasetsWanted, optimizersWanted, archsWanted; // display-ready values
  QMap<QString, int> activationCounts;
  QSet<QString> tripleKeys; // "dataset|archTag|optimizer|seed", for skip-duplicates on re-run
  {
    QFile f(summaryPath);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream ts(&f);
      const QStringList header = parseCsvLine(ts.readLine());
      auto colIndex = [&](const QString& name) { return header.indexOf(name); };
      const int cDataset = colIndex("dataset"), cArch = colIndex("arch"),
                cOpt = colIndex("optimizer"), cSeed = colIndex("seed");
      while (!ts.atEnd()) {
        const QString line = ts.readLine();
        if (line.trimmed().isEmpty()) continue;
        const QStringList row = parseCsvLine(line);
        if (row.size() <= std::max({cDataset, cArch, cOpt, cSeed})) continue;
        const QString dataset = row[cDataset].trimmed();
        const QString archTag = row[cArch].trimmed();
        const QString optimizer = row[cOpt].trimmed();
        const QString seed = row[cSeed].trimmed();
        if (dataset.isEmpty() || archTag.isEmpty() || optimizer.isEmpty()) continue;

        if (!datasetsWanted.contains(dataset)) datasetsWanted << dataset;
        if (!optimizersWanted.contains(optimizer)) optimizersWanted << optimizer;

        // archTag was written as hidden.replace(',', '-') + "_" + activation
        // (see onNNLandscapeTrainProcessFinished) -- reverse it. Activation
        // names never contain '_', so splitting on the LAST underscore is safe.
        const int lastUnderscore = archTag.lastIndexOf('_');
        QString hidden = lastUnderscore >= 0 ? archTag.left(lastUnderscore) : archTag;
        const QString activation = lastUnderscore >= 0 ? archTag.mid(lastUnderscore + 1) : QString("relu");
        hidden.replace('-', ',');
        if (!archsWanted.contains(hidden)) archsWanted << hidden;
        activationCounts[activation] += 1;

        tripleKeys.insert(dataset + "|" + archTag + "|" + optimizer + "|" + seed);
      }
      f.close();
    }
  }

  if (datasetsWanted.isEmpty() || optimizersWanted.isEmpty() || archsWanted.isEmpty()) {
    if (nnLoadExperimentSpinner_) nnLoadExperimentSpinner_->stop();
    nnLoadExperimentBtn_->setEnabled(true);
    QMessageBox::warning(this, "Load previous experiment",
      "nn_summary.csv doesn't look right (missing dataset/arch/optimizer columns, or no usable rows).");
    return;
  }

  // Restore Datasets selection (match by basename -- nn_summary.csv stores
  // the bare dataset name, list items store the .train filename + full path).
  populateNNDatasetList();
  const QSet<QString> wantedDatasets(datasetsWanted.begin(), datasetsWanted.end());
  for (int i = 0; i < nnDatasetsList_->count(); ++i) {
    if (auto* it = nnDatasetsList_->item(i)) {
      const QString base = QFileInfo(it->data(Qt::UserRole).toString()).completeBaseName();
      it->setSelected(wantedDatasets.contains(base));
    }
  }

  // Restore Optimizers selection.
  populateNNOptimizerList();
  const QSet<QString> wantedOptimizers(optimizersWanted.begin(), optimizersWanted.end());
  for (int i = 0; i < nnOptimizersList_->count(); ++i) {
    if (auto* it = nnOptimizersList_->item(i)) {
      it->setSelected(wantedOptimizers.contains(it->text()));
    }
  }

  // Restore Hidden layers (';'-joined, same convention the field already uses
  // for multiple architectures) and Activation (most frequent one found --
  // this mode applies a single activation to every architecture at once, so
  // a mixed-activation experiment can only be approximated here).
  if (nnHiddenEdit_) nnHiddenEdit_->setText(archsWanted.join(';'));
  if (nnActivationBox_ && !activationCounts.isEmpty()) {
    QString bestAct; int bestCount = -1;
    for (auto it = activationCounts.constBegin(); it != activationCounts.constEnd(); ++it)
      if (it.value() > bestCount) { bestCount = it.value(); bestAct = it.key(); }
    const int idx = nnActivationBox_->findText(bestAct);
    if (idx >= 0) nnActivationBox_->setCurrentIndex(idx);
  }

  nnLoadedRunDir_ = dir;
  nnLoadedTripleKeys_ = tripleKeys;
  nnStage1Done_ = true; // unlock "2. Analyze" so it can be re-run against the loaded data too
  refreshNNButtonStates();

  // Create the output tab FIRST: rebuildNNStats()/loadNNAnalysisTables() both
  // populate widgets (nnResultsTable_, nnMeanTable_, nnWilcoxonPlot_,
  // nnRankingsTable_, nnLodoTable_, nnImportanceTable_) that only get
  // constructed the first time createNNLandscapeOutputTab() runs. Calling
  // them in the old order (before the tab existed) meant that loading an
  // experiment as the very first action in a fresh session silently filled
  // nothing at all -- rebuildNNStats() guards on
  // "!nnWilcoxonPlot_ || !nnRankingsTable_" and just returns.
  const int outIdx = createNNLandscapeOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  rebuildNNStats();
  loadNNAnalysisTables(QDir(dir).filePath("nn_analysis"));
  if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainCheckmark") : nullptr)
    cm->setVisible(true);

  appendLog("----------------------------------------");
  appendLog(QString("Loaded previous experiment from: %1 (%2 dataset(s), %3 architecture(s), "
                     "%4 optimizer(s), %5 combination(s)). Selection restored above -- check more "
                     "items to add them, or uncheck to skip them, then run \"1. Train + sample "
                     "landscape\" again; already-present (dataset, architecture, optimizer, seed) "
                     "combinations are skipped automatically.")
                .arg(dir).arg(datasetsWanted.size()).arg(archsWanted.size())
                .arg(optimizersWanted.size()).arg(tripleKeys.size()));

  if (nnLoadExperimentSpinner_) nnLoadExperimentSpinner_->stop();
  nnLoadExperimentBtn_->setEnabled(true);
}

// ── Output tab: Log + Wilcoxon + Rankings ───────────────────────────────────
int MainWindow::createNNLandscapeOutputTab() {
  if (nnOutputTabIndex_ >= 0 && outputTab(nnOutputTabIndex_)) {
    return nnOutputTabIndex_; // reuse across both stages / repeated batches
  }
  if (!outputTabs_) return -1;

  const QString title = "NN landscape";
  auto* page = new QWidget(outputTabs_);
  auto* outerLay = new QVBoxLayout(page);
  outerLay->setContentsMargins(8, 8, 8, 8);

  auto* info = new QLabel(
      "Idle. Use \"1. Train + sample landscape\" then \"2. Analyze\" in the NN "
      "landscape panel to the left.", page);
  info->setObjectName("nnLandscapeInfoLbl");
  info->setWordWrap(true);
  outerLay->addWidget(info);

  auto* innerTabs = new QTabWidget(page);
  innerTabs->setObjectName("nnInnerTabs");
  outerLay->addWidget(innerTabs, 1);

  // Reusable table-tab builder (table + "Export XLSX..." button above it),
  // used by every results table below -- Results/Mean/Rankings/LODO/
  // Feature Importances all follow the exact same layout.
  auto makeTableTab = [&](const QString& tabTitle, const QString& objName,
                          const QStringList& columns, const QString& xlsxDefaultName) -> QTableWidget* {
    auto* p = new QWidget(innerTabs);
    auto* lay = new QVBoxLayout(p);
    lay->setContentsMargins(4, 4, 4, 4);
    auto* table = new QTableWidget(p);
    table->setObjectName(objName);
    table->setColumnCount(columns.size());
    table->setHorizontalHeaderLabels(columns);
    table->horizontalHeader()->setStretchLastSection(true);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setAlternatingRowColors(true);
    auto* exportBtn = new QPushButton("Export XLSX...", p);
    connect(exportBtn, &QPushButton::clicked, this, [this, table, xlsxDefaultName]() {
      const QString out = QFileDialog::getSaveFileName(this, "Export table (XLSX)",
                                                         QDir::homePath() + "/" + xlsxDefaultName,
                                                         "Excel Workbook (*.xlsx)");
      if (out.isEmpty()) return;
      if (!exportTableWidgetToXlsxNative(table, out))
        QMessageBox::warning(this, "Export", "Failed to export the table (it may be empty).");
    });
    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(exportBtn);
    btnRow->addStretch(1);
    lay->addLayout(btnRow);
    lay->addWidget(table, 1);
    innerTabs->addTab(p, tabTitle);
    return table;
  };

  // -- Log tab --
  auto* logPage = new QWidget(innerTabs);
  auto* logLay = new QVBoxLayout(logPage);
  logLay->setContentsMargins(4, 4, 4, 4);
  auto* log = new QTextEdit(logPage);
  log->setReadOnly(true);
  log->setLineWrapMode(QTextEdit::NoWrap);
  logLay->addWidget(log);
  innerTabs->addTab(logPage, "Log");

  // -- Results Table: one row per run, every nn_summary.csv column verbatim.
  //    The master raw-data table -- everything else here is a view/summary
  //    derived from this. Populated by rebuildNNStats(). --
  nnResultsTable_ = makeTableTab("Results Table", "nnResultsTable", {}, "nn_results.xlsx");

  // -- Mean Table: one row per (dataset, architecture, optimizer), mean+std
  //    of each metric across its seeds, plus N -- the table most directly
  //    usable as an article's own results table. --
  nnMeanTable_ = makeTableTab("Mean Table", "nnMeanTable", {}, "nn_mean_table.xlsx");

  // -- Wilcoxon tab (metric/theme/grid dropdowns, same convention as the
  //    framework's own Pairwise Wilcoxon tab -- "problem" = (dataset,
  //    architecture), "method" = optimizer) --
  auto* wTab = new QWidget(innerTabs);
  auto* wLay = new QVBoxLayout(wTab);
  wLay->setContentsMargins(4, 4, 4, 4);

  auto* wPlotRow = new QHBoxLayout();
  nnWilcoxonPlot_ = new WilcoxonBoxPlotWidget(wTab);
  wPlotRow->addWidget(nnWilcoxonPlot_, 1);

  auto* wControls = new QWidget(wTab);
  auto* wControlsLay = new QVBoxLayout(wControls);
  wControlsLay->setContentsMargins(8, 0, 0, 0);
  wControls->setFixedWidth(150);

  wControlsLay->addWidget(new QLabel("Metric", wControls));
  nnWilcoxonMetricCombo_ = new QComboBox(wControls);
  nnWilcoxonMetricCombo_->addItem("Generalization gap (loss)", "gen_gap_loss");
  nnWilcoxonMetricCombo_->addItem("Validation loss", "val_loss");
  nnWilcoxonMetricCombo_->addItem("Validation accuracy", "val_acc");
  nnWilcoxonMetricCombo_->addItem("Training loss", "train_loss");
  nnWilcoxonMetricCombo_->addItem("Generalization gap (error)", "gen_gap_err");
  connect(nnWilcoxonMetricCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &MainWindow::onNNStatsUiChanged);
  wControlsLay->addWidget(nnWilcoxonMetricCombo_);

  wControlsLay->addSpacing(6);
  wControlsLay->addWidget(new QLabel("Theme", wControls));
  auto* wThemeCombo = new QComboBox(wControls);
  wThemeCombo->addItem("Dark");
  wThemeCombo->addItem("Light");
  wThemeCombo->addItem("Transparent");
  connect(wThemeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
    if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(nnWilcoxonPlot_)) {
      auto tm = WilcoxonBoxPlotWidget::ThemeMode::Dark;
      if (index == 1) tm = WilcoxonBoxPlotWidget::ThemeMode::Light;
      else if (index == 2) tm = WilcoxonBoxPlotWidget::ThemeMode::Transparent;
      wp->setThemeMode(tm);
    }
  });
  wControlsLay->addWidget(wThemeCombo);

  wControlsLay->addSpacing(6);
  wControlsLay->addWidget(new QLabel("Grid", wControls));
  auto* wGridCombo = new QComboBox(wControls);
  wGridCombo->addItem("Sparse");
  wGridCombo->addItem("Medium");
  wGridCombo->addItem("Dense");
  wGridCombo->setCurrentIndex(1);
  connect(wGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
    if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(nnWilcoxonPlot_)) {
      auto gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
      if (index == 0) gd = WilcoxonBoxPlotWidget::GridDensity::Sparse;
      else if (index == 2) gd = WilcoxonBoxPlotWidget::GridDensity::Dense;
      wp->setGridDensity(gd);
    }
  });
  wControlsLay->addWidget(wGridCombo);
  wControlsLay->addStretch(1);

  wPlotRow->addWidget(wControls, 0);

  auto* wBtnRow = new QHBoxLayout();
  auto* wExportPngBtn = new QPushButton("Export PNG (300 DPI)...", wTab);
  wExportPngBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  wExportPngBtn->setIconSize(QSize(18, 18));
  connect(wExportPngBtn, &QPushButton::clicked, this, [this, wThemeCombo]() {
    const QString out = QFileDialog::getSaveFileName(this, "Export plot (PNG, 300 DPI)",
                                                       QDir::homePath() + "/nn_wilcoxon.png",
                                                       "PNG Image (*.png)");
    if (out.isEmpty()) return;
    const bool isTransparent = (wThemeCombo && wThemeCombo->currentIndex() == 2);
    if (!exportPipelinePlotPng(nnWilcoxonPlot_, out, isTransparent))
      QMessageBox::warning(this, "Export", "Failed to export the plot.");
  });
  wBtnRow->addWidget(wExportPngBtn);
  wBtnRow->addStretch(1);
  wLay->addLayout(wBtnRow);       // above the plot
  wLay->addLayout(wPlotRow, 1);   // plot + side controls, below the button

  nnWilcoxonSummaryLbl_ = new QLabel("No runs yet -- run \"1. Train + sample landscape\" first.", wTab);
  nnWilcoxonSummaryLbl_->setWordWrap(true);
  wLay->addWidget(nnWilcoxonSummaryLbl_, 0);

  innerTabs->addTab(wTab, "Wilcoxon");

  // -- Rankings tab --
  nnRankingsTable_ = makeTableTab("Rankings", "nnRankingsTable",
      {"Rank", "Optimizer", "Mean rank", "Mean metric", "# problems"}, "nn_rankings.xlsx");

  // -- LODO Results tab: stage 2 (Python)'s meta-only vs meta+landscape
  //    comparison -- the core scientific claim of this study, as an actual
  //    table (not just log text). Populated by loadNNAnalysisTables()
  //    after "2. Analyze" succeeds; empty until then. --
  nnLodoTable_ = makeTableTab("LODO Results", "nnLodoTable",
      {"Feature set", "N", "R2", "MAE", "Spearman"}, "nn_lodo_results.xlsx");

  // -- Feature Importances tab: same source (stage 2), which local-landscape
  //    statistics (and at which radius) actually carried predictive weight. --
  nnImportanceTable_ = makeTableTab("Feature Importances", "nnImportanceTable",
      {"Feature", "Importance"}, "nn_feature_importance.xlsx");

  // -- Bootstrap CI tab: stage 3 (optional, Python), cluster-bootstrapped
  //    95% confidence intervals on the LODO R2/MAE/Spearman numbers already
  //    shown in the LODO Results tab -- addresses "how stable is this with
  //    only a couple dozen datasets?" directly. Populated by
  //    loadNNBootstrapTable() after "3. Bootstrap CI" succeeds; empty (and
  //    the stage itself disabled) until "2. Analyze" has completed. --
  nnBootstrapTable_ = makeTableTab("Bootstrap CI", "nnBootstrapTable",
      {"Feature set", "Metric", "Median", "CI low", "CI high", "N boot"}, "nn_bootstrap_ci.xlsx");

  const int tabIndex = outputTabs_->insertTab(0, page, title);

  OutputRunTab run;
  run.title = title;
  run.page = page;
  run.log = log;
  outputRuns_.push_back(run);
  nnOutputTabIndex_ = tabIndex;

  return tabIndex;
}

// ── Stage 1: batch train + sample (C++ CLI, looped) ─────────────────────────
// Reads [nnlandscape] parallel_jobs from optimsolution.cfg. Absent, non-numeric,
// or <= 1 (the default) means "leave stage 1 exactly as it always was": one
// --nnlandscape process at a time, via nnTrainProc_. Only an explicit value
// > 1 opts into running that many independent --nnlandscape OS processes
// concurrently via startNNLandscapeJobSlots()/nnActiveProcs_. Each such
// process is fully self-contained (its own PID, its own NNProblem instance,
// its own RNG seeded from --seed) and shares no mutable state with any other
// -- unlike [global] parallel_runs elsewhere in the framework, which
// parallelizes independent runs as OpenMP *threads inside one process* and
// therefore has to special-case non-thread-safe problems (see main.cpp's
// GKLS check). Process-level parallelism has no such concern: the only
// shared resource across jobs is the nn_summary.csv file, and that is
// handled separately (see startNNLandscapeJobSlots() and mergeNNSummaryRow()).
int MainWindow::nnLandscapeParallelJobsFromSettings() const {
  if (!cfg_) return 1;
  bool ok = false;
  const int n = cfg_->value("nnlandscape", "parallel_jobs", "1").toInt(&ok);
  return (ok && n > 1) ? n : 1;
}

// See the [nnlandscape] bootstrap_jobs comment in optimsolution.cfg and
// bootstrap_lodo_ci.py's own --jobs docstring for the full reasoning
// (independent bootstrap replicates -> safe to parallelize across worker
// processes; exact draws differ slightly from a --jobs 1 run with the same
// seed, though both are equally valid).
int MainWindow::nnLandscapeBootstrapJobsFromSettings() const {
  if (!cfg_) return 1;
  bool ok = false;
  const int n = cfg_->value("nnlandscape", "bootstrap_jobs", "1").toInt(&ok);
  return (ok && n > 1) ? n : 1;
}

void MainWindow::onRunNNLandscapeTrain() {
  const bool serialRunning = nnTrainProc_ && nnTrainProc_->state() != QProcess::NotRunning;
  const bool pooledRunning = !nnActiveProcs_.isEmpty();
  if (serialRunning || pooledRunning) {
    appendLog("Stopping NN landscape batch...");
    nnQueue_.clear(); // don't advance/refill after these are killed
    if (serialRunning) {
      nnTrainProc_->terminate();
      if (!nnTrainProc_->waitForFinished(2000)) nnTrainProc_->kill();
    }
    // Iterate a COPY of nnActiveProcs_: waitForFinished() below pumps the
    // event loop, which can deliver a process's finished() signal
    // synchronously and re-enter onNNLandscapeJobSlotFinished(), which
    // mutates nnActiveProcs_ itself (removeAll) -- iterating the live
    // member here would risk invalidating the iterator mid-loop.
    const QVector<QProcess*> toStop = nnActiveProcs_;
    for (QProcess* p : toStop) {
      if (!p) continue;
      p->terminate();
      if (!p->waitForFinished(2000)) p->kill();
    }
    return;
  }

  // ---- Gather selection ----
  QVector<QPair<QString,QString>> datasets; // {absolutePath, displayName}
  QMap<QString, QStringList> datasetsByTaskType; // taskType (detailed) -> dataset display names
  QMap<QString, QStringList> datasetsByTaskFamily; // coarse "classification"/"regression" -> dataset names, for the warning below
  if (nnDatasetsList_) {
    for (auto* it : nnDatasetsList_->selectedItems()) {
      const QString path = it->data(Qt::UserRole).toString();
      if (path.isEmpty()) continue;
      datasets.push_back({path, it->text()});
      const QString taskType = it->data(Qt::UserRole + 1).toString();
      if (taskType.isEmpty()) continue;
      const QString name = QFileInfo(path).fileName();
      datasetsByTaskType[taskType] << name;
      // "3-class" vs "4-class" vs "binary" are all classification (same
      // cross-entropy loss scale) and mixing them is completely normal --
      // only regression-vs-classification actually mixes incompatible loss
      // scales, so that coarser split is what the warning below checks.
      datasetsByTaskFamily[taskType.startsWith("regression") ? "regression" : "classification"] << name;
    }
  }
  QStringList optimizers;
  if (nnOptimizersList_) {
    for (auto* it : nnOptimizersList_->selectedItems()) optimizers << it->text();
  }
  QStringList archs;
  if (nnHiddenEdit_) {
    for (const QString& a : nnHiddenEdit_->text().split(';', Qt::SkipEmptyParts))
      if (!a.trimmed().isEmpty()) archs << a.trimmed();
  }

  if (datasets.isEmpty() || optimizers.isEmpty() || archs.isEmpty()) {
    QMessageBox::warning(this, "NN landscape",
      "Select at least one dataset, one optimizer, and specify at least one "
      "architecture (Hidden layers field) first.");
    return;
  }

  // Mixing task types (e.g. a regression-formulated dataset alongside
  // classification ones) in the same batch is not an error -- each run
  // trains and evaluates its own dataset just fine -- but pooling the
  // resulting generalization gaps into one Wilcoxon/Rankings/LODO analysis
  // afterward is close to meaningless: regression losses and classification
  // losses live on unrelated scales, which is exactly what produced a
  // wildly negative LODO R^2/Spearman (an artifact of scale, not of the
  // landscape features) the first time this was tried with a same-named
  // regression/classification pair of the same underlying dataset. Warn
  // and let the person decide, rather than silently producing numbers that
  // look like a result but aren't a meaningful comparison.
  if (datasetsByTaskFamily.size() > 1) {
    QStringList parts;
    for (auto it = datasetsByTaskType.constBegin(); it != datasetsByTaskType.constEnd(); ++it)
      parts << QString("%1: %2").arg(it.key(), it.value().join(", "));
    const auto reply = QMessageBox::question(this, "Mixed task types selected",
      "The selected datasets don't all have the same task type:\n\n" + parts.join("\n") +
      "\n\nPooling regression and classification generalization gaps into one Wilcoxon/"
      "Rankings/LODO analysis is usually not meaningful, since their loss scales aren't "
      "comparable. Continue anyway?",
      QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) return;
  }

  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found",
      "optimsolution executable was not found next to the GUI or in build/Debug.");
    return;
  }

  const QString radii = nnRadiiEdit_ ? nnRadiiEdit_->text().trimmed() : QString("0.02,0.05,0.1");
  if (radii.isEmpty()) {
    QMessageBox::warning(this, "NN landscape", "Radii cannot be empty.");
    return;
  }

  // Unlike before: a new training run now targets nnActiveRunDir(), i.e. it
  // CONTINUES writing into whatever folder "Load previous experiment..."
  // last pointed at, if any -- so loading an experiment and pressing
  // "1. Train" again genuinely extends that same nn_summary.csv rather than
  // starting a second, disconnected one at the default location. To start
  // a fully fresh experiment instead, load the default optimsolution_gui_run
  // folder explicitly (or delete its nn_summary.csv via "Delete output files").

  // ---- Seed comes from optimsolution.cfg, NOT a GUI field: each combination
  // is repeated "Runs" times with seed = seed_base + run_index, exactly like
  // every other run mode already does. ----
  const int runs = std::max(1, globalRunsFromSettings());
  const unsigned long long seedBase = nnSeedBaseFromSettings(settingsPath_);
  refreshNNSeedInfoLabel();

  // ---- Build the batch queue: datasets x archs x optimizers x runs,
  // skipping any (dataset, architecture, optimizer, seed) combination
  // already present in a loaded experiment's nn_summary.csv (see
  // onRunNNLandscapeLoadExperiment()) so re-running only adds new rows. ----
  const QString activation = nnActivationBox_ ? nnActivationBox_->currentText() : QString("relu");
  nnQueue_.clear();
  int skippedDuplicates = 0;
  for (const auto& ds : datasets) {
    const QString datasetName = QFileInfo(ds.first).completeBaseName();
    for (const QString& arch : archs) {
      const QString archTag = QString(arch).replace(',', '-') + "_" + activation;
      for (const QString& opt : optimizers) {
        for (int r = 0; r < runs; ++r) {
          const QString seed = QString::number(seedBase + (unsigned long long)r);
          const QString key = datasetName + "|" + archTag + "|" + opt + "|" + seed;
          if (nnLoadedTripleKeys_.contains(key)) { ++skippedDuplicates; continue; }
          nnQueue_.push_back({ds.first, arch, opt, seed});
        }
      }
    }
  }
  nnQueueTotal_ = nnQueue_.size();
  nnQueueOk_ = 0;
  nnQueueFailed_ = 0;
  nnParallelJobs_ = nnLandscapeParallelJobsFromSettings();
  nnActiveProcs_.clear();
  nnPendingRowPath_.clear();

  if (nnQueue_.isEmpty()) {
    QMessageBox::information(this, "NN landscape",
      QString("Every combination in the current selection (%1 total) is already present in the "
              "loaded experiment -- nothing new to run. Add a dataset/optimizer/seed count, or "
              "change Runs in Settings -> Global, to run something new.").arg(skippedDuplicates));
    return;
  }

  const int outIdx = createNNLandscapeOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  appendLog("----------------------------------------");
  if (skippedDuplicates > 0)
    appendLog(QString("  (%1 combination(s) already present in the loaded experiment were skipped.)")
                  .arg(skippedDuplicates));
  appendLog(QString("[1/2] NN landscape batch: %1 dataset(s) x %2 architecture(s) x %3 optimizer(s) "
                     "x %4 run(s) = %5 combination(s) queued.")
                .arg(datasets.size()).arg(archs.size()).arg(optimizers.size()).arg(runs).arg(nnQueueTotal_));
  if (nnParallelJobs_ > 1) {
    appendLog(QString("  (optimsolution.cfg [nnlandscape] parallel_jobs=%1: running up to %1 combinations "
                       "concurrently.)").arg(nnParallelJobs_));
  }
  if (nnTrainSpinner_) nnTrainSpinner_->start();
  if (nnRunTrainBtn_) nnRunTrainBtn_->setText("Stop");
  if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainCheckmark") : nullptr)
    cm->setVisible(false);
  setNNLandscapeUiEnabled(false);
  startPipelineProgress(QString("NN landscape (0/%1)...").arg(nnQueueTotal_));
  nnStage1Timer_.start();
  if (nnParallelJobs_ > 1) {
    startNNLandscapeJobSlots(); // fills up to nnParallelJobs_ concurrent slots from nnQueue_
  } else {
    onNNLandscapeTrainProcessFinished(0, QProcess::NormalExit); // kicks off the first combination (unchanged serial path)
  }
}

// Advances nnQueue_ by one combination each time it's called -- once as a
// direct kick-off (nnTrainProc_ == nullptr) then once per QProcess::finished
// until the queue is empty. Mirrors onPipelineFeaturesProcessFinished()'s
// exact queue-draining pattern.
void MainWindow::onNNLandscapeTrainProcessFinished(int exitCode, QProcess::ExitStatus status) {
  const bool isInitialKickoff = (nnTrainProc_ == nullptr);
  if (!isInitialKickoff) {
    if (status == QProcess::NormalExit && exitCode == 0) ++nnQueueOk_;
    else {
      ++nnQueueFailed_;
      appendLog(QString("  [nnlandscape] failed (exit code %1) -- continuing with remaining combinations.")
                    .arg(exitCode));
    }
  }

  if (nnQueue_.isEmpty()) {
    if (nnTrainSpinner_) nnTrainSpinner_->stop();
    if (nnRunTrainBtn_) nnRunTrainBtn_->setText("1. Train + sample landscape");
    if (isInitialKickoff) return; // nothing was ever queued (shouldn't happen; guarded earlier)

    const double secs = nnStage1Timer_.isValid() ? nnStage1Timer_.elapsed() / 1000.0 : 0.0;
    finishPipelineProgress(nnQueueFailed_ == 0);
    setNNLandscapeUiEnabled(true);
    appendLog(QString("[1/2] Batch done in %1: %2 ok, %3 failed, out of %4. "
                       "Rows appended to nn_summary.csv; *_features.csv written to optimsolution_gui_run/.")
                  .arg(formatNNDuration(secs)).arg(nnQueueOk_).arg(nnQueueFailed_).arg(nnQueueTotal_));
    if (auto* lbl = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainTimeLbl") : nullptr)
      lbl->setText(QString("(%1, %2/%3 ok)").arg(formatNNDuration(secs)).arg(nnQueueOk_).arg(nnQueueTotal_));
    if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainCheckmark") : nullptr)
      cm->setVisible(nnQueueOk_ > 0);

    if (nnQueueOk_ > 0) {
      nnStage1Done_ = true;
      refreshNNButtonStates();
      rebuildNNStats();
    }
    return;
  }

  if (outputProgressBar_ && !isInitialKickoff) {
    const int doneSoFar = nnQueueTotal_ - nnQueue_.size();
    outputProgressBar_->setFormat(QString("NN landscape (%1/%2)...").arg(doneSoFar).arg(nnQueueTotal_));
  }

  const QStringList combo = nnQueue_.front();
  nnQueue_.erase(nnQueue_.begin());
  const QString datasetPath = combo[0];
  const QString arch = combo[1];
  const QString optimizer = combo[2];
  const QString seed = combo[3];

  const QString runtimeWd = nnActiveRunDir();
  lastRuntimeWorkingDir_ = runtimeWd;

  const QString datasetName = QFileInfo(datasetPath).completeBaseName();
  const QString activation = nnActivationBox_ ? nnActivationBox_->currentText() : QString("relu");
  const QString archTag = QString(arch).replace(',', '-') + "_" + activation;
  const QString prefix = QDir(runtimeWd).filePath(
      QString("%1_%2_%3_seed%4").arg(datasetName, archTag, optimizer, seed));
  const QString summaryCsv = QDir(runtimeWd).filePath("nn_summary.csv");
  const QString label = nnLabelEdit_ ? nnLabelEdit_->text().trimmed() : QString();

  QStringList args;
  args << "--nnlandscape"
       << "--dataset" << datasetPath
       << "--hidden" << arch
       << "--activation" << activation
       << "--optimizer" << optimizer
       << "--iterations" << QString::number(nnIterationsSpin_ ? nnIterationsSpin_->value() : 200)
       << "--population" << QString::number(nnPopulationSpin_ ? nnPopulationSpin_->value() : 40)
       << "--wbound" << QString::number(nnWBoundSpin_ ? nnWBoundSpin_->value() : 5.0, 'g', 6)
       << "--seed" << seed
       << "--radii" << (nnRadiiEdit_ ? nnRadiiEdit_->text().trimmed() : QString("0.02,0.05,0.1"))
       << "--local-samples" << QString::number(nnLocalSamplesSpin_ ? nnLocalSamplesSpin_->value() : 0)
       << "--out-prefix" << prefix
       << "--out-summary" << summaryCsv;
  if (!label.isEmpty()) args << "--label" << label;

  const QString cli = cliPath();
  if (nnTrainProc_) { nnTrainProc_->deleteLater(); nnTrainProc_ = nullptr; }
  nnTrainProc_ = new QProcess(this);
  nnTrainProc_->setProgram(cli);
  nnTrainProc_->setWorkingDirectory(runtimeWd);
  nnTrainProc_->setArguments(args);

  connect(nnTrainProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onNNLandscapeTrainProcessFinished);
  connect(nnTrainProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (nnTrainProc_) appendLog("  " + QString::fromUtf8(nnTrainProc_->readAllStandardOutput()).trimmed());
  });
  connect(nnTrainProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (nnTrainProc_) appendLog("  " + QString::fromUtf8(nnTrainProc_->readAllStandardError()).trimmed());
  });

  const int doneCount = nnQueueTotal_ - (int)nnQueue_.size();
  appendLog(QString("  (%1/%2) %3 hidden=%4 act=%5 opt=%6 seed=%7 ...")
                .arg(doneCount).arg(nnQueueTotal_).arg(datasetName, arch, activation, optimizer, seed));
  nnTrainProc_->start();
}

// ── Stage 1, pooled variant (nnParallelJobs_ > 1 only) ──────────────────────
// Fills every currently-free slot (up to nnParallelJobs_ concurrent
// --nnlandscape processes) from nnQueue_. Called once to seed the pool, and
// again from onNNLandscapeJobSlotFinished() every time a slot frees up, so it
// naturally keeps exactly nnParallelJobs_ jobs in flight until the queue runs
// dry. Mirrors onNNLandscapeTrainProcessFinished()'s argument-building
// exactly, with ONE deliberate difference: --out-summary points at a private,
// per-job row file instead of the shared nn_summary.csv (see
// mergeNNSummaryRow() for why).
void MainWindow::startNNLandscapeJobSlots() {
  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) return; // already validated before the batch started

  const QString runtimeWd = nnActiveRunDir();
  lastRuntimeWorkingDir_ = runtimeWd;
  const QString activation = nnActivationBox_ ? nnActivationBox_->currentText() : QString("relu");
  const QString label = nnLabelEdit_ ? nnLabelEdit_->text().trimmed() : QString();

  while (nnActiveProcs_.size() < nnParallelJobs_ && !nnQueue_.isEmpty()) {
    const QStringList combo = nnQueue_.front();
    nnQueue_.erase(nnQueue_.begin());
    const QString datasetPath = combo[0];
    const QString arch = combo[1];
    const QString optimizer = combo[2];
    const QString seed = combo[3];

    const QString datasetName = QFileInfo(datasetPath).completeBaseName();
    const QString archTag = QString(arch).replace(',', '-') + "_" + activation;
    const QString prefix = QDir(runtimeWd).filePath(
        QString("%1_%2_%3_seed%4").arg(datasetName, archTag, optimizer, seed));
    // Unique per job (dataset/arch/optimizer/seed are already unique per
    // combination) -- two concurrent jobs never target the same row file,
    // so no locking is needed for THIS file either.
    const QString rowPath = prefix + "_row_summary.csv";

    QStringList args;
    args << "--nnlandscape"
         << "--dataset" << datasetPath
         << "--hidden" << arch
         << "--activation" << activation
         << "--optimizer" << optimizer
         << "--iterations" << QString::number(nnIterationsSpin_ ? nnIterationsSpin_->value() : 200)
         << "--population" << QString::number(nnPopulationSpin_ ? nnPopulationSpin_->value() : 40)
         << "--wbound" << QString::number(nnWBoundSpin_ ? nnWBoundSpin_->value() : 5.0, 'g', 6)
         << "--seed" << seed
         << "--radii" << (nnRadiiEdit_ ? nnRadiiEdit_->text().trimmed() : QString("0.02,0.05,0.1"))
         << "--local-samples" << QString::number(nnLocalSamplesSpin_ ? nnLocalSamplesSpin_->value() : 0)
         << "--out-prefix" << prefix
         << "--out-summary" << rowPath;
    if (!label.isEmpty()) args << "--label" << label;

    QProcess* proc = new QProcess(this);
    proc->setProgram(cli);
    proc->setWorkingDirectory(runtimeWd);
    proc->setArguments(args);

    connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onNNLandscapeJobSlotFinished);
    connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc]() {
      appendLog("  " + QString::fromUtf8(proc->readAllStandardOutput()).trimmed());
    });
    connect(proc, &QProcess::readyReadStandardError, this, [this, proc]() {
      appendLog("  " + QString::fromUtf8(proc->readAllStandardError()).trimmed());
    });

    nnActiveProcs_.push_back(proc);
    nnPendingRowPath_[proc] = rowPath;

    const int startedSoFar = nnQueueTotal_ - nnQueue_.size();
    appendLog(QString("  (%1/%2) %3 hidden=%4 act=%5 opt=%6 seed=%7 ...")
                  .arg(startedSoFar).arg(nnQueueTotal_).arg(datasetName, arch, activation, optimizer, seed));
    proc->start();
  }
}

// Slot for QProcess::finished on any pooled job; identifies which process via
// sender() since several may be in flight. On success, merges that job's
// private row file into the shared nn_summary.csv (mergeNNSummaryRow()) --
// this call, like every other Qt slot invocation, runs on the GUI's single
// main thread, so it can never race with another job's merge even though the
// jobs themselves ran concurrently. Refills the freed slot from nnQueue_, and
// once the queue is empty AND every slot has drained, reports exactly like
// the original serial path's "queue empty" branch did.
void MainWindow::onNNLandscapeJobSlotFinished(int exitCode, QProcess::ExitStatus status) {
  QProcess* proc = qobject_cast<QProcess*>(sender());
  if (!proc) return;

  const QString rowPath = nnPendingRowPath_.take(proc);
  nnActiveProcs_.removeAll(proc);

  const bool ok = (status == QProcess::NormalExit && exitCode == 0);
  if (ok) {
    const QString summaryCsv = QDir(nnActiveRunDir()).filePath("nn_summary.csv");
    QString mergeErr;
    if (mergeNNSummaryRow(summaryCsv, rowPath, &mergeErr)) {
      ++nnQueueOk_;
    } else {
      ++nnQueueFailed_;
      appendLog(QString("  [nnlandscape] training succeeded but its summary row could not be merged "
                         "(%1) -- counting this combination as failed.").arg(mergeErr));
    }
  } else {
    ++nnQueueFailed_;
    appendLog(QString("  [nnlandscape] failed (exit code %1) -- continuing with remaining combinations.")
                  .arg(exitCode));
    QFile::remove(rowPath); // best-effort cleanup of a possibly partial/missing row file
  }

  proc->deleteLater();

  const int doneSoFar = nnQueueTotal_ - nnQueue_.size() - nnActiveProcs_.size();
  if (outputProgressBar_) {
    outputProgressBar_->setFormat(QString("NN landscape (%1/%2)...").arg(doneSoFar).arg(nnQueueTotal_));
  }

  if (!nnQueue_.isEmpty()) {
    startNNLandscapeJobSlots(); // keep the pool full
    return;
  }
  if (!nnActiveProcs_.isEmpty()) return; // other jobs still in flight -- wait for them too

  // Every combination (across every slot) is now accounted for -- identical
  // wrap-up to onNNLandscapeTrainProcessFinished()'s "queue empty" branch.
  if (nnTrainSpinner_) nnTrainSpinner_->stop();
  if (nnRunTrainBtn_) nnRunTrainBtn_->setText("1. Train + sample landscape");
  const double secs = nnStage1Timer_.isValid() ? nnStage1Timer_.elapsed() / 1000.0 : 0.0;
  finishPipelineProgress(nnQueueFailed_ == 0);
  setNNLandscapeUiEnabled(true);
  appendLog(QString("[1/2] Batch done in %1: %2 ok, %3 failed, out of %4. "
                     "Rows appended to nn_summary.csv; *_features.csv written to optimsolution_gui_run/.")
                .arg(formatNNDuration(secs)).arg(nnQueueOk_).arg(nnQueueFailed_).arg(nnQueueTotal_));
  if (auto* lbl = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainTimeLbl") : nullptr)
    lbl->setText(QString("(%1, %2/%3 ok)").arg(formatNNDuration(secs)).arg(nnQueueOk_).arg(nnQueueTotal_));
  if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnTrainCheckmark") : nullptr)
    cm->setVisible(nnQueueOk_ > 0);

  if (nnQueueOk_ > 0) {
    nnStage1Done_ = true;
    refreshNNButtonStates();
    rebuildNNStats();
  }
}

// ── Stage 2: analyze (Python) ───────────────────────────────────────────────
void MainWindow::onRunNNLandscapeAnalyze() {
  if (nnAnalyzeProc_ && nnAnalyzeProc_->state() != QProcess::NotRunning) {
    appendLog("Stopping analysis...");
    nnAnalyzeProc_->terminate();
    if (!nnAnalyzeProc_->waitForFinished(2000)) nnAnalyzeProc_->kill();
    return;
  }

  const QString runtimeWd = nnActiveRunDir();
  const QString summaryCsv = QDir(runtimeWd).filePath("nn_summary.csv");
  if (!QFileInfo::exists(summaryCsv)) {
    QMessageBox::information(this, "NN landscape",
      "No nn_summary.csv yet -- run \"1. Train + sample landscape\" at least once first.");
    return;
  }

  const QStringList required = {"pandas", "numpy", "scipy", "sklearn"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, scipy, and "
      "scikit-learn all importable (tried \"py\" with several version "
      "selectors, \"python\", and \"python3\").\n\n"
      "Install with: pip install pandas numpy scipy scikit-learn");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("train_gap_regressor_logtransform.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  const QString outDir = QDir(runtimeWd).filePath("nn_analysis");
  QDir().mkpath(outDir);

  const int outIdx = createNNLandscapeOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (nnAnalyzeProc_) { nnAnalyzeProc_->deleteLater(); nnAnalyzeProc_ = nullptr; }
  nnAnalyzeProc_ = new QProcess(this);
  nnAnalyzeProc_->setProgram(pythonProg);
  nnAnalyzeProc_->setWorkingDirectory(runtimeWd);
  QStringList analyzeArgs = pythonParts + QStringList{script, runtimeWd, summaryCsv, outDir};
  const bool logTransform = nnLogTransformCheck_ && nnLogTransformCheck_->isChecked();
  if (logTransform) analyzeArgs << "--log-transform";
  nnAnalyzeProc_->setArguments(analyzeArgs);

  connect(nnAnalyzeProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onNNLandscapeAnalyzeProcessFinished);
  connect(nnAnalyzeProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (nnAnalyzeProc_) appendLog(QString::fromUtf8(nnAnalyzeProc_->readAllStandardOutput()).trimmed());
  });
  connect(nnAnalyzeProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (nnAnalyzeProc_) appendLog(QString::fromUtf8(nnAnalyzeProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[2/2] Analyzing (%1%2): merging landscape samples with nn_summary.csv, "
                     "running Leave-One-Dataset-Out regression...")
                .arg(pythonSpec).arg(logTransform ? ", log-transformed" : ""));
  if (nnAnalyzeSpinner_) nnAnalyzeSpinner_->start();
  if (nnRunAnalyzeBtn_) nnRunAnalyzeBtn_->setText("Stop");
  if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnAnalyzeCheckmark") : nullptr)
    cm->setVisible(false);
  setNNLandscapeUiEnabled(false);
  startPipelineProgress("Analyzing...");
  nnStage2Timer_.start();
  nnAnalyzeProc_->start();
}

void MainWindow::onNNLandscapeAnalyzeProcessFinished(int exitCode, QProcess::ExitStatus status) {
  if (nnAnalyzeSpinner_) nnAnalyzeSpinner_->stop();
  if (nnRunAnalyzeBtn_) { nnRunAnalyzeBtn_->setText("2. Analyze (Python)"); nnRunAnalyzeBtn_->setEnabled(nnStage1Done_); }
  setNNLandscapeUiEnabled(true);

  const double secs = nnStage2Timer_.isValid() ? nnStage2Timer_.elapsed() / 1000.0 : 0.0;
  const bool ok = (status == QProcess::NormalExit && exitCode == 0);
  finishPipelineProgress(ok);
  if (ok) {
    appendLog(QString("[2/2] Done in %1. See the LODO Results / Feature Importances tabs; "
                       "merged_dataset.csv written to optimsolution_gui_run/nn_analysis/.")
                  .arg(formatNNDuration(secs)));
    if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnAnalyzeCheckmark") : nullptr)
      cm->setVisible(true);
    loadNNAnalysisTables(QDir(nnActiveRunDir()).filePath("nn_analysis"));
    nnStage2Done_ = true;
    if (nnRunBootstrapBtn_) nnRunBootstrapBtn_->setEnabled(true);
  } else {
    appendLog(QString("[2/2] Failed (exit code %1) after %2 -- see log above for details.")
                  .arg(exitCode).arg(formatNNDuration(secs)));
  }
  if (auto* lbl = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnAnalyzeTimeLbl") : nullptr)
    lbl->setText(QString("(%1)").arg(formatNNDuration(secs)));
}

// ── Stage 3 (optional): bootstrap CI on the LODO numbers (Python) ──────────
// Reads stage 2's own merged_dataset.csv -- no re-run of stage 1 or 2 is
// needed, and no *_features.csv access is required here either, since
// merged_dataset.csv already has every landscape/meta/optimizer column
// joined with the target. Cluster-bootstraps over datasets (the natural
// resampling unit for Leave-One-Dataset-Out) to put a 95% CI on R2/MAE/
// Spearman for both the meta-only and meta+landscape feature sets.
void MainWindow::onRunNNLandscapeBootstrap() {
  if (nnBootstrapProc_ && nnBootstrapProc_->state() != QProcess::NotRunning) {
    appendLog("Stopping bootstrap...");
    nnBootstrapProc_->terminate();
    if (!nnBootstrapProc_->waitForFinished(2000)) nnBootstrapProc_->kill();
    return;
  }

  const QString runtimeWd = nnActiveRunDir();
  const QString outDir = QDir(runtimeWd).filePath("nn_analysis");
  const QString mergedCsv = QDir(outDir).filePath("merged_dataset.csv");
  if (!QFileInfo::exists(mergedCsv)) {
    QMessageBox::information(this, "NN landscape",
      "No merged_dataset.csv yet -- run \"2. Analyze\" at least once first.");
    return;
  }

  const QStringList required = {"pandas", "numpy", "scipy", "sklearn"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, scipy, and "
      "scikit-learn all importable (tried \"py\" with several version "
      "selectors, \"python\", and \"python3\").\n\n"
      "Install with: pip install pandas numpy scipy scikit-learn");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("bootstrap_lodo_ci.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  const int outIdx = createNNLandscapeOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (nnBootstrapProc_) { nnBootstrapProc_->deleteLater(); nnBootstrapProc_ = nullptr; }
  nnBootstrapProc_ = new QProcess(this);
  nnBootstrapProc_->setProgram(pythonProg);
  nnBootstrapProc_->setWorkingDirectory(runtimeWd);
  QStringList bootstrapArgs = pythonParts + QStringList{script, mergedCsv, "--n-boot", "1000"};
  const int bootstrapJobs = nnLandscapeBootstrapJobsFromSettings();
  if (bootstrapJobs > 1) bootstrapArgs << "--jobs" << QString::number(bootstrapJobs);
  nnBootstrapProc_->setArguments(bootstrapArgs);

  connect(nnBootstrapProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onNNLandscapeBootstrapProcessFinished);
  connect(nnBootstrapProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (nnBootstrapProc_) appendLog(QString::fromUtf8(nnBootstrapProc_->readAllStandardOutput()).trimmed());
  });
  connect(nnBootstrapProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (nnBootstrapProc_) appendLog(QString::fromUtf8(nnBootstrapProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[3/3] Bootstrapping (%1): 1000 cluster resamples over datasets, "
                     "re-running LODO on each to get a 95%% CI on R2/MAE/Spearman. "
                     "This can take a while (roughly 1000x the cost of a single \"2. Analyze\" run) "
                     "since it retrains the random forest once per resample per feature set...")
                .arg(pythonSpec));
  if (bootstrapJobs > 1) {
    appendLog(QString("  (optimsolution.cfg [nnlandscape] bootstrap_jobs=%1: running that many "
                       "bootstrap replicates concurrently.)").arg(bootstrapJobs));
  }
  if (nnBootstrapSpinner_) nnBootstrapSpinner_->start();
  if (nnRunBootstrapBtn_) nnRunBootstrapBtn_->setText("Stop");
  if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnBootstrapCheckmark") : nullptr)
    cm->setVisible(false);
  setNNLandscapeUiEnabled(false);
  startPipelineProgress("Bootstrapping...");
  nnStage3Timer_.start();
  nnBootstrapProc_->start();
}

void MainWindow::onNNLandscapeBootstrapProcessFinished(int exitCode, QProcess::ExitStatus status) {
  if (nnBootstrapSpinner_) nnBootstrapSpinner_->stop();
  if (nnRunBootstrapBtn_) { nnRunBootstrapBtn_->setText("3. Bootstrap CI (optional)"); nnRunBootstrapBtn_->setEnabled(nnStage2Done_); }
  setNNLandscapeUiEnabled(true);

  const double secs = nnStage3Timer_.isValid() ? nnStage3Timer_.elapsed() / 1000.0 : 0.0;
  const bool ok = (status == QProcess::NormalExit && exitCode == 0);
  finishPipelineProgress(ok);
  if (ok) {
    appendLog(QString("[3/3] Done in %1. See the Bootstrap CI tab; "
                       "merged_dataset_bootstrap_ci.csv written to optimsolution_gui_run/nn_analysis/.")
                  .arg(formatNNDuration(secs)));
    if (auto* cm = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnBootstrapCheckmark") : nullptr)
      cm->setVisible(true);
    loadNNBootstrapTable(QDir(nnActiveRunDir()).filePath("nn_analysis"));
  } else {
    appendLog(QString("[3/3] Failed (exit code %1) after %2 -- see log above for details.")
                  .arg(exitCode).arg(formatNNDuration(secs)));
  }
  if (auto* lbl = nnActionPanel_ ? nnActionPanel_->findChild<QLabel*>("nnBootstrapTimeLbl") : nullptr)
    lbl->setText(QString("(%1)").arg(formatNNDuration(secs)));
}

void MainWindow::onNNStatsUiChanged() {
  rebuildNNStats();
}

// Reads lodo_results.csv and feature_importance.csv (written by
// py_scripts/train_gap_regressor.py's stage 2) into their own tables --
// the core scientific comparison (meta-only vs meta+landscape features)
// and which landscape statistics carried the predictive weight, both as
// actual exportable tables rather than only log text.
void MainWindow::loadNNAnalysisTables(const QString& outDir) {
  auto loadCsvIntoTable = [](const QString& path, QTableWidget* table) {
    if (!table) return;
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      table->setRowCount(0);
      return;
    }
    QTextStream ts(&f);
    QVector<QStringList> rows;
    while (!ts.atEnd()) {
      const QString line = ts.readLine();
      if (line.trimmed().isEmpty()) continue;
      rows.push_back(parseCsvLine(line));
    }
    f.close();
    if (rows.isEmpty()) { table->setRowCount(0); return; }

    const QStringList header = rows.front();
    table->setColumnCount(header.size());
    table->setHorizontalHeaderLabels(header);
    table->setRowCount(rows.size() - 1);
    for (int r = 1; r < rows.size(); ++r) {
      const QStringList& row = rows[r];
      for (int c = 0; c < header.size(); ++c)
        table->setItem(r - 1, c, new QTableWidgetItem(c < row.size() ? row[c] : QString()));
    }
    table->resizeColumnsToContents();
  };

  loadCsvIntoTable(QDir(outDir).filePath("lodo_results.csv"), nnLodoTable_);
  loadCsvIntoTable(QDir(outDir).filePath("feature_importance.csv"), nnImportanceTable_);
}

// Stage 3 (optional): reads merged_dataset_bootstrap_ci.csv (written by
// py_scripts/bootstrap_lodo_ci.py) into its own table -- one row per
// (feature set, metric), with median + 95% CI, cluster-bootstrapped over
// datasets. Same loadCsvIntoTable helper/layout as loadNNAnalysisTables().
void MainWindow::loadNNBootstrapTable(const QString& outDir) {
  auto loadCsvIntoTable = [](const QString& path, QTableWidget* table) {
    if (!table) return;
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      table->setRowCount(0);
      return;
    }
    QTextStream ts(&f);
    QVector<QStringList> rows;
    while (!ts.atEnd()) {
      const QString line = ts.readLine();
      if (line.trimmed().isEmpty()) continue;
      rows.push_back(parseCsvLine(line));
    }
    f.close();
    if (rows.isEmpty()) { table->setRowCount(0); return; }

    const QStringList header = rows.front();
    table->setColumnCount(header.size());
    table->setHorizontalHeaderLabels(header);
    table->setRowCount(rows.size() - 1);
    for (int r = 1; r < rows.size(); ++r) {
      const QStringList& row = rows[r];
      for (int c = 0; c < header.size(); ++c)
        table->setItem(r - 1, c, new QTableWidgetItem(c < row.size() ? row[c] : QString()));
    }
    table->resizeColumnsToContents();
  };

  // bootstrap_lodo_ci.py derives this name from its <merged_csv> argument
  // (".../merged_dataset.csv" -> ".../merged_dataset_bootstrap_ci.csv").
  loadCsvIntoTable(QDir(outDir).filePath("merged_dataset_bootstrap_ci.csv"), nnBootstrapTable_);
}

// ── Wilcoxon + Rankings: read nn_summary.csv fresh, aggregate per
//    (problem=dataset+arch, method=optimizer), pairwise Wilcoxon between
//    methods paired by problem, Friedman-style average rank per method. ────
void MainWindow::rebuildNNStats() {
  if (!nnWilcoxonPlot_ || !nnRankingsTable_) return;
  auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(nnWilcoxonPlot_);

  const QString runtimeWd = nnActiveRunDir();
  const QString summaryCsv = QDir(runtimeWd).filePath("nn_summary.csv");

  QFile f(summaryCsv);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    if (wp) wp->clear();
    nnRankingsTable_->setRowCount(0);
    if (nnResultsTable_) { nnResultsTable_->setColumnCount(0); nnResultsTable_->setRowCount(0); }
    if (nnMeanTable_) { nnMeanTable_->setColumnCount(0); nnMeanTable_->setRowCount(0); }
    if (nnWilcoxonSummaryLbl_) nnWilcoxonSummaryLbl_->setText("No nn_summary.csv yet -- run \"1. Train + sample landscape\" first.");
    return;
  }

  QTextStream ts(&f);
  const QStringList header = parseCsvLine(ts.readLine());
  auto colIndex = [&](const QString& name) { return header.indexOf(name); };
  const int cDataset = colIndex("dataset"), cArch = colIndex("arch"), cOpt = colIndex("optimizer");

  // Read every row once; everything below (Results Table, Mean Table,
  // Wilcoxon, Rankings) is derived from this same in-memory copy.
  QVector<QStringList> rows;
  while (!ts.atEnd()) {
    const QString line = ts.readLine();
    if (line.trimmed().isEmpty()) continue;
    rows.push_back(parseCsvLine(line));
  }
  f.close();

  // ---- Results Table: every column, every run, verbatim -- "the most
  // important information", straight from disk, for copy-into-the-article
  // use without any aggregation or interpretation applied. ----
  if (nnResultsTable_) {
    nnResultsTable_->setColumnCount(header.size());
    nnResultsTable_->setHorizontalHeaderLabels(header);
    nnResultsTable_->setRowCount(rows.size());
    for (int r = 0; r < rows.size(); ++r) {
      const QStringList& row = rows[r];
      for (int c = 0; c < header.size(); ++c)
        nnResultsTable_->setItem(r, c, new QTableWidgetItem(c < row.size() ? row[c] : QString()));
    }
    nnResultsTable_->resizeColumnsToContents();
  }

  // ---- Mean Table: one row per (dataset, architecture, optimizer), with
  // mean+std+N for every numeric metric column -- the table shape most
  // directly usable as an article's own results table. ----
  if (nnMeanTable_ && cDataset >= 0 && cArch >= 0 && cOpt >= 0) {
    static const QVector<QPair<QString,QString>> meanCols = {
        {"train_loss", "Train loss"}, {"val_loss", "Val loss"},
        {"train_acc", "Train acc"}, {"val_acc", "Val acc"},
        {"gen_gap_loss", "Gen. gap (loss)"}, {"gen_gap_err", "Gen. gap (err)"}};
    QVector<int> mIdx;
    for (const auto& mc : meanCols) mIdx.push_back(colIndex(mc.first));

    // key = dataset|arch|optimizer -> list of row indices
    QMap<QString, QVector<int>> groups;
    QMap<QString, QStringList> groupLabel; // key -> {dataset, arch, optimizer}
    for (int r = 0; r < rows.size(); ++r) {
      const QStringList& row = rows[r];
      if (row.size() <= std::max({cDataset, cArch, cOpt})) continue;
      const QString key = row[cDataset] + "\x1f" + row[cArch] + "\x1f" + row[cOpt];
      groups[key].push_back(r);
      groupLabel[key] = {row[cDataset], row[cArch], row[cOpt]};
    }

    QStringList meanHeader = {"Dataset", "Arch", "Optimizer", "N"};
    for (const auto& mc : meanCols) meanHeader << (mc.second + " (mean)") << (mc.second + " (std)");
    nnMeanTable_->setColumnCount(meanHeader.size());
    nnMeanTable_->setHorizontalHeaderLabels(meanHeader);
    nnMeanTable_->setRowCount(groups.size());

    int r = 0;
    for (auto it = groups.constBegin(); it != groups.constEnd(); ++it, ++r) {
      const QStringList& lbl = groupLabel[it.key()];
      nnMeanTable_->setItem(r, 0, new QTableWidgetItem(lbl[0]));
      nnMeanTable_->setItem(r, 1, new QTableWidgetItem(lbl[1]));
      nnMeanTable_->setItem(r, 2, new QTableWidgetItem(lbl[2]));
      nnMeanTable_->setItem(r, 3, new QTableWidgetItem(QString::number(it.value().size())));
      for (int m = 0; m < mIdx.size(); ++m) {
        double sum = 0.0, sumsq = 0.0; int n = 0;
        if (mIdx[m] >= 0) {
          for (int rowIdx : it.value()) {
            const QStringList& row = rows[rowIdx];
            if (mIdx[m] >= row.size()) continue;
            bool ok = false; const double v = row[mIdx[m]].toDouble(&ok);
            if (!ok || !std::isfinite(v)) continue;
            sum += v; sumsq += v * v; ++n;
          }
        }
        const double mean = n > 0 ? sum / n : std::numeric_limits<double>::quiet_NaN();
        const double var = n > 1 ? std::max(0.0, sumsq / n - mean * mean) : 0.0;
        nnMeanTable_->setItem(r, 4 + 2 * m, new QTableWidgetItem(n > 0 ? QString::number(mean, 'g', 6) : QString("-")));
        nnMeanTable_->setItem(r, 5 + 2 * m, new QTableWidgetItem(n > 0 ? QString::number(std::sqrt(var), 'g', 4) : QString("-")));
      }
    }
    nnMeanTable_->resizeColumnsToContents();
  }

  const QString metricCol = nnWilcoxonMetricCombo_ ? nnWilcoxonMetricCombo_->currentData().toString() : QString("gen_gap_loss");
  const int cMetric = colIndex(metricCol);
  const double metricScale = nnMetricUnitScale(metricCol);
  if (cDataset < 0 || cArch < 0 || cOpt < 0 || cMetric < 0) {
    if (wp) wp->clear();
    nnRankingsTable_->setRowCount(0);
    if (nnWilcoxonSummaryLbl_) nnWilcoxonSummaryLbl_->setText("nn_summary.csv is missing expected columns.");
    return;
  }

  // raw[problem][method] = list of metric values (one per seed/run),
  // scaled by metricScale (100x for ratio-valued metrics -- val_acc,
  // gen_gap_err -- so the numbers actually match the "[%]" unit shown on
  // the Wilcoxon plot's Y-axis; 1x, i.e. unchanged, for loss-valued ones).
  //
  // Regression runs carry train_acc/val_acc as the sentinel -1.0
  // (NNProblem::trainAcc()/valAcc() -- accuracy is meaningless for a
  // regression task), NOT a genuine 0-1 accuracy value. Confirmed as the
  // exact cause of an exported "Validation accuracy" plot on a
  // regression-only pool showing every point pinned to -100% (the sentinel,
  // scaled) rather than the intended "nothing to plot" -- checked on the
  // RAW (pre-scale) value, before it's mistaken for a real -100%
  // measurement, and skipped exactly like a non-finite value already is.
  QMap<QString, QMap<QString, QVector<double>>> raw;
  const bool metricCanBeNA = (metricCol == "val_acc" || metricCol == "train_acc");
  int rowsSeen = 0, rowsSkippedForMetric = 0;
  for (const QStringList& row : rows) {
    if (row.size() <= std::max({cDataset, cArch, cOpt, cMetric})) continue;
    ++rowsSeen;
    bool ok = false;
    const double rawV = row[cMetric].toDouble(&ok);
    if (!ok || !std::isfinite(rawV)) { ++rowsSkippedForMetric; continue; }
    if (metricCanBeNA && rawV <= -1.0 + 1e-9 && rawV >= -1.0 - 1e-9) { ++rowsSkippedForMetric; continue; } // "not applicable" sentinel
    const double v = rawV * metricScale;
    const QString problem = row[cDataset] + " | " + row[cArch];
    raw[problem][row[cOpt]].push_back(v);
  }
  f.close();

  if (raw.isEmpty()) {
    if (wp) {
      if (rowsSeen > 0 && rowsSkippedForMetric == rowsSeen) {
        wp->setEmptyMessage(QString("\"%1\" is not applicable to this data (e.g. an "
                                     "accuracy-based metric on a regression-only pool)")
                                 .arg(nnWilcoxonMetricCombo_ ? nnWilcoxonMetricCombo_->currentText() : metricCol));
      } else {
        wp->setEmptyMessage(QString()); // reset to the default "No Wilcoxon data"
      }
      wp->clear();
    }
    nnRankingsTable_->setRowCount(0);
    if (nnWilcoxonSummaryLbl_) {
      if (rowsSeen > 0 && rowsSkippedForMetric == rowsSeen) {
        // Every row in nn_summary.csv exists, but none of them have a
        // usable value for THIS metric -- e.g. "Generalization gap [%]"/
        // "Validation accuracy [%]" on a regression-only pool, where
        // accuracy is undefined by construction (see the comment above),
        // rather than an actual absence of data.
        nnWilcoxonSummaryLbl_->setText(
          QString("No usable \"%1\" values in nn_summary.csv (%2 row(s) present, but this metric is "
                   "not applicable to any of them -- e.g. an accuracy-based metric on a regression-only "
                   "pool). Pick a different metric above, such as \"Generalization gap (loss)\".")
              .arg(nnWilcoxonMetricCombo_ ? nnWilcoxonMetricCombo_->currentText() : metricCol)
              .arg(rowsSeen));
      } else {
        nnWilcoxonSummaryLbl_->setText("No runs yet -- run \"1. Train + sample landscape\" first.");
      }
    }
    return;
  }

  // agg[problem][method] = mean metric value across that (problem,method)'s runs
  QMap<QString, QMap<QString, double>> agg;
  QSet<QString> methodSet;
  for (auto pit = raw.constBegin(); pit != raw.constEnd(); ++pit) {
    for (auto mit = pit.value().constBegin(); mit != pit.value().constEnd(); ++mit) {
      double s = 0.0; for (double v : mit.value()) s += v;
      agg[pit.key()][mit.key()] = s / mit.value().size();
      methodSet.insert(mit.key());
    }
  }
  QStringList methods = methodSet.values();
  methods.sort(Qt::CaseInsensitive);
  const bool lowerBetter = nnMetricLowerIsBetter(metricCol);

  // ---- Wilcoxon box-plot groups: per method, its per-problem aggregated values ----
  QVector<WilcoxonBoxPlotWidget::Group> groups;
  QStringList problems = agg.keys();
  for (const QString& m : methods) {
    WilcoxonBoxPlotWidget::Group g;
    g.label = m;
    for (const QString& p : problems) {
      if (agg[p].contains(m)) g.values.push_back(agg[p][m]);
    }
    groups.push_back(g);
  }

  // ---- Pairwise Wilcoxon (only for method pairs sharing >=6 common problems) ----
  QVector<WilcoxonBoxPlotWidget::Annotation> anns;
  int nSig = 0, nTested = 0;
  for (int i = 0; i < methods.size(); ++i) {
    for (int j = i + 1; j < methods.size(); ++j) {
      QVector<double> a, b;
      for (const QString& p : problems) {
        if (agg[p].contains(methods[i]) && agg[p].contains(methods[j])) {
          a.push_back(agg[p][methods[i]]);
          b.push_back(agg[p][methods[j]]);
        }
      }
      if (a.size() < 6) continue;
      const double pval = wilcoxonSignedRankP(a, b);
      if (std::isnan(pval)) continue;
      ++nTested;
      const bool sig = pval < 0.05;
      if (sig) ++nSig;
      WilcoxonBoxPlotWidget::Annotation ann;
      ann.i = i; ann.j = j;
      ann.significant = sig;
      ann.text = sig ? QString("p=%1 *").arg(pval, 0, 'g', 2) : QString("p=%1").arg(pval, 0, 'g', 2);
      anns.push_back(ann);
    }
  }

  if (wp) {
    wp->setData(groups, anns,
                QString("%1 across %2 problem(s) (dataset x architecture)")
                    .arg(nnWilcoxonMetricCombo_ ? nnWilcoxonMetricCombo_->currentText() : metricCol)
                    .arg(problems.size()),
                lowerBetter ? "lower is better" : "higher is better",
                nnMetricAxisLabel(metricCol));
  }
  if (nnWilcoxonSummaryLbl_) {
    nnWilcoxonSummaryLbl_->setText(QString(
        "%1 optimizer(s) x %2 problem(s). %3 pair(s) tested (>=6 shared problems), "
        "%4 significant at alpha=0.05.").arg(methods.size()).arg(problems.size()).arg(nTested).arg(nSig));
  }

  // ---- Rankings: Friedman-style average rank per method across problems ----
  QMap<QString, double> rankSum;
  QMap<QString, int> rankCount;
  for (const QString& p : problems) {
    QVector<QPair<QString,double>> here;
    for (const QString& m : methods)
      if (agg[p].contains(m)) here.push_back({m, agg[p][m]});
    if (here.isEmpty()) continue;
    std::sort(here.begin(), here.end(), [&](const QPair<QString,double>& x, const QPair<QString,double>& y) {
      return lowerBetter ? x.second < y.second : x.second > y.second;
    });
    for (int r = 0; r < here.size(); ++r) {
      rankSum[here[r].first] += (r + 1);
      rankCount[here[r].first] += 1;
    }
  }

  QVector<QPair<QString,double>> ranked; // method, mean rank
  for (const QString& m : methods) {
    if (rankCount.value(m, 0) > 0) ranked.push_back({m, rankSum[m] / rankCount[m]});
  }
  std::sort(ranked.begin(), ranked.end(), [](const QPair<QString,double>& a, const QPair<QString,double>& b) {
    return a.second < b.second;
  });

  nnRankingsTable_->setRowCount(ranked.size());
  for (int r = 0; r < ranked.size(); ++r) {
    const QString& m = ranked[r].first;
    double sum = 0.0; int cnt = 0;
    for (const QString& p : problems) if (agg[p].contains(m)) { sum += agg[p][m]; ++cnt; }
    const double meanMetric = cnt > 0 ? sum / cnt : std::numeric_limits<double>::quiet_NaN();

    nnRankingsTable_->setItem(r, 0, new QTableWidgetItem(QString::number(r + 1)));
    nnRankingsTable_->setItem(r, 1, new QTableWidgetItem(m));
    nnRankingsTable_->setItem(r, 2, new QTableWidgetItem(QString::number(ranked[r].second, 'f', 2)));
    nnRankingsTable_->setItem(r, 3, new QTableWidgetItem(QString::number(meanMetric, 'g', 6)));
    nnRankingsTable_->setItem(r, 4, new QTableWidgetItem(QString::number(cnt)));
  }
  nnRankingsTable_->resizeColumnsToContents();
}

} // namespace optimsolution_gui
