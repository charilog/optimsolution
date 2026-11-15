// MainWindow_BatchStats.cpp
// Batch execution and post-processing, statistics tables, rankings, Wilcoxon/Friedman analysis, plot widgets and XLSX export.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"

namespace optimsolution_gui {
void MainWindow::startBatch() {
  if (batchActive_) {
    return;
  }
  if (!batchMethodsList_ || !batchProblemsList_) {
    return;
  }

  const QStringList methods = selectedBatchMethodShortNames();
  const QStringList problems = selectedBatchProblemShortNames();

  if (methods.isEmpty() || problems.isEmpty()) {
    QMessageBox::warning(this, "Batch run", "Select at least one method and one problem.");
    return;
  }

  // Pin the dimension of every problem of this batch as an explicit entry of
  // the Dims table (e.g. sphere -> "30"), exactly like "Load experiment CSV"
  // already does. Without this, a problem without its own Dims entry follows
  // the (hidden in Batch mode) default-dimension spin box, which is shared
  // with Single run: changing it later renamed "sphere [d=30]" to e.g.
  // "sphere [d=6]", so the finished results vanished from the Output tabs and
  // "Run batch" ran the whole experiment again. Same dimensions as before;
  // only newly added problems keep following the default dimension.
  // All of the problem's current dimensions are pinned (a restricted-
  // dimension problem such as CEC2017 defaults to several, e.g. 10, 30, 50;
  // pinning only the first one made the other dimensions' results vanish).
  for (const QString& problemKey : problems) {
    const QString base = batchBaseProblemShort(problemKey);
    if (base.isEmpty() || fixedDimForProblem(base) > 0) continue;
    if (batchProblemDimOverride_.contains(base)) continue;   // already explicit
    const QVector<int> dims = batchDimsForProblem(base, dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
    QStringList parts;
    for (int d : dims) if (d > 1) parts << QString::number(d);
    if (!parts.isEmpty()) batchProblemDimOverride_.insert(base, parts);
  }

  // Reload settings from disk to pick up any external edits (e.g. runs value).
  if (cfg_ && !settingsPath_.isEmpty()) cfg_->load(settingsPath_);

  // Runs are taken from [global].runs (mirrored read-only in the batch panel).
  const int runs = globalRunsFromSettings();

  // Build only the missing/incompatible jobs. Existing cached cells are reused.
  batchQueue_.clear();
  batchJobIndex_ = 0;
  batchTotalJobs_ = 0;
  batchStopRequested_ = false;
  batchCurMethodShort_.clear();
  batchCurProblemShort_.clear();
  batchCurRunIndex_ = 0;
  batchCurSeed_ = 0;

  const int userDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
  int reusedJobs = 0;

  for (const QString& problemKey : problems) {
    const QString prob = batchBaseProblemShort(problemKey);
    const int d = batchDimForProblem(problemKey, userDim);
    for (const QString& meth : methods) {
      if (batchHasCachedCell(problemKey, meth, d)) {
        ++reusedJobs;
        continue;
      }
      BatchJob job;
      job.methodShort = meth;
      job.problemShort = prob;
      job.problemKey = problemKey;
      job.dim = d;
      job.startMsecsUtc = 0;
      batchQueue_.push_back(job);
    }
  }

  batchTotalJobs_ = batchQueue_.size();

  // Ensure the batch panel is visible and the summary reflects the current selection immediately.
  updateBatchPanelVisibility();

  int summaryIdx = -1;
  QWidget* summaryPage = nullptr;
  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 && active->page && outputTabs_) {
      summaryPage = active->page;
      summaryIdx = outputTabs_->currentIndex();
    }
  }
  if (!summaryPage) {
    summaryIdx = createOutputRunTab("System", QString(), 0);
    if (summaryIdx >= 0 && outputTabs_) {
      outputTabs_->setCurrentIndex(summaryIdx);
      activeOutputRunIndex_ = summaryIdx;
      if (const auto* out = outputTab(summaryIdx)) {
        summaryPage = out->page;
      }
    }
  }

  // Rows shown while (and after) this batch runs. Reordered below to the
  // experiment's own row order when an existing batch summary is extended.
  QStringList shownProblems = problems;

  if (summaryPage) {
    g_activeBatchSummaryPage = summaryPage;
    bindBatchSummaryUiForPage(summaryPage);
    const BatchSummarySnapshot prevSnap = g_batchSummarySnapshots.value(summaryPage);
    BatchSummarySnapshot snap;
    // keep the experiment manifest of this page (a loaded experiment that is
    // extended with more methods/problems keeps writing to its own manifest)
    snap.manifestPath = prevSnap.manifestPath;
    // The experiment keeps every method/problem it already had, including the
    // ones currently hidden in the Selection area (ablation study), plus the
    // newly selected ones. Only the selected ones are shown.
    snap.allMethods = batchOrderedUnion(batchExperimentMethods(prevSnap), methods);
    // Methods are always shown alphabetically (same as selectedBatchMethodShortNames()
    // and "Load experiment CSV"), also after the experiment is extended.
    std::sort(snap.allMethods.begin(), snap.allMethods.end(), [](const QString& a, const QString& b) {
      return a.toLower() < b.toLower();
    });
    snap.allProblems = batchOrderedUnion(batchExperimentProblems(prevSnap), problems);
    shownProblems = batchOrderLike(problems, snap.allProblems);
    snap.methods = methods;
    snap.problems = shownProblems;
    snap.csvPaths = g_liveBatchCsvPaths;
    snap.cachedProblemDims = batchCachedProblemDims_;
    snap.problemDimOverrides = batchProblemDimOverride_;
    snap.metricComboIndex = (batchMetricCombo_ ? batchMetricCombo_->currentIndex() : -1);
    snap.aggText = (batchAggCombo_ ? batchAggCombo_->currentText() : QString("Mean"));
    snap.showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                      ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                      : true);
    snap.showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
    snap.showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
      snap.showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
    g_batchSummarySnapshots.insert(summaryPage, snap);
    batchSummaryCellsByPage_[summaryPage] = batchCells_;
  }

  initStatsTableForBatch(methods, shownProblems);
  for (const QString& prob : shownProblems) {
    for (const QString& meth : methods) {
      updateStatsTableCell(prob, meth);
    }
  }

  // Keep statsBatchTableActive_ enabled while there are additional jobs pending,
  // otherwise updateStatsTableCell() would early-return during the batch and the
  // Results table would appear to fill only at the very end.
  if (batchTotalJobs_ <= 0) {
    finalizeStatsTableAfterBatch();
  } else if (statsTable_) {
    statsTable_->resizeColumnsToContents();
  }

  // Force the initial empty/cached state of the Results table to become visible
  // before the first additional batch job starts, so the user can watch cells fill
  // incrementally during execution.
  if (statsTable_) {
    statsTable_->viewport()->update();
    statsTable_->viewport()->repaint();
    qApp->processEvents(QEventLoop::ExcludeUserInputEvents, 25);
  }

  if (batchTotalJobs_ <= 0) {
    if (batchProgress_) {
      batchProgress_->setRange(0, 1);
      batchProgress_->setValue(1);
    }
    if (outputProgressBar_) {
      outputProgressBar_->setRange(0, 1);
      outputProgressBar_->setValue(1);
      outputProgressBar_->setFormat("Up to date");
      progressWidget_->setVisible(true);
      QTimer::singleShot(1500, this, [this](){
        if (outputProgressBar_) progressWidget_->setVisible(false);
      });
    }
    if (outputStatusLbl_) {
      outputStatusLbl_->setText("No additional jobs.");
    }
    if (g_activeBatchSummaryPage) {
      bindBatchSummaryUiForPage(g_activeBatchSummaryPage);
    }
    refreshBatchSelectionView();
    appendLog(QString("Batch is up to date. Reused %1 cached job(s); no additional jobs were required.")
                .arg(reusedJobs));
    return;
  }

  batchActive_ = true;
  startBatchLogWriter();
  batchMetrics_.clear(); // legacy

  // Initialize progress UI.
  if (batchProgress_) {
    batchProgress_->setRange(0, batchTotalJobs_);
    batchProgress_->setValue(0);
  }
  if (outputStatusLbl_) {
    outputStatusLbl_->setText(QString("Batch: %1 additional job(s), %2 cached")
                               .arg(batchTotalJobs_)
                               .arg(reusedJobs));
  }

  // Lock the UI for the whole batch.
  updateRunBtn(true);
  if (methodBox_) methodBox_->setEnabled(false);
  if (problemBox_) problemBox_->setEnabled(false);
  if (dimSpin_) dimSpin_->setEnabled(false);
  if (tabs_) tabs_->setEnabled(false);
  if (refreshBtn_) refreshBtn_->setEnabled(false);
  if (runModeBox_) runModeBox_->setEnabled(false);
  if (batchPanel_) batchPanel_->setEnabled(false);
  if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(false);
  if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(false);
  if (clearCsvBtn_) clearCsvBtn_->setEnabled(false);
  if (saveBatchSelectionBtn_) saveBatchSelectionBtn_->setEnabled(false);
  if (loadBatchSelectionBtn_) loadBatchSelectionBtn_->setEnabled(false);
  if (selectCfgBtn_) selectCfgBtn_->setEnabled(false);
  if (reloadCfgBtn_) reloadCfgBtn_->setEnabled(false);
  if (saveCfgBtn_) saveCfgBtn_->setEnabled(false);
  updateStatsTabsEnabled();

  QTimer::singleShot(0, this, &MainWindow::startNextBatchJob);
}

void MainWindow::startNextBatchJob() {
  if (!batchActive_) {
    return;
  }
  if (batchStopRequested_) {
    finalizeBatch();
    return;
  }
  if (batchJobIndex_ < 0 || batchJobIndex_ >= batchQueue_.size()) {
    finalizeBatch();
    return;
  }

  BatchJob& job = batchQueue_[batchJobIndex_];
  job.startMsecsUtc = QDateTime::currentMSecsSinceEpoch();

  const bool isFixed = (fixedDimForProblem(job.problemShort) > 0);
  int dim = job.dim;
  if (dim <= 0 && dimSpin_) {
    dim = dimSpin_->value();
  }
  if (dim <= 0) {
    const QString dimKey = job.problemKey.trimmed().isEmpty() ? job.problemShort : job.problemKey;
    dim = batchDimForProblem(dimKey, dim);
  }
  job.dim = dim;

  batchCurMethodShort_ = job.methodShort;
  batchCurProblemShort_ = job.problemShort;

  if (job.problemKey.trimmed().isEmpty()) {
    job.problemKey = batchProblemDisplayKey(job.problemShort, dim);
  }

  if (outputStatusLbl_) {
    outputStatusLbl_->setText(QString("Running: %1 on %2").arg(job.methodShort, job.problemKey));
  }

  // CLI uses positional args: <method> <problem> [dim]
  QStringList args;
  args << job.methodShort;
  args << job.problemShort;
  if (!isFixed && dim > 0) {
    args << QString::number(dim);
  }

  lastArgs_ = args;
  lastHadDimension_ = !isFixed;

  appendLog(QString("Batch job %1/%2: method=%3, problem=%4")
              .arg(batchJobIndex_ + 1)
              .arg(batchTotalJobs_)
              .arg(job.methodShort)
              .arg(job.problemKey.isEmpty() ? job.problemShort : job.problemKey));

  processTextBuffer_.clear();
  currentRunNumber_ = 1;
  startCliProcess(args);
}


void MainWindow::startBatchPostProcessAsync(const BatchJob& job, int exitCode, const QString& runtimeWorkingDir) {
  if (!batchActive_) {
    return;
  }
  if (batchPostInFlight_) {
    return;
  }
  batchPostInFlight_ = true;

  if (outputStatusLbl_) {
    outputStatusLbl_->setText(QString("Post-processing: %1 on %2").arg(job.methodShort, job.problemShort));
  }
  if (outputProgressBar_) {
    progressWidget_->setVisible(true);
  }

  if (batchPostWatcher_) {
    batchPostWatcher_->disconnect(this);
    batchPostWatcher_->deleteLater();
    batchPostWatcher_ = nullptr;
  }

  batchPostWatcher_ = new QFutureWatcher<BatchPostResult>(this);

  connect(batchPostWatcher_, &QFutureWatcher<BatchPostResult>::finished, this, [this, exitCode]() {
    if (!batchPostWatcher_) {
      batchPostInFlight_ = false;
      return;
    }
    const BatchPostResult result = batchPostWatcher_->result();
    batchPostWatcher_->deleteLater();
    batchPostWatcher_ = nullptr;
    batchPostInFlight_ = false;
    finishBatchPostProcess(exitCode, result);
  });

  const QString workingDir = runtimeWorkingDir;
  const QString meth = job.methodShort;
  const QString prob = job.problemShort;
  const int dim = job.dim;
  const qint64 startMsecsUtc = job.startMsecsUtc;
  const QString problemKey = job.problemKey;
  const double timePerRunSecs = job.timePerRunSecs;

  const QFuture<BatchPostResult> future = QtConcurrent::run([this, workingDir, meth, prob, problemKey, dim, startMsecsUtc, timePerRunSecs, exitCode]() -> BatchPostResult {
    BatchPostResult r;
    r.methodShort = meth;
    r.problemShort = prob;
    r.problemKey = problemKey;
    r.dim = dim;
    r.startMsecsUtc = startMsecsUtc;

    if (exitCode != 0) {
      r.ok = false;
      r.message = QString("Batch: job failed for %1/%2 (exit code %3).")
                    .arg(meth, prob)
                    .arg(exitCode);
      return r;
    }

    QString err;
    const QString csvPath = findBestConvergenceCsvForJob(workingDir, meth, prob, dim, startMsecsUtc, &err);
    r.csvPath = csvPath;

    BatchCellData cell;
    if (csvPath.isEmpty()) {
      r.cell = cell;
      r.ok = false;
      r.message = QString("Batch: convergence CSV not found for %1/%2: %3")
                    .arg(meth, prob, err);
      return r;
    }

    QString perr;
    if (loadBatchCellFromConvergence(csvPath, cell, &perr)) {
      // Attach timing info from CLI summary output (if available).
      if (std::isfinite(timePerRunSecs) && timePerRunSecs > 0.0)
        cell.meanTimePerRun = timePerRunSecs;
      r.cell = cell;
      r.ok = true;
      r.message = QString("Batch: parsed convergence CSV: %1").arg(QFileInfo(csvPath).fileName());
    } else {
      r.cell = cell;
      r.ok = false;
      r.message = QString("Batch: failed to parse convergence CSV for %1/%2: %3")
                    .arg(meth, prob, perr);
    }
    return r;
  });

  batchPostWatcher_->setFuture(future);
}

void MainWindow::finishBatchPostProcess(int /*exitCode*/, const BatchPostResult& result) {
  if (!batchActive_) {
    return;
  }

  const QString problemKey = result.problemKey.trimmed().isEmpty()
                               ? batchProblemDisplayKey(result.problemShort, result.dim)
                               : result.problemKey.trimmed();

  // Persist the metrics for the completed job (derived from convergence CSV).
  batchCells_[problemKey][result.methodShort] = result.cell;
  if (result.dim > 0) {
    batchCachedProblemDims_[problemKey] = result.dim;
  }
  if (!result.csvPath.trimmed().isEmpty()) {
    g_liveBatchCsvPaths[problemKey][result.methodShort] = result.csvPath.trimmed();
  }
  if (!result.message.isEmpty()) {
    appendLog(result.message);
  }

  // Record the experiment's exact cells in its manifest (*.osexp) so that
  // "Load experiment CSV" later restores THIS experiment only and never
  // picks up unrelated CSV files of the same folder (single runs, other
  // batches). See writeBatchManifest() in MainWindowShared.h.
  if (g_activeBatchSummaryPage && !result.csvPath.trimmed().isEmpty()) {
    BatchSummarySnapshot snap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    QString mpath = snap.manifestPath;
    if (mpath.isEmpty() || !QFileInfo(mpath).absoluteDir().exists()) {
      mpath = QDir(QFileInfo(result.csvPath).absolutePath()).filePath(
          QString("batch_experiment_%1.%2")
              .arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), batchManifestFileSuffix()));
    }
    // Use the FULL experiment (also the methods/problems that are currently
    // hidden in the Selection area for an ablation study), otherwise running
    // more jobs while something is hidden would silently drop the hidden
    // cells from the manifest.
    QStringList probs = batchExperimentProblems(snap);
    if (probs.isEmpty()) probs = g_liveBatchCsvPaths.keys();
    QList<BatchManifestEntry> entries;
    for (const QString& pk : probs) {
      if (!g_liveBatchCsvPaths.contains(pk)) continue;
      const QMap<QString, QString>& byMethod = g_liveBatchCsvPaths[pk];
      QStringList meths = batchExperimentMethods(snap);
      if (meths.isEmpty()) meths = byMethod.keys();
      for (const QString& m : meths) {
        const QString csv = byMethod.value(m).trimmed();
        if (csv.isEmpty()) continue;
        BatchManifestEntry e;
        e.method  = m;
        e.problem = batchBaseProblemShort(pk);
        e.dim     = batchCachedProblemDims_.value(pk, parseDimFromConvergenceCsvFilename(csv));
        e.csvPath = QFileInfo(csv).absoluteFilePath();
        if (e.dim > 0 && !e.problem.isEmpty()) entries.push_back(e);
      }
    }
    QString merr;
    if (!entries.isEmpty()) {
      if (writeBatchManifest(mpath, entries, &merr)) {
        snap.manifestPath = mpath;
        g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, snap);
      } else {
        appendLog(QString("Batch: %1").arg(merr));
      }
    }
  }

  // v41: Incremental refresh (single cell) to keep the GUI responsive for large batches.
  if (!problemKey.isEmpty() && !result.methodShort.isEmpty()) {
    if (g_activeBatchSummaryPage) {
      bindBatchSummaryUiForPage(g_activeBatchSummaryPage);
    }
    updateStatsTableCell(problemKey, result.methodShort);

    // Paint the newly computed cell immediately before the next job starts.
    if (statsTable_) {
      statsTable_->viewport()->update();
      statsTable_->viewport()->repaint();
      qApp->processEvents(QEventLoop::ExcludeUserInputEvents, 25);
    }
  }

  ++batchJobIndex_;

  int prog = batchJobIndex_;
  if (prog < 0) prog = 0;
  if (prog > batchTotalJobs_) prog = batchTotalJobs_;

  if (batchProgress_) {
    batchProgress_->setRange(0, batchTotalJobs_);
    batchProgress_->setValue(prog);
  }
  if (outputProgressBar_) {
    const int pct = (batchTotalJobs_ > 0)
                    ? qBound(0, int(double(prog) / batchTotalJobs_ * 100.0), 99) : 0;
    outputProgressBar_->setRange(0, 100);
    outputProgressBar_->setValue(pct);
    outputProgressBar_->setFormat(QString("Job %1/%2 — %3%").arg(prog).arg(batchTotalJobs_).arg(pct));
  }

  if (batchStopRequested_ || batchJobIndex_ >= batchTotalJobs_) {
    finalizeBatch();
    return;
  }

  // Keep the UI locked during batch.
  updateRunBtn(true);
  if (methodBox_) methodBox_->setEnabled(false);
  if (problemBox_) problemBox_->setEnabled(false);
  if (dimSpin_) dimSpin_->setEnabled(false);
  if (tabs_) tabs_->setEnabled(false);
  if (refreshBtn_) refreshBtn_->setEnabled(false);
  if (runModeBox_) runModeBox_->setEnabled(false);
  if (batchPanel_) batchPanel_->setEnabled(false);
  if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(false);
  if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(false);
  if (clearCsvBtn_) clearCsvBtn_->setEnabled(false);
  if (saveBatchSelectionBtn_) saveBatchSelectionBtn_->setEnabled(false);
  if (loadBatchSelectionBtn_) loadBatchSelectionBtn_->setEnabled(false);
  if (selectCfgBtn_) selectCfgBtn_->setEnabled(false);
  if (reloadCfgBtn_) reloadCfgBtn_->setEnabled(false);
  if (saveCfgBtn_) saveCfgBtn_->setEnabled(false);

  QTimer::singleShot(0, this, &MainWindow::startNextBatchJob);
}

void MainWindow::finalizeBatch() {
  if (!batchActive_) {
    return;
  }

  stopBatchLogWriter();

  batchActive_ = false;

  // Clean up the temporary per-run config snapshot written into
  // optimsolution_gui_run (created because the CLI has no --config
  // argument; see startCliProcess()). The on-disk root optimsolution.cfg
  // is never touched here — only the runtime copy inside that folder.
  if (!lastRuntimeCfgPath_.isEmpty()) {
    const QFileInfo cfgFi(lastRuntimeCfgPath_);
    if (cfgFi.exists() && cfgFi.dir().dirName().compare("optimsolution_gui_run", Qt::CaseInsensitive) == 0) {
      QFile::remove(cfgFi.absoluteFilePath());
    }
  }

  // Fully restore the end-of-batch UI state.
  if (busySpinner_) busySpinner_->stop();

  if (outputProgressBar_) {
    outputProgressBar_->setRange(0, 100);
    outputProgressBar_->setValue(100);
    outputProgressBar_->setFormat("Finished");
    QTimer::singleShot(1500, this, [this](){
      if (outputProgressBar_) progressWidget_->setVisible(false);
    });
  }

  if (outputStatusLbl_) {
    if (runElapsedTimer_) runElapsedTimer_->stop();
    if (outputElapsedLbl_) outputElapsedLbl_->setVisible(false);
    const qint64 ms = runElapsed_.elapsed();
    const int h = int(ms / 3600000);
    const int m = int((ms % 3600000) / 60000);
    const int s = int((ms % 60000) / 1000);
    const QString elapsed = h > 0
      ? QString("%1:%2:%3").arg(h).arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0'))
      : QString("%1:%2").arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0'));
    outputStatusLbl_->setText(QString("Completed. Total time: %1").arg(elapsed));
  }

  // Restore UI.
  if (selectionBox_) selectionBox_->setEnabled(true);
  if (settingsBox_) settingsBox_->setEnabled(true);
  if (methodBox_) methodBox_->setEnabled(true);
  if (problemBox_) problemBox_->setEnabled(true);
  if (dimSpin_) dimSpin_->setEnabled(true);
  if (tabs_) tabs_->setEnabled(true);
  if (refreshBtn_) refreshBtn_->setEnabled(true);
  if (runModeBox_) runModeBox_->setEnabled(true);
  if (batchPanel_) batchPanel_->setEnabled(true);
  if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(true);
  if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(true);
  if (clearCsvBtn_) clearCsvBtn_->setEnabled(true);
  if (saveBatchSelectionBtn_) saveBatchSelectionBtn_->setEnabled(true);
  if (loadBatchSelectionBtn_) loadBatchSelectionBtn_->setEnabled(true);
  if (selectCfgBtn_) selectCfgBtn_->setEnabled(true);
  if (reloadCfgBtn_) reloadCfgBtn_->setEnabled(true);
  if (saveCfgBtn_) saveCfgBtn_->setEnabled(true);
  if (auto* c = findChild<QComboBox*>("batchMethodSettingsCombo")) c->setEnabled(true);
  updateStatsTabsEnabled();

  // NOTE: Do NOT call updateDimUiForProblem(currentProblemShort()) here.
  // currentProblemShort() reads the Single-Run "Problem" combo box, which is
  // completely unrelated to the batch's own problem selection. If that
  // combo happened to be sitting on a fixed-dimension problem, this used to
  // silently overwrite dimSpin_'s value (e.g. from 500 down to that
  // problem's fixed dim) right before refreshBatchSelectionView() below
  // reconstructs the batch problem keys from dimSpin_. That mismatch made
  // the just-computed results "disappear" from the Results table whenever
  // there was no per-problem dimension override recorded (most easily
  // reproduced with a single selected batch problem) -- the data was still
  // safely on disk/in batchCells_, so "Load experiment CSV" always could
  // still show it. The Selection-area dim spinbox for Batch mode is kept in
  // sync separately via batchDimForProblem()/batchProblemDimOverride_, so
  // this call is unnecessary in Batch mode and only ever caused harm here.
  if (!(runModeBox_ && runModeBox_->currentData().toInt() == 1)) {
    updateDimUiForProblem(currentProblemShort());
  }

  updateBatchPanelVisibility();
  if (g_activeBatchSummaryPage) {
    bindBatchSummaryUiForPage(g_activeBatchSummaryPage);
  }

  // FIX 7: Commit the live batch results into the per-page cache AND into the
  // snapshot BEFORE calling refreshBatchSelectionView().  That function
  // unconditionally restores batchCells_ from batchSummaryCellsByPage_ and
  // g_liveBatchCsvPaths from the snapshot — both of which still held the
  // values saved at batch START (typically empty).  Without this commit every
  // result accumulated during the run is silently discarded the moment the
  // batch finishes and the table is wiped clean.
  if (g_activeBatchSummaryPage) {
    batchSummaryCellsByPage_[g_activeBatchSummaryPage] = batchCells_;
    if (g_batchSummarySnapshots.contains(g_activeBatchSummaryPage)) {
      BatchSummarySnapshot liveSnap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
      liveSnap.csvPaths            = g_liveBatchCsvPaths;
      liveSnap.cachedProblemDims   = batchCachedProblemDims_;
      liveSnap.problemDimOverrides = batchProblemDimOverride_;
      g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, liveSnap);
    }
  }

  refreshBatchSelectionView();
  finalizeStatsTableAfterBatch();

  if (g_activeBatchSummaryPage) {
    BatchSummarySnapshot snap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    if (snap.methods.isEmpty()) snap.methods = effectiveStatsMethodShortNames();
    if (snap.problems.isEmpty()) snap.problems = effectiveStatsProblemShortNames();
    snap.csvPaths = g_liveBatchCsvPaths;
    snap.cachedProblemDims = batchCachedProblemDims_;
    snap.problemDimOverrides = batchProblemDimOverride_;
    snap.metricComboIndex = (batchMetricCombo_ ? batchMetricCombo_->currentIndex() : -1);
    snap.aggText = (batchAggCombo_ ? batchAggCombo_->currentText() : QString("Mean"));
    snap.showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                      ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                      : true);
    snap.showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
    snap.showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
      snap.showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
    g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, snap);
    batchSummaryCellsByPage_[g_activeBatchSummaryPage] = batchCells_;
  }

  appendLog(QString("Batch finished. Completed %1/%2 jobs.")
              .arg(std::min(batchJobIndex_, batchTotalJobs_))
              .arg(batchTotalJobs_));

  // U-score ("trial-based dominance") comparison report -- additive only;
  // never touches batchCells_/statsTable_/anything above. See
  // exportUScoreReport() and include/uscore.h for details.
  try {
    exportUScoreReport();
  } catch (const std::exception& e) {
    appendLog(QString("U-score report: skipped (%1).").arg(e.what()));
  } catch (...) {
    appendLog("U-score report: skipped (unknown error).");
  }
}

double MainWindow::extractBatchMetric(const QString& text, bool* ok) const {
  if (ok) {
    *ok = false;
  }

  // Try to extract a "Function calls" / "Function evaluations" style metric from CLI output.
  const QRegularExpression re1(
    R"((Function\s*(?:Evaluations|calls)\s*[:=]\s*)([0-9]+(?:\.[0-9]+)?))",
    QRegularExpression::CaseInsensitiveOption
  );
  double last = 0.0;
  bool found = false;
  auto it1 = re1.globalMatch(text);
  while (it1.hasNext()) {
    const auto m = it1.next();
    const QString num = m.captured(2);
    bool okNum = false;
    const double v = num.toDouble(&okNum);
    if (okNum) {
      last = v;
      found = true;
    }
  }

  if (!found) {
    const QRegularExpression re2(
      R"(\bFE(?:s)?\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?))",
      QRegularExpression::CaseInsensitiveOption
    );
    auto it2 = re2.globalMatch(text);
    while (it2.hasNext()) {
      const auto m = it2.next();
      const QString num = m.captured(1);
      bool okNum = false;
      const double v = num.toDouble(&okNum);
      if (okNum) {
        last = v;
        found = true;
      }
    }
  }

  if (found && ok) {
    *ok = true;
  }
  return last;
}

MainWindow::BatchMetricMode MainWindow::currentBatchMetricMode() const {
  if (!batchMetricCombo_) {
    return BatchMetricMode::BestFinalBestF;
  }
  bool ok = false;
  const int v = batchMetricCombo_->currentData().toInt(&ok);
  if (ok) {
    return static_cast<BatchMetricMode>(v);
  }
  return static_cast<BatchMetricMode>(batchMetricCombo_->currentIndex());
}

void MainWindow::onBatchMetricUiChanged() {
  if (batchActive_) {
    return;
  }
  const BatchMetricMode mode = currentBatchMetricMode();

  // Aggregation is only relevant for time-to-best metrics.
  const bool aggEnabled = (mode == BatchMetricMode::IterationAtBest || mode == BatchMetricMode::EvalsAtBest);
  if (batchAggCombo_) {
    batchAggCombo_->setEnabled(aggEnabled);
  }
  // Also keep the secondary (Friedman panel) agg combo in sync.
  if (g_activeBatchSummaryPage && g_batchSummaryUi.contains(g_activeBatchSummaryPage)) {
    if (auto* aggCombo2 = g_batchSummaryUi.value(g_activeBatchSummaryPage).aggCombo2)
      aggCombo2->setEnabled(aggEnabled);
  }

  if (batchCells_.isEmpty()) {
    finalizeStatsTableAfterBatch();
    if (const auto* active = activeOutputTab()) {
      if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 && active->page && g_batchSummarySnapshots.contains(active->page)) {
        BatchSummarySnapshot snap = g_batchSummarySnapshots.value(active->page);
        snap.metricComboIndex = (batchMetricCombo_ ? batchMetricCombo_->currentIndex() : -1);
        snap.aggText = (batchAggCombo_ ? batchAggCombo_->currentText() : QString("Mean"));
            snap.cachedProblemDims = batchCachedProblemDims_;
        snap.problemDimOverrides = batchProblemDimOverride_;
        snap.showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                          ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                          : true);
        snap.showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
        snap.showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
      snap.showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
        g_batchSummarySnapshots.insert(active->page, snap);
        batchSummaryCellsByPage_[active->page] = batchCells_;
      }
    }
    return;
  }

  rebuildStatsTable();

  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 && active->page) {
      BatchSummarySnapshot snap = g_batchSummarySnapshots.value(active->page);
      snap.metricComboIndex = (batchMetricCombo_ ? batchMetricCombo_->currentIndex() : -1);
      snap.aggText = (batchAggCombo_ ? batchAggCombo_->currentText() : QString("Mean"));
      snap.showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                        ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                        : true);
      snap.showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
      snap.showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
      snap.showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
      g_batchSummarySnapshots.insert(active->page, snap);
      batchSummaryCellsByPage_[active->page] = batchCells_;
    }
  }
}

QString MainWindow::batchMetricTitle() const {
  const BatchMetricMode mode = currentBatchMetricMode();
  switch (mode) {
    case BatchMetricMode::BestFinalBestF: return "Best (final best_f)";
    case BatchMetricMode::MeanFinalBestF: return "Mean (final best_f)";
    case BatchMetricMode::IterationAtBest: return "Iterations at best";
    case BatchMetricMode::EvalsAtBest: return "Function evaluations at best";
  }
  return "Metric";
}

QString MainWindow::batchMetricAxisLabel() const {
  const BatchMetricMode mode = currentBatchMetricMode();
  switch (mode) {
    case BatchMetricMode::BestFinalBestF:
    case BatchMetricMode::MeanFinalBestF:
      return "best_f";
    case BatchMetricMode::IterationAtBest:
      return "Iterations";
    case BatchMetricMode::EvalsAtBest:
      return "Function evaluations";
  }
  return "Value";
}

QString MainWindow::batchMetricYAxisLabel() const {
  // Used by the Wilcoxon box plot. Capitalized-first-word title with the
  // unit in square brackets, no underscores -- same convention as mode 7's
  // own Wilcoxon tab (nnMetricAxisLabel() in MainWindow_NNLandscape.cpp).
  const BatchMetricMode mode = currentBatchMetricMode();
  const QString agg = (batchAggCombo_ ? batchAggCombo_->currentText() : QString());
  if (mode == BatchMetricMode::IterationAtBest) {
    QString base = "Iterations to best";
    if (!agg.isEmpty()) base += QString(" (%1)").arg(agg);
    return base;
  }
  if (mode == BatchMetricMode::EvalsAtBest) {
    QString base = "Function evaluations to best";
    if (!agg.isEmpty()) base += QString(" (%1)").arg(agg);
    return base;
  }
  return "Best objective value";
}

QString MainWindow::batchMetricUnitSuffix() const {
  const BatchMetricMode mode = currentBatchMetricMode();
  switch (mode) {
    case BatchMetricMode::BestFinalBestF:
    case BatchMetricMode::MeanFinalBestF:
      return "";
    case BatchMetricMode::IterationAtBest:
      return "iters";
    case BatchMetricMode::EvalsAtBest:
      return "FEs";
  }
  return "";
}

int MainWindow::globalRunsFromSettings() const {
  // Read directly from the cfg file on disk for the freshest value,
  // bypassing any cached state in cfg_.
  const QString path = settingsPath_.isEmpty()
                       ? PathUtils::findExistingFileUpwards(QDir::currentPath(), "optimsolution.cfg")
                       : settingsPath_;
  if (!path.isEmpty()) {
    QFile f(path);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
      bool inGlobal = false;
      QTextStream ts(&f);
      while (!ts.atEnd()) {
        const QString line = ts.readLine().trimmed();
        if (line.startsWith('[')) {
          inGlobal = (line.toLower() == "[global]");
          continue;
        }
        if (!inGlobal) continue;
        if (line.startsWith(';') || line.startsWith('#')) continue;
        const int eq = line.indexOf('=');
        if (eq < 0) continue;
        const QString key = line.left(eq).trimmed().toLower();
        if (key == "runs") {
          const QString val = line.mid(eq + 1).section(';', 0, 0).trimmed();
          bool ok = false;
          const int v = val.toInt(&ok);
          if (ok && v > 0) return v;
        }
      }
    }
  }
  // Fall back to cfg_ object if file read failed.
  if (!cfg_) return 30;
  bool ok = false;
  int runs = cfg_->value("global", "runs", "30").toInt(&ok);
  if (!ok || runs <= 0) runs = 30;
  return runs;
}

QStringList MainWindow::selectedBatchMethodShortNames() const {
  QStringList out;
  if (!batchMethodsList_) {
    return out;
  }
  const QList<QListWidgetItem*> items = batchMethodsList_->selectedItems();
  for (QListWidgetItem* it : items) {
    QString key = it->data(Qt::UserRole).toString().trimmed();
    if (key.contains(" - ")) {
      key = key.section(" - ", 0, 0).trimmed();
    }
    if (key.isEmpty()) {
      key = it->text().section(" - ", 0, 0).trimmed();
    }
    if (!key.isEmpty()) {
      out.push_back(key);
    }
  }
  // QListWidget::selectedItems() does not guarantee any particular order
  // (it follows internal selection-range bookkeeping, not list position),
  // so manual picks in the Selection/Methods list or a ".batchsel" loaded
  // via "Load batch selection" could come back in an arbitrary order.
  // Sort alphabetically here so the ordering always matches what
  // "Load experiment CSV" already produces.
  std::sort(out.begin(), out.end(), [](const QString& a, const QString& b) {
    return a.toLower() < b.toLower();
  });
  return out;
}

QStringList MainWindow::selectedBatchProblemShortNames() const {
  QStringList out;
  if (!batchProblemsList_) {
    return out;
  }

  const int defaultDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
  const QList<QListWidgetItem*> items = batchProblemsList_->selectedItems();
  for (QListWidgetItem* it : items) {
    QString key = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
    if (key.isEmpty()) {
      key = batchBaseProblemShort(it->text().trimmed());
    }
    if (key.isEmpty()) {
      continue;
    }

    const QVector<int> dims = batchDimsForProblem(key, defaultDim);
    for (int dim : dims) {
      const QString problemKey = batchProblemDisplayKey(key, dim);
      if (!problemKey.isEmpty()) {
        out.push_back(problemKey);
      }
    }
  }
  return out;
}

QStringList MainWindow::effectiveStatsMethodShortNames() const {
  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 &&
        active->page && g_batchSummarySnapshots.contains(active->page)) {
      const BatchSummarySnapshot snap = g_batchSummarySnapshots.value(active->page);
      if (!snap.methods.isEmpty()) {
        return snap.methods;
      }
    }
  }

  QStringList methods = selectedBatchMethodShortNames();
  if (!methods.isEmpty()) {
    return methods;
  }

  if (!statsBatchMethods_.isEmpty()) {
    return statsBatchMethods_;
  }

  QSet<QString> ms;
  for (auto pit = batchCells_.begin(); pit != batchCells_.end(); ++pit) {
    for (auto mit = pit.value().begin(); mit != pit.value().end(); ++mit) {
      ms.insert(mit.key());
    }
  }

  methods = QStringList(ms.begin(), ms.end());
  std::sort(methods.begin(), methods.end(), [](const QString& a, const QString& b) {
    return a.toLower() < b.toLower();
  });
  return methods;
}

QStringList MainWindow::effectiveStatsProblemShortNames() const {
  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 &&
        active->page && g_batchSummarySnapshots.contains(active->page)) {
      const BatchSummarySnapshot snap = g_batchSummarySnapshots.value(active->page);
      if (!snap.problems.isEmpty()) {
        return snap.problems;
      }
    }
  }

  QStringList problems = selectedBatchProblemShortNames();
  if (!problems.isEmpty()) {
    return problems;
  }

  if (!statsBatchProblems_.isEmpty()) {
    return statsBatchProblems_;
  }

  problems = batchCells_.keys();
  std::sort(problems.begin(), problems.end(), [this](const QString& a, const QString& b) {
    const QString baseA = batchBaseProblemShort(a).toLower();
    const QString baseB = batchBaseProblemShort(b).toLower();
    if (baseA != baseB) return baseA < baseB;
    return batchDimForProblem(a, 2) < batchDimForProblem(b, 2);
  });
  return problems;
}

QString MainWindow::findBestConvergenceCsvForJob(const QString& workingDir,
                                               const QString& methodShort,
                                               const QString& problemShort,
                                               int dim,
                                               qint64 newerThanMsecsUtc,
                                               QString* err) const {
  if (err) err->clear();

  const QString m = methodShort.toLower();
  const QString p = problemShort.toLower();
  const QString dTok = (dim > 0) ? QString("d%1").arg(dim) : QString();

  const QStringList dirs = QStringList()
      << (workingDir.isEmpty() ? QString() : (workingDir + "/csv"))
      << workingDir
      << (workingDir.isEmpty() ? QString() : (workingDir + "/output"))
      << (workingDir.isEmpty() ? QString() : (workingDir + "/results"));

  QFileInfo bestFi;
  int bestScore = -1;

  for (const QString& dirPath : dirs) {
    if (dirPath.isEmpty()) continue;
    QDir dir(dirPath);
    if (!dir.exists()) continue;

    // Use name filters to avoid scanning the full directory when many CSV files exist.
    QStringList nameFilters;
    if (!m.isEmpty() && !p.isEmpty() && !dTok.isEmpty()) {
      nameFilters << QString("*%1*%2*%3*convergence.csv").arg(m, p, dTok)
                  << QString("*%2*%1*%3*convergence.csv").arg(m, p, dTok)
                  << QString("*%1*%2*convergence.csv").arg(m, p)
                  << QString("*%2*%1*convergence.csv").arg(m, p);
    } else if (!m.isEmpty() && !p.isEmpty()) {
      nameFilters << QString("*%1*%2*convergence.csv").arg(m, p)
                  << QString("*%2*%1*convergence.csv").arg(m, p);
    } else if (!m.isEmpty()) {
      nameFilters << QString("*%1*convergence.csv").arg(m);
    } else if (!p.isEmpty()) {
      nameFilters << QString("*%1*convergence.csv").arg(p);
    } else {
      nameFilters << "*convergence.csv";
    }

    QDirIterator it(dirPath, nameFilters, QDir::Files, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
      const QString path = it.next();
      const QFileInfo fi(path);
      const QString name = fi.fileName().toLower();
      const qint64 msec = fi.lastModified().toMSecsSinceEpoch();

      int score = 0;
      if (name.contains("convergence")) score += 1;
      if (!m.isEmpty() && name.contains(m)) score += 3;
      if (!p.isEmpty() && name.contains(p)) score += 3;
      if (!dTok.isEmpty() && name.contains(dTok)) score += 2;
      if (newerThanMsecsUtc > 0 && msec >= (newerThanMsecsUtc - 1000)) score += 1;

      if (score > bestScore) {
        bestScore = score;
        bestFi = fi;
      } else if (score == bestScore && bestFi.exists() && msec > bestFi.lastModified().toMSecsSinceEpoch()) {
        bestFi = fi;
      }
    }
  }

  if (!bestFi.exists()) {
    if (err) {
      *err = QString("No convergence CSV found in '%1'.").arg(workingDir);
    }
    return QString();
  }
  return bestFi.absoluteFilePath();
}

static inline bool isFinite(double v) {
  return std::isfinite(v);
}

double MainWindow::aggregateValues(const QVector<double>& vals, const QString& agg) const {
  QVector<double> x;
  x.reserve(vals.size());
  for (double v : vals) {
    if (isFinite(v)) x.push_back(v);
  }
  if (x.isEmpty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (agg.compare("Min", Qt::CaseInsensitive) == 0) {
    double mn = x[0];
    for (double v : x) mn = std::min(mn, v);
    return mn;
  }
  if (agg.compare("Max", Qt::CaseInsensitive) == 0) {
    double mx = x[0];
    for (double v : x) mx = std::max(mx, v);
    return mx;
  }
  if (agg.compare("Median", Qt::CaseInsensitive) == 0) {
    std::sort(x.begin(), x.end());
    const int n = x.size();
    if ((n % 2) == 1) {
      return x[n / 2];
    }
    return 0.5 * (x[n / 2 - 1] + x[n / 2]);
  }

  // Mean (default)
  double sum = 0.0;
  for (double v : x) sum += v;
  return sum / double(x.size());
}

double MainWindow::stdevValues(const QVector<double>& vals) const {
  QVector<double> x;
  x.reserve(vals.size());
  for (double v : vals) {
    if (isFinite(v)) x.push_back(v);
  }
  const int n = x.size();
  if (n <= 1) {
    return 0.0;
  }
  double mean = 0.0;
  for (double v : x) mean += v;
  mean /= double(n);

  double s2 = 0.0;
  for (double v : x) {
    const double d = v - mean;
    s2 += d * d;
  }
  s2 /= double(n - 1); // sample SD
  return std::sqrt(s2);
}

bool MainWindow::loadBatchCellFromConvergence(const QString& csvPath, BatchCellData& out, QString* err) const {
  if (err) err->clear();
  out = BatchCellData{};

  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly)) {
    if (err) *err = QString("Cannot open file '%1'.").arg(csvPath);
    return false;
  }

  // Fast path: read the file once and parse the bytes in place. The former
  // QTextStream + QString::split() per line allocated several QStrings per
  // row and parsed only ~30 MB/s, so loading an experiment with hundreds of
  // multi-MB convergence CSVs froze the GUI for minutes. The accepted input
  // is the same as before: same header columns, '\n' or "\r\n" line ends,
  // optional UTF-8 BOM, blank lines skipped, rows with too few columns or
  // non-numeric fields skipped, same numeric parsing (C locale, surrounding
  // whitespace ignored).
  const QByteArray data = f.readAll();
  f.close();
  const qsizetype dataSize = data.size();
  qsizetype pos = 0;
  if (dataSize >= 3 && uchar(data[0]) == 0xEF && uchar(data[1]) == 0xBB && uchar(data[2]) == 0xBF) {
    pos = 3; // UTF-8 BOM (QTextStream used to skip it as well)
  }
  auto nextLine = [&data, dataSize, &pos](QByteArrayView& line) -> bool {
    if (pos >= dataSize) return false;
    qsizetype e = data.indexOf('\n', pos);
    if (e < 0) e = dataSize;
    line = QByteArrayView(data.constData() + pos, e - pos);
    if (!line.isEmpty() && line.back() == '\r') line.chop(1);
    pos = e + 1;
    return true;
  };

  QByteArrayView headerView;
  if (!nextLine(headerView) || headerView.isEmpty()) {
    if (err) *err = "Empty CSV file.";
    return false;
  }
  const QString headerLine = QString::fromUtf8(headerView);

  const QStringList header = headerLine.split(',', Qt::KeepEmptyParts);
  auto colIndex = [&](const QString& name) -> int {
    for (int i = 0; i < header.size(); ++i) {
      if (header[i].trimmed().compare(name, Qt::CaseInsensitive) == 0) {
        return i;
      }
    }
    return -1;
  };

  const int cRun = colIndex("run");
  const int cIter = colIndex("iter");
  const int cEvals = colIndex("evals");
  const int cBest = colIndex("best_f");

  if (cRun < 0 || cIter < 0 || cEvals < 0 || cBest < 0) {
    if (err) *err = "CSV header missing one of: run, iter, evals, best_f.";
    return false;
  }

  struct Rec { int iter; qint64 evals; double best; };
  QMap<int, QVector<Rec>> byRun;

  const int maxCol = std::max(std::max(cRun, cIter), std::max(cEvals, cBest));
  QByteArrayView line;
  while (nextLine(line)) {
    line = line.trimmed();
    if (line.isEmpty()) continue;

    // Split on ',' only as far as the last needed column.
    QByteArrayView fRun, fIter, fEvals, fBest;
    int col = 0;
    qsizetype fieldStart = 0;
    bool haveAll = false;
    const qsizetype len = line.size();
    for (qsizetype i = 0; i <= len; ++i) {
      if (i < len && line[i] != ',') continue;
      const QByteArrayView field = line.sliced(fieldStart, i - fieldStart).trimmed();
      if (col == cRun) fRun = field;
      if (col == cIter) fIter = field;
      if (col == cEvals) fEvals = field;
      if (col == cBest) fBest = field;
      if (col == maxCol) { haveAll = true; break; }
      ++col;
      fieldStart = i + 1;
    }
    if (!haveAll) continue; // fewer columns than needed

    bool okRun = false, okIter = false, okEval = false, okBest = false;
    const int run = fRun.toInt(&okRun);
    const int itv = fIter.toInt(&okIter);
    const qint64 ev = fEvals.toLongLong(&okEval);
    const double bf = fBest.toDouble(&okBest);
    if (!okRun || !okIter || !okEval || !okBest) continue;

    byRun[run].push_back(Rec{itv, ev, bf});
  }

  if (byRun.isEmpty()) {
    if (err) *err = "No data rows parsed.";
    return false;
  }

  // Final best_f per run and global best.
  double globalBest = std::numeric_limits<double>::quiet_NaN();
  for (auto it = byRun.begin(); it != byRun.end(); ++it) {
    const QVector<Rec>& seq = it.value();
    if (seq.isEmpty()) continue;
    const double lastBest = seq.back().best;
    out.finalBestF.push_back(lastBest);
    if (!isFinite(globalBest) || lastBest < globalBest) {
      globalBest = lastBest;
    }
  }

  out.nRuns = out.finalBestF.size();
  out.bestOverall = globalBest;
  if (!isFinite(globalBest)) {
    if (err) *err = "Global best_f is NaN.";
    return false;
  }

  out.tol = std::max(1e-12, 1e-9 * std::abs(globalBest));

  // Time-to-best per run (first time best_f reaches the global best within tolerance).
  out.hitIter.reserve(byRun.size());
  out.hitEvals.reserve(byRun.size());

  int success = 0;
  for (auto it = byRun.begin(); it != byRun.end(); ++it) {
    const QVector<Rec>& seq = it.value();
    double hitI = std::numeric_limits<double>::quiet_NaN();
    double hitE = std::numeric_limits<double>::quiet_NaN();

    for (const Rec& r : seq) {
      if (r.best <= (globalBest + out.tol)) {
        hitI = double(r.iter);
        hitE = double(r.evals);
        break;
      }
    }

    out.hitIter.push_back(hitI);
    out.hitEvals.push_back(hitE);

    const double lastBest = seq.isEmpty() ? std::numeric_limits<double>::quiet_NaN() : seq.back().best;
    if (isFinite(lastBest) && std::abs(lastBest - globalBest) <= out.tol) {
      ++success;
    }
  }

  out.nSuccess = success;
  out.ratePct = (out.nRuns > 0) ? (100.0 * double(out.nSuccess) / double(out.nRuns)) : 0.0;

  // Try to read meanTimePerRun from the corresponding summary CSV.
  // The summary file has the same base name with _summary.csv suffix.
  const QString summaryPath = QString(csvPath).replace(
      QRegularExpression("_convergence\\.csv$", QRegularExpression::CaseInsensitiveOption),
      "_summary.csv");
  if (QFileInfo::exists(summaryPath)) {
    QFile sf(summaryPath);
    if (sf.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream sts(&sf);
      const QString shdr = sts.readLine();
      const QStringList scols = shdr.split(',', Qt::KeepEmptyParts);
      // Find iter_mean_ms_mean column (ms per iteration → convert to s per run).
      // Also accept mean_time, time_per_run, iter_mean_ms columns.
      int tcol = -1;
      double msPerIter = std::numeric_limits<double>::quiet_NaN();
      double msPerRun  = std::numeric_limits<double>::quiet_NaN();
      for (int i = 0; i < scols.size(); ++i) {
        const QString c = scols[i].trimmed().toLower();
        if (c == "iter_mean_ms_mean")   { tcol = i; break; }
        if (c == "mean_time" || c == "time_per_run" || c == "time") { tcol = i; break; }
      }
      if (!sts.atEnd()) {
        const QStringList srow = sts.readLine().split(',', Qt::KeepEmptyParts);
        if (tcol >= 0 && tcol < srow.size()) {
          bool ok = false;
          const double val = srow[tcol].trimmed().toDouble(&ok);
          if (ok && val > 0.0) msPerIter = val;
        }
        // Also try to find runs and max_evals for ms→s conversion.
        int maxEvalsCol = -1, runsCol = -1;
        for (int i = 0; i < scols.size(); ++i) {
          const QString c = scols[i].trimmed().toLower();
          if (c == "max_evals") maxEvalsCol = i;
          if (c == "runs") runsCol = i;
        }
        // iter_mean_ms_mean is ms per iteration; total run time = msPerIter × max_iters
        // But simpler: look for a direct time column in seconds.
        // iter_mean_ms_mean × (max_evals/population) gives approx time/run in ms.
        // Best approach: read max_evals and compute.
        // Actually iter_mean_ms_mean * (number of iters per run) = total ms per run.
        // We don't have iters here directly, but we can use mean_evals / population.
        // Simplest: find mean_time_s or similar direct column.
        int directTimeCol = -1;
        for (int i = 0; i < scols.size(); ++i) {
          const QString c = scols[i].trimmed().toLower();
          if (c == "mean_time_s" || c == "time_per_run_s" ||
              c == "wall_time_mean" || c == "elapsed_mean_s") {
            directTimeCol = i;
            break;
          }
        }
        if (directTimeCol >= 0 && directTimeCol < srow.size()) {
          bool ok = false;
          const double val = srow[directTimeCol].trimmed().toDouble(&ok);
          if (ok && val > 0.0) msPerRun = val * 1000.0; // already in seconds
        }
        // Use iter_mean_ms_mean × mean_iters_per_run if we have max_evals and population.
        if (!std::isfinite(msPerRun) && std::isfinite(msPerIter)) {
          // find mean_evals or max_evals column as denominator proxy
          int meanEvalsCol = -1, popCol = -1, maxItCol = -1;
          for (int i = 0; i < scols.size(); ++i) {
            const QString c = scols[i].trimmed().toLower();
            if (c == "mean_evals" || c == "evals_mean") meanEvalsCol = i;
            if (c == "population" || c == "pop" || c == "pop_size") popCol = i;
            if (c == "max_iters") maxItCol = i;
          }
          double meanEvals = std::numeric_limits<double>::quiet_NaN();
          double pop       = std::numeric_limits<double>::quiet_NaN();
          double maxIters  = std::numeric_limits<double>::quiet_NaN();
          if (meanEvalsCol >= 0 && meanEvalsCol < srow.size()) {
            bool ok = false; meanEvals = srow[meanEvalsCol].trimmed().toDouble(&ok); if (!ok) meanEvals = std::numeric_limits<double>::quiet_NaN();
          }
          if (popCol >= 0 && popCol < srow.size()) {
            bool ok = false; pop = srow[popCol].trimmed().toDouble(&ok); if (!ok) pop = std::numeric_limits<double>::quiet_NaN();
          }
          if (maxItCol >= 0 && maxItCol < srow.size()) {
            bool ok = false; maxIters = srow[maxItCol].trimmed().toDouble(&ok); if (!ok) maxIters = std::numeric_limits<double>::quiet_NaN();
          }
          // iters per run ≈ meanEvals / pop
          if (std::isfinite(meanEvals) && std::isfinite(pop) && pop > 0)
            msPerRun = msPerIter * (meanEvals / pop);
          else if (std::isfinite(maxIters))
            msPerRun = msPerIter * maxIters;
        }
        if (std::isfinite(msPerRun) && msPerRun > 0.0)
          out.meanTimePerRun = msPerRun / 1000.0;  // convert ms → s
      }
    }
  }

  return true;
}



void MainWindow::initStatsTableForBatch(const QStringList& methods, const QStringList& problems) {
  if (!statsTable_ || !exportStatsBtn_) {
    return;
  }

  statsBatchTableActive_ = true;
  statsBatchMethods_ = methods;
  statsBatchProblems_ = problems;
  statsRowByProblem_.clear();
  statsColBaseByMethod_.clear();

  const bool statsShowMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                              ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                              : true);
  statsShowRate_ = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
  statsShowSd_   = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
  statsShowTime_ = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);
  statsTable_->setProperty("statsShowMean", statsShowMean);

  statsMode_ = currentBatchMetricMode();
  statsAgg_ = (batchAggCombo_ ? batchAggCombo_->currentText() : "Mean");

  statsPerMethodCols_ = 1 + (statsShowMean ? 1 : 0) + (statsShowRate_ ? 1 : 0) + (statsShowSd_ ? 1 : 0) + (statsShowTime_ ? 1 : 0);
  const int cols = 2 + statsBatchMethods_.size() * statsPerMethodCols_;

  statsTable_->setUpdatesEnabled(false);

  statsTable_->clear();
  statsTable_->setRowCount(statsBatchProblems_.size());
  statsTable_->setColumnCount(cols);

  QStringList headers;
  headers << "PROBLEM" << "DIM";
  for (const QString& m : statsBatchMethods_) {
    headers << QString("%1 (Value)").arg(m);
    if (statsShowMean) headers << QString("%1 (Mean)").arg(m);
    if (statsShowRate_) headers << QString("%1 (Rate%)").arg(m);
    if (statsShowSd_) headers << QString("%1 (SD)").arg(m);
    if (statsShowTime_) headers << QString("%1 (Time/run s)").arg(m);
  }
  statsTable_->setHorizontalHeaderLabels(headers);

  auto mkItem = [](const QString& t) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    return it;
  };

  const int defaultDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);

  for (int r = 0; r < statsBatchProblems_.size(); ++r) {
    const QString prob = statsBatchProblems_[r];
    statsRowByProblem_[prob] = r;
    statsTable_->setItem(r, 0, mkItem(prob));
    const int dim = batchCachedProblemDims_.contains(prob)
                      ? batchCachedProblemDims_.value(prob)
                      : batchDimForProblem(prob, defaultDim);
    statsTable_->setItem(r, 1, mkItem(QString::number(dim)));
  }

  int c = 2;
  for (const QString& meth : statsBatchMethods_) {
    statsColBaseByMethod_[meth] = c;
    c += statsPerMethodCols_;
  }

  // Initialize all metric cells as empty and let them fill incrementally as each
  // batch job is post-processed. Cached cells, if any, will be painted explicitly
  // by the caller via updateStatsTableCell().
  for (int r = 0; r < statsBatchProblems_.size(); ++r) {
    for (const QString& meth : statsBatchMethods_) {
      const int colBase = statsColBaseByMethod_.value(meth, -1);
      if (colBase < 0) continue;
      int cc = colBase;
      statsTable_->setItem(r, cc++, mkItem(QString()));
      if (statsShowMean) statsTable_->setItem(r, cc++, mkItem(QString()));
      if (statsShowRate_) statsTable_->setItem(r, cc++, mkItem(QString()));
      if (statsShowSd_) statsTable_->setItem(r, cc++, mkItem(QString()));
      if (statsShowTime_) statsTable_->setItem(r, cc++, mkItem(QString()));
    }
  }

  // Avoid expensive auto-resize during batch; do it once at the end.
  if (statsTable_->horizontalHeader()) {
    statsTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
  }

  exportStatsBtn_->setEnabled(!batchCells_.isEmpty());
  if (generateReportBtn_) generateReportBtn_->setEnabled(!batchCells_.isEmpty());
  if (exportRankPlotBtn_) exportRankPlotBtn_->setEnabled(!batchCells_.isEmpty());

  statsTable_->setUpdatesEnabled(true);
}

void MainWindow::updateStatsTableCell(const QString& problemShort, const QString& methodShort) {
  if (!statsTable_ || !statsBatchTableActive_) {
    return;
  }
  if (!statsRowByProblem_.contains(problemShort) || !statsColBaseByMethod_.contains(methodShort)) {
    return;
  }

  const int row = statsRowByProblem_.value(problemShort);
  const int colBase = statsColBaseByMethod_.value(methodShort);

  if (statsTable_) {
    const int dim = batchCachedProblemDims_.contains(problemShort)
                      ? batchCachedProblemDims_.value(problemShort)
                      : batchDimForProblem(problemShort, dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
    if (auto* dimItem = statsTable_->item(row, 1)) {
      dimItem->setText(QString::number(dim));
    }
  }

  // Resolve current mode/agg from cached batch settings.
  const BatchMetricMode mode = statsMode_;
  const QString agg = statsAgg_.isEmpty() ? (batchAggCombo_ ? batchAggCombo_->currentText() : "Mean") : statsAgg_;

  const bool statsShowMean = statsTable_->property("statsShowMean").toBool();

  double value = std::numeric_limits<double>::quiet_NaN();
  double mean = std::numeric_limits<double>::quiet_NaN();
  double sd = std::numeric_limits<double>::quiet_NaN();
  double rate = 0.0;
  double timePerRun = std::numeric_limits<double>::quiet_NaN();
  bool hasComputedCell = false;

  if (batchCells_.contains(problemShort) && batchCells_[problemShort].contains(methodShort)) {
    const BatchCellData& cell = batchCells_[problemShort][methodShort];
    hasComputedCell = (cell.nRuns > 0)
                      || !cell.finalBestF.isEmpty()
                      || !cell.hitIter.isEmpty()
                      || !cell.hitEvals.isEmpty();
    if (hasComputedCell) {
      rate = cell.ratePct;
      mean = aggregateValues(cell.finalBestF, "Mean");
      timePerRun = cell.meanTimePerRun;

      if (mode == BatchMetricMode::BestFinalBestF) {
        value = aggregateValues(cell.finalBestF, "Min");
        sd = stdevValues(cell.finalBestF);
      } else if (mode == BatchMetricMode::MeanFinalBestF) {
        value = aggregateValues(cell.finalBestF, "Mean");
        sd = stdevValues(cell.finalBestF);
      } else if (mode == BatchMetricMode::IterationAtBest) {
        value = aggregateValues(cell.hitIter, agg);
        sd = stdevValues(cell.hitIter);
      } else if (mode == BatchMetricMode::EvalsAtBest) {
        value = aggregateValues(cell.hitEvals, agg);
        sd = stdevValues(cell.hitEvals);
      }
    }
  }

  const bool intLike = (mode == BatchMetricMode::IterationAtBest || mode == BatchMetricMode::EvalsAtBest);

  auto fmt = [&](double v) -> QString {
    if (!std::isfinite(v)) return QString();
    if (intLike) return QString::number(qint64(std::llround(v)));
    return QString::number(v, 'g', 15);
  };
  auto fmtSd = [&](double v) -> QString {
    if (!std::isfinite(v)) return QString();
    if (intLike) return QString::number(v, 'f', 2);
    return QString::number(v, 'g', 10);
  };

  auto mkItem = [](const QString& t) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    return it;
  };
  auto mkItemColored = [](const QString& t, const QColor& color) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    it->setForeground(QBrush(color));
    return it;
  };
  // Keep in sync with rebuildStatsTable()'s per-column colors.
  static const QColor kValueColor(0xff, 0xff, 0xff);   // white
  static const QColor kMeanColor (0x8f, 0xe3, 0xff);   // pale cyan
  static const QColor kRateColor (0xb6, 0xf5, 0x8a);   // pale green
  static const QColor kSdColor   (0xff, 0xe0, 0x8a);   // pale gold
  static const QColor kTimeColor (0xe3, 0xc4, 0xff);   // pale lavender

  const QString valueText = hasComputedCell ? fmt(value) : QString();
  const QString meanText = (hasComputedCell && std::isfinite(mean)) ? QString::number(mean, 'g', 15) : QString();
  const QString rateText = hasComputedCell ? QString::number(int(std::llround(rate))) : QString();
  const QString sdText = hasComputedCell ? fmtSd(sd) : QString();
  const QString timeText = (hasComputedCell && std::isfinite(timePerRun))
                             ? QString::number(timePerRun, 'f', 3)
                             : QString();

  statsTable_->setItem(row, colBase, mkItemColored(valueText, kValueColor));

  int c = colBase + 1;
  if (statsShowMean) {
    statsTable_->setItem(row, c++, mkItemColored(meanText, kMeanColor));
  }
  if (statsShowRate_) {
    statsTable_->setItem(row, c++, mkItemColored(rateText, kRateColor));
  }
  if (statsShowSd_) {
    statsTable_->setItem(row, c++, mkItemColored(sdText, kSdColor));
  }
  if (statsShowTime_) {
    statsTable_->setItem(row, c++, mkItemColored(timeText, kTimeColor));
  }

  exportStatsBtn_->setEnabled(!batchCells_.isEmpty());
  if (generateReportBtn_) generateReportBtn_->setEnabled(!batchCells_.isEmpty());
  if (exportRankPlotBtn_) exportRankPlotBtn_->setEnabled(!batchCells_.isEmpty());
}

void MainWindow::finalizeStatsTableAfterBatch() {
  // Called once, after the batch completes.
  statsBatchTableActive_ = false;

  if (statsTable_) {
    statsTable_->setUpdatesEnabled(false);
    statsTable_->resizeColumnsToContents();
    statsTable_->setUpdatesEnabled(true);
  }

  rebuildStatsComparisons();
}

void MainWindow::rebuildStatsTable() {
  if (!statsTable_ || !exportStatsBtn_) {
    return;
  }

  const bool showMean = (batchPanel_ && batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")
                         ? batchPanel_->findChild<QCheckBox*>("batchShowMeanChk")->isChecked()
                         : true);
  const bool showRate = (batchShowRateChk_ ? batchShowRateChk_->isChecked() : true);
  const bool showSd = (batchShowSdChk_ ? batchShowSdChk_->isChecked() : true);
  const bool showTime = (batchShowTimeChk_ ? batchShowTimeChk_->isChecked() : false);

  const QStringList methods = effectiveStatsMethodShortNames();
  const QStringList problems = effectiveStatsProblemShortNames();

  statsBatchMethods_ = methods;
  statsBatchProblems_ = problems;

  const int perMethodCols = 1 + (showMean ? 1 : 0) + (showRate ? 1 : 0) + (showSd ? 1 : 0) + (showTime ? 1 : 0);
  const int cols = 2 + methods.size() * perMethodCols;

  // FIX 6: rebuildStatsTable() rebuilds the physical table but previously forgot
  // to update statsRowByProblem_ and statsColBaseByMethod_.  Those maps are
  // normally set only by initStatsTableForBatch().  After a tab switch,
  // bindBatchSummaryUiForPage() replaces statsTable_ with a different widget
  // that may have a completely different row/column layout; any subsequent call
  // to updateStatsTableCell() would then write into wrong cells or do nothing.
  // NOTE: do NOT set statsBatchTableActive_ here — rebuildStatsTable() is
  // called from non-batch contexts and must not interfere with the flag that
  // guards incremental cell updates during a live batch run.
  statsRowByProblem_.clear();
  statsColBaseByMethod_.clear();
  for (int r = 0; r < problems.size(); ++r)
    statsRowByProblem_[problems[r]] = r;
  {
    int c = 2;
    for (const QString& meth : methods) {
      statsColBaseByMethod_[meth] = c;
      c += perMethodCols;
    }
  }
  statsShowRate_      = showRate;
  statsShowSd_        = showSd;
  statsShowTime_      = showTime;
  statsPerMethodCols_ = perMethodCols;

  statsTable_->clear();
  statsTable_->setRowCount(problems.size());
  statsTable_->setColumnCount(cols);

  QStringList headers;
  headers << "PROBLEM" << "DIM";
  for (const QString& m : methods) {
    headers << QString("%1 (Value)").arg(m);
    if (showMean) headers << QString("%1 (Mean)").arg(m);
    if (showRate) headers << QString("%1 (Rate%)").arg(m);
    if (showSd) headers << QString("%1 (SD)").arg(m);
    if (showTime) headers << QString("%1 (Time/run s)").arg(m);
  }
  statsTable_->setHorizontalHeaderLabels(headers);

  auto mkItem = [](const QString& t) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    return it;
  };
  auto mkItemColored = [](const QString& t, const QColor& color) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    it->setForeground(QBrush(color));
    return it;
  };
  // Distinct, light colors per metric column so Value/Mean/Rate/SD/Time are
  // visually easy to tell apart at a glance against a dark background.
  static const QColor kValueColor(0xff, 0xff, 0xff);   // white
  static const QColor kMeanColor (0x8f, 0xe3, 0xff);   // pale cyan
  static const QColor kRateColor (0xb6, 0xf5, 0x8a);   // pale green
  static const QColor kSdColor   (0xff, 0xe0, 0x8a);   // pale gold
  static const QColor kTimeColor (0xe3, 0xc4, 0xff);   // pale lavender

  const BatchMetricMode mode = currentBatchMetricMode();
  const QString agg = (batchAggCombo_ ? batchAggCombo_->currentText() : "Mean");
  const int defaultDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);

  for (int r = 0; r < problems.size(); ++r) {
    const QString prob = problems[r];
    statsTable_->setItem(r, 0, mkItem(prob));
    statsTable_->setItem(r, 1, mkItem(QString::number(batchDimForProblem(prob, defaultDim))));

    int c = 2;
    for (const QString& meth : methods) {
      double value = std::numeric_limits<double>::quiet_NaN();
      double mean = std::numeric_limits<double>::quiet_NaN();
      double sd = std::numeric_limits<double>::quiet_NaN();
      double rate = 0.0;
      double timePerRun = std::numeric_limits<double>::quiet_NaN();
      bool hasComputedCell = false;

      if (batchCells_.contains(prob) && batchCells_[prob].contains(meth)) {
        const BatchCellData& cell = batchCells_[prob][meth];
        hasComputedCell = (cell.nRuns > 0)
                          || !cell.finalBestF.isEmpty()
                          || !cell.hitIter.isEmpty()
                          || !cell.hitEvals.isEmpty();
        if (hasComputedCell) {
          rate = cell.ratePct;
          mean = aggregateValues(cell.finalBestF, "Mean");
          timePerRun = cell.meanTimePerRun;

          if (mode == BatchMetricMode::BestFinalBestF) {
            // Value = min(final best_f)
            value = aggregateValues(cell.finalBestF, "Min");
            sd = stdevValues(cell.finalBestF);
          } else if (mode == BatchMetricMode::MeanFinalBestF) {
            // Value = mean(final best_f)
            value = aggregateValues(cell.finalBestF, "Mean");
            sd = stdevValues(cell.finalBestF);
          } else if (mode == BatchMetricMode::IterationAtBest) {
            // Value = aggregation of iteration where global best is first reached (successful runs)
            value = aggregateValues(cell.hitIter, agg);
            sd = stdevValues(cell.hitIter);
          } else if (mode == BatchMetricMode::EvalsAtBest) {
            // Value = aggregation of evals where global best is first reached (successful runs)
            value = aggregateValues(cell.hitEvals, agg);
            sd = stdevValues(cell.hitEvals);
          }
        }
      }

      const bool intLike = (mode == BatchMetricMode::IterationAtBest || mode == BatchMetricMode::EvalsAtBest);

      auto fmt = [&](double v) -> QString {
        if (!std::isfinite(v)) return QString();
        if (intLike) {
          return QString::number(qint64(std::llround(v)));
        }
        return QString::number(v, 'g', 15);
      };
      auto fmtSd = [&](double v) -> QString {
        if (!std::isfinite(v)) return QString();
        if (intLike) {
          return QString::number(v, 'f', 2);
        }
        return QString::number(v, 'g', 10);
      };

      const QString valueText = hasComputedCell ? fmt(value) : QString();
      const QString meanText = (hasComputedCell && std::isfinite(mean)) ? QString::number(mean, 'g', 15) : QString();
      const QString rateText = hasComputedCell ? QString::number(int(std::llround(rate))) : QString();
      const QString sdText = hasComputedCell ? fmtSd(sd) : QString();
      const QString timeText = (hasComputedCell && std::isfinite(timePerRun))
                                 ? QString::number(timePerRun, 'f', 3)
                                 : QString();

      statsTable_->setItem(r, c++, mkItemColored(valueText, kValueColor));
      if (showMean) {
        statsTable_->setItem(r, c++, mkItemColored(meanText, kMeanColor));
      }
      if (showRate) {
        statsTable_->setItem(r, c++, mkItemColored(rateText, kRateColor));
      }
      if (showSd) {
        statsTable_->setItem(r, c++, mkItemColored(sdText, kSdColor));
      }
      if (showTime) {
        statsTable_->setItem(r, c++, mkItemColored(timeText, kTimeColor));
      }
    }
  }

  statsTable_->resizeColumnsToContents();
  exportStatsBtn_->setEnabled(!batchCells_.isEmpty());
  if (generateReportBtn_) generateReportBtn_->setEnabled(!batchCells_.isEmpty());
  if (exportRankPlotBtn_) {
    exportRankPlotBtn_->setEnabled(!batchCells_.isEmpty());
  }

  rebuildStatsComparisons();
}


namespace {

static double normalCdf(double z) {
  // Standard normal CDF.
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

static double twoSidedNormalP(double z) {
  const double az = std::abs(z);
  const double p = 1.0 - normalCdf(az);
  return std::max(0.0, std::min(1.0, 2.0 * p));
}

static double gammq(double a, double x) {
  // Regularized upper incomplete gamma Q(a,x). Numerical Recipes style implementation.
  if (a <= 0.0 || x < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (x == 0.0) {
    return 1.0;
  }

  constexpr int ITMAX = 200;
  constexpr double EPS = 3e-14;
  constexpr double FPMIN = 1e-300;

  const double gln = std::lgamma(a);

  if (x < a + 1.0) {
    // Series for P(a,x), then return Q = 1-P.
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 1; n <= ITMAX; ++n) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (std::abs(del) < std::abs(sum) * EPS) {
        const double p = sum * std::exp(-x + a * std::log(x) - gln);
        return std::max(0.0, std::min(1.0, 1.0 - p));
      }
    }
    const double p = sum * std::exp(-x + a * std::log(x) - gln);
    return std::max(0.0, std::min(1.0, 1.0 - p));
  }

  // Continued fraction for Q(a,x).
  double b = x + 1.0 - a;
  double c = 1.0 / FPMIN;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= ITMAX; ++i) {
    const double an = -double(i) * (double(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (std::abs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < EPS) break;
  }
  const double q = std::exp(-x + a * std::log(x) - gln) * h;
  return std::max(0.0, std::min(1.0, q));
}

static double chisqSurvival(double chi2, int df) {
  if (df <= 0 || !std::isfinite(chi2) || chi2 < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return gammq(0.5 * double(df), 0.5 * chi2);
}

static QVector<double> ranksForMinimization(const QVector<double>& vals) {
  // Returns ranks in [1..k] where smaller vals are better. Ties receive average ranks.
  const int k = vals.size();
  QVector<double> ranks(k, std::numeric_limits<double>::quiet_NaN());

  QVector<int> idx(k);
  for (int i = 0; i < k; ++i) idx[i] = i;

  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    const double va = vals[a];
    const double vb = vals[b];
    if (!std::isfinite(va) && !std::isfinite(vb)) return a < b;
    if (!std::isfinite(va)) return false;
    if (!std::isfinite(vb)) return true;
    if (va == vb) return a < b;
    return va < vb;
  });

  auto tieEps = [&](double v) {
    return std::max(1e-12, 1e-9 * std::abs(v));
  };

  int pos = 0;
  while (pos < k) {
    const int i0 = pos;
    const double v0 = vals[idx[i0]];
    int i1 = i0;
    while (i1 + 1 < k) {
      const double v1 = vals[idx[i1 + 1]];
      if (!std::isfinite(v0) || !std::isfinite(v1)) break;
      const double eps = std::max(tieEps(v0), tieEps(v1));
      if (std::abs(v1 - v0) > eps) break;
      ++i1;
    }

    const double rankStart = double(i0 + 1);
    const double rankEnd = double(i1 + 1);
    const double rAvg = 0.5 * (rankStart + rankEnd);
    for (int t = i0; t <= i1; ++t) {
      ranks[idx[t]] = rAvg;
    }
    pos = i1 + 1;
  }

  return ranks;
}

struct PairwiseResult {
  QString a;
  QString b;
  double p = std::numeric_limits<double>::quiet_NaN();
  double pAdj = std::numeric_limits<double>::quiet_NaN();
  QString better;
  bool significant = false;
};

static void holmAdjust(QVector<PairwiseResult>& res) {
  const int m = res.size();
  QVector<int> ord(m);
  for (int i = 0; i < m; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int i, int j){
    const double pi = res[i].p;
    const double pj = res[j].p;
    if (!std::isfinite(pi) && !std::isfinite(pj)) return i < j;
    if (!std::isfinite(pi)) return false;
    if (!std::isfinite(pj)) return true;
    if (pi == pj) return i < j;
    return pi < pj;
  });

  double prev = 0.0;
  for (int r = 0; r < m; ++r) {
    const int i = ord[r];
    const double pi = res[i].p;
    double adj = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(pi)) {
      adj = std::min(1.0, double(m - r) * pi);
      if (adj < prev) adj = prev;
      prev = adj;
    }
    res[i].pAdj = adj;
  }
}

// Nemenyi post-hoc critical values q_alpha (Demsar, 2006, Table 5), indexed by
// number of methods k (2..20). CD = q_alpha * sqrt(k(k+1)/(6N)).
static double nemenyiQAlpha(int k, double alpha) {
  static const QVector<double> q05 = {
    /*k=2*/ 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164,
    3.218, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544 // k=20
  };
  static const QVector<double> q01 = {
    /*k=2*/ 2.576, 2.913, 3.113, 3.255, 3.364, 3.454, 3.526, 3.590, 3.646,
    3.696, 3.741, 3.781, 3.818, 3.853, 3.884, 3.914, 3.941, 3.967, 3.992 // k=20
  };
  const QVector<double>& table = (alpha <= 0.01 + 1e-9) ? q01 : q05;
  const int kMax = table.size() + 1; // table[0] corresponds to k=2

  if (k < 2) k = 2;
  if (k <= kMax) {
    return table[k - 2];
  }

  // Beyond the tabulated range, extrapolate using the known asymptotic growth
  // rate of the range of k i.i.d. standard normals (~ sqrt(2*ln k)), calibrated
  // to match the table's last entry so the curve is continuous. This is an
  // approximation flagged to the user via the "approx." note in the UI.
  const double qLast = table.last();
  const double a = qLast / std::sqrt(2.0 * std::log(double(kMax)));
  return a * std::sqrt(2.0 * std::log(double(k)));
}

} // namespace


void MainWindow::rebuildStatsComparisons() {
  if (!rankPlot_ || !statsPairwiseTable_ || !statsSummaryLbl_) {
    return;
  }
  auto* plot = dynamic_cast<RankPlotWidget*>(rankPlot_);
  if (!plot) {
    return;
  }

  // Gather methods/problems.
  const QStringList methods = effectiveStatsMethodShortNames();
  const QStringList problems = effectiveStatsProblemShortNames();

  BatchSummaryUiBundle activeUi;
  bool haveActiveUi = false;
  if (const OutputRunTab* active = outputTab(activeOutputRunIndex_)) {
    if (active->page && g_batchSummaryUi.contains(active->page)) {
      activeUi = g_batchSummaryUi.value(active->page);
      haveActiveUi = true;
    }
  }
  // Fallback: if the currently indexed tab is not the batch summary (e.g. a
  // single-run tab is still selected), resolve the bundle from the known active
  // batch summary page so that Best/Mean/Ranking tables are still populated.
  if (!haveActiveUi && g_activeBatchSummaryPage
      && g_batchSummaryUi.contains(g_activeBatchSummaryPage)) {
    activeUi = g_batchSummaryUi.value(g_activeBatchSummaryPage);
    haveActiveUi = true;
  }

  auto clearDerivedTables = [&]() {
    if (!haveActiveUi) return;
    auto clearOne = [](QTableWidget* t, QPushButton* b) {
      if (t) {
        t->clear();
        t->setRowCount(0);
        t->setColumnCount(0);
      }
      if (b) b->setEnabled(false);
    };
    clearOne(activeUi.bestTable, activeUi.exportBestTableBtn);
    clearOne(activeUi.meanTable, activeUi.exportMeanTableBtn);
    clearOne(activeUi.bestRankingTable, activeUi.exportBestRankingBtn);
    clearOne(activeUi.meanRankingTable, activeUi.exportMeanRankingBtn);
    clearOne(activeUi.finalRankingTable, activeUi.exportFinalRankingBtn);
  };

  const BatchMetricMode mode = currentBatchMetricMode();
  const QString agg = (batchAggCombo_ ? batchAggCombo_->currentText() : "Mean");

  auto cellValue = [&](const QString& prob, const QString& meth) -> double {
    if (!batchCells_.contains(prob) || !batchCells_[prob].contains(meth)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const BatchCellData& cell = batchCells_[prob][meth];
    if (mode == BatchMetricMode::BestFinalBestF) {
      return aggregateValues(cell.finalBestF, "Min");
    }
    if (mode == BatchMetricMode::MeanFinalBestF) {
      return aggregateValues(cell.finalBestF, "Mean");
    }
    if (mode == BatchMetricMode::IterationAtBest) {
      return aggregateValues(cell.hitIter, agg);
    }
    // EvalsAtBest
    return aggregateValues(cell.hitEvals, agg);
  };

  // Keep only problems that have finite values for all methods.
  QStringList usedProblems;
  for (const QString& prob : problems) {
    bool okAll = true;
    for (const QString& meth : methods) {
      const double v = cellValue(prob, meth);
      if (!std::isfinite(v)) { okAll = false; break; }
    }
    if (okAll) usedProblems << prob;
  }

  const int k = methods.size();
  const int N = usedProblems.size();

  if (k < 2 || N < 2) {
    plot->clear();
    statsSummaryLbl_->setText("No statistical comparisons: need at least 2 methods and 2 problems with complete data.");
    statsPairwiseTable_->clear();
    statsPairwiseTable_->setRowCount(0);
    statsPairwiseTable_->setColumnCount(0);

    // Keep Wilcoxon tab consistent.
    if (wilcoxonPlot_) {
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        wp->setData(QStringList(), QVector<QVector<double>>(), QString(), QString(), QVector<WilcoxonBoxPlotWidget::Annotation>());
      }
    }
    if (wilcoxonSummaryLbl_) wilcoxonSummaryLbl_->setText("No Wilcoxon comparisons: need at least 2 methods and 2 problems with complete data.");
    if (exportWilcoxonPlotBtn_) exportWilcoxonPlotBtn_->setEnabled(false);
    if (exportRankPlotBtn_) exportRankPlotBtn_->setEnabled(false);

    // Keep the Critical Difference (Nemenyi) tab consistent.
    if (cdPlot_) {
      if (auto* cp = dynamic_cast<CDDiagramWidget*>(cdPlot_)) cp->clear();
    }
    if (cdSummaryLbl_) cdSummaryLbl_->setText("No Nemenyi comparisons: need at least 2 methods and 2 problems with complete data.");
    if (exportCdPlotBtn_) exportCdPlotBtn_->setEnabled(false);

    clearDerivedTables();
    return;
  }

  // Values matrix (N x k).
  QVector<QVector<double>> V;
  V.reserve(N);
  for (const QString& prob : usedProblems) {
    QVector<double> row;
    row.reserve(k);
    for (const QString& meth : methods) row.push_back(cellValue(prob, meth));
    V.push_back(std::move(row));
  }

  // Ranks per problem.
  QVector<QVector<double>> R;
  R.reserve(N);
  QVector<double> sumRanks(k, 0.0);
  for (int i = 0; i < N; ++i) {
    QVector<double> r = ranksForMinimization(V[i]);
    R.push_back(r);
    for (int j = 0; j < k; ++j) sumRanks[j] += r[j];
  }
  QVector<double> avgRanks(k, 0.0);
for (int j = 0; j < k; ++j) avgRanks[j] = sumRanks[j] / double(N);

// Build rank distributions per method for boxplots.
QVector<QVector<double>> rankDist(k);
for (int j = 0; j < k; ++j) rankDist[j].reserve(N);
for (int i = 0; i < N; ++i) {
  for (int j = 0; j < k; ++j) {
    rankDist[j].push_back(R[i][j]);
  }
}

// Update plot.
const QString title = "Ranks distribution (lower is better)";
const QString subtitle = QString("N=%1 problems, k=%2 methods").arg(N).arg(k);
plot->setData(methods, rankDist, title, subtitle);

// Alpha.
  double alpha = 0.05;
  if (statsAlphaCombo_) {
    bool ok = false;
    const double a = statsAlphaCombo_->currentData().toDouble(&ok);
    if (ok && a > 0.0 && a < 1.0) alpha = a;
  }

  // Friedman omnibus + post-hoc (pairwise based on rank differences; normal approximation) + Holm adjustment.
  double sumR2 = 0.0;
  for (double sr : sumRanks) sumR2 += sr * sr;
  // Standard Friedman statistic: chi2 = 12/(N·k·(k+1)) × Σ Rⱼ² − 3·N·(k+1)
  // where Rⱼ = sum of ranks for method j across all N problems (sumRanks[j]).
  const double chi2 = (12.0 / (double(N) * double(k) * double(k + 1))) * sumR2 - 3.0 * double(N) * double(k + 1);
  const double pOmni = chisqSurvival(chi2, k - 1);

  const double denom = std::sqrt(double(k) * double(k + 1) / (6.0 * double(N)));

  QVector<PairwiseResult> pairs;
  for (int i = 0; i < k; ++i) {
    for (int j = i + 1; j < k; ++j) {
      PairwiseResult pr;
      pr.a = methods[i];
      pr.b = methods[j];
      const double z = (denom > 0.0) ? std::abs(avgRanks[i] - avgRanks[j]) / denom : 0.0;
      pr.p = twoSidedNormalP(z);
      pr.better = (avgRanks[i] < avgRanks[j]) ? pr.a : pr.b;
      pairs.push_back(pr);
    }
  }
  holmAdjust(pairs);
  for (auto& pr : pairs) pr.significant = (std::isfinite(pr.pAdj) && pr.pAdj <= alpha);

  const QString summary = QString("Friedman omnibus: chi^2=%1 (df=%2), p=%3. Post-hoc: pairwise rank-difference tests (normal approx) with Holm correction (alpha=%4).")
    .arg(chi2, 0, 'g', 8)
    .arg(k - 1)
    .arg(std::isfinite(pOmni) ? QString::number(pOmni, 'g', 6) : "-")
    .arg(alpha, 0, 'g', 3);

  statsSummaryLbl_->setText(summary + QString("\nUsed problems: %1/%2 (complete rows).").arg(N).arg(problems.size()));

  // ---- Critical Difference (Nemenyi) diagram: companion visualization to the Friedman test above. ----
  if (cdPlot_) {
    if (auto* cp = dynamic_cast<CDDiagramWidget*>(cdPlot_)) {
      const bool approxTable = (k > 20);
      const double qAlpha = nemenyiQAlpha(k, alpha);
      const double cd = qAlpha * std::sqrt(double(k) * double(k + 1) / (6.0 * double(N)));

      const QString cdTitle = "Nemenyi critical-difference diagram";
      const QString cdSubtitle = QString("alpha=%1, CD=%2, N=%3 problems, k=%4 methods")
        .arg(alpha, 0, 'g', 3)
        .arg(cd, 0, 'f', 3)
        .arg(N)
        .arg(k);

      cp->setData(methods, avgRanks, cd, alpha, N, approxTable, cdTitle, cdSubtitle);

      if (cdSummaryLbl_) {
        QString cdNote = QString("Bars connect groups of methods whose average ranks differ by less than the critical difference "
                                  "(i.e., not significantly different, Nemenyi post-hoc, alpha=%1). CD=%2.")
          .arg(alpha, 0, 'g', 3)
          .arg(cd, 0, 'f', 3);
        if (approxTable) cdNote += " Note: q_alpha for k>20 methods is approximated (no exact tabulated value).";
        cdSummaryLbl_->setText(cdNote);
      }
      if (exportCdPlotBtn_) exportCdPlotBtn_->setEnabled(true);
    }
  }

  // Render the key statistical information inside the plot area (instead of a separate table below).
  // List all pairwise comparisons sorted by raw p-value.
  QString infoText;
  infoText += summary;
  infoText += QString("\nUsed problems: %1/%2 (complete rows).\n\n").arg(N).arg(problems.size());

  QVector<PairwiseResult> pairsByP = pairs;
  std::sort(pairsByP.begin(), pairsByP.end(), [](const PairwiseResult& x, const PairwiseResult& y){
    const double px = x.p;
    const double py = y.p;
    if (!std::isfinite(px) && !std::isfinite(py)) {
      if (x.a == y.a) return x.b.toLower() < y.b.toLower();
      return x.a.toLower() < y.a.toLower();
    }
    if (!std::isfinite(px)) return false;
    if (!std::isfinite(py)) return true;
    if (px == py) {
      if (x.a == y.a) return x.b.toLower() < y.b.toLower();
      return x.a.toLower() < y.a.toLower();
    }
    return px < py;
  });

  auto fmtPInfo = [](double p) -> QString {
    if (!std::isfinite(p)) return "-";
    return QString::number(p, 'g', 6);
  };

  for (const auto& pr : pairsByP) {
    infoText += QString("%1 vs %2: p=%3, p_adj=%4, better=%5, %6\n")
      .arg(pr.a)
      .arg(pr.b)
      .arg(fmtPInfo(pr.p))
      .arg(fmtPInfo(pr.pAdj))
      .arg(pr.better)
      .arg(pr.significant ? "*" : "ns");
  }
  plot->setInfoText(infoText);

  // Pairwise post-hoc annotations for all pairs (Holm-adjusted p-values).
  {
    QVector<RankPlotWidget::Annotation> ann;
    ann.reserve(pairs.size());
    auto fmtPadj = [](double p) -> QString {
      if (!std::isfinite(p)) p = 1.0;
      return QString("p_adj=%1").arg(QString::number(p, 'g', 6));
    };
    for (const auto& pr : pairs) {
      RankPlotWidget::Annotation a;
      a.a = pr.a;
      a.b = pr.b;
      a.text = fmtPadj(pr.pAdj);
      a.significant = pr.significant;
      ann.push_back(std::move(a));
    }
    plot->setAnnotations(std::move(ann));
  }

  // Fill pairwise table (hidden, kept for export; k*(k-1)/2 rows). Its
  // horizontal header is ResizeToContents, the vertical one Interactive
  // (see createOutputRunTab()).
  TableBulkFillGuard pairwiseBulk(statsPairwiseTable_, QHeaderView::ResizeToContents, QHeaderView::Interactive);
  statsPairwiseTable_->clear();
  statsPairwiseTable_->setColumnCount(6);
  statsPairwiseTable_->setRowCount(pairs.size());
  statsPairwiseTable_->setHorizontalHeaderLabels(QStringList() << "Method A" << "Method B" << "p" << "p_adj (Holm)" << "Better" << "Sig");

  auto mk = [](const QString& t) {
    auto* it = new QTableWidgetItem(t);
    it->setFlags(it->flags() & ~Qt::ItemIsEditable);
    return it;
  };
  auto fmtP = [](double p) -> QString {
    if (!std::isfinite(p)) return "-";
    return QString::number(p, 'g', 6);
  };

  for (int r = 0; r < pairs.size(); ++r) {
    const auto& pr = pairs[r];
    statsPairwiseTable_->setItem(r, 0, mk(pr.a));
    statsPairwiseTable_->setItem(r, 1, mk(pr.b));
    statsPairwiseTable_->setItem(r, 2, mk(fmtP(pr.p)));
    statsPairwiseTable_->setItem(r, 3, mk(fmtP(pr.pAdj)));
    statsPairwiseTable_->setItem(r, 4, mk(pr.better));
    statsPairwiseTable_->setItem(r, 5, mk(pr.significant ? "*" : "ns"));
  }

  statsPairwiseTable_->resizeColumnsToContents();

  if (haveActiveUi) {
    const int defaultDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);

    auto fillValueTable = [&](QTableWidget* table, QPushButton* button, bool useMeanMetric) {
      if (!table) return;
      // Same table setup as setupDerivedAnalysisTable(): ResizeToContents on both headers.
      TableBulkFillGuard bulk(table, QHeaderView::ResizeToContents, QHeaderView::ResizeToContents);
      QStringList headers;
      headers << "PROBLEM" << "DIM";
      for (const QString& m : methods) headers << m;
      table->clear();
      table->setRowCount(problems.size());
      table->setColumnCount(headers.size());
      table->setHorizontalHeaderLabels(headers);
      for (int r = 0; r < problems.size(); ++r) {
        const QString& prob = problems[r];
        table->setItem(r, 0, makeReadOnlyItem(prob));
        table->setItem(r, 1, makeReadOnlyItem(QString::number(batchDimForProblem(prob, defaultDim))));
        for (int j = 0; j < methods.size(); ++j) {
          const QString& meth = methods[j];
          QString text = "-";
          if (batchCells_.contains(prob) && batchCells_[prob].contains(meth)) {
            const BatchCellData& cell = batchCells_[prob][meth];
            const double val = useMeanMetric ? aggregateValues(cell.finalBestF, "Mean")
                                             : aggregateValues(cell.finalBestF, "Min");
            if (std::isfinite(val)) text = QString::number(val, 'g', 15);
          }
          auto* it = makeReadOnlyItem(text);
          it->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
          table->setItem(r, j + 2, it);
        }
      }
      table->resizeColumnsToContents();
      if (button) button->setEnabled(table->rowCount() > 0 && table->columnCount() > 0);
    };

    fillValueTable(activeUi.bestTable, activeUi.exportBestTableBtn, false);
    fillValueTable(activeUi.meanTable, activeUi.exportMeanTableBtn, true);

    struct RankRow {
      QString problem;
      int dim = 0;
      QVector<int> bestRanks;
      QVector<int> meanRanks;
    };

    QVector<RankRow> rankRows;
    rankRows.reserve(problems.size());
    QVector<double> bestTotals(methods.size(), 0.0);
    QVector<double> meanTotals(methods.size(), 0.0);

    for (const QString& prob : problems) {
      QVector<double> bestVals;
      QVector<double> meanVals;
      bestVals.reserve(methods.size());
      meanVals.reserve(methods.size());
      bool okAll = true;
      for (const QString& meth : methods) {
        if (!batchCells_.contains(prob) || !batchCells_[prob].contains(meth)) {
          okAll = false;
          break;
        }
        const BatchCellData& cell = batchCells_[prob][meth];
        const double bestVal = aggregateValues(cell.finalBestF, "Min");
        const double meanVal = aggregateValues(cell.finalBestF, "Mean");
        if (!std::isfinite(bestVal) || !std::isfinite(meanVal)) {
          okAll = false;
          break;
        }
        bestVals.push_back(bestVal);
        meanVals.push_back(meanVal);
      }
      if (!okAll) continue;

      RankRow row;
      row.problem = prob;
      row.dim = batchDimForProblem(prob, defaultDim);
      row.bestRanks = competitionRanksForMinimization(bestVals);
      row.meanRanks = competitionRanksForMinimization(meanVals);
      for (int j = 0; j < methods.size(); ++j) {
        bestTotals[j] += row.bestRanks[j];
        meanTotals[j] += row.meanRanks[j];
      }
      rankRows.push_back(std::move(row));
    }

    auto fillRankingTable = [&](QTableWidget* table, QPushButton* button, bool useMeanMetric) {
      if (!table) return;
      TableBulkFillGuard bulk(table, QHeaderView::ResizeToContents, QHeaderView::ResizeToContents);
      QStringList headers;
      headers << "PROBLEM" << "DIM";
      for (const QString& m : methods) headers << m;
      table->clear();
      table->setRowCount(rankRows.size());
      table->setColumnCount(headers.size());
      table->setHorizontalHeaderLabels(headers);
      for (int r = 0; r < rankRows.size(); ++r) {
        const RankRow& rr = rankRows[r];
        table->setItem(r, 0, makeReadOnlyItem(rr.problem));
        table->setItem(r, 1, makeReadOnlyItem(QString::number(rr.dim)));
        const QVector<int>& ranks = useMeanMetric ? rr.meanRanks : rr.bestRanks;
        for (int j = 0; j < methods.size(); ++j) {
          auto* it = makeReadOnlyItem(QString::number(ranks[j]));
          it->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
          applyTopRankHighlight(it, ranks[j]);
          table->setItem(r, j + 2, it);
        }
      }
      table->resizeColumnsToContents();
      if (button) button->setEnabled(table->rowCount() > 0 && table->columnCount() > 0);
    };

    fillRankingTable(activeUi.bestRankingTable, activeUi.exportBestRankingBtn, false);
    fillRankingTable(activeUi.meanRankingTable, activeUi.exportMeanRankingBtn, true);

    if (activeUi.finalRankingTable) {
      struct FinalRow {
        QString method;
        double bestTotal = 0.0;
        double meanTotal = 0.0;
        double overall = 0.0;
        double avg = 0.0;
      };
      QVector<FinalRow> finalRows;
      finalRows.reserve(methods.size());
      const double denomFinal = rankRows.isEmpty() ? 1.0 : (2.0 * double(rankRows.size()));
      for (int j = 0; j < methods.size(); ++j) {
        FinalRow fr;
        fr.method = methods[j];
        fr.bestTotal = bestTotals[j];
        fr.meanTotal = meanTotals[j];
        fr.overall = fr.bestTotal + fr.meanTotal;
        fr.avg = fr.overall / denomFinal;
        finalRows.push_back(std::move(fr));
      }
      std::sort(finalRows.begin(), finalRows.end(), [](const FinalRow& a, const FinalRow& b) {
        if (a.overall != b.overall) return a.overall < b.overall;
        if (a.bestTotal != b.bestTotal) return a.bestTotal < b.bestTotal;
        if (a.meanTotal != b.meanTotal) return a.meanTotal < b.meanTotal;
        return a.method.toLower() < b.method.toLower();
      });

      TableBulkFillGuard bulk(activeUi.finalRankingTable, QHeaderView::ResizeToContents, QHeaderView::ResizeToContents);
      activeUi.finalRankingTable->clear();
      activeUi.finalRankingTable->setRowCount(finalRows.size());
      activeUi.finalRankingTable->setColumnCount(5);
      activeUi.finalRankingTable->setHorizontalHeaderLabels(QStringList()
        << "Algorithm" << "Best Total Rank" << "Mean Total Rank" << "Overall Rank Sum" << "Average Rank");

      QVector<double> finalOverallVals;
      finalOverallVals.reserve(finalRows.size());
      for (const auto& fr : finalRows) finalOverallVals.push_back(fr.overall);
      const QVector<int> finalRanks = competitionRanksForMinimization(finalOverallVals);

      for (int r = 0; r < finalRows.size(); ++r) {
        const FinalRow& fr = finalRows[r];
        const int finalRank = (r < finalRanks.size() ? finalRanks[r] : (r + 1));
        auto* it0 = makeReadOnlyItem(fr.method);
        auto* it1 = makeReadOnlyItem(QString::number(fr.bestTotal, 'g', 15));
        auto* it2 = makeReadOnlyItem(QString::number(fr.meanTotal, 'g', 15));
        auto* it3 = makeReadOnlyItem(QString::number(fr.overall, 'g', 15));
        auto* it4 = makeReadOnlyItem(QString::number(fr.avg, 'f', 3));
        it1->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        it2->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        it3->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        it4->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        applyTopRankHighlight(it0, finalRank);
        applyTopRankHighlight(it1, finalRank);
        applyTopRankHighlight(it2, finalRank);
        applyTopRankHighlight(it3, finalRank);
        applyTopRankHighlight(it4, finalRank);
        activeUi.finalRankingTable->setItem(r, 0, it0);
        activeUi.finalRankingTable->setItem(r, 1, it1);
        activeUi.finalRankingTable->setItem(r, 2, it2);
        activeUi.finalRankingTable->setItem(r, 3, it3);
        activeUi.finalRankingTable->setItem(r, 4, it4);
      }
      activeUi.finalRankingTable->resizeColumnsToContents();
      if (activeUi.exportFinalRankingBtn) activeUi.exportFinalRankingBtn->setEnabled(activeUi.finalRankingTable->rowCount() > 0);
    }

    if (statsNoteLbl_) {
      QString note = "Notes: lower Value is better; tests use one aggregated value per (problem, method) cell; p-values are Holm-adjusted.";
      note += QString(" Ranking tabs use Best=min(final best_f), Mean=mean(final best_f), and competition ranking (1,1,3). Final ranking uses complete rows: %1/%2 problems.")
                .arg(rankRows.size())
                .arg(problems.size());
      statsNoteLbl_->setText(note);
    }
  }

  if (exportRankPlotBtn_) exportRankPlotBtn_->setEnabled(true);

  // Wilcoxon tab is updated separately, but uses the same selected methods/problems and alpha.
  rebuildWilcoxonPlot();
}

void MainWindow::rebuildWilcoxonPlot() {
  if (!wilcoxonPlot_ || !wilcoxonSummaryLbl_) return;
  auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_);
  if (!wp) return;

  const QStringList methods = effectiveStatsMethodShortNames();
  const QStringList problems = effectiveStatsProblemShortNames();

  const BatchMetricMode mode = currentBatchMetricMode();
  const QString agg = (batchAggCombo_ ? batchAggCombo_->currentText() : "Mean");

  auto cellValue = [&](const QString& prob, const QString& meth) -> double {
    if (!batchCells_.contains(prob) || !batchCells_[prob].contains(meth)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const BatchCellData& cell = batchCells_[prob][meth];
    if (mode == BatchMetricMode::BestFinalBestF) {
      return aggregateValues(cell.finalBestF, "Min");
    }
    if (mode == BatchMetricMode::MeanFinalBestF) {
      return aggregateValues(cell.finalBestF, "Mean");
    }
    if (mode == BatchMetricMode::IterationAtBest) {
      return aggregateValues(cell.hitIter, agg);
    }
    return aggregateValues(cell.hitEvals, agg);
  };

  // Keep only problems that have finite values for all methods (paired tests need aligned rows).
  QStringList usedProblems;
  for (const QString& prob : problems) {
    bool okAll = true;
    for (const QString& meth : methods) {
      const double v = cellValue(prob, meth);
      if (!std::isfinite(v)) { okAll = false; break; }
    }
    if (okAll) usedProblems << prob;
  }

  const int k = methods.size();
  const int N = usedProblems.size();

  if (k < 2 || N < 2) {
    wp->setData(QStringList(), QVector<QVector<double>>(), QString(), QString(), QVector<WilcoxonBoxPlotWidget::Annotation>());
    wilcoxonSummaryLbl_->setText("No Wilcoxon comparisons: need at least 2 methods and 2 problems with complete data.");
    if (exportWilcoxonPlotBtn_) exportWilcoxonPlotBtn_->setEnabled(false);
    return;
  }

  // Groups: one vector per method.
  QVector<QVector<double>> groups;
  groups.reserve(k);
  for (int j = 0; j < k; ++j) {
    QVector<double> g;
    g.reserve(N);
    for (const QString& prob : usedProblems) {
      g.push_back(cellValue(prob, methods[j]));
    }
    groups.push_back(std::move(g));
  }

  // Alpha.
  double alpha = 0.05;
  if (statsAlphaCombo_) {
    bool ok = false;
    const double a = statsAlphaCombo_->currentData().toDouble(&ok);
    if (ok && a > 0.0 && a < 1.0) alpha = a;
  }

  // Paired Wilcoxon signed-rank across problems + Holm adjustment.
  QVector<PairwiseResult> pairs;
  pairs.reserve(k * (k - 1) / 2);

  for (int i = 0; i < k; ++i) {
    for (int j = i + 1; j < k; ++j) {
      QVector<double> diffs;
      diffs.reserve(N);
      for (int p = 0; p < N; ++p) {
        diffs.push_back(groups[i][p] - groups[j][p]);
      }

      // Remove zero diffs (ties).
      QVector<double> d;
      d.reserve(diffs.size());
      for (double x : diffs) {
        const double eps = std::max(1e-12, 1e-9 * std::max(std::abs(x), 1.0));
        if (std::abs(x) > eps) d.push_back(x);
      }

      PairwiseResult pr;
      pr.a = methods[i];
      pr.b = methods[j];
      pr.better = QString();
      pr.p = std::numeric_limits<double>::quiet_NaN();

      const int n = d.size();
      if (n >= 2) {
        // Rank absolute differences.
        QVector<int> idx(n);
        for (int t = 0; t < n; ++t) idx[t] = t;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
          const double aa = std::abs(d[a]);
          const double bb = std::abs(d[b]);
          if (aa == bb) return a < b;
          return aa < bb;
        });

        QVector<double> ranks(n, 0.0);
        int pos = 0;
        while (pos < n) {
          const int i0 = pos;
          const double v0 = std::abs(d[idx[i0]]);
          int i1 = i0;
          while (i1 + 1 < n) {
            const double v1 = std::abs(d[idx[i1 + 1]]);
            const double eps = std::max(1e-12, 1e-9 * std::max(std::abs(v0), 1.0));
            if (std::abs(v1 - v0) > eps) break;
            ++i1;
          }
          const double rStart = double(i0 + 1);
          const double rEnd = double(i1 + 1);
          const double rAvg = 0.5 * (rStart + rEnd);
          for (int t = i0; t <= i1; ++t) ranks[idx[t]] = rAvg;
          pos = i1 + 1;
        }

        double Wplus = 0.0;
        for (int t = 0; t < n; ++t) {
          if (d[t] > 0.0) Wplus += ranks[t];
        }

        const double meanW = double(n) * double(n + 1) / 4.0;
        const double sdW = std::sqrt(double(n) * double(n + 1) * double(2 * n + 1) / 24.0);
        double z = 0.0;
        if (sdW > 0.0) {
          const double cc = 0.5 * ((Wplus > meanW) ? 1.0 : (Wplus < meanW ? -1.0 : 0.0));
          z = (Wplus - meanW - cc) / sdW;
        }
        pr.p = twoSidedNormalP(z);
      }

      // Better method = smaller mean across problems.
      double mi = 0.0, mj = 0.0;
      for (int p = 0; p < N; ++p) { mi += groups[i][p]; mj += groups[j][p]; }
      mi /= double(N);
      mj /= double(N);
      pr.better = (mi < mj) ? pr.a : pr.b;

      pairs.push_back(pr);
    }
  }

  holmAdjust(pairs);
  for (auto& pr : pairs) pr.significant = (std::isfinite(pr.pAdj) && pr.pAdj <= alpha);

  auto fmtPadj = [](double p) -> QString {
    if (!std::isfinite(p)) p = 1.0;
    return QString("p_adj=%1").arg(QString::number(p, 'g', 6));
  };

  // Determine which comparisons to display.
  int pairsMode = 0;
  if (wilcoxonPairsCombo_) {
    bool ok = false;
    pairsMode = wilcoxonPairsCombo_->currentData().toInt(&ok);
    if (!ok) pairsMode = wilcoxonPairsCombo_->currentIndex();
  }

  int bestIdx = 0;
  if (pairsMode == 1) {
    // Determine the "best" method by *average rank* (same logic as the Friedman
    // Ranking tab) so that the Wilcoxon "Best vs others" selection is always
    // consistent with the ranking tables.
    // Using a raw arithmetic mean of metric values can silently pick the wrong
    // winner when two methods are numerically tied within epsilon on every
    // problem — the ranking functions handle those ties correctly with average
    // rank assignment, whereas a bare float comparison does not.
    QVector<double> sumRanksLocal(k, 0.0);
    for (int p = 0; p < N; ++p) {
      QVector<double> rowVals(k);
      for (int j = 0; j < k; ++j) rowVals[j] = groups[j][p];
      const QVector<double> r = ranksForMinimization(rowVals);
      for (int j = 0; j < k; ++j) sumRanksLocal[j] += r[j];
    }
    double bestAvgRank = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
      const double ar = sumRanksLocal[j] / double(N);
      if (ar < bestAvgRank) { bestAvgRank = ar; bestIdx = j; }
    }
  }

  QVector<WilcoxonBoxPlotWidget::Annotation> ann;
  ann.reserve(pairs.size());

  auto idxOf = [&](const QString& name) -> int {
    for (int j = 0; j < methods.size(); ++j) {
      if (methods[j] == name) return j;
    }
    return -1;
  };

  for (const auto& pr : pairs) {
    const int i = idxOf(pr.a);
    const int j = idxOf(pr.b);
    if (i < 0 || j < 0) continue;
    if (pairsMode == 1 && !(i == bestIdx || j == bestIdx)) continue;

    WilcoxonBoxPlotWidget::Annotation a;
    a.i = i;
    a.j = j;
    a.text = fmtPadj(pr.pAdj);
    a.significant = pr.significant;
    ann.push_back(a);
  }

  const QString title = "Box plots per method";
  const QString subtitle = QString("Paired Wilcoxon (Holm-adjusted), alpha=%1, N=%2 (%3)")
    .arg(alpha, 0, 'g', 3)
    .arg(N)
    .arg(pairsMode == 1 ? "Best vs others" : "All pairs");

  wp->setYLabel(batchMetricYAxisLabel());
  wp->setData(methods, groups, title, subtitle, ann);

  wilcoxonSummaryLbl_->setText(QString("Used problems: %1/%2 (complete rows). Displayed comparisons: %3.")
    .arg(N)
    .arg(problems.size())
    .arg(ann.size()));

  if (exportWilcoxonPlotBtn_) exportWilcoxonPlotBtn_->setEnabled(true);
}

namespace {

static QString xmlEscape(QString s) {
  s.replace('&', "&amp;");
  s.replace('<', "&lt;");
  s.replace('>', "&gt;");
  s.replace('"', "&quot;");
  s.replace('\'', "&apos;");
  return s;
}

static QString excelColName(int col0) {
  // 0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ...
  QString out;
  int n = col0;
  do {
    const int rem = n % 26;
    out.prepend(QChar('A' + rem));
    n = (n / 26) - 1;
  } while (n >= 0);
  return out;
}

static quint32 crc32TableEntry(quint32 i) {
  quint32 c = i;
  for (int k = 0; k < 8; ++k) {
    if (c & 1U) c = 0xEDB88320U ^ (c >> 1);
    else c >>= 1;
  }
  return c;
}

static quint32 crc32(const QByteArray& data) {
  static quint32 table[256];
  static bool init = false;
  if (!init) {
    for (quint32 i = 0; i < 256; ++i) table[i] = crc32TableEntry(i);
    init = true;
  }
  quint32 c = 0xFFFFFFFFU;
  for (unsigned char b : data) {
    c = table[(c ^ quint32(b)) & 0xFFU] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFU;
}

static void writeLe16(QIODevice& dev, quint16 v) {
  char b[2];
  b[0] = char(v & 0xFF);
  b[1] = char((v >> 8) & 0xFF);
  dev.write(b, 2);
}

static void writeLe32(QIODevice& dev, quint32 v) {
  char b[4];
  b[0] = char(v & 0xFF);
  b[1] = char((v >> 8) & 0xFF);
  b[2] = char((v >> 16) & 0xFF);
  b[3] = char((v >> 24) & 0xFF);
  dev.write(b, 4);
}

struct ZipEntry {
  QByteArray nameUtf8;
  QByteArray data;
  quint32 crc = 0;
  quint32 localHeaderOffset = 0;
};

static bool writeZipStored(const QString& outPath, const QVector<ZipEntry>& entries, QString* err) {
  QSaveFile f(outPath);
  if (!f.open(QIODevice::WriteOnly)) {
    if (err) *err = QString("Failed to open file for writing: %1").arg(outPath);
    return false;
  }

  QVector<ZipEntry> local = entries;

  // Local headers + data
  for (int i = 0; i < local.size(); ++i) {
    ZipEntry& e = local[i];
    e.localHeaderOffset = quint32(f.pos());
    const quint32 sz = quint32(e.data.size());
    const quint32 c = crc32(e.data);
    e.crc = c;

    // Local file header
    writeLe32(f, 0x04034b50U);
    writeLe16(f, 20);         // version needed
    writeLe16(f, 0);          // flags
    writeLe16(f, 0);          // compression (0 = store)
    writeLe16(f, 0);          // mod time
    writeLe16(f, 0);          // mod date
    writeLe32(f, c);
    writeLe32(f, sz);
    writeLe32(f, sz);
    writeLe16(f, quint16(e.nameUtf8.size()));
    writeLe16(f, 0);          // extra len
    f.write(e.nameUtf8);
    f.write(e.data);
  }

  const quint32 centralDirOffset = quint32(f.pos());

  // Central directory
  for (const ZipEntry& e : local) {
    const quint32 sz = quint32(e.data.size());
    writeLe32(f, 0x02014b50U);
    writeLe16(f, 20);         // version made by
    writeLe16(f, 20);         // version needed
    writeLe16(f, 0);          // flags
    writeLe16(f, 0);          // compression
    writeLe16(f, 0);          // mod time
    writeLe16(f, 0);          // mod date
    writeLe32(f, e.crc);
    writeLe32(f, sz);
    writeLe32(f, sz);
    writeLe16(f, quint16(e.nameUtf8.size()));
    writeLe16(f, 0);          // extra
    writeLe16(f, 0);          // comment
    writeLe16(f, 0);          // disk number
    writeLe16(f, 0);          // internal attrs
    writeLe32(f, 0);          // external attrs
    writeLe32(f, e.localHeaderOffset);
    f.write(e.nameUtf8);
  }

  const quint32 centralDirSize = quint32(f.pos()) - centralDirOffset;

  // End of central directory
  writeLe32(f, 0x06054b50U);
  writeLe16(f, 0); // disk
  writeLe16(f, 0); // start disk
  writeLe16(f, quint16(local.size()));
  writeLe16(f, quint16(local.size()));
  writeLe32(f, centralDirSize);
  writeLe32(f, centralDirOffset);
  writeLe16(f, 0); // comment len

  if (!f.commit()) {
    if (err) *err = QString("Failed to finalize file: %1").arg(outPath);
    return false;
  }
  return true;
}

static QByteArray makeXmlSheet(const QVector<QVector<QVariant>>& grid) {
  QByteArray out;
  out.reserve(16384);
  out += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>";
  out += "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">";
  out += "<sheetData>";

  for (int r = 0; r < grid.size(); ++r) {
    const int row1 = r + 1;
    out += "<row r=\"" + QByteArray::number(row1) + "\">";
    const auto& row = grid[r];
    for (int c = 0; c < row.size(); ++c) {
      const QString cellRef = excelColName(c) + QString::number(row1);
      const QVariant& v = row[c];

      if (!v.isValid() || v.isNull()) {
        // Skip empty cells.
        continue;
      }

      if (v.typeId() == QMetaType::Double || v.typeId() == QMetaType::Int || v.typeId() == QMetaType::LongLong || v.typeId() == QMetaType::UInt || v.typeId() == QMetaType::ULongLong) {
        const double dv = v.toDouble();
        // NaN/Inf serialize (via QByteArray::number) to the literal text
        // "nan"/"inf", which is not valid content for a numeric cell with no
        // t="..." override (OOXML requires <v> to be a plain number there) --
        // confirmed reproducible: openpyxl (and likely Excel's own strict
        // parser) refuses to open a sheet containing such a cell at all.
        // Leaving the cell blank matches the existing "skip empty" convention
        // just above, and is what a NaN generalization-gap-error value (e.g.
        // for a regression run, where accuracy/error isn't defined) should
        // look like in the exported spreadsheet anyway.
        if (std::isnan(dv) || std::isinf(dv)) {
          continue;
        }
        out += "<c r=\"" + cellRef.toUtf8() + "\"><v>" + QByteArray::number(dv, 'g', 15) + "</v></c>";
      } else {
        const QString sv = v.toString();
        out += "<c r=\"" + cellRef.toUtf8() + "\" t=\"inlineStr\"><is><t>" + xmlEscape(sv).toUtf8() + "</t></is></c>";
      }
    }
    out += "</row>";
  }

  out += "</sheetData></worksheet>";
  return out;
}

// One named worksheet's data (row 0 = header). Reused by
// writeMultiSheetXlsxNative() below to build a single .xlsx workbook with
// several tabs -- used by exportUScoreReport() so the U-score batch
// report can ship as a real multi-sheet .xlsx instead of one flat .csv.
struct XlsxSheet {
  QString name;
  QVector<QVector<QVariant>> grid;
};

static bool writeMultiSheetXlsxNative(const QString& outPath,
                                       const QVector<XlsxSheet>& sheets,
                                       QString* err) {
  if (sheets.isEmpty()) {
    if (err) *err = "No sheets to write.";
    return false;
  }

  QVector<ZipEntry> entries;
  entries.reserve(4 + sheets.size());

  QString sheetsXml, relsXml, typesOverrides;
  for (int i = 0; i < sheets.size(); ++i) {
    const int idx1 = i + 1;
    const QByteArray sheetXml = makeXmlSheet(sheets[i].grid);
    entries.push_back({QString("xl/worksheets/sheet%1.xml").arg(idx1).toUtf8(), sheetXml});

    // Excel worksheet-name rules: <=31 chars, none of []:*?/\ .
    QString safeName = sheets[i].name;
    safeName.replace(QRegularExpression("[\\[\\]:\\*\\?/\\\\]"), "_");
    if (safeName.size() > 31) safeName = safeName.left(31);
    if (safeName.isEmpty()) safeName = QString("Sheet%1").arg(idx1);

    sheetsXml += QString("<sheet name=\"%1\" sheetId=\"%2\" r:id=\"rId%2\"/>")
                   .arg(xmlEscape(safeName)).arg(idx1);
    relsXml += QString("<Relationship Id=\"rId%1\" "
                        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
                        "Target=\"worksheets/sheet%1.xml\"/>").arg(idx1);
    typesOverrides += QString("<Override PartName=\"/xl/worksheets/sheet%1.xml\" "
                               "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>")
                        .arg(idx1);
  }

  const QByteArray workbook = (
    QString("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            "<sheets>")
    + sheetsXml + "</sheets></workbook>").toUtf8();

  const QByteArray relsRoot =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
    "</Relationships>";

  const QByteArray relsWb = (
    QString("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">")
    + relsXml + "</Relationships>").toUtf8();

  const QByteArray contentTypes = (
    QString("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>")
    + typesOverrides + "</Types>").toUtf8();

  entries.push_back({QByteArray("[Content_Types].xml"), contentTypes});
  entries.push_back({QByteArray("_rels/.rels"), relsRoot});
  entries.push_back({QByteArray("xl/workbook.xml"), workbook});
  entries.push_back({QByteArray("xl/_rels/workbook.xml.rels"), relsWb});

  return writeZipStored(outPath, entries, err);
}

} // namespace

bool MainWindow::exportStatsToXlsxNative(const QString& outPath) const {
  if (!statsTable_ || statsTable_->rowCount() == 0 || statsTable_->columnCount() == 0) {
    return false;
  }

  // Build a simple 2D grid: header + rows.
  QVector<QVector<QVariant>> grid;
  grid.reserve(statsTable_->rowCount() + 1);

  QVector<QVariant> header;
  header.reserve(statsTable_->columnCount());
  for (int c = 0; c < statsTable_->columnCount(); ++c) {
    const QString h = statsTable_->horizontalHeaderItem(c) ? statsTable_->horizontalHeaderItem(c)->text() : QString();
    header.push_back(h);
  }
  grid.push_back(header);

  for (int r = 0; r < statsTable_->rowCount(); ++r) {
    QVector<QVariant> row;
    row.reserve(statsTable_->columnCount());
    for (int c = 0; c < statsTable_->columnCount(); ++c) {
      const QTableWidgetItem* it = statsTable_->item(r, c);
      const QString s = it ? it->text().trimmed() : QString();

      if (s.isEmpty()) {
        row.push_back(QVariant());
        continue;
      }

      if (c == 0) {
        // PROBLEM column is always text.
        row.push_back(s);
      } else {
        bool okNum = false;
        const double d = s.toDouble(&okNum);
        row.push_back(okNum ? QVariant(d) : QVariant(s));
      }
    }
    grid.push_back(row);
  }

  const QByteArray sheet1 = makeXmlSheet(grid);

  const QByteArray workbook =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
    "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
    "<sheets><sheet name=\"stats\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
    "</workbook>";

  const QByteArray relsRoot =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
    "</Relationships>";

  const QByteArray relsWb =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
    "</Relationships>";

  const QByteArray contentTypes =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
    "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
    "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
    "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
    "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
    "</Types>";

  QVector<ZipEntry> entries;
  entries.reserve(5);
  entries.push_back({QByteArray("[Content_Types].xml"), contentTypes});
  entries.push_back({QByteArray("_rels/.rels"), relsRoot});
  entries.push_back({QByteArray("xl/workbook.xml"), workbook});
  entries.push_back({QByteArray("xl/_rels/workbook.xml.rels"), relsWb});
  entries.push_back({QByteArray("xl/worksheets/sheet1.xml"), sheet1});

  QString err;
  const bool ok = writeZipStored(outPath, entries, &err);
  if (!ok) {
    CrashLog::append(QString("XLSX export failed: %1").arg(err));
  }
  return ok;
}

bool MainWindow::exportTableWidgetToXlsxNative(QTableWidget* table, const QString& outPath) const {
  if (!table || table->rowCount() == 0 || table->columnCount() == 0) {
    return false;
  }

  QVector<QVector<QVariant>> grid;
  grid.reserve(table->rowCount() + 1);

  QVector<QVariant> header;
  header.reserve(table->columnCount());
  for (int c = 0; c < table->columnCount(); ++c) {
    const QString h = table->horizontalHeaderItem(c) ? table->horizontalHeaderItem(c)->text() : QString();
    header.push_back(h);
  }
  grid.push_back(header);

  for (int r = 0; r < table->rowCount(); ++r) {
    QVector<QVariant> row;
    row.reserve(table->columnCount());
    for (int c = 0; c < table->columnCount(); ++c) {
      const QTableWidgetItem* it = table->item(r, c);
      const QString s = it ? it->text().trimmed() : QString();
      if (s.isEmpty()) {
        row.push_back(QVariant());
        continue;
      }
      bool okNum = false;
      const double d = s.toDouble(&okNum);
      row.push_back(okNum ? QVariant(d) : QVariant(s));
    }
    grid.push_back(row);
  }

  const QByteArray sheet1 = makeXmlSheet(grid);

  const QByteArray workbook =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
    "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
    "<sheets><sheet name=\"results\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
    "</workbook>";

  const QByteArray relsRoot =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
    "</Relationships>";

  const QByteArray relsWb =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
    "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
    "</Relationships>";

  const QByteArray contentTypes =
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
    "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
    "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
    "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
    "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
    "</Types>";

  QVector<ZipEntry> entries;
  entries.reserve(5);
  entries.push_back({QByteArray("[Content_Types].xml"), contentTypes});
  entries.push_back({QByteArray("_rels/.rels"), relsRoot});
  entries.push_back({QByteArray("xl/workbook.xml"), workbook});
  entries.push_back({QByteArray("xl/_rels/workbook.xml.rels"), relsWb});
  entries.push_back({QByteArray("xl/worksheets/sheet1.xml"), sheet1});

  QString err2;
  const bool ok2 = writeZipStored(outPath, entries, &err2);
  if (!ok2) {
    CrashLog::append(QString("XLSX export failed: %1").arg(err2));
  }
  return ok2;
}

bool MainWindow::exportStatsToXlsxViaPython(const QString& outPath, const QString& csvPath) {
  // Attempts XLSX export via the system Python interpreter + openpyxl. If unavailable, returns false.
  QProcess p;
  QStringList args;
  args << "-c";

  const QString code =
    "import sys,csv\n"
    "from openpyxl import Workbook\n"
    "wb=Workbook()\n"
    "ws=wb.active\n"
    "with open(sys.argv[1], newline='', encoding='utf-8') as f:\n"
    "  r=csv.reader(f)\n"
    "  for row in r:\n"
    "    ws.append(row)\n"
    "wb.save(sys.argv[2])\n";

  args << code << csvPath << outPath;

  p.start("python", args);
  if (!p.waitForFinished(60000)) {
    p.kill();
    return false;
  }
  return (p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0);
}

void MainWindow::exportTableWidgetToXlsxOrCsv(QTableWidget* table, const QString& dialogTitle, const QString& defaultPath) {
  if (!table || table->rowCount() == 0) return;

  const QString out = QFileDialog::getSaveFileName(
    this, dialogTitle, defaultPath, "Excel Workbook (*.xlsx);;CSV (*.csv)");
  if (out.isEmpty()) return;

  // exportStatsToXlsxNative() is hard-coded to read from statsTable_ (see
  // above); point it at the caller's table for the duration of this export,
  // exactly as the Batch run derived-table buttons already do.
  QTableWidget* savedStatsTable = statsTable_;
  statsTable_ = table;

  const QString lower = out.toLower();
  if (lower.endsWith(".xlsx")) {
    if (exportStatsToXlsxNative(out)) {
      statsTable_ = savedStatsTable;
      return;
    }
  }

  QTemporaryFile tmp(QDir::tempPath() + "/optimsolution_table_XXXXXX.csv");
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
      QMessageBox::information(this, "Export",
        "XLSX export requires Python + openpyxl on the system.\n"
        "A CSV file was written instead:\n" + csvOut);
    }
  } else {
    QFile::remove(out);
    QFile::copy(tmp.fileName(), out);
  }

  statsTable_ = savedStatsTable;
}

bool MainWindow::exportWidgetToPng300Dpi(QWidget* widget, const QString& filePath) const {
  if (!widget) return false;
  const int targetDpi = 300, baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);
  const QSize sz = widget->size();
  if (sz.width() <= 0 || sz.height() <= 0) return false;
  const QSize outSz(int(std::ceil(sz.width() * scale)), int(std::ceil(sz.height() * scale)));
  QImage img(outSz, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::transparent);
  const int dpm = int(std::llround(double(targetDpi) / 25.4 * 1000.0));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);
  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  widget->render(&painter);
  painter.end();
  return img.save(filePath, "PNG");
}

void MainWindow::onExportStats() {
  if (!statsTable_ || statsTable_->rowCount() == 0) {
    return;
  }

  const QString out = QFileDialog::getSaveFileName(
    this,
    "Export statistics table",
    QDir::homePath() + "/stats.xlsx",
    "Excel Workbook (*.xlsx);;CSV (*.csv)"
  );
  if (out.isEmpty()) {
    return;
  }

  const QString lower = out.toLower();
  if (lower.endsWith(".xlsx")) {
    // Primary path: native XLSX writer (cross-platform, no external dependencies).
    if (exportStatsToXlsxNative(out)) {
      return;
    }
    // If native export fails, fall back to the existing CSV->Python(openpyxl) bridge.
  }

  // Write to a temporary CSV first.
  QTemporaryFile tmp(QDir::tempPath() + "/optimsolution_stats_XXXXXX.csv");
  tmp.setAutoRemove(true);
  if (!tmp.open()) {
    QMessageBox::warning(this, "Export", "Failed to create a temporary file.");
    return;
  }

  QTextStream ts(&tmp);
  ts.setEncoding(QStringConverter::Utf8);

  // Header
  QStringList header;
  for (int c = 0; c < statsTable_->columnCount(); ++c) {
    header << statsTable_->horizontalHeaderItem(c)->text();
  }
  ts << header.join(",") << "\n";

  for (int r = 0; r < statsTable_->rowCount(); ++r) {
    QStringList row;
    for (int c = 0; c < statsTable_->columnCount(); ++c) {
      QTableWidgetItem* it = statsTable_->item(r, c);
      QString cell = it ? it->text() : "";
      // Escape commas/quotes.
      if (cell.contains(',') || cell.contains('"') || cell.contains('\n') || cell.contains('\r')) {
        cell.replace("\"", "\"\"");
        cell = "\"" + cell + "\"" ;
      }
      row << cell;
    }
    ts << row.join(",") << "\n";
  }
  ts.flush();
  tmp.flush();

  if (lower.endsWith(".xlsx")) {
    if (!exportStatsToXlsxViaPython(out, tmp.fileName())) {
      // Fallback: export CSV next to the requested XLSX.
      const QString csvOut = out.left(out.size() - 5) + ".csv";
      QFile::remove(csvOut);
      QFile::copy(tmp.fileName(), csvOut);
      QMessageBox::information(
        this,
        "Export",
        "XLSX export requires Python + openpyxl on the system.\n"
        "A CSV file was written instead:\n" + csvOut
      );
    }
  } else {
    QFile::remove(out);
    QFile::copy(tmp.fileName(), out);
  }
}

void MainWindow::onStatsTestChanged(int) {
  rebuildStatsComparisons();
}

void MainWindow::onExportRankPlotPng() {
  if (!rankPlot_) {
    return;
  }
  const QString out = QFileDialog::getSaveFileName(
    this,
    "Export rank plot (PNG, 300 DPI)",
    QDir::homePath() + "/rank_plot.png",
    "PNG Image (*.png)"
  );
  if (out.isEmpty()) {
    return;
  }
  if (!exportRankPlotPng(out)) {
    QMessageBox::warning(this, "Export", "Failed to export the rank plot.");
  }
}

void MainWindow::onExportWilcoxonPlotPng() {
  if (!wilcoxonPlot_) {
    return;
  }
  const QString out = QFileDialog::getSaveFileName(
    this,
    "Export Wilcoxon box plot (PNG, 300 DPI)",
    QDir::homePath() + "/wilcoxon_boxplot.png",
    "PNG Image (*.png)"
  );
  if (out.isEmpty()) {
    return;
  }
  if (!exportWilcoxonPlotPng(out)) {
    QMessageBox::warning(this, "Export", "Failed to export the Wilcoxon box plot.");
  }
}

bool MainWindow::exportWilcoxonPlotPng(const QString& filePath) {
  auto* plot = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_);
  if (!plot) {
    return false;
  }

  const int targetDpi = 300;
  const int baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);

  const QSize sz = plot->size();
  if (sz.width() <= 0 || sz.height() <= 0) {
    return false;
  }

  const QSize outSz(int(std::ceil(sz.width() * scale)), int(std::ceil(sz.height() * scale)));
  QImage img(outSz, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::transparent);

  const int dpm = int(std::llround(double(targetDpi) / 25.4 * 1000.0));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}

bool MainWindow::exportRankPlotPng(const QString& filePath) {
  auto* plot = dynamic_cast<RankPlotWidget*>(rankPlot_);
  if (!plot) { return false; }
  const int targetDpi = 300, baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);
  const QSize sz = plot->size();
  if (sz.width() <= 0 || sz.height() <= 0) { return false; }
  plot->setInfoVisible(false);
  const QSize outSz(int(std::ceil(sz.width()*scale)), int(std::ceil(sz.height()*scale)));
  QImage img(outSz, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::transparent);
  const int dpm = int(std::llround(double(targetDpi) / 25.4 * 1000.0));
  img.setDotsPerMeterX(dpm); img.setDotsPerMeterY(dpm);
  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();
  const bool ok = img.save(filePath, "PNG");
  plot->setInfoVisible(true);
  return ok;
}

void MainWindow::onExportCdPlotPng() {
  if (!cdPlot_) {
    return;
  }
  const QString out = QFileDialog::getSaveFileName(
    this,
    "Export CD diagram (PNG, 300 DPI)",
    QDir::homePath() + "/cd_diagram.png",
    "PNG Image (*.png)"
  );
  if (out.isEmpty()) {
    return;
  }
  if (!exportCdPlotPng(out)) {
    QMessageBox::warning(this, "Export", "Failed to export the CD diagram.");
  }
}

bool MainWindow::exportCdPlotPng(const QString& filePath) {
  auto* plot = dynamic_cast<CDDiagramWidget*>(cdPlot_);
  if (!plot) {
    return false;
  }

  const int targetDpi = 300;
  const int baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);

  const QSize sz = plot->size();
  if (sz.width() <= 0 || sz.height() <= 0) {
    return false;
  }

  const QSize outSz(int(std::ceil(sz.width() * scale)), int(std::ceil(sz.height() * scale)));
  QImage img(outSz, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::transparent);

  const int dpm = int(std::llround(double(targetDpi) / 25.4 * 1000.0));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}


namespace {

QString reportTableHtml(QTableWidget* table) {
  if (!table || table->rowCount() == 0 || table->columnCount() == 0) {
    return QStringLiteral("<p><i>(no data)</i></p>");
  }
  QString html = QStringLiteral(
    "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\" style=\"border-collapse:collapse; font-size:9pt;\">");
  html += QStringLiteral("<tr>");
  for (int c = 0; c < table->columnCount(); ++c) {
    const QString h = table->horizontalHeaderItem(c) ? table->horizontalHeaderItem(c)->text() : QString();
    html += QStringLiteral("<th style=\"background:#dddddd;\">%1</th>").arg(h.toHtmlEscaped());
  }
  html += QStringLiteral("</tr>");
  for (int r = 0; r < table->rowCount(); ++r) {
    html += QStringLiteral("<tr>");
    for (int c = 0; c < table->columnCount(); ++c) {
      const QString v = table->item(r, c) ? table->item(r, c)->text() : QString();
      html += QStringLiteral("<td>%1</td>").arg(v.toHtmlEscaped());
    }
    html += QStringLiteral("</tr>");
  }
  html += QStringLiteral("</table>");
  return html;
}

QByteArray reportRenderWidgetPng(QWidget* w, double scale = 2.0) {
  if (!w) return QByteArray();
  const QSize sz = w->size();
  if (sz.width() <= 0 || sz.height() <= 0) return QByteArray();
  const QSize outSz(int(std::ceil(sz.width() * scale)), int(std::ceil(sz.height() * scale)));
  QImage img(outSz, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::white); // white background reads well on a printed page
  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  w->render(&painter);
  painter.end();
  QByteArray bytes;
  QBuffer buf(&bytes);
  buf.open(QIODevice::WriteOnly);
  img.save(&buf, "PNG");
  return bytes;
}

QString reportImgTag(QWidget* w, int displayWidthPx = 640) {
  const QByteArray png = reportRenderWidgetPng(w);
  if (png.isEmpty()) return QStringLiteral("<p><i>(plot not available)</i></p>");
  return QStringLiteral("<p><img src=\"data:image/png;base64,%1\" width=\"%2\"></p>")
    .arg(QString::fromLatin1(png.toBase64()))
    .arg(displayWidthPx);
}

QString reportEscaped(const QString& s) {
  QString out = s.toHtmlEscaped();
  out.replace(QStringLiteral("\n"), QStringLiteral("<br>"));
  return out;
}

} // namespace

QString MainWindow::buildBatchReportHtml(bool includeConvergence) const {
  BatchSummaryUiBundle activeUi;
  bool haveActiveUi = false;
  if (const OutputRunTab* active = outputTab(activeOutputRunIndex_)) {
    if (active->page && g_batchSummaryUi.contains(active->page)) {
      activeUi = g_batchSummaryUi.value(active->page);
      haveActiveUi = true;
    }
  }
  if (!haveActiveUi && g_activeBatchSummaryPage && g_batchSummaryUi.contains(g_activeBatchSummaryPage)) {
    activeUi = g_batchSummaryUi.value(g_activeBatchSummaryPage);
    haveActiveUi = true;
  }
  if (!haveActiveUi || !activeUi.statsTable || activeUi.statsTable->rowCount() == 0) {
    return QString();
  }

  const QStringList methods = effectiveStatsMethodShortNames();
  const QStringList problems = effectiveStatsProblemShortNames();
  const QString generatedAt = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");

  QString html;
  html += QStringLiteral("<html><body style=\"font-family:sans-serif;\">");
  html += QStringLiteral("<h1>Optimization Batch Report</h1>");
  html += QStringLiteral("<p><b>Generated:</b> %1<br>").arg(generatedAt.toHtmlEscaped());
  html += QStringLiteral("<b>Methods (%1):</b> %2<br>").arg(methods.size()).arg(methods.join(", ").toHtmlEscaped());
  html += QStringLiteral("<b>Problems (%1):</b> %2</p>").arg(problems.size()).arg(problems.join(", ").toHtmlEscaped());

  html += QStringLiteral("<h2>Results Table</h2>");
  html += reportTableHtml(activeUi.statsTable);
  if (activeUi.statsNoteLbl) {
    html += QStringLiteral("<p><i>%1</i></p>").arg(reportEscaped(activeUi.statsNoteLbl->text()));
  }

  html += QStringLiteral("<h2>Rankings</h2>");
  if (activeUi.bestRankingTable && activeUi.bestRankingTable->rowCount() > 0) {
    html += QStringLiteral("<h3>Best Ranking</h3>") + reportTableHtml(activeUi.bestRankingTable);
  }
  if (activeUi.meanRankingTable && activeUi.meanRankingTable->rowCount() > 0) {
    html += QStringLiteral("<h3>Mean Ranking</h3>") + reportTableHtml(activeUi.meanRankingTable);
  }
  if (activeUi.finalRankingTable && activeUi.finalRankingTable->rowCount() > 0) {
    html += QStringLiteral("<h3>Final Ranking</h3>") + reportTableHtml(activeUi.finalRankingTable);
  }

  html += QStringLiteral("<h2>Pairwise Wilcoxon</h2>");
  if (activeUi.wilcoxonSummaryLbl) {
    html += QStringLiteral("<p>%1</p>").arg(reportEscaped(activeUi.wilcoxonSummaryLbl->text()));
  }
  html += reportImgTag(activeUi.wilcoxonPlot);

  html += QStringLiteral("<h2>Friedman Ranking</h2>");
  if (activeUi.statsSummaryLbl) {
    html += QStringLiteral("<p>%1</p>").arg(reportEscaped(activeUi.statsSummaryLbl->text()));
  }
  html += reportImgTag(activeUi.rankPlot);
  if (activeUi.statsPairwiseTable && activeUi.statsPairwiseTable->rowCount() > 0) {
    html += QStringLiteral("<h3>Pairwise comparisons (Holm-adjusted)</h3>") + reportTableHtml(activeUi.statsPairwiseTable);
  }

  html += QStringLiteral("<h2>Critical Difference (Nemenyi)</h2>");
  if (activeUi.cdSummaryLbl) {
    html += QStringLiteral("<p>%1</p>").arg(reportEscaped(activeUi.cdSummaryLbl->text()));
  }
  html += reportImgTag(activeUi.cdPlot);

  if (includeConvergence) {
    const QSet<QString> methodSet(methods.begin(), methods.end());
    const QSet<QString> problemSet(problems.begin(), problems.end());
    const int kMaxConvergencePlots = 40;
    int shown = 0;

    html += QStringLiteral("<h2>Convergence Curves</h2>");
    for (int i = 0; i < outputRuns_.size(); ++i) {
      const OutputRunTab& t = outputRuns_[i];
      if (t.methodShort.compare("System", Qt::CaseInsensitive) == 0) continue;
      if (!methodSet.contains(t.methodShort) || !problemSet.contains(t.problemShort)) continue;
      if (!t.convergencePlot) continue;

      if (shown >= kMaxConvergencePlots) {
        html += QStringLiteral(
          "<p><i>... and more runs not shown (limit of %1 reached). "
          "Export the batch CSV for the complete data.</i></p>").arg(kMaxConvergencePlots);
        break;
      }

      html += QStringLiteral("<h3>%1 / %2 (dim=%3)</h3>")
        .arg(t.methodShort.toHtmlEscaped())
        .arg(t.problemShort.toHtmlEscaped())
        .arg(t.problemDim);
      html += reportImgTag(t.convergencePlot, 560);
      ++shown;
    }
    if (shown == 0) {
      html += QStringLiteral("<p><i>No convergence data loaded for the runs in this batch.</i></p>");
    }
  }

  html += QStringLiteral("</body></html>");
  return html;
}

QString MainWindow::buildSensitivityReportHtml(int tabIndex) const {
  const OutputRunTab* tab = outputTab(tabIndex);
  if (!tab || !tab->sensitivitySummaryTable || tab->sensitivitySummaryTable->rowCount() == 0) {
    return QString();
  }

  const QString generatedAt = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
  const QString param = (tab->sensitivityParamCombo && !tab->sensitivityParamCombo->currentText().isEmpty())
    ? tab->sensitivityParamCombo->currentText() : QStringLiteral("-");

  QString html;
  html += QStringLiteral("<html><body style=\"font-family:sans-serif;\">");
  html += QStringLiteral("<h1>Parameter Sensitivity Report</h1>");
  html += QStringLiteral("<p><b>Generated:</b> %1<br>").arg(generatedAt.toHtmlEscaped());
  html += QStringLiteral("<b>Method:</b> %1<br>").arg(tab->methodShort.toHtmlEscaped());
  html += QStringLiteral("<b>Problem:</b> %1 (dim=%2)<br>").arg(tab->problemShort.toHtmlEscaped()).arg(tab->problemDim);
  html += QStringLiteral("<b>Parameter:</b> %1</p>").arg(param.toHtmlEscaped());

  html += QStringLiteral("<h2>Results Table</h2>");
  html += reportTableHtml(tab->sensitivitySummaryTable);

  html += QStringLiteral("<h2>Parameter Sensitivity Chart</h2>");
  html += reportImgTag(tab->sensitivityPlot);

  if (tab->sensitivityLog) {
    const QString logText = tab->sensitivityLog->toPlainText().trimmed();
    if (!logText.isEmpty()) {
      html += QStringLiteral("<h2>Summary</h2>");
      html += QStringLiteral("<pre style=\"font-size:9pt;\">%1</pre>").arg(logText.toHtmlEscaped());
    }
  }

  html += QStringLiteral("</body></html>");
  return html;
}

bool MainWindow::writeReportDocument(const QString& html, const QString& filePath) const {
  QTextDocument doc;
  doc.setHtml(html);

  const QString lower = filePath.toLower();
  if (lower.endsWith(".odt")) {
    QTextDocumentWriter writer(filePath, "ODF");
    return writer.write(&doc);
  }

  // Default: PDF (native Qt printing, no extra dependencies).
  QString outPath = filePath;
  if (!lower.endsWith(".pdf")) outPath += ".pdf";

  QPrinter printer(QPrinter::HighResolution);
  printer.setOutputFormat(QPrinter::PdfFormat);
  printer.setOutputFileName(outPath);
  printer.setPageSize(QPageSize(QPageSize::A4));
  printer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout::Millimeter);
  doc.print(&printer);
  return true;
}

void MainWindow::onGenerateReport() {
  const auto reply = QMessageBox::question(
    this, "Generate report",
    "Include per-run convergence curves in the report?\n\n"
    "Recommended for small batches (a few methods/problems). For large batches, choose "
    "\"No\" to keep the report smaller and faster to generate.",
    QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
  const bool includeConvergence = (reply == QMessageBox::Yes);

  const QString html = buildBatchReportHtml(includeConvergence);
  if (html.isEmpty()) {
    QMessageBox::warning(this, "Generate report", "No batch results are available yet.");
    return;
  }

  const QString out = QFileDialog::getSaveFileName(
    this, "Generate batch report",
    QDir::homePath() + "/batch_report.pdf",
    "PDF Document (*.pdf);;Word-compatible Document (*.odt)");
  if (out.isEmpty()) return;

  if (!writeReportDocument(html, out)) {
    QMessageBox::warning(this, "Generate report", "Failed to write the report file.");
    return;
  }
  appendLog("Report generated: " + out);
}


// ---------------------------------------------------------------------
// U-score ("trial-based dominance") batch report.
// See include/uscore.h for the ranking algorithm (Qt-free, unit tested in
// tests/test_uscore.cpp) and CEC_2025_Comparison_Slides.pdf for the table
// layouts this reproduces as an .xlsx workbook: pp. 19-20's "Algorithm |
// Ranking" table -> the "Ranking" sheet; pp. 22-23's "Prob\Algo" table
// (incl. the trailing Sum/RS row) -> the "Per-Problem U-scores" sheet.
// ---------------------------------------------------------------------

// Reads every run's (evals, best_f) trace from a convergence CSV. Free
// function (no MainWindow state) so the U-score report can run it on a
// worker thread. Same accepted input as the former QTextStream/QString
// version (same header columns, blank lines skipped, rows with too few
// columns or non-numeric fields skipped, C-locale numbers), but it parses
// the bytes in place: ~8x faster, which matters because the report reads
// every convergence CSV of the experiment at the end of each batch.
static bool parseRunTracesFromConvergenceCsv(
    const QString& csvPath,
    std::vector<std::vector<optimsolution::uscore::EvalPoint>>& outRuns,
    QString* err) {
  using optimsolution::uscore::EvalPoint;

  outRuns.clear();
  if (err) err->clear();

  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly)) {
    if (err) *err = QString("Cannot open file '%1'.").arg(csvPath);
    return false;
  }
  const QByteArray data = f.readAll();
  f.close();
  const qsizetype dataSize = data.size();
  qsizetype pos = 0;
  if (dataSize >= 3 && uchar(data[0]) == 0xEF && uchar(data[1]) == 0xBB && uchar(data[2]) == 0xBF) {
    pos = 3; // UTF-8 BOM
  }
  auto nextLine = [&data, dataSize, &pos](QByteArrayView& line) -> bool {
    if (pos >= dataSize) return false;
    qsizetype e = data.indexOf('\n', pos);
    if (e < 0) e = dataSize;
    line = QByteArrayView(data.constData() + pos, e - pos);
    if (!line.isEmpty() && line.back() == '\r') line.chop(1);
    pos = e + 1;
    return true;
  };

  QByteArrayView headerView;
  if (!nextLine(headerView) || headerView.isEmpty()) {
    if (err) *err = "Empty CSV file.";
    return false;
  }
  const QString headerLine = QString::fromUtf8(headerView);

  const QStringList header = headerLine.split(',', Qt::KeepEmptyParts);
  auto colIndex = [&](const QString& name) -> int {
    for (int i = 0; i < header.size(); ++i) {
      if (header[i].trimmed().compare(name, Qt::CaseInsensitive) == 0) {
        return i;
      }
    }
    return -1;
  };

  const int cRun   = colIndex("run");
  const int cEvals = colIndex("evals");
  const int cBest  = colIndex("best_f");

  if (cRun < 0 || cEvals < 0 || cBest < 0) {
    if (err) *err = "CSV header missing one of: run, evals, best_f.";
    return false;
  }

  QMap<int, std::vector<EvalPoint>> byRun;

  const int maxCol = std::max(cRun, std::max(cEvals, cBest));
  QByteArrayView line;
  while (nextLine(line)) {
    line = line.trimmed();
    if (line.isEmpty()) continue;

    QByteArrayView fRun, fEvals, fBest;
    int col = 0;
    qsizetype fieldStart = 0;
    bool haveAll = false;
    const qsizetype len = line.size();
    for (qsizetype i = 0; i <= len; ++i) {
      if (i < len && line[i] != ',') continue;
      const QByteArrayView field = line.sliced(fieldStart, i - fieldStart).trimmed();
      if (col == cRun) fRun = field;
      if (col == cEvals) fEvals = field;
      if (col == cBest) fBest = field;
      if (col == maxCol) { haveAll = true; break; }
      ++col;
      fieldStart = i + 1;
    }
    if (!haveAll) continue;

    bool okRun = false, okEval = false, okBest = false;
    const int    run = fRun.toInt(&okRun);
    const qint64 ev  = fEvals.toLongLong(&okEval);
    const double bf  = fBest.toDouble(&okBest);
    if (!okRun || !okEval || !okBest) continue;

    byRun[run].push_back(EvalPoint{double(ev), bf});
  }

  if (byRun.isEmpty()) {
    if (err) *err = "No data rows parsed.";
    return false;
  }

  outRuns.reserve(byRun.size());
  for (auto it = byRun.begin(); it != byRun.end(); ++it) {
    std::vector<EvalPoint> trace = it.value();
    // Defensive: the CLI already appends rows in increasing evals order,
    // but sort explicitly so resolveTrial()'s "first point <= target"
    // scan is correct even if that were ever not the case.
    std::stable_sort(trace.begin(), trace.end(),
                      [](const EvalPoint& a, const EvalPoint& b) { return a.evals < b.evals; });
    outRuns.push_back(std::move(trace));
  }
  return true;
}

bool MainWindow::loadRunTracesFromConvergenceCsv(
    const QString& csvPath,
    std::vector<std::vector<optimsolution::uscore::EvalPoint>>& outRuns,
    QString* err) const {
  return parseRunTracesFromConvergenceCsv(csvPath, outRuns, err);
}

// Computes the U-score report and writes uscore_report_<timestamp>.xlsx.
// Runs on a worker thread (see exportUScoreReport()): it uses only its
// argument and free functions, never MainWindow state. Returns the line for
// the log ("" = nothing to report).
static QString computeAndWriteUScoreReport(const QMap<QString, QMap<QString, QString>> csvPaths) {
  using namespace optimsolution::uscore;

  const QStringList problemKeys = csvPaths.keys();
  if (problemKeys.isEmpty()) {
    return QString(); // no batch data available
  }

  struct ProblemRow {
    QString problemKey;
    std::vector<ProblemScore> scores; // one entry per method compared on this problem
  };
  std::vector<ProblemRow> rows;
  QStringList allMethodsSeen; // first-seen order, used as the report's column order

  for (const QString& problemKey : problemKeys) {
    const QMap<QString, QString> methodToCsv = csvPaths.value(problemKey);
    if (methodToCsv.size() < 2) {
      continue; // U-score needs at least two algorithms to compare on this problem
    }

    struct MethodTraceSet {
      QString methodShort;
      std::vector<std::vector<EvalPoint>> runs;
    };
    std::vector<MethodTraceSet> traceSets;
    std::vector<double> allFinals; // final best_f of every run of every method on this problem

    for (auto it = methodToCsv.begin(); it != methodToCsv.end(); ++it) {
      const QString methodShort = it.key();
      const QString csvPath = it.value().trimmed();
      if (csvPath.isEmpty()) continue;

      std::vector<std::vector<EvalPoint>> runs;
      QString perr;
      if (!parseRunTracesFromConvergenceCsv(csvPath, runs, &perr) || runs.empty()) {
        continue; // skip only this method; still compare whichever others parsed fine
      }
      for (const auto& trace : runs) {
        if (!trace.empty()) allFinals.push_back(trace.back().bestF);
      }
      traceSets.push_back(MethodTraceSet{methodShort, std::move(runs)});
      if (!allMethodsSeen.contains(methodShort)) {
        allMethodsSeen << methodShort;
      }
    }

    if (traceSets.size() < 2 || allFinals.empty()) {
      continue;
    }

    double target = 0.0, tol = 0.0;
    computeSharedTarget(allFinals, target, tol);

    std::vector<MethodTrials> methods;
    methods.reserve(traceSets.size());
    for (const auto& ts : traceSets) {
      MethodTrials mt;
      mt.methodName = ts.methodShort.toStdString();
      mt.trials.reserve(ts.runs.size());
      for (const auto& trace : ts.runs) {
        mt.trials.push_back(resolveTrial(trace, target, tol));
      }
      methods.push_back(std::move(mt));
    }

    ProblemRow row;
    row.problemKey = problemKey;
    row.scores = computeUScoreForProblem(methods);
    rows.push_back(std::move(row));
  }

  if (rows.empty()) {
    return QString(); // e.g. every batch problem only had a single method -- nothing to rank
  }

  std::vector<std::vector<ProblemScore>> perProblemScores;
  perProblemScores.reserve(rows.size());
  for (const auto& r : rows) perProblemScores.push_back(r.scores);
  const std::vector<OverallScore> overall = aggregateUScore(perProblemScores);

  // Write the report next to the batch's own convergence CSVs (same
  // directory as whichever CSV path we happen to find first).
  QString outDir;
  for (const QString& problemKey : problemKeys) {
    const QMap<QString, QString> methodToCsv = csvPaths.value(problemKey);
    for (auto it = methodToCsv.begin(); it != methodToCsv.end() && outDir.isEmpty(); ++it) {
      if (!it.value().trimmed().isEmpty()) {
        outDir = QFileInfo(it.value()).absolutePath();
      }
    }
    if (!outDir.isEmpty()) break;
  }
  if (outDir.isEmpty()) outDir = QDir::homePath();

  const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
  const QString outPath = outDir + "/uscore_report_" + stamp + ".xlsx";

  // --- Sheet 1: "About" -- methodology / citation, replacing the '#'
  // comment header lines a CSV would have used. ---
  QVector<QVector<QVariant>> aboutGrid;
  aboutGrid.push_back({QVariant(QString("U-score (trial-based dominance) report"))});
  aboutGrid.push_back({QVariant(QString(
      "Methodology: K.V. Price, A. Kumar, P.N. Suganthan, \"Trial-based dominance for "
      "comparing both the speed and accuracy of stochastic optimizers with standard "
      "non-parametric tests\", Swarm and Evolutionary Computation, 78, 101287, 2023."))});
  aboutGrid.push_back({QVariant(QString(
      "Table layout follows: Kangjia Qiao et al., \"Performance comparison of CEC 2025 "
      "competition entries on numerical optimization considering accuracy and speed\" "
      "(see the 'Ranking' and 'Per-Problem U-scores' sheets, matching pp. 19-23 of that "
      "presentation)."))});
  aboutGrid.push_back({});
  aboutGrid.push_back({QVariant(QString("Per problem: higher U-score = better; rank 1 = best algorithm on that problem."))});
  aboutGrid.push_back({QVariant(QString("Overall: RS = sum of per-problem ranks; lower RS = better; Ranking 1 = best."))});
  aboutGrid.push_back({});
  aboutGrid.push_back({QVariant(QString("Generated: %1")
      .arg(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss")))});
  aboutGrid.push_back({QVariant(QString("Problems compared: %1   |   Methods compared: %2")
      .arg(rows.size()).arg(allMethodsSeen.size()))});

  // --- Sheet 2: "Ranking" -- mirrors the CEC-2025-slides "Algorithm |
  // Ranking" tables (pp. 19-20), sorted best (1) first. ---
  std::vector<OverallScore> overallSorted = overall;
  std::stable_sort(overallSorted.begin(), overallSorted.end(),
                    [](const OverallScore& a, const OverallScore& b) { return a.overallRank < b.overallRank; });

  QVector<QVector<QVariant>> rankGrid;
  rankGrid.push_back({QVariant(QString("Algorithm")), QVariant(QString("Ranking")), QVariant(QString("Sum (U-score)")), QVariant(QString("RS (rank-sum)"))});
  for (const auto& o : overallSorted) {
    rankGrid.push_back({
      QVariant(QString::fromStdString(o.methodName)),
      QVariant(o.overallRank),
      QVariant(qint64(std::llround(o.sumUScore))),
      QVariant(qint64(std::llround(o.sumRank)))
    });
  }

  // --- Sheet 3: "Per-Problem U-scores" -- mirrors the CEC-2025-slides
  // "Prob\Algo" tables (pp. 22-23): one row per problem, one U-score/rank
  // column pair per method, and a final "Sum / RS" row (same column
  // layout, reusing the aggregated OverallScore totals). ---
  QMap<QString, const OverallScore*> overallByName;
  for (const auto& o : overall) overallByName[QString::fromStdString(o.methodName)] = &o;

  QVector<QVariant> probHeader;
  probHeader.push_back(QVariant(QString("Problem")));
  for (const QString& m : allMethodsSeen) {
    probHeader.push_back(QVariant(m + "_uscore"));
    probHeader.push_back(QVariant(m + "_rank"));
  }
  QVector<QVector<QVariant>> probGrid;
  probGrid.push_back(probHeader);

  for (const auto& row : rows) {
    QVector<QVariant> line;
    line.push_back(QVariant(row.problemKey));
    QMap<QString, const ProblemScore*> byName;
    for (const auto& s : row.scores) byName[QString::fromStdString(s.methodName)] = &s;
    for (const QString& m : allMethodsSeen) {
      auto found = byName.find(m);
      if (found != byName.end()) {
        const ProblemScore* s = found.value();
        line.push_back(QVariant(qint64(std::llround(s->uScore))));
        line.push_back(QVariant(s->rankOnProblem));
      } else {
        line.push_back(QVariant());
        line.push_back(QVariant());
      }
    }
    probGrid.push_back(line);
  }

  // Final "Sum / RS" row, same two-columns-per-method layout as the data
  // rows above, so each method's total lands directly under its own
  // U-score/rank columns -- exactly like the "sum/RS" row at the bottom
  // of the slides' per-problem tables.
  {
    QVector<QVariant> sumLine;
    sumLine.push_back(QVariant(QString("Sum / RS")));
    for (const QString& m : allMethodsSeen) {
      auto it2 = overallByName.find(m);
      if (it2 != overallByName.end()) {
        const OverallScore* o = it2.value();
        sumLine.push_back(QVariant(qint64(std::llround(o->sumUScore))));
        sumLine.push_back(QVariant(qint64(std::llround(o->sumRank))));
      } else {
        sumLine.push_back(QVariant());
        sumLine.push_back(QVariant());
      }
    }
    probGrid.push_back(sumLine);
  }

  QVector<XlsxSheet> sheets;
  sheets.push_back(XlsxSheet{"About", aboutGrid});
  sheets.push_back(XlsxSheet{"Ranking", rankGrid});
  sheets.push_back(XlsxSheet{"Per-Problem U-scores", probGrid});

  QString xerr;
  if (!writeMultiSheetXlsxNative(outPath, sheets, &xerr)) {
    return QString("U-score report: could not write '%1' (%2).").arg(outPath, xerr);
  }

  return QString("U-score report written: %1 (%2 problem(s), %3 method(s) compared).")
           .arg(outPath)
           .arg(rows.size())
           .arg(allMethodsSeen.size());
}

void MainWindow::exportUScoreReport() {
  using namespace optimsolution::uscore;

  // [global] uscore_report master switch: only export the uscore_report_*.xlsx
  // file when uscore_report = 1 in optimsolution.cfg (default: disabled).
  // NOTE: ConfigFile::value() returns the raw text after "=", including any
  // inline "; ..." comment (e.g. "1                    ; Export ..."), so the
  // comment must be stripped before comparing against "1".
  if (!cfg_) return;
  QString uscoreReportRaw = cfg_->value("global", "uscore_report", "0");
  int cpos = uscoreReportRaw.indexOf(';'); if (cpos >= 0) uscoreReportRaw = uscoreReportRaw.left(cpos);
  cpos = uscoreReportRaw.indexOf('#');      if (cpos >= 0) uscoreReportRaw = uscoreReportRaw.left(cpos);
  if (uscoreReportRaw.trimmed() != "1") {
    return;
  }

  // g_liveBatchCsvPaths: problemKey -> methodShort -> convergence CSV path of
  // the batch that just ran (see finishBatchPostProcess()). A COPY is handed
  // to a worker thread: the report re-reads every convergence CSV of the
  // experiment, which used to run here on the GUI thread and froze the
  // window at "99%" at the end of every batch (for minutes on large
  // experiments). The batch is already finished and its results are on
  // screen; the report line appears in the log when the file is written.
  const QMap<QString, QMap<QString, QString>> csvPaths = g_liveBatchCsvPaths;
  if (csvPaths.isEmpty()) {
    return; // no batch data available (e.g. called outside of a batch run)
  }

  appendLog("U-score report: computing in the background...");
  auto* watcher = new QFutureWatcher<QString>(this);
  connect(watcher, &QFutureWatcher<QString>::finished, this, [this, watcher]() {
    const QString msg = watcher->result();
    watcher->deleteLater();
    appendLog(msg.isEmpty() ? QString("U-score report: nothing to compare (needs at least two methods on a problem).")
                            : msg);
  });
  watcher->setFuture(QtConcurrent::run([csvPaths]() -> QString {
    try {
      return computeAndWriteUScoreReport(csvPaths);
    } catch (const std::exception& e) {
      return QString("U-score report: skipped (%1).").arg(e.what());
    } catch (...) {
      return QString("U-score report: skipped (unknown error).");
    }
  }));
}

} // namespace optimsolution_gui
