// MainWindow_ProcessSettings.cpp
// CLI process management and I/O, log rendering, settings tables (load/save/edit), sensitivity configuration tables.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"

namespace optimsolution_gui {
void MainWindow::startCliProcess(const QStringList& args) {
  QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found", "optimsolution executable was not found next to the GUI or in build/Debug.");
    return;
  }

  if (proc_) {
    proc_->deleteLater();
    proc_ = nullptr;
  }

  proc_ = new QProcess(this);
  proc_->setProgram(cli);

  // Build a runtime config snapshot for this run so GUI edits act as overrides without
  // modifying the on-disk optimsolution.cfg. If the CLI supports "--config", the snapshot
  // is passed explicitly; otherwise the CLI is executed from a dedicated folder containing
  // an "optimsolution.cfg" copy.
  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }

  QString runtimeCfgPath;
  QString runtimeWd = baseWd;
  QStringList finalArgs = args;

  if (cliSupportsConfigArg_ < 0) {
    cliSupportsConfigArg_ = detectCliSupportsConfigArg(cli) ? 1 : 0;
  }

  // In sensitivity mode, writeSensitivityConfigFromUi() was already called by
  // startNextSensJob() with the table in the correct state. Calling it again
  // from startCliProcess (after outputTabs_->setCurrentIndex fires signals)
  // risks clearing params/values. Skip it here for sensitivity mode.
  const bool isSensitivityMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));

  if (cfg_ && cfg_->isLoaded()) {
    // Clear convergence plot at the start of each run (active output tab).
    if (auto* tab = activeOutputTab()) {
      if (tab->convergenceInfo) {
        const QString m = tab->methodShort.isEmpty() ? "-" : tab->methodShort;
        const QString p = tab->problemShort.isEmpty() ? "-" : tab->problemShort;
        const QString dimInfo = (tab->problemDim > 0) ? QString(" | D=%1").arg(tab->problemDim) : QString();
        tab->convergenceInfo->setText(QString("Method: %1 | Problem: %2%3\nNo convergence data loaded.")
                                        .arg(m)
                                        .arg(p)
                                        .arg(dimInfo));
      }
      if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    }

    // Optional runtime-only override: force CSV convergence so the GUI can plot best-per-iteration.
    const bool forceCsv = batchActive_ || (forceConvergenceCsvChk_ && forceConvergenceCsvChk_->isChecked());
    const auto globalMap = cfg_->sectionMap("global");
    const bool hadCsvEnable = globalMap.contains("csv_enable");
    const bool hadCsvConv   = globalMap.contains("csv_convergence");
    const QString prevCsvEnable = cfg_->value("global", "csv_enable");
    const QString prevCsvConv   = cfg_->value("global", "csv_convergence");

    auto applyForceCsv = [&](){
      if (!forceCsv) return;
      cfg_->setValue("global", "csv_enable", "1");
      cfg_->setValue("global", "csv_convergence", "1");
    };

    auto restoreForceCsv = [&](){
      if (!forceCsv) return;
      if (hadCsvEnable) cfg_->setValue("global", "csv_enable", prevCsvEnable);
      else cfg_->removeKey("global", "csv_enable");
      if (hadCsvConv) cfg_->setValue("global", "csv_convergence", prevCsvConv);
      else cfg_->removeKey("global", "csv_convergence");
    };

    // In sensitivity mode, forcibly clear the [sensitivity] section right before
    // saving so that any signal-driven config writes that may have re-enabled it
    // (e.g. from outputTabs_->setCurrentIndex() firing after startNextSensJob's
    // own cleanup) cannot cause the CLI to run its own sweep and produce a single
    // N=runs*values aggregate CSV that all GUI-driven jobs would then re-read.
    auto forceClearSensitivityForCliRun = [&]() {
      if (!isSensitivityMode || !cfg_ || !cfg_->isLoaded()) return;
      cfg_->setValue("sensitivity", "enabled", "0");
      cfg_->setValue("sensitivity", "enable",  "0");
      cfg_->setValue("sensitivity", "params",  "");
      const auto sm = cfg_->sectionMap("sensitivity");
      for (auto it = sm.cbegin(); it != sm.cend(); ++it) {
        const QString k = it.key().trimmed();
        if (k.startsWith("values."))
          cfg_->removeKey("sensitivity", k);
      }
    };

    if (cliSupportsConfigArg_ == 1) {
      runtimeCfgPath = QDir(baseWd).filePath("optimsolution_gui_merged.cfg");
      applyForceCsv();
      if (!isSensitivityMode) writeSensitivityConfigFromUi();
      forceClearSensitivityForCliRun();
      const bool saved = cfg_->save(runtimeCfgPath);
      restoreForceCsv();
      if (saved) {
        finalArgs.clear();
        finalArgs << "--config" << runtimeCfgPath;
        finalArgs << args;
      } else {
        runtimeCfgPath.clear();
      }
      runtimeWd = baseWd;
    } else {
      runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
      QDir().mkpath(runtimeWd);
      runtimeCfgPath = QDir(runtimeWd).filePath("optimsolution.cfg");
      applyForceCsv();
      if (!isSensitivityMode) writeSensitivityConfigFromUi();
      forceClearSensitivityForCliRun();
      const bool saved = cfg_->save(runtimeCfgPath);
      restoreForceCsv();
      if (!saved) {
        runtimeCfgPath.clear();
      }
    }
  }

  lastRuntimeWorkingDir_ = runtimeWd;
  lastRuntimeCfgPath_ = runtimeCfgPath;

  if (auto* tab = activeOutputTab()) {
    tab->runtimeWorkingDir  = runtimeWd;
    tab->runtimeCfgPath     = runtimeCfgPath;
    tab->runStartMsecsUtc   = QDateTime::currentMSecsSinceEpoch();
    tab->convLoadedCsvPath.clear(); // invalidate cached CSV so next switch re-scans
  }

  proc_->setArguments(finalArgs);
  proc_->setWorkingDirectory(runtimeWd);

  if (!runtimeCfgPath.isEmpty()) appendLog(QString("Using config: %1").arg(runtimeCfgPath));
  appendLog(QString("Working directory: %1").arg(runtimeWd));

  connect(proc_, &QProcess::readyReadStandardOutput, this, &MainWindow::onProcessReadyReadStdout);
  connect(proc_, &QProcess::readyReadStandardError,  this, &MainWindow::onProcessReadyReadStderr);
  connect(proc_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, &MainWindow::onProcessFinished);
  connect(proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError err){
    CrashLog::append(QString("Process: errorOccurred=%1").arg(static_cast<int>(err)));
    appendLog(QString("Process error: %1").arg(static_cast<int>(err)));
  });

  ansiCarryStdout_.clear();
  ansiCarryStderr_.clear();

  connect(proc_, &QProcess::started, this, [this, runtimeWd](){
    // Running state
    if (busySpinner_) busySpinner_->start();
    updateRunBtn(true);
    // Disable entire Selection and Settings areas. (Code Wizard now lives
    // as a Settings tab, so settingsBox_->setEnabled() already covers it.)
    if (selectionBox_) selectionBox_->setEnabled(false);
    if (settingsBox_) settingsBox_->setEnabled(false);
    // runBtn_ and busySpinner_ must stay active (Stop + spinner).
    if (runBtn_) runBtn_->setEnabled(true);
    if (busySpinner_) busySpinner_->setEnabled(true);
    if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(false);
    if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(false);
    if (clearCsvBtn_) clearCsvBtn_->setEnabled(false);

    // Start elapsed timer only once — not reset between batch/sensitivity jobs.
    if (!runElapsedTimer_ || !runElapsedTimer_->isActive()) {
      runElapsed_.start();
      if (!runElapsedTimer_) {
        runElapsedTimer_ = new QTimer(this);
        connect(runElapsedTimer_, &QTimer::timeout, this, [this](){
          if (!outputElapsedLbl_) return;
          const qint64 ms = runElapsed_.elapsed();
          const int h  = int(ms / 3600000);
          const int m  = int((ms % 3600000) / 60000);
          const int s  = int((ms % 60000) / 1000);
          outputElapsedLbl_->setText(h > 0
            ? QString("%1:%2:%3").arg(h).arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0'))
            : QString("%1:%2").arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0')));
        });
      }
      runElapsedTimer_->start(1000);
      if (outputElapsedLbl_) { outputElapsedLbl_->setVisible(true); outputElapsedLbl_->setText("0:00"); }
    }

    // Set status label — only for non-batch modes; batch sets it in startNextBatchJob.
    const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
    if (mode != 1) {
      const QString modeStr = (mode == 2) ? "Sensitivity" :
                              (mode == 3) ? "Problem sensitivity" : "Running";
      if (outputStatusLbl_) outputStatusLbl_->setText(modeStr + "...");
    }

    // Output progress bar: start determinate.
    if (outputProgressBar_) {
      outputProgressDeterminate_ = false;
      outputProgressValue_ = 0;
      outputProgressMaximum_ = 0;
      progressWidget_->setVisible(true);
      outputProgressBar_->setRange(0, 0); // indeterminate
      outputProgressBar_->setValue(0);
      const bool isSens = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
      outputProgressBar_->setFormat(isSens ? "Sensitivity running..." : "Running...");
    }

    // For sensitivity runs: start CSV-based progress polling.
    if (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3) && sensCsvExpectedRows_ > 0) {
      // Resolve relative CSV path against the working directory.
      QString pollPath = sensCsvPollPath_;
      if (!QFileInfo(pollPath).isAbsolute())
        pollPath = QDir(runtimeWd).filePath(pollPath);
      // Read runsPerPoint from config — check common key names across sections.
      int runsPerPoint = 30; // sensible default
      if (cfg_ && cfg_->isLoaded()) {
        const QStringList sections = {"global", "settings", "run", ""};
        const QStringList keys     = {"runs", "num_runs", "n_runs", "repeats", "replications"};
        bool found = false;
        for (const QString& sec : sections) {
          for (const QString& key : keys) {
            bool ok = false;
            const int v = cfg_->value(sec, key, "-1").toInt(&ok);
            if (ok && v > 0) { runsPerPoint = v; found = true; break; }
          }
          if (found) break;
        }
      }
      startSensCsvPoll(pollPath, sensCsvExpectedRows_, runsPerPoint);
    }

    // Reset single-run output throttling state for this process.
    if (!batchActive_) {
      singleUiPendingLog_.clear();
      singleProgressThrottle_.invalidate();
      if (singleUiFlushTimer_) {
        singleUiFlushTimer_->stop();
      }
    }

    // Initialize progress tracking: read runs from config, reset current run number.
    {
      // Reload cfg from disk to pick up external edits.
      if (cfg_ && !settingsPath_.isEmpty()) cfg_->load(settingsPath_);
      currentRunNumber_ = 1;
      singleRunsPerJob_ = globalRunsFromSettings();
      progressPollDir_  = runtimeWd;
      progressPollStartTime_ = QDateTime::currentDateTimeUtc();

      // Start polling the convergence CSV for real-time run progress.
      if (!progressPollTimer_) {
        progressPollTimer_ = new QTimer(this);
        connect(progressPollTimer_, &QTimer::timeout, this, &MainWindow::onProgressPollTick);
      }
      progressPollTimer_->start(500);

      // Progress bar: determinate from the start.
      if (outputProgressBar_) {
        outputProgressBar_->setRange(0, 100);
        outputProgressDeterminate_ = true;
        updateSingleBatchProgress();
      }
    }
  });

  CrashLog::append(QString("Process: start program=%1 args=%2").arg(cli).arg(finalArgs.join(" ")));
  proc_->start();
}

void MainWindow::onProcessReadyReadStdout() {
  QByteArray b = proc_->readAllStandardOutput();
  QString chunk = QString::fromUtf8(b);
  QString cleaned = AnsiStrip::stripStreaming(chunk, ansiCarryStdout_);
  if (cleaned.isEmpty()) {
    return;
  }

  if (batchActive_) {
    batchHandleProcessOutput(cleaned);
    return;
  }

  singleHandleProcessOutput(cleaned);
}

void MainWindow::onProcessReadyReadStderr() {
  QByteArray b = proc_->readAllStandardError();
  QString chunk = QString::fromUtf8(b);
  QString cleaned = AnsiStrip::stripStreaming(chunk, ansiCarryStderr_);
  if (cleaned.isEmpty()) {
    return;
  }

  // In sensitivity mode, capture stderr to CrashLog for diagnosis.
  if (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3)) {
    const QString trimmed = cleaned.trimmed();
    if (!trimmed.isEmpty()) {
      CrashLog::append(QString("[sens-stderr] %1").arg(trimmed.left(500)));
    }
  }

  if (batchActive_) {
    batchHandleProcessOutput(cleaned);
    return;
  }

  singleHandleProcessOutput(cleaned);
}

void MainWindow::updateOutputProgressFromText(const QString& text) {
  if (!outputProgressBar_) return;
  if (!outputProgressBar_->isVisible()) return;

  // Accept either raw chunks or already-normalized text. Split robustly.
  const QStringList lines = text.split(QRegularExpression(R"(
|
|
)"));

  // Heuristics:
  // 1) If a "progress ... %" token exists, prefer it.
  // 2) Sensitivity-specific: "point X/Y", "combo X/Y", "sweep X/Y", "[sensitivity] X/Y".
  // 3) Otherwise look for "iter/eval/run x/y" patterns.
  // The GUI remains indeterminate if nothing parsable is found.
  static const QRegularExpression rePct(QStringLiteral(R"((?:^|\b)(?:progress|completed|completion)\b[^0-9%]*([0-9]{1,3}(?:\.[0-9]+)?)\s*%)"),
                                       QRegularExpression::CaseInsensitiveOption);
  static const QRegularExpression reSens(QStringLiteral(
    R"((?:(?:\[sensitivity\]|\bsensitivity\b|\bpoint\b|\bcombo\b|\bcombination\b|\bsweep\b|\bconfig\b|\bparam(?:eter)?\b)[^0-9]*([0-9]+)\s*(?:/|of)\s*([0-9]+)))"),
    QRegularExpression::CaseInsensitiveOption);
  static const QRegularExpression reIter(QStringLiteral(R"((?:^|\b)(?:iter(?:ation)?|gen|step)\b[^0-9]*([0-9]+)\s*(?:/|of)\s*([0-9]+))"),
                                        QRegularExpression::CaseInsensitiveOption);
  static const QRegularExpression reEval(QStringLiteral(R"((?:^|\b)(?:eval(?:s|uations)?|fe?s|evaluations)\b[^0-9]*([0-9]+)\s*(?:/|of)\s*([0-9]+))"),
                                        QRegularExpression::CaseInsensitiveOption);
  static const QRegularExpression reRun(QStringLiteral(R"((?:^|\b)(?:run|trial)\b[^0-9]*([0-9]+)\s*(?:/|of)\s*([0-9]+))"),
                                       QRegularExpression::CaseInsensitiveOption);

  const bool isSensRun = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));

  // Scan the last few lines in the chunk so the UI remains responsive.
  for (int i = lines.size() - 1; i >= 0 && i >= lines.size() - 25; --i) {
    const QString s = lines[i].trimmed();
    if (s.isEmpty()) continue;
    // Avoid interpreting "Success rate: 100%" as progress.
    if (s.startsWith("Success rate", Qt::CaseInsensitive)) continue;

    {
      const auto m = rePct.match(s);
      if (m.hasMatch()) {
        bool ok = false;
        const double pct = m.captured(1).toDouble(&ok);
        if (ok) {
          const int val = qBound(0, int(std::round(pct)), 100);
          outputProgressDeterminate_ = true;
          outputProgressValue_ = val;
          outputProgressMaximum_ = 100;
          outputProgressBar_->setRange(0, 100);
          outputProgressBar_->setValue(val);
          outputProgressBar_->setFormat(QString("%1%").arg(val));
          return;
        }
      }
    }

    auto applyXY = [&](const QString& label, const QRegularExpression& re){
      const auto m = re.match(s);
      if (!m.hasMatch()) return false;
      bool okA = false, okB = false;
      const int a = m.captured(1).toInt(&okA);
      const int b = m.captured(2).toInt(&okB);
      if (!okA || !okB || b <= 0) return false;
      const int v = qBound(0, a, b);
      outputProgressDeterminate_ = true;
      outputProgressValue_ = v;
      outputProgressMaximum_ = b;
      outputProgressBar_->setRange(0, b);
      outputProgressBar_->setValue(v);
      outputProgressBar_->setFormat(QString("%1 %2/%3").arg(label).arg(v).arg(b));
      return true;
    };

    if (isSensRun) {
      // In sensitivity mode: sensitivity-level progress takes priority over
      // inner iter/eval counters (which reflect a single run, not overall progress).
      if (applyXY("Sensitivity", reSens)) return;
      if (applyXY("Sensitivity", reRun))  return;
      if (applyXY("Iter",  reIter)) return;
      if (applyXY("Evals", reEval)) return;
    } else {
      if (applyXY("Iter",  reIter)) return;
      if (applyXY("Evals", reEval)) return;
      if (applyXY("Run",   reRun))  return;
      if (applyXY("Sensitivity", reSens)) return;
    }
  }
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus status) {
  Q_UNUSED(status);

  // Stop the convergence CSV poll timer and do one final read.
  if (progressPollTimer_) progressPollTimer_->stop();
  onProgressPollTick();

  // Update elapsed display (timer keeps running between batch/sensitivity jobs).
  if (outputElapsedLbl_) {
    const qint64 ms = runElapsed_.elapsed();
    const int h = int(ms / 3600000);
    const int m = int((ms % 3600000) / 60000);
    const int s = int((ms % 60000) / 1000);
    outputElapsedLbl_->setText(h > 0
      ? QString("%1:%2:%3").arg(h).arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0'))
      : QString("%1:%2").arg(m,2,10,QChar('0')).arg(s,2,10,QChar('0')));
  }

  const bool inSensQueue = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3) && !sensJobQueue_.isEmpty());

  // GUI-driven sensitivity: parse the summary CSV from this run, store result,
  // write output CSV when the group (problem/dim) is complete.
  if (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3) && !sensJobQueue_.isEmpty()) {

    const SensQueueJob curJob = sensJobQueue_[sensJobIndex_];
    const int groupTabIdx = sensGroupTabIdx_.value(curJob.groupId, activeOutputRunIndex_);
    const QString wd = (groupTabIdx >= 0 && groupTabIdx < int(outputRuns_.size()))
                       ? outputRuns_[groupTabIdx].runtimeWorkingDir
                       : lastRuntimeWorkingDir_;

    CrashLog::append(QString("[sens] Job %1/%2 finished: exitCode=%3 problem=%4 dim=%5 wd=%6")
                     .arg(sensJobIndex_+1).arg(sensJobQueue_.size())
                     .arg(exitCode).arg(curJob.problem).arg(curJob.dim).arg(wd));
    QString injStr;
    for (auto it = curJob.injectedParams.cbegin(); it != curJob.injectedParams.cend(); ++it)
      injStr += QString("%1=%2 ").arg(it.key(), it.value());
    CrashLog::append(QString("[sens] injectedParams: %1").arg(injStr));

    if (exitCode == 0 && !wd.isEmpty() && curJob.groupId < int(sensGroups_.size())) {
      // Find the *_summary.csv written by this specific run (newer than job start).
      QString summaryCsv;
      QStringList allFound;
      {
        QDirIterator it(wd, {"*_summary.csv"}, QDir::Files);
        QDateTime best;
        while (it.hasNext()) {
          const QString p = it.next();
          const QFileInfo fi(p);
          allFound << QString("%1 (mtime=%2)").arg(fi.fileName(), fi.lastModified().toUTC().toString(Qt::ISODate));
          if (fi.lastModified().toUTC() >= sensJobStartTime_.addSecs(-2)) {
            if (best.isNull() || fi.lastModified() > best) {
              best = fi.lastModified();
              summaryCsv = p;
            }
          }
        }
      }

      // Debug: log what we found.
      CrashLog::append(QString("[sens] sensJobStartTime=%1").arg(sensJobStartTime_.toString(Qt::ISODate)));
      CrashLog::append(QString("[sens] summary CSVs in wd: %1").arg(allFound.isEmpty() ? "(none)" : allFound.join("; ")));
      CrashLog::append(QString("[sens] selected: %1").arg(summaryCsv.isEmpty() ? "(none)" : summaryCsv));

      // Parse summary CSV into a result point.
      SensPointResult result;
      result.paramValues = curJob.injectedParams;
      if (!summaryCsv.isEmpty()) {
        parseSummaryCsvForSensPoint(summaryCsv, result);
        CrashLog::append(QString("[sens] parsed: valid=%1 mean_f=%2 runs=%3")
                         .arg(result.valid).arg(result.meanF).arg(result.runs));
      }

      SensGroup& group = sensGroups_[curJob.groupId];
      while (group.results.size() <= curJob.pointIdx)
        group.results.push_back({});
      group.results[curJob.pointIdx] = result;

      // Count jobs in this group and check if complete.
      int totalInGroup = 0;
      for (const auto& j : sensJobQueue_)
        if (j.groupId == curJob.groupId) ++totalInGroup;

      if (int(group.results.size()) >= totalInGroup) {
        // All points done for this group — write aggregated sensitivity CSV.
        writeSensGroupCsv(group, wd);
        const QString csvName = group.sweepParams.size() == 1
          ? QString("sensitivity_%1_%2_D%3.csv").arg(group.sweepParams[0], group.problem).arg(group.dim)
          : QString("sensitivity_%1_D%2.csv").arg(group.problem).arg(group.dim);
        const QString csvPath = QDir(wd).filePath(csvName);
        if (groupTabIdx >= 0 && groupTabIdx < int(outputRuns_.size()))
          outputRuns_[groupTabIdx].sensitivityCsvPath = csvPath;
        tryLoadSensitivityForTab(groupTabIdx >= 0 ? groupTabIdx : activeOutputRunIndex_);
      }

      // Restore injected param to original value (every point, not just group-end).
      if (cfg_ && cfg_->isLoaded()) {
        if (sensIsProblemMode_) {
          // Problem sensitivity: restore to the problem section.
          const QString problemSection = curJob.problem;
          for (auto it = curJob.injectedParams.cbegin(); it != curJob.injectedParams.cend(); ++it) {
            const QString orig = sensOrigProblemParams_.value(it.key());
            if (!orig.isEmpty()) cfg_->setValue(problemSection, it.key(), orig);
            else cfg_->removeKey(problemSection, it.key());
          }
        } else {
          // Method sensitivity: restore to the method section.
          for (auto it = curJob.injectedParams.cbegin(); it != curJob.injectedParams.cend(); ++it) {
            const QString orig = sensOrigMethodParams_.value(it.key());
            if (!orig.isEmpty()) cfg_->setValue(group.method, it.key(), orig);
            else cfg_->removeKey(group.method, it.key());
          }
        }
      }
    } // end if (exitCode == 0 ...)

    if (singleUiFlushTimer_) singleUiFlushTimer_->stop();
    flushSingleUiLog();

    if (sensJobIndex_ < int(sensJobQueue_.size()) - 1) {
      ++sensJobIndex_;
      QTimer::singleShot(100, this, &MainWindow::startNextSensJob);
      return;
    }

    // All sensitivity jobs done — clean up.
    sensJobQueue_.clear();
    sensJobIndex_ = 0;
    sensGroups_.clear();
    sensOrigMethodParams_.clear();
    sensOrigProblemParams_.clear();
    sensIsProblemMode_ = false;
    sensGroupTabIdx_.clear();
  }
  if (sensCsvPollTimer_ && sensCsvPollTimer_->isActive()) {
    onSensCsvPollTick(); // synchronous final read
  }
  stopSensCsvPoll();

  // flush any carry
  if (!ansiCarryStdout_.isEmpty()) appendLog(AnsiStrip::strip(ansiCarryStdout_));
  if (!ansiCarryStderr_.isEmpty()) appendLog(AnsiStrip::strip(ansiCarryStderr_));
  ansiCarryStdout_.clear();
  ansiCarryStderr_.clear();

  // Ensure any buffered single-run output is flushed before emitting the final status line.
  if (!batchActive_) {
    if (singleUiFlushTimer_) singleUiFlushTimer_->stop();
    flushSingleUiLog();
    singleProgressThrottle_.invalidate();
  }

  appendLog(QString("Process finished with exit code %1").arg(exitCode));

  if (!batchActive_) {
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

    // Restore idle state
    if (busySpinner_) busySpinner_->stop();

    // Stop elapsed timer and show final time.
    if (runElapsedTimer_) runElapsedTimer_->stop();
    if (outputElapsedLbl_) outputElapsedLbl_->setVisible(false);
    if (outputStatusLbl_) {
      const qint64 ms = runElapsed_.elapsed();
      const int h = int(ms / 3600000);
      const int mn = int((ms % 3600000) / 60000);
      const int s  = int((ms % 60000) / 1000);
      const QString elapsed = h > 0
        ? QString("%1:%2:%3").arg(h).arg(mn,2,10,QChar('0')).arg(s,2,10,QChar('0'))
        : QString("%1:%2").arg(mn,2,10,QChar('0')).arg(s,2,10,QChar('0'));
      outputStatusLbl_->setText(QString("Completed. Total time: %1").arg(elapsed));
    }
    updateRunBtn(false);
    if (selectionBox_) selectionBox_->setEnabled(true);
    if (settingsBox_) settingsBox_->setEnabled(true);
    if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(true);
    if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(true);
    if (clearCsvBtn_) clearCsvBtn_->setEnabled(true);
    updateDimUiForProblem(problemNameFromDisplay(problemBox_->currentText()));

    // Output progress bar: stop updates and report the end state.
    if (outputProgressBar_) {
      const bool isSens = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
      outputProgressBar_->setRange(0, 100);
      outputProgressBar_->setValue(exitCode == 0 ? 100 : 0);
      if (isSens)
        outputProgressBar_->setFormat(exitCode == 0 ? "100%" : "Failed");
      else
        outputProgressBar_->setFormat(exitCode == 0 ? "Finished" : "Failed");
      // Keep visible briefly, then hide.
      QTimer::singleShot(isSens ? 3000 : 1500, this, [this](){
        if (outputProgressBar_) progressWidget_->setVisible(false);
      });
    }
  } else {
    // During batch: keep the UI in the running state and keep the progress visible.
    if (outputProgressBar_) progressWidget_->setVisible(true);
  }

  // Auto-retry: if we passed a dimension but the problem is fixed-dimension, the CLI prints a diagnostic and exits(1).
  // Detect that message and re-run once without the dimension.
  if (exitCode == 1 && lastHadDimension_ && !lastAutoRetriedFixedDim_) {
    const QRegularExpression reFixed(QStringLiteral(R"(fixed dimension of\s+([0-9]+))"),
                                    QRegularExpression::CaseInsensitiveOption);
    const auto mm = reFixed.match(processTextBuffer_);
    if (mm.hasMatch()) {
      const int fixedDim = mm.captured(1).toInt();
      lastAutoRetriedFixedDim_ = true;

      // Update UI to reflect fixed dimension for this problem.
      currentFixedDim_ = fixedDim;
      dimSpin_->setValue(fixedDim);
      dimSpin_->setEnabled(false);

      appendLog(QString("Detected fixed dimension (%1). Re-running without an explicit dimension...").arg(fixedDim));

      // Build args without the last dimension
      QStringList args = lastArgs_;
      if (!args.isEmpty()) args.removeLast(); // remove dimension
      lastArgs_ = args;
      lastHadDimension_ = false;
      processTextBuffer_.clear();

      appendLog("Running: " + cliPath() + " " + args.join(' '));
      startCliProcess(args);
      return;
    }
  }

  // Try loading convergence CSV if available. Skip sensitivity load for queue mode
  // (handled by the queue block above with the correct per-group CSV path).
  if (exitCode == 0 && !batchActive_) {
    tryLoadConvergenceForTab(activeOutputRunIndex_);
    tryLoadLandscapeForTab(activeOutputRunIndex_);
    if (!inSensQueue) tryLoadSensitivityForTab(activeOutputRunIndex_);
    tryLoadPortfolioForTab(activeOutputRunIndex_);
  }

  // Batch progression (v41): CSV scanning/parsing is performed off the GUI thread.
  if (batchActive_) {
    if (batchPostInFlight_) {
      // A previous job is still being post-processed; do not advance the queue twice.
      return;
    }
    if (batchJobIndex_ < 0 || batchJobIndex_ >= batchQueue_.size()) {
      finalizeBatch();
      return;
    }

    BatchJob job = batchQueue_[batchJobIndex_];
    // Extract "Time per run (mean): X.XXX s" from CLI output and attach to job.
    {
      const QString tStr = lastSummaryField(processTextBuffer_, "Time per run (mean):");
      if (!tStr.isEmpty()) {
        bool ok = false;
        const double t = tStr.split(' ').first().trimmed().toDouble(&ok);
        if (ok && t > 0.0) job.timePerRunSecs = t;
      }
    }
    startBatchPostProcessAsync(job, exitCode, lastRuntimeWorkingDir_);
    return;
  }


}

void MainWindow::appendLog(const QString& text) {
  auto* tab = activeOutputTab();
  if (!tab || !tab->log) return;
  QTextEdit* log = tab->log;

  // Normalize line endings, then append non-empty lines only (no blank lines).
  // The GUI applies lightweight styling to the CLI's built-in summary block, and does
  // not add a second synthetic "RESULT SUMMARY" section.
  QString t = text;
  t.replace("\r\n", "\n");
  t.replace('\r', '\n');

  const QStringList lines = t.split('\n');
  QTextCursor c = log->textCursor();
  c.movePosition(QTextCursor::End);

  // Theme-aware defaults
  const QColor plainColor = log->palette().color(QPalette::Text);

  // Summary styling colors
  const QColor cTitle(0, 92, 185);   // blue
  const QColor cOk(0, 128, 0);       // green
  const QColor cErr(180, 0, 0);      // red
  const QColor cWarn(196, 109, 0);   // orange
  const QColor cSep(90, 90, 90);     // gray

  auto mkFmt = [&](const QColor& color, bool bold) {
    QTextCharFormat fmt;
    fmt.setForeground(QBrush(color));
    fmt.setFontWeight(bold ? QFont::Bold : QFont::Normal);
    return fmt;
  };

  const QTextCharFormat fmtPlain = mkFmt(plainColor, false);
  const QTextCharFormat fmtBlueB = mkFmt(cTitle, true);
  const QTextCharFormat fmtGreenB = mkFmt(cOk, true);
  const QTextCharFormat fmtRedB = mkFmt(cErr, true);
  const QTextCharFormat fmtOrangeB = mkFmt(cWarn, true);
  const QTextCharFormat fmtGray = mkFmt(cSep, false);

  auto insertLine = [&](const QList<QPair<QString, QTextCharFormat>>& parts) {
    for (const auto& p : parts) {
      c.setCharFormat(p.second);
      c.insertText(p.first);
    }
    c.setCharFormat(fmtPlain);
    c.insertText("\n");
  };

  auto insertPlain = [&](const QString& s) {
    insertLine({{s, fmtPlain}});
  };

  auto insertStyled = [&](const QString& s, const QTextCharFormat& fmt) {
    insertLine({{s, fmt}});
  };

  auto looksLikeRule = [](const QString& s, QChar ch) -> bool {
    if (s.size() < 8) return false;
    int cnt = 0;
    for (QChar cc : s) if (cc == ch) ++cnt;
    return cnt >= (s.size() * 8) / 10; // 80% same character
  };

  auto looksLikeHeading = [&](const QString& s) -> bool {
    const QString t = s.trimmed();
    if (t.size() < 8) return false;
    // CLI headings are typically dash/equal padded with a word group in the center.
    if (looksLikeRule(t, '=') || looksLikeRule(t, '-')) return true;
    if ((t.startsWith("===") && t.endsWith("===")) || (t.startsWith("---") && t.endsWith("---"))) return true;
    // Common summary titles.
    if (t.contains("RUN SUMMARY", Qt::CaseInsensitive)) return true;
    if (t.contains("RESULT SUMMARY", Qt::CaseInsensitive)) return true;
    return false;
  };

  auto splitLabelValue = [](const QString& s) -> QPair<QString, QString> {
    const int idx = s.indexOf(':');
    if (idx < 0) return {s, QString()};
    return {s.left(idx + 1), s.mid(idx + 1).trimmed()};
  };

  for (const QString& line : lines) {
    const QString s = line.trimmed();
    if (s.isEmpty()) continue;

    // Section titles / separators in the CLI summary
    if (looksLikeHeading(s)) {
      insertStyled(s, fmtBlueB);
      continue;
    }

    // Highlight key fields
    if (s.startsWith("Status:", Qt::CaseInsensitive)) {
      const auto lv = splitLabelValue(s);
      const QString vUpper = lv.second.toUpper();
      if (vUpper.contains("SUCCESS")) {
        insertLine({{lv.first + " ", fmtPlain}, {lv.second, fmtGreenB}});
      } else {
        insertLine({{lv.first + " ", fmtPlain}, {lv.second, fmtRedB}});
      }
      continue;
    }
    if (s.startsWith("Best f", Qt::CaseInsensitive)) {
      // Best objective value: value in red, label in blue.
      const auto lv = splitLabelValue(s);
      if (!lv.second.isEmpty()) {
        insertLine({{lv.first + " ", fmtBlueB}, {lv.second, fmtRedB}});
      } else {
        insertStyled(s, fmtBlueB);
      }
      continue;
    }
    if (s.startsWith("Success rate:", Qt::CaseInsensitive)) {
      // Success rate: percentage in green if 100%, otherwise orange/red.
      const auto lv = splitLabelValue(s);
      const QRegularExpression rePct(QStringLiteral(R"(([-+]?[0-9]+(?:\.[0-9]+)?)\s*%)"));
      const auto m = rePct.match(lv.second);
      if (m.hasMatch()) {
        const QString pctToken = m.captured(1);
        const double pct = m.captured(1).toDouble();
        QTextCharFormat pctFmt = fmtOrangeB;
        if (pct >= 99.999) pctFmt = fmtGreenB;
        else if (pct <= 0.000001) pctFmt = fmtRedB;

        const QString before = lv.second.left(m.capturedStart(1));
        const QString after  = lv.second.mid(m.capturedEnd(1));
        QList<QPair<QString, QTextCharFormat>> parts;
        parts << qMakePair(lv.first + " ", fmtPlain);
        if (!before.isEmpty()) parts << qMakePair(before, fmtPlain);
        parts << qMakePair(pctToken, pctFmt);
        if (!after.isEmpty()) parts << qMakePair(after, fmtPlain);
        insertLine(parts);
      } else {
        insertLine({{lv.first + " ", fmtPlain}, {lv.second, fmtOrangeB}});
      }
      continue;
    }
    if (s.startsWith("Process finished", Qt::CaseInsensitive)) {
      insertStyled(s, fmtGray);
      continue;
    }

    // Default
    insertPlain(s);
  }

  log->setTextCursor(c);
  log->ensureCursorVisible();
}

void MainWindow::appendLogStyled(const QString& line, const QColor& color, bool bold) {
  auto* tab = activeOutputTab();
  if (!tab || !tab->log) return;
  QTextEdit* log = tab->log;
  const QString trimmed = line.trimmed();
  if (trimmed.isEmpty()) return;

  QTextCursor c = log->textCursor();
  c.movePosition(QTextCursor::End);

  QTextCharFormat fmt;
  fmt.setForeground(QBrush(color));
  fmt.setFontWeight(bold ? QFont::Bold : QFont::Normal);

  c.setCharFormat(fmt);
  c.insertText(trimmed + "\n");

  log->setTextCursor(c);
  log->ensureCursorVisible();
}

// lastSummaryField(): moved to MainWindowShared.h

void MainWindow::appendFinishSummaryColored(int exitCode) {
  // Colors chosen to remain readable on light/dark themes.
  const QColor cTitle(0, 92, 185);   // blue
  const QColor cOk(0, 128, 0);       // green
  const QColor cErr(180, 0, 0);      // red
  const QColor cWarn(196, 109, 0);   // orange
  const QColor cSep(90, 90, 90);     // gray
  const QColor cText(0, 0, 0);       // black (Qt will map to theme as needed)

  const QString methodShort = lastSummaryField(processTextBuffer_, "Method:");
  const QString methodFull  = lastSummaryField(processTextBuffer_, "Method full name:");
  const QString problem     = lastSummaryField(processTextBuffer_, "Problem:");
  const QString bestf       = lastSummaryField(processTextBuffer_, "Best f (min):");
  const QString evals       = lastSummaryField(processTextBuffer_, "Evals:");
  const QString timePerRun  = lastSummaryField(processTextBuffer_, "Time per run (mean):");
  const QString success     = lastSummaryField(processTextBuffer_, "Success rate:");

  appendLogStyled("======================================================", cTitle, true);
  appendLogStyled("RESULT SUMMARY", cTitle, true);
  appendLogStyled("======================================================", cTitle, true);

  if (exitCode == 0) appendLogStyled("Status: SUCCESS", cOk, true);
  else appendLogStyled(QString("Status: FAILED (exit code %1)").arg(exitCode), cErr, true);

  if (!methodFull.isEmpty() || !methodShort.isEmpty()) {
    const QString m = !methodFull.isEmpty()
      ? (methodShort.isEmpty() ? methodFull : (methodFull + " (" + methodShort + ")"))
      : methodShort;
    appendLogStyled("Method: " + m, cText, false);
  }
  if (!problem.isEmpty()) appendLogStyled("Problem: " + problem, cText, false);
  if (!bestf.isEmpty())   appendLogStyled("Best f: " + bestf, cTitle, true);
  if (!evals.isEmpty())   appendLogStyled("Evals: " + evals, cText, false);
  if (!timePerRun.isEmpty()) appendLogStyled("Time per run: " + timePerRun, cText, false);

  if (!success.isEmpty()) {
    if (success.contains("0%", Qt::CaseInsensitive)) appendLogStyled("Success rate: " + success, cWarn, true);
    else appendLogStyled("Success rate: " + success, cOk, true);
  }

  appendLogStyled("------------------------------------------------------", cSep, false);
}


void MainWindow::loadSettings() {
  if (settingsPath_.isEmpty()) {
    // keep empty; still allow user to select
    cfg_->load(QString());
    setTablesEditable(false);
    appendLog("Settings file not found. Use 'Select settings...' to choose a cfg.");
    return;
  }

  if (!cfg_->load(settingsPath_)) {
    setTablesEditable(false);
    appendLog("Failed to load settings.");
    return;
  }
  CrashLog::append("populateSettingsTables: setTablesEditable...");
  setTablesEditable(true);
  CrashLog::append("populateSettingsTables: end (success).");
}

void MainWindow::setTablesEditable(bool editable) {
  // Changing item flags may emit itemChanged(). Block signals to avoid re-entrant updates
  // that would otherwise continuously repopulate the tables.
  QSignalBlocker b1(runTable_);
  QSignalBlocker b2(stopTable_);
  QSignalBlocker b3(initTable_);
  QSignalBlocker b4(methodTable_);
  QSignalBlocker b5(sensitivityTable_);

  Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
  Qt::ItemFlags flagsEdit = flags | Qt::ItemIsEditable;

  // Settings-area tables:
  // - Global: Key | Value | Info | Source
  // - Termination rule: Key | Value | Info
  // - Initialization: Key | Value | Info
  // Value is edited via widgets (click-based), so the underlying Value/Info/Source items remain non-editable.
  auto apply = [&](QTableWidget* t, bool hasInfo, bool hasSource, bool keyEditable, bool valueEditable) {
    if (!t) return;
    const int srcCol = (hasSource ? (hasInfo ? 3 : 2) : -1);
    for (int r = 0; r < t->rowCount(); ++r) {
      for (int c = 0; c < t->columnCount(); ++c) {
        QTableWidgetItem* item = t->item(r, c);
        if (!item) continue;

        if (c == 0) {
          item->setFlags((keyEditable && editable) ? flagsEdit : flags);
          continue;
        }

        if (hasInfo && c == 2) {
          item->setFlags(flags);
          continue;
        }

        if (srcCol >= 0 && c == srcCol) {
          item->setFlags(flags);
          continue;
        }

        if (c == 1 && !valueEditable) {
          item->setFlags(flags);
          continue;
        }

        item->setFlags(editable ? flagsEdit : flags);
      }
    }
  };

  apply(runTable_, true, true, true, false);
  apply(stopTable_, true, false, true, false);
  apply(initTable_, true, false, true, false);
  apply(methodTable_, false, false, true, true);


  // Sensitivity table has custom columns: Parameter | Analyze | Values
  if (sensitivityTable_) {
    for (int r = 0; r < sensitivityTable_->rowCount(); ++r) {
      QTableWidgetItem* p = sensitivityTable_->item(r, 0);
      QTableWidgetItem* a = sensitivityTable_->item(r, 1);
      QTableWidgetItem* v = sensitivityTable_->item(r, 2);
      if (p) p->setFlags(flags);
      if (a) {
        a->setFlags(flags | Qt::ItemIsUserCheckable);
      }
      if (v) {
        const bool inSensMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
        const bool enabled = inSensMode; // mode-only: checkbox obsolete
        const bool analyze = (a && a->checkState() == Qt::Checked);
        const bool canEdit = editable && enabled && analyze;
        v->setFlags(canEdit ? flagsEdit : flags);
      }
    }
  }

  saveCfgBtn_->setEnabled(editable);
  reloadCfgBtn_->setEnabled(true);
}


namespace {
struct SettingsValueParts {
  QString value;
  QString suffix;
  QString delim;
};

static SettingsValueParts splitSettingsValue(const QString& raw) {
  const QString s = raw;

  auto splitAt = [&](int idx, int delimStart, int delimLen) -> SettingsValueParts {
    const QString value = s.left(delimStart).trimmed();
    const QString suffix = s.mid(idx + 1).trimmed();
    const QString delim = s.mid(delimStart, delimLen);
    return {value, suffix, delim};
  };

  int idx = s.indexOf(QStringLiteral(" : "));
  if (idx >= 0) {
    // Keep delimiter exactly as found (" : ").
    return {s.left(idx).trimmed(), s.mid(idx + 3).trimmed(), QStringLiteral(" : ")};
  }

  idx = s.indexOf(QStringLiteral(" ; "));
  if (idx >= 0) {
    // Keep delimiter exactly as found (" ; ").
    return {s.left(idx).trimmed(), s.mid(idx + 3).trimmed(), QStringLiteral(" ; ")};
  }

  // Fallback: plain ':' or ';' (keep optional surrounding spaces).
  idx = s.indexOf(QLatin1Char(':'));
  if (idx >= 0) {
    int start = idx;
    int len = 1;
    if (start > 0 && s.at(start - 1) == QLatin1Char(' ')) { start -= 1; len += 1; }
    if (idx + 1 < s.size() && s.at(idx + 1) == QLatin1Char(' ')) { len += 1; }
    return splitAt(idx, start, len);
  }

  idx = s.indexOf(QLatin1Char(';'));
  if (idx >= 0) {
    int start = idx;
    int len = 1;
    if (start > 0 && s.at(start - 1) == QLatin1Char(' ')) { start -= 1; len += 1; }
    if (idx + 1 < s.size() && s.at(idx + 1) == QLatin1Char(' ')) { len += 1; }
    return splitAt(idx, start, len);
  }

  return {s.trimmed(), QString(), QString()};
}

static QString composeSettingsValue(const QString& valuePart, const SettingsValueParts& parts) {
  if (parts.suffix.isEmpty()) return valuePart;
  const QString d = parts.delim.isEmpty() ? QStringLiteral(" : ") : parts.delim;
  return valuePart + d + parts.suffix;
}

static bool keySuggestsBoolForSettings(const QString& key) {
  const QString k = key.trimmed().toLower();

  // Explicit boolean flags in [global] that must render as checkboxes.
  if (k == "end_local_refine" || k == "summary_enable" || k.startsWith("summary_show_")) {
    return true;
  }

  static const QStringList tokens = {
    "enable", "enabled", "use", "active", "flag", "csv", "plot", "verbose", "debug",
    "print", "save", "export", "log", "normalize", "shuffle", "parallel"
  };
  for (const QString& t : tokens) {
    if (k.contains(t)) return true;
  }
  return k.startsWith("is_") || k.startsWith("has_") || k.endsWith("_on") || k.endsWith("_off");
}

enum class SettingsValueKind { Bool, Int, Double, Text };

static SettingsValueKind inferSettingsValueKind(const QString& key, const QString& valuePart) {
  const QString s = valuePart.trimmed();
  const QString sl = s.toLower();

  // Bool-like textual values
  if (sl == "true" || sl == "false" || sl == "yes" || sl == "no" || sl == "on" || sl == "off") {
    return SettingsValueKind::Bool;
  }

  // Numeric bool (0/1) if key suggests it
  if ((s == "0" || s == "1") && keySuggestsBoolForSettings(key)) {
    return SettingsValueKind::Bool;
  }

  // Integer
  {
    static const QRegularExpression reInt(QStringLiteral("^[\\\\+\\\\-]?\\\\d+$"));
    if (reInt.match(s).hasMatch()) return SettingsValueKind::Int;
  }

  // Floating-point (with optional exponent)
  {
    static const QRegularExpression reDbl(
      QStringLiteral("^[\\\\+\\\\-]?(?:\\\\d+(?:\\\\.\\\\d*)?|\\\\.\\\\d+)(?:[eE][\\\\+\\\\-]?\\\\d+)?$")
    );
    if (reDbl.match(s).hasMatch()) return SettingsValueKind::Double;
  }

  return SettingsValueKind::Text;
}

static bool parseBoolValue(const QString& valuePart) {
  const QString s = valuePart.trimmed().toLower();
  if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
  if (s == "0" || s == "false" || s == "no" || s == "off") return false;
  return false;
}
}  // namespace

void MainWindow::installSettingsValueWidgets(QTableWidget* table, const QString& section) {
  if (!table) return;

  for (int r = 0; r < table->rowCount(); ++r) {
    installSettingsValueWidgetRow(table, section, r);
  }

  // Global tab: when summary_enable is unchecked, disable all summary_show_* flags.
  if (section.trimmed().toLower() != QStringLiteral("global")) return;

  int enableRow = -1;
  for (int r = 0; r < table->rowCount(); ++r) {
    QTableWidgetItem* keyItem = table->item(r, 0);
    const QString k = keyItem ? keyItem->text().trimmed().toLower() : QString();
    if (k == QStringLiteral("summary_enable")) {
      enableRow = r;
      break;
    }
  }
  if (enableRow < 0) return;

  QWidget* enableWrap = table->cellWidget(enableRow, 1);
  if (!enableWrap) return;

  QCheckBox* enableCb = enableWrap->findChild<QCheckBox*>();
  if (!enableCb) return;

  // Remove any previous hook tied to this table population cycle.
  QObject::disconnect(enableCb, nullptr, table, nullptr);

  auto applyEnabled = [table](bool enabled) {
    for (int r = 0; r < table->rowCount(); ++r) {
      QTableWidgetItem* keyItem = table->item(r, 0);
      const QString k = keyItem ? keyItem->text().trimmed().toLower() : QString();
      if (!k.startsWith(QStringLiteral("summary_show_"))) continue;

      if (QWidget* w = table->cellWidget(r, 1)) {
        w->setEnabled(enabled);
      }
    }
  };

  applyEnabled(enableCb->isChecked());
  QObject::connect(enableCb, &QCheckBox::toggled, table, applyEnabled);
}


void MainWindow::installSettingsValueWidgetRow(QTableWidget* table, const QString& section, int row) {
  if (!table) return;
  if (row < 0 || row >= table->rowCount()) return;

  const bool hasInfoCol = (table->columnCount() >= 3);
  const int infoCol = hasInfoCol ? 2 : -1;

  QTableWidgetItem* keyItem = table->item(row, 0);
  QTableWidgetItem* valItem = table->item(row, 1);
  if (!valItem) {
    valItem = new QTableWidgetItem(QString());
    table->setItem(row, 1, valItem);
  }

  QTableWidgetItem* infoItem = nullptr;
  if (hasInfoCol) {
    infoItem = table->item(row, infoCol);
    if (!infoItem) {
      infoItem = new QTableWidgetItem(QString());
      table->setItem(row, infoCol, infoItem);
    }
  }

  const QString key = keyItem ? keyItem->text().trimmed() : QString();
  if (key.isEmpty()) {
    if (QWidget* old = table->cellWidget(row, 1)) {
      table->removeCellWidget(row, 1);
      old->deleteLater();
    }
    if (infoItem) infoItem->setText(QString());
    return;
  }

  // Prefer the stored full text (with suffix) if available.
  QString rawFull = valItem->data(Qt::UserRole + 1).toString();
  if (rawFull.isEmpty()) rawFull = valItem->text();

  const SettingsValueParts parts = splitSettingsValue(rawFull);
  const SettingsValueKind kind = inferSettingsValueKind(key, parts.value);

  // Ensure Info shows the suffix and keeps the delimiter (":", ";", etc.).
  if (infoItem) {
    if (parts.suffix.isEmpty()) {
      infoItem->setText(QString());
    } else {
      const QString d = parts.delim.isEmpty() ? QStringLiteral(" : ") : parts.delim;
      infoItem->setText(d + parts.suffix);
    }
  }

  // Remove any previous widget.
  if (QWidget* old = table->cellWidget(row, 1)) {
    table->removeCellWidget(row, 1);
    old->deleteLater();
  }

  // Never paint text behind the widget (prevents overlaps).
  valItem->setText(QString());
  valItem->setData(Qt::UserRole, parts.value);
  valItem->setData(Qt::UserRole + 1, rawFull);

  auto commit = [this, table, section, row, keyItem, valItem, infoItem, parts](const QString& valuePart) {
    if (!cfg_ || !cfg_->isLoaded()) return;
    if (!table || row < 0 || row >= table->rowCount()) return;

    const QString k = keyItem ? keyItem->text().trimmed() : QString();
    if (k.isEmpty()) return;

    const QString full = composeSettingsValue(valuePart, parts);

    QScopedValueRollback<bool> guard(internalUpdate_, true);
    table->blockSignals(true);
    valItem->setText(QString());
    valItem->setData(Qt::UserRole, valuePart);
    valItem->setData(Qt::UserRole + 1, full);
    if (infoItem) {
      if (parts.suffix.isEmpty()) infoItem->setText(QString());
      else {
        const QString d = parts.delim.isEmpty() ? QStringLiteral(" : ") : parts.delim;
        infoItem->setText(d + parts.suffix);
      }
    }
    table->blockSignals(false);

    cfg_->setValue(section, k, full);
  };

  const QString sectionLower = section.trimmed().toLower();
  const QString keyLower = key.trimmed().toLower();

  // List dropdowns (explicit UI requirement)
  if (sectionLower == QStringLiteral("stop") && keyLower == QStringLiteral("rule")) {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* combo = new QComboBox(wrap);
    combo->setEditable(false);
    const QStringList opts = {
      QStringLiteral("bss"), QStringLiteral("wss"), QStringLiteral("tss"), QStringLiteral("boss"),
      QStringLiteral("srs"), QStringLiteral("irs"), QStringLiteral("doublebox"),
      QStringLiteral("maxevals"), QStringLiteral("maxiters"), QStringLiteral("all")
    };
    combo->addItems(opts);

    const QString cur = parts.value.trimmed().toLower();
    int idx = combo->findText(cur, Qt::MatchFixedString);
    if (idx < 0 && !parts.value.trimmed().isEmpty()) {
      combo->insertItem(0, parts.value.trimmed());
      idx = 0;
    }
    if (idx < 0) idx = 0;

    {
      QSignalBlocker b(combo);
      combo->setCurrentIndex(idx);
    }

    lay->addWidget(combo, 0, Qt::AlignVCenter);
    lay->addStretch(1);
    table->setCellWidget(row, 1, wrap);

    connect(combo, &QComboBox::currentTextChanged, this, [commit](const QString& t) {
      commit(t.trimmed());
    });
    return;
  }

  if (sectionLower == QStringLiteral("init") && keyLower == QStringLiteral("type")) {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* combo = new QComboBox(wrap);
    combo->setEditable(false);
    const QStringList opts = {
      QStringLiteral("uniform"), QStringLiteral("normal"), QStringLiteral("cauchy"), QStringLiteral("laplace"),
      QStringLiteral("lognormal"), QStringLiteral("exponential"), QStringLiteral("beta"), QStringLiteral("levy"),
      QStringLiteral("lhs"), QStringLiteral("halton"), QStringLiteral("oppositional")
    };
    combo->addItems(opts);

    const QString cur = parts.value.trimmed().toLower();
    int idx = combo->findText(cur, Qt::MatchFixedString);
    if (idx < 0 && !parts.value.trimmed().isEmpty()) {
      combo->insertItem(0, parts.value.trimmed());
      idx = 0;
    }
    if (idx < 0) idx = 0;

    {
      QSignalBlocker b(combo);
      combo->setCurrentIndex(idx);
    }

    lay->addWidget(combo, 0, Qt::AlignVCenter);
    lay->addStretch(1);
    table->setCellWidget(row, 1, wrap);

    connect(combo, &QComboBox::currentTextChanged, this, [commit](const QString& t) {
      commit(t.trimmed());
    });
    return;
  }

  // Click-based editors
  if (kind == SettingsValueKind::Bool) {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* cb = new QCheckBox(wrap);
    {
      QSignalBlocker b(cb);
      cb->setChecked(parseBoolValue(parts.value));
    }
    lay->addWidget(cb, 0, Qt::AlignVCenter);
    lay->addStretch(1);

    table->setCellWidget(row, 1, wrap);

    const bool numericOut = (parts.value.trimmed() == "0" || parts.value.trimmed() == "1");
    connect(cb, &QCheckBox::toggled, this, [commit, numericOut](bool checked) {
      const QString out = numericOut ? (checked ? QStringLiteral("1") : QStringLiteral("0"))
                                     : (checked ? QStringLiteral("true") : QStringLiteral("false"));
      commit(out);
    });

    return;
  }

  if (kind == SettingsValueKind::Int) {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* sb = new QSpinBox(wrap);
    sb->setFrame(false);
    sb->setReadOnly(true);                // no typing; arrows/wheel still work
    sb->setKeyboardTracking(false);
    sb->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    {
      bool ok = false;
      const int v = parts.value.trimmed().toInt(&ok);
      QSignalBlocker b(sb);
      sb->setValue(ok ? v : 0);
    }

    auto* btn = new QToolButton(wrap);
    btn->setText(QStringLiteral("..."));
    btn->setToolTip(QStringLiteral("Set value"));

    lay->addWidget(sb, 0, Qt::AlignVCenter);
    lay->addWidget(btn, 0, Qt::AlignVCenter);
    lay->addStretch(1);

    table->setCellWidget(row, 1, wrap);

    connect(sb, qOverload<int>(&QSpinBox::valueChanged), this, [commit](int v) {
      commit(QString::number(v));
    });

    connect(btn, &QToolButton::clicked, this, [this, sb]() {
      bool ok = false;
      const int v = QInputDialog::getInt(this, QStringLiteral("Set value"), QStringLiteral("Value"),
                                         sb->value(), sb->minimum(), sb->maximum(), 1, &ok);
      if (ok) sb->setValue(v);
    });

    return;
  }

  if (kind == SettingsValueKind::Double) {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* dsb = new QDoubleSpinBox(wrap);
    dsb->setFrame(false);
    dsb->setReadOnly(true);               // no typing; arrows/wheel still work
    dsb->setKeyboardTracking(false);
    dsb->setDecimals(12);
    dsb->setRange(-1.0e300, 1.0e300);
    dsb->setSingleStep(0.01);

    {
      bool ok = false;
      const double v = QLocale::c().toDouble(parts.value.trimmed(), &ok);
      QSignalBlocker b(dsb);
      dsb->setValue(ok ? v : 0.0);
    }

    auto* btn = new QToolButton(wrap);
    btn->setText(QStringLiteral("..."));
    btn->setToolTip(QStringLiteral("Set value"));

    lay->addWidget(dsb, 0, Qt::AlignVCenter);
    lay->addWidget(btn, 0, Qt::AlignVCenter);
    lay->addStretch(1);

    table->setCellWidget(row, 1, wrap);

    connect(dsb, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [commit](double v) {
      commit(QLocale::c().toString(v, 'g', 15));
    });

    connect(btn, &QToolButton::clicked, this, [this, dsb]() {
      bool ok = false;
      const double v = QInputDialog::getDouble(this, QStringLiteral("Set value"), QStringLiteral("Value"),
                                               dsb->value(), dsb->minimum(), dsb->maximum(),
                                               dsb->decimals(), &ok);
      if (ok) dsb->setValue(v);
    });

    return;
  }

  // Text: click button (no typing in-cell).
  {
    auto* wrap = new QWidget(table);
    auto* lay = new QHBoxLayout(wrap);
    lay->setContentsMargins(4, 0, 4, 0);
    lay->setSpacing(6);

    auto* btn = new QToolButton(wrap);
    btn->setToolButtonStyle(Qt::ToolButtonTextOnly);
    btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    auto applyBtnText = [btn](const QString& v) {
      btn->setText(v.isEmpty() ? QStringLiteral("<empty>") : v);
      btn->setToolTip(v);
    };
    applyBtnText(parts.value);

    lay->addWidget(btn, 1, Qt::AlignVCenter);
    table->setCellWidget(row, 1, wrap);

    connect(btn, &QToolButton::clicked, this, [this, commit, btn, applyBtnText]() {
      bool ok = false;
      const QString current = (btn->text() == QStringLiteral("<empty>")) ? QString() : btn->text();
      const QString v = QInputDialog::getText(this, QStringLiteral("Set value"), QStringLiteral("Value"),
                                              QLineEdit::Normal, current, &ok);
      if (!ok) return;
      applyBtnText(v);
      commit(v);
    });
  }
}



static void fillKVTable(QTableWidget* t, const QMap<QString, QString>& kv, bool includeSource,
                        const std::function<QString(const QString&)>& sourceFn = {}) {
  if (!t) return;

  t->blockSignals(true);
  t->setRowCount(0);

  QStringList keys = kv.keys();
  keys.sort(Qt::CaseInsensitive);

  t->setRowCount(keys.size());

  const bool hasInfoCol = (!includeSource && t->columnCount() >= 3) || (includeSource && t->columnCount() >= 4);
  const int infoCol = hasInfoCol ? 2 : -1;
  const int srcCol = includeSource ? (hasInfoCol ? 3 : 2) : -1;

  for (int row = 0; row < keys.size(); ++row) {
    const QString k = keys[row];
    const QString raw = kv.value(k);

    auto* keyItem = new QTableWidgetItem(k);
    keyItem->setData(Qt::UserRole, k);
    t->setItem(row, 0, keyItem);

    if (hasInfoCol) {
      const SettingsValueParts parts = splitSettingsValue(raw);

      auto* valItem = new QTableWidgetItem(QString());
      valItem->setData(Qt::UserRole, parts.value);
      valItem->setData(Qt::UserRole + 1, raw);
      t->setItem(row, 1, valItem);

      auto* infoItem = new QTableWidgetItem(QString());
      if (!parts.suffix.isEmpty()) {
        const QString d = parts.delim.isEmpty() ? QStringLiteral(" : ") : parts.delim;
        infoItem->setText(d + parts.suffix);
      }
      t->setItem(row, infoCol, infoItem);
    } else {
      auto* valItem = new QTableWidgetItem(raw);
      t->setItem(row, 1, valItem);
    }

    if (srcCol >= 0) {
      const QString src = sourceFn ? sourceFn(k) : QString();
      auto* srcItem = new QTableWidgetItem(src);
      t->setItem(row, srcCol, srcItem);
    }
  }

  t->resizeColumnsToContents();
  t->blockSignals(false);
}

void MainWindow::populateSettingsTables() {
  QScopedValueRollback<bool> guard(internalUpdate_, true);

  CrashLog::append("populateSettingsTables: begin.");
  CrashLog::append(QString("populateSettingsTables: cfg_loaded=%1").arg(cfg_ && cfg_->isLoaded() ? "true" : "false"));
  if (!cfg_ || !cfg_->isLoaded()) {
    runTable_->setRowCount(0);
    stopTable_->setRowCount(0);
    initTable_->setRowCount(0);
    methodTable_->setRowCount(0);
    sensitivityTable_->setRowCount(0);
    return;
  }

  QString method = currentMethodShort();
  CrashLog::append(QString("populateSettingsTables: methodText=%1").arg(method));
  if (method.isEmpty()) method = "method";

  // Global tab must reflect exactly the keys present in the [global] section of the loaded
  // settings file. Method overrides are shown only in the Method tab.
  CrashLog::append("populateSettingsTables: reading [global]...");
  const auto globalMap = cfg_->sectionMap("global");
  CrashLog::append(QString("populateSettingsTables: global keys=%1").arg(globalMap.size()));
  CrashLog::append("populateSettingsTables: fill Global table...");
  fillKVTable(runTable_, globalMap, true, [&](const QString&){ return QStringLiteral("global"); });
  CrashLog::append("populateSettingsTables: Global table filled.");

  // Stop
  CrashLog::append("populateSettingsTables: reading [stop]...");
  auto stopMap = cfg_->sectionMap("stop");
  CrashLog::append(QString("populateSettingsTables: stop keys=%1").arg(stopMap.size()));
  CrashLog::append("populateSettingsTables: fill Stop table...");
  fillKVTable(stopTable_, stopMap, false);
  CrashLog::append("populateSettingsTables: Stop table filled.");

  // Init
  CrashLog::append("populateSettingsTables: reading [init]...");
  auto initMap = cfg_->sectionMap("init");
  CrashLog::append(QString("populateSettingsTables: init keys=%1").arg(initMap.size()));
  CrashLog::append("populateSettingsTables: fill Init table...");
  fillKVTable(initTable_, initMap, false);
  CrashLog::append("populateSettingsTables: Init table filled.");

  // Method section
  CrashLog::append(QString("populateSettingsTables: reading method section [%1]...").arg(method));
  auto mMap = cfg_->sectionMap(method);
  CrashLog::append(QString("populateSettingsTables: method keys=%1").arg(mMap.size()));
  CrashLog::append("populateSettingsTables: fill Method table...");
  fillKVTable(methodTable_, mMap, false);
  CrashLog::append("populateSettingsTables: Method table filled.");

  // Sensitivity (method-based or problem-based parameter sweep UI)
  CrashLog::append("populateSettingsTables: sync Sensitivity UI...");
  syncSensitivityEnableUiFromConfig();
  const int runMode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
  if (runMode == 3) {
    const QString problem = currentProblemShort();
    if (!problem.isEmpty())
      populateSensitivityTableForProblem(problem);
    else
      populateSensitivityTableForMethod(method);
  } else {
    populateSensitivityTableForMethod(method);
  }
  CrashLog::append("populateSettingsTables: Sensitivity UI synced.");

  CrashLog::append("populateSettingsTables: setTablesEditable...");
  setTablesEditable(true);

  // Ensure value widgets are visible for all parameters on the requested Settings tabs.
  installSettingsValueWidgets(runTable_, "global");
  installSettingsValueWidgets(stopTable_, "stop");
  installSettingsValueWidgets(initTable_, "init");

  CrashLog::append("populateSettingsTables: end (success).");
}

void MainWindow::onSelectSettings() {
  QString file = QFileDialog::getOpenFileName(this, "Select settings", projectRoot_, "Config files (*.cfg *.ini *.txt);;All files (*.*)");
  if (file.isEmpty()) return;
  settingsPath_ = file;
  loadSettings();
  populateSettingsTables();
}

void MainWindow::onReloadSettings() {
  loadSettings();
  populateSettingsTables();
  if (auto* lbl = findChild<QLabel*>("selectionRunsLbl"))
    lbl->setText(QString::number(globalRunsFromSettings()));

  appendLog("Settings reloaded.");
  QMessageBox::information(this, "Reload settings", "Settings reloaded successfully.");
}

void MainWindow::onSaveSettings() {
  if (!cfg_ || !cfg_->isLoaded() || settingsPath_.isEmpty()) {
    QMessageBox::warning(this, "No settings", "No settings file is loaded.");
    return;
  }
  if (!cfg_->save(settingsPath_)) {
    QMessageBox::critical(this, "Save failed", "Could not save settings.");
    return;
  }
  appendLog("Settings saved.");
  if (auto* lbl = findChild<QLabel*>("selectionRunsLbl"))
    lbl->setText(QString::number(globalRunsFromSettings()));

  QMessageBox::information(this, "Save settings", QString("Settings saved successfully to:\n%1").arg(settingsPath_));
}

void MainWindow::onSettingItemChanged(int row, int column) {
  if (internalUpdate_) return;

  // Identify which table emitted the change
  QTableWidget* senderTable = qobject_cast<QTableWidget*>(sender());
  if (!senderTable) return;
  if (!cfg_ || !cfg_->isLoaded()) return;
  if (senderTable == sensitivityTable_) return;

  // Global tab edits map strictly to the [global] section of the loaded settings file.
  if (senderTable == runTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    QString value = valItem->data(Qt::UserRole + 1).toString();
    if (value.isEmpty()) {
      const QString vp = valItem->data(Qt::UserRole).toString();
      QTableWidgetItem* infoItem = (senderTable->columnCount() >= 3) ? senderTable->item(row, 2) : nullptr;
      const QString info = infoItem ? infoItem->text() : QString();
      value = vp + info;
      if (value.trimmed().isEmpty()) value = valItem->text();
    }
    if (newKey.isEmpty()) return;

    const QString targetSection = "global";

    const QString oldKey = keyItem->data(Qt::UserRole).toString().trimmed();

    if (column == 0) {
      // Rename or create.
      if (!oldKey.isEmpty() && oldKey.compare(newKey, Qt::CaseInsensitive) != 0) {
        cfg_->removeKey(targetSection, oldKey);
      }
      cfg_->setValue(targetSection, newKey, value);
      keyItem->setData(Qt::UserRole, newKey);
    } else {
      // Value change.
      cfg_->setValue(targetSection, newKey, value);
      if (oldKey.isEmpty()) keyItem->setData(Qt::UserRole, newKey);
    }

    populateSettingsTables();
    // If the "runs" key was edited, refresh the Selection area display.
    if (newKey.compare("runs", Qt::CaseInsensitive) == 0) {
      if (auto* lbl = findChild<QLabel*>("selectionRunsLbl"))
        lbl->setText(QString::number(globalRunsFromSettings()));
    }
    return;
  }

  // Stop parameters: names/count may vary, so allow editing both Key and Value columns.
  if (senderTable == stopTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    QString value = valItem->data(Qt::UserRole + 1).toString();
    if (value.isEmpty()) {
      const QString vp = valItem->data(Qt::UserRole).toString();
      QTableWidgetItem* infoItem = (senderTable->columnCount() >= 3) ? senderTable->item(row, 2) : nullptr;
      const QString info = infoItem ? infoItem->text() : QString();
      value = vp + info;
      if (value.trimmed().isEmpty()) value = valItem->text();
    }
    if (newKey.isEmpty()) return;

    const QString oldKey = keyItem->data(Qt::UserRole).toString().trimmed();

    if (column == 0) {
      if (!oldKey.isEmpty() && oldKey.compare(newKey, Qt::CaseInsensitive) != 0) {
        cfg_->removeKey("stop", oldKey);
      }
      cfg_->setValue("stop", newKey, value);
      keyItem->setData(Qt::UserRole, newKey);
    } else {
      cfg_->setValue("stop", newKey, value);
      if (oldKey.isEmpty()) keyItem->setData(Qt::UserRole, newKey);
    }

    populateSettingsTables();
    return;
  }

  // Init parameters: names/count may vary, so allow editing both Key and Value columns.
  if (senderTable == initTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    QString value = valItem->data(Qt::UserRole + 1).toString();
    if (value.isEmpty()) {
      const QString vp = valItem->data(Qt::UserRole).toString();
      QTableWidgetItem* infoItem = (senderTable->columnCount() >= 3) ? senderTable->item(row, 2) : nullptr;
      const QString info = infoItem ? infoItem->text() : QString();
      value = vp + info;
      if (value.trimmed().isEmpty()) value = valItem->text();
    }
    if (newKey.isEmpty()) return;

    const QString oldKey = keyItem->data(Qt::UserRole).toString().trimmed();

    if (column == 0) {
      if (!oldKey.isEmpty() && oldKey.compare(newKey, Qt::CaseInsensitive) != 0) {
        cfg_->removeKey("init", oldKey);
      }
      cfg_->setValue("init", newKey, value);
      keyItem->setData(Qt::UserRole, newKey);
    } else {
      cfg_->setValue("init", newKey, value);
      if (oldKey.isEmpty()) keyItem->setData(Qt::UserRole, newKey);
    }

    populateSettingsTables();
    return;
  }

  // Method parameters: names/count vary per method, so allow editing both Key and Value columns.
  if (senderTable == methodTable_) {
    if (column != 0 && column != 1) return;

    QString method = currentMethodShort();
    if (method.isEmpty()) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    QString value = valItem->data(Qt::UserRole + 1).toString();
    if (value.isEmpty()) {
      const QString vp = valItem->data(Qt::UserRole).toString();
      QTableWidgetItem* infoItem = (senderTable->columnCount() >= 3) ? senderTable->item(row, 2) : nullptr;
      const QString info = infoItem ? infoItem->text() : QString();
      value = vp + info;
      if (value.trimmed().isEmpty()) value = valItem->text();
    }
    if (newKey.isEmpty()) return;

    const QString oldKey = keyItem->data(Qt::UserRole).toString().trimmed();

    if (column == 0) {
      // Rename or create.
      if (!oldKey.isEmpty() && oldKey.compare(newKey, Qt::CaseInsensitive) != 0) {
        cfg_->removeKey(method, oldKey);
      }
      cfg_->setValue(method, newKey, value);
      keyItem->setData(Qt::UserRole, newKey);
    } else {
      // Value change.
      cfg_->setValue(method, newKey, value);
      if (oldKey.isEmpty()) keyItem->setData(Qt::UserRole, newKey);
    }

    populateSettingsTables();
    return;
  }
}



static bool isLocalOptimizationKeyName(const QString& key) {
  const QString k = key.trimmed().toLower();
  // Heuristic filter for local-search / local-optimization related keys.
  return k.contains("local") || k.contains("localsearch") || k.contains("bfgs") || k.contains("lbfgs") ||
         k.contains("nelder") || k.contains("simplex") || k.contains("cmaes") || k.startsWith("ls_") ||
         k.startsWith("local_") || k.startsWith("bfgs_");
}

void MainWindow::cleanupSensitivityLegacyKeys() {
  if (!cfg_) return;
  const auto sMap = cfg_->sectionMap("sensitivity");
  for (auto it = sMap.begin(); it != sMap.end(); ++it) {
    const QString k = it.key();
    const QString kl = k.trimmed().toLower();
    const bool keep = (kl == "enabled" || kl == "mode" || kl == "output" || kl.startsWith("values."));
    if (!keep) {
      cfg_->removeKey("sensitivity", k);
    }
  }
  // Remove common legacy keys explicitly (if they exist with different naming).
  cfg_->removeKey("sensitivity", "enable");
  cfg_->removeKey("sensitivity", "params");
  cfg_->removeKey("sensitivity", "metric");
}

void MainWindow::writeSensitivityConfigFromUi() {
  if (internalUpdate_) {
    return;
  }

  static const QString SEC_SENS = QStringLiteral("sensitivity");

  const bool inSensRunMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  // The "Enable sensitivity analysis" checkbox is obsolete; enabled is determined
  // entirely by the run mode: only Sensitivity run (mode 2) enables it.
  const bool enabled = inSensRunMode;

  // For single/batch run modes: write a clean disabled state and stop.
  // This prevents any stale enabled=1 / params from previous sensitivity runs
  // from leaking into the CLI invocation.
  if (!enabled) {
    cfg_->setValue(SEC_SENS, QStringLiteral("enabled"), QStringLiteral("0"));
    cfg_->setValue(SEC_SENS, QStringLiteral("enable"),  QStringLiteral("0"));
    cfg_->setValue(SEC_SENS, QStringLiteral("params"),  QStringLiteral(""));
    return;
  }

  // Helper: strip inline comments ("; ..." or "# ...") from a config value.
  auto stripComment = [](const QString& raw) -> QString {
    QString s = raw;
    int p = s.indexOf(';'); if (p >= 0) s = s.left(p);
    p = s.indexOf('#'); if (p >= 0) s = s.left(p);
    return s.trimmed();
  };

  // Preserve user-provided mode/output if present, otherwise apply defaults.
  const QString mode   = stripComment(cfg_->value(SEC_SENS, QStringLiteral("mode"),   QStringLiteral("grid")));
  const QString output = stripComment(cfg_->value(SEC_SENS, QStringLiteral("output"), QStringLiteral("sensitivity_results.csv")));
  const QString modeClean   = mode.isEmpty()   ? QStringLiteral("grid")                       : mode;
  const QString outputClean = output.isEmpty() ? QStringLiteral("sensitivity_results.csv")    : output;

  QStringList enabledParams;
  QMap<QString, QString> valuesByParam;

  if (sensitivityTable_) {
    for (int r = 0; r < sensitivityTable_->rowCount(); ++r) {
      auto* pItem = sensitivityTable_->item(r, 0);
      auto* eItem = sensitivityTable_->item(r, 1);
      auto* vItem = sensitivityTable_->item(r, 2);
      if (!pItem || !eItem || !vItem) {
        continue;
      }

      const QString param = pItem->text().trimmed();
      const bool checked  = (eItem->checkState() == Qt::Checked);
      const QString vals  = vItem->text().trimmed();

      if (!checked || param.isEmpty() || vals.isEmpty()) {
        continue;
      }

      enabledParams.push_back(param);
      valuesByParam.insert(param, vals);
    }
  }

  // Keep only a deterministic set of keys inside [sensitivity], and rewrite it from scratch.
  // This prevents stale parameters from other methods (e.g., CR/F) from leaking into a new run.
  QSet<QString> keep;
  keep.insert(QStringLiteral("enabled"));
  keep.insert(QStringLiteral("enable"));   // legacy compatibility
  keep.insert(QStringLiteral("mode"));
  keep.insert(QStringLiteral("output"));
  keep.insert(QStringLiteral("params"));   // list of parameter names

  for (auto it = valuesByParam.cbegin(); it != valuesByParam.cend(); ++it) {
    keep.insert(it.key());                               // legacy key: <param> = ...
    keep.insert(QStringLiteral("values.") + it.key());   // new key: values.<param> = ...
  }

  const auto sMap = cfg_->sectionMap(SEC_SENS);
  for (auto it = sMap.cbegin(); it != sMap.cend(); ++it) {
    const QString key = it.key().trimmed();
    if (!keep.contains(key)) {
      cfg_->removeKey(SEC_SENS, key);
    }
  }

  cfg_->setValue(SEC_SENS, QStringLiteral("enabled"), enabled ? QStringLiteral("1") : QStringLiteral("0"));
  cfg_->setValue(SEC_SENS, QStringLiteral("enable"),  enabled ? QStringLiteral("1") : QStringLiteral("0"));
  cfg_->setValue(SEC_SENS, QStringLiteral("mode"),    modeClean);
  cfg_->setValue(SEC_SENS, QStringLiteral("output"),  outputClean);
  cfg_->setValue(SEC_SENS, QStringLiteral("params"),  enabledParams.join(QStringLiteral(",")));

  for (auto it = valuesByParam.cbegin(); it != valuesByParam.cend(); ++it) {
    cfg_->setValue(SEC_SENS, it.key(), it.value());  // legacy
    cfg_->setValue(SEC_SENS, QStringLiteral("values.") + it.key(), it.value());
  }
}

void MainWindow::applySensitivityRowFlags(int row) {
  if (!sensitivityTable_) return;
  if (row < 0 || row >= sensitivityTable_->rowCount()) return;

  QTableWidgetItem* a = sensitivityTable_->item(row, 1);
  QTableWidgetItem* v = sensitivityTable_->item(row, 2);
  if (!a || !v) return;

  const bool inSensMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  const bool enabled = inSensMode; // mode-only: checkbox obsolete
  const bool analyze = (a->checkState() == Qt::Checked);
  const bool editable = saveCfgBtn_ && saveCfgBtn_->isEnabled();

  Qt::ItemFlags base = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
  v->setFlags((enabled && analyze && editable) ? (base | Qt::ItemIsEditable) : base);
}

void MainWindow::syncSensitivityEnableUiFromConfig() {
  if (!cfg_ || !cfg_->isLoaded() || !sensitivityEnableChk_ || !sensitivityTable_) return;

  QString en = cfg_->value("sensitivity", "enabled");
  if (en.isEmpty()) en = cfg_->value("sensitivity", "enable"); // legacy
  const bool checked = (!en.trimmed().isEmpty() && en.trimmed() != "0");

  QSignalBlocker b1(sensitivityEnableChk_);
  sensitivityEnableChk_->setChecked(checked);

  const bool inSensRunMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  sensitivityTable_->setEnabled(inSensRunMode);
}

void MainWindow::populateSensitivityTableForMethod(const QString& methodSection) {
  if (!cfg_ || !cfg_->isLoaded() || !sensitivityTable_) return;

  QSignalBlocker b(sensitivityTable_);
  QScopedValueRollback<bool> guard(internalUpdate_, true);

  sensitivityTable_->setRowCount(0);

  const auto mMap = cfg_->sectionMap(methodSection);
  QStringList params = mMap.keys();
  params.sort(Qt::CaseInsensitive);

  // Build rows from method parameters; exclude local-search keys and empty keys.
  for (const QString& p : params) {
    const QString param = p.trimmed();
    if (param.isEmpty()) continue;
    if (isLocalOptimizationKeyName(param)) continue;

    const int row = sensitivityTable_->rowCount();
    sensitivityTable_->insertRow(row);

    auto* pItem = new QTableWidgetItem(param);
    pItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    sensitivityTable_->setItem(row, 0, pItem);

    auto* aItem = new QTableWidgetItem();
    aItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
    const QString k = QString("values.%1").arg(param);
    const QString valNew = cfg_->value("sensitivity", k).trimmed();
    const QString valOld = cfg_->value("sensitivity", param).trimmed();
    const QString effectiveVal = !valNew.isEmpty() ? valNew : valOld;

    const QStringList enabledKeys = cfg_->value("sensitivity", QStringLiteral("params"), QString()).split(",", Qt::SkipEmptyParts);
    const bool isEnabled = (!effectiveVal.isEmpty() && (enabledKeys.isEmpty() || enabledKeys.contains(param) || enabledKeys.contains(k)));

    aItem->setCheckState(isEnabled ? Qt::Checked : Qt::Unchecked);
    sensitivityTable_->setItem(row, 1, aItem);

    auto* vItem = new QTableWidgetItem(effectiveVal);
    sensitivityTable_->setItem(row, 2, vItem);

    applySensitivityRowFlags(row);
  }

  // Ensure UI enable state is applied after repopulating.
  const bool inSensRunMode2 = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  const bool enabled = inSensRunMode2; // mode-only: checkbox obsolete
  sensitivityTable_->setEnabled(enabled);
}

void MainWindow::populateSensitivityTableForProblem(const QString& problemSection) {
  if (!cfg_ || !cfg_->isLoaded() || !sensitivityTable_) return;

  QSignalBlocker b(sensitivityTable_);
  QScopedValueRollback<bool> guard(internalUpdate_, true);

  sensitivityTable_->setRowCount(0);

  const auto pMap = cfg_->sectionMap(problemSection);
  if (pMap.isEmpty()) {
    // No config section for this problem — nothing to sweep.
    sensitivityTable_->setEnabled(false);
    return;
  }

  QStringList params = pMap.keys();
  params.sort(Qt::CaseInsensitive);

  // Build rows from problem parameters.
  for (const QString& p : params) {
    const QString param = p.trimmed();
    if (param.isEmpty()) continue;

    const int row = sensitivityTable_->rowCount();
    sensitivityTable_->insertRow(row);

    auto* pItem = new QTableWidgetItem(param);
    pItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    sensitivityTable_->setItem(row, 0, pItem);

    auto* aItem = new QTableWidgetItem();
    aItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);

    // Try to find pre-configured sensitivity values from [sensitivity] section.
    const QString k = QString("values.%1").arg(param);
    const QString valNew = cfg_->value("sensitivity", k).trimmed();
    const QString valOld = cfg_->value("sensitivity", param).trimmed();
    const QString effectiveVal = !valNew.isEmpty() ? valNew : valOld;

    const QStringList enabledKeys = cfg_->value("sensitivity", QStringLiteral("params"), QString()).split(",", Qt::SkipEmptyParts);
    const bool isEnabled = (!effectiveVal.isEmpty() && (enabledKeys.isEmpty() || enabledKeys.contains(param) || enabledKeys.contains(k)));

    aItem->setCheckState(isEnabled ? Qt::Checked : Qt::Unchecked);
    sensitivityTable_->setItem(row, 1, aItem);

    // Pre-fill "Values" column from [sensitivity] section values.
    auto* vItem = new QTableWidgetItem(effectiveVal);
    sensitivityTable_->setItem(row, 2, vItem);

    applySensitivityRowFlags(row);
  }

  // Ensure UI enable state is applied.
  const bool inSensRunMode = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  sensitivityTable_->setEnabled(inSensRunMode);
}

bool MainWindow::problemHasConfigSection(const QString& problemShort) const {
  if (!cfg_ || !cfg_->isLoaded()) return false;
  const auto pMap = cfg_->sectionMap(problemShort);
  return !pMap.isEmpty();
}

void MainWindow::onSensitivityEnableToggled(bool checked) {
  if (internalUpdate_) return;
  if (!cfg_ || !cfg_->isLoaded()) return;
  if (!sensitivityTable_) return;

  QScopedValueRollback<bool> guard(internalUpdate_, true);
  cfg_->setValue("sensitivity", "enabled", checked ? "1" : "0");

  sensitivityTable_->setEnabled(checked);

  // Re-apply editability based on enabled state.
  for (int r = 0; r < sensitivityTable_->rowCount(); ++r) {
    applySensitivityRowFlags(r);
  }

  writeSensitivityConfigFromUi();
}

void MainWindow::onSensitivityCellChanged(int row, int column) {
  if (internalUpdate_) return;
  if (!cfg_ || !cfg_->isLoaded()) return;
  if (!sensitivityTable_) return;

  // Only column 1 (Analyze) or 2 (Values) affects sensitivity config.
  if (column != 1 && column != 2) return;

  QScopedValueRollback<bool> guard(internalUpdate_, true);

  applySensitivityRowFlags(row);
  writeSensitivityConfigFromUi();
}

} // namespace optimsolution_gui
