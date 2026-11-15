// MainWindow_RunControl.cpp
// Run mode handling (Single / Batch / Sensitivity), run & stop control, batch selection UI, process output routing, sensitivity job scheduling and CSV polling.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"

namespace optimsolution_gui {
void MainWindow::onRunClicked() {
  // Multi-objective mode uses its own QProcess (moProc_), completely separate
  // from proc_ used by Single/Batch/Sensitivity -- so this toggle never
  // touches (and is never touched by) the logic below.
  const int currentMode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
  if (moProc_ && moProc_->state() != QProcess::NotRunning) {
    appendLog("Stopping multi-objective run...");
    moProc_->terminate();
    if (!moProc_->waitForFinished(2000)) {
      moProc_->kill();
    }
    return;
  }
  if (currentMode == 4) {
    onRunMultiObjective();
    return;
  }

  // Toggle behavior: Run when idle, Stop when a process is running.
  if (proc_ && proc_->state() != QProcess::NotRunning) {
    if (batchActive_) {
      batchStopRequested_ = true;
    }
    requestStop();
    return;
  }

  // If the batch is currently post-processing (no QProcess running), treat the button as Stop.
  if (batchActive_ && batchPostInFlight_) {
    batchStopRequested_ = true;
    appendLog("Stopping batch after post-processing...");
    return;
  }

  // Batch mode: run the selected method/problem combinations.
  if (runModeBox_ && runModeBox_->currentData().toInt() == 1) {
    // Ensure sensitivity is disabled for batch runs.
    if (cfg_ && cfg_->isLoaded()) {
      cfg_->setValue("sensitivity", "enabled", "0");
      cfg_->setValue("sensitivity", "enable",  "0");
    }
    startBatch();
    return;
  }

  // Portfolio mode (mode 7): mid-run dynamic algorithm switching. Candidates
  // come from the Batch panel's Methods list (reused here as the candidate
  // pool -- see updateBatchPanelVisibility()); problem/dim come from the
  // main dropdowns, exactly like Single run. Everything is launched through
  // the exact same startCliProcess()/proc_ plumbing Single run already uses
  // -- Portfolio is, from the GUI's point of view, just another CLI mode.
  // Advanced knobs (policy, segment/probe evals, warm-start radius/decay)
  // are read from an optional [portfolio] section of optimsolution.cfg; a
  // dedicated settings panel (mirroring NN Landscape's own panel) is a
  // natural follow-up rather than something guessed at here.
  if (currentMode == 7) {
    QStringList candidates;
    if (batchMethodsList_) {
      for (auto* item : batchMethodsList_->selectedItems()) {
        const QString shortName = item->data(Qt::UserRole).toString();
        candidates << (shortName.isEmpty() ? item->text().trimmed() : shortName);
      }
    }
    if (candidates.size() < 2) {
      QMessageBox::warning(this, "Portfolio run",
        "Select at least two candidate methods in the Methods list on the left "
        "-- Portfolio mode needs a pool of methods to switch between.");
      return;
    }

    // Problems now come from the same Batch-panel Problems list Batch run
    // itself uses (multi-select, with per-problem dim overrides already
    // supported there -- see syncBatchProblemDimsTable()/selectedBatch-
    // ProblemShortNames()), not the single problem/dim dropdowns. Every
    // candidate method is relayed across every selected problem.
    const QStringList problemKeys = selectedBatchProblemShortNames();
    if (problemKeys.isEmpty()) {
      QMessageBox::warning(this, "Portfolio run",
        "Select at least one problem in the Problems list on the left.");
      return;
    }
    struct ProbDim { QString base; int dim; };
    QVector<ProbDim> probs;
    const int userDim = (dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
    for (const QString& key : problemKeys) {
      const QString base = batchBaseProblemShort(key);
      const int dim = batchDimForProblem(key, userDim);
      probs.push_back({ base, dim });
    }
    const bool multiProblem = (probs.size() > 1);

    // ConfigFile::value() (gui/ConfigFile.cpp) returns the raw rest-of-line
    // after "=", INCLUDING any trailing "; comment" -- unlike the CLI's own
    // C++ config parser (src/config.cpp), which strips ';'/'#'/'//' comments.
    // The [portfolio] section in optimsolution.cfg documents each key with
    // exactly that kind of trailing comment, so every value read here must
    // be stripped locally before it is placed straight into a CLI argument,
    // or the comment text itself ends up on the command line.
    auto stripInlineComment = [](QString s) -> QString {
      const int cut = s.indexOf(';');
      if (cut >= 0) s = s.left(cut);
      return s.trimmed();
    };
    const QString policy      = stripInlineComment(cfg_ ? cfg_->value("portfolio", "policy", "greedy_probe") : "greedy_probe");
    const QString totalEvals  = stripInlineComment(cfg_ ? cfg_->value("portfolio", "total_evals", "150000") : "150000");
    const QString segEvals    = stripInlineComment(cfg_ ? cfg_->value("portfolio", "segment_evals", "5000") : "5000");
    const QString probeEvals  = stripInlineComment(cfg_ ? cfg_->value("portfolio", "probe_evals", "300") : "300");
    const QString radius0     = stripInlineComment(cfg_ ? cfg_->value("portfolio", "warmstart_radius0", "0.35") : "0.35");
    const QString radiusDecay = stripInlineComment(cfg_ ? cfg_->value("portfolio", "warmstart_decay", "0.85") : "0.85");
    const QString radiusMin   = stripInlineComment(cfg_ ? cfg_->value("portfolio", "warmstart_radius_min", "0.02") : "0.02");
    const QString withBase    = stripInlineComment(cfg_ ? cfg_->value("portfolio", "with_baselines", "1") : "1");
    const QString runsStr     = QString::number(globalRunsFromSettings());

    const QString tabTitle = multiProblem
      ? QString("portfolio[%1] | %2 problems").arg(candidates.join('+')).arg(probs.size())
      : QString("portfolio[%1]").arg(candidates.join('+'));
    const int outIdx = createOutputRunTab(tabTitle, probs.front().base, probs.front().dim);
    if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
    activeOutputRunIndex_ = outIdx;

    const QString prefix = multiProblem
      ? QString("portfolio_multi%1_%2").arg(probs.size()).arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"))
      : QString("portfolio_%1_d%2_%3").arg(probs.front().base).arg(probs.front().dim)
                                       .arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
    if (auto* tab = outputTab(outIdx)) {
      tab->portfolioOutPrefix = prefix;
      tab->portfolioMultiProblem = multiProblem;
    }

    QStringList problemsArg;
    for (const auto& p : probs) problemsArg << QString("%1:%2").arg(p.base).arg(p.dim);

    QStringList args;
    args << "--portfolio" << "--problems" << problemsArg.join(',')
         << "--methods" << candidates.join(',')
         << "--policy" << policy
         << "--total-evals" << totalEvals
         << "--segment-evals" << segEvals
         << "--probe-evals" << probeEvals
         << "--warmstart-radius0" << radius0
         << "--warmstart-decay" << radiusDecay
         << "--warmstart-radius-min" << radiusMin
         << "--runs" << runsStr
         << "--out-prefix" << prefix;
    // Baselines are ON by default: without them the run produces a single
    // number with nothing to compare it against, and the Ranking/Friedman/
    // Wilcoxon tabs stay empty. Set with_baselines = 0 in [portfolio] to
    // skip them (much faster, but the comparison tabs will be empty).
    if (withBase != "0" && withBase.compare("false", Qt::CaseInsensitive) != 0) {
      args << "--with-baselines";
    }
    // NOTE: no explicit --config here -- startCliProcess() already launches
    // the CLI with its working directory set to a folder containing a
    // freshly written optimsolution.cfg snapshot of the GUI's current
    // settings (see the "runtimeWd"/"runtimeCfgPath" logic there), and Mode
    // 8's own cfg lookup defaults to "optimsolution.cfg" relative to CWD --
    // exactly the convention every other mode already relies on.

    lastArgs_ = args;
    lastHadDimension_ = true;
    processTextBuffer_.clear();

    appendLog("----------------------------------------");
    appendLog("Running: " + cliPath() + " " + args.join(' '));

    startCliProcess(args);
    return;
  }

  // For Single run: also ensure sensitivity is disabled.
  const bool isSensRun = (runModeBox_ && (runModeBox_->currentData().toInt() == 2 || runModeBox_->currentData().toInt() == 3));
  if (!isSensRun && cfg_ && cfg_->isLoaded()) {
    cfg_->setValue("sensitivity", "enabled", "0");
    cfg_->setValue("sensitivity", "enable",  "0");
  }
  if (isSensRun) {
    const bool isProblemSens = (runModeBox_->currentData().toInt() == 3);
    sensIsProblemMode_ = isProblemSens;

    // Collect (problem, dim) combinations from the panel.
    struct ProbDim { QString problem; int dim; };
    QVector<ProbDim> probDims;

    if (isProblemSens) {
      // Problem sensitivity mode: use the currently selected problem + dim.
      const QString prob = currentProblemShort();
      const int dim = dimSpin_ ? dimSpin_->value() : 2;
      if (prob.isEmpty()) {
        QMessageBox::warning(this, "Problem sensitivity",
          "No problem selected.\n\nSelect a problem from the dropdown.");
        return;
      }
      probDims.push_back({prob, dim});
    } else if (sensProblemsList_ && sensProblemsList_->selectedItems().count() > 0) {
      const int defaultDim = dimSpin_ ? dimSpin_->value() : 2;
      const int maxDim     = dimSpin_ ? dimSpin_->maximum() : 100000;
      for (int r = 0; r < (sensProblemDimsTable_ ? sensProblemDimsTable_->rowCount() : 0); ++r) {
        auto* probItem = sensProblemDimsTable_->item(r, 0);
        if (!probItem) continue;
        // The cell shows problemDisplayName(); the short name is its data.
        QString prob = probItem->data(Qt::UserRole).toString().trimmed();
        if (prob.isEmpty()) prob = probItem->text().trimmed();
        if (prob.isEmpty()) continue;
        QVector<int> dims;
        const int fixed = fixedDimForProblem(prob);
        if (fixed > 0) {
          dims = {fixed};
        } else if (auto* w = sensProblemDimsTable_->cellWidget(r, 1)) {
          if (auto* le = qobject_cast<QLineEdit*>(w)) {
            const bool restricted = !allowedDimsForProblem(prob).empty();
            dims = parseBatchDimensionList(le->text(), defaultDim, 2, restricted ? 0 : maxDim);
            if (restricted) dims = restrictToAllowedDims(prob, dims);
          }
        }
        if (dims.isEmpty()) dims = {defaultDim};
        for (int d : dims) probDims.push_back({prob, d});
      }
    }
    if (probDims.isEmpty()) {
      QMessageBox::warning(this, "Sensitivity run",
        "No problems selected.\n\nSelect one or more problems in the 'Problems' panel.");
      return;
    }

    // Collect parameters + values from sensitivity table.
    // Each row: param name, whether to analyze, comma-separated values.
    struct SweepParam { QString name; QStringList values; };
    QVector<SweepParam> sweepParams;
    if (sensitivityTable_) {
      for (int r = 0; r < sensitivityTable_->rowCount(); ++r) {
        auto* pItem = sensitivityTable_->item(r, 0);
        auto* aItem = sensitivityTable_->item(r, 1);
        auto* vItem = sensitivityTable_->item(r, 2);
        if (!pItem || !aItem || !vItem) continue;
        if (aItem->checkState() != Qt::Checked) continue;
        const QString param = pItem->text().trimmed();
        const QStringList vals = vItem->text().split(',', Qt::SkipEmptyParts);
        if (param.isEmpty() || vals.isEmpty()) continue;
        QStringList trimmed;
        for (const QString& v : vals) trimmed << v.trimmed();
        sweepParams.push_back({param, trimmed});
      }
    }
    if (sweepParams.isEmpty()) {
      QMessageBox::warning(this, "Sensitivity run",
        "No parameters configured for sensitivity analysis.\n\n"
        "Check ✓ 'Analyze' and fill 'Values' in Settings → Sensitivity.");
      return;
    }

    // Build cartesian product of parameter values (grid search).
    // E.g. delta=[0.1,0.3] × alpha=[0.5,0.8] → 4 combinations.
    // NOTE: For problem sensitivity (mode 3), we use OAT (One-At-a-Time)
    // instead of cartesian product — each parameter is swept individually
    // while others stay at their default values.  This produces one output
    // tab per parameter.
    // For method sensitivity (mode 2), we keep the existing cartesian product
    // behaviour for backward compatibility.
    const bool useOAT = isProblemSens;

    // Save original param values so we can restore after the run.
    sensOrigMethodParams_.clear();
    sensOrigProblemParams_.clear();
    const QString methodSection = currentMethodShort();
    if (isProblemSens) {
      // For problem sensitivity: save originals from the problem section.
      const QString problemSection = currentProblemShort();
      for (const auto& sp : sweepParams) {
        sensOrigProblemParams_[sp.name] = cfg_ ? cfg_->value(problemSection, sp.name) : QString();
      }
    } else {
      // For method sensitivity: save originals from the method section.
      for (const auto& sp : sweepParams) {
        sensOrigMethodParams_[sp.name] = cfg_ ? cfg_->value(methodSection, sp.name) : QString();
      }
    }

    // Build sensGroups_ and sensJobQueue_.
    sensGroups_.clear();
    sensJobQueue_.clear();
    sensJobIndex_ = 0;
    sensGroupTabIdx_.clear();

    if (useOAT) {
      // ── OAT mode: one group per parameter ──────────────────────────
      // Each group sweeps ONE parameter across its values, keeping all
      // others at their default.  This produces one tab per parameter.
      for (int sp = 0; sp < sweepParams.size(); ++sp) {
        const SweepParam& param = sweepParams[sp];

        for (int gi = 0; gi < probDims.size(); ++gi) {
          SensGroup group;
          group.method      = methodSection;
          group.problem     = probDims[gi].problem;
          group.dim         = probDims[gi].dim;
          group.sweepParams = QStringList{param.name};  // single param per group
          const int groupId = static_cast<int>(sensGroups_.size());
          sensGroups_.push_back(group);

          for (int vi = 0; vi < param.values.size(); ++vi) {
            SensQueueJob job;
            job.problem        = probDims[gi].problem;
            job.dim            = probDims[gi].dim;
            job.injectedParams = {{param.name, param.values[vi]}};
            job.groupId        = groupId;
            job.pointIdx       = vi;
            sensJobQueue_.push_back(job);
          }
        }
      }
    } else {
      // ── Cartesian mode (method sensitivity): all params combined ────
      QVector<QMap<QString,QString>> grid;
      grid.push_back({});
      for (const auto& sp : sweepParams) {
        QVector<QMap<QString,QString>> expanded;
        for (const auto& existing : grid) {
          for (const QString& val : sp.values) {
            QMap<QString,QString> combo = existing;
            combo[sp.name] = val;
            expanded.push_back(combo);
          }
        }
        grid = expanded;
      }

      QStringList paramNames;
      for (const auto& sp : sweepParams) paramNames << sp.name;

      for (int gi = 0; gi < probDims.size(); ++gi) {
        SensGroup group;
        group.method      = methodSection;
        group.problem     = probDims[gi].problem;
        group.dim         = probDims[gi].dim;
        group.sweepParams = paramNames;
        const int groupId = static_cast<int>(sensGroups_.size());
        sensGroups_.push_back(group);

        for (int pi = 0; pi < grid.size(); ++pi) {
          SensQueueJob job;
          job.problem        = probDims[gi].problem;
          job.dim            = probDims[gi].dim;
          job.injectedParams = grid[pi];
          job.groupId        = groupId;
          job.pointIdx       = pi;
          sensJobQueue_.push_back(job);
        }
      }
    }

    startNextSensJob();
    return;
  }

  QString method = currentMethodShort();
  // Text-based as before (editable combo), minus the "(D: ...)" display suffix.
  QString problem = problemNameFromDisplay(problemBox_->currentText());
  if (method.isEmpty() || problem.isEmpty()) {
    QMessageBox::warning(this, "Missing selection", "Please select a method and a problem.");
    return;
  }

  // Create a dedicated output tab for this run: <method>+<problem> (short names).
  // Store the effective dimension for later legend rendering.
  const bool isFixed = (currentFixedDim_ > 0) || (!dimSpin_->isEnabled());
  const int dimUi = dimSpin_->value();
  const int dimEff = (isFixed && currentFixedDim_ > 0) ? currentFixedDim_ : dimUi;

  const int outIdx = createOutputRunTab(method, problem, dimEff);
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  // Decide args: omit dimension for fixed-D problems (if known)
  const int dim = dimUi;

  QStringList args;
  args << method << problem;
  if (!isFixed) args << QString::number(dim);

  // Remember last run (for auto-retry and UI sync)
  lastArgs_ = args;
  lastHadDimension_ = !isFixed;
  lastAutoRetriedFixedDim_ = false;
  processTextBuffer_.clear();

  appendLog("----------------------------------------");
  appendLog("Running: " + cliPath() + " " + args.join(' '));

  startCliProcess(args);
}


void MainWindow::requestStop() {
  if (!proc_) return;
  if (proc_->state() == QProcess::NotRunning) return;

  appendLog("Stopping...");
  proc_->terminate();
  if (!proc_->waitForFinished(2000)) {
    proc_->kill();
  }
}

void MainWindow::updateStatsTabsEnabled() {
  if (!statsTabs_) return;
  // Tab 0 = Results Table — always enabled during/after batch.
  // Tabs 1..N = analysis tabs — disabled during run OR when < 2 methods.
  const bool running = batchActive_ || (proc_ && proc_->state() != QProcess::NotRunning);
  const int methodCount = batchMethodsList_ ? batchMethodsList_->count() : 0;
  const bool analysisOk = !running && (methodCount >= 2);
  const int n = statsTabs_->count();
  for (int i = 1; i < n; ++i)
    statsTabs_->setTabEnabled(i, analysisOk);
}

void MainWindow::onNewMethod() {
  optimsolution::NewMethodDialog dlg(projectRoot_, this);
  dlg.exec();
}

void MainWindow::onNewProblem() {
  optimsolution::NewProblemDialog dlg(projectRoot_, this);
  dlg.exec();
}

void MainWindow::onDeleteMethod() {
  optimsolution::DeleteItemDialog dlg(true, projectRoot_, this);
  if (dlg.exec() == QDialog::Accepted)
    onRefreshFactory();  // refresh method/problem lists in GUI
}

void MainWindow::onDeleteProblem() {
  optimsolution::DeleteItemDialog dlg(false, projectRoot_, this);
  if (dlg.exec() == QDialog::Accepted)
    onRefreshFactory();
}

void MainWindow::updateRunBtn(bool running) {
  if (!runBtn_) return;
  if (running) {
    // Red square = stop
    runBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_MediaStop), QColor(210, 0, 0), QSize(20, 20), 2.0));
    runBtn_->setStyleSheet("QPushButton { color: #cc0000; font-weight: bold; }");
    const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
    runBtn_->setText(mode == 1 ? "Stop batch" : "Stop");
  } else {
    // Green play = run
    const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
    runBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_MediaPlay), QColor(0, 160, 0), QSize(20, 20), 2.0));
    runBtn_->setStyleSheet("QPushButton { color: #00a000; font-weight: bold; }");
    runBtn_->setText(mode == 1 ? "Run batch" : mode == 2 ? "Run sensitivity" : mode == 3 ? "Run problem sens." : "Run");
  }
}

void MainWindow::onRunModeChanged(int /*index*/) {
  const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;

  // Ensure the batch lists are populated the moment the user switches to batch mode.
  if (mode == 1) {
    populateBatchLists();
    onBatchMetricUiChanged();
  }
  if (mode == 5) {
    // Unified pipeline mode reuses the same selection lists as Batch run,
    // but has no post-run statistics UI of its own, so
    // onBatchMetricUiChanged() is deliberately not called here.
    populateBatchLists();
    // Every (re-)entry into pipeline mode starts by clearing any stale
    // in-memory flags (selection/settings may have changed, or output may
    // have been deleted, since an earlier visit) and then immediately
    // re-derives them from whatever is actually on disk in
    // optimsolution_gui_run/ -- so resuming earlier work does not require
    // blindly re-running stages whose output is already there.
    resetPipelineButtonStates();
    detectPipelineExistingOutputs();
  }
  if (mode == 2 || mode == 3) {
    populateBatchLists(); // also populates sensProblemsList_
  }
  if (mode == 6) {
    populateNNDatasetList();
    populateNNOptimizerList();
    refreshNNButtonStates();
    refreshNNSeedInfoLabel();
  }
  if (mode == 7) {
    // Portfolio mode reuses the Batch panel's METHOD list as the candidate
    // pool to switch between; its problems list is not used (Portfolio
    // always targets the single problem/dim picked in the main dropdown,
    // exactly like Single run).
    populateBatchLists();
  }

  // Keep the runs info label up to date.
  if (auto* lbl = findChild<QLabel*>("selectionRunsLbl"))
    lbl->setText(QString::number(globalRunsFromSettings()));

  // When switching away from Sensitivity run mode, immediately clear the
  // sensitivity enable flag in the config so it does not bleed into subsequent
  // single/batch runs (even if the user never presses Run).
  if (mode != 2 && mode != 3 && cfg_ && cfg_->isLoaded()) {
    cfg_->setValue("sensitivity", "enabled", "0");
    cfg_->setValue("sensitivity", "enable",  "0");
  }

  // When switching to/from problem sensitivity mode, refresh the sensitivity table
  // to show the appropriate parameters (method params for mode 2, problem params for mode 3).
  if (mode == 3) {
    const QString problem = currentProblemShort();
    if (!problem.isEmpty())
      populateSensitivityTableForProblem(problem);
  } else if (mode == 2) {
    const QString method = currentMethodShort();
    if (!method.isEmpty())
      populateSensitivityTableForMethod(method);
  }

  // Modes that SHOW the Problem/Dimension controls (Single run, Problem
  // sensitivity): bring the dimension spin box back in line with the selected
  // problem (fixed / allowed dimensions). updateDimUiForProblem() leaves the
  // spin box alone while in Batch mode, so it may be stale after a batch.
  if (mode == 0 || mode == 3) {
    updateDimUiForProblem(currentProblemShort());
  }

  updateBatchPanelVisibility();
}

void MainWindow::updateBatchPanelVisibility() {
  const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
  const bool isBatch = (mode == 1);
  const bool isSensitivity = (mode == 2);
  const bool isProblemSensitivity = (mode == 3);
  const bool isAnySensitivity = (isSensitivity || isProblemSensitivity);
  const bool isMultiObjective = (mode == 4);
  const bool isPipeline = (mode == 5);
  const bool isNNLandscape = (mode == 6);
  const bool isPortfolio = (mode == 7);
  // The unified pipeline mode reuses the exact same method/problem selection
  // lists as Batch run (batchMethodsList_/batchProblemsList_ inside
  // batchPanel_), including per-problem dimension overrides -- see
  // onRunPipelineBatch() / onRunPipelineFeatures(). Portfolio mode reuses
  // only the METHODS half of that same panel, as its candidate pool.
  const bool showBatchSelectionLists = (isBatch || isPipeline || isPortfolio);

  if (batchPanel_) batchPanel_->setVisible(showBatchSelectionLists);
  // Portfolio mode now uses the Batch panel's Problems list too (multi-select,
  // with the same per-problem dim overrides Batch run already supports),
  // not just its Methods list -- every candidate method is relayed across
  // every selected problem, with cross-problem ranking/Friedman/Wilcoxon
  // written in addition to the per-problem ones. See onRunClicked().
  if (moPanel_) moPanel_->setVisible(isMultiObjective);
  if (pipelinePanel_) pipelinePanel_->setVisible(isPipeline);
  if (nnLandscapePanel_) nnLandscapePanel_->setVisible(isNNLandscape);
  if (nnActionPanel_) nnActionPanel_->setVisible(isNNLandscape);
  if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setVisible(isBatch || isAnySensitivity);
  if (sensProblemPanel_) sensProblemPanel_->setVisible(isSensitivity);  // only mode 2; mode 3 uses main dropdown
  if (settingsBox_) {
    if (auto* row = settingsBox_->findChild<QWidget*>("batchMethodSettingsRow")) {
      row->setVisible(isBatch);
    }
  }
  // "Batch settings" (Mean/Rate/SD/Time column toggles) only applies to
  // Batch run's own post-run statistics tables -- irrelevant clutter for
  // the pipeline mode, which has no such tables.
  if (auto* box = findChild<QGroupBox*>("batchSettingsBox")) {
    box->setVisible(isBatch);
  }
  // Save/Load batch selection are useful for the pipeline mode too, since
  // it uses the exact same method/problem selection lists.
  if (auto* btn = findChild<QPushButton*>("saveBatchSelectionBtn")) {
    btn->setVisible(isBatch || isPipeline);
  }
  if (auto* btn = findChild<QPushButton*>("loadBatchSelectionBtn")) {
    btn->setVisible(isBatch || isPipeline);
  }
  if (nnSaveSelectionBtn_) nnSaveSelectionBtn_->setVisible(isNNLandscape);
  if (nnLoadSelectionBtn_) nnLoadSelectionBtn_->setVisible(isNNLandscape);
  // "Refresh plots" re-loads convergence/sensitivity plots, which the
  // pipeline mode and mode 7 (NN landscape) have none of.
  if (reloadConvergenceBtn_) reloadConvergenceBtn_->setVisible(!isPipeline && !isNNLandscape);
  // "Load previous experiment" only makes sense for the pipeline mode.
  if (pipelineLoadExperimentBtn_) pipelineLoadExperimentBtn_->setVisible(isPipeline);
  if (pipelineLoadExperimentSpinner_) pipelineLoadExperimentSpinner_->setVisible(isPipeline);
  // Mode 7's own "Load previous experiment...", same idea, different data shape.
  if (nnLoadExperimentBtn_) nnLoadExperimentBtn_->setVisible(isNNLandscape);
  if (nnLoadExperimentSpinner_) nnLoadExperimentSpinner_->setVisible(isNNLandscape);
  // The pipeline mode has its own 4 dedicated buttons instead of the shared
  // Run/Stop button -- there is no single "the" action to toggle anymore.
  // busySpinner_ only ever spins for that shared Run/Stop button, so it's
  // hidden alongside it here (the pipeline stages have their own spinners).
  // Mode 7 (NN landscape) follows the exact same convention with its own 2
  // dedicated buttons.
  if (runBtn_) runBtn_->setVisible(!isPipeline && !isNNLandscape);
  if (busySpinner_) busySpinner_->setVisible(!isPipeline && !isNNLandscape);

  // In sensitivity mode: method stays visible, problem/dim are hidden (panel used instead).
  // In problem sensitivity mode: both method and problem stay visible (sweep is on problem params).
  // In batch mode / multi-objective mode / pipeline mode / NN landscape mode: all
  // single-run dropdowns are hidden (their own panel is used instead; the pipeline
  // mode is problem-only for stage 2 but method+problem for stage 1, both covered
  // by batchPanel_; NN landscape mode has its own dataset/optimizer selection).
  const bool showMethod   = !isBatch && !isPipeline && !isMultiObjective && !isNNLandscape && !isPortfolio;
  const bool showProbDim  = !isBatch && !isPipeline && !isSensitivity && !isMultiObjective && !isNNLandscape && !isPortfolio;  // mode 3 keeps problem/dim visible; mode 7 (Portfolio) now sources problems from the Batch panel's Problems list instead (multi-problem support)

  if (methodBox_) methodBox_->setVisible(showMethod);
  if (problemBox_) problemBox_->setVisible(showProbDim);
  if (dimSpin_) dimSpin_->setVisible(showProbDim);
  // "Samples per landscape" (Selection, below Run mode) only matters for
  // the pipeline mode's stage 2.
  if (pipelineSamplesSpin_) pipelineSamplesSpin_->setVisible(isPipeline);

  if (selectionForm_) {
    if (auto* lbl = selectionForm_->labelForField(methodBox_)) lbl->setVisible(showMethod);
    if (auto* lbl = selectionForm_->labelForField(problemBox_)) lbl->setVisible(showProbDim);
    if (auto* lbl = selectionForm_->labelForField(dimSpin_)) lbl->setVisible(showProbDim);
    if (auto* lbl = selectionForm_->labelForField(pipelineSamplesSpin_)) lbl->setVisible(isPipeline);
  }

  // Update run button to reflect the active mode.
  if (!batchActive_ && (!proc_ || proc_->state() == QProcess::NotRunning)) {
    updateRunBtn(false);
  }

  // The "Enable sensitivity analysis" checkbox is obsolete — sensitivity is now
  // controlled by the Run mode selector. Keep it hidden in all modes.
  if (sensitivityEnableChk_) {
    sensitivityEnableChk_->setVisible(false);
  }

  // In Sensitivity or Problem Sensitivity run mode, ensure the sensitivity table
  // is always editable (no need to check the enable checkbox).
  if (sensitivityTable_) {
    if (isAnySensitivity) {
      sensitivityTable_->setEnabled(true);
    }
  }
}

void MainWindow::populateBatchLists() {
  if (!batchMethodsList_ || !batchProblemsList_) {
    return;
  }

  QStringList prevMethods = selectedBatchMethodShortNames();
  QStringList prevProblems = selectedBatchProblemShortNames();
  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 && active->page && g_batchSummarySnapshots.contains(active->page)) {
      const BatchSummarySnapshot snap = g_batchSummarySnapshots.value(active->page);
      if (prevMethods.isEmpty()) prevMethods = snap.methods;
      if (prevProblems.isEmpty()) prevProblems = snap.problems;
      if (batchProblemDimOverride_.isEmpty() && !snap.problemDimOverrides.isEmpty()) {
        batchProblemDimOverride_ = snap.problemDimOverrides;
      }
    }
  }

  batchMethodsList_->clear();
  for (int i = 0; i < methodBox_->count(); ++i) {
    const QString shortName = methodBox_->itemData(i).toString();
    const QString fullDisplay = methodBox_->itemText(i);
    const QString display = shortName.isEmpty() ? fullDisplay : shortName;
    auto* it = new QListWidgetItem(display);
    const QString key = shortName.isEmpty() ? fullDisplay : shortName;
    it->setData(Qt::UserRole, key);
    if (display != fullDisplay) {
      it->setToolTip(fullDisplay);
    }
    batchMethodsList_->addItem(it);
  }

  batchProblemsList_->clear();
  for (int i = 0; i < problemBox_->count(); ++i) {
    const QString shortName = problemBox_->itemData(i).toString();
    const QString display = shortName.isEmpty() ? problemBox_->itemText(i) : problemDisplayName(shortName);
    auto* it = new QListWidgetItem(display);
    const QString key = shortName.isEmpty() ? display : shortName;
    it->setData(Qt::UserRole, key);
    batchProblemsList_->addItem(it);
  }

  // Preserve the current batch selections when refreshing; otherwise fall back to the current single-run method/problem.
  const QString curM = currentMethodShort();
  const QString curP = currentProblemShort();
  QSet<QString> wantedMethods;
  QSet<QString> wantedProblems;
  if (prevMethods.isEmpty()) {
    if (!curM.isEmpty()) wantedMethods.insert(curM);
  } else {
    for (const QString& m : prevMethods) wantedMethods.insert(m);
  }
  if (prevProblems.isEmpty()) {
    if (!curP.isEmpty()) wantedProblems.insert(batchBaseProblemShort(curP));
  } else {
    for (const QString& p : prevProblems) wantedProblems.insert(batchBaseProblemShort(p));
  }

  for (int i = 0; i < batchMethodsList_->count(); ++i) {
    if (auto* it = batchMethodsList_->item(i)) {
      const QString key = it->data(Qt::UserRole).toString().trimmed();
      it->setSelected(wantedMethods.contains(key));
    }
  }
  for (int i = 0; i < batchProblemsList_->count(); ++i) {
    if (auto* it = batchProblemsList_->item(i)) {
      const QString key = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
      it->setSelected(wantedProblems.contains(key));
    }
  }

  syncBatchProblemDimsTable();

  // Also populate the sensitivity problems list (mirrors batchProblemsList_ but independent).
  if (sensProblemsList_) {
    const QStringList prevSensProblems = [this]() {
      QStringList lst;
      for (int i = 0; i < sensProblemsList_->count(); ++i)
        if (auto* it = sensProblemsList_->item(i); it && it->isSelected())
          lst << batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
      return lst;
    }();

    sensProblemsList_->clear();
    const QString curP = currentProblemShort();
    QSet<QString> wantedSens;
    if (prevSensProblems.isEmpty()) {
      if (!curP.isEmpty()) wantedSens.insert(batchBaseProblemShort(curP));
    } else {
      for (const QString& p : prevSensProblems) wantedSens.insert(p);
    }

    for (int i = 0; i < problemBox_->count(); ++i) {
      const QString shortName = problemBox_->itemData(i).toString();
      const QString display   = shortName.isEmpty() ? problemBox_->itemText(i) : problemDisplayName(shortName);
      const QString key       = shortName.isEmpty() ? display : shortName;
      auto* it = new QListWidgetItem(display);
      it->setData(Qt::UserRole, key);
      sensProblemsList_->addItem(it);
      it->setSelected(wantedSens.contains(batchBaseProblemShort(key)));
    }

    syncSensProblemDimsTable();
  } // end if (sensProblemsList_)

  if (settingsBox_) {
    auto* combo = settingsBox_->findChild<QComboBox*>("batchMethodSettingsCombo");
    if (combo) {
      QString preferredShort = curM;
      if (batchMethodsList_ && batchMethodsList_->currentItem()) {
        const QString curSelected = batchMethodsList_->currentItem()->data(Qt::UserRole).toString().trimmed();
        if (!curSelected.isEmpty()) preferredShort = curSelected;
      }

      QStringList selectedShorts;
      for (int i = 0; i < batchMethodsList_->count(); ++i) {
        if (auto* it = batchMethodsList_->item(i); it && it->isSelected()) {
          const QString shortName = it->data(Qt::UserRole).toString().trimmed();
          if (!shortName.isEmpty() && !selectedShorts.contains(shortName)) {
            selectedShorts << shortName;
          }
        }
      }

      {
        QSignalBlocker blocker(combo);
        combo->clear();
        for (const QString& shortName : selectedShorts) {
          QString display = shortName;
          int idx = methodBox_->findData(shortName);
          if (idx < 0) idx = methodBox_->findText(shortName);
          if (idx >= 0) display = methodBox_->itemText(idx);
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
}

QString MainWindow::batchBaseProblemShort(const QString& problemKey) const {
  QString key = problemKey.trimmed();
  static const QRegularExpression re(QStringLiteral(R"(^(.*)\s+\[d=(\d+)\]\s*$)"),
                                     QRegularExpression::CaseInsensitiveOption);
  const QRegularExpressionMatch m = re.match(key);
  if (m.hasMatch()) {
    key = m.captured(1).trimmed();
  }
  return key;
}

QString MainWindow::batchProblemDisplayKey(const QString& problemShort, int dim) const {
  const QString base = batchBaseProblemShort(problemShort);
  const int fixed = fixedDimForProblem(base);
  if (fixed > 0) {
    return base;
  }
  if (dim > 0) {
    return QString("%1 [d=%2]").arg(base).arg(dim);
  }
  return base;
}

QVector<int> MainWindow::batchDimsForProblem(const QString& problemShort, int defaultDim) const {
  const QString base = batchBaseProblemShort(problemShort);
  const int fixed = fixedDimForProblem(base);
  if (fixed > 0) {
    return QVector<int>{fixed};
  }

  static const QRegularExpression re(QStringLiteral(R"(\[d=(\d+)\]\s*$)"),
                                     QRegularExpression::CaseInsensitiveOption);
  const QRegularExpressionMatch m = re.match(problemShort.trimmed());
  if (m.hasMatch()) {
    bool ok = false;
    const int explicitDim = m.captured(1).toInt(&ok);
    if (ok && explicitDim > 1) {
      return QVector<int>{explicitDim};
    }
  }

  QVector<int> dims;
  QSet<int> seen;
  const QStringList stored = batchProblemDimOverride_.value(base);
  for (const QString& part : stored) {
    bool ok = false;
    int v = part.trimmed().toInt(&ok);
    if (!ok || v <= 1) continue;
    if (!seen.contains(v)) {
      seen.insert(v);
      dims.push_back(v);
    }
  }

  // Restricted-dimension problems (CEC2017 / CEC2022): without an entry of
  // their own they get ALL supported dimensions (not the default spin box
  // value, which may be unsupported); a stored entry keeps only its
  // supported values (also cleans settings saved before this rule).
  if (!allowedDimsForProblem(base).empty()) {
    return restrictToAllowedDims(base, dims);
  }

  if (dims.isEmpty()) {
    dims.push_back(std::max(2, defaultDim));
  }

  std::sort(dims.begin(), dims.end());
  return dims;
}

int MainWindow::batchDimForProblem(const QString& problemShort, int defaultDim) const {
  const QVector<int> dims = batchDimsForProblem(problemShort, defaultDim);
  if (!dims.isEmpty()) {
    return dims.front();
  }
  return std::max(2, defaultDim);
}

void MainWindow::syncBatchProblemDimsTable() {
  if (!batchProblemDimsTable_ || !batchProblemsList_ || !dimSpin_) {
    return;
  }

  QScopedValueRollback<bool> guard(internalUpdate_, true);

  QStringList probs;
  const QList<QListWidgetItem*> selectedItems = batchProblemsList_->selectedItems();
  for (QListWidgetItem* it : selectedItems) {
    const QString p = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
    if (!p.isEmpty() && !probs.contains(p)) {
      probs << p;
    }
  }

  // Drop overrides for problems that are no longer selected.
  const QSet<QString> selectedSet = QSet<QString>(probs.begin(), probs.end());
  for (auto it = batchProblemDimOverride_.begin(); it != batchProblemDimOverride_.end(); ) {
    if (!selectedSet.contains(it.key())) {
      it = batchProblemDimOverride_.erase(it);
    } else {
      ++it;
    }
  }

  batchProblemDimsTable_->clearContents();
  batchProblemDimsTable_->setRowCount(probs.size());

  const int defaultDim = std::max(2, dimSpin_->value());
  const int maxDim = dimSpin_->maximum();
  for (int r = 0; r < probs.size(); ++r) {
    const QString p = probs[r];

    auto* probItem = new QTableWidgetItem(problemDisplayName(p));
    probItem->setData(Qt::UserRole, p);
    probItem->setToolTip(problemDisplayName(p));   // full text if the column is narrow
    probItem->setFlags(probItem->flags() & ~Qt::ItemIsEditable);
    batchProblemDimsTable_->setItem(r, 0, probItem);

    const int fixed = fixedDimForProblem(p);
    if (fixed > 0) {
      auto* dimItem = new QTableWidgetItem(QString::number(fixed));
      dimItem->setFlags(dimItem->flags() & ~Qt::ItemIsEditable);
      batchProblemDimsTable_->setItem(r, 1, dimItem);
      continue;
    }

    const bool restricted = !allowedDimsForProblem(p).empty();
    QVector<int> dims;
    if (restricted) {
      // All supported dimensions are filled in automatically; the user only
      // removes the ones not wanted.
      dims = batchDimsForProblem(p, defaultDim);
    } else if (batchProblemDimOverride_.contains(p)) {
      dims = parseBatchDimensionList(batchProblemDimOverride_.value(p).join(", "), defaultDim, 2, maxDim);
    } else {
      dims = QVector<int>{defaultDim};
    }

    auto* line = new QLineEdit(batchProblemDimsTable_);
    line->setPlaceholderText("e.g. 10, 30");
    if (restricted) {
      const QString all = batchDimensionListToText(restrictToAllowedDims(p, {}));
      line->setPlaceholderText(all);
      line->setToolTip(QString("%1 supports only D = %2. All are filled in; delete the ones you "
                               "do not want (an empty box restores all).").arg(p, all));
    }
    line->setMinimumWidth(120);
    line->setText(batchDimensionListToText(dims));

    connect(line, &QLineEdit::editingFinished, this, [this, line, p, defaultDim, maxDim, restricted]() {
      // Restricted problems: no clamping to the spin box range (it belongs
      // to the Single-run problem), then keep only supported values.
      QVector<int> dims = parseBatchDimensionList(line->text(), defaultDim, 2, restricted ? 0 : maxDim);
      if (restricted) {
        const QVector<int> kept = restrictToAllowedDims(p, dims);
        QStringList dropped;
        for (int d : dims) if (!kept.contains(d)) dropped << QString::number(d);
        if (!dropped.isEmpty() && !line->text().trimmed().isEmpty()) {
          appendLog(QString("%1: dimension(s) %2 not supported -- removed (supported: %3).")
                      .arg(p, dropped.join(", "), batchDimensionListToText(restrictToAllowedDims(p, {}))));
        }
        dims = kept;
      }
      const QString canonicalText = batchDimensionListToText(dims);
      {
        QSignalBlocker blocker(line);
        line->setText(canonicalText);
      }

      const bool isDefault = restricted ? (dims == restrictToAllowedDims(p, {}))
                                        : (dims.size() == 1 && dims.front() == defaultDim);
      if (isDefault) {
        batchProblemDimOverride_.remove(p);
      } else {
        QStringList parts;
        for (int d : dims) parts << QString::number(d);
        batchProblemDimOverride_[p] = parts;
      }

      // NOTE: no invalidateBatchProblemCache(p) here any more. Results are
      // stored per dimension ("ackley [d=10]", "ackley [d=30]"), so changing
      // the dimension list only changes WHICH results are shown/run; the
      // results of the other dimensions stay valid and are reused (instead
      // of being deleted and re-run) when that dimension is selected again.
      refreshBatchSelectionView();
    });

    batchProblemDimsTable_->setCellWidget(r, 1, line);
  }

  batchProblemDimsTable_->setVisible(!probs.isEmpty());
  batchProblemDimsTable_->setColumnWidth(1, 160);
}

void MainWindow::syncSensProblemDimsTable() {
  if (!sensProblemDimsTable_ || !sensProblemsList_ || !dimSpin_) return;

  QScopedValueRollback<bool> guard(internalUpdate_, true);

  QStringList probs;
  for (QListWidgetItem* it : sensProblemsList_->selectedItems()) {
    const QString p = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
    if (!p.isEmpty() && !probs.contains(p)) probs << p;
  }

  sensProblemDimsTable_->clearContents();
  sensProblemDimsTable_->setRowCount(probs.size());

  const int defaultDim = std::max(2, dimSpin_->value());
  const int maxDim     = dimSpin_->maximum();

  for (int r = 0; r < probs.size(); ++r) {
    const QString p = probs[r];
    auto* probItem = new QTableWidgetItem(problemDisplayName(p));
    probItem->setData(Qt::UserRole, p);
    probItem->setToolTip(problemDisplayName(p));   // full text if the column is narrow
    probItem->setFlags(probItem->flags() & ~Qt::ItemIsEditable);
    sensProblemDimsTable_->setItem(r, 0, probItem);

    const int fixed = fixedDimForProblem(p);
    if (fixed > 0) {
      auto* dimItem = new QTableWidgetItem(QString::number(fixed));
      dimItem->setFlags(dimItem->flags() & ~Qt::ItemIsEditable);
      sensProblemDimsTable_->setItem(r, 1, dimItem);
      continue;
    }

    const bool restricted = !allowedDimsForProblem(p).empty();
    auto* line = new QLineEdit(sensProblemDimsTable_);
    line->setPlaceholderText("e.g. 10, 30");
    line->setMinimumWidth(120);
    if (restricted) {
      const QString all = batchDimensionListToText(restrictToAllowedDims(p, {}));
      line->setPlaceholderText(all);
      line->setToolTip(QString("%1 supports only D = %2. All are filled in; delete the ones you "
                               "do not want (an empty box restores all).").arg(p, all));
      line->setText(all);
    } else {
      line->setText(QString::number(defaultDim));
    }
    connect(line, &QLineEdit::editingFinished, this, [this, line, p, defaultDim, maxDim, restricted]() {
      QVector<int> dims = parseBatchDimensionList(line->text(), defaultDim, 2, restricted ? 0 : maxDim);
      if (restricted) {
        const QVector<int> kept = restrictToAllowedDims(p, dims);
        QStringList dropped;
        for (int d : dims) if (!kept.contains(d)) dropped << QString::number(d);
        if (!dropped.isEmpty() && !line->text().trimmed().isEmpty()) {
          appendLog(QString("%1: dimension(s) %2 not supported -- removed (supported: %3).")
                      .arg(p, dropped.join(", "), batchDimensionListToText(restrictToAllowedDims(p, {}))));
        }
        dims = kept;
      }
      QSignalBlocker b(line);
      line->setText(batchDimensionListToText(dims));
    });
    sensProblemDimsTable_->setCellWidget(r, 1, line);
  }

  sensProblemDimsTable_->setVisible(!probs.isEmpty());
  sensProblemDimsTable_->setColumnWidth(1, 160);
}

void MainWindow::bindBatchSummaryUiForPage(QWidget* page) {
  if (!page || !g_batchSummaryUi.contains(page)) {
    return;
  }

  const BatchSummaryUiBundle ui = g_batchSummaryUi.value(page);
  statsTabs_ = ui.statsTabs;
  exportStatsBtn_ = ui.exportStatsBtn;
  statsTable_ = ui.statsTable;
  statsNoteLbl_ = ui.statsNoteLbl;
  generateReportBtn_ = ui.generateReportBtn;
  wilcoxonPairsCombo_ = ui.wilcoxonPairsCombo;
  statsAlphaCombo_ = ui.statsAlphaCombo;
  exportWilcoxonPlotBtn_ = ui.exportWilcoxonPlotBtn;
  wilcoxonPlot_ = ui.wilcoxonPlot;
  wilcoxonSummaryLbl_ = ui.wilcoxonSummaryLbl;
  exportRankPlotBtn_ = ui.exportRankPlotBtn;
  rankPlot_ = ui.rankPlot;
  statsSummaryLbl_ = ui.statsSummaryLbl;
  statsPairwiseTable_ = ui.statsPairwiseTable;
  exportCdPlotBtn_ = ui.exportCdPlotBtn;
  cdPlot_ = ui.cdPlot;
  cdSummaryLbl_ = ui.cdSummaryLbl;
  // Bind metric/aggregate member pointers to the primary (Wilcoxon) combos.
  batchMetricCombo_ = ui.metricCombo;
  batchAggCombo_    = ui.aggCombo;
}

void MainWindow::refreshBatchSelectionView() {
  if (batchActive_) {
    return;
  }

  if (const auto* active = activeOutputTab()) {
    if (active->methodShort.compare("System", Qt::CaseInsensitive) == 0 && active->page) {
      g_activeBatchSummaryPage = active->page;
    }
  }
  if (g_activeBatchSummaryPage) {
    bindBatchSummaryUiForPage(g_activeBatchSummaryPage);
  }

  QStringList methods;
  QStringList problems;

  // Full experiment lists of the active batch summary (never shrunk by a
  // deselection; see BatchSummarySnapshot::allMethods).
  QStringList experimentMethods;
  QStringList experimentProblems;

  if (g_activeBatchSummaryPage && g_batchSummarySnapshots.contains(g_activeBatchSummaryPage)) {
    const BatchSummarySnapshot existing = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    experimentMethods = batchExperimentMethods(existing);
    experimentProblems = batchExperimentProblems(existing);
    batchCells_ = batchSummaryCellsByPage_.value(g_activeBatchSummaryPage);
    batchCachedProblemDims_ = existing.cachedProblemDims;
    g_liveBatchCsvPaths = existing.csvPaths;

    // Per-problem dimension overrides. syncBatchProblemDimsTable() (run just
    // before this on every Problems-list selection change) drops the override
    // of every problem that is NOT selected; take those back from the
    // experiment's snapshot so that a problem hidden for an ablation study
    // comes back with its own dimensions (e.g. "ackley [d=10]" and
    // "ackley [d=30]") when it is selected again. For the problems that ARE
    // selected the current overrides stay authoritative (they may have just
    // been edited in the Dims table or loaded from a .batchsel file).
    {
      QSet<QString> selectedBases;
      if (batchProblemsList_) {
        for (QListWidgetItem* it : batchProblemsList_->selectedItems()) {
          const QString b = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
          if (!b.isEmpty()) selectedBases.insert(b);
        }
      }
      for (auto it = existing.problemDimOverrides.constBegin(); it != existing.problemDimOverrides.constEnd(); ++it) {
        if (!selectedBases.contains(it.key()) && !batchProblemDimOverride_.contains(it.key())) {
          batchProblemDimOverride_.insert(it.key(), it.value());
        }
      }
    }

    if (batchCells_.isEmpty() && !existing.csvPaths.isEmpty()) {
      QMap<QString, QSet<int>> restoredDimsByBase;
      for (auto pit = existing.csvPaths.constBegin(); pit != existing.csvPaths.constEnd(); ++pit) {
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

    // Read the Selection area only AFTER the per-problem dimension overrides
    // of this experiment are back in place: syncBatchProblemDimsTable() (run
    // just before this, on every Problems-list selection change) drops the
    // override of every problem that is not selected, so a problem that is
    // re-selected after being hidden must get its experiment dimensions
    // (e.g. "ackley [d=10]", "ackley [d=30]") from the restored overrides,
    // not from the default dimension.
    methods = selectedBatchMethodShortNames();
    problems = selectedBatchProblemShortNames();
    if (methods.isEmpty()) methods = existing.methods;
    if (problems.isEmpty()) problems = existing.problems;

    // Ablation view: only the selected methods/problems are shown in the
    // Output tabs. Methods are already alphabetical; problems keep the
    // experiment's own row order, so a problem that is hidden and shown again
    // returns to exactly the row it had (instead of being appended last).
    problems = batchOrderLike(problems, experimentProblems);
  } else {
    methods = selectedBatchMethodShortNames();
    problems = selectedBatchProblemShortNames();
  }

  if (g_activeBatchSummaryPage && !internalUpdate_) {
    BatchSummarySnapshot snap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    // Remember the full experiment once, so later deselections (which
    // overwrite snap.methods/problems below) never shrink it.
    if (snap.allMethods.isEmpty()) snap.allMethods = experimentMethods;
    if (snap.allProblems.isEmpty()) snap.allProblems = experimentProblems;
    if (!methods.isEmpty()) snap.methods = methods;
    if (!problems.isEmpty()) snap.problems = problems;
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

  if (!statsTable_ || methods.isEmpty() || problems.isEmpty()) {
    if (statsTable_) {
      initStatsTableForBatch(methods, problems);
      finalizeStatsTableAfterBatch();
    }
    rebuildStatsComparisons();
    return;
  }

  initStatsTableForBatch(methods, problems);
  for (const QString& prob : problems) {
    const int row = statsRowByProblem_.value(prob, -1);
    if (row >= 0 && statsTable_) {
      const int dim = batchCachedProblemDims_.contains(prob)
                        ? batchCachedProblemDims_.value(prob)
                        : batchDimForProblem(prob, dimSpin_ ? std::max(2, dimSpin_->value()) : 2);
      if (auto* dimItem = statsTable_->item(row, 1)) {
        dimItem->setText(QString::number(dim));
      }
    }
    for (const QString& meth : methods) {
      updateStatsTableCell(prob, meth);
    }
  }
  finalizeStatsTableAfterBatch();
  rebuildStatsComparisons();
}

bool MainWindow::batchHasCachedCell(const QString& problemShort, const QString& methodShort, int dim) const {
  if (!batchCells_.contains(problemShort)) return false;
  if (!batchCells_.value(problemShort).contains(methodShort)) return false;
  const int cachedDim = batchCachedProblemDims_.value(problemShort, -1);
  if (dim > 0 && cachedDim > 0 && cachedDim != dim) return false;
  return true;
}

void MainWindow::invalidateBatchProblemCache(const QString& problemShort) {
  const QString baseProblem = batchBaseProblemShort(problemShort);
  if (baseProblem.isEmpty()) return;

  QStringList keysToRemove;
  for (auto it = batchCells_.begin(); it != batchCells_.end(); ++it) {
    if (batchBaseProblemShort(it.key()) == baseProblem) {
      keysToRemove << it.key();
    }
  }
  for (const QString& key : keysToRemove) {
    batchCells_.remove(key);
    batchCachedProblemDims_.remove(key);
    g_liveBatchCsvPaths.remove(key);
  }

  if (g_activeBatchSummaryPage) {
    BatchSummarySnapshot snap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    snap.csvPaths = g_liveBatchCsvPaths;
    snap.cachedProblemDims = batchCachedProblemDims_;
    snap.problemDimOverrides = batchProblemDimOverride_;
    g_batchSummarySnapshots.insert(g_activeBatchSummaryPage, snap);
    batchSummaryCellsByPage_[g_activeBatchSummaryPage] = batchCells_;
  }
}


void MainWindow::startBatchLogWriter() {
  if (batchLogThread_ || batchLogWriter_) {
    return;
  }

  // Log file writing to disk is disabled — batch output is shown in the UI only.
  batchLogFilePath_.clear();

  if (!batchUiFlushTimer_) {
    batchUiFlushTimer_ = new QTimer(this);
    batchUiFlushTimer_->setInterval(250);
    connect(batchUiFlushTimer_, &QTimer::timeout, this, &MainWindow::flushBatchUiLog);
  }
  batchUiPendingLog_.clear();
  batchProgressThrottle_.invalidate();
  batchUiFlushTimer_->start();
}

void MainWindow::stopBatchLogWriter() {
  if (batchUiFlushTimer_) {
    batchUiFlushTimer_->stop();
  }
  flushBatchUiLog();

  batchLogThread_ = nullptr;
  batchLogWriter_ = nullptr;
  batchLogFilePath_.clear();
  batchUiPendingLog_.clear();
  batchProgressThrottle_.invalidate();
}

void MainWindow::batchHandleProcessOutput(const QString& cleaned) {
  // Keep recent output only; it is used for parsing summary fields at the end.
  constexpr int kMaxBufferChars = 400000;
  processTextBuffer_ += cleaned;
  if (processTextBuffer_.size() > kMaxBufferChars) {
    processTextBuffer_.remove(0, processTextBuffer_.size() - kMaxBufferChars);
  }

  if (batchLogWriter_) {
    BatchLogWriter* writer = batchLogWriter_;
    const QByteArray bytes = cleaned.toUtf8();
    QMetaObject::invokeMethod(writer, [writer, bytes]() { writer->appendUtf8(bytes); }, Qt::QueuedConnection);
  }

  batchUiPendingLog_ += cleaned;

  // Progress is driven by onProgressPollTick() which polls the convergence CSV.
}

void MainWindow::flushBatchUiLog() {
  if (batchUiPendingLog_.isEmpty()) {
    return;
  }

  const QString pending = batchUiPendingLog_;
  batchUiPendingLog_.clear();

  QStringList lines = pending.split(QRegularExpression(R"(\r\n|\n|\r)"), Qt::SkipEmptyParts);
  if (lines.isEmpty()) {
    return;
  }

  QStringList keep;
  keep.reserve(lines.size());
  for (const QString& l : lines) {
    const QString t = l.trimmed();
    if (t.isEmpty()) continue;
    if (t.contains("Batch job", Qt::CaseInsensitive) ||
        t.contains("RUN SUMMARY", Qt::CaseInsensitive) ||
        t.contains("Process finished", Qt::CaseInsensitive) ||
        t.contains("ERROR", Qt::CaseInsensitive) ||
        t.contains("WARNING", Qt::CaseInsensitive) ||
        t.contains("Final best", Qt::CaseInsensitive) ||
        t.contains("best_f", Qt::CaseInsensitive) ||
        t.contains("mean", Qt::CaseInsensitive) ||
        t.startsWith("FixedDimension:", Qt::CaseInsensitive)) {
      keep << t;
    }
  }

  if (keep.isEmpty()) {
    keep << lines.last().trimmed();
  }
  if (keep.size() > 12) {
    keep = keep.mid(keep.size() - 12);
  }

  const QString msg = keep.join("\n");
  if (msg.isEmpty()) {
    return;
  }

  const bool wasBatch = batchActive_;
  batchActive_ = false;
  appendLog(msg);
  batchActive_ = wasBatch;
}

void MainWindow::singleHandleProcessOutput(const QString& cleaned) {
  // Keep recent output only; it is used for parsing summary fields at the end.
  constexpr int kMaxBufferChars = 400000;
  processTextBuffer_ += cleaned;
  if (processTextBuffer_.size() > kMaxBufferChars) {
    processTextBuffer_.remove(0, processTextBuffer_.size() - kMaxBufferChars);
  }

  // Buffer UI log updates and flush on a timer to avoid freezing when stdout is very verbose.
  singleUiPendingLog_ += cleaned;
  if (!singleUiFlushTimer_) {
    singleUiFlushTimer_ = new QTimer(this);
    singleUiFlushTimer_->setInterval(200);
    connect(singleUiFlushTimer_, &QTimer::timeout, this, &MainWindow::flushSingleUiLog);
  }
  if (!singleUiFlushTimer_->isActive()) {
    singleUiFlushTimer_->start();
  }
  // Progress is driven by onProgressPollTick() which polls the convergence CSV.
}

void MainWindow::flushSingleUiLog() {
  if (singleUiPendingLog_.isEmpty()) {
    return;
  }

  const QString pending = singleUiPendingLog_;
  singleUiPendingLog_.clear();

  // Reduce UI workload: keep a small, representative subset of lines.
  QStringList lines = pending.split(QRegularExpression(R"(\r\n|\n|\r)"), Qt::SkipEmptyParts);
  if (lines.isEmpty()) {
    return;
  }

  QStringList trimmed;
  trimmed.reserve(lines.size());
  for (const QString& l : lines) {
    const QString t = l.trimmed();
    if (!t.isEmpty()) trimmed << t;
  }
  if (trimmed.isEmpty()) {
    return;
  }

  QStringList keep;
  keep.reserve(trimmed.size());

  // If a run-summary block exists in this chunk, keep it verbatim (v40 behavior).
  int lastSummaryIdx = -1;
  for (int i = trimmed.size() - 1; i >= 0; --i) {
    const QString& t = trimmed[i];
    if (t.contains("RUN SUMMARY", Qt::CaseInsensitive) ||
        t.contains("RESULT SUMMARY", Qt::CaseInsensitive)) {
      lastSummaryIdx = i;
      break;
    }
  }

  auto looksLikeRule = [](const QString& s, QChar ch) -> bool {
    if (s.size() < 8) return false;
    int cnt = 0;
    for (QChar cc : s) if (cc == ch) ++cnt;
    return cnt >= (s.size() * 8) / 10; // 80% same character
  };

  if (lastSummaryIdx >= 0) {
    const int start = qMax(0, lastSummaryIdx - 3); // include separators above the title
    for (int i = start; i < trimmed.size(); ++i) {
      keep << trimmed[i];
    }
  } else {
    static const QRegularExpression kProgressRe(R"(^\s*\[\s*Run\s+\d+\s*\]\s*)", QRegularExpression::CaseInsensitiveOption);

    for (const QString& t : trimmed) {
      if (kProgressRe.match(t).hasMatch() ||
          looksLikeRule(t, '=') ||
          looksLikeRule(t, '-') ||
          t.contains("Process finished", Qt::CaseInsensitive) ||
          t.contains("ERROR", Qt::CaseInsensitive) ||
          t.contains("WARNING", Qt::CaseInsensitive) ||
          t.contains("Final best", Qt::CaseInsensitive) ||
          t.contains("best_f", Qt::CaseInsensitive) ||
          t.contains("mean", Qt::CaseInsensitive) ||
          t.startsWith("Status:", Qt::CaseInsensitive) ||
          t.startsWith("Method:", Qt::CaseInsensitive) ||
          t.startsWith("Method full name:", Qt::CaseInsensitive) ||
          t.startsWith("Problem:", Qt::CaseInsensitive) ||
          t.startsWith("Best f", Qt::CaseInsensitive) ||
          t.startsWith("Evals:", Qt::CaseInsensitive) ||
          t.startsWith("Time per run", Qt::CaseInsensitive) ||
          t.startsWith("Success rate:", Qt::CaseInsensitive) ||
          t.startsWith("FixedDimension:", Qt::CaseInsensitive)) {
        keep << t;
      }
    }
  }

  // If nothing matched, keep just the last non-empty line.
  if (keep.isEmpty()) {
    keep << trimmed.last();
  }

  // Keep only the most recent lines to avoid long-run slowdown.
  constexpr int kMaxUiLines = 60;
  if (keep.size() > kMaxUiLines) {
    keep = keep.mid(keep.size() - kMaxUiLines);
  }

  const QString msg = keep.join("\n");
  if (msg.isEmpty()) {
    return;
  }

  // Use the v40 log appender so the CLI summary block retains its colors/styling.
  appendLog(msg);

  // Cap document size to avoid quadratic QTextEdit updates.
  OutputRunTab* tab = activeOutputTab();
  if (!tab || !tab->log) {
    return;
  }
  QTextDocument* doc = tab->log->document();

  constexpr int kMaxDocChars = 300000;
  if (doc->characterCount() > kMaxDocChars) {
    constexpr int kTargetChars = 250000;
    const int removeChars = doc->characterCount() - kTargetChars;
    if (removeChars > 0) {
      QTextCursor c(doc);
      c.setPosition(0);
      c.setPosition(removeChars, QTextCursor::KeepAnchor);
      c.removeSelectedText();
    }
  }
}


// ---------------------------------------------------------------------------
// Sensitivity job queue (GUI-driven: one CLI run per sensitivity point)
// ---------------------------------------------------------------------------
void MainWindow::startNextSensJob() {
  if (sensJobIndex_ >= sensJobQueue_.size()) return;

  const SensQueueJob& job = sensJobQueue_[sensJobIndex_];
  CrashLog::append(QString("[sens] Starting job %1/%2: problem=%3 dim=%4")
                   .arg(sensJobIndex_+1).arg(sensJobQueue_.size())
                   .arg(job.problem).arg(job.dim));

  // Show status: method, problem, current parameter(s) and point index.
  if (outputStatusLbl_) {
    const QString method = (!sensGroups_.isEmpty() && job.groupId < (int)sensGroups_.size())
                           ? sensGroups_[job.groupId].method : currentMethodShort();
    const QStringList& params = (!sensGroups_.isEmpty() && job.groupId < (int)sensGroups_.size())
                                ? sensGroups_[job.groupId].sweepParams : QStringList();
    const QString paramStr = params.isEmpty() ? QString() : params.join(", ");
    QString msg = sensIsProblemMode_
      ? QString("Problem sens. — %1 | %2").arg(job.problem).arg(paramStr)
      : QString("%1 | %2").arg(method).arg(paramStr);
    if (!paramStr.isEmpty())
      msg += QString("  [point %1/%2]").arg(job.pointIdx + 1)
                                        .arg(sensJobQueue_.size() / qMax(1, (int)sensGroups_.size()));
    outputStatusLbl_->setText(msg);
  }

  // Set problem + dim (blocked — avoid unwanted signal cascades).
  if (problemBox_) {
    QSignalBlocker b(problemBox_);
    const int idx = problemBox_->findData(job.problem);
    if (idx >= 0) problemBox_->setCurrentIndex(idx);
    else {
      const int ti = problemBox_->findText(job.problem, Qt::MatchFixedString | Qt::MatchCaseSensitive);
      if (ti >= 0) problemBox_->setCurrentIndex(ti);
    }
  }
  if (dimSpin_) { QSignalBlocker b(dimSpin_); dimSpin_->setValue(job.dim); }

  // Inject parameter values into the appropriate config section.
  // Mode 2: inject into METHOD section.  Mode 3: inject into PROBLEM section.
  // No [sensitivity] section is needed — the CLI runs as a plain multi-run experiment.
  if (cfg_ && cfg_->isLoaded()) {
    const QString methodSection = (!sensGroups_.isEmpty() && job.groupId < sensGroups_.size())
                                  ? sensGroups_[job.groupId].method : currentMethodShort();
    if (sensIsProblemMode_) {
      // Problem sensitivity: inject into the problem section (e.g. [weatherirrigation]).
      const QString problemSection = job.problem;
      for (auto it = job.injectedParams.cbegin(); it != job.injectedParams.cend(); ++it) {
        cfg_->setValue(problemSection, it.key(), it.value());
      }
    } else {
      // Method sensitivity: inject into the method section.
      for (auto it = job.injectedParams.cbegin(); it != job.injectedParams.cend(); ++it) {
        cfg_->setValue(methodSection, it.key(), it.value());
        CrashLog::append(QString("[sens] inject [%1] %2 = %3").arg(methodSection, it.key(), it.value()));
      }
    }
    // Aggressively clear the [sensitivity] section — leave only enabled=0.
    // Stale params (e.g. params=delta) would cause the CLI to attempt its own
    // sensitivity analysis and exit early without running.
    cfg_->removeKey("sensitivity", "params");
    cfg_->removeKey("sensitivity", "values.delta");
    cfg_->removeKey("sensitivity", "values.alpha");
    cfg_->removeKey("sensitivity", "values.Fhi");
    cfg_->removeKey("sensitivity", "values.Flo");
    // Remove all values.* keys from sensitivity section.
    const auto sensMap = cfg_->sectionMap("sensitivity");
    for (auto it = sensMap.cbegin(); it != sensMap.cend(); ++it) {
      const QString k = it.key().trimmed();
      if (k.startsWith("values.") || (k != "enabled" && k != "enable" && k != "mode" && k != "output"))
        cfg_->removeKey("sensitivity", k);
    }
    cfg_->setValue("sensitivity", "enabled", "0");
    cfg_->setValue("sensitivity", "enable",  "0");
    cfg_->setValue("sensitivity", "params",  "");
  }

  const QString method  = (!sensGroups_.isEmpty() && job.groupId < sensGroups_.size())
                          ? sensGroups_[job.groupId].method : currentMethodShort();
  const QString problem = job.problem;
  const int     dim     = job.dim;
  if (method.isEmpty() || problem.isEmpty()) return;

  QStringList args;
  args << method << problem;
  // For fixed-dimension problems (e.g. eld1, eld2), the CLI rejects a dim argument.
  if (fixedDimForProblem(problem) <= 0)
    args << QString::number(dim);

  // Create or reuse the output tab for this group (one tab per group).
  // In OAT mode each group has a single sweep param → one tab per param.
  int outIdx = -1;
  if (sensGroupTabIdx_.contains(job.groupId)) {
    outIdx = sensGroupTabIdx_[job.groupId];
  } else {
    outIdx = createOutputRunTab(method, problem, dim);
    if (outIdx >= 0) {
      sensGroupTabIdx_[job.groupId] = outIdx;
      // In OAT mode: suffix the tab title with the parameter name
      // so multiple params on the same problem/dim get distinct tabs.
      if (job.groupId < int(sensGroups_.size())
          && sensGroups_[job.groupId].sweepParams.size() == 1
          && outputTabs_) {
        const QString paramName = sensGroups_[job.groupId].sweepParams[0];
        const QString oldTitle  = outputTabs_->tabText(outIdx);
        outputTabs_->setTabText(outIdx, oldTitle + " [" + paramName + "]");
      }
    }
  }
  if (outIdx >= 0) {
    if (outputTabs_) outputTabs_->setCurrentIndex(outIdx);
    activeOutputRunIndex_ = outIdx;
  }

  sensJobStartTime_ = QDateTime::currentDateTimeUtc();
  currentRunNumber_ = 1;  // reset for new sensitivity point
  startCliProcess(args);
}

void MainWindow::parseSummaryCsvForSensPoint(const QString& csvPath, SensPointResult& out) {
  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;

  QStringList headers;
  QVector<QMap<QString,double>> rows;
  bool firstLine = true;
  QTextStream ts(&f);
  while (!ts.atEnd()) {
    const QString line = ts.readLine().trimmed();
    if (line.isEmpty()) continue;
    const QStringList parts = line.split(',');
    if (firstLine) {
      firstLine = false;
      for (const QString& p : parts) headers << p.trimmed().toLower();
      continue;
    }
    if (parts.size() < 2) continue;
    QMap<QString,double> row;
    for (int i = 0; i < qMin(headers.size(), parts.size()); ++i) {
      bool ok = false;
      const double v = parts[i].trimmed().toDouble(&ok);
      if (ok) row[headers[i]] = v;
    }
    if (!row.isEmpty()) rows << row;
  }
  f.close();
  if (rows.isEmpty()) return;

  // Helper: find first matching column name.
  auto findCol = [&](const QStringList& names) -> QString {
    for (const QString& n : names)
      if (headers.contains(n)) return n;
    return {};
  };

  // Try aggregate format first (one row with mean/stdev columns).
  const QString cMean    = findCol({"mean_f","mean_best_f","mean_fitness","mean"});
  const QString cStdev   = findCol({"stdev_f","stdev_best_f","sd_f","std_f","stdev","std"});
  const QString cMin     = findCol({"min_f","min_best_f","min_fitness","min"});
  const QString cMax     = findCol({"max_f","max_best_f","max_fitness","max"});
  const QString cRuns    = findCol({"runs","num_runs","n_runs","n","samples"});
  const QString cMeanEv  = findCol({"mean_evals","mean_fe","mean_evaluations"});
  const QString cStdevEv = findCol({"stdev_evals","stdev_fe","sd_evals"});
  const QString cSuccess = findCol({"success_rate","success"});

  if (!cMean.isEmpty() && rows.size() == 1) {
    // Aggregate row.
    out.meanF       = rows[0].value(cMean, 0.0);
    out.stdevF      = rows[0].value(cStdev, 0.0);
    out.minF        = rows[0].value(cMin, out.meanF);
    out.maxF        = !cMax.isEmpty() ? rows[0].value(cMax, out.meanF) : out.minF;
    out.runs        = int(rows[0].value(cRuns, double(rows.size())));
    out.meanEvals   = rows[0].value(cMeanEv, 0.0);
    out.stdevEvals  = rows[0].value(cStdevEv, 0.0);
    out.successRate = rows[0].value(cSuccess, 100.0);
    out.valid       = true;

    // If max_f was NOT in the summary CSV, try the companion convergence CSV
    // to compute per-run max(best_f) — the worst run's final result.
    if (cMax.isEmpty()) {
      const QString convPath = QString(csvPath).replace(QRegularExpression("_summary\\.csv$"), "_convergence.csv");
      QFile convFile(convPath);
      if (convFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream cts(&convFile);
        const QString cHeader = cts.readLine().trimmed();
        const QStringList cHdrs = cHeader.split(',', Qt::KeepEmptyParts);
        int cRunIdx = -1, cBestIdx = -1;
        for (int ci = 0; ci < cHdrs.size(); ++ci) {
          const QString h = cHdrs[ci].trimmed().toLower();
          if (h == "run") cRunIdx = ci;
          else if (h == "best_f" || h == "best") cBestIdx = ci;
        }
        if (cRunIdx >= 0 && cBestIdx >= 0) {
          // Collect per-run final best_f: for each run, the best_f of the
          // last row (highest iter) is the converged result for that run.
          QMap<int, double> runFinalBest;
          while (!cts.atEnd()) {
            const QString cline = cts.readLine().trimmed();
            if (cline.isEmpty()) continue;
            const QStringList cparts = cline.split(',', Qt::KeepEmptyParts);
            if (cparts.size() <= qMax(cRunIdx, cBestIdx)) continue;
            bool okR = false, okB = false;
            const int runNum = cparts[cRunIdx].trimmed().toInt(&okR);
            const double bestF = cparts[cBestIdx].trimmed().toDouble(&okB);
            if (okR && okB) {
              runFinalBest[runNum] = bestF; // last row per run wins
            }
          }
          if (!runFinalBest.isEmpty()) {
            double maxBest = -std::numeric_limits<double>::infinity();
            double minBest =  std::numeric_limits<double>::infinity();
            for (auto it = runFinalBest.cbegin(); it != runFinalBest.cend(); ++it) {
              if (it.value() > maxBest) maxBest = it.value();
              if (it.value() < minBest) minBest = it.value();
            }
            out.maxF = maxBest;
            // Also correct minF if needed (convergence CSV is more precise).
            if (minBest < out.minF) out.minF = minBest;
          }
        }
      }
    }

    return;
  }

  // Per-run format: compute aggregate from multiple rows.
  const QString fCol = !cMean.isEmpty() ? cMean
                     : findCol({"best_f","f","fitness","best_fitness","value"});
  const QString eCol = !cMeanEv.isEmpty() ? cMeanEv
                     : findCol({"evals","evaluations","fe","function_evals"});
  const QString sCol = cSuccess.isEmpty() ? findCol({"success","converged"}) : cSuccess;

  if (fCol.isEmpty()) return;

  QVector<double> fVals, eVals;
  int successCount = 0;
  for (const auto& row : rows) {
    if (row.contains(fCol)) fVals << row[fCol];
    if (!eCol.isEmpty() && row.contains(eCol)) eVals << row[eCol];
    if (!sCol.isEmpty() && row.contains(sCol))
      successCount += (row[sCol] >= 0.5) ? 1 : 0;
  }
  if (fVals.isEmpty()) return;

  const int n = fVals.size();
  double sum = 0, sum2 = 0, minVal = fVals[0], maxVal = fVals[0];
  for (double v : fVals) { sum += v; sum2 += v*v; minVal = qMin(minVal, v); maxVal = qMax(maxVal, v); }
  const double mean = sum / n;
  const double var  = n > 1 ? (sum2 - double(n)*mean*mean) / double(n-1) : 0.0;

  out.meanF       = mean;
  out.stdevF      = std::sqrt(qMax(0.0, var));
  out.minF        = minVal;
  out.maxF        = maxVal;
  out.runs        = n;
  out.successRate = sCol.isEmpty() ? 100.0 : successCount * 100.0 / n;

  if (!eVals.isEmpty()) {
    double es = 0, es2 = 0;
    for (double v : eVals) { es += v; es2 += v*v; }
    const double em = es / eVals.size();
    const double ev = eVals.size() > 1 ? (es2 - eVals.size()*em*em) / (eVals.size()-1) : 0.0;
    out.meanEvals  = em;
    out.stdevEvals = std::sqrt(qMax(0.0, ev));
  }
  out.valid = true;
}

void MainWindow::writeSensGroupCsv(const SensGroup& group, const QString& wd) {
  if (group.results.isEmpty()) return;

  // Filename: sensitivity_<param>_<problem>_D<dim>.csv (single param)
  //       or: sensitivity_<problem>_D<dim>.csv (multi-param)
  QString csvName;
  if (group.sweepParams.size() == 1)
    csvName = QString("sensitivity_%1_%2_D%3.csv")
              .arg(group.sweepParams[0], group.problem).arg(group.dim);
  else
    csvName = QString("sensitivity_%1_D%2.csv").arg(group.problem).arg(group.dim);

  const QString csvPath = QDir(wd).filePath(csvName);

  QFile f(csvPath);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);

  // Header.
  QStringList hdr;
  hdr << "method" << "problem" << "dim";
  for (const QString& p : group.sweepParams) hdr << p;
  hdr << "runs" << "mean_f" << "stdev_f" << "min_f" << "max_f" << "mean_evals" << "stdev_evals" << "success_rate";
  ts << hdr.join(",") << "\n";

  // Rows.
  for (const SensPointResult& r : group.results) {
    QStringList row;
    row << group.method << group.problem << QString::number(group.dim);
    for (const QString& p : group.sweepParams)
      row << r.paramValues.value(p, "");
    if (r.valid) {
      row << QString::number(r.runs)
          << QString::number(r.meanF,       'g', 10)
          << QString::number(r.stdevF,      'g', 10)
          << QString::number(r.minF,        'g', 10)
          << QString::number(r.maxF,        'g', 10)
          << QString::number(r.meanEvals,   'g', 10)
          << QString::number(r.stdevEvals,  'g', 10)
          << QString::number(r.successRate, 'f', 3);
    } else {
      // Write empty cells if parsing failed — at least the user sees the structure.
      row << "0" << "" << "" << "" << "" << "" << "" << "";
    }
    ts << row.join(",") << "\n";
  }
}


void MainWindow::updateSingleBatchProgress() {
  if (!outputProgressBar_ || singleRunsPerJob_ <= 0) return;

  const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;

  // currentRunNumber_ is set by parsing "[ Run X ]" from CLI output.
  // It is 1-based, so run 1 = start of first run (0% done for that run).
  // We treat run X starting as (X-1) completed runs.
  const int completedRuns = qBound(0, currentRunNumber_ - 1, singleRunsPerJob_);

  double completed = double(completedRuns);
  double total     = double(singleRunsPerJob_);

  if (mode == 1 && batchActive_ && batchTotalJobs_ > 0) {
    completed = double(batchJobIndex_) * singleRunsPerJob_ + completedRuns;
    total     = double(batchTotalJobs_) * singleRunsPerJob_;
  } else if ((mode == 2 || mode == 3) && !sensJobQueue_.isEmpty()) {
    completed = double(sensJobIndex_) * singleRunsPerJob_ + completedRuns;
    total     = double(sensJobQueue_.size()) * singleRunsPerJob_;
  }

  if (total <= 0) total = 1;
  const int pct = qBound(0, int(completed / total * 100.0), 99);

  outputProgressBar_->setRange(0, 100);
  outputProgressBar_->setValue(pct);
  outputProgressDeterminate_ = true;

  if (mode == 1 && batchActive_ && batchTotalJobs_ > 0) {
    outputProgressBar_->setFormat(QString("Job %1/%2  Run %3/%4 — %5%")
      .arg(batchJobIndex_ + 1).arg(batchTotalJobs_)
      .arg(currentRunNumber_).arg(singleRunsPerJob_).arg(pct));
  } else if ((mode == 2 || mode == 3) && !sensJobQueue_.isEmpty()) {
    outputProgressBar_->setFormat(QString("Point %1/%2  Run %3/%4 — %5%")
      .arg(sensJobIndex_ + 1).arg(sensJobQueue_.size())
      .arg(currentRunNumber_).arg(singleRunsPerJob_).arg(pct));
  } else {
    outputProgressBar_->setFormat(QString("Run %1/%2 — %3%")
      .arg(currentRunNumber_).arg(singleRunsPerJob_).arg(pct));
  }
}

void MainWindow::onProgressPollTick() {
  if (!outputProgressBar_ || singleRunsPerJob_ <= 0) return;

  // Search for a convergence CSV created after this process started.
  const QStringList searchDirs = { progressPollDir_ + "/csv", progressPollDir_ };
  QString bestPath;
  QDateTime bestTime;
  for (const QString& dir : searchDirs) {
    if (dir.isEmpty()) continue;
    QDirIterator it(dir, QStringList() << "*_convergence.csv", QDir::Files);
    while (it.hasNext()) {
      const QFileInfo fi(it.next());
      // Only consider files created/modified after this process started.
      const QDateTime fileTime = fi.birthTime().isValid() ? fi.birthTime() : fi.lastModified();
      if (fileTime < progressPollStartTime_.addSecs(-2)) continue;
      if (bestPath.isEmpty() || fi.lastModified() > bestTime) {
        bestPath = fi.absoluteFilePath();
        bestTime = fi.lastModified();
      }
    }
  }
  if (bestPath.isEmpty()) return;

  // Read only the last line efficiently — seek from end.
  QFile f(bestPath);
  if (!f.open(QIODevice::ReadOnly)) return;
  const qint64 size = f.size();
  if (size <= 0) { f.close(); return; }

  // Seek back up to 4KB from end to find the last complete line.
  const qint64 seekPos = qMax(qint64(0), size - 4096);
  f.seek(seekPos);
  const QByteArray tail = f.read(size - seekPos);
  f.close();

  // Split and take the last non-empty line.
  const QList<QByteArray> tailLines = tail.split('\n');
  QByteArray lastLine;
  for (int i = tailLines.size() - 1; i >= 0; --i) {
    const QByteArray& ln = tailLines[i].trimmed();
    if (!ln.isEmpty() && !ln.startsWith("method")) { lastLine = ln; break; }
  }
  if (lastLine.isEmpty()) return;

  const QList<QByteArray> parts = lastLine.split(',');
  if (parts.size() < 4) return;
  bool ok = false;
  const int runIdx = parts[3].trimmed().toInt(&ok);   // 'run' column, 0-based
  if (!ok || runIdx < 0) return;

  // runIdx is 0-based: run 0 = first run in progress → display as Run 1.
  // Only advance currentRunNumber_ — never go backwards.
  const int newRun = qMin(runIdx + 1, singleRunsPerJob_);
  if (newRun > currentRunNumber_) currentRunNumber_ = newRun;
  updateSingleBatchProgress();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void MainWindow::startSensCsvPoll(const QString& csvPath, int expectedPoints, int runsPerPoint) {
  stopSensCsvPoll();

  sensCsvPollPath_       = csvPath;
  sensCsvExpectedRows_   = expectedPoints;
  sensCsvLastRowCount_   = 0;
  sensRunsPerPoint_      = (runsPerPoint > 0) ? runsPerPoint : 1;
  sensCurrentRunInPoint_ = 0;
  sensRunsSeenTotal_     = 0;
  sensTotalExpectedRuns_ = expectedPoints * sensRunsPerPoint_;
  sensCompletedRuns_     = 0;
  sensCurrentEvals_      = 0;
  sensPrevEvals_         = 0;

  // Read max_evals from config for evals-based progress calculation.
  sensMaxEvalsPerRun_ = 150000; // safe default
  if (cfg_ && cfg_->isLoaded()) {
    const QStringList evKeys = {"max_evals", "max_evaluations", "eval_budget", "max_fe", "budget"};
    for (const QString& k : evKeys) {
      bool ok = false;
      const int v = cfg_->value("global", k, "-1").toInt(&ok);
      if (ok && v > 0) { sensMaxEvalsPerRun_ = v; break; }
    }
  }

  sensElapsed_.start();
  sensFirstRunMs_ = -1;

  // Switch to determinate mode immediately.
  if (outputProgressBar_) {
    outputProgressBar_->setRange(0, 100);
    outputProgressBar_->setValue(0);
    outputProgressBar_->setFormat("0%");
    outputProgressDeterminate_ = true;
  }

  if (!sensCsvPollTimer_) {
    sensCsvPollTimer_ = new QTimer(this);
    sensCsvPollTimer_->setInterval(150);
    connect(sensCsvPollTimer_, &QTimer::timeout, this, &MainWindow::onSensCsvPollTick);
  }
  sensCsvPollTimer_->start();
  // Fire one tick immediately (next event loop iteration) so fast runs don't miss the first update.
  QTimer::singleShot(0, this, &MainWindow::onSensCsvPollTick);
}

void MainWindow::stopSensCsvPoll() {
  if (sensCsvPollTimer_) sensCsvPollTimer_->stop();
  sensCsvPollPath_.clear();
  sensCsvExpectedRows_    = 0;
  sensCsvLastRowCount_    = 0;
  sensRunsPerPoint_       = 0;
  sensCurrentRunInPoint_  = 0;
  sensRunsSeenTotal_      = 0;
  sensTotalExpectedRuns_  = 0;
  sensFirstChunkLogged_   = false;
  sensPollPathLogged_     = false;
  sensMaxEvalsPerRun_     = 0;
  sensCompletedRuns_      = 0;
  sensCurrentEvals_       = 0;
  sensPrevEvals_          = 0;
  sensFirstRunMs_         = -1;
}

void MainWindow::onSensCsvPollTick() {
  if (!outputProgressBar_ || sensCsvExpectedRows_ <= 0 || sensRunsPerPoint_ <= 0) return;

  // One-time debug: confirm the CSV path we're polling.
  if (!sensPollPathLogged_) {
    sensPollPathLogged_ = true;
    appendLog(QString("[debug] polling CSV: \"%1\"  exists=%2  expectedPoints=%3  runsPerPoint=%4")
              .arg(sensCsvPollPath_)
              .arg(QFileInfo::exists(sensCsvPollPath_) ? "yes" : "NO")
              .arg(sensCsvExpectedRows_)
              .arg(sensRunsPerPoint_));
  }

  updateSingleBatchProgress();
}


void MainWindow::onRunMultiObjective() {
  if (!moProblemBox_ || !moMethodBox_) return;
  const QString problem = moProblemBox_->currentText().trimmed();
  const QString method  = moMethodBox_->currentText().trimmed();
  if (problem.isEmpty() || method.isEmpty() || !moProblemBox_->isEnabled()) {
    QMessageBox::warning(this, "Multi-objective",
      "No multi-objective-capable problem/method is available yet.\n\n"
      "Convert a problem by overriding numObjectives()/evaluateMultiCore() "
      "(see portfoliomv for an example).");
    return;
  }

  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found",
      "optimsolution executable was not found next to the GUI or in build/Debug.");
    return;
  }

  const int dim  = moDimSpin_ ? moDimSpin_->value() : 2;

  const int outIdx = createMultiObjectiveOutputTab(method, problem, dim);
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (moProc_) {
    moProc_->deleteLater();
    moProc_ = nullptr;
  }
  moProc_ = new QProcess(this);
  moProc_->setProgram(cli);

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  // Same convention as Single/Batch/Sensitivity: keep generated CSVs out of
  // the root/app folder, under optimsolution_gui_run instead.
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(runtimeWd);
  moProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;

  // Population/generations are read by the CLI itself from optimsolution.cfg
  // (the chosen method's own section, e.g. [nsga2]) -- not passed as args.
  QStringList args;
  args << "--multi" << method << problem << QString::number(dim);
  moProc_->setArguments(args);

  connect(moProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onMultiObjectiveProcessFinished);
  connect(moProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (moProc_) appendLog(QString::fromUtf8(moProc_->readAllStandardOutput()).trimmed());
  });
  connect(moProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (moProc_) appendLog(QString::fromUtf8(moProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog("Running (multi-objective): " + cli + " " + args.join(' '));

  updateRunBtn(true);
  moProc_->start();
}

void MainWindow::onMultiObjectiveProcessFinished(int exitCode, QProcess::ExitStatus status) {
  updateRunBtn(false);

  if (status != QProcess::NormalExit || exitCode != 0) {
    appendLog(QString("Multi-objective run failed (exit code %1). See log above for details.").arg(exitCode));
    return;
  }

  auto* tab = outputTab(activeOutputRunIndex_);
  if (!tab || !tab->isMultiObjective) return;

  const QString workDir = moProc_ ? moProc_->workingDirectory() : QDir::currentPath();
  QDir dir(workDir.isEmpty() ? QDir::currentPath() : workDir);
  const QStringList filters = { tab->methodShort + "_" + tab->problemShort + "_d*_multi_*_pareto.csv" };
  const QFileInfoList matches = dir.entryInfoList(filters, QDir::Files, QDir::Time);

  if (matches.isEmpty()) {
    appendLog("Multi-objective run finished, but no _pareto.csv output file was found in: " + dir.absolutePath());
    if (tab->moInfoLbl) tab->moInfoLbl->setText("Run finished, but no Pareto CSV was found.");
    return;
  }

  loadParetoCsvIntoTab(activeOutputRunIndex_, matches.first().absoluteFilePath());
  appendLog("Multi-objective run finished: " + matches.first().fileName());
}

int MainWindow::createMultiObjectiveOutputTab(const QString& method, const QString& problem, int dim) {
  if (!outputTabs_) return -1;

  const QString title = method + " + " + problem;
  auto* page = new QWidget(outputTabs_);
  auto* lay = new QVBoxLayout(page);
  lay->setContentsMargins(8, 8, 8, 8);

  auto* top = new QHBoxLayout();
  auto* info = new QLabel("No results yet. Click Run to start.", page);
  info->setWordWrap(true);
  top->addWidget(info, 1);

  auto* exportPngBtn = new QPushButton("Export PNG (300 DPI)...", page);
  exportPngBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  exportPngBtn->setIconSize(QSize(18, 18));
  connect(exportPngBtn, &QPushButton::clicked, this, &MainWindow::onExportParetoPng);
  top->addWidget(exportPngBtn);

  auto* exportXlsxBtn = new QPushButton("Export table...", page);
  exportXlsxBtn->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
  exportXlsxBtn->setIconSize(QSize(18, 18));
  connect(exportXlsxBtn, &QPushButton::clicked, this, &MainWindow::onExportParetoTable);
  top->addWidget(exportXlsxBtn);
  lay->addLayout(top);

  auto* plotRow = new QHBoxLayout();
  auto* plot = new ParetoFrontWidget(page);
  plotRow->addWidget(plot, 1);

  auto* controlsPanel = new QWidget(page);
  auto* controlsLay = new QVBoxLayout(controlsPanel);
  controlsLay->setContentsMargins(8, 0, 0, 0);
  controlsLay->setSpacing(10);
  controlsPanel->setFixedWidth(120);

  auto* themeLbl = new QLabel("Theme", controlsPanel);
  auto* themeCombo = new QComboBox(controlsPanel);
  themeCombo->addItem("Dark");
  themeCombo->addItem("Light");
  themeCombo->addItem("Transparent");
  themeCombo->setCurrentIndex(0);
  connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plot](int index) {
    if (auto* pf = dynamic_cast<ParetoFrontWidget*>(plot)) {
      ParetoFrontWidget::ThemeMode tm = ParetoFrontWidget::ThemeMode::Dark;
      if (index == 1) tm = ParetoFrontWidget::ThemeMode::Light;
      else if (index == 2) tm = ParetoFrontWidget::ThemeMode::Transparent;
      pf->setThemeMode(tm);
    }
  });

  auto* gridLbl = new QLabel("Grid", controlsPanel);
  auto* gridCombo = new QComboBox(controlsPanel);
  gridCombo->addItem("Sparse");
  gridCombo->addItem("Medium");
  gridCombo->addItem("Dense");
  gridCombo->setCurrentIndex(1);
  connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plot](int index) {
    if (auto* pf = dynamic_cast<ParetoFrontWidget*>(plot)) {
      ParetoFrontWidget::GridDensity gd = ParetoFrontWidget::GridDensity::Medium;
      if (index == 0) gd = ParetoFrontWidget::GridDensity::Sparse;
      else if (index == 2) gd = ParetoFrontWidget::GridDensity::Dense;
      pf->setGridDensity(gd);
    }
  });

  controlsLay->addWidget(themeLbl);
  controlsLay->addWidget(themeCombo);
  controlsLay->addSpacing(6);
  controlsLay->addWidget(gridLbl);
  controlsLay->addWidget(gridCombo);
  controlsLay->addStretch(1);

  plotRow->addWidget(controlsPanel, 0);
  lay->addLayout(plotRow, 2);

  auto* table = new QTableWidget(page);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->setSelectionBehavior(QAbstractItemView::SelectItems);
  table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  table->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  lay->addWidget(table, 1);

  const int tabIndex = outputTabs_->addTab(page, title);

  OutputRunTab run;
  run.title = title;
  run.methodShort = method;
  run.problemShort = problem;
  run.problemDim = dim;
  run.page = page;
  run.isMultiObjective = true;
  run.moParetoPlot = plot;
  run.moParetoTable = table;
  run.moInfoLbl = info;
  run.moExportPngBtn = exportPngBtn;
  run.moExportXlsxBtn = exportXlsxBtn;
  outputRuns_.push_back(run);

  return tabIndex;
}

// ---------------------------------------------------------------------
// Unified "Meta-dataset pipeline" mode (run mode 5).
//
// Four stages, each driven by its own dedicated button (not the shared
// Run/Stop button, since there is no single "the" action for this mode):
//   1. Performance batch + ranking  -- "optimsolution --batch ... ." then
//      automatically "optimsolution --rank batch_master.csv batch".
//   2. Landscape sampling            -- "optimsolution --features <p> <d> <n>"
//      looped once per selected (problem, dim) pair.
//   3. Extract ELA features (Python) -- py_scripts/extract_ela_features.py
//   4. Train meta-learner (Python)   -- py_scripts/train_meta_learner.py
//
// All four write into optimsolution_gui_run/ and share ONE output tab
// (created lazily, reused across stages) so the whole pipeline reads as a
// single running log rather than four disconnected tabs.
//
// Selection (methods + problems, including per-problem dimension override)
// reuses batchPanel_ exactly as Batch run does -- no separate file picker,
// no separate dropdowns. The batch list .txt the CLI reads is generated on
// the fly, same as before; it remains a CLI-only implementation detail.
// ---------------------------------------------------------------------

void MainWindow::startPipelineProgress(const QString& label) {
  if (!outputProgressBar_ || !progressWidget_) return;
  progressWidget_->setVisible(true);
  outputProgressBar_->setRange(0, 0); // indeterminate
  outputProgressBar_->setValue(0);
  outputProgressBar_->setFormat(label);

  // Start the live elapsed-time readout (see pipelineElapsedLbl_'s comment
  // in MainWindow.h). Independent of whatever per-stage QElapsedTimer the
  // caller (pipeline or NN landscape) keeps for its own final "(Xs, ...)"
  // label once the stage finishes -- this one exists purely to tick
  // visibly WHILE the stage is running, so it is started fresh here on
  // every call regardless of which of those already-existing timers is
  // also running.
  if (pipelineElapsedLbl_ && pipelineElapsedTimer_) {
    pipelineElapsedClock_.start();
    pipelineElapsedLbl_->setText("Elapsed: 0:00");
    pipelineElapsedLbl_->setVisible(true);
    pipelineElapsedTimer_->start();
  }
}

void MainWindow::finishPipelineProgress(bool success) {
  if (!outputProgressBar_ || !progressWidget_) return;
  outputProgressBar_->setRange(0, 100);
  outputProgressBar_->setValue(success ? 100 : 0);
  outputProgressBar_->setFormat(success ? "Finished" : "Failed");
  if (pipelineElapsedTimer_) pipelineElapsedTimer_->stop();
  if (pipelineElapsedLbl_) pipelineElapsedLbl_->setVisible(false);
  QTimer::singleShot(1500, this, [this]() {
    if (progressWidget_) progressWidget_->setVisible(false);
  });
}

// Ticks once a second while pipelineElapsedTimer_ is running (started/
// stopped only from startPipelineProgress()/finishPipelineProgress() above).
void MainWindow::updatePipelineElapsedLabel() {
  if (!pipelineElapsedLbl_) return;
  const qint64 ms = pipelineElapsedClock_.isValid() ? pipelineElapsedClock_.elapsed() : 0;
  const qint64 totalSec = ms / 1000;
  const qint64 h = totalSec / 3600;
  const qint64 m = (totalSec % 3600) / 60;
  const qint64 s = totalSec % 60;
  const QString text = (h > 0)
      ? QString("Elapsed: %1:%2:%3").arg(h).arg(m, 2, 10, QChar('0')).arg(s, 2, 10, QChar('0'))
      : QString("Elapsed: %1:%2").arg(m).arg(s, 2, 10, QChar('0'));
  pipelineElapsedLbl_->setText(text);
}

bool MainWindow::exportPipelinePlotPng(QWidget* widget, const QString& outPath, bool isTransparent) const {
  if (!widget) return false;
  const int targetDpi = 300, baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);
  const QSize sz = widget->size();
  if (sz.width() <= 0 || sz.height() <= 0) return false;
  const QSize outSz(int(std::ceil(sz.width() * scale)), int(std::ceil(sz.height() * scale)));
  // Format_ARGB32_Premultiplied does NOT survive QWidget::render() with a
  // truly transparent background -- Qt's default DrawWindowBackground
  // render flag ends up compositing an opaque backdrop onto a premultiplied
  // target regardless of what the widget's own paintEvent draws. Batch
  // run's own PNG export (MainWindow_Plots.cpp) already discovered this and
  // switches to plain (non-premultiplied) Format_ARGB32 specifically for
  // the transparent case; mirrored here for the same reason.
  QImage img(outSz, isTransparent ? QImage::Format_ARGB32 : QImage::Format_ARGB32_Premultiplied);
  img.fill(isTransparent ? Qt::transparent : Qt::white);
  const int dpm = int(std::llround(double(targetDpi) / 25.4 * 1000.0));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);
  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.scale(scale, scale);
  widget->render(&painter);
  painter.end();
  return img.save(outPath, "PNG");
}

// Minimal CSV reader matching the plain, unquoted format OptimSolution's own
// --rank CLI mode produces (see loadSimpleCsv() in main.cpp) -- header as
// row 0, one QStringList per row thereafter.
static QVector<QStringList> readPipelineCsv(const QString& path) {
  QVector<QStringList> rows;
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return rows;
  QTextStream ts(&f);
  while (!ts.atEnd()) {
    QString line = ts.readLine();
    if (line.endsWith('\r')) line.chop(1);
    if (line.isEmpty()) continue;
    rows.push_back(line.split(','));
  }
  return rows;
}
static int pipelineCsvCol(const QStringList& header, const QString& name) {
  return (int)header.indexOf(name);
}
// Generic: fill any QTableWidget from CSV rows (row 0 = header). Used for
// every raw-data table tab (Batch results, Landscape files, ELA features,
// LOPO predictions) so a researcher can always see the actual numbers a
// stage produced, not just a derived chart or summary figure.
static void populateTableFromCsvRows(QTableWidget* table, const QVector<QStringList>& rows) {
  if (!table) return;
  if (rows.isEmpty()) { table->setRowCount(0); table->setColumnCount(0); return; }
  const QStringList& header = rows[0];
  table->setColumnCount(header.size());
  table->setHorizontalHeaderLabels(header);
  table->setRowCount(rows.size() - 1);
  for (int r = 1; r < rows.size(); ++r) {
    const QStringList& row = rows[r];
    for (int c = 0; c < row.size() && c < header.size(); ++c)
      table->setItem(r - 1, c, new QTableWidgetItem(row[c]));
  }
}

void MainWindow::populatePipelineStatsTabs() {
  auto* tab = outputTab(activeOutputRunIndex_);
  if (!tab || !tab->page) return;
  QWidget* page = tab->page;

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  // ---- Rankings tab: Friedman summary text + avg_rank bars + full method
  // summary table (enriched with the rank_sum column that used to live in
  // the separate "Friedman" tab's own table, so that data is still
  // exportable via this tab's "Export XLSX..." button). ----
  {
    // Parse batch_friedman.txt once: the omnibus summary text and the
    // per-method rank sums (Rj) both come from this file.
    QString friedmanSummaryText = "Friedman test: N/A (not enough complete problem/dim groups -- see batch_friedman.txt for details).";
    QMap<QString, double> rankSumByMethod;
    {
      QFile f(QDir(runtimeWd).filePath("batch_friedman.txt"));
      if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream ts(&f);
        QString chi2, df, p, conclusion;
        bool inRankSums = false;
        static const QRegularExpression reChi2(R"(chi\^2\s*=\s*([-\d.eE+]+))");
        static const QRegularExpression reDf(R"(df\s*=\s*(\d+))");
        static const QRegularExpression reP(R"(^p\s*=\s*([-\d.eE+]+))");
        static const QRegularExpression reRankLine(R"(^\s*(\S+):\s*([-\d.eE+]+)\s*$)");
        while (!ts.atEnd()) {
          const QString line = ts.readLine();
          if (line.contains("Rank sums")) { inRankSums = true; continue; }
          if (inRankSums) {
            auto m = reRankLine.match(line);
            if (m.hasMatch()) { rankSumByMethod[m.captured(1)] = m.captured(2).toDouble(); continue; }
          }
          auto mChi2 = reChi2.match(line); if (mChi2.hasMatch()) chi2 = mChi2.captured(1);
          auto mDf = reDf.match(line); if (mDf.hasMatch()) df = mDf.captured(1);
          auto mP = reP.match(line); if (mP.hasMatch()) p = mP.captured(1);
          if (line.startsWith("->")) conclusion = line.trimmed();
        }
        if (!chi2.isEmpty()) {
          friedmanSummaryText = QString("Friedman omnibus: chi^2=%1 (df=%2), p=%3. %4").arg(chi2, df, p, conclusion);
        }
      }
    }
    if (auto* summaryLbl = page->findChild<QLabel*>("pipelineFriedmanSummaryLbl")) {
      summaryLbl->setText(friedmanSummaryText);
    }

    const auto rows = readPipelineCsv(QDir(runtimeWd).filePath("batch_method_summary.csv"));
    auto* plot = page->findChild<RankPlotWidget*>("pipelineRankingsPlot");
    auto* table = page->findChild<QTableWidget*>("pipelineRankingsTable");
    if (!rows.isEmpty() && plot && table) {
      const QStringList& header = rows[0];
      const int cMethod = pipelineCsvCol(header, "method");
      const int cAvgRank = pipelineCsvCol(header, "avg_rank");
      QStringList methods;
      QVector<double> avgRanks;
      // Append a "rank_sum" column (Rj, from batch_friedman.txt) after the
      // csv's own columns, if we managed to parse it -- this is the one
      // piece of data the old Friedman tab's table had that this one didn't.
      const bool haveRankSums = !rankSumByMethod.isEmpty();
      table->setColumnCount(header.size() + (haveRankSums ? 1 : 0));
      QStringList headerWithExtra = header;
      if (haveRankSums) headerWithExtra << "rank_sum";
      table->setHorizontalHeaderLabels(headerWithExtra);
      table->setRowCount(rows.size() - 1);
      for (int r = 1; r < rows.size(); ++r) {
        const QStringList& row = rows[r];
        for (int c = 0; c < row.size() && c < header.size(); ++c)
          table->setItem(r - 1, c, new QTableWidgetItem(row[c]));
        if (cMethod >= 0 && cAvgRank >= 0 && cMethod < row.size() && cAvgRank < row.size()) {
          methods << row[cMethod];
          avgRanks << row[cAvgRank].toDouble();
        }
        if (haveRankSums && cMethod >= 0 && cMethod < row.size() && rankSumByMethod.contains(row[cMethod])) {
          table->setItem(r - 1, header.size(),
                          new QTableWidgetItem(QString::number(rankSumByMethod[row[cMethod]])));
        }
      }
      // Reviewer 2, comment 3: the "(lower = better)" note is now drawn
      // by RankPlotWidget itself (see MainWindowShared.h), so it no
      // longer needs to be passed in here as caller-supplied subtitle text.
      plot->setData(methods, avgRanks, "Average rank per method", "");
    }
  }


  // ---- Wilcoxon tab: per-method rank-distribution box plots + pairwise table ----
  // (also populates the "Batch results" raw-data tab below, from the same
  // batch_ranked.csv read -- no need to read the file twice.)
  {
    const auto rankedRows = readPipelineCsv(QDir(runtimeWd).filePath("batch_ranked.csv"));
    populateTableFromCsvRows(page->findChild<QTableWidget*>("pipelineBatchResultsTable"), rankedRows);
    if (auto* hint = page->findChild<QLabel*>("pipelineBatchResultsTableHint")) {
      hint->setText(rankedRows.isEmpty()
        ? "No results yet -- run stage 1 (Batch + ranking) first."
        : QString("%1 row(s) -- full per-(method, problem, dim) results with rank/is_best (batch_ranked.csv).")
            .arg(rankedRows.size() - 1));
    }
    auto* plot = page->findChild<WilcoxonBoxPlotWidget*>("pipelineWilcoxonPlot");
    QStringList methodOrder;
    QMap<QString, QVector<double>> ranksByMethod;
    if (!rankedRows.isEmpty()) {
      const QStringList& header = rankedRows[0];
      const int cMethod = pipelineCsvCol(header, "method");
      const int cRank = pipelineCsvCol(header, "rank");
      if (cMethod >= 0 && cRank >= 0) {
        for (int r = 1; r < rankedRows.size(); ++r) {
          const QStringList& row = rankedRows[r];
          if (cMethod >= row.size() || cRank >= row.size()) continue;
          const QString m = row[cMethod];
          if (!ranksByMethod.contains(m)) methodOrder << m;
          ranksByMethod[m].push_back(row[cRank].toDouble());
        }
      }
    }

    const auto wilcoxonRows = readPipelineCsv(QDir(runtimeWd).filePath("batch_wilcoxon.csv"));
    auto* table = page->findChild<QTableWidget*>("pipelineWilcoxonTable");
    QVector<WilcoxonBoxPlotWidget::Annotation> annotations;
    if (!wilcoxonRows.isEmpty() && table) {
      const QStringList& header = wilcoxonRows[0];
      const int cA = pipelineCsvCol(header, "method_a");
      const int cB = pipelineCsvCol(header, "method_b");
      const int cSig = pipelineCsvCol(header, "significant_at_0.05");
      const int cPHolm = pipelineCsvCol(header, "p_holm");
      table->setColumnCount(header.size());
      table->setHorizontalHeaderLabels(header);
      table->setRowCount(wilcoxonRows.size() - 1);
      for (int r = 1; r < wilcoxonRows.size(); ++r) {
        const QStringList& row = wilcoxonRows[r];
        for (int c = 0; c < row.size() && c < header.size(); ++c)
          table->setItem(r - 1, c, new QTableWidgetItem(row[c]));
        if (cA >= 0 && cB >= 0 && cA < row.size() && cB < row.size()) {
          const int i = (int)methodOrder.indexOf(row[cA]);
          const int j = (int)methodOrder.indexOf(row[cB]);
          if (i >= 0 && j >= 0) {
            const bool sig = (cSig >= 0 && cSig < row.size() && row[cSig].trimmed() == "1");
            WilcoxonBoxPlotWidget::Annotation ann;
            ann.i = i; ann.j = j; ann.significant = sig;
            // "p_adj=" is the Holm-corrected-p-value convention the widget
            // already renders specially (italic "p", italic subscript
            // "adj", and -- for scientific notation -- a real raised
            // superscript exponent instead of raw "e-06" text). Falls back
            // to the previous "*"/"ns" only if this file doesn't have a
            // p_holm column (older pipeline runs, before it was added).
            if (cPHolm >= 0 && cPHolm < row.size()) {
              ann.text = "p_adj=" + row[cPHolm].trimmed() + (sig ? " *" : "");
            } else {
              ann.text = sig ? "*" : "ns";
            }
            annotations.push_back(ann);
          }
        }
      }
    }

    if (plot && !methodOrder.isEmpty()) {
      QVector<WilcoxonBoxPlotWidget::Group> groups;
      for (const auto& m : methodOrder) {
        WilcoxonBoxPlotWidget::Group g;
        g.label = m;
        g.values = ranksByMethod.value(m);
        groups.push_back(g);
      }
      plot->setData(std::move(groups), annotations, "Per-method rank distribution",
                     "Across all problem/dim instances (from batch_ranked.csv)", "Rank");
    }
  }
}

void MainWindow::populatePipelineLandscapeFilesTable() {
  auto* tab = outputTab(activeOutputRunIndex_);
  if (!tab || !tab->page) return;
  QWidget* page = tab->page;
  auto* table = page->findChild<QTableWidget*>("pipelineLandscapeFilesTable");
  auto* hint = page->findChild<QLabel*>("pipelineLandscapeFilesTableHint");
  if (!table) return;

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  const QStringList headerOut = {"file", "problem", "dim", "n_samples", "modality", "separability"};
  table->setColumnCount(headerOut.size());
  table->setHorizontalHeaderLabels(headerOut);

  QDir dir(runtimeWd);
  const QStringList files = dir.entryList(QStringList() << "*_features.csv", QDir::Files, QDir::Name);
  table->setRowCount(files.size());
  for (int i = 0; i < files.size(); ++i) {
    const auto rows = readPipelineCsv(dir.filePath(files[i]));
    QString problem, dim, nSamples, modality, separability;
    if (rows.size() >= 2) {
      const QStringList& h = rows[0];
      const QStringList& d = rows[1];
      auto get = [&](const QString& name) -> QString {
        const int c = pipelineCsvCol(h, name);
        return (c >= 0 && c < d.size()) ? d[c] : QString();
      };
      problem = get("problem");
      dim = get("dim");
      nSamples = get("n_samples");
      modality = get("modality");
      separability = get("separability");
    }
    table->setItem(i, 0, new QTableWidgetItem(files[i]));
    table->setItem(i, 1, new QTableWidgetItem(problem));
    table->setItem(i, 2, new QTableWidgetItem(dim));
    table->setItem(i, 3, new QTableWidgetItem(nSamples));
    table->setItem(i, 4, new QTableWidgetItem(modality));
    table->setItem(i, 5, new QTableWidgetItem(separability));
  }
  if (hint) {
    hint->setText(files.isEmpty()
      ? "No files yet -- run stage 2 (Landscape sampling) first."
      : QString("%1 file(s) found in optimsolution_gui_run/.").arg(files.size()));
  }
}

void MainWindow::populatePipelineElaFeaturesTable() {
  auto* tab = outputTab(activeOutputRunIndex_);
  if (!tab || !tab->page) return;
  QWidget* page = tab->page;

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  const auto rows = readPipelineCsv(QDir(runtimeWd).filePath("ela_features.csv"));
  populateTableFromCsvRows(page->findChild<QTableWidget*>("pipelineElaFeaturesTable"), rows);
  if (auto* hint = page->findChild<QLabel*>("pipelineElaFeaturesTableHint")) {
    hint->setText(rows.isEmpty()
      ? "No features yet -- run stage 3 (Extract ELA features) first."
      : QString("%1 row(s), %2 column(s) -- full ela_features.csv.").arg(rows.size() - 1).arg(rows[0].size()));
  }
}

void MainWindow::refreshPipelineButtonStates() {
  // Stage 3 needs stage 2's output (this session); stage 4 needs both
  // stage 1's and stage 3's output (this session). Stages 1 and 2 have no
  // prerequisites of their own. The optional ablation stage (Reviewer 2,
  // comment 4) shares stage 4's own prerequisites, not stage 4's
  // completion -- it re-derives everything stage 4 does internally.
  if (pipelineExtractElaBtn_) pipelineExtractElaBtn_->setEnabled(pipelineStage2Done_);
  if (pipelineTrainModelBtn_) pipelineTrainModelBtn_->setEnabled(pipelineStage1Done_ && pipelineStage3Done_);
  if (pipelineAblationBtn_) pipelineAblationBtn_->setEnabled(pipelineStage1Done_ && pipelineStage3Done_);
  if (pipelineCompareBtn_) pipelineCompareBtn_->setEnabled(pipelineStage1Done_ && pipelineStage3Done_);

  if (pipelinePanel_) {
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineBatchCheckmark")) cm->setVisible(pipelineStage1Done_);
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineFeaturesCheckmark")) cm->setVisible(pipelineStage2Done_);
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineExtractCheckmark")) cm->setVisible(pipelineStage3Done_);
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineTrainCheckmark")) cm->setVisible(pipelineStage4Done_);
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineAblationCheckmark")) cm->setVisible(pipelineAblationDone_);
    if (auto* cm = pipelinePanel_->findChild<QLabel*>("pipelineCompareCheckmark")) cm->setVisible(pipelineCompareDone_);
  }
}

// Narrower reset: only the 4 stage-completion flags/checkmarks, WITHOUT
// touching pipelineExperimentLoaded_/pipelineLoadedTripleKeys_. Used by
// onLoadPipelineExperiment() itself -- loading an experiment for add/remove
// must clear the checkmarks (they reflect a DIFFERENT, possibly stale
// session's completions, not necessarily what's true for the combination
// the user is about to run), but must NOT also wipe out the very
// loaded-experiment context it just set up.
// Formats a duration in seconds as "Xs", "Xm Ys", or "Xh Ym Zs" -- whichever
// units are actually needed, so short stages don't show "0h 0m 12s".
static QString formatPipelineDuration(double seconds) {
  if (seconds < 0) return QString();
  const qint64 totalSec = qRound64(seconds);
  const qint64 h = totalSec / 3600;
  const qint64 m = (totalSec % 3600) / 60;
  const qint64 s = totalSec % 60;
  if (h > 0) return QString("%1h %2m %3s").arg(h).arg(m).arg(s);
  if (m > 0) return QString("%1m %2s").arg(m).arg(s);
  return QString("%1s").arg(s);
}

// The 4 pipeline stage keys, in a fixed order -- used both for the on-disk
// CSV's row order and for looking up each stage's UI label/checkmark.
static const QStringList kPipelineStageKeys = {
  "batch_ranking", "landscape_sampling", "extract_ela", "train_model"
};
// Reviewer 2, comment 4: the optional ablation stage's own timing is
// persisted too, but its key is deliberately kept OUT of
// kPipelineStageKeys itself, so "Total pipeline time (all 4 stages)"
// below keeps its existing meaning (and keeps showing up) for the many
// users who never run the optional 5th stage.
static const QStringList kPipelineWritableStageKeys = kPipelineStageKeys + QStringList{"ablation", "compare_classifiers"};

void MainWindow::savePipelineStageTiming(const QString& stageKey, double seconds) {
  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(runtimeWd);
  const QString path = QDir(runtimeWd).filePath("pipeline_timings.csv");

  // Read-modify-write: load whatever is already on disk (other stages'
  // timings from this run or an earlier one), overwrite just this stage's
  // row, and write everything back -- so saving stage 3's time never loses
  // stage 1/2's already-recorded times.
  QMap<QString, QPair<double, QString>> rows; // stageKey -> (seconds, ISO timestamp)
  {
    QFile inFile(path);
    if (inFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream ts(&inFile);
      if (!ts.atEnd()) ts.readLine(); // header
      while (!ts.atEnd()) {
        QString line = ts.readLine();
        if (line.endsWith('\r')) line.chop(1);
        if (line.isEmpty()) continue;
        const QStringList parts = line.split(',');
        if (parts.size() >= 3) rows[parts[0]] = qMakePair(parts[1].toDouble(), parts[2]);
      }
    }
  }
  rows[stageKey] = qMakePair(seconds, QDateTime::currentDateTime().toString(Qt::ISODate));

  QFile outFile(path);
  if (!outFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
    appendLog("[warn] Could not write pipeline_timings.csv -- stage timing for this run won't persist across a reload.");
    return;
  }
  QTextStream ts(&outFile);
  ts << "stage,duration_seconds,completed_at\n";
  for (const QString& key : kPipelineWritableStageKeys) {
    if (rows.contains(key)) {
      ts << key << "," << QString::number(rows[key].first, 'f', 3) << "," << rows[key].second << "\n";
    }
  }
}

QMap<QString, double> MainWindow::loadPipelineStageTimings() {
  QMap<QString, double> result;
  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QFile inFile(QDir(runtimeWd).filePath("pipeline_timings.csv"));
  if (!inFile.open(QIODevice::ReadOnly | QIODevice::Text)) return result;
  QTextStream ts(&inFile);
  if (!ts.atEnd()) ts.readLine(); // header
  while (!ts.atEnd()) {
    QString line = ts.readLine();
    if (line.endsWith('\r')) line.chop(1);
    if (line.isEmpty()) continue;
    const QStringList parts = line.split(',');
    if (parts.size() >= 2) result[parts[0]] = parts[1].toDouble();
  }
  return result;
}

void MainWindow::refreshPipelineTimingLabels() {
  if (!pipelinePanel_) return;
  const QMap<QString, double> timings = loadPipelineStageTimings();

  static const QMap<QString, QString> stageToLabelObjName = {
    {"batch_ranking",      "pipelineBatchTimeLbl"},
    {"landscape_sampling", "pipelineFeaturesTimeLbl"},
    {"extract_ela",        "pipelineExtractTimeLbl"},
    {"train_model",        "pipelineTrainTimeLbl"},
    {"ablation",           "pipelineAblationTimeLbl"},
    {"compare_classifiers","pipelineCompareTimeLbl"},
  };
  for (auto it = stageToLabelObjName.constBegin(); it != stageToLabelObjName.constEnd(); ++it) {
    auto* lbl = pipelinePanel_->findChild<QLabel*>(it.value());
    if (!lbl) continue;
    if (timings.contains(it.key())) {
      lbl->setText("(" + formatPipelineDuration(timings.value(it.key())) + ")");
    } else {
      lbl->setText(QString());
    }
  }

  auto* totalLbl = pipelinePanel_->findChild<QLabel*>("pipelineTotalTimeLbl");
  if (!totalLbl) return;
  bool allPresent = true;
  double total = 0.0;
  for (const QString& key : kPipelineStageKeys) {
    if (!timings.contains(key)) { allPresent = false; break; }
    total += timings.value(key);
  }
  totalLbl->setText(allPresent
    ? QString("Total pipeline time (all 4 stages): %1").arg(formatPipelineDuration(total))
    : QString());
}

void MainWindow::resetPipelineStageFlags() {
  pipelineStage1Done_ = false;
  pipelineStage2Done_ = false;
  pipelineStage3Done_ = false;
  pipelineStage4Done_ = false;
  pipelineAblationDone_ = false;
  pipelineCompareDone_ = false;
  refreshPipelineButtonStates();
  refreshPipelineTimingLabels();
}

void MainWindow::resetPipelineButtonStates() {
  resetPipelineStageFlags();
  setPipelineUiEnabled(true);
  // A fresh session (re-entering the mode, or after "Delete output files")
  // must not silently keep treating an old "loaded experiment" baseline as
  // still valid -- the add/remove diffing only applies once the user
  // explicitly loads an experiment again.
  pipelineExperimentLoaded_ = false;
  pipelineLoadedTripleKeys_.clear();
}

// "Load previous experiment": reads an existing batch_master.csv, restores
// the Selection area's method/problem/dimension checkboxes to match exactly
// what it contains, and remembers every (method,problem,dim) triple it
// already has. From this point on, clicking "1. Batch + ranking" will only
// actually run combinations that are newly checked (added) and will delete
// the summary CSV of any combination that got unchecked (removed) --
// Batch run (mode 1)'s own CSV loader has no equivalent "remove" half.
// "Metric" dropdown (Rankings/Wilcoxon): re-runs --rank against
// the ALREADY-COMPUTED batch_master.csv using a different ranking column,
// then refreshes both tabs. Does not touch stage 1's actual
// optimization runs -- this is purely a re-analysis of existing results.
void MainWindow::onPipelineMetricChanged(const QString& column) {
  if (column.isEmpty() || column == pipelineMetricColumn_) return;
  pipelineMetricColumn_ = column;

  if (pipelineMetricRerankProc_ && pipelineMetricRerankProc_->state() != QProcess::NotRunning) {
    appendLog("A metric re-rank is already running; the new choice will apply on the next run.");
    return;
  }
  if ((pipelineBatchProc_ && pipelineBatchProc_->state() != QProcess::NotRunning) ||
      (pipelineRankProc_ && pipelineRankProc_->state() != QProcess::NotRunning)) {
    appendLog("Stage 1 is currently running; the new metric will be used for that run.");
    return;
  }
  if (!pipelineStage1Done_) {
    appendLog(QString("Metric set to \"%1\" -- will be used once stage 1 (Batch + ranking) runs.").arg(column));
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  const QString masterCsv = QDir(runtimeWd).filePath("batch_master.csv");
  if (!QFileInfo::exists(masterCsv)) {
    appendLog(QString("Metric set to \"%1\", but batch_master.csv is missing -- run stage 1 first.").arg(column));
    return;
  }

  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found",
      "optimsolution executable was not found next to the GUI or in build/Debug.");
    return;
  }

  if (pipelineMetricRerankProc_) { pipelineMetricRerankProc_->deleteLater(); pipelineMetricRerankProc_ = nullptr; }
  pipelineMetricRerankProc_ = new QProcess(this);
  pipelineMetricRerankProc_->setProgram(cli);
  pipelineMetricRerankProc_->setWorkingDirectory(runtimeWd);
  pipelineMetricRerankProc_->setArguments({"--rank", "batch_master.csv", "batch", column});
  connect(pipelineMetricRerankProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelineMetricRerankProc_) appendLog(QString::fromUtf8(pipelineMetricRerankProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelineMetricRerankProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelineMetricRerankProc_) appendLog(QString::fromUtf8(pipelineMetricRerankProc_->readAllStandardError()).trimmed());
  });
  connect(pipelineMetricRerankProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, [this, column](int exitCode, QProcess::ExitStatus status) {
    if (status == QProcess::NormalExit && exitCode == 0) {
      appendLog(QString("Re-ranked by \"%1\".").arg(column));
      populatePipelineStatsTabs();
    } else {
      QMessageBox::warning(this, "Re-rank failed",
        QString("Re-ranking by \"%1\" failed (exit code %2). See the log for details.").arg(column).arg(exitCode));
    }
  });
  appendLog(QString("Re-ranking by \"%1\"...").arg(column));
  pipelineMetricRerankProc_->start();
}

// See declaration in MainWindow.h. openOutputFolderBtn_ and the "Maximize"
// buttons are intentionally never touched here.
void MainWindow::setPipelineUiEnabled(bool enabled) {
  if (selectionBox_) selectionBox_->setEnabled(enabled);
  if (settingsBox_) settingsBox_->setEnabled(enabled);
  if (clearCsvBtn_) clearCsvBtn_->setEnabled(enabled);
  if (pipelineLoadExperimentBtn_) pipelineLoadExperimentBtn_->setEnabled(enabled);
}

void MainWindow::onLoadPipelineExperiment() {
  if (!batchMethodsList_ || !batchProblemsList_) return;

  // Prefer the canonical runtime folder (same one every pipeline stage
  // writes into) so the dialog opens exactly where batch_master.csv
  // actually lives, even on a fresh session where lastRuntimeWorkingDir_
  // is still empty. Falling back to lastRuntimeWorkingDir_/projectRoot_
  // was opening the dialog in the source tree instead, far from any real
  // output -- easy to mistake for "loading doesn't work".
  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  QString startDir = QDir(baseWd).filePath("optimsolution_gui_run");
  if (!QDir(startDir).exists()) {
    startDir = !lastRuntimeWorkingDir_.trimmed().isEmpty()
      ? lastRuntimeWorkingDir_
      : (!projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath());
  }

  const QString csvPath = QFileDialog::getOpenFileName(
    this, "Load previous experiment (batch_master.csv)", QDir(startDir).filePath("batch_master.csv"),
    "batch_master.csv (batch_master.csv);;CSV files (*.csv);;All files (*)"
  );
  if (csvPath.isEmpty()) return;

  // Visible feedback that something is actually happening -- reading and
  // parsing a large batch_master.csv (hundreds/thousands of rows) plus
  // restoring the selection lists is not instantaneous, and a log line
  // alone is easy to miss if the Log tab isn't in view. In practice the
  // whole load finishes in well under a second even for large files, so
  // the spinner alone would just flash a single, imperceptible frame --
  // update() only *schedules* a repaint, and nothing yields back to the
  // event loop again until the (synchronous) work below is done. Pumping
  // the event loop for a small guaranteed minimum keeps the 33ms-interval
  // spinner animation actually visible, regardless of how fast the real
  // work turns out to be.
  QElapsedTimer spinnerTimer;
  spinnerTimer.start();
  if (pipelineLoadExperimentSpinner_) pipelineLoadExperimentSpinner_->start();
  if (pipelineLoadExperimentBtn_) pipelineLoadExperimentBtn_->setEnabled(false);
  QApplication::processEvents();

  auto stopSpinner = [this, &spinnerTimer]() {
    constexpr qint64 kMinVisibleMs = 450;
    while (spinnerTimer.elapsed() < kMinVisibleMs) {
      QApplication::processEvents(QEventLoop::AllEvents, 20);
    }
    if (pipelineLoadExperimentSpinner_) pipelineLoadExperimentSpinner_->stop();
    if (pipelineLoadExperimentBtn_) pipelineLoadExperimentBtn_->setEnabled(true);
  };

  const auto rows = readPipelineCsv(csvPath);
  if (rows.isEmpty()) {
    stopSpinner();
    QMessageBox::warning(this, "Load previous experiment", "The selected file is empty or could not be read.");
    return;
  }
  const QStringList& header = rows[0];
  const int cMethod = pipelineCsvCol(header, "method");
  const int cProblem = pipelineCsvCol(header, "problem");
  const int cDim = pipelineCsvCol(header, "dim");
  if (cMethod < 0 || cProblem < 0 || cDim < 0) {
    stopSpinner();
    QMessageBox::warning(this, "Load previous experiment",
      "This file doesn't look like a batch_master.csv (missing method/problem/dim columns).");
    return;
  }

  QStringList methodsWanted;
  QStringList problemsWanted;
  QMap<QString, QSet<int>> dimsPerProblem;
  QSet<QString> tripleKeys;
  for (int r = 1; r < rows.size(); ++r) {
    const QStringList& row = rows[r];
    if (cMethod >= row.size() || cProblem >= row.size() || cDim >= row.size()) continue;
    const QString method = row[cMethod].trimmed();
    const QString problem = row[cProblem].trimmed();
    const int dim = row[cDim].trimmed().toInt();
    if (method.isEmpty() || problem.isEmpty()) continue;
    if (!methodsWanted.contains(method)) methodsWanted << method;
    if (!problemsWanted.contains(problem)) problemsWanted << problem;
    dimsPerProblem[problem].insert(dim);
    tripleKeys.insert(method + "|" + problem + "|" + QString::number(dim));
  }
  if (methodsWanted.isEmpty() || problemsWanted.isEmpty()) {
    stopSpinner();
    QMessageBox::warning(this, "Load previous experiment", "No usable (method, problem, dim) rows were found in this file.");
    return;
  }

  populateBatchLists();

  QSet<QString> wantedMethods(methodsWanted.begin(), methodsWanted.end());
  for (int i = 0; i < batchMethodsList_->count(); ++i) {
    if (auto* it = batchMethodsList_->item(i)) {
      QString key = it->data(Qt::UserRole).toString().trimmed();
      if (key.contains(" - ")) key = key.section(" - ", 0, 0).trimmed();
      if (key.isEmpty()) key = it->text().section(" - ", 0, 0).trimmed();
      it->setSelected(wantedMethods.contains(key));
    }
  }

  QSet<QString> wantedProblems;
  for (const QString& p : problemsWanted) wantedProblems.insert(batchBaseProblemShort(p));
  for (int i = 0; i < batchProblemsList_->count(); ++i) {
    if (auto* it = batchProblemsList_->item(i)) {
      QString key = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
      if (key.isEmpty()) key = batchBaseProblemShort(it->text().trimmed());
      it->setSelected(wantedProblems.contains(key));
    }
  }

  batchProblemDimOverride_.clear();
  for (auto it = dimsPerProblem.begin(); it != dimsPerProblem.end(); ++it) {
    if (fixedDimForProblem(it.key()) > 0) continue; // fixed-dim problems don't use the override table
    QStringList dims;
    for (int d : it.value()) dims << QString::number(d);
    std::sort(dims.begin(), dims.end(), [](const QString& a, const QString& b){ return a.toInt() < b.toInt(); });
    if (!dims.isEmpty()) batchProblemDimOverride_.insert(it.key(), dims);
  }
  syncBatchProblemDimsTable();
  refreshBatchSelectionView();

  pipelineExperimentLoaded_ = true;
  pipelineLoadedTripleKeys_ = tripleKeys;
  // Start from a clean slate, then re-derive each stage's flag from
  // whatever output genuinely exists on disk for this (method, problem,
  // dim) selection -- an honest checkmark earned by real output, not a
  // stale one carried over from before this load, but also not a blank
  // one the user has to manually re-earn by re-running work that is
  // already sitting in optimsolution_gui_run/.
  resetPipelineStageFlags();
  detectPipelineExistingOutputs();
  stopSpinner();

  const QString msg = QString(
    "Loaded previous experiment: %1 method(s), %2 (problem,dim) instance(s) from %3.\n"
    "Adjust the checkboxes above (check to add, uncheck to remove), then run "
    "\"1. Batch + ranking\" -- only the new combinations will actually run, and "
    "unchecked ones will be dropped from batch_master.csv."
  ).arg(methodsWanted.size()).arg(tripleKeys.size()).arg(QFileInfo(csvPath).fileName());
  appendLog(msg);
  statusBar()->showMessage(QString("Loaded %1 previous instance(s).").arg(tripleKeys.size()), 6000);
  // A status-bar/log message alone is easy to miss -- confirm with a
  // popup too, so it's unambiguous the load actually happened (and what
  // it found), same reasoning as the pipeline stages' own failure popups.
  QMessageBox::information(this, "Previous experiment loaded", msg);
}

int MainWindow::createPipelineOutputTab() {
  if (pipelineOutputTabIndex_ >= 0 && outputTab(pipelineOutputTabIndex_)) {
    return pipelineOutputTabIndex_; // reuse the existing tab across stages
  }
  if (!outputTabs_) return -1;

  const QString title = "Meta-dataset pipeline";
  auto* page = new QWidget(outputTabs_);
  auto* outerLay = new QVBoxLayout(page);
  outerLay->setContentsMargins(8, 8, 8, 8);

  auto* info = new QLabel("Idle. Use the buttons in the pipeline panel to run a stage.", page);
  info->setObjectName("pipelineInfoLbl");
  info->setWordWrap(true);
  outerLay->addWidget(info);

  // Inner tabs: a live running "Log" (used by all 4 stages), and a separate
  // "Meta-learning results" tab that stays empty until stage 4 succeeds,
  // then shows the actual summary/feature-importance content instead of
  // making the user go dig through optimsolution_gui_run/ by hand.
  auto* innerTabs = new QTabWidget(page);
  innerTabs->setObjectName("pipelineInnerTabs");
  outerLay->addWidget(innerTabs, 1);

  auto* logPage = new QWidget(innerTabs);
  auto* logLay = new QVBoxLayout(logPage);
  logLay->setContentsMargins(4, 4, 4, 4);
  auto* log = new QTextEdit(logPage);
  log->setReadOnly(true);
  log->setLineWrapMode(QTextEdit::NoWrap);
  logLay->addWidget(log);
  innerTabs->addTab(logPage, "Log");

  // Small shared helper for the raw-data table tabs below: a researcher
  // should always be able to see the actual numbers a stage produced (not
  // just a derived chart or a single final accuracy figure), and export
  // them to XLSX for their own analysis.
  auto addXlsxTableTab = [&](const QString& tabTitle, const QString& tableObjName,
                             const QString& xlsxDefaultName, const QString& emptyHint) -> QTableWidget* {
    auto* p = new QWidget(innerTabs);
    auto* lay = new QVBoxLayout(p);
    lay->setContentsMargins(4, 4, 4, 4);
    auto* hint = new QLabel(emptyHint, p);
    hint->setObjectName(tableObjName + "Hint");
    hint->setWordWrap(true);
    lay->addWidget(hint);

    auto* table = new QTableWidget(p);
    table->setObjectName(tableObjName);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->horizontalHeader()->setStretchLastSection(true);

    auto* exportBtn = new QPushButton("Export XLSX...", p);
    connect(exportBtn, &QPushButton::clicked, this, [this, table, xlsxDefaultName]() {
      const QString out = QFileDialog::getSaveFileName(this, "Export table (XLSX)",
                                                         QDir::homePath() + "/" + xlsxDefaultName,
                                                         "Excel Workbook (*.xlsx)");
      if (out.isEmpty()) return;
      if (!exportTableWidgetToXlsxNative(table, out)) {
        QMessageBox::warning(this, "Export", "Failed to export the table (it may be empty).");
      }
    });
    // Button sits above the table, at its own natural width (not stretched
    // across the tab) -- same convention as every other mode's export buttons.
    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(exportBtn);
    btnRow->addStretch(1);
    lay->addLayout(btnRow);
    lay->addWidget(table, 1);

    innerTabs->addTab(p, tabTitle);
    return table;
  };

  // Stage 1 raw data: every individual (method, problem, dim) result with
  // its rank and is_best flag -- the full detail behind the Rankings and
  // Wilcoxon charts below, so a researcher can see exactly which numbers
  // produced a given conclusion instead of only the summarized view.
  addXlsxTableTab("Batch results", "pipelineBatchResultsTable", "batch_results.xlsx",
                   "No results yet -- run stage 1 (Batch + ranking) first. "
                   "Full per-(method, problem, dim) results with rank/is_best (batch_ranked.csv).");

  // Stage 2 raw data: which landscape sample files were actually generated,
  // and their basic metadata -- confirms stage 2 covered every selected
  // (problem, dim) pair before moving on to stage 3.
  addXlsxTableTab("Landscape files", "pipelineLandscapeFilesTable", "landscape_files.xlsx",
                   "No files yet -- run stage 2 (Landscape sampling) first. "
                   "One row per generated *_features.csv.");

  // Stage 3 raw data: the full ELA feature table exactly as stage 4 will
  // train on -- lets a researcher inspect the actual feature values instead
  // of only trusting the trained model's summary.
  addXlsxTableTab("ELA features", "pipelineElaFeaturesTable", "ela_features_view.xlsx",
                   "No features yet -- run stage 3 (Extract ELA features) first. Full ela_features.csv.");

  auto* resultsPage = new QWidget(innerTabs);
  resultsPage->setObjectName("pipelineResultsPage");
  auto* resultsLay = new QVBoxLayout(resultsPage);
  resultsLay->setContentsMargins(4, 4, 4, 4);

  // Both export buttons together, at the top, side by side (feature
  // importances first, then predictions) -- the old free-text summary box
  // that used to sit here was removed; its content (accuracy vs SBS/VBS
  // etc.) is now appended to the Log tab instead when stage 4 finishes.
  auto* importanceTable = new QTableWidget(resultsPage);
  importanceTable->setObjectName("pipelineImportanceTable");
  importanceTable->setColumnCount(2);
  importanceTable->setHorizontalHeaderLabels({"ELA feature", "Importance"});
  importanceTable->horizontalHeader()->setStretchLastSection(true);
  importanceTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
  importanceTable->setMaximumHeight(200);
  auto* exportImportanceBtn = new QPushButton("Export feature importances (XLSX)...", resultsPage);
  connect(exportImportanceBtn, &QPushButton::clicked, this, [this, importanceTable]() {
    const QString out = QFileDialog::getSaveFileName(this, "Export table (XLSX)",
                                                       QDir::homePath() + "/feature_importance.xlsx",
                                                       "Excel Workbook (*.xlsx)");
    if (out.isEmpty()) return;
    if (!exportTableWidgetToXlsxNative(importanceTable, out)) {
      QMessageBox::warning(this, "Export", "Failed to export the table (it may be empty).");
    }
  });

  auto* lopoTable = new QTableWidget(resultsPage);
  lopoTable->setObjectName("pipelineLopoTable");
  lopoTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
  lopoTable->horizontalHeader()->setStretchLastSection(true);
  auto* exportLopoBtn = new QPushButton("Export predictions (XLSX)...", resultsPage);
  connect(exportLopoBtn, &QPushButton::clicked, this, [this, lopoTable]() {
    const QString out = QFileDialog::getSaveFileName(this, "Export table (XLSX)",
                                                       QDir::homePath() + "/lopo_predictions.xlsx",
                                                       "Excel Workbook (*.xlsx)");
    if (out.isEmpty()) return;
    if (!exportTableWidgetToXlsxNative(lopoTable, out)) {
      QMessageBox::warning(this, "Export", "Failed to export the table (it may be empty).");
    }
  });

  // Both buttons together in one row, at their natural width -- predictions
  // to the right of feature importances.
  auto* exportBtnRow = new QHBoxLayout();
  exportBtnRow->addWidget(exportImportanceBtn);
  exportBtnRow->addWidget(exportLopoBtn);
  exportBtnRow->addStretch(1);
  resultsLay->addLayout(exportBtnRow);

  auto* importanceLbl = new QLabel("Top ELA feature importances (full-data fit, reporting only):", resultsPage);
  resultsLay->addWidget(importanceLbl);
  resultsLay->addWidget(importanceTable);

  // Full per-instance predictions (lopo_predictions.csv) -- this is the
  // detail behind the single headline accuracy number: exactly which
  // (problem, dim) instances the model got right or wrong, and what the
  // naive Single-Best-Solver baseline would have predicted for comparison.
  auto* lopoLbl = new QLabel("Full leave-one-problem-out predictions (lopo_predictions.csv):", resultsPage);
  resultsLay->addWidget(lopoLbl);
  resultsLay->addWidget(lopoTable, 1);
  innerTabs->addTab(resultsPage, "Meta-learning results");

  // Reviewer 2, comment 4: results of the optional ablation stage (5), if
  // it has been run. Plain flat table like the raw-data tabs above --
  // ablation_summary.csv already has one row per configuration, so no
  // custom widget is needed, just the same addXlsxTableTab helper.
  addXlsxTableTab("Feature ablation", "pipelineAblationTable", "ablation_summary.xlsx",
                   "No ablation results yet -- click \"5. Feature-group ablation\" in the "
                   "pipeline panel. Compares the full model against leaving out, or using "
                   "only, each ELA feature family "
                   "(meta_learning_results/ablation/ablation_summary.csv).");

  // Reviewer 2, comment 4: results of the optional classifier-comparison
  // stage (6), if it has been run.
  addXlsxTableTab("Classifier comparison", "pipelineCompareTable", "classifier_comparison.xlsx",
                   "No comparison results yet -- click \"6. Classifier comparison\" in the "
                   "pipeline panel. Compares this study's Random Forest against SVM, Gradient "
                   "Boosting, and Logistic Regression on the same features and folds "
                   "(meta_learning_results/classifier_comparison/classifier_comparison.csv).");

  // Shared helper: export buttons (top row) + a controls panel to the
  // RIGHT of the plot (Metric/Theme/Grid, stacked vertically, fixed width)
  // -- mirrors Batch run's own Wilcoxon/Friedman tab layout exactly
  // ("Controls panel to the RIGHT of the plot"). Metric re-runs --rank
  // with a different batch_master.csv column (exposes --rank's own
  // [metric_column] argument, previously unreachable from the GUI);
  // Theme/Grid are purely cosmetic, already supported by both
  // RankPlotWidget and WilcoxonBoxPlotWidget.
  auto addExportRow = [&](QWidget* parent, QVBoxLayout* lay, QWidget* plot, QTableWidget* table,
                          const QString& plotObjName, const QString& tableObjName,
                          const QString& pngDefaultName, const QString& xlsxDefaultName) {
    plot->setObjectName(plotObjName);
    table->setObjectName(tableObjName);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->horizontalHeader()->setStretchLastSection(true);
    table->setMaximumHeight(220);

    // Create the Theme combo FIRST (before the export buttons below), since
    // the PNG export needs to read its current selection to decide whether
    // to render with a truly transparent background.
    auto* metricLbl = new QLabel("Metric", parent);
    auto* metricCombo = new QComboBox(parent);
    metricCombo->setObjectName("pipelineMetricCombo");
    metricCombo->addItem("mean_f (default)", "mean_f");
    metricCombo->addItem("min_f (best found)", "min_f");
    metricCombo->addItem("mean_evals (efficiency)", "mean_evals");
    // Deliberately NOT offering success_rate: --rank always treats the
    // chosen column as "lower = better", which would rank backwards for a
    // higher-is-better column.
    metricCombo->setToolTip(
      "Which batch_master.csv column to rank methods by (--rank always "
      "treats it as \"lower = better\"). Changing this re-runs --rank and "
      "refreshes Rankings (including its Friedman summary) and Wilcoxon "
      "together; it does not re-run stage 1's actual optimization.");
    connect(metricCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, metricCombo](int) {
      onPipelineMetricChanged(metricCombo->currentData().toString());
    });

    auto* themeLbl = new QLabel("Theme", parent);
    auto* themeCombo = new QComboBox(parent);
    themeCombo->addItem("Dark");
    themeCombo->addItem("Light");
    themeCombo->addItem("Transparent");
    themeCombo->setCurrentIndex(0);
    auto* gridLbl = new QLabel("Grid", parent);
    auto* gridCombo = new QComboBox(parent);
    gridCombo->addItem("Sparse");
    gridCombo->addItem("Medium");
    gridCombo->addItem("Dense");
    gridCombo->setCurrentIndex(1);
    connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plot](int index) {
      if (auto* rp = dynamic_cast<RankPlotWidget*>(plot)) {
        auto tm = RankPlotWidget::ThemeMode::Dark;
        if (index == 1) tm = RankPlotWidget::ThemeMode::Light;
        else if (index == 2) tm = RankPlotWidget::ThemeMode::Transparent;
        rp->setThemeMode(tm);
      } else if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(plot)) {
        auto tm = WilcoxonBoxPlotWidget::ThemeMode::Dark;
        if (index == 1) tm = WilcoxonBoxPlotWidget::ThemeMode::Light;
        else if (index == 2) tm = WilcoxonBoxPlotWidget::ThemeMode::Transparent;
        wp->setThemeMode(tm);
      }
    });
    connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [plot](int index) {
      if (auto* rp = dynamic_cast<RankPlotWidget*>(plot)) {
        auto gd = RankPlotWidget::GridDensity::Medium;
        if (index == 0) gd = RankPlotWidget::GridDensity::Sparse;
        else if (index == 2) gd = RankPlotWidget::GridDensity::Dense;
        rp->setGridDensity(gd);
      } else if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(plot)) {
        auto gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
        if (index == 0) gd = WilcoxonBoxPlotWidget::GridDensity::Sparse;
        else if (index == 2) gd = WilcoxonBoxPlotWidget::GridDensity::Dense;
        wp->setGridDensity(gd);
      }
    });

    auto* btnRow = new QHBoxLayout();
    auto* exportPngBtn = new QPushButton("Export PNG (300 DPI)...", parent);
    auto* exportXlsxBtn = new QPushButton("Export XLSX...", parent);
    connect(exportPngBtn, &QPushButton::clicked, this, [this, plot, pngDefaultName, themeCombo]() {
      const QString out = QFileDialog::getSaveFileName(this, "Export plot (PNG, 300 DPI)",
                                                         QDir::homePath() + "/" + pngDefaultName,
                                                         "PNG Image (*.png)");
      if (out.isEmpty()) return;
      const bool isTransparent = (themeCombo && themeCombo->currentIndex() == 2);
      if (!exportPipelinePlotPng(plot, out, isTransparent)) {
        QMessageBox::warning(this, "Export", "Failed to export the plot.");
      }
    });
    connect(exportXlsxBtn, &QPushButton::clicked, this, [this, table, xlsxDefaultName]() {
      const QString out = QFileDialog::getSaveFileName(this, "Export table (XLSX)",
                                                         QDir::homePath() + "/" + xlsxDefaultName,
                                                         "Excel Workbook (*.xlsx)");
      if (out.isEmpty()) return;
      if (!exportTableWidgetToXlsxNative(table, out)) {
        QMessageBox::warning(this, "Export", "Failed to export the table (it may be empty).");
      }
    });
    btnRow->addWidget(exportPngBtn);
    btnRow->addWidget(exportXlsxBtn);
    btnRow->addStretch(1);
    lay->addLayout(btnRow);

    // Controls panel to the RIGHT of the plot (mirrors Batch run's layout).
    auto* controlsPanel = new QWidget(parent);
    auto* controlsLay = new QVBoxLayout(controlsPanel);
    controlsLay->setContentsMargins(8, 0, 0, 0);
    controlsLay->setSpacing(10);
    controlsPanel->setFixedWidth(150);
    metricLbl->setParent(controlsPanel);
    metricCombo->setParent(controlsPanel);
    themeLbl->setParent(controlsPanel);
    themeCombo->setParent(controlsPanel);
    gridLbl->setParent(controlsPanel);
    gridCombo->setParent(controlsPanel);

    controlsLay->addWidget(metricLbl);
    controlsLay->addWidget(metricCombo);
    controlsLay->addSpacing(6);
    controlsLay->addWidget(themeLbl);
    controlsLay->addWidget(themeCombo);
    controlsLay->addSpacing(6);
    controlsLay->addWidget(gridLbl);
    controlsLay->addWidget(gridCombo);
    controlsLay->addStretch(1);

    auto* plotRow = new QHBoxLayout();
    plotRow->addWidget(plot, 1);
    plotRow->addWidget(controlsPanel, 0);
    lay->addLayout(plotRow, 1);
    lay->addWidget(table);
  };

  // "Rankings" tab: Friedman omnibus summary text, proportional bar chart of
  // each method's average rank (lower = better), and the full per-method
  // summary table. The summary text comes from batch_friedman.txt; the bar
  // chart and table come from batch_method_summary.csv. These two files
  // used to back separate "Rankings" and "Friedman" tabs, but both bar
  // charts plotted the identical average-rank values (batch_friedman.txt's
  // rank sums are just that same average multiplied by N), so the second
  // chart was a pure duplicate and has been folded into this one tab.
  auto* rankingsPage = new QWidget(innerTabs);
  auto* rankingsLay = new QVBoxLayout(rankingsPage);
  rankingsLay->setContentsMargins(4, 4, 4, 4);
  auto* rankingsSummary = new QLabel("No Friedman results yet -- run stage 1 (Batch + ranking) first.", rankingsPage);
  rankingsSummary->setObjectName("pipelineFriedmanSummaryLbl");
  rankingsSummary->setWordWrap(true);
  rankingsLay->addWidget(rankingsSummary);
  auto* rankingsPlot = new RankPlotWidget(rankingsPage);
  auto* rankingsTable = new QTableWidget(rankingsPage);
  addExportRow(rankingsPage, rankingsLay, rankingsPlot, rankingsTable,
               "pipelineRankingsPlot", "pipelineRankingsTable",
               "pipeline_rankings.png", "pipeline_rankings.xlsx");
  innerTabs->addTab(rankingsPage, "Rankings");

  // "Wilcoxon" tab: per-method rank-distribution box plots (from
  // batch_ranked.csv) with pairwise significance markers, plus the full
  // pairwise test table -- both from batch_wilcoxon.csv.
  auto* wilcoxonPage = new QWidget(innerTabs);
  auto* wilcoxonLay = new QVBoxLayout(wilcoxonPage);
  wilcoxonLay->setContentsMargins(4, 4, 4, 4);
  auto* wilcoxonPlot = new WilcoxonBoxPlotWidget(wilcoxonPage);
  auto* wilcoxonTable = new QTableWidget(wilcoxonPage);
  addExportRow(wilcoxonPage, wilcoxonLay, wilcoxonPlot, wilcoxonTable,
               "pipelineWilcoxonPlot", "pipelineWilcoxonTable",
               "pipeline_wilcoxon.png", "pipeline_wilcoxon.xlsx");
  innerTabs->addTab(wilcoxonPage, "Wilcoxon");

  // "Open output folder" now lives once in the global Output toolbar
  // (next to "Delete output files") rather than duplicated per-tab.

  // Pinned as the first/leftmost tab (rather than wherever addTab happens to
  // append it) so it stays easy to find alongside the many per-run tabs
  // Single/Batch/Sensitivity/Multi-objective create.
  const int tabIndex = outputTabs_->insertTab(0, page, title);

  OutputRunTab run;
  run.title = title;
  run.page = page;
  run.log = log;
  outputRuns_.push_back(run);
  pipelineOutputTabIndex_ = tabIndex;

  return tabIndex;
}

// Drains pipelineLogBuffer_ into the visible log, keeping only lines that
// look like progress/status/errors and capping how many are shown at once
// (same technique as Batch run's flushBatchUiLog(), needed here because a
// full batch sweep can print tens of thousands of per-iteration lines that
// would otherwise freeze the UI if appended live).
void MainWindow::flushPipelineLog() {
  if (pipelineLogBuffer_.isEmpty()) return;

  const QString pending = pipelineLogBuffer_;
  pipelineLogBuffer_.clear();

  const QStringList lines = pending.split(QRegularExpression(R"(\r\n|\n|\r)"), Qt::SkipEmptyParts);
  if (lines.isEmpty()) return;

  QStringList keep;
  keep.reserve(lines.size());
  for (const QString& l : lines) {
    const QString t = l.trimmed();
    if (t.isEmpty()) continue;
    if (t.startsWith("[batch]", Qt::CaseInsensitive) ||
        t.startsWith("[features]", Qt::CaseInsensitive) ||
        t.startsWith("[rank]", Qt::CaseInsensitive) ||
        t.contains("RUN SUMMARY", Qt::CaseInsensitive) ||
        t.contains("ERROR", Qt::CaseInsensitive) ||
        t.contains("error:", Qt::CaseInsensitive) ||
        t.contains("FAILED", Qt::CaseInsensitive)) {
      keep << t;
    }
  }
  if (keep.isEmpty()) return; // the vast majority of lines are per-iteration noise; drop silently

  if (keep.size() > 12) keep = keep.mid(keep.size() - 12);
  appendLog(keep.join('\n'));
}

// Finds a Python interpreter that actually has every module in
// `requiredModules` importable -- not just "some python that runs". This
// matters because a bare "py" on Windows resolves to whichever version was
// installed most recently, which is often NOT the one the required packages
// were pip-installed into (e.g. the user has both Python 3.14 and 3.12, and
// only ran "py -3.12 -m pip install ..."). Tries several common version
// selectors and returns the first one whose "import <modules>" check
// actually succeeds. Returns a "<program> [args...]" spec (split on spaces
// by the caller) or empty if none qualify.
QString MainWindow::findPythonForModules(const QStringList& requiredModules) const {
  QStringList candidates;
#if defined(_WIN32)
  candidates << "py" << "py -3.12" << "py -3.13" << "py -3.11" << "py -3.10"
             << "python" << "python3";
#else
  candidates << "python3" << "python";
#endif
  const QString importCheck = QString("import %1").arg(requiredModules.join(", "));

  for (const auto& spec : candidates) {
    QStringList parts = spec.split(' ', Qt::SkipEmptyParts);
    if (parts.isEmpty()) continue;
    const QString prog = parts.takeFirst();
    QStringList args = parts;
    args << "-c" << importCheck;

    QProcess p;
    p.start(prog, args);
    if (p.waitForStarted(1500)) {
      p.waitForFinished(6000);
      if (p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0) {
        return spec;
      }
    }
  }
  return QString();
}

// py_scripts/ lives at the project root (alongside CMakeLists.txt, src/,
// gui/) -- a fixed, predictable location regardless of where the built
// executable happens to run from, using the same root-finding logic
// already used for cliPath()'s fallback (projectRoot_).
QString MainWindow::pyScriptsDir() const {
  return QDir(projectRoot_).filePath("py_scripts");
}

QString MainWindow::datasetsDir() const {
  return QDir(projectRoot_).filePath("datasets");
}

void MainWindow::onRunPipelineBatch() {
  if (pipelineBatchProc_ && pipelineBatchProc_->state() != QProcess::NotRunning) {
    appendLog("Stopping performance batch...");
    pipelineBatchProc_->terminate();
    if (!pipelineBatchProc_->waitForFinished(2000)) pipelineBatchProc_->kill();
    return;
  }

  const QStringList methods = selectedBatchMethodShortNames();
  const QStringList problemKeys = selectedBatchProblemShortNames();
  if (methods.isEmpty() || problemKeys.isEmpty()) {
    QMessageBox::warning(this, "Meta-dataset pipeline",
      "Select at least one method and one problem (same lists as Batch run).");
    return;
  }

  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found",
      "optimsolution executable was not found next to the GUI or in build/Debug.");
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(runtimeWd);

  static const QRegularExpression dimRe(QStringLiteral(R"(\[d=(\d+)\]\s*$)"),
                                        QRegularExpression::CaseInsensitiveOption);
  QStringList pairTokens; // "base:dim" tokens (dim may be 0 = wildcard), used only in the non-loaded default path
  QVector<QPair<QString,int>> baseAndDim; // resolved (base, dim-as-selected, possibly 0) pairs, used for the loaded-experiment diff path
  for (const QString& key : problemKeys) {
    const QString base = batchBaseProblemShort(key);
    int dim = 0;
    const QRegularExpressionMatch m = dimRe.match(key.trimmed());
    if (m.hasMatch()) dim = m.captured(1).toInt();
    pairTokens << QString("%1:%2").arg(base.toLower()).arg(dim);
    baseAndDim.push_back({base.toLower(), dim});
  }

  const QString listPath = QDir(runtimeWd).filePath("headless_batch_list.generated.txt");
  QFile listFile(listPath);
  if (!listFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
    QMessageBox::critical(this, "Meta-dataset pipeline",
      "Could not write generated batch list file:\n" + listPath);
    return;
  }

  pipelinePendingDesiredKeys_.clear();

  if (pipelineExperimentLoaded_) {
    // Add/remove workflow: figure out exactly which (method,problem,dim)
    // triples are wanted NOW, compare against what a previously loaded
    // experiment already had, and only run what's genuinely missing --
    // then delete the summary CSV of anything that got unchecked, so the
    // next merge/rank reflects the removal.
    QSet<QString> desiredKeys;
    QVector<std::tuple<QString,QString,int>> desiredTriples; // (method, problem, resolved dim)
    for (const QString& method : methods) {
      for (const auto& bd : baseAndDim) {
        const QString& base = bd.first;
        const int fixed = fixedDimForProblem(base);
        const int effDim = fixed > 0 ? fixed : bd.second;
        desiredKeys.insert(method + "|" + base + "|" + QString::number(effDim));
        desiredTriples.push_back({method, base, bd.second}); // keep the 0-wildcard form for the CLI; it resolves fixed-dim itself too
      }
    }

    const QSet<QString> toRemove = pipelineLoadedTripleKeys_ - desiredKeys;
    int removedFiles = 0;
    for (const QString& key : toRemove) {
      const QStringList parts = key.split('|');
      if (parts.size() != 3) continue;
      const QString fname = QString("%1_%2_d%3_summary.csv").arg(parts[0], parts[1], parts[2]);
      if (QFile::remove(QDir(runtimeWd).filePath(fname))) ++removedFiles;
    }
    if (removedFiles > 0) {
      appendLog(QString("Removed %1 unchecked combination(s) from the experiment.").arg(removedFiles));
    }

    QStringList tripleTokens;
    int skippedAlreadyDone = 0;
    for (const auto& tr : desiredTriples) {
      const QString& method = std::get<0>(tr);
      const QString& base = std::get<1>(tr);
      const int fixed = fixedDimForProblem(base);
      const int effDim = fixed > 0 ? fixed : std::get<2>(tr);
      const QString fname = QString("%1_%2_d%3_summary.csv").arg(method, base).arg(effDim);
      if (QFile::exists(QDir(runtimeWd).filePath(fname))) {
        ++skippedAlreadyDone; // already computed in an earlier session -- don't redo it
        continue;
      }
      tripleTokens << QString("%1:%2:%3").arg(method, base).arg(std::get<2>(tr));
    }
    if (skippedAlreadyDone > 0) {
      appendLog(QString("Skipping %1 combination(s) already computed in a previous session.").arg(skippedAlreadyDone));
    }

    QTextStream out(&listFile);
    out << "# Auto-generated (add/remove mode) from the GUI's method/problem selection -- do not edit by hand.\n";
    out << "triples: " << tripleTokens.join(", ") << "\n";

    pipelinePendingDesiredKeys_ = desiredKeys;
  } else {
    QTextStream out(&listFile);
    out << "# Auto-generated from the GUI's method/problem selection -- do not edit by hand.\n";
    out << "methods: " << methods.join(", ") << "\n";
    out << "pairs: " << pairTokens.join(", ") << "\n";
  }
  listFile.close();

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (pipelineBatchProc_) { pipelineBatchProc_->deleteLater(); pipelineBatchProc_ = nullptr; }
  pipelineBatchProc_ = new QProcess(this);
  pipelineBatchProc_->setProgram(cli);
  pipelineBatchProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;
  pipelineBatchProc_->setArguments({"--batch", listPath, "."});

  if (pipelineLogFile_) { pipelineLogFile_->close(); delete pipelineLogFile_; pipelineLogFile_ = nullptr; }
  pipelineLogFile_ = new QFile(QDir(runtimeWd).filePath("pipeline_batch_console.log"), this);
  if (!pipelineLogFile_->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    appendLog("[warn] Could not open pipeline_batch_console.log for writing -- "
              "the filtered log below will still work, just no full raw copy on disk.");
  }
  pipelineLogBuffer_.clear();

  if (!pipelineFlushTimer_) {
    pipelineFlushTimer_ = new QTimer(this);
    connect(pipelineFlushTimer_, &QTimer::timeout, this, &MainWindow::flushPipelineLog);
  }
  pipelineFlushTimer_->start(400);

  connect(pipelineBatchProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelineBatchProcessFinished);
  connect(pipelineBatchProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (!pipelineBatchProc_) return;
    const QByteArray b = pipelineBatchProc_->readAllStandardOutput();
    if (pipelineLogFile_ && pipelineLogFile_->isOpen()) pipelineLogFile_->write(b);
    pipelineLogBuffer_ += QString::fromUtf8(b);
    constexpr int kMaxBufferChars = 400000;
    if (pipelineLogBuffer_.size() > kMaxBufferChars)
      pipelineLogBuffer_.remove(0, pipelineLogBuffer_.size() - kMaxBufferChars);
  });
  connect(pipelineBatchProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (!pipelineBatchProc_) return;
    const QByteArray b = pipelineBatchProc_->readAllStandardError();
    if (pipelineLogFile_ && pipelineLogFile_->isOpen()) pipelineLogFile_->write(b);
    pipelineLogBuffer_ += QString::fromUtf8(b);
  });

  appendLog("----------------------------------------");
  appendLog("[1/4] Running performance batch: " + cli + " --batch " + listPath + " .");
  if (pipelineBatchSpinner_) pipelineBatchSpinner_->start();
  startPipelineProgress("Running batch...");
  pipelineStage1Timer_.start();
  setPipelineUiEnabled(false);
  pipelineBatchProc_->start();
}

void MainWindow::onPipelineBatchProcessFinished(int exitCode, QProcess::ExitStatus status) {
  if (pipelineFlushTimer_) pipelineFlushTimer_->stop();
  flushPipelineLog();
  if (pipelineLogFile_) pipelineLogFile_->close();

  const QString runtimeWd = lastRuntimeWorkingDir_;
  const QString masterCsv = QDir(runtimeWd).filePath("batch_master.csv");

  if (status == QProcess::NormalExit && exitCode == 0 && QFileInfo::exists(masterCsv)) {
    appendLog("Batch finished. Computing rankings (Friedman/Wilcoxon)...");
    startPipelineProgress("Computing rankings...");
    const QString cli = cliPath();
    if (pipelineRankProc_) { pipelineRankProc_->deleteLater(); pipelineRankProc_ = nullptr; }
    pipelineRankProc_ = new QProcess(this);
    pipelineRankProc_->setProgram(cli);
    pipelineRankProc_->setWorkingDirectory(runtimeWd);
    pipelineRankProc_->setArguments({"--rank", "batch_master.csv", "batch", pipelineMetricColumn_});
    connect(pipelineRankProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onPipelineRankProcessFinished);
    connect(pipelineRankProc_, &QProcess::readyReadStandardOutput, this, [this]() {
      if (pipelineRankProc_) appendLog(QString::fromUtf8(pipelineRankProc_->readAllStandardOutput()).trimmed());
    });
    connect(pipelineRankProc_, &QProcess::readyReadStandardError, this, [this]() {
      if (pipelineRankProc_) appendLog(QString::fromUtf8(pipelineRankProc_->readAllStandardError()).trimmed());
    });
    pipelineRankProc_->start();
    return;
  }

  QString msg;
  if (status != QProcess::NormalExit) msg = "Performance batch crashed.";
  else if (exitCode != 0) msg = QString("Performance batch finished with errors (exit code %1). See log above.").arg(exitCode);
  else msg = "Performance batch finished, but batch_master.csv was not found -- rankings were not computed.";
  appendLog(msg);
  if (pipelineBatchSpinner_) pipelineBatchSpinner_->stop();
  finishPipelineProgress(false); // only reached on failure/missing-csv paths
  if (auto* tab = outputTab(activeOutputRunIndex_))
    if (tab->page)
      if (auto* lbl = tab->page->findChild<QLabel*>("pipelineInfoLbl")) lbl->setText(msg);
}

// Diagnostic transparency: after an add/remove Stage-1 rerun, explicitly
// compares the CURRENT batch_ranked.csv against the baseline that was
// loaded (pipelineLoadedTripleKeys_) at the (problem,dim)-group level, and
// logs anything that changed a group's completeness -- this is exactly
// what shifts the Friedman/Wilcoxon "N (complete groups)" count between
// two exports taken before/after a reload, and previously had no visible
// explanation in the log.
void MainWindow::logPipelineGroupChanges() {
  if (!pipelineExperimentLoaded_ || pipelineLoadedTripleKeys_.isEmpty()) return;

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  const auto rows = readPipelineCsv(QDir(runtimeWd).filePath("batch_ranked.csv"));
  if (rows.isEmpty()) return;
  const QStringList& header = rows[0];
  const int cProblem = pipelineCsvCol(header, "problem");
  const int cDim = pipelineCsvCol(header, "dim");
  const int cMethod = pipelineCsvCol(header, "method");
  if (cProblem < 0 || cDim < 0 || cMethod < 0) return;

  QMap<QString, QSet<QString>> nowMethodsByGroup;
  QSet<QString> allMethodsNow;
  for (int r = 1; r < rows.size(); ++r) {
    const QStringList& row = rows[r];
    if (cMethod >= row.size() || cProblem >= row.size() || cDim >= row.size()) continue;
    const QString key = row[cProblem].trimmed() + "|" + row[cDim].trimmed();
    const QString method = row[cMethod].trimmed();
    nowMethodsByGroup[key].insert(method);
    allMethodsNow.insert(method);
  }
  const int totalMethodCount = allMethodsNow.size();
  if (totalMethodCount <= 0) return;

  QMap<QString, QSet<QString>> beforeMethodsByGroup;
  for (const QString& k : pipelineLoadedTripleKeys_) {
    const QStringList parts = k.split('|'); // method|problem|dim
    if (parts.size() != 3) continue;
    beforeMethodsByGroup[parts[1] + "|" + parts[2]].insert(parts[0]);
  }

  int newGroups = 0, newlyCompleted = 0, newlyIncomplete = 0;
  QStringList newlyCompletedNames;
  for (auto it = nowMethodsByGroup.begin(); it != nowMethodsByGroup.end(); ++it) {
    const bool completeNow = (it.value().size() >= totalMethodCount);
    if (!beforeMethodsByGroup.contains(it.key())) {
      ++newGroups;
      continue;
    }
    const bool completeBefore = (beforeMethodsByGroup.value(it.key()).size() >= totalMethodCount);
    if (completeNow && !completeBefore) { ++newlyCompleted; newlyCompletedNames << it.key(); }
    else if (!completeNow && completeBefore) { ++newlyIncomplete; }
  }

  if (newGroups > 0 || newlyCompleted > 0 || newlyIncomplete > 0) {
    QString msg = QString(
      "[transparency] vs. the previously loaded experiment: %1 brand-new (problem,dim) group(s), "
      "%2 group(s) that just became complete, %3 group(s) that became incomplete. "
      "This is exactly why Friedman/Wilcoxon's \"N (complete groups)\" can differ from an earlier export."
    ).arg(newGroups).arg(newlyCompleted).arg(newlyIncomplete);
    if (!newlyCompletedNames.isEmpty()) {
      msg += " Newly complete: " + newlyCompletedNames.join(", ") + ".";
    }
    appendLog(msg);
  }
}

void MainWindow::onPipelineRankProcessFinished(int exitCode, QProcess::ExitStatus status) {
  if (pipelineBatchSpinner_) pipelineBatchSpinner_->stop(); // stage 1 = batch + rank together
  QString msg;
  if (status != QProcess::NormalExit || exitCode != 0) {
    msg = "Batch finished, but computing rankings failed. See log above; "
          "optimsolution_gui_run/batch_master.csv is still available.";
  } else {
    msg = "Stage 1 done: batch_master.csv, batch_ranked.csv, batch_method_summary.csv, "
          "batch_friedman.txt, batch_wilcoxon.csv are in optimsolution_gui_run/. "
          "Next: stage 2 (landscape sampling).";
    populatePipelineStatsTabs();
    pipelineStage1Done_ = true;
    refreshPipelineButtonStates();
    savePipelineStageTiming("batch_ranking", pipelineStage1Timer_.elapsed() / 1000.0);
    refreshPipelineTimingLabels();
    // Report exactly what changed vs. the loaded baseline BEFORE rolling
    // that baseline forward -- otherwise there would be nothing left to
    // compare against.
    logPipelineGroupChanges();
    // Roll the add/remove baseline forward: whatever was just (re)computed
    // is now "the experiment" for any future load/add/remove cycle in this
    // session, without needing to explicitly reload it from disk.
    if (pipelineExperimentLoaded_ && !pipelinePendingDesiredKeys_.isEmpty()) {
      pipelineLoadedTripleKeys_ = pipelinePendingDesiredKeys_;
    }
  }
  finishPipelineProgress(status == QProcess::NormalExit && exitCode == 0);
  appendLog(msg);
  setPipelineUiEnabled(true);
  if (auto* tab = outputTab(activeOutputRunIndex_))
    if (tab->page)
      if (auto* lbl = tab->page->findChild<QLabel*>("pipelineInfoLbl")) lbl->setText(msg);
}

void MainWindow::onRunPipelineFeatures() {
  if (pipelineFeaturesProc_ && pipelineFeaturesProc_->state() != QProcess::NotRunning) {
    appendLog("Stopping landscape sampling...");
    pipelineFeatureQueue_.clear(); // don't advance to the next pair after this one is killed
    pipelineFeaturesProc_->terminate();
    if (!pipelineFeaturesProc_->waitForFinished(2000)) pipelineFeaturesProc_->kill();
    return;
  }

  const QStringList problemKeys = selectedBatchProblemShortNames();
  if (problemKeys.isEmpty()) {
    QMessageBox::warning(this, "Meta-dataset pipeline",
      "Select at least one problem first (same list as Batch run).");
    return;
  }

  static const QRegularExpression dimRe(QStringLiteral(R"(\[d=(\d+)\]\s*$)"),
                                        QRegularExpression::CaseInsensitiveOption);
  pipelineFeatureQueue_.clear();
  for (const QString& key : problemKeys) {
    const QString base = batchBaseProblemShort(key).toLower();
    int dim = 0;
    const QRegularExpressionMatch m = dimRe.match(key.trimmed());
    if (m.hasMatch()) dim = m.captured(1).toInt();
    pipelineFeatureQueue_.push_back({base, dim});
  }
  pipelineFeatureTotal_ = (int)pipelineFeatureQueue_.size();

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  appendLog("----------------------------------------");
  appendLog(QString("[2/4] Landscape sampling: %1 problem/dim pair(s) queued.").arg(pipelineFeatureTotal_));
  if (pipelineFeaturesSpinner_) pipelineFeaturesSpinner_->start();
  startPipelineProgress(QString("Landscape sampling (0/%1)...").arg(pipelineFeatureTotal_));
  pipelineStage2Timer_.start();
  setPipelineUiEnabled(false);
  onPipelineFeaturesProcessFinished(0, QProcess::NormalExit); // kicks off the first pair
}

// Advances pipelineFeatureQueue_ by one pair each time it's called -- once
// as a direct kick-off from onRunPipelineFeatures(), then once per
// QProcess::finished from here on, until the queue is empty.
void MainWindow::onPipelineFeaturesProcessFinished(int exitCode, QProcess::ExitStatus status) {
  // Report the pair that just finished (skipped on the initial kick-off call,
  // which is only ever made with a full, untouched queue).
  const bool isInitialKickoff = (pipelineFeaturesProc_ == nullptr);
  if (!isInitialKickoff) {
    if (status != QProcess::NormalExit || exitCode != 0) {
      appendLog(QString("  [features] failed (exit code %1) -- continuing with remaining pairs.").arg(exitCode));
    }
  }

  if (pipelineFeatureQueue_.empty()) {
    if (pipelineFeaturesSpinner_) pipelineFeaturesSpinner_->stop();
    setPipelineUiEnabled(true);
    if (!isInitialKickoff) {
      finishPipelineProgress(true);
      const QString msg = QString("Stage 2 done: %1 *_features.csv file(s) written to optimsolution_gui_run/. "
                                   "Next: stage 3 (extract ELA features).").arg(pipelineFeatureTotal_);
      appendLog(msg);
      populatePipelineLandscapeFilesTable();
      pipelineStage2Done_ = true;
      refreshPipelineButtonStates();
      savePipelineStageTiming("landscape_sampling", pipelineStage2Timer_.elapsed() / 1000.0);
      refreshPipelineTimingLabels();
      if (auto* tab = outputTab(activeOutputRunIndex_))
        if (tab->page)
          if (auto* lbl = tab->page->findChild<QLabel*>("pipelineInfoLbl")) lbl->setText(msg);
    }
    return;
  }

  if (outputProgressBar_ && !isInitialKickoff) {
    const int doneSoFar = pipelineFeatureTotal_ - (int)pipelineFeatureQueue_.size();
    outputProgressBar_->setFormat(QString("Landscape sampling (%1/%2)...").arg(doneSoFar).arg(pipelineFeatureTotal_));
  }

  const auto pair = pipelineFeatureQueue_.front();
  pipelineFeatureQueue_.erase(pipelineFeatureQueue_.begin());
  const QString problem = pair.first;
  const int dim = pair.second;
  const int fixed = fixedDimForProblem(problem);
  const int effDim = (fixed > 0) ? fixed : dim;
  const int samples = pipelineSamplesSpin_ ? pipelineSamplesSpin_->value() : 0;

  const QString cli = cliPath();
  if (cli.isEmpty() || !QFileInfo::exists(cli)) {
    QMessageBox::critical(this, "CLI not found",
      "optimsolution executable was not found next to the GUI or in build/Debug.");
    pipelineFeatureQueue_.clear();
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(runtimeWd);

  if (pipelineFeaturesProc_) { pipelineFeaturesProc_->deleteLater(); pipelineFeaturesProc_ = nullptr; }
  pipelineFeaturesProc_ = new QProcess(this);
  pipelineFeaturesProc_->setProgram(cli);
  pipelineFeaturesProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;

  QStringList args;
  args << "--features" << problem << (fixed > 0 ? "0" : QString::number(effDim));
  if (samples > 0) args << QString::number(samples);
  pipelineFeaturesProc_->setArguments(args);

  connect(pipelineFeaturesProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelineFeaturesProcessFinished);
  connect(pipelineFeaturesProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelineFeaturesProc_) appendLog("  " + QString::fromUtf8(pipelineFeaturesProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelineFeaturesProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelineFeaturesProc_) appendLog("  " + QString::fromUtf8(pipelineFeaturesProc_->readAllStandardError()).trimmed());
  });

  const int doneCount = pipelineFeatureTotal_ - (int)pipelineFeatureQueue_.size();
  appendLog(QString("  (%1/%2) %3 d%4 ...").arg(doneCount).arg(pipelineFeatureTotal_).arg(problem).arg(effDim));
  pipelineFeaturesProc_->start();
}

// See the [pipeline] extract_ela_jobs comment in optimsolution.cfg and
// extract_ela_features.py's own --jobs docstring for the full reasoning
// (each *_features.csv file is processed completely independently).
int MainWindow::pipelineExtractElaJobsFromSettings() const {
  if (!cfg_) return 1;
  bool ok = false;
  const int n = cfg_->value("pipeline", "extract_ela_jobs", "1").toInt(&ok);
  return (ok && n > 1) ? n : 1;
}

void MainWindow::onRunPipelineExtractEla() {
  if (pipelinePyProc_ && pipelinePyProc_->state() != QProcess::NotRunning) {
    if (pipelinePyRunningStage_ == PipelinePyStage::Train) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Meta-learner training is still running. Wait for it to finish, or "
        "click \"Train meta-learner\" again to stop it, before starting ELA extraction.");
      return;
    }
    appendLog("Stopping ELA extraction...");
    pipelinePyProc_->terminate();
    if (!pipelinePyProc_->waitForFinished(2000)) pipelinePyProc_->kill();
    return;
  }

  const QStringList required = {"pandas", "numpy", "pflacco"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, and pflacco "
      "all importable (tried \"py\" with several version selectors, "
      "\"python\", and \"python3\").\n\n"
      "See py_scripts/README_python_ela.md for the exact install commands "
      "(pflacco needs its dependencies installed manually -- \"pip install "
      "pflacco\" alone does not work).");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("extract_ela_features.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  QDir().mkpath(runtimeWd);

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (pipelinePyProc_) { pipelinePyProc_->deleteLater(); pipelinePyProc_ = nullptr; }
  pipelinePyProc_ = new QProcess(this);
  pipelinePyProc_->setProgram(pythonProg);
  pipelinePyProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;
  QStringList elaArgs = pythonParts + QStringList{script, ".", "ela_features.csv"};
  const int extractJobs = pipelineExtractElaJobsFromSettings();
  if (extractJobs > 1) elaArgs << "--jobs" << QString::number(extractJobs);
  pipelinePyProc_->setArguments(elaArgs);

  connect(pipelinePyProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelinePyProcessFinished);
  connect(pipelinePyProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelinePyProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[3/4] Extracting ELA features: %1 -> ela_features.csv").arg(pythonSpec));
  if (extractJobs > 1) {
    appendLog(QString("  (optimsolution.cfg [pipeline] extract_ela_jobs=%1: processing that many "
                       "files concurrently.)").arg(extractJobs));
  }
  if (pipelineExtractSpinner_) pipelineExtractSpinner_->start();
  startPipelineProgress("Extracting ELA features...");
  pipelinePyRunningStage_ = PipelinePyStage::Extract;
  pipelineStage3Timer_.start();
  setPipelineUiEnabled(false);
  pipelinePyProc_->start();
}

void MainWindow::onRunPipelineTrainModel() {
  if (pipelinePyProc_ && pipelinePyProc_->state() != QProcess::NotRunning) {
    if (pipelinePyRunningStage_ == PipelinePyStage::Extract) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "ELA extraction is still running. Wait for it to finish, or click "
        "\"Extract ELA features\" again to stop it, before starting training.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::Ablation) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Feature-group ablation is still running. Wait for it to finish, or click "
        "\"Feature-group ablation\" again to stop it, before starting training.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::CompareClassifiers) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Classifier comparison is still running. Wait for it to finish, or click "
        "\"Classifier comparison\" again to stop it, before starting training.");
      return;
    }
    appendLog("Stopping meta-learner training...");
    pipelinePyProc_->terminate();
    if (!pipelinePyProc_->waitForFinished(2000)) pipelinePyProc_->kill();
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  const QString elaCsv = QDir(runtimeWd).filePath("ela_features.csv");
  const QString rankedCsv = QDir(runtimeWd).filePath("batch_ranked.csv");
  QStringList missing;
  if (!QFileInfo::exists(elaCsv)) missing << "ela_features.csv (run stage 3 first)";
  if (!QFileInfo::exists(rankedCsv)) missing << "batch_ranked.csv (run stage 1 first)";
  if (!missing.isEmpty()) {
    QMessageBox::warning(this, "Meta-dataset pipeline",
      "Missing required file(s) in optimsolution_gui_run/:\n- " + missing.join("\n- "));
    return;
  }

  const QStringList required = {"pandas", "numpy", "sklearn"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, and "
      "scikit-learn all importable (tried \"py\" with several version "
      "selectors, \"python\", and \"python3\").\n\n"
      "See py_scripts/README_python_ela.md for the exact install commands.");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("train_meta_learner.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (pipelinePyProc_) { pipelinePyProc_->deleteLater(); pipelinePyProc_ = nullptr; }
  pipelinePyProc_ = new QProcess(this);
  pipelinePyProc_->setProgram(pythonProg);
  pipelinePyProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;
  pipelinePyProc_->setArguments(pythonParts + QStringList{script, "ela_features.csv", "batch_ranked.csv", "meta_learning_results"});

  connect(pipelinePyProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelinePyProcessFinished);
  connect(pipelinePyProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelinePyProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[4/4] Training meta-learner: %1 -> meta_learning_results/").arg(pythonSpec));
  if (pipelineTrainSpinner_) pipelineTrainSpinner_->start();
  startPipelineProgress("Training meta-learner...");
  pipelinePyRunningStage_ = PipelinePyStage::Train;
  pipelineStage4Timer_.start();
  setPipelineUiEnabled(false);
  pipelinePyProc_->start();
}

// Reviewer 2, comment 4: optional 5th pipeline stage. Runs the exact same
// script as stage 4, train_meta_learner.py, with an extra "--ablation"
// argument -- it always performs the normal stage-4 run first (so the
// "Meta-learning results" tab and its files are refreshed identically to
// clicking stage 4 directly), then additionally repeats the fold-safe LOPO
// evaluation once per ELA feature family, writing the comparison to
// optimsolution_gui_run/meta_learning_results/ablation/. Deliberately a
// SEPARATE button from stage 4 rather than a checkbox on it: the ablation
// takes roughly 13x longer (13 full LOPO passes vs. 1), so it should never
// run by accident when a user only wanted the quick stage-4 result.
void MainWindow::onRunPipelineAblation() {
  if (pipelinePyProc_ && pipelinePyProc_->state() != QProcess::NotRunning) {
    if (pipelinePyRunningStage_ == PipelinePyStage::Extract) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "ELA extraction is still running. Wait for it to finish, or click "
        "\"Extract ELA features\" again to stop it, before starting the ablation.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::Train) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Meta-learner training is still running. Wait for it to finish, or click "
        "\"Train model\" again to stop it, before starting the ablation.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::CompareClassifiers) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Classifier comparison is still running. Wait for it to finish, or click "
        "\"Classifier comparison\" again to stop it, before starting the ablation.");
      return;
    }
    appendLog("Stopping feature-group ablation...");
    pipelinePyProc_->terminate();
    if (!pipelinePyProc_->waitForFinished(2000)) pipelinePyProc_->kill();
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  const QString elaCsv = QDir(runtimeWd).filePath("ela_features.csv");
  const QString rankedCsv = QDir(runtimeWd).filePath("batch_ranked.csv");
  QStringList missing;
  if (!QFileInfo::exists(elaCsv)) missing << "ela_features.csv (run stage 3 first)";
  if (!QFileInfo::exists(rankedCsv)) missing << "batch_ranked.csv (run stage 1 first)";
  if (!missing.isEmpty()) {
    QMessageBox::warning(this, "Meta-dataset pipeline",
      "Missing required file(s) in optimsolution_gui_run/:\n- " + missing.join("\n- "));
    return;
  }

  const QStringList required = {"pandas", "numpy", "sklearn"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, and "
      "scikit-learn all importable (tried \"py\" with several version "
      "selectors, \"python\", and \"python3\").\n\n"
      "See py_scripts/README_python_ela.md for the exact install commands.");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("train_meta_learner.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (pipelinePyProc_) { pipelinePyProc_->deleteLater(); pipelinePyProc_ = nullptr; }
  pipelinePyProc_ = new QProcess(this);
  pipelinePyProc_->setProgram(pythonProg);
  pipelinePyProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;
  pipelinePyProc_->setArguments(pythonParts + QStringList{script, "ela_features.csv", "batch_ranked.csv", "meta_learning_results", "--ablation"});

  connect(pipelinePyProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelinePyProcessFinished);
  connect(pipelinePyProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelinePyProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[5/5] Feature-group ablation: %1 -> meta_learning_results/ablation/ "
                     "(13 LOPO passes -- this takes noticeably longer than stage 4)").arg(pythonSpec));
  if (pipelineAblationSpinner_) pipelineAblationSpinner_->start();
  startPipelineProgress("Running feature-group ablation...");
  pipelinePyRunningStage_ = PipelinePyStage::Ablation;
  pipelineAblationTimer_.start();
  setPipelineUiEnabled(false);
  pipelinePyProc_->start();
}

// Reviewer 2, comment 4: optional 6th pipeline stage, alongside ablation.
// Runs the exact same script as stage 4, train_meta_learner.py, with an
// extra "--compare-classifiers" argument -- it always performs the normal
// stage-4 run first (so "Meta-learning results" is refreshed identically
// to clicking stage 4 directly), then additionally repeats the fold-safe
// LOPO evaluation with three alternative classifiers (SVM, Gradient
// Boosting, Logistic Regression) on the same features, writing the
// comparison to optimsolution_gui_run/meta_learning_results/
// classifier_comparison/. A separate button from both stage 4 and the
// ablation stage, for the same reason the ablation stage itself is
// separate: it takes noticeably longer (4 full LOPO passes) than a single
// stage-4 click, so it should never run by accident.
void MainWindow::onRunPipelineCompareClassifiers() {
  if (pipelinePyProc_ && pipelinePyProc_->state() != QProcess::NotRunning) {
    if (pipelinePyRunningStage_ == PipelinePyStage::Extract) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "ELA extraction is still running. Wait for it to finish, or click "
        "\"Extract ELA features\" again to stop it, before starting the comparison.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::Train) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Meta-learner training is still running. Wait for it to finish, or click "
        "\"Train model\" again to stop it, before starting the comparison.");
      return;
    }
    if (pipelinePyRunningStage_ == PipelinePyStage::Ablation) {
      QMessageBox::information(this, "Meta-dataset pipeline",
        "Feature-group ablation is still running. Wait for it to finish, or click "
        "\"Feature-group ablation\" again to stop it, before starting the comparison.");
      return;
    }
    appendLog("Stopping classifier comparison...");
    pipelinePyProc_->terminate();
    if (!pipelinePyProc_->waitForFinished(2000)) pipelinePyProc_->kill();
    return;
  }

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");

  const QString elaCsv = QDir(runtimeWd).filePath("ela_features.csv");
  const QString rankedCsv = QDir(runtimeWd).filePath("batch_ranked.csv");
  QStringList missing;
  if (!QFileInfo::exists(elaCsv)) missing << "ela_features.csv (run stage 3 first)";
  if (!QFileInfo::exists(rankedCsv)) missing << "batch_ranked.csv (run stage 1 first)";
  if (!missing.isEmpty()) {
    QMessageBox::warning(this, "Meta-dataset pipeline",
      "Missing required file(s) in optimsolution_gui_run/:\n- " + missing.join("\n- "));
    return;
  }

  const QStringList required = {"pandas", "numpy", "sklearn"};
  const QString pythonSpec = findPythonForModules(required);
  if (pythonSpec.isEmpty()) {
    QMessageBox::critical(this, "Python not found or missing packages",
      "Could not find a Python installation with pandas, numpy, and "
      "scikit-learn all importable (tried \"py\" with several version "
      "selectors, \"python\", and \"python3\").\n\n"
      "See py_scripts/README_python_ela.md for the exact install commands.");
    return;
  }
  QStringList pythonParts = pythonSpec.split(' ', Qt::SkipEmptyParts);
  const QString pythonProg = pythonParts.takeFirst();

  const QString script = QDir(pyScriptsDir()).filePath("train_meta_learner.py");
  if (!QFileInfo::exists(script)) {
    QMessageBox::critical(this, "Script not found",
      "Expected to find:\n" + script + "\n\n"
      "Make sure py_scripts/ sits at the project root (next to CMakeLists.txt).");
    return;
  }

  const int outIdx = createPipelineOutputTab();
  if (outIdx >= 0 && outputTabs_) outputTabs_->setCurrentIndex(outIdx);
  activeOutputRunIndex_ = outIdx;

  if (pipelinePyProc_) { pipelinePyProc_->deleteLater(); pipelinePyProc_ = nullptr; }
  pipelinePyProc_ = new QProcess(this);
  pipelinePyProc_->setProgram(pythonProg);
  pipelinePyProc_->setWorkingDirectory(runtimeWd);
  lastRuntimeWorkingDir_ = runtimeWd;
  pipelinePyProc_->setArguments(pythonParts + QStringList{script, "ela_features.csv", "batch_ranked.csv", "meta_learning_results", "--compare-classifiers"});

  connect(pipelinePyProc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, &MainWindow::onPipelinePyProcessFinished);
  connect(pipelinePyProc_, &QProcess::readyReadStandardOutput, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardOutput()).trimmed());
  });
  connect(pipelinePyProc_, &QProcess::readyReadStandardError, this, [this]() {
    if (pipelinePyProc_) appendLog(QString::fromUtf8(pipelinePyProc_->readAllStandardError()).trimmed());
  });

  appendLog("----------------------------------------");
  appendLog(QString("[6/6] Classifier comparison: %1 -> meta_learning_results/classifier_comparison/ "
                     "(4 LOPO passes -- this takes noticeably longer than stage 4)").arg(pythonSpec));
  if (pipelineCompareSpinner_) pipelineCompareSpinner_->start();
  startPipelineProgress("Running classifier comparison...");
  pipelinePyRunningStage_ = PipelinePyStage::CompareClassifiers;
  pipelineCompareTimer_.start();
  setPipelineUiEnabled(false);
  pipelinePyProc_->start();
}

// Shared finished-handler for both Python stages (3 and 4) -- only one of
// them can ever be running at a time (both use pipelinePyProc_), so we
// distinguish which one just finished by inspecting the full argument list
// for the script name (the script may not be at a fixed index, since a
// versioned "py -3.12" spec inserts an extra leading argument).
void MainWindow::onPipelinePyProcessFinished(int exitCode, QProcess::ExitStatus status) {
  const QString allArgs = pipelinePyProc_ ? pipelinePyProc_->arguments().join(' ') : QString();
  const bool wasExtract = allArgs.contains("extract_ela_features.py");
  const bool wasTrain = allArgs.contains("train_meta_learner.py");
  // Reviewer 2, comment 4: the ablation invocation is train_meta_learner.py
  // with an extra "--ablation" flag, so wasTrain is also true for it --
  // wasAblation is checked FIRST below (it implies wasTrain) so ablation
  // runs get their own message and additionally populate the ablation
  // table, while still refreshing the ordinary stage-4 outputs too (the
  // script performs that same run internally before the ablation sweep).
  const bool wasAblation = wasTrain && allArgs.contains("--ablation");
  // Same reasoning as wasAblation: the classifier-comparison invocation is
  // also train_meta_learner.py, with an extra "--compare-classifiers" flag.
  const bool wasCompare = wasTrain && allArgs.contains("--compare-classifiers");

  // Both stages share pipelinePyProc_; stop whichever spinner was running.
  if (pipelineExtractSpinner_) pipelineExtractSpinner_->stop();
  if (pipelineTrainSpinner_) pipelineTrainSpinner_->stop();
  if (pipelineAblationSpinner_) pipelineAblationSpinner_->stop();
  if (pipelineCompareSpinner_) pipelineCompareSpinner_->stop();
  pipelinePyRunningStage_ = PipelinePyStage::None;

  QString msg;
  bool failed = false;
  if (status != QProcess::NormalExit) {
    msg = "Python stage crashed.";
    failed = true;
  } else if (exitCode != 0) {
    msg = QString("Python stage finished with errors (exit code %1). See the log for details -- "
                   "check py_scripts/README_python_ela.md for dependency setup.").arg(exitCode);
    failed = true;
  } else if (wasExtract) {
    msg = "Stage 3 done: ela_features.csv written to optimsolution_gui_run/. Next: stage 4 (train meta-learner).";
    populatePipelineElaFeaturesTable();
    pipelineStage3Done_ = true;
    refreshPipelineButtonStates();
    savePipelineStageTiming("extract_ela", pipelineStage3Timer_.elapsed() / 1000.0);
    refreshPipelineTimingLabels();
  } else if (wasAblation) {
    msg = "Stage 5 done: feature-group ablation results written to "
          "optimsolution_gui_run/meta_learning_results/ablation/ "
          "(and meta_learning_results/ refreshed with the same full-model run stage 4 produces).";
    pipelineStage4Done_ = true;     // the script re-ran the ordinary stage-4 pass internally
    pipelineAblationDone_ = true;
    refreshPipelineButtonStates();
    savePipelineStageTiming("train_model", pipelineStage4Timer_.elapsed() / 1000.0);
    savePipelineStageTiming("ablation", pipelineAblationTimer_.elapsed() / 1000.0);
    refreshPipelineTimingLabels();
  } else if (wasCompare) {
    msg = "Stage 6 done: classifier comparison results written to "
          "optimsolution_gui_run/meta_learning_results/classifier_comparison/ "
          "(and meta_learning_results/ refreshed with the same full-model run stage 4 produces).";
    pipelineStage4Done_ = true;     // the script re-ran the ordinary stage-4 pass internally
    pipelineCompareDone_ = true;
    refreshPipelineButtonStates();
    savePipelineStageTiming("train_model", pipelineStage4Timer_.elapsed() / 1000.0);
    savePipelineStageTiming("compare_classifiers", pipelineCompareTimer_.elapsed() / 1000.0);
    refreshPipelineTimingLabels();
  } else if (wasTrain) {
    msg = "Stage 4 done: results written to optimsolution_gui_run/meta_learning_results/.";
    pipelineStage4Done_ = true;
    refreshPipelineButtonStates();
    savePipelineStageTiming("train_model", pipelineStage4Timer_.elapsed() / 1000.0);
    refreshPipelineTimingLabels();
  } else {
    msg = "Python stage finished.";
  }
  appendLog(msg);
  finishPipelineProgress(!failed);
  setPipelineUiEnabled(true);
  if (failed) {
    // A log line alone is easy to miss if the tab isn't in focus -- a
    // failed stage needs to be impossible to overlook.
    const QString failTitle = wasExtract ? "ELA extraction failed"
                             : wasAblation ? "Feature-group ablation failed"
                             : wasCompare ? "Classifier comparison failed"
                             : wasTrain ? "Meta-learner training failed"
                             : "Python stage failed";
    QMessageBox::warning(this, failTitle, msg);
  }
  if (auto* tab = outputTab(activeOutputRunIndex_)) {
    if (tab->page) {
      if (auto* lbl = tab->page->findChild<QLabel*>("pipelineInfoLbl")) lbl->setText(msg);

      // On a successful training (or ablation) run, populate the
      // "Meta-learning results" tab with the actual output instead of
      // leaving the user to go dig through
      // optimsolution_gui_run/meta_learning_results/ by hand.
      if ((wasTrain || wasAblation || wasCompare) && !failed) {
        // The old free-text summary box in "Meta-learning results" was
        // removed -- its content (accuracy vs SBS/VBS, etc.) now goes into
        // the Log tab instead, right after the "Stage 4/5/6 done" line.
        // Only done here, for a run that just finished -- not from
        // detectPipelineExistingOutputs(), where re-dumping it on every
        // mode entry would just be log spam.
        QString baseWd = QCoreApplication::applicationDirPath();
        if (!settingsPath_.isEmpty()) {
          QFileInfo fi(settingsPath_);
          if (fi.exists()) baseWd = fi.absolutePath();
        }
        const QString resultsDir = QDir(baseWd).filePath("optimsolution_gui_run/meta_learning_results");
        QFile summaryFile(QDir(resultsDir).filePath("summary.txt"));
        if (summaryFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
          appendLog(QString::fromUtf8(summaryFile.readAll()));
        }

        populatePipelineMetaLearningResults(wasAblation, wasCompare, /*jumpToTab=*/true);
      }
    }
  }
}



// Shared by onPipelinePyProcessFinished() (stage 4/5/6 just finished) and
// detectPipelineExistingOutputs() (stage 4/5/6 output found already on
// disk): reads optimsolution_gui_run/meta_learning_results/ into the
// "Meta-learning results" tables, and, if includeAblation / includeCompare,
// its ablation/ / classifier_comparison/ subfolders into the "Feature
// ablation" / "Classifier comparison" tables too. Creates (or reuses) the
// pipeline output tab itself, so this is safe to call even when nothing
// has been run yet this session.
void MainWindow::populatePipelineMetaLearningResults(bool includeAblation, bool includeCompare, bool jumpToTab) {
  const int outIdx = createPipelineOutputTab();
  if (outIdx < 0) return;
  activeOutputRunIndex_ = outIdx;
  auto* tab = outputTab(outIdx);
  if (!tab || !tab->page) return;

  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString resultsDir = QDir(baseWd).filePath("optimsolution_gui_run/meta_learning_results");

  if (auto* table = tab->page->findChild<QTableWidget*>("pipelineImportanceTable")) {
    table->setRowCount(0);
    QFile impFile(QDir(resultsDir).filePath("feature_importance.csv"));
    if (impFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
      QTextStream ts(&impFile);
      bool isHeader = true;
      int row = 0;
      while (!ts.atEnd()) {
        const QString line = ts.readLine();
        if (line.trimmed().isEmpty()) continue;
        if (isHeader) { isHeader = false; continue; } // skip "feature,importance" header
        const QStringList parts = line.split(',');
        if (parts.size() < 2) continue;
        table->insertRow(row);
        table->setItem(row, 0, new QTableWidgetItem(parts[0]));
        table->setItem(row, 1, new QTableWidgetItem(parts[1]));
        ++row;
      }
    }
  }

  // Full leave-one-problem-out predictions table.
  if (auto* lopoTable = tab->page->findChild<QTableWidget*>("pipelineLopoTable")) {
    populateTableFromCsvRows(lopoTable, readPipelineCsv(QDir(resultsDir).filePath("lopo_predictions.csv")));
  }

  // Reviewer 2, comment 4: feature-group ablation comparison table,
  // written by train_meta_learner.py --ablation into its own subfolder
  // (kept separate from the main run's files above).
  if (includeAblation) {
    if (auto* ablationTable = tab->page->findChild<QTableWidget*>("pipelineAblationTable")) {
      const QString ablationDir = QDir(resultsDir).filePath("ablation");
      populateTableFromCsvRows(ablationTable, readPipelineCsv(QDir(ablationDir).filePath("ablation_summary.csv")));
    }
  }

  // Reviewer 2, comment 4: alternative-classifier comparison table,
  // written by train_meta_learner.py --compare-classifiers into its own
  // subfolder (kept separate from both the main run's files and ablation's).
  if (includeCompare) {
    if (auto* compareTable = tab->page->findChild<QTableWidget*>("pipelineCompareTable")) {
      const QString compareDir = QDir(resultsDir).filePath("classifier_comparison");
      populateTableFromCsvRows(compareTable, readPipelineCsv(QDir(compareDir).filePath("classifier_comparison.csv")));
    }
  }

  if (!jumpToTab) return;

  // Jump to the results tab automatically so the user sees them
  // immediately rather than having to notice and click over. Found by
  // object name (not a hardcoded index) since the exact tab order can
  // change as more raw-data tabs get added over time. An ablation or
  // comparison result jumps straight to ITS OWN tab, not back to
  // "Meta-learning results". If both happen to be requested at once (not
  // currently possible from the GUI, since each is its own button/run),
  // ablation takes priority, simply as a fixed, predictable choice.
  if (auto* innerTabs = tab->page->findChild<QTabWidget*>("pipelineInnerTabs")) {
    // addXlsxTableTab() adds the table directly into the QWidget page it
    // registers with innerTabs->addTab(), so the table's own
    // parentWidget() is exactly that page.
    auto jumpToTable = [&](const char* tableObjName) {
      if (auto* w = tab->page->findChild<QTableWidget*>(tableObjName)) {
        if (auto* pageW = w->parentWidget()) {
          const int idx = innerTabs->indexOf(pageW);
          if (idx >= 0) innerTabs->setCurrentIndex(idx);
        }
      }
    };
    if (includeAblation) {
      jumpToTable("pipelineAblationTable");
    } else if (includeCompare) {
      jumpToTable("pipelineCompareTable");
    } else if (auto* resultsPageW = tab->page->findChild<QWidget*>("pipelineResultsPage")) {
      const int idx = innerTabs->indexOf(resultsPageW);
      if (idx >= 0) innerTabs->setCurrentIndex(idx);
    }
  }
}

// Re-derives the pipeline stage-done flags from whatever output files
// already exist in optimsolution_gui_run/, instead of assuming "not done"
// just because this is a fresh GUI session -- entering pipeline mode (or
// loading a previous experiment's selection) used to always reset every
// checkmark and require re-clicking every stage button even when its
// output was already sitting on disk from an earlier session. Each stage
// that IS found done also has its table populated, exactly as if that
// stage had just finished, so resuming earlier work needs no clicking
// through stages whose work is already complete -- only whichever stage
// actually stops being done (e.g. because ela_features.csv is newer/older
// than expected) still needs a click.
//
// Deliberately conservative: this only ever turns a flag ON when it finds
// the corresponding output file non-empty on disk. It never turns a flag
// off and never deletes or overwrites anything -- if in doubt (e.g. a
// stage's file is missing or empty), that stage is simply left "not done",
// same as the previous behaviour, so a user who really did delete some
// output still sees an honest, un-checked stage rather than a false
// checkmark.
void MainWindow::detectPipelineExistingOutputs() {
  QString baseWd = QCoreApplication::applicationDirPath();
  if (!settingsPath_.isEmpty()) {
    QFileInfo fi(settingsPath_);
    if (fi.exists()) baseWd = fi.absolutePath();
  }
  const QString runtimeWd = QDir(baseWd).filePath("optimsolution_gui_run");
  if (!QDir(runtimeWd).exists()) return; // nothing to detect yet

  auto nonEmptyFileExists = [](const QString& path) {
    QFileInfo fi(path);
    return fi.exists() && fi.isFile() && fi.size() > 0;
  };

  // Stage 1: batch execution + ranking. batch_ranked.csv is written only
  // after --rank succeeds, so its presence implies stage 1 genuinely
  // completed (not merely started).
  if (nonEmptyFileExists(QDir(runtimeWd).filePath("batch_ranked.csv"))) {
    const int outIdx = createPipelineOutputTab();
    if (outIdx >= 0) {
      activeOutputRunIndex_ = outIdx;
      pipelineStage1Done_ = true;
      populatePipelineStatsTabs();
    }
  }

  // Stage 2: landscape sampling. One *_features.csv per (problem, dim)
  // pair -- excluding ela_features.csv itself, which matches the same
  // glob but is stage 3's output, not stage 2's.
  {
    QDir dir(runtimeWd);
    const QStringList files = dir.entryList(QStringList{"*_features.csv"}, QDir::Files);
    bool anyLandscapeFile = false;
    for (const QString& f : files) {
      if (f.compare("ela_features.csv", Qt::CaseInsensitive) == 0) continue;
      anyLandscapeFile = true;
      break;
    }
    if (anyLandscapeFile) {
      const int outIdx = createPipelineOutputTab();
      if (outIdx >= 0) {
        activeOutputRunIndex_ = outIdx;
        pipelineStage2Done_ = true;
        populatePipelineLandscapeFilesTable();
      }
    }
  }

  // Stage 3: ELA feature extraction.
  if (nonEmptyFileExists(QDir(runtimeWd).filePath("ela_features.csv"))) {
    const int outIdx = createPipelineOutputTab();
    if (outIdx >= 0) {
      activeOutputRunIndex_ = outIdx;
      pipelineStage3Done_ = true;
      populatePipelineElaFeaturesTable();
    }
  }

  // Stage 4: meta-learner training.
  const QString resultsDir = QDir(runtimeWd).filePath("meta_learning_results");
  if (nonEmptyFileExists(QDir(resultsDir).filePath("lopo_predictions.csv"))) {
    pipelineStage4Done_ = true;
    // Reviewer 2, comment 4: stage 5's ablation summary and stage 6's
    // classifier comparison, if present -- independent of each other,
    // since either can be run without the other.
    const bool hasAblation = nonEmptyFileExists(QDir(resultsDir).filePath("ablation/ablation_summary.csv"));
    const bool hasCompare = nonEmptyFileExists(QDir(resultsDir).filePath("classifier_comparison/classifier_comparison.csv"));
    if (hasAblation) pipelineAblationDone_ = true;
    if (hasCompare) pipelineCompareDone_ = true;
    populatePipelineMetaLearningResults(hasAblation, hasCompare, /*jumpToTab=*/false);
  }

  refreshPipelineButtonStates();
  refreshPipelineTimingLabels();
}


void MainWindow::loadParetoCsvIntoTab(int index, const QString& csvPath) {
  auto* tab = outputTab(index);
  if (!tab || !tab->isMultiObjective) return;

  QFile f(csvPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    if (tab->moInfoLbl) tab->moInfoLbl->setText("Could not open: " + csvPath);
    return;
  }

  QTextStream ts(&f);
  QStringList header;
  QVector<QStringList> rows;
  bool first = true;
  while (!ts.atEnd()) {
    const QString line = ts.readLine();
    if (line.trimmed().isEmpty()) continue;
    if (first) { header = line.split(','); first = false; continue; }
    rows.push_back(line.split(','));
  }
  f.close();

  if (header.isEmpty() || rows.isEmpty()) {
    if (tab->moInfoLbl) tab->moInfoLbl->setText("Pareto CSV was empty: " + csvPath);
    return;
  }

  // Objective columns are named f1..fK by the CLI (see runMultiObjectiveCli in main.cpp).
  QVector<int> objCols;
  for (int c = 0; c < header.size(); ++c) {
    if (header[c].startsWith('f')) objCols.push_back(c);
  }

  if (auto* table = tab->moParetoTable) {
    table->setColumnCount(header.size());
    table->setHorizontalHeaderLabels(header);
    table->setRowCount(rows.size());
    for (int r = 0; r < rows.size(); ++r) {
      for (int c = 0; c < header.size() && c < rows[r].size(); ++c) {
        table->setItem(r, c, new QTableWidgetItem(rows[r][c]));
      }
    }
    table->resizeColumnsToContents();
  }

  if (auto* plot = dynamic_cast<ParetoFrontWidget*>(tab->moParetoPlot)) {
    QVector<double> f1, f2;
    if (objCols.size() >= 2) {
      for (const auto& row : rows) {
        if (objCols[0] < row.size() && objCols[1] < row.size()) {
          f1.push_back(row[objCols[0]].toDouble());
          f2.push_back(row[objCols[1]].toDouble());
        }
      }
    }
    plot->setPoints(f1, f2, header.value(objCols.value(0, 0)), header.value(objCols.value(1, 0)));
    plot->setTitleSubtitle(
      QString("Pareto front — %1 / %2").arg(tab->methodShort, tab->problemShort),
      QString("%1 non-dominated solutions (dim=%2)%3")
        .arg(rows.size())
        .arg(tab->problemDim)
        .arg(objCols.size() > 2 ? QString(" — showing objectives 1 & 2 of %1").arg(objCols.size()) : QString()));
  }

  if (tab->moInfoLbl) {
    tab->moInfoLbl->setText(QString("Loaded %1 Pareto-optimal solutions from: %2").arg(rows.size()).arg(csvPath));
  }
}

bool MainWindow::exportParetoPng(int index, const QString& filePath) const {
  const auto* tab = outputTab(index);
  if (!tab) return false;
  auto* plot = dynamic_cast<ParetoFrontWidget*>(tab->moParetoPlot);
  if (!plot) return false;

  const int targetDpi = 300;
  const int baseDpi = 96;
  const double scale = double(targetDpi) / double(baseDpi);

  const QSize sz = plot->size();
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
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}

void MainWindow::onExportParetoPng() {
  auto* tab = activeOutputTab();
  if (!tab || !tab->isMultiObjective) return;
  const QString out = QFileDialog::getSaveFileName(
    this, "Export Pareto front (PNG, 300 DPI)",
    QDir::homePath() + "/pareto_front.png",
    "PNG Image (*.png)");
  if (out.isEmpty()) return;
  if (!exportParetoPng(activeOutputRunIndex_, out)) {
    QMessageBox::warning(this, "Export", "Failed to export the Pareto front plot.");
  }
}

void MainWindow::onExportParetoTable() {
  auto* tab = activeOutputTab();
  if (!tab || !tab->isMultiObjective || !tab->moParetoTable || tab->moParetoTable->rowCount() == 0) {
    QMessageBox::warning(this, "Export", "No Pareto results to export yet.");
    return;
  }
  const QString out = QFileDialog::getSaveFileName(
    this, "Export Pareto results table",
    QDir::homePath() + "/pareto_results.xlsx",
    "Excel Workbook (*.xlsx);;CSV (*.csv)");
  if (out.isEmpty()) return;

  const QString lower = out.toLower();
  if (lower.endsWith(".xlsx")) {
    if (exportTableWidgetToXlsxNative(tab->moParetoTable, out)) return;
    QMessageBox::warning(this, "Export", "Failed to write the XLSX file.");
    return;
  }

  // CSV fallback.
  QFile f(out);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, "Export", "Failed to write the CSV file.");
    return;
  }
  QTextStream ts(&f);
  QStringList header;
  for (int c = 0; c < tab->moParetoTable->columnCount(); ++c) {
    header << (tab->moParetoTable->horizontalHeaderItem(c) ? tab->moParetoTable->horizontalHeaderItem(c)->text() : QString());
  }
  ts << header.join(",") << "\n";
  for (int r = 0; r < tab->moParetoTable->rowCount(); ++r) {
    QStringList row;
    for (int c = 0; c < tab->moParetoTable->columnCount(); ++c) {
      auto* it = tab->moParetoTable->item(r, c);
      row << (it ? it->text() : QString());
    }
    ts << row.join(",") << "\n";
  }
}


} // namespace optimsolution_gui
