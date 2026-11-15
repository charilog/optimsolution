// MainWindow_Plots.cpp
// Convergence and distribution plots per output tab, overlays, axis/theme/grid options and PNG exports.
// Split out of the former monolithic MainWindow.cpp (v52).
#include "MainWindowShared.h"

namespace optimsolution_gui {
void MainWindow::updateConvergencePlotForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab) return;

  auto* plot = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot);
  if (!plot) return;

  // Apply display options (Theme + Grid density) even when no data is loaded.
  if (tab->themeCombo) {
    const int ti = tab->themeCombo->currentIndex();
    const auto tm = (ti == 1) ? ConvergencePlotWidget::ThemeMode::Light
                  : (ti == 2) ? ConvergencePlotWidget::ThemeMode::Transparent
                              : ConvergencePlotWidget::ThemeMode::Dark;
    plot->setThemeMode(tm);
  }
  if (tab->gridDensityCombo) {
    const int idx = tab->gridDensityCombo->currentIndex();
    const auto gd = (idx <= 0) ? ConvergencePlotWidget::GridDensity::Sparse
                               : (idx == 1 ? ConvergencePlotWidget::GridDensity::Medium
                                           : ConvergencePlotWidget::GridDensity::Dense);
    plot->setGridDensity(gd);
  }

  plot->clearBands();
  plot->clearYAxisLowerAnchor();

  // Keep all plot widgets in this output tab consistent with the Convergence display options.
  const bool darkTheme = (!tab->themeCombo || tab->themeCombo->currentIndex() != 1);
  const bool transpTheme = (tab->themeCombo && tab->themeCombo->currentIndex() == 2);
  if (auto* splot = dynamic_cast<SensitivityBarWidget*>(tab->sensitivityPlot)) {
    const auto stm = transpTheme ? SensitivityBarWidget::ThemeMode::Transparent
                   : darkTheme   ? SensitivityBarWidget::ThemeMode::Dark
                                 : SensitivityBarWidget::ThemeMode::Light;
    splot->setThemeMode(stm);
    if (tab->gridDensityCombo) {
      const int idx = tab->gridDensityCombo->currentIndex();
      const auto gd = (idx <= 0) ? SensitivityBarWidget::GridDensity::Sparse
                                 : (idx == 1 ? SensitivityBarWidget::GridDensity::Medium
                                             : SensitivityBarWidget::GridDensity::Dense);
      splot->setGridDensity(gd);
    }
  }

  if (auto* dplot = dynamic_cast<DistributionPlotWidget*>(tab->distributionPlot)) {
    // Always use themeCombo (convergence) as source of truth — it is already at the
    // new value when this runs. distThemeCombo is a visual mirror synced afterwards.
    const int dti = tab->themeCombo ? tab->themeCombo->currentIndex() : 0;
    const auto dtm = (dti == 1) ? DistributionPlotWidget::ThemeMode::Light
                   : (dti == 2) ? DistributionPlotWidget::ThemeMode::Transparent
                                : DistributionPlotWidget::ThemeMode::Dark;
    dplot->setThemeMode(dtm);
    if (tab->gridDensityCombo) {
      const int idx = tab->gridDensityCombo->currentIndex();
      const auto gd = (idx <= 0) ? DistributionPlotWidget::GridDensity::Sparse
                                 : (idx == 1 ? DistributionPlotWidget::GridDensity::Medium
                                             : DistributionPlotWidget::GridDensity::Dense);
      dplot->setGridDensity(gd);
    }
  }

  if (!tab->convLoaded) {
    plot->clear();
    return;
  }

  const bool overlayByMethod  = (tab->overlayMethodChk && tab->overlayMethodChk->isChecked());
  const bool overlayByProblem = (tab->overlayProblemChk && tab->overlayProblemChk->isChecked());

  auto extractKnownBestMinimum = [&](const OutputRunTab* src) -> double {
    if (!src || !src->log) return std::numeric_limits<double>::quiet_NaN();
    const QString text = src->log->toPlainText();
    if (text.trimmed().isEmpty()) return std::numeric_limits<double>::quiet_NaN();

    const QStringList lines = text.split(QRegularExpression(QStringLiteral("\\r\\n|\\n|\\r")));
    QString valueText;
    for (int i = lines.size() - 1; i >= 0; --i) {
      const QString ln = lines[i].trimmed();
      if (ln.startsWith("Best f (min):", Qt::CaseInsensitive)) {
        valueText = ln.mid(QString("Best f (min):").size()).trimmed();
        break;
      }
      if (ln.startsWith("Best f:", Qt::CaseInsensitive)) {
        valueText = ln.mid(QString("Best f:").size()).trimmed();
        break;
      }
    }
    if (valueText.isEmpty()) return std::numeric_limits<double>::quiet_NaN();

    const QRegularExpression numRe(QStringLiteral(R"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"));
    const QRegularExpressionMatch m = numRe.match(valueText);
    if (!m.hasMatch()) return std::numeric_limits<double>::quiet_NaN();

    bool ok = false;
    const double v = QLocale::c().toDouble(m.captured(0), &ok);
    return ok ? v : std::numeric_limits<double>::quiet_NaN();
  };

  const bool useEvalX = (tab->xAxisEvalRadio && tab->xAxisEvalRadio->isEnabled() && tab->xAxisEvalRadio->isChecked());
  plot->setXAxisTitle(useEvalX ? "Function evaluations" : "Iterations");

  auto selectX = [&](const OutputRunTab& r) -> const QVector<double>& {
    if (useEvalX && r.convHasEvalX && !r.convEvalX.isEmpty()) return r.convEvalX;
    return r.convIterX;
  };

  
  const bool bandEnabled = (tab->convBandChk && tab->convBandChk->isEnabled() && tab->convBandChk->isChecked());

  const double knownBestMinimum = (!overlayByMethod) ? extractKnownBestMinimum(tab)
                                                     : std::numeric_limits<double>::quiet_NaN();
  if (std::isfinite(knownBestMinimum)) {
    plot->setYAxisLowerAnchor(knownBestMinimum);
  }

  // Summary convergence: median line + IQR band across runs (only for non-overlay mode).
  if (bandEnabled && !overlayByMethod && !overlayByProblem) {
    const QVector<double>& x = selectX(*tab);
    const QVector<double>& y = tab->convY;
    const int nxy = (int(x.size()) < int(y.size()) ? int(x.size()) : int(y.size()));

    struct Run { QVector<double> x; QVector<double> y; };
    QVector<Run> runs;
    Run cur;

    bool havePrev = false;
    double prevX = 0.0;

    for (int i = 0; i < nxy; ++i) {
      const double xi = x[i];
      const double yi = y[i];
      if (!std::isfinite(xi) || !std::isfinite(yi)) continue;

      if (havePrev && xi < prevX) {
        if (cur.x.size() >= 2 && cur.y.size() >= 2) runs.push_back(std::move(cur));
        cur = Run{};
      }

      cur.x.push_back(xi);
      cur.y.push_back(yi);

      prevX = xi;
      havePrev = true;
    }
    if (cur.x.size() >= 2 && cur.y.size() >= 2) runs.push_back(std::move(cur));

    if (runs.size() >= 2) {
      // Build X grid spanning [minX, maxX] of actual run data (avoids leading NaN values).
      double minX = std::numeric_limits<double>::infinity();
      double maxX = 0.0;
      for (const auto& r : runs) {
        if (!r.x.isEmpty()) {
          minX = std::min(minX, r.x.front());
          maxX = std::max(maxX, r.x.back());
        }
      }
      if (!std::isfinite(minX) || minX < 0.0) minX = 0.0;
      if (maxX <= minX) maxX = minX + 1.0;

      const int maxPoints = 2500;
      const double span = maxX - minX;
      long long maxStep = (long long)std::llround(span);
      long long stride = 1;
      if (maxStep > maxPoints) stride = (long long)std::ceil(double(maxStep) / double(maxPoints));
      if (stride < 1) stride = 1;

      QVector<double> xGrid;
      xGrid.reserve(int(maxStep / stride) + 2);
      for (long long s = 0; s <= maxStep; s += stride) xGrid.push_back(minX + double(s));
      if (xGrid.isEmpty() || xGrid.back() < maxX) xGrid.push_back(maxX);

      // For each run, sample best_f at each xGrid position (last observation carried forward within the run).
      QVector<QVector<double>> sampled;
      sampled.reserve(runs.size());
      for (const auto& r : runs) {
        QVector<double> ys;
        ys.reserve(xGrid.size());

        int j = 0;
        double lastY = std::numeric_limits<double>::quiet_NaN();
        bool haveY = false;

        for (int gi = 0; gi < xGrid.size(); ++gi) {
          const double xt = xGrid[gi];
          while (j < r.x.size() && r.x[j] <= xt) {
            if (std::isfinite(r.y[j])) { lastY = r.y[j]; haveY = true; }
            ++j;
          }
          ys.push_back(haveY ? lastY : std::numeric_limits<double>::quiet_NaN());
        }

        sampled.push_back(std::move(ys));
      }

      auto quant = [&](QVector<double> v, double q) -> double {
        if (v.isEmpty()) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end());
        if (v.size() == 1) return v[0];
        const double pos = q * double(v.size() - 1);
        const int a = int(std::floor(pos));
        const int n = int(v.size());
        const int b = (a + 1 < n ? (a + 1) : (n - 1));
        const double t = pos - double(a);
        return v[a] * (1.0 - t) + v[b] * t;
      };

      QVector<double> yMed;
      QVector<double> yQ1;
      QVector<double> yQ3;
      yMed.reserve(xGrid.size());
      yQ1.reserve(xGrid.size());
      yQ3.reserve(xGrid.size());

      for (int gi = 0; gi < xGrid.size(); ++gi) {
        QVector<double> vals;
        vals.reserve(sampled.size());
        for (const auto& ys : sampled) {
          if (gi < ys.size() && std::isfinite(ys[gi])) vals.push_back(ys[gi]);
        }
        if (vals.isEmpty()) {
          yMed.push_back(std::numeric_limits<double>::quiet_NaN());
          yQ1.push_back(std::numeric_limits<double>::quiet_NaN());
          yQ3.push_back(std::numeric_limits<double>::quiet_NaN());
          continue;
        }
        yQ1.push_back(quant(vals, 0.25));
        yMed.push_back(quant(vals, 0.50));
        yQ3.push_back(quant(vals, 0.75));
      }

      plot->setXAxisTitle(useEvalX ? "Function evaluations" : "Iterations");
      plot->setIqrBand(xGrid, yQ1, yQ3, "IQR (25–75%)");
      plot->setSeries(xGrid, yMed, "Median");
      return;
    }
  }
if (!overlayByMethod && !overlayByProblem) {
    QString label = tab->problemShort;
    if (label.isEmpty()) label = tab->title;
    if (tab->problemDim > 0) label += QString(" (D=%1)").arg(tab->problemDim);
    plot->setSeries(selectX(*tab), tab->convY, label);
    return;
  }

  QVector<ConvergencePlotWidget::Series> series;
  if (overlayByMethod) {
    // Overlay: plot all problems that have convergence data for this method.
    for (const auto& r : outputRuns_) {
      if (!r.convLoaded) continue;
      if (r.methodShort.compare(tab->methodShort, Qt::CaseInsensitive) != 0) continue;

      ConvergencePlotWidget::Series s;
      s.label = r.problemShort.isEmpty() ? r.title : r.problemShort;
      if (r.problemDim > 0) s.label += QString(" (D=%1)").arg(r.problemDim);
      s.x = selectX(r);
      s.y = r.convY;
      series.push_back(std::move(s));
    }
  } else {
    // Overlay: plot all methods that have convergence data for this problem.
    for (const auto& r : outputRuns_) {
      if (!r.convLoaded) continue;
      if (r.problemShort.compare(tab->problemShort, Qt::CaseInsensitive) != 0) continue;
      if (tab->problemDim > 0 && r.problemDim > 0 && r.problemDim != tab->problemDim) continue;

      ConvergencePlotWidget::Series s;
      s.label = r.methodShort.isEmpty() ? r.title : r.methodShort;
      if (r.problemDim > 0) s.label += QString(" (D=%1)").arg(r.problemDim);
      s.x = selectX(r);
      s.y = r.convY;
      series.push_back(std::move(s));
    }
  }

  if (series.isEmpty()) {
    QString label = tab->problemShort;
    if (label.isEmpty()) label = tab->title;
    if (tab->problemDim > 0) label += QString(" (D=%1)").arg(tab->problemDim);
    plot->setSeries(selectX(*tab), tab->convY, label);
    return;
  }

  plot->setSeriesList(std::move(series));
}


void MainWindow::updateDistributionPlotForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab) return;

  auto* info = tab->distributionInfo;
  auto* plot = dynamic_cast<DistributionPlotWidget*>(tab->distributionPlot);
  if (!info || !plot) return;

  // themeCombo (convergence) is always the authoritative source: it is already at the
  // new value when this function runs, regardless of which combo triggered the update.
  // distThemeCombo is a visual mirror synced after the fact, so never read it here.
  const int themeIdx = tab->themeCombo ? tab->themeCombo->currentIndex() : 0;
  const auto dtm = (themeIdx == 1) ? DistributionPlotWidget::ThemeMode::Light
                 : (themeIdx == 2) ? DistributionPlotWidget::ThemeMode::Transparent
                                   : DistributionPlotWidget::ThemeMode::Dark;
  plot->setThemeMode(dtm);
  const int gridIdx = tab->gridDensityCombo ? tab->gridDensityCombo->currentIndex() : 1;
  {
    const auto gd = (gridIdx <= 0) ? DistributionPlotWidget::GridDensity::Sparse
                  : (gridIdx == 1)  ? DistributionPlotWidget::GridDensity::Medium
                                    : DistributionPlotWidget::GridDensity::Dense;
    plot->setGridDensity(gd);
  }

  plot->clear();

  if (!tab->convLoaded) {
    info->setText("Distribution: no convergence data loaded.");
    if (tab->convBandChk) tab->convBandChk->setEnabled(false);
    return;
  }

  const QVector<double>& x = !tab->convIterX.isEmpty() ? tab->convIterX
                                                      : (tab->convHasEvalX ? tab->convEvalX : tab->convIterX);
  const QVector<double>& y = tab->convY;
  const int nxy = (int(x.size()) < int(y.size()) ? int(x.size()) : int(y.size()));

  QVector<double> finals;
  finals.reserve(64);

  bool havePrev = false;
  double prevX = 0.0;

  double lastY = std::numeric_limits<double>::quiet_NaN();
  bool haveY = false;

  for (int i = 0; i < nxy; ++i) {
    const double xi = x[i];
    const double yi = y[i];
    if (!std::isfinite(xi) || !std::isfinite(yi)) continue;

    if (havePrev && xi < prevX) {
      if (haveY) finals.push_back(lastY);
      haveY = false;
      lastY = std::numeric_limits<double>::quiet_NaN();
    }

    lastY = yi;
    haveY = true;

    prevX = xi;
    havePrev = true;
  }
  if (haveY) finals.push_back(lastY);

  if (tab->convBandChk) tab->convBandChk->setEnabled(finals.size() >= 2);

  if (finals.isEmpty()) {
    info->setText("Distribution: no per-run final values could be extracted from the convergence CSV.");
    return;
  }

  DistributionPlotWidget::Group g;
  g.label = QString("final best_f");
  g.values = finals;
  plot->setGroups(QVector<DistributionPlotWidget::Group>() << g);

  // Summary line (discreet).
  QVector<double> vals;
  vals.reserve(finals.size());
  for (double v : finals) if (std::isfinite(v)) vals.push_back(v);
  std::sort(vals.begin(), vals.end());

  const int n = int(vals.size());

  const auto q = [&](double qq) -> double {
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();
    if (n == 1) return vals[0];
    const double pos = qq * double(n - 1);
    const int a = int(std::floor(pos));
    const int b = std::min(a + 1, n - 1);
    const double t = pos - double(a);
    return vals[a] * (1.0 - t) + vals[b] * t;
  };

  double mean = 0.0;
  for (double v : vals) mean += v;
  mean /= double(std::max(1, n));

  double var = 0.0;
  for (double v : vals) var += (v - mean) * (v - mean);
  var /= double(std::max(1, n - 1));
  const double sd = std::sqrt(std::max(0.0, var));


    const QString summary = QString("Distribution from convergence CSV: n=%1 | min=%2 | median=%3 | mean=%4 | sd=%5")
                            .arg(vals.size())
                            .arg(QString::number(vals.front(), 'g', 6))
                            .arg(QString::number(q(0.50), 'g', 6))
                            .arg(QString::number(mean, 'g', 6))
                            .arg(QString::number(sd, 'g', 6));
  info->setText(QString()); // summary is shown inside the plot image
  info->setToolTip(summary);
}

void MainWindow::onOverlayMethodToggled(bool) {
  // Identify which output tab triggered this toggle, then refresh that tab's plot.
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].overlayMethodChk == s) {
      if (outputRuns_[i].overlayMethodChk->isChecked() && outputRuns_[i].overlayProblemChk) {
        QSignalBlocker b(outputRuns_[i].overlayProblemChk);
        outputRuns_[i].overlayProblemChk->setChecked(false);
      }
      updateConvergencePlotForTab(i);
      break;
    }
  }
}


void MainWindow::onOverlayProblemToggled(bool checked) {
  // Identify which output tab triggered this toggle, then refresh that tab's plot.
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].overlayProblemChk == s) {
      if (checked && outputRuns_[i].overlayMethodChk) {
        QSignalBlocker b(outputRuns_[i].overlayMethodChk);
        outputRuns_[i].overlayMethodChk->setChecked(false);
      }
      updateConvergencePlotForTab(i);
      break;
    }
  }
}

void MainWindow::onConvergenceXAxisToggled(bool checked) {
  if (!checked) return;
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].xAxisIterRadio == s || outputRuns_[i].xAxisEvalRadio == s) {
      updateConvergencePlotForTab(i);
      break;
    }
  }
}

void MainWindow::onConvergenceThemeToggled(bool /*checked*/) {
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].themeCombo == s
     || outputRuns_[i].distThemeCombo == s
     || outputRuns_[i].gridDensityCombo == s) {
      updateConvergencePlotForTab(i);
      updateDistributionPlotForTab(i);
      break;
    }
  }
}

void MainWindow::onConvergenceGridChanged(int /*index*/) {
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].gridDensityCombo == s
     || outputRuns_[i].distGridDensityCombo == s) {
      updateConvergencePlotForTab(i);
      updateDistributionPlotForTab(i);
      break;
    }
  }
}

void MainWindow::onConvergenceBandToggled(bool /*checked*/) {
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].convBandChk == s) {
      updateConvergencePlotForTab(i);
      break;
    }
  }
}

void MainWindow::onOutputTabCloseRequested(int index) {
  if (index < 0) return;

  QWidget* summaryPageToForget = nullptr;
  if (const auto* tab = outputTab(index)) {
    if (tab->methodShort.compare("System", Qt::CaseInsensitive) == 0) {
      if (batchActive_ && tab->page == g_activeBatchSummaryPage) {
        QMessageBox::information(this, "Batch in progress",
                                 "The active batch summary tab cannot be closed while a batch run is still in progress.");
        return;
      }
      summaryPageToForget = tab->page;
    }
  }

  // Do not allow closing the active running tab.
  if (proc_ && proc_->state() != QProcess::NotRunning && index == activeOutputRunIndex_) {
    QMessageBox::information(this, "Run in progress",
                             "This output tab is attached to the currently running process and cannot be closed.");
    return;
  }

  if (!outputTabs_) return;
  if (index < 0 || index >= outputTabs_->count()) return;

  QWidget* w = outputTabs_->widget(index);
  outputTabs_->removeTab(index);

  // FIX 3: Clear member pointers that reference widgets owned by this page
  // BEFORE calling deleteLater().  deleteLater() is asynchronous — Qt processes
  // pending events before the destruction, so signal handlers (e.g. onExportStats,
  // rebuildStatsComparisons) could fire with a dangling statsTable_ / statsTabs_
  // pointer if we don't nullify them first.
  if (summaryPageToForget) {
    if (g_activeBatchSummaryPage == summaryPageToForget) {
      // Null out every member pointer that was bound to this page's UI bundle.
      statsTabs_           = nullptr;
      statsTable_          = nullptr;
      exportStatsBtn_      = nullptr;
      statsNoteLbl_        = nullptr;
      wilcoxonPairsCombo_  = nullptr;
      statsAlphaCombo_     = nullptr;
      exportWilcoxonPlotBtn_ = nullptr;
      wilcoxonPlot_        = nullptr;
      wilcoxonSummaryLbl_  = nullptr;
      exportRankPlotBtn_   = nullptr;
      rankPlot_            = nullptr;
      statsSummaryLbl_     = nullptr;
      statsPairwiseTable_  = nullptr;
      g_activeBatchSummaryPage = nullptr;
      g_liveBatchCsvPaths.clear();
    }
    g_batchSummaryUi.remove(summaryPageToForget);
    g_batchSummarySnapshots.remove(summaryPageToForget);
    batchSummaryCellsByPage_.remove(summaryPageToForget);
  }

  if (w) w->deleteLater();

  if (index >= 0 && index < int(outputRuns_.size())) {
    outputRuns_.erase(outputRuns_.begin() + index);
  }

  if (activeOutputRunIndex_ == index) {
    activeOutputRunIndex_ = 0;
  } else if (activeOutputRunIndex_ > index) {
    activeOutputRunIndex_--;
  }
}

void MainWindow::onReloadConvergence() {
  if (!outputTabs_) return;
  tryLoadConvergenceForTab(outputTabs_->currentIndex());
  tryLoadLandscapeForTab(outputTabs_->currentIndex());
}

void MainWindow::onClearCsvFiles() {
  QSet<QString> roots;

  auto addRunRoot = [&roots](const QString& path) {
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()) return;

    QFileInfo fi(trimmed);
    QString dirPath;
    if (fi.exists()) {
      dirPath = fi.isDir() ? fi.absoluteFilePath() : fi.absolutePath();
    } else {
      dirPath = trimmed;
    }

    const QDir dir(dirPath);
    if (dir.dirName().compare("optimsolution_gui_run", Qt::CaseInsensitive) != 0) return;
    if (!QFileInfo(dir.absolutePath()).exists()) return;

    roots.insert(dir.absolutePath());
  };

  // Current-session runtime folders, but only the exact optimsolution_gui_run folder.
  for (const auto& r : outputRuns_) {
    addRunRoot(r.runtimeWorkingDir);
  }
  addRunRoot(lastRuntimeWorkingDir_);

  // Fallback: allow deletion immediately after startup, but only inside
  // the optimsolution_gui_run folder.
  addRunRoot(QDir(QCoreApplication::applicationDirPath()).filePath("optimsolution_gui_run"));
  if (!settingsPath_.trimmed().isEmpty()) {
    const QFileInfo fi(settingsPath_);
    if (fi.exists()) {
      addRunRoot(QDir(fi.absolutePath()).filePath("optimsolution_gui_run"));
    }
  }

  QStringList rootList = roots.values();
  std::sort(rootList.begin(), rootList.end(), [](const QString& a, const QString& b){ return a.toLower() < b.toLower(); });

  // Count everything (any file, any subfolder) under each root -- this now
  // deletes the WHOLE optimsolution_gui_run folder tree, not just CSV/XLSX.
  qint64 totalFiles = 0;
  QStringList existingRoots;
  for (const QString& root : rootList) {
    QDir d(root);
    if (!d.exists()) continue;
    existingRoots.push_back(root);
    QDirIterator it(root, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) { it.next(); ++totalFiles; }
  }

  if (existingRoots.isEmpty() || totalFiles == 0) {
    QMessageBox::information(this, "Delete output files", "The optimsolution_gui_run folder is already empty.");
    return;
  }

  const QString prompt = QString(
    "This will delete EVERYTHING (%1 file(s), including all subfolders) under "
    "the following optimsolution_gui_run folder(s):\n\n%2\n\nContinue?"
  ).arg(totalFiles).arg(existingRoots.join("\n"));

  const auto btn = QMessageBox::question(this, "Delete output files", prompt, QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (btn != QMessageBox::Yes) return;

  int removedRoots = 0;
  int failedRoots = 0;
  for (const QString& root : existingRoots) {
    QDir d(root);
    // removeRecursively() deletes the folder itself too, so recreate an
    // empty one in its place immediately -- callers throughout the app
    // assume optimsolution_gui_run/ exists and can be written into right away.
    if (d.removeRecursively()) {
      QDir().mkpath(root);
      ++removedRoots;
    } else {
      ++failedRoots;
    }
  }

  const QString msg = failedRoots > 0
    ? QString("Deleted all output files from %1 optimsolution_gui_run folder(s) (%2 failed).")
        .arg(removedRoots)
        .arg(failedRoots)
    : QString("Deleted all output files (%1 total) from optimsolution_gui_run.")
        .arg(totalFiles);

  appendLog(msg);
  statusBar()->showMessage(msg, 6000);

  // Clear all in-memory batch data so the next run starts clean.
  g_liveBatchCsvPaths.clear();
  g_batchSummarySnapshots.clear();
  batchCells_.clear();
  batchSummaryCellsByPage_.clear();
  if (statsTable_) {
    statsTable_->clearContents();
    statsTable_->setRowCount(0);
    statsTable_->setColumnCount(0);
  }

  // Clear convergence data from all output tabs.
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    auto& run = outputRuns_[i];
    run.convLoaded = false;
    run.convLoadedCsvPath.clear();
    run.convIterX.clear();
    run.convEvalX.clear();
    run.convY.clear();
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(run.convergencePlot)) p->clear();
    if (run.convergenceInfo) run.convergenceInfo->setText("Output files deleted — run again to reload.");
    updateDistributionPlotForTab(i);
  }

  // The pipeline mode's progress (checkmarks, stage 3/4 enabled state) is
  // no longer valid once the output it was based on has been deleted.
  resetPipelineButtonStates();

  // Refresh plots for the active output tab.
  if (outputTabs_) {
    const int idx = outputTabs_->currentIndex();
    tryLoadConvergenceForTab(idx);
    tryLoadLandscapeForTab(idx);
    tryLoadSensitivityForTab(idx);
  }
}



void MainWindow::onExportConvergencePng() {
  QObject* s = sender();
  int idx = -1;
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].exportConvergencePngBtn == s) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    // Fallback: export the currently selected output tab.
    if (outputTabs_) idx = outputTabs_->currentIndex();
  }

  const auto* tab = outputTab(idx);
  if (!tab) return;

  const QString base = sanitizeExportFileStem(tab->title.isEmpty() ? QString("convergence") : tab->title);
  const QString defName = base + "_convergence.png";
  const QString defDir = (!tab->runtimeWorkingDir.isEmpty()) ? tab->runtimeWorkingDir : QDir::currentPath();

  const QString filePath = QFileDialog::getSaveFileName(
      this, "Export convergence plot (PNG, 300 DPI)", QDir(defDir).filePath(defName), "PNG image (*.png)");
  if (filePath.isEmpty()) return;

  if (!exportConvergencePlotPng(idx, filePath)) {
    QMessageBox::warning(this, "Export failed", "Could not export the convergence plot to a PNG file.");
    return;
  }
  statusBar()->showMessage(QString("Exported: %1").arg(filePath), 4000);
}

bool MainWindow::exportConvergencePlotPng(int outputTabIndex, const QString& filePath) {
  auto* tab = outputTab(outputTabIndex);
  if (!tab) return false;
  auto* plot = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot);
  if (!plot) return false;

  const int targetDpi = 300;
  const double baseDpi = std::max(1.0, double(plot->logicalDpiX()));
  const double scale = double(targetDpi) / baseDpi;

  const QSize src = plot->size();
  QSize dst(int(std::ceil(src.width() * scale)), int(std::ceil(src.height() * scale)));
  if (dst.width() < 10 || dst.height() < 10) return false;

  const bool isTransparent = (tab->themeCombo && tab->themeCombo->currentIndex() == 2);
  QImage img(dst, isTransparent ? QImage::Format_ARGB32 : QImage::Format_ARGB32_Premultiplied);
  img.fill(isTransparent ? Qt::transparent : Qt::white);

  // Embed 300 DPI metadata in the PNG.
  const int dpm = int(std::round(double(targetDpi) / 0.0254)); // dots per meter
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}

void MainWindow::onExportDistributionPng() {
  QObject* s = sender();
  int idx = -1;
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].exportDistributionPngBtn == s) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    if (outputTabs_) idx = outputTabs_->currentIndex();
  }

  const auto* tab = outputTab(idx);
  if (!tab) return;

  const QString base = sanitizeExportFileStem(tab->title.isEmpty() ? QString("distribution") : tab->title);
  const QString defName = base + "_distribution.png";
  const QString defDir = (!tab->runtimeWorkingDir.isEmpty()) ? tab->runtimeWorkingDir : QDir::currentPath();

  const QString filePath = QFileDialog::getSaveFileName(
      this, "Export distribution plot (PNG, 300 DPI)", QDir(defDir).filePath(defName), "PNG image (*.png)");
  if (filePath.isEmpty()) return;

  if (!exportDistributionPlotPng(idx, filePath)) {
    QMessageBox::warning(this, "Export failed", "Could not export the distribution plot to a PNG file.");
    return;
  }
  statusBar()->showMessage(QString("Exported: %1").arg(filePath), 4000);
}

bool MainWindow::exportDistributionPlotPng(int outputTabIndex, const QString& filePath) {
  auto* tab = outputTab(outputTabIndex);
  if (!tab) return false;
  auto* plot = dynamic_cast<DistributionPlotWidget*>(tab->distributionPlot);
  if (!plot) return false;

  const int targetDpi = 300;
  const double baseDpi = std::max(1.0, double(plot->logicalDpiX()));
  const double scale = double(targetDpi) / baseDpi;

  const QSize src = plot->size();
  QSize dst(int(std::ceil(src.width() * scale)), int(std::ceil(src.height() * scale)));
  if (dst.width() < 10 || dst.height() < 10) return false;

  QImage img(dst, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::white);

  const int dpm = int(std::round(double(targetDpi) / 0.0254));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}


void MainWindow::onExportSensitivityPng() {
  QObject* s = sender();
  int idx = -1;
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].exportSensitivityPngBtn == s) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    if (outputTabs_) idx = outputTabs_->currentIndex();
  }

  const auto* tab = outputTab(idx);
  if (!tab) return;

  QString defName;
  if (tab->isMethodParamSensitivity) {
    // e.g. "rastrigin_50_F.png"
    QString paramName = "param";
    if (tab->sensitivityParamCombo && !tab->sensitivityParamCombo->currentText().isEmpty()) {
      paramName = tab->sensitivityParamCombo->currentText().trimmed();
    }
    paramName.replace(QRegularExpression("[^A-Za-z0-9]+"), "_");
    while (paramName.startsWith('_')) paramName.remove(0, 1);
    while (paramName.endsWith('_')) paramName.chop(1);
    if (paramName.isEmpty()) paramName = "param";

    QString pName = tab->problemShort.isEmpty() ? "problem" : tab->problemShort;
    pName.replace(QRegularExpression("[^A-Za-z0-9]+"), "_");
    while (pName.startsWith('_')) pName.remove(0, 1);
    while (pName.endsWith('_')) pName.chop(1);

    defName = (tab->problemDim > 0)
                ? QString("%1_%2_%3.png").arg(pName).arg(tab->problemDim).arg(paramName)
                : QString("%1_%2.png").arg(pName, paramName);
  } else {
    const QString base = sanitizeExportFileStem(tab->title.isEmpty() ? QString("sensitivity") : tab->title);
    defName = base + "_sensitivity.png";
  }
  const QString defDir = (!tab->runtimeWorkingDir.isEmpty()) ? tab->runtimeWorkingDir : QDir::currentPath();

  const QString filePath = QFileDialog::getSaveFileName(
      this, "Export sensitivity plot (PNG, 300 DPI)", QDir(defDir).filePath(defName), "PNG image (*.png)");
  if (filePath.isEmpty()) return;

  if (!exportSensitivityPlotPng(idx, filePath)) {
    QMessageBox::warning(this, "Export failed", "Could not export the sensitivity plot to a PNG file.");
    return;
  }
  statusBar()->showMessage(QString("Exported: %1").arg(filePath), 4000);
}

bool MainWindow::exportSensitivityPlotPng(int outputTabIndex, const QString& filePath) {
  auto* tab = outputTab(outputTabIndex);
  if (!tab) return false;
  auto* plot = dynamic_cast<SensitivityBarWidget*>(tab->sensitivityPlot);
  if (!plot) return false;

  const int targetDpi = 300;
  const double baseDpi = std::max(1.0, double(plot->logicalDpiX()));
  const double scale = double(targetDpi) / baseDpi;

  const QSize src = plot->size();
  QSize dst(int(std::ceil(src.width() * scale)), int(std::ceil(src.height() * scale)));
  if (dst.width() < 10 || dst.height() < 10) return false;

  QImage img(dst, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::white);

  const int dpm = int(std::round(double(targetDpi) / 0.0254));
  img.setDotsPerMeterX(dpm);
  img.setDotsPerMeterY(dpm);

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setRenderHint(QPainter::TextAntialiasing, true);
  painter.scale(scale, scale);
  plot->render(&painter);
  painter.end();

  return img.save(filePath, "PNG");
}


// parseConvergenceCsvFile(): moved to MainWindowShared.h

void MainWindow::ensureLandscapeGridForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || !tab->landscapePlot || tab->problemShort.isEmpty()) return;
  if (tab->landscapeGridReady) return;

  auto* lw = dynamic_cast<LandscapeWidget*>(tab->landscapePlot);
  if (!lw) return;

  try {
    auto prob = optimsolution::makeProblem(tab->problemShort.toStdString());
    if (!prob) return;
    prob->init(2);
    const auto& lb = prob->lb();
    const auto& ub = prob->ub();
    if (lb.size() != 2 || ub.size() != 2) return;

    const int nx = 60;
    const int ny = 60;
    QVector<double> z(nx * ny, std::numeric_limits<double>::quiet_NaN());
    optimsolution::Vec x(2, 0.0);
    for (int j = 0; j < ny; ++j) {
      const double t2 = (ny > 1) ? double(j) / double(ny - 1) : 0.0;
      const double x2 = lb[1] + t2 * (ub[1] - lb[1]);
      for (int i = 0; i < nx; ++i) {
        const double t1 = (nx > 1) ? double(i) / double(nx - 1) : 0.0;
        const double x1 = lb[0] + t1 * (ub[0] - lb[0]);
        x[0] = x1;
        x[1] = x2;
        double v = std::numeric_limits<double>::quiet_NaN();
        try { v = prob->evaluate(x); } catch (...) {}
        z[j * nx + i] = v;
      }
    }
    lw->setGrid(nx, ny, z, lb[0], ub[0], lb[1], ub[1]);
    tab->landscapeGridReady = true;
  } catch (...) {
    // Leave the grid unset; the trajectory (if any) stays hidden until a grid exists.
  }
}

void MainWindow::tryLoadLandscapeForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || tab->isPortfolioRun) return;   // Portfolio tabs have no *_landscape.csv
  if (!tab || !tab->landscapePlot) return; // no Landscape tab for this run (not 2D, or sensitivity mode)

  auto* lw = dynamic_cast<LandscapeWidget*>(tab->landscapePlot);
  if (!lw) return;

  auto setInfo = [&](const QString& msg) {
    if (tab->landscapeInfo) tab->landscapeInfo->setText(msg);
  };
  auto disableControls = [&]() {
    if (tab->landscapePlayBtn) { QSignalBlocker b(tab->landscapePlayBtn); tab->landscapePlayBtn->setChecked(false); tab->landscapePlayBtn->setEnabled(false); }
    if (tab->exportLandscapePngBtn) tab->exportLandscapePngBtn->setEnabled(false);
    if (tab->landscapeSlider) { tab->landscapeSlider->setEnabled(false); tab->landscapeSlider->setRange(0, 0); }
  };

  static const QString kConvSuffix = "_convergence.csv";
  if (tab->convLoadedCsvPath.isEmpty() || !tab->convLoadedCsvPath.endsWith(kConvSuffix, Qt::CaseInsensitive)) {
    if (!tab->landscapeLoaded) {
      setInfo("Landscape: waiting for the run's convergence data.");
      disableControls();
    }
    return;
  }

  // Both files share the same CLI-generated prefix (see src/main.cpp).
  const QString landscapePath =
    tab->convLoadedCsvPath.left(tab->convLoadedCsvPath.size() - kConvSuffix.size()) + "_landscape.csv";

  if (tab->landscapeLoaded && tab->landscapeLoadedCsvPath == landscapePath && QFileInfo::exists(landscapePath)) {
    return; // already loaded and unchanged
  }

  if (!QFileInfo::exists(landscapePath)) {
    tab->landscapeLoaded = false;
    tab->landscapeLoadedCsvPath.clear();
    lw->setTrajectory(QVector<LandscapeWidget::Point>());
    setInfo("Landscape: no trajectory data for this run.\n"
            "(Only logged for population-based methods on 2D problems.)");
    disableControls();
    return;
  }

  QFile f(landscapePath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    setInfo("Landscape: could not open " + landscapePath);
    return;
  }

  QVector<LandscapeWidget::Point> traj;
  QTextStream ts(&f);
  bool first = true;
  int firstRun = -1;
  while (!ts.atEnd()) {
    const QString line = ts.readLine();
    if (first) { first = false; continue; } // header row
    if (line.trimmed().isEmpty()) continue;
    const QStringList parts = line.split(',');
    if (parts.size() < 6) continue;
    const int run = parts[0].toInt();
    if (firstRun < 0) firstRun = run;
    if (run != firstRun) continue; // only one illustrative run is logged/plotted

    LandscapeWidget::Point pt;
    pt.iter = parts[1].toInt();
    pt.f    = parts[3].toDouble();
    pt.x1   = parts[4].toDouble();
    pt.x2   = parts[5].toDouble();
    traj.push_back(pt);
  }
  f.close();

  if (traj.isEmpty()) {
    tab->landscapeLoaded = false;
    lw->setTrajectory(QVector<LandscapeWidget::Point>());
    setInfo("Landscape: trajectory file was empty.");
    disableControls();
    return;
  }

  tab->landscapeLoaded = true;
  tab->landscapeLoadedCsvPath = landscapePath;

  ensureLandscapeGridForTab(index);

  lw->setTrajectory(traj);
  lw->setTitleSubtitle(
    QString("Objective landscape — %1 / %2").arg(tab->methodShort, tab->problemShort),
    QString("Best-so-far trajectory over %1 iterations").arg(traj.size()));

  const int lastFrame = traj.size() - 1;
  if (tab->landscapeSlider) {
    QSignalBlocker b(tab->landscapeSlider);
    tab->landscapeSlider->setEnabled(true);
    tab->landscapeSlider->setRange(0, lastFrame);
    tab->landscapeSlider->setValue(lastFrame); // start at the final, converged state
  }
  lw->setFrame(lastFrame);
  if (tab->landscapePlayBtn) tab->landscapePlayBtn->setEnabled(true);
  if (tab->exportLandscapePngBtn) tab->exportLandscapePngBtn->setEnabled(true);

  const auto& last = traj.last();
  setInfo(QString("Landscape: %1 iterations loaded. Frame %2/%1 — iter=%3, best_f=%4")
    .arg(traj.size())
    .arg(traj.size())
    .arg(last.iter)
    .arg(QString::number(last.f, 'g', 8)));
}

void MainWindow::onLandscapeSliderMoved(int value) {
  auto* sliderObj = qobject_cast<QSlider*>(sender());
  OutputRunTab* targetTab = nullptr;
  if (sliderObj) {
    for (int i = 0; i < outputRuns_.size(); ++i) {
      if (outputRuns_[i].landscapeSlider == sliderObj) { targetTab = &outputRuns_[i]; break; }
    }
  }
  if (!targetTab) targetTab = activeOutputTab();
  if (!targetTab) return;

  auto* lw = dynamic_cast<LandscapeWidget*>(targetTab->landscapePlot);
  if (!lw) return;
  lw->setFrame(value);

  if (targetTab->landscapeInfo && lw->frameCount() > 0) {
    const auto pt = lw->currentPoint();
    targetTab->landscapeInfo->setText(
      QString("Landscape: %1 iterations loaded. Frame %2/%1 — iter=%3, best_f=%4")
        .arg(lw->frameCount())
        .arg(value + 1)
        .arg(pt.iter)
        .arg(QString::number(pt.f, 'g', 8)));
  }
}

void MainWindow::onLandscapePlayToggled(bool checked) {
  auto* tab = activeOutputTab();
  if (!tab || !tab->landscapeSlider) return;

  if (!landscapeTimer_) {
    landscapeTimer_ = new QTimer(this);
    connect(landscapeTimer_, &QTimer::timeout, this, &MainWindow::onLandscapeTimerTick);
  }

  if (checked) {
    if (tab->landscapeSlider->maximum() <= 0) {
      QSignalBlocker b(tab->landscapePlayBtn);
      if (tab->landscapePlayBtn) tab->landscapePlayBtn->setChecked(false);
      return;
    }
    if (tab->landscapeSlider->value() >= tab->landscapeSlider->maximum()) {
      tab->landscapeSlider->setValue(0); // restart from the beginning
    }
    landscapePlayingTabIndex_ = activeOutputRunIndex_;
    landscapeTimer_->start(120); // ms per frame
    if (tab->landscapePlayBtn) tab->landscapePlayBtn->setText(QString::fromUtf8("\xE2\x8F\xB8 Pause"));
  } else {
    landscapeTimer_->stop();
    landscapePlayingTabIndex_ = -1;
    if (tab->landscapePlayBtn) tab->landscapePlayBtn->setText(QString::fromUtf8("\xE2\x96\xB6 Play"));
  }
}

void MainWindow::onLandscapeTimerTick() {
  auto* tab = outputTab(landscapePlayingTabIndex_);
  if (!tab || !tab->landscapeSlider) {
    if (landscapeTimer_) landscapeTimer_->stop();
    landscapePlayingTabIndex_ = -1;
    return;
  }
  const int v = tab->landscapeSlider->value();
  if (v >= tab->landscapeSlider->maximum()) {
    landscapeTimer_->stop();
    landscapePlayingTabIndex_ = -1;
    if (tab->landscapePlayBtn) {
      QSignalBlocker b(tab->landscapePlayBtn);
      tab->landscapePlayBtn->setChecked(false);
      tab->landscapePlayBtn->setText(QString::fromUtf8("\xE2\x96\xB6 Play"));
    }
    return;
  }
  tab->landscapeSlider->setValue(v + 1); // triggers onLandscapeSliderMoved via valueChanged
}

void MainWindow::onExportLandscapePng() {
  auto* tab = activeOutputTab();
  if (!tab || !tab->landscapePlot) return;
  const QString out = QFileDialog::getSaveFileName(
    this,
    "Export landscape (PNG, 300 DPI)",
    QDir::homePath() + "/landscape.png",
    "PNG Image (*.png)"
  );
  if (out.isEmpty()) return;
  if (!exportLandscapePng(out)) {
    QMessageBox::warning(this, "Export", "Failed to export the landscape plot.");
  }
}

bool MainWindow::exportLandscapePng(const QString& filePath) {
  auto* tab = activeOutputTab();
  if (!tab) return false;
  auto* plot = dynamic_cast<LandscapeWidget*>(tab->landscapePlot);
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

} // namespace optimsolution_gui
