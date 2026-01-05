#include "MainWindow.h"
#include "PathUtils.h"
#include "ConfigFile.h"
#include "factory.h"
#include "fixed_dims.h"
#include "AnsiStrip.h"
#include "CrashLog.h"

#include <QScopedValueRollback>
#include <QSignalBlocker>

#include <QApplication>
#include <QComboBox>
#include <QSpinBox>
#include <QTextEdit>
#include <QTextDocument>
#include <QColor>
#include <QBrush>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QPalette>
#include <QTableWidget>
#include <QTabWidget>
#include <QTabBar>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QStatusBar>
#include <QFileInfo>
#include <QDir>
#include <QLabel>
#include <QStandardPaths>
#include <QDateTime>
#include <QFile>
#include <QImage>
#include <QFontMetrics>
#include <QTextStream>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <QPainter>
#include <QPainterPath>
#include <QStringConverter>
#include <QDirIterator>
#include <QCheckBox>

namespace optimsolution_gui {

// Forward declarations (file-scope helpers).
static bool parseConvergenceCsvFile(const QString& path,
                                   QVector<double>& outX,
                                   QVector<double>& outY,
                                   QString& outInfo);

// Lightweight plot widget (no QtCharts dependency).
class ConvergencePlotWidget final : public QWidget {
public:
  struct Series {
    QString label;
    QVector<double> x;
    QVector<double> y;
  };

  explicit ConvergencePlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void clear() {
    series_.clear();
    update();
  }

  void setSeries(QVector<double> x, QVector<double> y, const QString& label = QString()) {
    Series s;
    s.label = label;
    s.x = std::move(x);
    s.y = std::move(y);
    series_.clear();
    series_.push_back(std::move(s));
    update();
  }

  void setSeriesList(QVector<Series> series) {
    series_ = std::move(series);
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    p.fillRect(r, palette().base());

    // Compute overall bounds across all series.
    bool any = false;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;

    for (const auto& s : series_) {
      const int n = std::min(s.x.size(), s.y.size());
      if (n < 2) continue;
      if (!any) {
        xmin = s.x[0]; xmax = s.x[0];
        ymin = s.y[0]; ymax = s.y[0];
        any = true;
      }
      for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, s.x[i]);
        xmax = std::max(xmax, s.x[i]);
        ymin = std::min(ymin, s.y[i]);
        ymax = std::max(ymax, s.y[i]);
      }
    }

    if (!any) {
      p.setPen(palette().color(QPalette::Text));
      p.drawText(r.adjusted(10, 10, -10, -10), Qt::AlignCenter,
                 "No convergence data available.\nEnable CSV convergence and run once.");
      return;
    }

    if (xmax <= xmin) xmax = xmin + 1.0;
    if (ymax <= ymin) ymax = ymin + 1.0;

    // Pad Y slightly for readability.
    const double ypad = 0.02 * (ymax - ymin);
    ymin -= ypad;
    ymax += ypad;

    // Margins include tick labels plus axis titles.
    const int left = 72;
    const int right = 14;
    const int top = 12;
    const int bottom = 52;
    QRect plot = r.adjusted(left, top, -right, -bottom);
    if (plot.width() < 50 || plot.height() < 50) return;

    auto mapX = [&](double x) {
      return plot.left() + (x - xmin) * (plot.width() / (xmax - xmin));
    };
    auto mapY = [&](double y) {
      return plot.bottom() - (y - ymin) * (plot.height() / (ymax - ymin));
    };

    // Axes
    const QColor axis = palette().color(QPalette::Text);
    p.setPen(QPen(axis, 1));
    p.drawLine(plot.bottomLeft(), plot.bottomRight());
    p.drawLine(plot.bottomLeft(), plot.topLeft());

    // Ticks (5)
    const int ticks = 5;
    QFont f = font();
    f.setPointSize(std::max(8, f.pointSize() - 1));
    p.setFont(f);

    for (int i = 0; i <= ticks; ++i) {
      const double t = double(i) / ticks;
      const int xx = int(plot.left() + t * plot.width());
      const int yy = int(plot.bottom() - t * plot.height());

      // x ticks
      p.drawLine(QPoint(xx, plot.bottom()), QPoint(xx, plot.bottom() + 4));
      const double xv = xmin + t * (xmax - xmin);
      p.drawText(QRect(xx - 40, plot.bottom() + 6, 80, 18), Qt::AlignHCenter | Qt::AlignTop,
                 QString::number(xv, 'g', 6));

      // y ticks
      p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
      const double yv = ymin + t * (ymax - ymin);
      p.drawText(QRect(0, yy - 9, plot.left() - 8, 18), Qt::AlignRight | Qt::AlignVCenter,
                 QString::number(yv, 'g', 6));
    }

    // Axis titles.
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.setPen(axis);

      // X title.
      p.drawText(QRect(plot.left(), plot.bottom() + 26, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignTop, "Iterations");

      // Y title (rotated 90 degrees).
      p.save();
      p.translate(18, plot.center().y());
      p.rotate(-90.0);
      p.drawText(QRect(-plot.height() / 2, -18, plot.height(), 36),
                 Qt::AlignHCenter | Qt::AlignVCenter, "Best solution");
      p.restore();

      // Restore tick font for subsequent drawing.
      p.setFont(f);
    }

    // Series
    const int k = series_.size();
    int drawn = 0;
    for (int si = 0; si < series_.size(); ++si) {
      const auto& s = series_[si];
      const int n = std::min(s.x.size(), s.y.size());
      if (n < 2) continue;

      QPainterPath path;

      // NOTE:
      // Convergence CSV may concatenate multiple runs into a single series.
      // When a new run starts, the x-axis typically resets (e.g., iterations return to 0/1).
      // Do not draw a line segment across that boundary.
      bool penDown = false;
      double prevX = 0.0;

      for (int i = 0; i < n; ++i) {
        const double xi = s.x[i];
        const double yi = s.y[i];

        if (!std::isfinite(xi) || !std::isfinite(yi)) {
          penDown = false;
          continue;
        }

        if (!penDown) {
          path.moveTo(mapX(xi), mapY(yi));
          penDown = true;
        } else if (xi < prevX) {
          // New run boundary (x reset): start a new sub-path.
          path.moveTo(mapX(xi), mapY(yi));
        } else {
          path.lineTo(mapX(xi), mapY(yi));
        }

        prevX = xi;
      }

      QColor col;
      if (k <= 1) {
        col = palette().color(QPalette::Highlight);
      } else {
        const int hue = (si * 360 / std::max(1, k));
        col = QColor::fromHsv(hue, 200, 220);
      }

      p.setPen(QPen(col, 2));
      p.drawPath(path);
      drawn++;
    }

    // Legend (top-right)
    if (drawn > 1) {
      int lx = plot.right() - 160;
      int ly = plot.top() + 6;
      const int lh = 16;
      const int lw = 150;
      p.setPen(axis);
      for (int si = 0; si < series_.size(); ++si) {
        const auto& s = series_[si];
        if (s.label.isEmpty()) continue;
        QColor col = QColor::fromHsv((si * 360 / std::max(1, k)), 200, 220);
        p.fillRect(QRect(lx, ly + si * lh + 4, 10, 10), col);
        p.drawRect(QRect(lx, ly + si * lh + 4, 10, 10));
        const QRect textR(lx + 14, ly + si * lh, lw - 14, lh);
        const QString elided = QFontMetrics(p.font()).elidedText(s.label, Qt::ElideRight, textR.width());
        p.drawText(textR, Qt::AlignLeft | Qt::AlignVCenter, elided);
      }
    }
  }

private:
  QVector<Series> series_;
};

static void setupTable(QTableWidget* t, const QStringList& headers) {
  t->setColumnCount(headers.size());
  t->setHorizontalHeaderLabels(headers);
  t->horizontalHeader()->setStretchLastSection(true);
  t->verticalHeader()->setVisible(false);
  t->setSelectionBehavior(QAbstractItemView::SelectRows);
  t->setSelectionMode(QAbstractItemView::SingleSelection);
  t->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::EditKeyPressed);
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
  loadFactoryLists();
  CrashLog::append("MainWindow: factory lists loaded.");
  loadSettings();
  CrashLog::append("MainWindow: settings loaded.");
  populateSettingsTables();
  CrashLog::append("MainWindow: settings tables populated.");

  setWindowTitle("optimsolution GUI (v31)");
}

MainWindow::~MainWindow() = default;

void MainWindow::buildUi() {
  auto* central = new QWidget(this);
  auto* rootLay = new QVBoxLayout(central);

  auto* topBox = new QGroupBox("Selection", central);
  auto* form = new QFormLayout(topBox);

  methodBox_ = new QComboBox(topBox);
  methodBox_->setEditable(true);
  methodBox_->setInsertPolicy(QComboBox::NoInsert);
  methodBox_->setSizeAdjustPolicy(QComboBox::AdjustToContents);

  problemBox_ = new QComboBox(topBox);
  problemBox_->setEditable(true);
  problemBox_->setInsertPolicy(QComboBox::NoInsert);
  problemBox_->setSizeAdjustPolicy(QComboBox::AdjustToContents);

  dimSpin_ = new QSpinBox(topBox);
  dimSpin_->setRange(1, 100000);
  dimSpin_->setValue(30);

  form->addRow("Optimization method", methodBox_);
  form->addRow("Problem", problemBox_);
  form->addRow("Dimension", dimSpin_);

  auto* btnRow = new QHBoxLayout();
  refreshBtn_ = new QPushButton("Refresh lists", topBox);
  runBtn_ = new QPushButton("Run", topBox);
  btnRow->addWidget(refreshBtn_);
  btnRow->addStretch(1);
  btnRow->addWidget(runBtn_);
  form->addRow(btnRow);

  rootLay->addWidget(topBox);

  // Settings controls (no file/path displayed)
  auto* cfgBox = new QGroupBox("Settings", central);
  auto* cfgLay = new QVBoxLayout(cfgBox);
  auto* cfgBtnRow = new QHBoxLayout();
  selectCfgBtn_ = new QPushButton("Select settings…", cfgBox);
  reloadCfgBtn_ = new QPushButton("Reload", cfgBox);
  saveCfgBtn_   = new QPushButton("Save", cfgBox);
  cfgBtnRow->addWidget(selectCfgBtn_);
  cfgBtnRow->addStretch(1);
  cfgBtnRow->addWidget(reloadCfgBtn_);
  cfgBtnRow->addWidget(saveCfgBtn_);
  cfgLay->addLayout(cfgBtnRow);

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
  removeGlobalParamBtn_ = new QPushButton("Remove parameter", globalTab);
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
  removeStopParamBtn_ = new QPushButton("Remove parameter", stopTab);
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
  removeInitParamBtn_ = new QPushButton("Remove parameter", initTab);
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
  removeMethodParamBtn_ = new QPushButton("Remove parameter", methodTab);
  methodBtnRow->addWidget(addMethodParamBtn_);
  methodBtnRow->addWidget(removeMethodParamBtn_);
  methodBtnRow->addStretch(1);
  methodLay->addLayout(methodBtnRow);
  methodLay->addWidget(methodTable_);

  /* Sensitivity tab wrapper: allows adding/removing arbitrary sensitivity parameters. */
  auto* sensitivityTab = new QWidget(tabs_);
  auto* sensitivityLay = new QVBoxLayout(sensitivityTab);
  sensitivityLay->setContentsMargins(0, 0, 0, 0);
  auto* sensitivityBtnRow = new QHBoxLayout();
  addSensitivityParamBtn_ = new QPushButton("Add parameter", sensitivityTab);
  removeSensitivityParamBtn_ = new QPushButton("Remove parameter", sensitivityTab);
  sensitivityBtnRow->addWidget(addSensitivityParamBtn_);
  sensitivityBtnRow->addWidget(removeSensitivityParamBtn_);
  sensitivityBtnRow->addStretch(1);
  sensitivityLay->addLayout(sensitivityBtnRow);
  sensitivityLay->addWidget(sensitivityTable_);

  setupTable(runTable_, {"Key", "Value", "Source"});
  setupTable(stopTable_, {"Key", "Value"});
  setupTable(initTable_, {"Key", "Value"});
  setupTable(methodTable_, {"Key", "Value"});
  setupTable(sensitivityTable_, {"Key", "Value"});

  tabs_->addTab(globalTab, "Global");
  tabs_->addTab(stopTab, "Termination rule");
  tabs_->addTab(initTab, "Initialization");
  tabs_->addTab(methodTab, "Optimization method");
  tabs_->addTab(sensitivityTab, "Sensitivity");

  cfgLay->addWidget(tabs_);
  rootLay->addWidget(cfgBox, 1);

  auto* outBox = new QGroupBox("Output", central);
  auto* outLay = new QVBoxLayout(outBox);
  // Global output controls (apply to next run, and to the currently selected output tab when reloading).
  auto* outTop = new QHBoxLayout();
  forceConvergenceCsvChk_ = new QCheckBox("Force CSV convergence for plotting (runtime only)", outBox);
  forceConvergenceCsvChk_->setChecked(true);
  reloadConvergenceBtn_ = new QPushButton("Reload convergence", outBox);
  outTop->addWidget(forceConvergenceCsvChk_);
  outTop->addStretch(1);
  outTop->addWidget(reloadConvergenceBtn_);
  outLay->addLayout(outTop);

  outputTabs_ = new QTabWidget(outBox);
  outputTabs_->setDocumentMode(true);

  outputTabs_->setTabsClosable(true);
  connect(outputTabs_, &QTabWidget::tabCloseRequested, this, &MainWindow::onOutputTabCloseRequested);

  // Create an initial System tab, so configuration/diagnostic messages always have a visible target.
  createOutputRunTab("System", QString(), 0);
  outputTabs_->setCurrentIndex(0);
  activeOutputRunIndex_ = 0;

  if (auto* bar = outputTabs_->tabBar()) {
    bar->setTabButton(0, QTabBar::LeftSide, nullptr);
    bar->setTabButton(0, QTabBar::RightSide, nullptr);
  }

  outLay->addWidget(outputTabs_);
  rootLay->addWidget(outBox, 2);

  setCentralWidget(central);

  statusBar()->showMessage("Ready");

  // connections
  connect(refreshBtn_, &QPushButton::clicked, this, &MainWindow::onRefreshFactory);
  connect(problemBox_, &QComboBox::currentTextChanged, this, &MainWindow::onProblemChanged);
  connect(methodBox_, &QComboBox::currentTextChanged, this, &MainWindow::onMethodChanged);
  connect(runBtn_, &QPushButton::clicked, this, &MainWindow::onRunClicked);

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
  connect(addSensitivityParamBtn_, &QPushButton::clicked, this, &MainWindow::onAddSensitivityParam);
  connect(removeSensitivityParamBtn_, &QPushButton::clicked, this, &MainWindow::onRemoveSensitivityParam);

  connect(reloadConvergenceBtn_, &QPushButton::clicked, this, &MainWindow::onReloadConvergence);

  // item change hooks (editable settings)
  connect(runTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(stopTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(initTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(methodTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
  connect(sensitivityTable_, &QTableWidget::cellChanged, this, &MainWindow::onSettingItemChanged);
}

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
          display = QString::fromStdString(full) + " (" + shortName + ")";
        }
      }
    } catch (...) {
      // If a method cannot be instantiated here, keep the short name.
    }
    methodBox_->addItem(display, shortName);
  }

  // Problems: keep short name as display and itemData.
  for (const QString& p : problems) {
    problemBox_->addItem(p, p);
  }

  methodBox_->blockSignals(false);
  problemBox_->blockSignals(false);

  // try to keep current selections
  if (methodBox_->count() > 0 && methodBox_->currentText().isEmpty()) methodBox_->setCurrentIndex(0);
  if (problemBox_->count() > 0 && problemBox_->currentText().isEmpty()) problemBox_->setCurrentIndex(0);

  updateDimUiForProblem(problemBox_->currentText());

  appendLog("Factory lists loaded.");
}

void MainWindow::onProblemChanged(const QString& problem) {
  updateDimUiForProblem(problem);
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
  runTable_->setItem(row, 2, srcItem);

  setTablesEditable(true);
  runTable_->setCurrentCell(row, 0);
  runTable_->editItem(keyItem);
}

void MainWindow::onRemoveGlobalParam() {
  if (!runTable_) return;
  const int row = runTable_->currentRow();
  if (row < 0) return;

  QTableWidgetItem* keyItem = runTable_->item(row, 0);
  QTableWidgetItem* srcItem = runTable_->item(row, 2);
  const QString key = keyItem ? keyItem->text().trimmed() : QString();
  const QString src = srcItem ? srcItem->text().trimmed() : QString("global");

  if (cfg_ && cfg_->isLoaded() && !key.isEmpty()) {
    if (src.compare("method", Qt::CaseInsensitive) == 0) {
      const QString method = currentMethodShort();
      if (!method.isEmpty()) cfg_->removeKey(method, key);
    } else {
      cfg_->removeKey("global", key);
    }
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

void MainWindow::updateDimUiForProblem(const QString& problem) {
  currentFixedDim_ = fixedDimForProblem(problem);

  if (currentFixedDim_ > 0) {
    dimSpin_->setEnabled(false);
    dimSpin_->setValue(currentFixedDim_);
  } else {
    dimSpin_->setEnabled(true);
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

int MainWindow::createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim) {
  if (!outputTabs_) return -1;

  const QString m = methodShort.trimmed();
  const QString p = problemShort.trimmed();
  const QString baseTitle = p.isEmpty() ? m : (m + "+" + p);
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

  // Inner tabs keep the UI familiar (Log + Convergence) while the outer tabs are per-run.
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
  inner->addTab(logTab, "Log");

  // Convergence
  auto* convTab = new QWidget(inner);
  auto* convLay = new QVBoxLayout(convTab);
  convLay->setContentsMargins(8, 8, 8, 8);

  auto* convTop = new QHBoxLayout();
  auto* overlayMethodChk = new QCheckBox("Overlay all problems for this method", convTab);
  overlayMethodChk->setChecked(false);
  connect(overlayMethodChk, &QCheckBox::toggled, this, &MainWindow::onOverlayMethodToggled);
  convTop->addWidget(overlayMethodChk);

  auto* exportPngBtn = new QPushButton("Export PNG…", convTab);
  connect(exportPngBtn, &QPushButton::clicked, this, &MainWindow::onExportConvergencePng);
  convTop->addWidget(exportPngBtn);

  convTop->addStretch(1);
  convLay->addLayout(convTop);

  auto* convInfo = new QLabel("No convergence data loaded.", convTab);
  convInfo->setWordWrap(true);
  convLay->addWidget(convInfo);
  auto* convPlot = new ConvergencePlotWidget(convTab);
  convLay->addWidget(convPlot, 1);
  inner->addTab(convTab, "Convergence");

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
  run.page = page;
  run.log = log;
  run.convergenceInfo = convInfo;
  run.convergencePlot = convPlot;
  run.overlayMethodChk = overlayMethodChk;
  run.exportConvergencePngBtn = exportPngBtn;
  run.convLoaded = false;
  outputRuns_.push_back(std::move(run));

  return tabIndex;
}

MainWindow::OutputRunTab* MainWindow::outputTab(int index) {
  if (index < 0 || index >= int(outputRuns_.size())) return nullptr;
  return &outputRuns_[index];
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

void MainWindow::tryLoadConvergenceForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab || !tab->convergenceInfo) return;

  // Reset stored series for this tab.
  tab->convLoaded = false;
  tab->convX.clear();
  tab->convY.clear();

  if (tab->runtimeWorkingDir.isEmpty()) {
    tab->convergenceInfo->setText("No previous run information available.");
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
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
    tab->convergenceInfo->setText(
      "No CSV files were found near the run folder. "
      "Ensure [global] csv_enable=1 and csv_convergence=1 (or keep 'Force CSV convergence' enabled)."
    );
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    return;
  }

  std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.mtime > b.mtime;
  });

  QVector<double> x;
  QVector<double> y;
  QString info;
  QString loadedPath;

  // Try the best-scoring recent candidates first, but keep a hard cap on attempts.
  const int maxAttempts = std::min(40, int(candidates.size()));
  for (int i = 0; i < maxAttempts; ++i) {
    x.clear(); y.clear(); info.clear();
    if (parseConvergenceCsvFile(candidates[i].path, x, y, info)) {
      loadedPath = candidates[i].path;
      break;
    }
  }

  if (loadedPath.isEmpty()) {
    tab->convergenceInfo->setText(
      "CSV files were found, but none looked like a convergence trace. "
      "Enable convergence CSV output in the CLI, or adjust its filename/pattern."
    );
    if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    return;
  }

  tab->convX = x;
  tab->convY = y;
  tab->convLoaded = true;

  tab->convergenceInfo->setText(QString("%1\n%2").arg(info).arg(loadedPath));
  updateConvergencePlotForTab(index);

  // If other tabs for the same method are in overlay mode, refresh them as well.
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (!outputRuns_[i].overlayMethodChk) continue;
    if (!outputRuns_[i].overlayMethodChk->isChecked()) continue;
    if (outputRuns_[i].methodShort.compare(tab->methodShort, Qt::CaseInsensitive) != 0) continue;
    updateConvergencePlotForTab(i);
  }
}

void MainWindow::onRunClicked() {
  // Toggle behavior: Run when idle, Stop when a process is running.
  if (proc_ && proc_->state() != QProcess::NotRunning) {
    requestStop();
    return;
  }

  QString method = currentMethodShort();
  QString problem = problemBox_->currentText().trimmed();
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

  if (cfg_ && cfg_->isLoaded()) {
    // Clear convergence plot at the start of each run (active output tab).
    if (auto* tab = activeOutputTab()) {
      if (tab->convergenceInfo) tab->convergenceInfo->setText("No convergence data loaded.");
      if (auto* p = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot)) p->clear();
    }

    // Optional runtime-only override: force CSV convergence so the GUI can plot best-per-iteration.
    const bool forceCsv = (forceConvergenceCsvChk_ && forceConvergenceCsvChk_->isChecked());
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

    if (cliSupportsConfigArg_ == 1) {
      runtimeCfgPath = QDir(baseWd).filePath("optimsolution_gui_merged.cfg");
      applyForceCsv();
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
    tab->runtimeWorkingDir = runtimeWd;
    tab->runtimeCfgPath = runtimeCfgPath;
  }

  // Persist per-run runtime paths (used for plotting CSV convergence for that specific tab).
  if (auto* tab = activeOutputTab()) {
    tab->runtimeWorkingDir = runtimeWd;
    tab->runtimeCfgPath = runtimeCfgPath;
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

  CrashLog::append(QString("Process: start program=%1 args=%2").arg(cli).arg(finalArgs.join(" ")));
  proc_->start();
  if (!proc_->waitForStarted(3000)) {
    CrashLog::append("Process: failed to start (waitForStarted timeout).");
    QMessageBox::critical(this, "Failed to start", "Could not start the CLI process.");
    return;
  }

  // Running state
  runBtn_->setText("Stop");
  methodBox_->setEnabled(false);
  problemBox_->setEnabled(false);
}

void MainWindow::onProcessReadyReadStdout() {
  QByteArray b = proc_->readAllStandardOutput();
  QString chunk = QString::fromUtf8(b);
  QString cleaned = AnsiStrip::stripStreaming(chunk, ansiCarryStdout_);
  processTextBuffer_ += cleaned;
  if (!cleaned.isEmpty()) appendLog(cleaned);
}

void MainWindow::onProcessReadyReadStderr() {
  QByteArray b = proc_->readAllStandardError();
  QString chunk = QString::fromUtf8(b);
  QString cleaned = AnsiStrip::stripStreaming(chunk, ansiCarryStderr_);
  processTextBuffer_ += cleaned;
  if (!cleaned.isEmpty()) appendLog(cleaned);
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus status) {
  Q_UNUSED(status);

  // flush any carry
  if (!ansiCarryStdout_.isEmpty()) appendLog(AnsiStrip::strip(ansiCarryStdout_));
  if (!ansiCarryStderr_.isEmpty()) appendLog(AnsiStrip::strip(ansiCarryStderr_));
  ansiCarryStdout_.clear();
  ansiCarryStderr_.clear();

  appendLog(QString("Process finished with exit code %1").arg(exitCode));

  // Restore idle state
  runBtn_->setText("Run");
  methodBox_->setEnabled(true);
  problemBox_->setEnabled(true);
  updateDimUiForProblem(problemBox_->currentText().trimmed());

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

  // Try loading convergence CSV (best-per-iteration) if available.
  if (exitCode == 0) {
    tryLoadConvergenceForTab(activeOutputRunIndex_);
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
      const QRegularExpression rePct(QStringLiteral(R"(([-+]?[0-9]+(?:\.[0-9]+)?)\s*%))"));
      const auto m = rePct.match(lv.second);
      if (m.hasMatch()) {
        const QString pctToken = m.captured(1);
        const double pct = m.captured(2).toDouble();
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

static QString lastSummaryField(const QString& text, const QString& prefix) {
  const QStringList lines = text.split(QRegularExpression(R"(\r\n|\n|\r)"));
  for (int i = lines.size() - 1; i >= 0; --i) {
    const QString ln = lines[i].trimmed();
    if (ln.startsWith(prefix, Qt::CaseInsensitive)) {
      return ln.mid(prefix.size()).trimmed().simplified();
    }
  }
  return QString();
}

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
    appendLog("Settings file not found. Use 'Select settings…' to choose a cfg.");
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
  auto apply = [&](QTableWidget* t, bool hasSource, bool keyEditable) {
    for (int r = 0; r < t->rowCount(); ++r) {
      for (int c = 0; c < t->columnCount(); ++c) {
        QTableWidgetItem* item = t->item(r, c);
        if (!item) continue;
        if (c == 0) {
          item->setFlags((keyEditable && editable) ? flagsEdit : flags);
        } else if (hasSource && c == 2) {
          item->setFlags(flags);
        } else {
          item->setFlags(editable ? flagsEdit : flags);
        }
      }
    }
  };
  apply(runTable_, true, true);
  apply(stopTable_, false, true);
  apply(initTable_, false, true);
  apply(methodTable_, false, true);
  apply(sensitivityTable_, false, true);

  saveCfgBtn_->setEnabled(editable);
  reloadCfgBtn_->setEnabled(true);
}

static void fillKVTable(QTableWidget* t, const QMap<QString, QString>& kv, bool includeSource,
                        const std::function<QString(const QString&)>& sourceFn = {}) {
  t->blockSignals(true);
  t->setRowCount(0);
  int row = 0;

  QStringList keys = kv.keys();
  keys.sort(Qt::CaseInsensitive);

  t->setRowCount(keys.size());
  for (const auto& k : keys) {
    auto* keyItem = new QTableWidgetItem(k);
    keyItem->setData(Qt::UserRole, k);
    auto* valItem = new QTableWidgetItem(kv.value(k));
    t->setItem(row, 0, keyItem);
    t->setItem(row, 1, valItem);

    if (includeSource) {
      QString src = sourceFn ? sourceFn(k) : QString();
      auto* srcItem = new QTableWidgetItem(src);
      t->setItem(row, 2, srcItem);
    }
    ++row;
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

  // Effective Global = [global] + [method]
  CrashLog::append("populateSettingsTables: computing effectiveGlobalMap...");
  QMap<QString, QString> eff = cfg_->effectiveRunMap(method);
  CrashLog::append(QString("populateSettingsTables: effectiveRunMap keys=%1").arg(eff.size()));
  CrashLog::append("populateSettingsTables: fill Global table...");
  fillKVTable(runTable_, eff, true, [&](const QString& k){ return cfg_->sourceOfEffectiveKey(method, k); });
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

  // Sensitivity
  CrashLog::append("populateSettingsTables: reading [sensitivity]...");
  auto sMap = cfg_->sectionMap("sensitivity");
  CrashLog::append(QString("populateSettingsTables: sensitivity keys=%1").arg(sMap.size()));
  CrashLog::append("populateSettingsTables: fill Sensitivity table...");
  fillKVTable(sensitivityTable_, sMap, false);
  CrashLog::append("populateSettingsTables: Sensitivity table filled.");

  CrashLog::append("populateSettingsTables: setTablesEditable...");
  setTablesEditable(true);
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
}

void MainWindow::onSettingItemChanged(int row, int column) {
  if (internalUpdate_) return;

  // Identify which table emitted the change
  QTableWidget* senderTable = qobject_cast<QTableWidget*>(sender());
  if (!senderTable) return;
  if (!cfg_ || !cfg_->isLoaded()) return;

  // Global tab is an "effective" view (global + method overrides). Allow adding/removing/renaming
  // arbitrary keys by editing the Key/Value columns; the Source column determines which section
  // is updated (global vs current method section).
  if (senderTable == runTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    QTableWidgetItem* srcItem = senderTable->item(row, 2);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    const QString value = valItem->text();
    if (newKey.isEmpty()) return;

    QString targetSection = "global";
    const QString src = srcItem ? srcItem->text().trimmed() : QString("global");
    if (src.compare("method", Qt::CaseInsensitive) == 0) {
      const QString method = currentMethodShort();
      if (!method.isEmpty()) targetSection = method;
    }

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
    return;
  }

  // Stop parameters: names/count may vary, so allow editing both Key and Value columns.
  if (senderTable == stopTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    const QString value = valItem->text();
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
    const QString value = valItem->text();
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
    const QString value = valItem->text();
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

  // Sensitivity parameters: names/count vary, so allow editing both Key and Value columns.
  if (senderTable == sensitivityTable_) {
    if (column != 0 && column != 1) return;

    QTableWidgetItem* keyItem = senderTable->item(row, 0);
    QTableWidgetItem* valItem = senderTable->item(row, 1);
    if (!keyItem || !valItem) return;

    const QString newKey = keyItem->text().trimmed();
    const QString value = valItem->text();
    if (newKey.isEmpty()) return;

    const QString oldKey = keyItem->data(Qt::UserRole).toString().trimmed();

    if (column == 0) {
      if (!oldKey.isEmpty() && oldKey.compare(newKey, Qt::CaseInsensitive) != 0) {
        cfg_->removeKey("sensitivity", oldKey);
      }
      cfg_->setValue("sensitivity", newKey, value);
      keyItem->setData(Qt::UserRole, newKey);
    } else {
      cfg_->setValue("sensitivity", newKey, value);
      if (oldKey.isEmpty()) keyItem->setData(Qt::UserRole, newKey);
    }

    populateSettingsTables();
    return;
  }
}


void MainWindow::updateConvergencePlotForTab(int index) {
  auto* tab = outputTab(index);
  if (!tab) return;

  auto* plot = dynamic_cast<ConvergencePlotWidget*>(tab->convergencePlot);
  if (!plot) return;

  if (!tab->convLoaded) {
    plot->clear();
    return;
  }

  const bool overlay = (tab->overlayMethodChk && tab->overlayMethodChk->isChecked());
  if (!overlay) {
    QString label = tab->problemShort;
    if (label.isEmpty()) label = tab->title;
    if (tab->problemDim > 0) label += QString(" (D=%1)").arg(tab->problemDim);
    plot->setSeries(tab->convX, tab->convY, label);
    return;
  }

  // Overlay: plot all problems that have convergence data for this method.
  QVector<ConvergencePlotWidget::Series> series;
  for (const auto& r : outputRuns_) {
    if (!r.convLoaded) continue;
    if (r.methodShort.compare(tab->methodShort, Qt::CaseInsensitive) != 0) continue;

    ConvergencePlotWidget::Series s;
    s.label = r.problemShort.isEmpty() ? r.title : r.problemShort;
    if (r.problemDim > 0) s.label += QString(" (D=%1)").arg(r.problemDim);
    s.x = r.convX;
    s.y = r.convY;
    series.push_back(std::move(s));
  }

  if (series.isEmpty()) {
    QString label = tab->problemShort;
    if (label.isEmpty()) label = tab->title;
    if (tab->problemDim > 0) label += QString(" (D=%1)").arg(tab->problemDim);
    plot->setSeries(tab->convX, tab->convY, label);
    return;
  }

  plot->setSeriesList(std::move(series));
}

void MainWindow::onOverlayMethodToggled(bool) {
  // Identify which output tab triggered this toggle, then refresh that tab's plot.
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].overlayMethodChk == s) {
      updateConvergencePlotForTab(i);
      break;
    }
  }
}

void MainWindow::onOutputTabCloseRequested(int index) {
  // Keep the System tab (index 0) always present.
  if (index <= 0) return;

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

  const QString base = tab->title.isEmpty() ? QString("convergence") : tab->title;
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

  QImage img(dst, QImage::Format_ARGB32_Premultiplied);
  img.fill(Qt::white);

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

static bool parseConvergenceCsvFile(const QString& path, QVector<double>& outX, QVector<double>& outY, QString& outInfo) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    outInfo = QString("Could not open: %1").arg(path);
    return false;
  }

  QTextStream ts(&f);
	#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
	  ts.setEncoding(QStringConverter::Utf8);
	#else
	  ts.setCodec("UTF-8");
	#endif

  auto detectDelim = [](const QString& line) -> QChar {
    const int cComma = line.count(',');
    const int cSemi  = line.count(';');
    const int cTab   = line.count('\t');
    if (cTab  >= cComma && cTab  >= cSemi) return '\t';
    if (cSemi >= cComma) return ';';
    return ',';
  };

  auto splitLine = [](const QString& line, QChar delim) {
    return line.split(delim, Qt::KeepEmptyParts);
  };

  QString firstNonEmpty;
  while (!ts.atEnd()) {
    const QString ln = ts.readLine();
    const QString t = ln.trimmed();
    if (t.isEmpty()) continue;
    if (t.startsWith('#') || t.startsWith(';')) continue;
    firstNonEmpty = ln;
    break;
  }
  if (firstNonEmpty.isEmpty()) {
    outInfo = "CSV is empty.";
    return false;
  }

  const QChar delim = detectDelim(firstNonEmpty);
  QStringList fields = splitLine(firstNonEmpty, delim);
  auto looksNumeric = [](const QString& s) {
    bool ok = false;
    s.trimmed().toDouble(&ok);
    return ok;
  };

  bool hasHeader = false;
  for (const QString& v : fields) {
    const QString t = v.trimmed();
    if (t.isEmpty()) continue;
    if (!looksNumeric(t)) { hasHeader = true; break; }
  }

  int xCol = 0;
  int yCol = 1;
  QStringList header;
  if (hasHeader) {
    header = fields;
    // Read next line as first data
  }

  auto findCol = [&](const QStringList& h, const QStringList& needles) -> int {
    for (int i = 0; i < h.size(); ++i) {
      const QString key = h[i].trimmed().toLower();
      for (const QString& n : needles) {
        if (key.contains(n)) return i;
      }
    }
    return -1;
  };

  if (hasHeader) {
    const int iterCol = findCol(header, {"iter", "iteration", "gen", "step"});
    const int evalCol = findCol(header, {"eval", "fe", "nfe", "calls"});
    const int bestCol = findCol(header, {"best", "fbest", "bestf", "best_f", "gbest"});
    if (iterCol >= 0) xCol = iterCol;
    else if (evalCol >= 0) xCol = evalCol;
    else xCol = 0;

    if (bestCol >= 0) yCol = bestCol;
    else {
      // fallback: first numeric column that is not x
      yCol = -1;
      for (int i = 0; i < header.size(); ++i) {
        if (i == xCol) continue;
        yCol = i;
        break;
      }
	      if (yCol < 0) yCol = std::min(1, int(header.size()) - 1);
    }
  } else {
    // No header: assume x,y in first two columns.
    xCol = 0;
	    yCol = std::min(1, int(fields.size()) - 1);
  }

  // Parse helper
  auto parseDataLine = [&](const QString& ln) {
    const QString t = ln.trimmed();
    if (t.isEmpty()) return;
    if (t.startsWith('#') || t.startsWith(';')) return;
    const QStringList cols = splitLine(ln, delim);
    if (cols.size() <= std::max(xCol, yCol)) return;
    bool okX=false, okY=false;
    const double x = cols[xCol].trimmed().toDouble(&okX);
    const double y = cols[yCol].trimmed().toDouble(&okY);
    if (!okX || !okY) return;
    outX.push_back(x);
    outY.push_back(y);
  };

  // If first line was data, parse it.
  if (!hasHeader) parseDataLine(firstNonEmpty);

  // Parse remaining lines
  int lineCount = 0;
  while (!ts.atEnd()) {
    const QString ln = ts.readLine();
    parseDataLine(ln);
    if (++lineCount > 2000000) break; // hard safety cap
  }

  const int n = std::min(outX.size(), outY.size());
  if (n < 2) {
    outX.clear();
    outY.clear();
    outInfo = QString("No usable convergence data in: %1").arg(QFileInfo(path).fileName());
    return false;
  }

  // Downsample for rendering if extremely large.
  const int maxPts = 6000;
  if (n > maxPts) {
    QVector<double> dx;
    QVector<double> dy;
    dx.reserve(maxPts);
    dy.reserve(maxPts);
    const int stride = int(std::ceil(double(n) / maxPts));
    for (int i = 0; i < n; i += stride) {
      dx.push_back(outX[i]);
      dy.push_back(outY[i]);
    }
    outX = std::move(dx);
    outY = std::move(dy);
  }

  outInfo = QString("Loaded %1 points from %2").arg(outX.size()).arg(QFileInfo(path).fileName());
  return true;
}

} // namespace optimsolution_gui
