#include "MainWindow.h"
#include "PathUtils.h"
#include "ConfigFile.h"
#include "CodeGenDialog.h"
#include "factory.h"
#include "fixed_dims.h"
#include "AnsiStrip.h"
#include "CrashLog.h"
#include "BusySpinner.h"

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
#include <QToolButton>
#include <QInputDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QListWidget>
#include <QTemporaryFile>
#include <QSaveFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QProcess>
#include <QStatusBar>
#include <QFileInfo>
#include <QDir>
#include <QLabel>
#include <QStandardPaths>
#include <QDateTime>
#include <QRegularExpression>
#include <QProgressBar>
#include <QTimer>
#include <QThread>
#include <QFile>
#include <QImage>
#include <QFontMetrics>
#include <QTextStream>
#include <QSet>
#include <QSplitter>
#include <QPlainTextEdit>
#include <QRandomGenerator>
#include <QLocale>
#include <algorithm>
#include <limits>
#include <cmath>
#include <QPainter>
#include <QPainterPath>
#include <QStringConverter>
#include <QDirIterator>
#include <QtConcurrent/QtConcurrent>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QStyledItemDelegate>
#include <QRadioButton>
#include <QButtonGroup>
#include <QSizePolicy>
#include <QStyle>
#include <QSize>
#include <QHash>
#include <QPointer>

// File-scope helper for non-blocking batch log writing.
class BatchLogWriter : public QObject
{
public:
    explicit BatchLogWriter(const QString& filePath, QObject* parent=nullptr)
        : QObject(parent), file_(filePath)
    {
        const bool ok = file_.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);

        Q_UNUSED(ok);
}

    void appendUtf8(const QByteArray& bytes)
    {
        if (!file_.isOpen()) return;
        file_.write(bytes);
    }

    void close()
    {
        if (!file_.isOpen()) return;
        file_.flush();
        file_.close();
    }

private:
    QFile file_;
};

namespace optimsolution_gui {

struct BatchSummaryUiBundle {
  QTabWidget* statsTabs = nullptr;
  QPushButton* exportStatsBtn = nullptr;
  QTableWidget* statsTable = nullptr;
  QLabel* statsNoteLbl = nullptr;
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
  QComboBox* wilcoxonPairsCombo = nullptr;
  QComboBox* statsAlphaCombo = nullptr;
  QPushButton* exportWilcoxonPlotBtn = nullptr;
  QWidget* wilcoxonPlot = nullptr;
  QLabel* wilcoxonSummaryLbl = nullptr;
  QPushButton* exportRankPlotBtn = nullptr;
  QWidget* rankPlot = nullptr;
  QLabel* statsSummaryLbl = nullptr;
  QTableWidget* statsPairwiseTable = nullptr;
};

struct BatchSummarySnapshot {
  QStringList methods;
  QStringList problems;
  QMap<QString, QMap<QString, QString>> csvPaths;
  QMap<QString, int> cachedProblemDims;
  QMap<QString, QStringList> problemDimOverrides;
  int metricComboIndex = -1;
  QString aggText;
  bool showMean = true;
  bool showRate = true;
  bool showSd = true;
  bool showTime = true;
};

static QHash<QWidget*, BatchSummaryUiBundle> g_batchSummaryUi;
static QHash<QWidget*, BatchSummarySnapshot> g_batchSummarySnapshots;
static QWidget* g_activeBatchSummaryPage = nullptr;
static QMap<QString, QMap<QString, QString>> g_liveBatchCsvPaths;

// Forward declaration — defined later in this file.
static QString lastSummaryField(const QString& text, const QString& prefix);

static QString sanitizeExportFileStem(const QString& text) {
  QString s = text.trimmed();
  if (s.isEmpty()) s = "batch";
  s.replace(QRegularExpression(QStringLiteral(R"([\/:*?"<>|]+)")), "_");
  s.replace(QRegularExpression(QStringLiteral(R"(\s+)")), "_");
  s.replace(QRegularExpression(QStringLiteral("_+")), "_");
  s.remove(QRegularExpression(QStringLiteral("^_+|_+$")));
  if (s.isEmpty()) s = "batch";
  return s;
}

static QString sanitizeSheetName(QString s) {
  s = s.trimmed();
  if (s.isEmpty()) s = "sheet";
  s.replace(QRegularExpression(QStringLiteral(R"([:\/?*\[\]])")), " ");
  s.replace(QRegularExpression(QStringLiteral(R"(\s+)")), " ");
  s = s.trimmed();
  if (s.isEmpty()) s = "sheet";
  if (s.size() > 31) s = s.left(31);
  return s;
}

static QTableWidgetItem* makeReadOnlyItem(const QString& text) {
  auto* it = new QTableWidgetItem(text);
  it->setFlags(it->flags() & ~Qt::ItemIsEditable);
  return it;
}

static void setupDerivedAnalysisTable(QTableWidget* t) {
  if (!t) return;
  t->setEditTriggers(QAbstractItemView::NoEditTriggers);
  t->setSelectionBehavior(QAbstractItemView::SelectItems);
  t->setSelectionMode(QAbstractItemView::ExtendedSelection);
  t->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  t->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  if (t->horizontalHeader()) t->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  if (t->verticalHeader()) t->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
}

static void applyTopRankHighlight(QTableWidgetItem* it, int rank) {
  if (!it || rank <= 0) return;

  it->setBackground(QBrush(Qt::NoBrush));

  QFont f = it->font();
  if (rank >= 1 && rank <= 3) {
    f.setBold(true);
    f.setPointSize(std::max(10, f.pointSize()));
    it->setFont(f);
  }

  if (rank == 1) {
    it->setForeground(QBrush(QColor(0, 200, 0)));     // strong green
  } else if (rank == 2) {
    it->setForeground(QBrush(QColor(0, 120, 255)));   // strong blue
  } else if (rank == 3) {
    it->setForeground(QBrush(QColor(220, 0, 0)));     // strong red
  }
}

static QVector<int> competitionRanksForMinimization(const QVector<double>& vals) {
  const int n = vals.size();
  QVector<int> ranks(n, 0);
  QVector<int> idx;
  idx.reserve(n);
  for (int i = 0; i < n; ++i) if (std::isfinite(vals[i])) idx.push_back(i);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (vals[a] == vals[b]) return a < b;
    return vals[a] < vals[b];
  });
  int pos = 0;
  while (pos < idx.size()) {
    const int start = pos;
    const double v0 = vals[idx[start]];
    int end = start;
    while (end + 1 < idx.size()) {
      const double v1 = vals[idx[end + 1]];
      const double eps = std::max(1e-12, 1e-9 * std::max(std::abs(v0), 1.0));
      if (std::abs(v1 - v0) > eps) break;
      ++end;
    }
    const int rank = start + 1;
    for (int i = start; i <= end; ++i) ranks[idx[i]] = rank;
    pos = end + 1;
  }
  return ranks;
}

// Icon tint helper used only for a few explicit buttons (per user-requested icon patches).
static QIcon makeTintedIcon(const QIcon& base, const QColor& color, const QSize& size, double alphaMul = 1.0) {
  QPixmap pm = base.pixmap(size, QIcon::Normal, QIcon::Off);
  if (pm.isNull()) return base;

  QImage img = pm.toImage().convertToFormat(QImage::Format_ARGB32);
  const int w = img.width();
  const int h = img.height();
  const int r = color.red();
  const int g = color.green();
  const int b = color.blue();

  for (int y = 0; y < h; ++y) {
    QRgb* line = reinterpret_cast<QRgb*>(img.scanLine(y));
    for (int x = 0; x < w; ++x) {
      int a = qAlpha(line[x]);
      if (a == 0) continue;
      if (alphaMul != 1.0) a = std::min(255, static_cast<int>(std::lround(a * alphaMul)));
      line[x] = qRgba(r, g, b, a);
    }
  }

  QPixmap out = QPixmap::fromImage(img);
  QIcon ic;
  ic.addPixmap(out, QIcon::Normal, QIcon::Off);
  ic.addPixmap(out, QIcon::Active, QIcon::Off);
  ic.addPixmap(out, QIcon::Selected, QIcon::Off);
  return ic;
}


// Variant that preserves the underlying icon's luminance/shading so the glyph remains crisp
// even when the base icon is multi-colored (e.g., Windows style standard icons).
static QIcon makeTintedIconPreserveDetails(const QIcon& base, const QColor& color, const QSize& size, double alphaMul = 1.0) {
  QPixmap pm = base.pixmap(size, QIcon::Normal, QIcon::Off);
  if (pm.isNull()) return base;

  QImage img = pm.toImage().convertToFormat(QImage::Format_ARGB32);
  const int w = img.width();
  const int h = img.height();
  const int cr = color.red();
  const int cg = color.green();
  const int cb = color.blue();

  // Keep a minimum intensity so dark strokes remain visible on dark themes.
  const double minF = 0.18;

  for (int y = 0; y < h; ++y) {
    QRgb* line = reinterpret_cast<QRgb*>(img.scanLine(y));
    for (int x = 0; x < w; ++x) {
      const QRgb px = line[x];
      int a = qAlpha(px);
      if (a == 0) continue;

      if (alphaMul != 1.0) a = std::min(255, static_cast<int>(std::lround(a * alphaMul)));

      const int gray = qGray(px); // 0..255
      double f = gray / 255.0;
      f = minF + (1.0 - minF) * f;

      const int r = std::clamp(static_cast<int>(std::lround(cr * f)), 0, 255);
      const int g = std::clamp(static_cast<int>(std::lround(cg * f)), 0, 255);
      const int b = std::clamp(static_cast<int>(std::lround(cb * f)), 0, 255);

      line[x] = qRgba(r, g, b, a);
    }
  }

  QPixmap out = QPixmap::fromImage(img);
  QIcon ic;
  ic.addPixmap(out, QIcon::Normal, QIcon::Off);
  ic.addPixmap(out, QIcon::Active, QIcon::Off);
  ic.addPixmap(out, QIcon::Selected, QIcon::Off);
  return ic;
}

// Forward declarations (file-scope helpers).


// Settings table value editor delegate (used only for Settings tabs: Global / Termination rule / Initialization).
// It infers a reasonable editor based on the current key/value text and commits edits back to the table model.
class SettingsValueDelegate final : public QStyledItemDelegate {
public:
  explicit SettingsValueDelegate(QObject* parent = nullptr) : QStyledItemDelegate(parent) {}

  QWidget* createEditor(QWidget* parent,
                        const QStyleOptionViewItem& option,
                        const QModelIndex& index) const override {
    Q_UNUSED(option);
    if (!index.isValid()) return nullptr;

    const QString key = index.sibling(index.row(), 0).data(Qt::DisplayRole).toString().trimmed();
    const QString val = index.data(Qt::DisplayRole).toString().trimmed();

    const EditorKind kind = inferKind(key, val);

    if (kind == EditorKind::Bool) {
      auto* cb = new QCheckBox(parent);
      cb->setText(QString());
      cb->setProperty("bool_numeric", isNumericBool(val));
      auto* that = const_cast<SettingsValueDelegate*>(this);
      QObject::connect(cb, &QCheckBox::toggled, that, [that, cb]() {
        Q_EMIT that->commitData(cb);
      });
      return cb;
    }

    if (kind == EditorKind::Int) {
      auto* sb = new QSpinBox(parent);
      sb->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
      sb->setSingleStep(1);
      auto* that = const_cast<SettingsValueDelegate*>(this);
      QObject::connect(sb, QOverload<int>::of(&QSpinBox::valueChanged), that, [that, sb](int) {
        Q_EMIT that->commitData(sb);
      });
      return sb;
    }

    if (kind == EditorKind::Double) {
      auto* dsb = new QDoubleSpinBox(parent);
      dsb->setLocale(QLocale::c());
      dsb->setDecimals(10);
      dsb->setRange(-1e100, 1e100);
      dsb->setSingleStep(0.01);
      auto* that = const_cast<SettingsValueDelegate*>(this);
      QObject::connect(dsb, QOverload<double>::of(&QDoubleSpinBox::valueChanged), that, [that, dsb](double) {
        Q_EMIT that->commitData(dsb);
      });
      return dsb;
    }

    auto* le = new QLineEdit(parent);
    auto* that = const_cast<SettingsValueDelegate*>(this);
    QObject::connect(le, &QLineEdit::editingFinished, that, [that, le]() {
      Q_EMIT that->commitData(le);
    });
    return le;
  }

  void setEditorData(QWidget* editor, const QModelIndex& index) const override {
    if (!index.isValid() || !editor) return;

    const QString key = index.sibling(index.row(), 0).data(Qt::DisplayRole).toString().trimmed();
    const QString val = index.data(Qt::DisplayRole).toString().trimmed();
    const EditorKind kind = inferKind(key, val);

    if (kind == EditorKind::Bool) {
      auto* cb = qobject_cast<QCheckBox*>(editor);
      if (!cb) return;
      cb->setChecked(parseBool(val));
      cb->setProperty("bool_numeric", isNumericBool(val));
      return;
    }

    if (kind == EditorKind::Int) {
      auto* sb = qobject_cast<QSpinBox*>(editor);
      if (!sb) return;
      bool ok = false;
      const int v = val.toInt(&ok);
      sb->setValue(ok ? v : 0);
      return;
    }

    if (kind == EditorKind::Double) {
      auto* dsb = qobject_cast<QDoubleSpinBox*>(editor);
      if (!dsb) return;
      bool ok = false;
      const double v = QLocale::c().toDouble(val, &ok);
      dsb->setValue(ok ? v : 0.0);
      return;
    }

    auto* le = qobject_cast<QLineEdit*>(editor);
    if (!le) return;
    le->setText(val);
  }

  void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override {
    if (!index.isValid() || !editor || !model) return;

    const QString key = index.sibling(index.row(), 0).data(Qt::DisplayRole).toString().trimmed();
    const QString cur = index.data(Qt::DisplayRole).toString().trimmed();
    const EditorKind kind = inferKind(key, cur);

    if (kind == EditorKind::Bool) {
      auto* cb = qobject_cast<QCheckBox*>(editor);
      if (!cb) return;
      const bool numeric = cb->property("bool_numeric").toBool();
      const QString out = numeric ? (cb->isChecked() ? "1" : "0") : (cb->isChecked() ? "true" : "false");
      model->setData(index, out, Qt::EditRole);
      return;
    }

    if (kind == EditorKind::Int) {
      auto* sb = qobject_cast<QSpinBox*>(editor);
      if (!sb) return;
      model->setData(index, QString::number(sb->value()), Qt::EditRole);
      return;
    }

    if (kind == EditorKind::Double) {
      auto* dsb = qobject_cast<QDoubleSpinBox*>(editor);
      if (!dsb) return;
      const QString out = QLocale::c().toString(dsb->value(), 'g', 15);
      model->setData(index, out, Qt::EditRole);
      return;
    }

    auto* le = qobject_cast<QLineEdit*>(editor);
    if (!le) return;
    model->setData(index, le->text(), Qt::EditRole);
  }

  void updateEditorGeometry(QWidget* editor,
                            const QStyleOptionViewItem& option,
                            const QModelIndex& index) const override {
    Q_UNUSED(index);
    if (editor) editor->setGeometry(option.rect);
  }

private:
  enum class EditorKind { Bool, Int, Double, Text };

  static bool isNumericBool(const QString& v) {
    const QString s = v.trimmed();
    return (s == "0" || s == "1");
  }

  static bool parseBool(const QString& v) {
    const QString s = v.trimmed().toLower();
    if (s == "true" || s == "yes" || s == "on") return true;
    if (s == "false" || s == "no" || s == "off") return false;
    if (s == "1") return true;
    if (s == "0") return false;
    return false;
  }

  static bool keySuggestsBool(const QString& key) {
    const QString k = key.trimmed().toLower();
    static const QStringList tokens = {
      "enable", "enabled", "use", "flag", "verbose", "debug", "show", "print", "log", "save", "plot"
    };
    for (const auto& t : tokens) {
      if (k.contains(t)) return true;
    }
    if (k.startsWith("is_") || k.startsWith("has_")) return true;
    if (k.endsWith("_enabled") || k.endsWith("_enable") || k.endsWith("_flag")) return true;
    return false;
  }

  static EditorKind inferKind(const QString& key, const QString& val) {
    const QString s = val.trimmed();
    const QString lower = s.toLower();

    if (lower == "true" || lower == "false" || lower == "yes" || lower == "no" || lower == "on" || lower == "off") {
      return EditorKind::Bool;
    }

    if ((s == "0" || s == "1") && keySuggestsBool(key)) {
      return EditorKind::Bool;
    }

    // Int?
    {
      static const QRegularExpression reInt(QStringLiteral("^[\\+\\-]?\\d+$"));
      if (reInt.match(s).hasMatch()) return EditorKind::Int;
    }

    // Double?
    {
      static const QRegularExpression reDbl(QStringLiteral("^[\\+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][\\+\\-]?\\d+)?$"));
      if (reDbl.match(s).hasMatch()) return EditorKind::Double;
    }

    return EditorKind::Text;
  }
};

static inline void openPersistentValueEditors(QTableWidget* t, int valueCol) {
  if (!t) return;
  for (int r = 0; r < t->rowCount(); ++r) {
    if (auto* it = t->item(r, valueCol)) {
      t->openPersistentEditor(it);
    }
  }
}










static bool parseConvergenceCsvFile(const QString& path,
                                   QVector<double>& outIterX,
                                   QVector<double>& outEvalX,
                                   QVector<double>& outY,
                                   QString& outInfo,
                                   bool* outHasEvalX = nullptr);

// Lightweight plot widget (no QtCharts dependency).
class ConvergencePlotWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  struct Series {
    QString label;
    QVector<double> x;
    QVector<double> y;
  };


  struct Band {
    QString label;
    QVector<double> x;
    QVector<double> yLow;
    QVector<double> yHigh;
  };

  explicit ConvergencePlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TranslucentBackground, false);
  }

  void setThemeMode(ThemeMode mode) {
    themeMode_ = mode;
    setAttribute(Qt::WA_TranslucentBackground, mode == ThemeMode::Transparent);
    update();
  }

  void setGridDensity(GridDensity density) {
    gridDensity_ = density;
    update();
  }

  void clear() {
    series_.clear();
    bands_.clear();
    yAxisLowerAnchorEnabled_ = false;
    yAxisLowerAnchor_ = std::numeric_limits<double>::quiet_NaN();
    update();
  }

  void clearYAxisLowerAnchor() {
    yAxisLowerAnchorEnabled_ = false;
    yAxisLowerAnchor_ = std::numeric_limits<double>::quiet_NaN();
    update();
  }

  void setYAxisLowerAnchor(double value) {
    if (!std::isfinite(value)) {
      clearYAxisLowerAnchor();
      return;
    }
    yAxisLowerAnchorEnabled_ = true;
    yAxisLowerAnchor_ = value;
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


  void clearBands() {
    bands_.clear();
    update();
  }

  void setIqrBand(QVector<double> x, QVector<double> yLow, QVector<double> yHigh, const QString& label = QString()) {
    Band b;
    b.label = label;
    b.x = std::move(x);
    b.yLow = std::move(yLow);
    b.yHigh = std::move(yHigh);
    bands_.clear();
    bands_.push_back(std::move(b));
    update();
  }

  void setXAxisTitle(const QString& title) {
    xTitle_ = title;
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();

    const bool dark = (themeMode_ == ThemeMode::Dark);
    const bool transparent = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transparent) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = (dark || transparent) ? QColor(180, 180, 180) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transparent ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transparent ? 65 : 38);

    if (!transparent) p.fillRect(r, bg);

    // Compute overall bounds across all series.
    bool any = false;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;

    for (const auto& s : series_) {
      const int n = (int(s.x.size()) < int(s.y.size()) ? int(s.x.size()) : int(s.y.size()));
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
      p.setPen(fg);
      p.drawText(r.adjusted(10, 10, -10, -10), Qt::AlignCenter,
                 "No convergence data available.\nEnable CSV convergence and run once.");
      return;
    }

    if (xmax <= xmin) xmax = xmin + 1.0;

    const double dataYMax = ymax;

    if (yAxisLowerAnchorEnabled_ && std::isfinite(yAxisLowerAnchor_)) {
      ymin = yAxisLowerAnchor_;
      ymax = std::max(dataYMax, ymin);
    }

    if (ymax <= ymin) {
      const double span = std::max(1.0, std::abs(ymin));
      ymax = ymin + span;
    }

    // Keep the lower bound anchored exactly; add headroom only on the upper side.
    const double ypadTop = 0.02 * (ymax - ymin);
    ymax += ypadTop;

    // Margins include tick labels plus axis titles.
    const int left = 90;   // title(~20) + gap(4) + tick labels(~60) + gap(6)
    const int right = 50;   // increased from 14 to prevent last x-tick label from clipping
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

    // Grid (drawn first, behind axes/series).
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int i = 1; i < div; ++i) {
        const int xx = plot.left() + int(double(i) * plot.width() / div);
        const int yy = plot.bottom() - int(double(i) * plot.height() / div);
        p.drawLine(QPoint(xx, plot.top()), QPoint(xx, plot.bottom()));
        p.drawLine(QPoint(plot.left(), yy), QPoint(plot.right(), yy));
      }
    }

    // Axes
    const QColor axis = fg;
    p.setPen(QPen(axis, 1));
    p.drawLine(plot.bottomLeft(), plot.bottomRight());
    p.drawLine(plot.bottomLeft(), plot.topLeft());

    // Ticks (5)
    const int ticks = 5;
    QFont f = font();
    f.setPointSize(std::max(8, f.pointSize() - 1));
    p.setFont(f);

    // Helper: format a value as a×10ⁿ using unicode superscripts.
    // For values that don't need scientific notation (small exponent), fall back to 'g'.
    auto formatSciNotation = [](double v) -> QString {
      if (v == 0.0) return QStringLiteral("0");
      const int exp = int(std::floor(std::log10(std::abs(v))));
      if (exp >= -3 && exp <= 4) {
        // No scientific notation needed for small numbers.
        return QString::number(v, 'g', 4);
      }
      const double mantissa = v / std::pow(10.0, exp);
      // Unicode superscript digits and minus.
      static const QChar sup[] = {
        QChar(0x2070), QChar(0x00B9), QChar(0x00B2), QChar(0x00B3),
        QChar(0x2074), QChar(0x2075), QChar(0x2076), QChar(0x2077),
        QChar(0x2078), QChar(0x2079)
      };
      const QChar supMinus(0x207B);
      QString expStr;
      int absExp = std::abs(exp);
      if (absExp == 0) { expStr = sup[0]; }
      else {
        while (absExp > 0) { expStr.prepend(sup[absExp % 10]); absExp /= 10; }
      }
      if (exp < 0) expStr.prepend(supMinus);
      // Format mantissa to 3 sig figs, strip trailing zeros.
      QString mStr = QString::number(mantissa, 'f', 2);
      while (mStr.endsWith('0') && mStr.contains('.')) mStr.chop(1);
      if (mStr.endsWith('.')) mStr.chop(1);
      return mStr + QStringLiteral("\u00D710") + expStr;  // ×
    };

    for (int i = 0; i <= ticks; ++i) {
      const double t = double(i) / ticks;
      const int xx = int(plot.left() + t * plot.width());
      const int yy = int(plot.bottom() - t * plot.height());

      // x ticks — last tick right-aligned to avoid clipping at widget edge.
      p.drawLine(QPoint(xx, plot.bottom()), QPoint(xx, plot.bottom() + 4));
      const double xv = xmin + t * (xmax - xmin);
      const Qt::Alignment xAlign = (i == ticks) ? (Qt::AlignRight | Qt::AlignTop)
                                                 : (Qt::AlignHCenter | Qt::AlignTop);
      p.drawText(QRect(xx - 40, plot.bottom() + 6, 80, 18), xAlign,
                 QString::number(xv, 'g', 6));

      // y ticks — use scientific notation with superscript exponents.
      p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
      const double yv = ymin + t * (ymax - ymin);
      p.drawText(QRect(0, yy - 9, plot.left() - 8, 18), Qt::AlignRight | Qt::AlignVCenter,
                 formatSciNotation(yv));
    }
    // No separate compact info panel here.
    // The convergence plot keeps only the standard colored legend when multiple
    // series are displayed, avoiding the extra black box on the right side.
  // Axis titles.
  {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.setPen(axis);

      // X title.
      p.drawText(QRect(plot.left(), plot.bottom() + 26, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignTop, xTitle_.isEmpty() ? "Iterations" : xTitle_);

      // Y title (rotated 90 degrees) — drawn at the far left, before tick labels.
      p.save();
      p.translate(10, plot.center().y());
      p.rotate(-90.0);
      p.drawText(QRect(-plot.height() / 2, -18, plot.height(), 36),
                 Qt::AlignHCenter | Qt::AlignVCenter, "Best solution");
      p.restore();

      // Restore tick font for subsequent drawing.
      p.setFont(f);
    }

    // IQR band (if provided). Draw behind the line series.
    if (!bands_.isEmpty()) {
      QColor bandFill = dark ? QColor(230, 230, 230) : QColor(40, 40, 40);
      bandFill.setAlpha(dark ? 38 : 28);

      for (int bi = 0; bi < bands_.size(); ++bi) {
        const auto& b = bands_[bi];
        int n = (int(b.x.size()) < int(b.yLow.size()) ? int(b.x.size()) : int(b.yLow.size()));
        n = (n < int(b.yHigh.size()) ? n : int(b.yHigh.size()));
        if (n < 2) continue;

        QPainterPath poly;
        // Upper edge
        bool started = false;
        for (int i = 0; i < n; ++i) {
          const double xi = b.x[i];
          const double yh = b.yHigh[i];
          if (!std::isfinite(xi) || !std::isfinite(yh)) continue;
          QPointF pt(mapX(xi), mapY(yh));
          if (!started) {
            poly.moveTo(pt);
            started = true;
          } else {
            poly.lineTo(pt);
          }
        }
        // Lower edge (reverse)
        for (int i = n - 1; i >= 0; --i) {
          const double xi = b.x[i];
          const double yl = b.yLow[i];
          if (!std::isfinite(xi) || !std::isfinite(yl)) continue;
          poly.lineTo(QPointF(mapX(xi), mapY(yl)));
        }
        poly.closeSubpath();

        p.fillPath(poly, bandFill);

        // Small, unobtrusive label.
        const QString lab = b.label.isEmpty() ? QString("IQR (25–75%)") : b.label;
        p.setPen(QPen(fg, 1));
        p.setFont(f);
        p.drawText(QRect(plot.left() + 6, plot.top() + 4, plot.width() - 12, 16),
                   Qt::AlignLeft | Qt::AlignTop, lab);
      }
    }

    // Series
    const int k = series_.size();
    int drawn = 0;
    for (int si = 0; si < series_.size(); ++si) {
      const auto& s = series_[si];
      const int n = (int(s.x.size()) < int(s.y.size()) ? int(s.x.size()) : int(s.y.size()));
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
        col = dark ? QColor(90, 180, 255) : QColor(0, 120, 215);
      } else {
        const int hue = (si * 360 / std::max(1, k));
        col = dark ? QColor::fromHsv(hue, 200, 235) : QColor::fromHsv(hue, 190, 200);
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
        QColor col = dark ? QColor::fromHsv((si * 360 / std::max(1, k)), 200, 235)
                          : QColor::fromHsv((si * 360 / std::max(1, k)), 190, 200);
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
  QVector<Band> bands_;
  QString xTitle_ = "Iterations";
  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
  bool yAxisLowerAnchorEnabled_ = false;
  double yAxisLowerAnchor_ = std::numeric_limits<double>::quiet_NaN();
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


static QStringList splitCsvLineRespectingQuotes(const QString& line) {
  QStringList out;
  QString cur;
  bool inQuotes = false;

  for (int i = 0; i < line.size(); ++i) {
    const QChar ch = line.at(i);

    if (ch == QChar('"')) {
      if (inQuotes && (i + 1) < line.size() && line.at(i + 1) == QChar('"')) {
        cur += QChar('"');
        ++i;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch == QChar(',') && !inQuotes) {
      out.push_back(cur.trimmed());
      cur.clear();
      continue;
    }

    cur += ch;
  }

  out.push_back(cur.trimmed());
  return out;
}


static QVector<int> parseBatchDimensionList(const QString& text, int defaultDim, int minDim, int maxDim) {
  QVector<int> dims;
  QSet<int> seen;

  const QStringList parts = text.split(QRegularExpression(QStringLiteral(R"([,;\s]+)")), Qt::SkipEmptyParts);
  for (const QString& part : parts) {
    bool ok = false;
    int v = part.trimmed().toInt(&ok);
    if (!ok) continue;
    if (v < minDim) v = minDim;
    if (maxDim > 0 && v > maxDim) v = maxDim;
    if (!seen.contains(v)) {
      seen.insert(v);
      dims.push_back(v);
    }
  }

  if (dims.isEmpty()) {
    int v = std::max(minDim, defaultDim);
    if (maxDim > 0) v = std::min(v, maxDim);
    dims.push_back(v);
  }

  std::sort(dims.begin(), dims.end());
  return dims;
}

static QString batchDimensionListToText(const QVector<int>& dims) {
  QStringList parts;
  parts.reserve(dims.size());
  for (int v : dims) {
    if (v > 0) parts << QString::number(v);
  }
  return parts.join(", ");
}

static bool parseBatchSummaryCsvMetadata(const QString& summaryPath,
                                         QString& outMethod,
                                         QString& outProblem,
                                         int& outDim,
                                         QString& outError) {
  outMethod.clear();
  outProblem.clear();
  outDim = 0;
  outError.clear();

  QFile f(summaryPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    outError = QString("Cannot open summary CSV '%1'.").arg(summaryPath);
    return false;
  }

  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);

  QString headerLine;
  while (!ts.atEnd() && headerLine.trimmed().isEmpty()) {
    headerLine = ts.readLine();
  }
  if (headerLine.trimmed().isEmpty()) {
    outError = "Summary CSV is empty.";
    return false;
  }

  QString dataLine;
  while (!ts.atEnd() && dataLine.trimmed().isEmpty()) {
    dataLine = ts.readLine();
  }
  if (dataLine.trimmed().isEmpty()) {
    outError = "Summary CSV does not contain a data row.";
    return false;
  }

  const QStringList headersRaw = splitCsvLineRespectingQuotes(headerLine);
  const QStringList valuesRaw = splitCsvLineRespectingQuotes(dataLine);

  QStringList headers;
  headers.reserve(headersRaw.size());
  for (const QString& h : headersRaw) headers.push_back(h.trimmed());

  auto colIndex = [&](const QString& name) -> int {
    for (int i = 0; i < headers.size(); ++i) {
      if (headers[i].compare(name, Qt::CaseInsensitive) == 0) return i;
    }
    return -1;
  };

  const int cMethod = colIndex("method");
  const int cProblem = colIndex("problem");
  const int cDim = colIndex("dim");

  if (cMethod < 0 || cProblem < 0 || cDim < 0) {
    outError = "Summary CSV header missing one of: method, problem, dim.";
    return false;
  }
  if (valuesRaw.size() <= std::max(cMethod, std::max(cProblem, cDim))) {
    outError = "Summary CSV data row is shorter than expected.";
    return false;
  }

  outMethod = valuesRaw[cMethod].trimmed();
  outProblem = valuesRaw[cProblem].trimmed();

  bool okDim = false;
  outDim = valuesRaw[cDim].trimmed().toInt(&okDim);
  if (outMethod.isEmpty() || outProblem.isEmpty() || !okDim || outDim <= 0) {
    outError = "Failed to parse method/problem/dim from summary CSV.";
    return false;
  }

  return true;
}

static QString siblingConvergenceCsvForSummary(const QString& summaryPath) {
  const QFileInfo fi(summaryPath);
  const QString dirPath = fi.absolutePath();
  const QString fileName = fi.fileName();

  QString stem = fileName;
  stem.replace(QRegularExpression(QStringLiteral("_summary\\.csv$"),
                                  QRegularExpression::CaseInsensitiveOption),
               QString());

  const QString exact = QDir(dirPath).absoluteFilePath(stem + "_convergence.csv");
  if (QFileInfo::exists(exact)) return exact;

  QDir dir(dirPath);
  const QFileInfoList matches = dir.entryInfoList(
      QStringList() << (stem + "*_convergence.csv"),
      QDir::Files,
      QDir::Time);
  if (!matches.isEmpty()) {
    return matches.front().absoluteFilePath();
  }

  return QString();
}


static int parseDimFromConvergenceCsvFilename(const QString& path) {
  const QString fileName = QFileInfo(path).fileName().trimmed();
  static const QRegularExpression re(
      QStringLiteral(R"(_d(\d+)_\d{8}-\d{6}_convergence\.csv$)"),
      QRegularExpression::CaseInsensitiveOption);
  const QRegularExpressionMatch m = re.match(fileName);
  if (!m.hasMatch()) return -1;
  bool ok = false;
  const int dim = m.captured(1).toInt(&ok);
  return (ok && dim > 0) ? dim : -1;
}



// Lightweight bar chart for sensitivity analysis (no QtCharts dependency).
class SensitivityBarWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  struct Point {
    double x    = 0.0;
    double mean = 0.0;
    double sd   = 0.0;
    int    n    = 0;
    double minV = std::numeric_limits<double>::quiet_NaN();
    double maxV = std::numeric_limits<double>::quiet_NaN();
  };

  explicit SensitivityBarWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void setThemeMode(ThemeMode mode) {
    themeMode_ = mode;
    update();
  }

  void setGridDensity(GridDensity density) {
    gridDensity_ = density;
    update();
  }

  void clear() {
    paramName_.clear();
    points_.clear();
    hasReference_ = false;
    referenceX_ = 0.0;
    update();
  }

  void setData(const QString& paramName, QVector<Point> pts) {
    paramName_ = paramName;
    points_ = std::move(pts);
    update();
  }

  void setReferenceX(double x) {
    referenceX_ = x;
    hasReference_ = true;
    update();
  }

  void clearReference() {
    hasReference_ = false;
    referenceX_ = 0.0;
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();

    const bool dark = (themeMode_ == ThemeMode::Dark);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = dark ? QColor(230, 230, 230) : QColor(24, 24, 24);

    QColor grid = dark ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : 38);

    p.fillRect(r, bg);

    if (points_.isEmpty()) {
      p.setPen(fg);
      p.drawText(r.adjusted(10, 10, -10, -10), Qt::AlignCenter,
                 "No sensitivity data available.\nRun with sensitivity enabled to generate sensitivity_results.csv.");
      return;
    }

    // Compute bounds.
    double ymin = points_[0].mean, ymax = points_[0].mean;
    for (const auto& pt : points_) {
      ymin = std::min(ymin, pt.mean);
      ymax = std::max(ymax, pt.mean);
      if (pt.sd > 0.0) {
        ymin = std::min(ymin, pt.mean - pt.sd);
        ymax = std::max(ymax, pt.mean + pt.sd);
      }
    }
    if (ymax <= ymin) ymax = ymin + 1.0;

    // Padding.
    const double ypad = 0.06 * (ymax - ymin);
    ymin -= ypad;
    ymax += ypad;

    // Plot rect.
    const int left = 72;
    const int right = 16;
    const int top = 18;
    const int bottom = 58; // room for x labels + axis title
    QRect plot = r.adjusted(left, top, -right, -bottom);
    if (plot.width() < 60 || plot.height() < 60) return;

    auto mapY = [&](double y) {
      return plot.bottom() - (y - ymin) * (plot.height() / (ymax - ymin));
    };

    // Grid (behind axes/bars).
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int i = 1; i < div; ++i) {
        const int xx = plot.left() + int(double(i) * plot.width() / div);
        const int yy = plot.bottom() - int(double(i) * plot.height() / div);
        p.drawLine(QPoint(xx, plot.top()), QPoint(xx, plot.bottom()));
        p.drawLine(QPoint(plot.left(), yy), QPoint(plot.right(), yy));
      }
    }

    // Axes.
    p.setPen(QPen(fg, 1));
    p.drawLine(plot.bottomLeft(), plot.bottomRight());
    p.drawLine(plot.bottomLeft(), plot.topLeft());

    // Y ticks (5).
    const int ticks = 5;
    QFont tickFont = font();
    tickFont.setPointSize(std::max(8, tickFont.pointSize() - 1));
    p.setFont(tickFont);

    for (int i = 0; i <= ticks; ++i) {
      const double t = double(i) / ticks;
      const double yv = ymin + (1.0 - t) * (ymax - ymin);
      const int yy = int(plot.top() + t * plot.height());
      p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
      const QString lab = QString::number(yv, 'g', 6);
      p.drawText(QRect(0, yy - 10, plot.left() - 6, 20), Qt::AlignRight | Qt::AlignVCenter, lab);
    }

    // Bar geometry (categorical slots).
    const int n = points_.size();
    const double gap = 0.16; // gap fraction inside each slot
    const double slotW = double(plot.width()) / std::max(1, n);
    const double barW = slotW * (1.0 - gap);

    auto slotCenterX = [&](int i) -> double {
      return plot.left() + (i + 0.5) * slotW;
    };

    // Identify best (minimum mean) and (optional) reference.
    int bestIdx = 0;
    for (int i = 1; i < n; ++i) if (points_[i].mean < points_[bestIdx].mean) bestIdx = i;

    int refIdx = -1;
    if (hasReference_) {
      // Pick the nearest slot to the reference value to keep the highlighting stable even
      // when the reference value is not an exact CSV x-value due to reflect/formatting.
      double bestDx = std::numeric_limits<double>::infinity();
      for (int i = 0; i < n; ++i) {
        const double dx = std::abs(points_[i].x - referenceX_);
        if (dx < bestDx) {
          bestDx = dx;
          refIdx = i;
        }
      }
    }

    // Colors (theme-aware, deterministic).
    QColor baseBar = dark ? QColor(90, 180, 255) : QColor(0, 120, 215);
    baseBar.setAlpha(dark ? 110 : 130);

    QColor bestBar = dark ? QColor(0, 200, 120) : QColor(0, 160, 80);
    bestBar.setAlpha(dark ? 180 : 185);

    QColor refBar = dark ? QColor(255, 170, 60) : QColor(230, 140, 0);
    refBar.setAlpha(dark ? 175 : 175);

    // Optional reference vertical line (use slot-based x mapping for consistency with bars).
    if (hasReference_) {
      double refPx = slotCenterX(0);

      if (n == 1) {
        refPx = slotCenterX(0);
      } else if (refIdx >= 0) {
        refPx = slotCenterX(refIdx);
      } else {
        // Interpolate between neighbor slots using the numeric x values.
        int hi = 0;
        while (hi < n && points_[hi].x < referenceX_) ++hi;
        if (hi <= 0) {
          refPx = slotCenterX(0);
        } else if (hi >= n) {
          refPx = slotCenterX(n - 1);
        } else {
          const int lo = hi - 1;
          const double x0 = points_[lo].x;
          const double x1 = points_[hi].x;
          const double t = (x1 != x0) ? ((referenceX_ - x0) / (x1 - x0)) : 0.0;
          refPx = slotCenterX(lo) * (1.0 - t) + slotCenterX(hi) * t;
        }
      }

      QColor refLine = refBar;
      refLine.setAlpha(dark ? 190 : 190);
      p.setPen(QPen(refLine, 2, Qt::DashLine));
      p.drawLine(QPointF(refPx, plot.top()), QPointF(refPx, plot.bottom()));

      // Label near the top (kept short to avoid clutter).
      p.setPen(fg);
      p.drawText(QRectF(refPx + 6, plot.top() + 2, plot.width(), 14),
                 Qt::AlignLeft | Qt::AlignVCenter, "Reference");
    }

    // Bars + error bars.
    for (int i = 0; i < n; ++i) {
      const double cx = slotCenterX(i);
      const double x0 = cx - barW / 2.0;
      const double x1 = cx + barW / 2.0;

      const int yMean = int(mapY(points_[i].mean));
      const int yBase = plot.bottom();

      int yTop = std::min(yMean, yBase);
      int yBot = std::max(yMean, yBase);

      // Guarantee visible bars even if yMean == yBase (common when the best == ymin).
      const int minH = 2;
      if ((yBot - yTop) < minH) {
        yTop = std::max(plot.top(), yBot - minH);
      }

      QRectF barRect(QPointF(x0, yTop), QPointF(x1, yBot));
      p.setPen(Qt::NoPen);

      QColor fill = baseBar;
      if (i == bestIdx) fill = bestBar;
      else if (i == refIdx) fill = refBar;

      p.setBrush(fill);
      p.drawRect(barRect);

      // If bar is both best and reference, add an outline.
      if (i == bestIdx && i == refIdx) {
        p.setPen(QPen(refBar, 2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(barRect.adjusted(0.5, 0.5, -0.5, -0.5));
      }

      // Error bar (sd) around the mean.
      if (points_[i].sd > 0.0) {
        const int yHi = int(mapY(points_[i].mean + points_[i].sd));
        const int yLo = int(mapY(points_[i].mean - points_[i].sd));
        p.setPen(QPen(fg, 1));
        p.drawLine(QPointF(cx, yHi), QPointF(cx, yLo));
        p.drawLine(QPointF(cx - 5, yHi), QPointF(cx + 5, yHi));
        p.drawLine(QPointF(cx - 5, yLo), QPointF(cx + 5, yLo));
      }

      // X label.
      const QString xlab = QString::number(points_[i].x, 'g', 6);
      p.setPen(fg);
      p.drawText(QRectF(x0, plot.bottom() + 6, barW, 18), Qt::AlignHCenter | Qt::AlignTop, xlab);
    }

    // Title.
    p.setPen(fg);
    const QString title = paramName_.isEmpty()
      ? "Sensitivity (Mean best_f)"
      : QString("Sensitivity: %1 (Mean best_f)").arg(paramName_);
    p.drawText(QRect(plot.left(), r.top() + 2, plot.width(), 16), Qt::AlignLeft | Qt::AlignVCenter, title);

    // X axis label.
    if (!paramName_.isEmpty()) {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.drawText(QRect(plot.left(), r.bottom() - 30, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignVCenter, paramName_);
    }
  }

private:
  QString paramName_;
  QVector<Point> points_;
  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
  bool hasReference_ = false;
  double referenceX_ = 0.0;
};


// Boxplot-style distribution plot (no QtCharts dependency).
class DistributionPlotWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  struct Group {
    QString label;
    QVector<double> values; // per-run final best_f
  };

  explicit DistributionPlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }

  void clear() { groups_.clear(); update(); }

  void setGroups(QVector<Group> g) { groups_ = std::move(g); update(); }

protected:
  static double quantileLinear(const QVector<double>& sorted, double q) {
    if (sorted.isEmpty()) return std::numeric_limits<double>::quiet_NaN();
    if (sorted.size() == 1) return sorted[0];
    const double pos = q * double(sorted.size() - 1);
    const int i = int(std::floor(pos));
    const int n = int(sorted.size());
    const int j = (i + 1 < n ? (i + 1) : (n - 1));
    const double t = pos - double(i);
    return sorted[i] * (1.0 - t) + sorted[j] * t;
  }

  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);

    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = dark ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = dark ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : 38);

    p.fillRect(r, bg);

    if (groups_.isEmpty()) {
      p.setPen(fg);
      p.drawText(r.adjusted(10, 10, -10, -10), Qt::AlignCenter,
                 "No distribution data available.\nRun with multiple independent runs to show a boxplot.");
      return;
    }

    // Compute global bounds.
    bool any = false;
    double ymin = 0.0, ymax = 0.0;
    for (const auto& g : groups_) {
      for (double v : g.values) {
        if (!std::isfinite(v)) continue;
        if (!any) { ymin = ymax = v; any = true; }
        ymin = std::min(ymin, v);
        ymax = std::max(ymax, v);
      }
    }
    if (!any) {
      p.setPen(fg);
      p.drawText(r.adjusted(10, 10, -10, -10), Qt::AlignCenter,
                 "No finite values were found in the distribution data.");
      return;
    }
    if (ymax <= ymin) ymax = ymin + 1.0;

    const double ypad = 0.08 * (ymax - ymin);
    ymin -= ypad;
    ymax += ypad;

    // Plot rect.
    const int left = 72;
    const int right = 16;
    const int top = 18;
    const int bottom = 58;
    QRect plot = r.adjusted(left, top, -right, -bottom);
    if (plot.width() < 60 || plot.height() < 60) return;

    auto mapY = [&](double y) {
      return plot.bottom() - (y - ymin) * (plot.height() / (ymax - ymin));
    };

    // Grid
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int i = 1; i < div; ++i) {
        const int xx = plot.left() + int(double(i) * plot.width() / div);
        const int yy = plot.bottom() - int(double(i) * plot.height() / div);
        p.drawLine(QPoint(xx, plot.top()), QPoint(xx, plot.bottom()));
        p.drawLine(QPoint(plot.left(), yy), QPoint(plot.right(), yy));
      }
    }

    // Axes.
    p.setPen(QPen(fg, 1));
    p.drawLine(plot.bottomLeft(), plot.bottomRight());
    p.drawLine(plot.bottomLeft(), plot.topLeft());

    // Y ticks.
    const int ticks = 5;
    QFont tickFont = font();
    tickFont.setPointSize(std::max(8, tickFont.pointSize() - 1));
    p.setFont(tickFont);
    for (int i = 0; i <= ticks; ++i) {
      const double t = double(i) / ticks;
      const double yv = ymin + (1.0 - t) * (ymax - ymin);
      const int yy = int(plot.top() + t * plot.height());
      p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
      p.drawText(QRect(0, yy - 10, plot.left() - 6, 20), Qt::AlignRight | Qt::AlignVCenter,
                 QString::number(yv, 'g', 6));
    }

    // Box geometry per group (categorical).
    const int gN = groups_.size();
    const double slotW = double(plot.width()) / std::max(1, gN);
    const double boxW = slotW * 0.42;

    auto slotCenterX = [&](int gi) -> double { return plot.left() + (gi + 0.5) * slotW; };

    QColor boxFill = dark ? QColor(80, 140, 230) : QColor(0, 120, 215);
    boxFill.setAlpha(dark ? 85 : 70);
    const QColor boxStroke = dark ? QColor(190, 190, 190) : QColor(30, 30, 30);
    const QColor medianCol = dark ? QColor(255, 230, 150) : QColor(180, 120, 0);

    for (int gi = 0; gi < gN; ++gi) {
      QVector<double> vals;
      vals.reserve(groups_[gi].values.size());
      for (double v : groups_[gi].values) if (std::isfinite(v)) vals.push_back(v);
      if (vals.isEmpty()) continue;
      std::sort(vals.begin(), vals.end());

      const double q1 = quantileLinear(vals, 0.25);
      const double med = quantileLinear(vals, 0.50);
      const double q3 = quantileLinear(vals, 0.75);
      const double iqr = q3 - q1;

      // Tukey whiskers (1.5*IQR), clipped to observed values.
      const double loLim = q1 - 1.5 * iqr;
      const double hiLim = q3 + 1.5 * iqr;

      double wLo = vals.first();
      for (int i = 0; i < vals.size(); ++i) {
        if (vals[i] >= loLim) { wLo = vals[i]; break; }
      }
      double wHi = vals.last();
      for (int i = vals.size() - 1; i >= 0; --i) {
        if (vals[i] <= hiLim) { wHi = vals[i]; break; }
      }

      const double cx = slotCenterX(gi);
      const double x0 = cx - boxW / 2.0;
      const double x1 = cx + boxW / 2.0;

      const int yQ1 = int(mapY(q1));
      const int yQ3 = int(mapY(q3));
      const int yMed = int(mapY(med));
      const int yWL = int(mapY(wLo));
      const int yWH = int(mapY(wHi));

      // Whiskers
      p.setPen(QPen(boxStroke, 1));
      p.drawLine(QPointF(cx, yQ1), QPointF(cx, yWL));
      p.drawLine(QPointF(cx, yQ3), QPointF(cx, yWH));
      const double capW = boxW * 0.55;
      p.drawLine(QPointF(cx - capW / 2.0, yWL), QPointF(cx + capW / 2.0, yWL));
      p.drawLine(QPointF(cx - capW / 2.0, yWH), QPointF(cx + capW / 2.0, yWH));

      // Box
      QRectF boxRect(QPointF(x0, yQ3), QPointF(x1, yQ1));
      p.fillRect(boxRect, boxFill);
      p.drawRect(boxRect);

      // Median line
      p.setPen(QPen(medianCol, 2));
      p.drawLine(QPointF(x0, yMed), QPointF(x1, yMed));
      // Sample points (all runs). Inliers are filled; outliers are hollow (Tukey 1.5*IQR).
      // Deterministic jitter keeps points readable without requiring a RNG.
      auto jitter01 = [](int k) -> double {
        // xorshift-like hash to [0,1)
        uint32_t x = uint32_t(k) * 2654435761u;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return double(x & 0x00FFFFFFu) / double(0x01000000u);
      };

      const double jitterW = boxW * 0.32;
      const double rIn = 2.0;
      const double rOut = 2.4;

      QColor inFill = boxStroke;
       inFill.setAlpha(dark ? 190 : 160);

      p.setPen(Qt::NoPen);
      p.setBrush(inFill);

      // Draw inliers first (filled).
      for (int i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const bool isOut = (v < wLo - 1e-12) || (v > wHi + 1e-12);
        if (isOut) continue;
        const double j = (jitter01(i) - 0.5) * 2.0 * jitterW;
        const int yy = int(mapY(v));
        p.drawEllipse(QPointF(cx + j, yy), rIn, rIn);
      }

      // Draw outliers on top (hollow).
      p.setPen(QPen(boxStroke, 1));
      p.setBrush(Qt::NoBrush);
      for (int i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const bool isOut = (v < wLo - 1e-12) || (v > wHi + 1e-12);
        if (!isOut) continue;
        const double j = (jitter01(i) - 0.5) * 2.0 * jitterW;
        const int yy = int(mapY(v));
        p.drawEllipse(QPointF(cx + j, yy), rOut, rOut);
      }
      // X label
      const QString lab = groups_[gi].label;
      const QRect textR(int(cx - slotW / 2.0), plot.bottom() + 8, int(slotW), 18);
      p.setPen(fg);
      p.drawText(textR, Qt::AlignHCenter | Qt::AlignTop,
                 QFontMetrics(p.font()).elidedText(lab, Qt::ElideRight, textR.width()));
    }

    // Inline info panel (kept inside the image).
    {
      QFont lf = font();
      lf.setPointSize(std::max(8, lf.pointSize() - 1));
      p.setFont(lf);

      QStringList lines;
      lines << "Whiskers: 1.5*IQR; Filled dots: inliers; Hollow dots: outliers";

      const int gN = groups_.size();
      for (int gi = 0; gi < gN; ++gi) {
        QVector<double> vals;
        vals.reserve(groups_[gi].values.size());
        for (double v : groups_[gi].values) {
          if (std::isfinite(v)) vals.push_back(v);
        }
        if (vals.isEmpty()) continue;
        std::sort(vals.begin(), vals.end());

        const int n = int(vals.size());
        const double vmin = vals.front();
        const double med  = quantileLinear(vals, 0.50);

        double sum = 0.0;
        for (double v : vals) sum += v;
        const double mean = sum / double(std::max(1, n));

        double ss = 0.0;
        for (double v : vals) {
          const double d = v - mean;
          ss += d * d;
        }
        const double sd = (n > 1) ? std::sqrt(ss / double(n - 1)) : 0.0;

        QString lab = groups_[gi].label.trimmed();
        if (lab.isEmpty()) lab = QString("Group %1").arg(gi + 1);
        if (lab.size() > 18) lab = lab.left(17) + "...";

        lines << QString("%1: n=%2  min=%3  med=%4  mean=%5  sd=%6")
                    .arg(lab)
                    .arg(n)
                    .arg(vmin, 0, 'g', 6)
                    .arg(med,  0, 'g', 6)
                    .arg(mean, 0, 'g', 6)
                    .arg(sd,   0, 'g', 6);
      }

      const QFontMetrics fm(p.font());
      int w = 0;
      int h = 0;
      for (const QString& ln : lines) {
        w = std::max(w, fm.horizontalAdvance(ln));
        h += fm.height();
      }

      const int pad = 8;
      QRect panel(plot.right() - (w + 2 * pad) - 6,
                  plot.top() + 6,
                  w + 2 * pad,
                  h + 2 * pad);

      QColor panelBg = dark ? QColor(0, 0, 0, 140) : QColor(255, 255, 255, 200);
      QColor panelBorder = dark ? QColor(200, 200, 200, 90) : QColor(0, 0, 0, 70);

      p.setPen(QPen(panelBorder, 1));
      p.setBrush(panelBg);
      p.drawRoundedRect(panel, 6, 6);

      p.setPen(fg);
      int yy = panel.top() + pad;
      for (const QString& ln : lines) {
        p.drawText(panel.left() + pad, yy + fm.ascent(), ln);
        yy += fm.height();
      }
    }


    // Axis titles.
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.setPen(fg);

      p.drawText(QRect(plot.left(), plot.bottom() + 28, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignTop, "Runs (final best_f)");

      // Y title (rotated 90 degrees).
      p.save();
      p.translate(18, plot.center().y());
      p.rotate(-90.0);
      p.drawText(QRect(-plot.height() / 2, -18, plot.height(), 36),
                 Qt::AlignHCenter | Qt::AlignVCenter, "Best solution");
      p.restore();
    }
  }

private:
  QVector<Group> groups_;
  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
};




// Box plot widget for paired Wilcoxon results, with p-value annotations drawn inside the plot.
class WilcoxonBoxPlotWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }

  struct Group {
    QString label;
    QVector<double> values;
  };

  struct Annotation {
    int i = -1;
    int j = -1;
    QString text;
    bool significant = false;
  };

  explicit WilcoxonBoxPlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(240);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void clear() {
    groups_.clear();
    ann_.clear();
    title_.clear();
    subtitle_.clear();
    yLabel_.clear();
    update();
  }

  void setData(QVector<Group> groups, QVector<Annotation> ann, const QString& title,
              const QString& subtitle, const QString& yLabel) {
    groups_ = std::move(groups);
    ann_ = std::move(ann);
    title_ = title;
    subtitle_ = subtitle;
    yLabel_ = yLabel;
    update();
  }

  void setYLabel(const QString& y) {
    yLabel_ = y;
    update();
  }

  // Backward-compatible API used by existing MainWindow code.
  // Converts (labels + value vectors) to Group objects and forwards to the main setData().
  void setData(const QStringList& labels, const QVector<QVector<double>>& valueGroups,
               const QString& title, const QString& subtitle, const QVector<Annotation>& ann) {
    if (valueGroups.isEmpty()) {
      clear();
      return;
    }
    QVector<Group> groups;
    groups.reserve(valueGroups.size());
    for (int i = 0; i < valueGroups.size(); ++i) {
      Group g;
      if (i < labels.size()) g.label = labels[i];
      else g.label = QString("G%1").arg(i + 1);
      g.values = valueGroups[i];
      groups.push_back(std::move(g));
    }
    setData(std::move(groups), ann, title, subtitle, yLabel_);
  }

protected:
  static double quantileLinear(const QVector<double>& sorted, double q) {
    if (sorted.isEmpty()) return std::numeric_limits<double>::quiet_NaN();
    if (q <= 0.0) return sorted.front();
    if (q >= 1.0) return sorted.back();
    const double pos = q * double(sorted.size() - 1);
    const int i0 = int(std::floor(pos));
    const int i1 = int(std::ceil(pos));
    if (i0 == i1) return sorted[i0];
    const double t = pos - double(i0);
    return sorted[i0] * (1.0 - t) + sorted[i1] * t;
  }
  static void assignAnnotationLevels(const QVector<Annotation>& ann, QVector<int>& outLevels, int& outLevelCount) {
    outLevels = QVector<int>(ann.size(), 0);
    outLevelCount = 0;

    struct Span { int l; int r; int idx; };
    QVector<Span> spans;
    spans.reserve(ann.size());
    for (int a = 0; a < ann.size(); ++a) {
      const int i = ann[a].i;
      const int j = ann[a].j;
      if (i < 0 || j < 0) continue;
      spans.push_back(Span{std::min(i, j), std::max(i, j), a});
    }

    std::sort(spans.begin(), spans.end(), [](const Span& A, const Span& B){
      const int wA = A.r - A.l;
      const int wB = B.r - B.l;
      if (wA != wB) return wA < wB;
      return A.l < B.l;
    });

    QVector<QVector<Span>> levels;
    for (const Span& s : spans) {
      int lev = 0;
      for (; lev < levels.size(); ++lev) {
        bool overlap = false;
        for (const Span& t : levels[lev]) {
          if (!(s.r < t.l || s.l > t.r)) { overlap = true; break; }
        }
        if (!overlap) break;
      }
      if (lev >= levels.size()) levels.push_back(QVector<Span>{});
      levels[lev].push_back(s);
      outLevels[s.idx] = lev;
    }

    outLevelCount = levels.size();
  }

  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = dark ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = dark ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : 38);

    p.fillRect(r, bg);

    if (groups_.isEmpty()) {
      p.setPen(fg);
      p.drawText(r, Qt::AlignCenter, "No Wilcoxon data");
      return;
    }

    // Precompute per-group stats.
    struct Stats { double mn, q1, med, q3, mx; };
    QVector<Stats> stats;
    stats.reserve(groups_.size());

    double globalMin = std::numeric_limits<double>::infinity();
    double globalMax = -std::numeric_limits<double>::infinity();

    for (const Group& g : groups_) {
      QVector<double> v;
      v.reserve(g.values.size());
      for (double x : g.values) {
        if (std::isfinite(x)) v.push_back(x);
      }
      std::sort(v.begin(), v.end());
      Stats s{};
      if (v.isEmpty()) {
        s.mn = s.q1 = s.med = s.q3 = s.mx = std::numeric_limits<double>::quiet_NaN();
      } else {
        s.mn = v.front();
        s.mx = v.back();
        s.q1 = quantileLinear(v, 0.25);
        s.med = quantileLinear(v, 0.50);
        s.q3 = quantileLinear(v, 0.75);
      }
      stats.push_back(s);
      if (std::isfinite(s.mn)) globalMin = std::min(globalMin, s.mn);
      if (std::isfinite(s.mx)) globalMax = std::max(globalMax, s.mx);
    }

    if (!std::isfinite(globalMin) || !std::isfinite(globalMax)) {
      p.setPen(fg);
      p.drawText(r, Qt::AlignCenter, "No Wilcoxon data");
      return;
    }

    double range = globalMax - globalMin;
    if (range <= 0.0) range = std::max(1.0, std::abs(globalMax));

    // Annotation levels (stack brackets).
    QVector<int> levels;
    int levelCount = 0;
    // Simple greedy: each new span goes to the first level with no overlap.
    struct Span { int l; int r; int idx; };
    QVector<Span> spans;
    spans.reserve(ann_.size());
    for (int a = 0; a < ann_.size(); ++a) {
      int i = ann_[a].i;
      int j = ann_[a].j;
      if (i < 0 || j < 0) continue;
      int l = std::min(i, j);
      int rr = std::max(i, j);
      spans.push_back({l, rr, a});
    }
    std::sort(spans.begin(), spans.end(), [](const Span& A, const Span& B){
      const int wA = A.r - A.l;
      const int wB = B.r - B.l;
      if (wA != wB) return wA < wB;
      return A.l < B.l;
    });
    QVector<QVector<Span>> levSpans;
    levels = QVector<int>(ann_.size(), 0);
    for (const Span& s : spans) {
      int lev = 0;
      for (; lev < levSpans.size(); ++lev) {
        bool overlap = false;
        for (const Span& t : levSpans[lev]) {
          if (!(s.r < t.l || s.l > t.r)) { overlap = true; break; }
        }
        if (!overlap) break;
      }
      if (lev >= levSpans.size()) levSpans.push_back({});
      levSpans[lev].push_back(s);
      levels[s.idx] = lev;
    }
    levelCount = levSpans.size();

    const double annStep = 0.06 * range;
    const double annPad = (levelCount > 0) ? (annStep * (levelCount + 1)) : (0.12 * range);

    const double yMin = globalMin;
    const double yMax = globalMax + annPad;

    // Layout.
    const int left = 70;
    const int right = 20;
    const int bottom = 58;
    const int topBase = 26;
    const int top = topBase + std::min(140, levelCount * 14 + 18);

    QRect plot(left, top, r.width() - left - right, r.height() - top - bottom);
    if (plot.width() <= 10 || plot.height() <= 10) {
      p.setPen(fg);
      p.drawText(r, Qt::AlignCenter, "Plot area too small");
      return;
    }

    auto yToPix = [&](double y) -> int {
      const double t = (y - yMin) / (yMax - yMin);
      return plot.bottom() - int(std::llround(t * double(plot.height())));
    };

    // Title/subtitle.
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(10, tf.pointSize()));
      p.setFont(tf);
      p.setPen(fg);
      p.drawText(QRect(plot.left(), 6, plot.width(), 18), Qt::AlignHCenter | Qt::AlignVCenter, title_);
      QFont sf = font();
      sf.setPointSize(std::max(8, sf.pointSize() - 1));
      p.setFont(sf);
      p.drawText(QRect(plot.left(), 22, plot.width(), 16), Qt::AlignHCenter | Qt::AlignVCenter, subtitle_);
    }

    // Grid (mirrors Convergence).
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int gi = 1; gi < div; ++gi) {
        const int xx = plot.left() + int(double(gi) * plot.width() / div);
        const int yy = plot.top() + int(double(gi) * plot.height() / div);
        p.drawLine(QPoint(xx, plot.top()), QPoint(xx, plot.bottom()));
        p.drawLine(QPoint(plot.left(), yy), QPoint(plot.right(), yy));
      }
    }

    // Axes.
    p.setPen(QPen(fg, 1));
    p.drawLine(QPoint(plot.left(), plot.bottom()), QPoint(plot.right(), plot.bottom()));
    p.drawLine(QPoint(plot.left(), plot.bottom()), QPoint(plot.left(), plot.top()));

    // Y ticks.
    {
      QFont tickFont = font();
      tickFont.setPointSize(std::max(8, tickFont.pointSize() - 1));
      p.setFont(tickFont);

      const int ticks = 5;
      for (int t = 0; t <= ticks; ++t) {
        const double yy = yMin + (yMax - yMin) * double(t) / double(ticks);
        const int yp = yToPix(yy);

        p.setPen(fg);
        p.drawLine(QPoint(plot.left() - 4, yp), QPoint(plot.left(), yp));

        QString lab;
        if (std::abs(yy) >= 1e6 || std::abs(yy) < 1e-3) lab = QString::number(yy, 'g', 4);
        else lab = QString::number(yy, 'g', 6);
        p.drawText(QRect(0, yp - 8, left - 6, 16), Qt::AlignRight | Qt::AlignVCenter, lab);
      }
    }

    // X positions.
    const int k = groups_.size();
    const double dx = plot.width() / double(k);

    // Draw boxplots (outlined boxes + points).
    auto groupColor = [&](int idx) {
      // Deterministic, distinct hues.
      const int hue = (idx * 47) % 360;
      QColor c = QColor::fromHsv(hue, 180, (dark ? 230 : 170));
      c.setAlpha(230);
      return c;
    };

    for (int i = 0; i < k; ++i) {
      const double cx = plot.left() + (i + 0.5) * dx;
      const int xMid = int(std::llround(cx));
      const int w = int(std::max(10.0, dx * 0.55));

      const Stats s = stats[i];
      if (!std::isfinite(s.med)) continue;

      const QColor c = groupColor(i);
      QPen pen(c, 2.0);
      pen.setCosmetic(true);

      const int yMn = yToPix(s.mn);
      const int yMx = yToPix(s.mx);
      const int yQ1 = yToPix(s.q1);
      const int yQ3 = yToPix(s.q3);
      const int yMed = yToPix(s.med);

      // Whisker.
      p.setPen(pen);
      p.setBrush(Qt::NoBrush);
      p.drawLine(xMid, yMn, xMid, yMx);
      p.drawLine(xMid - w/4, yMn, xMid + w/4, yMn);
      p.drawLine(xMid - w/4, yMx, xMid + w/4, yMx);

      // Box (outline only).
      QRect box(xMid - w/2, yQ3, w, yQ1 - yQ3);
      p.drawRect(box);

      // Median.
      p.drawLine(xMid - w/2, yMed, xMid + w/2, yMed);

      // Points (jittered).
      {
        QColor pc = c;
        pc.setAlpha(dark ? 200 : 190);
        p.setPen(Qt::NoPen);
        p.setBrush(pc);

        // Larger points than default.
        const double rad = std::min(2.8, std::max(1.6, dx * 0.028));
        QRandomGenerator rng(quint32(0xC0FFEEu ^ quint32(i * 2654435761u)));

        const auto& vals = groups_[i].values;
        for (int t = 0; t < vals.size(); ++t) {
          const double v = vals[t];
          if (!std::isfinite(v)) continue;
          const double jitter = (rng.generateDouble() - 0.5) * double(w) * 0.55;
          const int xp = int(std::llround(double(xMid) + jitter));
          const int yp = yToPix(v);
          p.drawEllipse(QPointF(xp, yp), rad, rad);
        }
      }

      // X label.
      p.setPen(fg);
      QFont lf = font();
      lf.setPointSize(std::max(7, lf.pointSize() - 1));
      p.setFont(lf);
      const QString lab = groups_[i].label;
      p.save();
      p.translate(xMid, plot.bottom() + 6);
      p.rotate(-35.0);
      p.drawText(QRect(-int(dx*0.6), 0, int(dx*1.2), 22), Qt::AlignHCenter | Qt::AlignTop, lab);
      p.restore();
    }

    // Axis titles.
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.setPen(fg);

      p.drawText(QRect(plot.left(), plot.bottom() + 34, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignTop, "Methods (aggregated per problem)");

      // Y title (rotated 90 degrees).
      p.save();
      p.translate(18, plot.center().y());
      p.rotate(-90.0);
      p.drawText(QRect(-plot.height() / 2, -18, plot.height(), 36),
                 Qt::AlignHCenter | Qt::AlignVCenter, yLabel_.isEmpty() ? "Metric" : yLabel_);
      p.restore();
    }

    // Annotations (brackets + p text).
    if (!ann_.isEmpty()) {
      QFont af = font();
      af.setPointSize(std::max(8, af.pointSize() - 1));
      p.setFont(af);

      const double y0 = globalMax;

      for (int a = 0; a < ann_.size(); ++a) {
        const Annotation& an = ann_[a];
        if (an.i < 0 || an.j < 0) continue;
        if (an.i >= k || an.j >= k) continue;

        const int i = an.i;
        const int j = an.j;
        const int l = std::min(i, j);
        const int rr = std::max(i, j);

        const double cxL = plot.left() + (l + 0.5) * dx;
        const double cxR = plot.left() + (rr + 0.5) * dx;

        const int xL = int(std::llround(cxL));
        const int xR = int(std::llround(cxR));

        const int lev = (a < levels.size()) ? levels[a] : 0;
        const double yData = y0 + annStep * double(lev + 1);
        const int yP = yToPix(yData);

        QColor pen = an.significant ? (dark ? QColor(255, 210, 0) : QColor(160, 100, 0)) : (dark ? QColor(190, 190, 190) : QColor(90, 90, 90));
        pen.setAlpha(230);
        p.setPen(QPen(pen, 1.5));

        // Bracket.
        p.drawLine(xL, yP, xR, yP);
        p.drawLine(xL, yP, xL, yP + 6);
        p.drawLine(xR, yP, xR, yP + 6);

        // Text.
        p.setPen(fg);
        const int mid = (xL + xR) / 2;
        p.drawText(QRect(mid - 120, yP - 16, 240, 14), Qt::AlignHCenter | Qt::AlignVCenter, an.text);
      }
    }
  }

private:
  QVector<Group> groups_;
  QVector<Annotation> ann_;
  QString title_;
  QString subtitle_;
  QString yLabel_;

  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
};
// Rank plot for batch statistics (average ranks across problems; lower rank is better).
class RankPlotWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  struct Annotation {
    QString a;
    QString b;
    QString text;
    bool significant = false;
  };

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }
  void setAnnotations(QVector<Annotation> ann) { ann_ = std::move(ann); update(); }

  explicit RankPlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    infoBox_ = new QPlainTextEdit(this);
    infoBox_->setReadOnly(true);
    infoBox_->setFrameStyle(QFrame::NoFrame);
    infoBox_->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    infoBox_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    infoBox_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    infoBox_->setFocusPolicy(Qt::NoFocus);
    infoBox_->setVisible(false);
  }

  void clear() {
    methods_.clear();
    avgRanks_.clear();
    ranksByMethod_.clear();
    title_.clear();
    subtitle_.clear();
    infoText_.clear();
    ann_.clear();
    if (infoBox_) {
      infoBox_->clear();
      infoBox_->setVisible(false);
    }
    update();
  }

  void setData(QStringList methods, QVector<double> avgRanks, const QString& title, const QString& subtitle) {
  methods_ = std::move(methods);
  avgRanks_ = std::move(avgRanks);
  ranksByMethod_.clear();
  title_ = title;
  subtitle_ = subtitle;
  ann_.clear();
  update();
}

void setData(QStringList methods, QVector<QVector<double>> ranksByMethod, const QString& title, const QString& subtitle) {
  methods_ = std::move(methods);
  ranksByMethod_ = std::move(ranksByMethod);

  // Keep an average-rank summary for ordering and labels.
  avgRanks_.clear();
  avgRanks_.reserve(methods_.size());
  for (int i = 0; i < ranksByMethod_.size(); ++i) {
    const QVector<double>& v = ranksByMethod_[i];
    double s = 0.0;
    int c = 0;
    for (double x : v) {
      if (std::isfinite(x)) { s += x; ++c; }
    }
    avgRanks_.push_back(c > 0 ? (s / double(c)) : std::numeric_limits<double>::quiet_NaN());
  }

  title_ = title;
  subtitle_ = subtitle;
  ann_.clear();
  update();
}

  void setInfoText(const QString& text) {
    infoText_ = text;
    if (infoBox_) {
      infoBox_->setPlainText(infoText_);
      infoBox_->setVisible(!infoText_.trimmed().isEmpty());
      updateInfoBoxGeometry();
    }
    update();
  }

protected:
  void resizeEvent(QResizeEvent* e) override {
    QWidget::resizeEvent(e);
    updateInfoBoxGeometry();
  }

  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = dark ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = dark ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : 38);

    p.fillRect(r, bg);

    const bool hasBoxes = (!ranksByMethod_.isEmpty() && ranksByMethod_.size() == methods_.size());
    const bool hasBars  = (!avgRanks_.isEmpty() && avgRanks_.size() == methods_.size());
    if (methods_.isEmpty() || (!hasBoxes && !hasBars)) {
      p.setPen(fg);
      QFont f = font();
      f.setBold(true);
      p.setFont(f);
      p.drawText(r.adjusted(12, 12, -12, -12), Qt::AlignCenter, "No Friedman data.Run a batch first.");
      return;
    }

    const int k = methods_.size();

    auto quantileLinear = [&](const QVector<double>& sorted, double q) -> double {
      if (sorted.isEmpty()) return std::numeric_limits<double>::quiet_NaN();
      if (sorted.size() == 1) return sorted[0];
      const double pos = q * double(sorted.size() - 1);
      const int i = int(std::floor(pos));
      const int n = int(sorted.size());
      const int j = (i + 1 < n ? (i + 1) : (n - 1));
      const double a = sorted[i];
      const double b = sorted[j];
      return a + (pos - double(i)) * (b - a);
    };

    struct BoxStats {
      double q1 = std::numeric_limits<double>::quiet_NaN();
      double med = std::numeric_limits<double>::quiet_NaN();
      double q3 = std::numeric_limits<double>::quiet_NaN();
      double wLo = std::numeric_limits<double>::quiet_NaN();
      double wHi = std::numeric_limits<double>::quiet_NaN();
      double mean = std::numeric_limits<double>::quiet_NaN();
      QVector<double> out;
    };

    struct Item {
      QString m;
      double key = std::numeric_limits<double>::quiet_NaN();  // sort key (median rank)
      double label = std::numeric_limits<double>::quiet_NaN(); // shown above (median)
      BoxStats bs;
      QVector<double> vals;
    };

    QVector<Item> items;
    items.reserve(k);

    if (hasBoxes) {
      for (int i = 0; i < k; ++i) {
        QVector<double> vals;
        vals.reserve(ranksByMethod_[i].size());
        for (double v : ranksByMethod_[i]) if (std::isfinite(v)) vals.push_back(v);
        std::sort(vals.begin(), vals.end());

        BoxStats bs;
        if (!vals.isEmpty()) {
          bs.q1 = quantileLinear(vals, 0.25);
          bs.med = quantileLinear(vals, 0.50);
          bs.q3 = quantileLinear(vals, 0.75);
          const double iqr = bs.q3 - bs.q1;

          const double loLim = bs.q1 - 1.5 * iqr;
          const double hiLim = bs.q3 + 1.5 * iqr;

          bs.wLo = vals.first();
          for (int t = 0; t < vals.size(); ++t) {
            if (vals[t] >= loLim) { bs.wLo = vals[t]; break; }
          }
          bs.wHi = vals.last();
          for (int t = vals.size() - 1; t >= 0; --t) {
            if (vals[t] <= hiLim) { bs.wHi = vals[t]; break; }
          }

          double s = 0.0;
          for (double v : vals) s += v;
          bs.mean = s / double(vals.size());

          for (double v : vals) {
            if (v < bs.wLo || v > bs.wHi) bs.out.push_back(v);
          }
        }

        Item it;
        it.m = methods_[i];
        it.key = bs.med;
        it.label = bs.med;
        it.bs = std::move(bs);
        it.vals = vals;
        items.push_back(std::move(it));
      }
    } else {
      // Fallback: bar plot of average ranks.
      for (int i = 0; i < k; ++i) {
        Item it;
        it.m = methods_[i];
        it.key = avgRanks_[i];
        it.label = avgRanks_[i];
        items.push_back(std::move(it));
      }
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
      const double ra = a.key;
      const double rb = b.key;
      if (!std::isfinite(ra) && !std::isfinite(rb)) return a.m.toLower() < b.m.toLower();
      if (!std::isfinite(ra)) return false;
      if (!std::isfinite(rb)) return true;
      if (ra == rb) return a.m.toLower() < b.m.toLower();
      return ra < rb;
    });

    // Rank range is [1..k]. Add padding for aesthetics.
    // Extra headroom is reserved at the top for pairwise annotations.
    QHash<QString, int> displayIndex;
    for (int i = 0; i < items.size(); ++i) displayIndex.insert(items[i].m, i);

    struct AnnDraw { int l; int r; QString text; bool significant; int level; };
    QVector<AnnDraw> annDraw;
    annDraw.reserve(ann_.size());
    for (const auto& an : ann_) {
      const int ia = displayIndex.contains(an.a) ? displayIndex.value(an.a) : -1;
      const int ib = displayIndex.contains(an.b) ? displayIndex.value(an.b) : -1;
      if (ia < 0 || ib < 0 || ia == ib) continue;
      AnnDraw ad;
      ad.l = std::min(ia, ib);
      ad.r = std::max(ia, ib);
      ad.text = an.text;
      ad.significant = an.significant;
      ad.level = 0;
      annDraw.push_back(std::move(ad));
    }

    // Assign non-overlapping levels for bracket stacking.
    QVector<QVector<int>> levels;
    QVector<int> order(annDraw.size());
    for (int i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){
      const int wa = annDraw[a].r - annDraw[a].l;
      const int wb = annDraw[b].r - annDraw[b].l;
      if (wa != wb) return wa < wb;
      return annDraw[a].l < annDraw[b].l;
    });
    for (int idx : order) {
      int lev = 0;
      for (; lev < levels.size(); ++lev) {
        bool overlap = false;
        for (int jdx : levels[lev]) {
          if (!(annDraw[idx].r < annDraw[jdx].l || annDraw[idx].l > annDraw[jdx].r)) { overlap = true; break; }
        }
        if (!overlap) break;
      }
      if (lev >= levels.size()) levels.push_back(QVector<int>{});
      levels[lev].push_back(idx);
      annDraw[idx].level = lev;
    }
    const int levelCount = levels.size();
    const double annStep = 0.28; // rank units

    double rmin = 0.5 - annStep * double(levelCount + 2);
    double rmax = double(k) + 0.5;

    // Compute left margin so tick labels do not collide with the plot.
    QFont tickFont = font();
    tickFont.setPointSize(std::max(8, tickFont.pointSize() - 1));
    QFontMetrics fmTick(tickFont);
    int maxTickW = 0;
    if (hasBoxes) {
      for (int rv = 1; rv <= k; ++rv) {
        maxTickW = std::max(maxTickW, fmTick.horizontalAdvance(QString::number(rv)));
      }
    } else {
      const int ticks = 5;
      for (int i = 0; i < ticks; ++i) {
        const double t = double(i) / double(std::max(1, ticks - 1));
        const double rv = rmin + t * (rmax - rmin);
        maxTickW = std::max(maxTickW, fmTick.horizontalAdvance(QString::number(rv, 'f', 2)));
      }
    }

    const int left = std::max(56, maxTickW + 14);
    int right = 18;
    const int top = 34;
    const int bottom = 64;

    const bool hasInfo = (infoBox_ && infoBox_->isVisible());
    if (hasInfo) {
      const int panelW = std::max(260, std::min(420, int(double(r.width()) * 0.42)));
      right = right + panelW + 10;
    }

    QRect plot = r.adjusted(left, top, -right, -bottom);
    if (plot.width() < 80 || plot.height() < 60) return;

    auto mapY = [&](double rv) {
      return plot.top() + (rv - rmin) * (plot.height() / (rmax - rmin));
    };

    // Grid (mirrors Convergence).
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int gi = 1; gi < div; ++gi) {
        const int xx = plot.left() + int(double(gi) * plot.width() / div);
        const int yy = plot.top() + int(double(gi) * plot.height() / div);
        p.drawLine(QPoint(xx, plot.top()), QPoint(xx, plot.bottom()));
        p.drawLine(QPoint(plot.left(), yy), QPoint(plot.right(), yy));
      }
    }

    // Axes.
    p.setPen(QPen(fg, 1));
    p.drawLine(QPoint(plot.left(), plot.top()), QPoint(plot.left(), plot.bottom()));
    p.drawLine(QPoint(plot.left(), plot.bottom()), QPoint(plot.right(), plot.bottom()));

    // Y ticks.
    p.setFont(tickFont);
    p.setPen(fg);
    if (hasBoxes) {
      for (int rv = 1; rv <= k; ++rv) {
        const int yy = int(mapY(double(rv)));
        p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
        p.drawText(QRect(2, yy - 10, left - 8, 20), Qt::AlignRight | Qt::AlignVCenter, QString::number(rv));
      }
    } else {
      const int ticks = 5;
      for (int i = 0; i < ticks; ++i) {
        const double t = double(i) / double(std::max(1, ticks - 1));
        const double rv = rmin + t * (rmax - rmin);
        const int yy = int(mapY(rv));
        p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
        p.drawText(QRect(2, yy - 10, left - 8, 20), Qt::AlignRight | Qt::AlignVCenter, QString::number(rv, 'f', 2));
      }
    }

    // Plot elements.
    const int n = items.size();
    const double slotW = double(plot.width()) / std::max(1, n);
    const double gap = 0.18;
    const double boxW = slotW * (1.0 - gap);
    const double x0 = plot.left();

    if (hasBoxes) {
      for (int i = 0; i < n; ++i) {
        const Item& it = items[i];
        const double cx = x0 + i * slotW + 0.5 * slotW;
        const double xx = x0 + i * slotW + 0.5 * (slotW - boxW);

        QColor fill = dark ? QColor(70, 130, 255) : QColor(50, 110, 235);
        fill.setAlpha(dark ? 110 : 95);
        if (i == 0) {
          fill = dark ? QColor(0, 200, 120) : QColor(0, 160, 80);
          fill.setAlpha(dark ? 170 : 150);
        }

        const double yQ1  = mapY(it.bs.q1);
        const double yMed = mapY(it.bs.med);
        const double yQ3  = mapY(it.bs.q3);
        const double yLo  = mapY(it.bs.wLo);
        const double yHi  = mapY(it.bs.wHi);

        // Whiskers.
        p.setPen(QPen(fg, 1));
        p.drawLine(QPointF(cx, yLo), QPointF(cx, yHi));
        p.drawLine(QPointF(cx - 0.22 * boxW, yLo), QPointF(cx + 0.22 * boxW, yLo));
        p.drawLine(QPointF(cx - 0.22 * boxW, yHi), QPointF(cx + 0.22 * boxW, yHi));

        // Box.
        const QRectF box(xx, yQ1, boxW, std::max(1.0, (yQ3 - yQ1)));
        p.fillRect(box, fill);
        p.setPen(QPen(fg, 1));
        p.drawRect(box);

        // Median.
        p.drawLine(QPointF(box.left(), yMed), QPointF(box.right(), yMed));

        // Points (jittered), like Wilcoxon.
        {
          QColor pc = fg;
          pc.setAlpha(dark ? 190 : 170);
          p.setPen(Qt::NoPen);
          p.setBrush(pc);

          const double rad = std::min(2.4, std::max(1.1, slotW * 0.055));
          QRandomGenerator rng(quint32(qHash(it.m) ^ 0xA5A5A5A5u));

          for (double v : it.vals) {
            if (!std::isfinite(v)) continue;
            const double jitter = (rng.generateDouble() - 0.5) * double(boxW) * 0.55;
            const double xp = cx + jitter;
            const double yp = mapY(v);
            p.drawEllipse(QPointF(xp, yp), rad, rad);
          }
        }

        p.setBrush(Qt::NoBrush);

        // Mean marker.
        if (std::isfinite(it.bs.mean)) {
          const double yMean = mapY(it.bs.mean);
          p.setBrush(fg);
          p.drawEllipse(QPointF(cx, yMean), 2.4, 2.4);
          p.setBrush(Qt::NoBrush);
        }

        // Outliers.
        for (double ov : it.bs.out) {
          const double yO = mapY(ov);
          p.drawEllipse(QPointF(cx, yO), 2.0, 2.0);
        }

        // Median label.
        p.drawText(QRectF(xx, std::min(yLo, yHi) - 18, boxW, 16), Qt::AlignHCenter | Qt::AlignVCenter,
                   std::isfinite(it.label) ? QString::number(it.label, 'f', 3) : "-");

        // Method label.
        const QString lab = QFontMetrics(p.font()).elidedText(it.m, Qt::ElideRight, int(boxW));
        p.drawText(QRectF(xx, plot.bottom() + 6, boxW, 34), Qt::AlignHCenter | Qt::AlignTop, lab);
      }
    } else {
      // Legacy bars.
      const double barW = boxW;
      for (int i = 0; i < n; ++i) {
        const double rv = items[i].label;
        const double xx = x0 + i * slotW + 0.5 * (slotW - barW);
        const double y = mapY(rv);
        const QRectF bar(xx, y, barW, plot.bottom() - y);

        QColor col = dark ? QColor(70, 130, 255) : QColor(50, 110, 235);
        col.setAlpha(dark ? 150 : 150);
        if (i == 0) {
          col = dark ? QColor(0, 200, 120) : QColor(0, 160, 80);
          col.setAlpha(dark ? 200 : 200);
        }
        p.fillRect(bar, col);
        p.setPen(QPen(fg, 1));
        p.drawRect(bar);

        p.setPen(fg);
        p.drawText(QRectF(bar.left(), bar.top() - 18, bar.width(), 16), Qt::AlignHCenter | Qt::AlignVCenter,
                   QString::number(rv, 'f', 3));

        const QString lab = QFontMetrics(p.font()).elidedText(items[i].m, Qt::ElideRight, int(barW));
        p.drawText(QRectF(bar.left(), plot.bottom() + 6, bar.width(), 34), Qt::AlignHCenter | Qt::AlignTop, lab);
      }
    }

    // Pairwise annotations (all pairs).
    if (!annDraw.isEmpty()) {
      QFont af = font();
      af.setPointSize(std::max(8, af.pointSize() - 1));
      p.setFont(af);

      for (const auto& ad : annDraw) {
        const double cxL = x0 + ad.l * slotW + 0.5 * slotW;
        const double cxR = x0 + ad.r * slotW + 0.5 * slotW;

        const double yRv = 0.5 - annStep * double(ad.level + 1);
        const int yP = int(std::llround(mapY(yRv)));

        QColor penCol = ad.significant ? (dark ? QColor(255, 210, 0) : QColor(160, 100, 0))
                                       : (dark ? QColor(190, 190, 190) : QColor(90, 90, 90));
        penCol.setAlpha(230);
        QPen pen(penCol, 1.5);
        pen.setCosmetic(true);
        p.setPen(pen);

        const int xL = int(std::llround(cxL));
        const int xR = int(std::llround(cxR));
        p.drawLine(xL, yP, xR, yP);
        p.drawLine(xL, yP, xL, yP + 6);
        p.drawLine(xR, yP, xR, yP + 6);

        p.setPen(fg);
        const int mid = (xL + xR) / 2;
        p.drawText(QRect(mid - 120, yP - 16, 240, 14), Qt::AlignHCenter | Qt::AlignVCenter, ad.text);
      }
    }

    // Title + subtitle.
    p.setPen(fg);
    QFont tf = font();
    tf.setBold(true);
    tf.setPointSize(std::max(10, tf.pointSize()));
    p.setFont(tf);
    const QString t = title_.isEmpty() ? (hasBoxes ? "Ranks distribution (lower is better)" : "Average ranks (lower is better)") : title_;
    p.drawText(QRect(plot.left(), r.top() + 4, plot.width(), 18), Qt::AlignLeft | Qt::AlignVCenter, t);

    QFont sf = font();
    sf.setPointSize(std::max(8, sf.pointSize() - 1));
    p.setFont(sf);
    if (!subtitle_.isEmpty()) {
      p.drawText(QRect(plot.left(), r.top() + 20, plot.width(), 14), Qt::AlignLeft | Qt::AlignVCenter, subtitle_);
    }

    // Y axis label (placed inside the plot to avoid collisions with tick labels).
    p.save();
    QColor yLab = fg;
    yLab.setAlpha(200);
    p.setPen(yLab);
    p.translate(plot.left() + 14, plot.center().y());
    p.rotate(-90);
    p.drawText(QRect(-plot.height() / 2, -10, plot.height(), 18), Qt::AlignHCenter | Qt::AlignVCenter, "Rank");
    p.restore();

    // Info panel (right).
    if (hasInfo) {
      const int rightPad = 8;
      const int panelW = std::max(260, std::min(420, int(double(r.width()) * 0.42)));
      const int panelH = std::max(140, r.height() - top - bottom);
      const QRect panelOuter(r.right() - rightPad - panelW, r.top() + top, panelW, panelH);

      QColor panelBg = dark ? QColor(32, 32, 32) : QColor(248, 248, 248);
      QColor panelBorder = dark ? QColor(70, 70, 70) : QColor(200, 200, 200);
      p.setPen(QPen(panelBorder, 1));
      p.setBrush(panelBg);
      p.drawRoundedRect(panelOuter, 10, 10);

      infoBox_->setGeometry(panelOuter.adjusted(6, 6, -6, -6));
    }
  }

private:
  void updateInfoBoxGeometry() {
    if (!infoBox_ || !infoBox_->isVisible()) return;

    const QRect r = rect();
    const int top = 34;
    const int bottom = 18;
    const int rightPad = 18;

    const int panelW = std::max(260, std::min(420, int(double(r.width()) * 0.42)));
    const int panelH = std::max(140, r.height() - top - bottom);
    const QRect panelOuter(r.right() - rightPad - panelW, r.top() + top, panelW, panelH);

    infoBox_->setGeometry(panelOuter.adjusted(6, 6, -6, -6));
  }

  QStringList methods_;
  QVector<double> avgRanks_;
  QVector<QVector<double>> ranksByMethod_;
  QString title_;
  QString subtitle_;

  QString infoText_;
  QPlainTextEdit* infoBox_ = nullptr;

  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
  QVector<Annotation> ann_;
};




void MainWindow::tryLoadSensitivityForTab(int index) {
  if (index < 0 || index >= static_cast<int>(outputRuns_.size())) return;
  OutputRunTab& tab = outputRuns_[index];

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

  setWindowTitle("OptimSolution v51");
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
  const QString flagPath = QDir(projectRoot_).filePath(".rebuild_pending");
  if (!QFileInfo::exists(flagPath)) return;

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
  // Runs silently after this process exits; re-creates the flag on failure.
  const QString nRoot = QDir::toNativeSeparators(projectRoot_).replace("'", "''");
  const QString nFlag = QDir::toNativeSeparators(flagPath).replace("'", "''");
  const QString nExe  = QDir::toNativeSeparators(appExe).replace("'", "''");

  const QString psCmd = QString(
      "Start-Sleep -Seconds 2; "
      "Set-Location '%1'; "
      "cmake --build build --config Release; "
      "if ($LASTEXITCODE -ne 0) { 'Rebuild needed' | Out-File '%2' }; "
      "Start-Process '%3'%4")
      .arg(nRoot, nFlag, nExe,
           appArgsStr.isEmpty()
               ? QString()
               : QString(" -ArgumentList '%1'").arg(appArgsStr.trimmed()));

  QProcess::startDetached("powershell.exe",
      {"-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
       "-Command", psCmd});
#else
  // ── Linux / macOS: write a shell script ───────────────────────────
  const QString scriptPath = QDir(projectRoot_).filePath("_rebuild_restart.sh");
  {
    QFile s(scriptPath);
    if (s.open(QIODevice::WriteOnly | QIODevice::Text)) {
      QTextStream t(&s);
      t << "#!/bin/bash\n";
      t << "sleep 2\n";
      t << "cd \"" << projectRoot_ << "\"\n";
      t << "cmake --build build --config Release\n";
      t << "if [ $? -ne 0 ]; then\n";
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

  runBtn_ = new QPushButton("Run", topBox);
  runBtn_->setIconSize(QSize(20, 20));
  runBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_MediaPlay), QColor(0, 160, 0), QSize(20, 20), 2.0));
  runBtn_->setStyleSheet("QPushButton { color: #00a000; font-weight: bold; }");

  busySpinner_ = new BusySpinner(topBox);
  busySpinner_->setFixedSize(18, 18);
  busySpinner_->stop();

  selectionMaxBtn_ = new QPushButton("Maximize", topBox);
  selectionMaxBtn_->setProperty("role", "regionMax");
  selectionMaxBtn_->setToolTip("Expand the Selection area and hide Settings and Output. Click again to restore all areas.");
  selectionMaxBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_TitleBarMaxButton), QColor(0, 120, 215), QSize(18, 18), 2.0));
  selectionMaxBtn_->setIconSize(QSize(18, 18));
  selectionMaxBtn_->setFlat(true);
  connect(selectionMaxBtn_, &QPushButton::clicked, this, [this]() { toggleRegionFocus(FocusArea::Selection); });

  {
    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(refreshBtn_);
    btnRow->addWidget(saveBatchSelectionBtn_);
    btnRow->addWidget(loadBatchSelectionBtn_);
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
  form->addRow("Run mode", runModeBox_);

  form->addRow("Optimization method", methodBox_);
  form->addRow("Problem", problemBox_);
  form->addRow("Dimension", dimSpin_);

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
        [this, refreshBatchMethodSettingsSelector](QListWidgetItem*, QListWidgetItem*) {
          refreshBatchMethodSettingsSelector();
          refreshBatchSelectionView();
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
    const int defaultDim = std::max(2, newDefaultDim);
    const QList<QListWidgetItem*> items = (batchProblemsList_ ? batchProblemsList_->selectedItems() : QList<QListWidgetItem*>());
    for (QListWidgetItem* it : items) {
      const QString p = batchBaseProblemShort(it->data(Qt::UserRole).toString().trimmed());
      if (p.isEmpty() || fixedDimForProblem(p) > 0) continue;
      if (batchProblemDimOverride_.contains(p)) continue;
      const int cachedDim = batchCachedProblemDims_.value(batchProblemDisplayKey(p, defaultDim), -1);
      if (cachedDim > 0 && cachedDim != defaultDim) {
        invalidateBatchProblemCache(p);
      }
    }
    syncBatchProblemDimsTable();
    refreshBatchSelectionView();
  }
});

batchListsRow->addWidget(batchProblemsBox, 1);
batchLay->addLayout(batchListsRow, 1);

auto* batchSettingsBox = new QGroupBox("Batch settings", batchPanel_);
auto* batchSettingsLay = new QFormLayout(batchSettingsBox);
batchSettingsLay->setContentsMargins(8, 8, 8, 8);

// Runs are taken from [global].runs (batch does not override runs).
batchRunsSpin_ = new QSpinBox(batchSettingsBox);
batchRunsSpin_->setRange(1, 1000000);
batchRunsSpin_->setValue(30);
batchRunsSpin_->setEnabled(false);

batchMetricCombo_ = new QComboBox(batchSettingsBox);
batchMetricCombo_->addItem("best_f (best of runs)", int(BatchMetricMode::BestFinalBestF));
batchMetricCombo_->addItem("best_f (mean of runs)", int(BatchMetricMode::MeanFinalBestF));
batchMetricCombo_->addItem("Iteration of best_f", int(BatchMetricMode::IterationAtBest));
batchMetricCombo_->addItem("Function evaluations of best_f", int(BatchMetricMode::EvalsAtBest));
batchMetricCombo_->setCurrentIndex(0);

batchAggCombo_ = new QComboBox(batchSettingsBox);
batchAggCombo_->addItems(QStringList() << "Mean" << "Median" << "Min" << "Max");
batchAggCombo_->setCurrentText("Mean");

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

batchSettingsLay->addRow("Runs", batchRunsSpin_);
batchSettingsLay->addRow("Metric", batchMetricCombo_);
batchSettingsLay->addRow("Aggregate", batchAggCombo_);
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
      const int batchIndex = runModeBox_->findData(1);
      if (batchIndex >= 0 && runModeBox_->currentIndex() != batchIndex) {
        runModeBox_->setCurrentIndex(batchIndex);
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
  settingsMaxBtn_ = new QPushButton("Maximize", cfgBox);
  settingsMaxBtn_->setProperty("role", "regionMax");
  settingsMaxBtn_->setToolTip("Expand the Settings area and hide Selection and Output. Click again to restore all areas.");
  settingsMaxBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_TitleBarMaxButton), QColor(0, 120, 215), QSize(18, 18), 2.0));
  settingsMaxBtn_->setIconSize(QSize(18, 18));
settingsMaxBtn_->setFlat(true);
  connect(settingsMaxBtn_, &QPushButton::clicked, this, [this]() { toggleRegionFocus(FocusArea::Settings); });
  cfgBtnRow->addSpacing(6);
  cfgBtnRow->addWidget(settingsMaxBtn_);
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
  selectionSettingsSplitter_ = new QSplitter(Qt::Horizontal, central);
  auto* topSplit = selectionSettingsSplitter_;
  topSplit->setChildrenCollapsible(false);
  topSplit->addWidget(topBox);
  topSplit->addWidget(cfgBox);
  topSplit->setStretchFactor(0, 2);
  topSplit->setStretchFactor(1, 3);
  rootLay->addWidget(topSplit, 1);

  outputBox_ = new QGroupBox("Output", central);
  outputBox_->setObjectName("outputBox");
  outputBox_->setProperty("focused", false);
  auto* outBox = outputBox_;
  auto* outLay = new QVBoxLayout(outBox);
  // Global output controls (apply to next run, and to the currently selected output tab when reloading).
  auto* outTop = new QHBoxLayout();
  forceConvergenceCsvChk_ = nullptr;
  loadExperimentCsvBtn_ = new QPushButton("Load experiment CSV...", outBox);
  loadExperimentCsvBtn_->setIcon(selectCfgBtn_ ? selectCfgBtn_->icon() : style()->standardIcon(QStyle::SP_DialogOpenButton));
  loadExperimentCsvBtn_->setIconSize(QSize(18, 18));
  loadExperimentCsvBtn_->setToolTip("Load a previously saved experiment CSV. If the selected folder contains batch summary CSV files, the whole batch is reconstructed.");
  reloadConvergenceBtn_ = new QPushButton("Refresh plots", outBox);
  reloadConvergenceBtn_->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
  reloadConvergenceBtn_->setIconSize(QSize(18, 18));
  clearCsvBtn_ = new QPushButton("Delete CSV files", outBox);
  clearCsvBtn_->setIcon(makeTintedIconPreserveDetails(style()->standardIcon(QStyle::SP_TrashIcon), QColor(220, 0, 0), QSize(18, 18), 2.0));
  clearCsvBtn_->setIconSize(QSize(18, 18));
  clearCsvBtn_->setToolTip("Delete all *.csv files under the GUI runtime working folders (current and previous runs).");

  outTop->addWidget(runBtn_);
  outTop->addWidget(busySpinner_);
  outTop->addSpacing(8);
  outTop->addStretch(1);
  outTop->addWidget(loadExperimentCsvBtn_);
  outTop->addWidget(reloadConvergenceBtn_);
  outTop->addWidget(clearCsvBtn_);
  outputMaxBtn_ = new QPushButton("Maximize", outBox);
  outputMaxBtn_->setProperty("role", "regionMax");
  outputMaxBtn_->setToolTip("Expand the Output area and hide Selection and Settings. Click again to restore all areas.");
  outputMaxBtn_->setIcon(makeTintedIcon(style()->standardIcon(QStyle::SP_TitleBarMaxButton), QColor(0, 120, 215), QSize(18, 18), 2.0));
  outputMaxBtn_->setIconSize(QSize(18, 18));
outputMaxBtn_->setFlat(true);
  connect(outputMaxBtn_, &QPushButton::clicked, this, [this]() { toggleRegionFocus(FocusArea::Output); });
  outTop->addSpacing(6);
  outTop->addWidget(outputMaxBtn_);
  outLay->addLayout(outTop);

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
  progressWidget_ = new QWidget(outBox);
  progressWidget_->setLayout(progRow);
  progressWidget_->setVisible(false);
  outLay->addWidget(progressWidget_);

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

  outputTabs_ = new QTabWidget(outBox);
  outputTabs_->setDocumentMode(true);

  outputTabs_->setTabsClosable(true);
  connect(outputTabs_, &QTabWidget::tabCloseRequested, this, &MainWindow::onOutputTabCloseRequested);
  connect(outputTabs_, &QTabWidget::currentChanged, this, &MainWindow::onOutputTabChanged);

  // Do not create a batch summary tab eagerly.
  // It is created only when a batch run starts or when external batch results are loaded.
  activeOutputRunIndex_ = -1;


  outLay->addWidget(outputTabs_);
  // ── Code Wizard panel (right of Output) ──────────────────────────
  auto* wizardBox = new QGroupBox("Code Wizard", central);
  wizardBox_ = wizardBox;
  wizardBox->setObjectName("wizardBox");
  auto* wizLay = new QVBoxLayout(wizardBox);
  wizLay->setSpacing(8);

  auto* wizDescLbl = new QLabel(
    "Generate skeleton .h/.cpp files\nfor new methods or problems.\n\n"
    "Files are placed in src/methods/\nor src/problems/ of your project.", wizardBox);
  wizDescLbl->setWordWrap(true);
  wizDescLbl->setAlignment(Qt::AlignTop | Qt::AlignLeft);

  auto* newMethodBtn = new QPushButton("New method…", wizardBox);
  newMethodBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  newMethodBtn->setIconSize(QSize(18, 18));
  newMethodBtn->setToolTip("Open the Method Wizard to generate a new optimizer .h/.cpp skeleton");

  auto* newProblemBtn = new QPushButton("New problem…", wizardBox);
  newProblemBtn->setIcon(style()->standardIcon(QStyle::SP_FileDialogNewFolder));
  newProblemBtn->setIconSize(QSize(18, 18));
  newProblemBtn->setToolTip("Open the Problem Wizard to generate a new benchmark problem .h/.cpp skeleton");

  wizLay->addWidget(wizDescLbl);
  wizLay->addSpacing(4);
  wizLay->addWidget(newMethodBtn);
  wizLay->addWidget(newProblemBtn);
  wizLay->addStretch(1);

  connect(newMethodBtn,  &QPushButton::clicked, this, &MainWindow::onNewMethod);
  connect(newProblemBtn, &QPushButton::clicked, this, &MainWindow::onNewProblem);

  auto* delMethodBtn = new QPushButton("Delete method…", wizardBox);
  delMethodBtn->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
  delMethodBtn->setIconSize(QSize(18, 18));
  delMethodBtn->setToolTip("Remove a method: deletes .h/.cpp, factory.cpp entry, CMakeLists.txt entry");

  auto* delProblemBtn = new QPushButton("Delete problem…", wizardBox);
  delProblemBtn->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
  delProblemBtn->setIconSize(QSize(18, 18));
  delProblemBtn->setToolTip("Remove a problem: deletes .h/.cpp, factory.cpp entry, CMakeLists.txt entry");

  wizLay->addWidget(delMethodBtn);
  wizLay->addWidget(delProblemBtn);

  connect(delMethodBtn,  &QPushButton::clicked, this, &MainWindow::onDeleteMethod);
  connect(delProblemBtn, &QPushButton::clicked, this, &MainWindow::onDeleteProblem);

  auto* openFileBtn = new QPushButton("Open file…", wizardBox);
  openFileBtn->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
  openFileBtn->setIconSize(QSize(18, 18));
  openFileBtn->setToolTip("Open any .h/.cpp file pair for editing and building");
  wizLay->addWidget(openFileBtn);
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

  // Bottom split: Output (left, wide) | Code Wizard (right, narrow)
  auto* bottomSplit = new QSplitter(Qt::Horizontal, central);
  bottomSplit_ = bottomSplit;
  bottomSplit->setChildrenCollapsible(false);
  bottomSplit->addWidget(outBox);
  bottomSplit->addWidget(wizardBox);
  bottomSplit->setStretchFactor(0, 6);
  bottomSplit->setStretchFactor(1, 1);
  bottomSplit->setSizes({900, 160});
  rootLay->addWidget(bottomSplit, 2);

  setCentralWidget(central);

  applyRegionStyling();
  updateRegionFocusStyling();

  statusBar()->showMessage("OptimSolution by OptimTeam");

  // connections
  connect(refreshBtn_, &QPushButton::clicked, this, &MainWindow::onRefreshFactory);
  connect(problemBox_, &QComboBox::currentTextChanged, this, &MainWindow::onProblemChanged);
  connect(methodBox_, &QComboBox::currentTextChanged, this, &MainWindow::onMethodChanged);
  connect(runBtn_, &QPushButton::clicked, this, &MainWindow::onRunClicked);
  connect(runModeBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onRunModeChanged);
  // Batch metric UI
  if (batchMetricCombo_) {
    connect(batchMetricCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onBatchMetricUiChanged);
  }
  if (batchAggCombo_) {
    connect(batchAggCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onBatchMetricUiChanged);
  }
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

    QFileInfoList summaryFiles;
    {
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
    if (useSummaryFiles && summaryFiles.size() < 2 && !selectedName.endsWith("_summary.csv")) {
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

    if (useSummaryFiles) {
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

        BatchCellData cell;
        QString loadErr;
        if (!loadBatchCellFromConvergence(convPath, cell, &loadErr)) {
          skippedLines << QString("Skipped %1: %2").arg(QFileInfo(convPath).fileName(), loadErr);
          continue;
        }

        const QString problemKey = batchProblemDisplayKey(problem, dim);
        loadedCells[problemKey][method] = cell;
        loadedCsvPaths[problemKey][method] = convPath;
        loadedDims[problemKey] = dim;
        methodSet.insert(method);
        problemSet.insert(problemKey);
        loadedProblemDimsByBase[batchBaseProblemShort(problemKey)].insert(dim);
        loadedLines << QString("Loaded batch cell: %1 + %2").arg(method, problemKey);
        ++loadedCount;
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

        BatchCellData cell;
        QString loadErr;
        if (!loadBatchCellFromConvergence(cfi.absoluteFilePath(), cell, &loadErr)) {
          skippedLines << QString("Skipped %1: %2").arg(cfi.fileName(), loadErr);
          continue;
        }

        const QString problemKey = batchProblemDisplayKey(problem, dim);
        loadedCells[problemKey][method] = cell;
        loadedCsvPaths[problemKey][method] = cfi.absoluteFilePath();
        loadedDims[problemKey] = dim;
        methodSet.insert(method);
        problemSet.insert(problemKey);
        loadedProblemDimsByBase[batchBaseProblemShort(problemKey)].insert(dim);
        loadedLines << QString("Loaded batch cell: %1 + %2").arg(method, problemKey);
        ++loadedCount;
      }
    }

    if (loadedCells.isEmpty()) {
      QMessageBox::warning(
        this,
        "Load experiment CSV",
        "No batch results could be reconstructed from the selected folder."
      );
      return false;
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
      g_batchSummarySnapshots.insert(summaryPage, snap);
      g_activeBatchSummaryPage = summaryPage;
    }

    lastRuntimeWorkingDir_ = dir.absolutePath();

    QStringList logLines;
    logLines << QString("Loaded external batch results from: %1").arg(dir.absolutePath());
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

  connect(loadExperimentCsvBtn_, &QPushButton::clicked, this, [this, loadExternalBatchFromDirectory]() {
    const QString startDir = !lastRuntimeWorkingDir_.trimmed().isEmpty()
      ? lastRuntimeWorkingDir_
      : (!projectRoot_.trimmed().isEmpty() ? projectRoot_ : QDir::currentPath());

    const QString csvPath = QFileDialog::getOpenFileName(
      this,
      "Load experiment CSV",
      startDir,
      "CSV files (*.csv);;All files (*.*)"
    );
    if (csvPath.isEmpty()) return;

    const QFileInfo fi(csvPath);
    if (loadExternalBatchFromDirectory(fi.absolutePath(), fi.absoluteFilePath())) {
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
    tryLoadSensitivityForTab(tabIndex);
    statusBar()->showMessage(QString("Loaded experiment CSV: %1").arg(fi.fileName()), 5000);
  });

  connect(reloadConvergenceBtn_, &QPushButton::clicked, this, &MainWindow::onReloadConvergence);
  connect(clearCsvBtn_, &QPushButton::clicked, this, &MainWindow::onClearCsvFiles);

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

  const QColor selAccent = clampSaturation(rotateHue(baseAccent, 0),   120);
  const QColor setAccent = clampSaturation(rotateHue(baseAccent, 120), 120);
  const QColor outAccent = clampSaturation(rotateHue(baseAccent, 240), 120);

  const QColor focusBorder = blend(win, hi, 0.38);

  const QString qss = QString(
R"(
QGroupBox#selectionBox { border: 1px solid palette(mid); border-left: 5px solid %1; border-radius: 8px; margin-top: 10px; background-color: %5; }
QGroupBox#settingsBox  { border: 1px solid palette(mid); border-left: 5px solid %2; border-radius: 8px; margin-top: 10px; background-color: %6; }
QGroupBox#outputBox    { border: 1px solid palette(mid); border-left: 5px solid %3; border-radius: 8px; margin-top: 10px; background-color: %7; }

QGroupBox#selectionBox::title, QGroupBox#settingsBox::title, QGroupBox#outputBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; font-weight: 600; }

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

  if (selectionSettingsSplitter_) {
    selectionSettingsSplitterSizes_ = selectionSettingsSplitter_->sizes();
  }

  focusedArea_ = area;

  if (area == FocusArea::Selection) {
    if (selectionSettingsSplitter_) selectionSettingsSplitter_->setVisible(true);
    if (selectionBox_) selectionBox_->setVisible(true);
    if (settingsBox_)  settingsBox_->setVisible(false);
    if (outputBox_)    outputBox_->setVisible(false);
    if (wizardBox_)    wizardBox_->setVisible(false);
    if (bottomSplit_)  bottomSplit_->setVisible(false);
  } else if (area == FocusArea::Settings) {
    if (selectionSettingsSplitter_) selectionSettingsSplitter_->setVisible(true);
    if (selectionBox_) selectionBox_->setVisible(false);
    if (settingsBox_)  settingsBox_->setVisible(true);
    if (outputBox_)    outputBox_->setVisible(false);
    if (wizardBox_)    wizardBox_->setVisible(false);
    if (bottomSplit_)  bottomSplit_->setVisible(false);
  } else if (area == FocusArea::Output) {
    if (selectionSettingsSplitter_) selectionSettingsSplitter_->setVisible(false);
    if (outputBox_)    outputBox_->setVisible(true);
    if (wizardBox_)    wizardBox_->setVisible(false);
    if (bottomSplit_)  bottomSplit_->setVisible(true);
  }

  updateRegionFocusStyling();
}

void MainWindow::restoreRegionLayout() {
  focusedArea_ = FocusArea::None;

  if (selectionSettingsSplitter_) selectionSettingsSplitter_->setVisible(true);
  if (selectionBox_) selectionBox_->setVisible(true);
  if (settingsBox_)  settingsBox_->setVisible(true);
  if (outputBox_)    outputBox_->setVisible(true);
  if (wizardBox_)    wizardBox_->setVisible(true);
  if (bottomSplit_)  bottomSplit_->setVisible(true);

  if (selectionSettingsSplitter_ && !selectionSettingsSplitterSizes_.isEmpty() && selectionSettingsSplitterSizes_.size() >= 2) {
    selectionSettingsSplitter_->setSizes(selectionSettingsSplitterSizes_);
  }

  updateRegionFocusStyling();
}

void MainWindow::updateRegionFocusStyling() {
  const bool hasFocus = (focusedArea_ != FocusArea::None);

  if (selectionBox_) selectionBox_->setProperty("focused", focusedArea_ == FocusArea::Selection);
  if (settingsBox_)  settingsBox_->setProperty("focused", focusedArea_ == FocusArea::Settings);
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
                    ? "Restore all three areas (Selection, Settings, Output)."
                    : "Expand this area and hide the other two.");
    refreshWidgetStyle(b);
  };

  updateBtn(selectionMaxBtn_, focusedArea_ == FocusArea::Selection);
  updateBtn(settingsMaxBtn_, focusedArea_ == FocusArea::Settings);
  updateBtn(outputMaxBtn_, focusedArea_ == FocusArea::Output);

  refreshWidgetStyle(selectionBox_);
  refreshWidgetStyle(settingsBox_);
  refreshWidgetStyle(outputBox_);
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
          display = QString::fromStdString(full) + " (" + shortName.toLower() + ")";
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

void MainWindow::updateDimUiForProblem(const QString& problem) {
  currentFixedDim_ = fixedDimForProblem(problem);

  if (currentFixedDim_ > 0) {
    dimSpin_->setEnabled(false);
    dimSpin_->setValue(currentFixedDim_);
  } else {
    dimSpin_->setEnabled(true);

    // In batch mode, keep the Selection-area dimension aligned with any
    // per-problem override that is already known (for example after
    // "Load experiment CSV" reconstructs batch cells with a specific dim).
    if (runModeBox_ && runModeBox_->currentData().toInt() == 1) {
      const int v = batchDimForProblem(problem.trimmed(), std::max(2, dimSpin_->value()));
      if (v > 0 && dimSpin_->value() != v) {
        dimSpin_->setValue(v);
      }
    }
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
  const QString txt = problemBox_ ? problemBox_->currentText().trimmed() : QString();
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

int MainWindow::createOutputRunTab(const QString& methodShort, const QString& problemShort, int problemDim) {
  if (!outputTabs_) return -1;

  const QString m = methodShort.trimmed();
  const QString p = problemShort.trimmed();
  const bool isSystemTab = (m.compare("System", Qt::CaseInsensitive) == 0);
  const int runMode = (runModeBox_ ? runModeBox_->currentData().toInt() : 0);
  const bool isBatchUi = (runMode == 1);
  const bool isSensitivityUi = (runMode == 2 || runMode == 3);

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
    wLay->addWidget(wilcoxonPlot_, 1);

    // Display options (Theme + Grid density) under the plot (mirrors Convergence controls).
    auto* wDispRow = new QHBoxLayout();
    wDispRow->addStretch(1);

    auto* wThemeLbl = new QLabel("Theme", wTab);
    auto* wThemeDarkRadio  = new QRadioButton("Dark", wTab);
    auto* wThemeLightRadio = new QRadioButton("Light", wTab);
    wThemeDarkRadio->setChecked(true);
    wThemeLightRadio->setChecked(false);

    auto* wThemeGroup = new QButtonGroup(wTab);
    wThemeGroup->setExclusive(true);
    wThemeGroup->addButton(wThemeDarkRadio);
    wThemeGroup->addButton(wThemeLightRadio);

    auto* wGridLbl = new QLabel("Grid", wTab);
    auto* wGridCombo = new QComboBox(wTab);
    wGridCombo->addItem("Sparse");
    wGridCombo->addItem("Medium");
    wGridCombo->addItem("Dense");
    wGridCombo->setCurrentIndex(1);

    connect(wThemeDarkRadio, &QRadioButton::toggled, this, [this](bool checked){
      if (!checked) return;
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        wp->setThemeMode(WilcoxonBoxPlotWidget::ThemeMode::Dark);
      }
    });
    connect(wThemeLightRadio, &QRadioButton::toggled, this, [this](bool checked){
      if (!checked) return;
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        wp->setThemeMode(WilcoxonBoxPlotWidget::ThemeMode::Light);
      }
    });
    connect(wGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* wp = dynamic_cast<WilcoxonBoxPlotWidget*>(wilcoxonPlot_)) {
        WilcoxonBoxPlotWidget::GridDensity gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
        if (index == 0) gd = WilcoxonBoxPlotWidget::GridDensity::Sparse;
        else if (index == 1) gd = WilcoxonBoxPlotWidget::GridDensity::Medium;
        else gd = WilcoxonBoxPlotWidget::GridDensity::Dense;
        wp->setGridDensity(gd);
      }
    });

    wDispRow->addWidget(wThemeLbl);
    wDispRow->addSpacing(8);
    wDispRow->addWidget(wThemeDarkRadio);
    wDispRow->addSpacing(10);
    wDispRow->addWidget(wThemeLightRadio);
    wDispRow->addSpacing(22);
    wDispRow->addWidget(wGridLbl);
    wDispRow->addSpacing(8);
    wDispRow->addWidget(wGridCombo);
    wDispRow->addStretch(1);
    wLay->addLayout(wDispRow);


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
    fLay->addWidget(rankPlot_, 1);

    // Display options (Theme + Grid density) under the plot (mirrors Convergence controls).
    auto* fDispRow = new QHBoxLayout();
    fDispRow->addStretch(1);

    auto* fThemeLbl = new QLabel("Theme", fTab);
    auto* fThemeDarkRadio  = new QRadioButton("Dark", fTab);
    auto* fThemeLightRadio = new QRadioButton("Light", fTab);
    fThemeDarkRadio->setChecked(true);
    fThemeLightRadio->setChecked(false);

    auto* fThemeGroup = new QButtonGroup(fTab);
    fThemeGroup->setExclusive(true);
    fThemeGroup->addButton(fThemeDarkRadio);
    fThemeGroup->addButton(fThemeLightRadio);

    auto* fGridLbl = new QLabel("Grid", fTab);
    auto* fGridCombo = new QComboBox(fTab);
    fGridCombo->addItem("Sparse");
    fGridCombo->addItem("Medium");
    fGridCombo->addItem("Dense");
    fGridCombo->setCurrentIndex(1);

    connect(fThemeDarkRadio, &QRadioButton::toggled, this, [this](bool checked){
      if (!checked) return;
      if (auto* rp = dynamic_cast<RankPlotWidget*>(rankPlot_)) {
        rp->setThemeMode(RankPlotWidget::ThemeMode::Dark);
      }
    });
    connect(fThemeLightRadio, &QRadioButton::toggled, this, [this](bool checked){
      if (!checked) return;
      if (auto* rp = dynamic_cast<RankPlotWidget*>(rankPlot_)) {
        rp->setThemeMode(RankPlotWidget::ThemeMode::Light);
      }
    });
    connect(fGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
      if (auto* rp = dynamic_cast<RankPlotWidget*>(rankPlot_)) {
        RankPlotWidget::GridDensity gd = RankPlotWidget::GridDensity::Medium;
        if (index == 0) gd = RankPlotWidget::GridDensity::Sparse;
        else if (index == 1) gd = RankPlotWidget::GridDensity::Medium;
        else gd = RankPlotWidget::GridDensity::Dense;
        rp->setGridDensity(gd);
      }
    });

    fDispRow->addWidget(fThemeLbl);
    fDispRow->addSpacing(8);
    fDispRow->addWidget(fThemeDarkRadio);
    fDispRow->addSpacing(10);
    fDispRow->addWidget(fThemeLightRadio);
    fDispRow->addSpacing(22);
    fDispRow->addWidget(fGridLbl);
    fDispRow->addSpacing(8);
    fDispRow->addWidget(fGridCombo);
    fDispRow->addStretch(1);
    fLay->addLayout(fDispRow);


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
    // Tabs 1+ start disabled until batch completes with ≥2 methods.
    updateStatsTabsEnabled();

    inner->addTab(statsTab, "Batch Analysis");
    inner->addTab(logTab, "Batch Log");
  }


  // In Batch mode, per-job output tabs are intentionally lightweight (log only).
  // Statistics are available only in the System tab.
  if (isBatchUi && !isSystemTab) {
    inner->addTab(logTab, "Job Log");
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

    outputTabs_->setTabToolTip(tabIndex, title);
    outputTabs_->setCurrentIndex(tabIndex);
    return tabIndex;
  }

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
  QComboBox*    distGridCombo       = nullptr;
  QPushButton*  exportPngBtn        = nullptr;
  QCheckBox*    bandChk             = nullptr;
  QLabel*       distInfo            = nullptr;
  QWidget*      distPlot            = nullptr;
  QPushButton*  exportDistBtn       = nullptr;
  QWidget*      convTab             = nullptr;
  QWidget*      distTab             = nullptr;

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

  distPlot = new DistributionPlotWidget(distTab);
  distLay->addWidget(distPlot, 1);

  // Display options (Theme + Grid density) under the plot (mirrors Convergence controls).
  auto* distDispRow = new QHBoxLayout();
  distDispRow->addStretch(1);

  auto* distThemeLbl = new QLabel("Theme", distTab);
  distThemeDarkRadio  = new QRadioButton("Dark", distTab);
  distThemeLightRadio = new QRadioButton("Light", distTab);
  distThemeDarkRadio->setChecked(true);
  distThemeLightRadio->setChecked(false);

  auto* distThemeGroup = new QButtonGroup(distTab);
  distThemeGroup->setExclusive(true);
  distThemeGroup->addButton(distThemeDarkRadio);
  distThemeGroup->addButton(distThemeLightRadio);

  auto* distGridLbl = new QLabel("Grid", distTab);
  distGridCombo = new QComboBox(distTab);
  distGridCombo->addItem("Sparse");
  distGridCombo->addItem("Medium");
  distGridCombo->addItem("Dense");
  distGridCombo->setCurrentIndex(1); // default: Medium

  // Mirror dist radio -> conv themeCombo (Dark=0, Light=1)
  connect(distThemeDarkRadio, &QRadioButton::toggled, this, [themeCombo](bool checked) {
    if (checked && themeCombo) themeCombo->setCurrentIndex(0);
  });
  connect(distThemeLightRadio, &QRadioButton::toggled, this, [themeCombo](bool checked) {
    if (checked && themeCombo) themeCombo->setCurrentIndex(1);
  });
  connect(distGridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [gridCombo](int idx) {
    if (gridCombo) gridCombo->setCurrentIndex(idx);
  });

  // Conv themeCombo -> dist radio mirror
  connect(themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [distThemeDarkRadio, distThemeLightRadio](int idx) {
    if (!distThemeDarkRadio || !distThemeLightRadio) return;
    QSignalBlocker b1(distThemeDarkRadio), b2(distThemeLightRadio);
    distThemeDarkRadio->setChecked(idx != 1);
    distThemeLightRadio->setChecked(idx == 1);
  });
  connect(gridCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [distGridCombo](int idx) {
    if (!distGridCombo) return;
    QSignalBlocker b(distGridCombo);
    distGridCombo->setCurrentIndex(idx);
  });

  distDispRow->addWidget(distThemeLbl);
  distDispRow->addSpacing(8);
  distDispRow->addWidget(distThemeDarkRadio);
  distDispRow->addSpacing(10);
  distDispRow->addWidget(distThemeLightRadio);
  distDispRow->addSpacing(22);
  distDispRow->addWidget(distGridLbl);
  distDispRow->addSpacing(8);
  distDispRow->addWidget(distGridCombo);
  distDispRow->addStretch(1);
  distLay->addLayout(distDispRow);


  auto* distHint = new QLabel("Boxplot: whiskers = 1.5×IQR, dots = outliers. Values are final best_f per run from the convergence CSV.", distTab);
  distHint->setWordWrap(true);
  distHint->setStyleSheet("opacity: 0.75;");
  distHint->hide();
  distLay->addWidget(distHint);

  } // end if (!isSensitivityUi)


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
    connect(exportTableBtn, &QPushButton::clicked, this, [this]() {
      auto* tab = activeOutputTab();
      if (!tab || !tab->sensitivitySummaryTable) return;

      // Build default filename: sensitivity_<param>_<problem>.xlsx
      const QString param   = (tab->sensitivityParamCombo && !tab->sensitivityParamCombo->currentText().isEmpty())
                                ? tab->sensitivityParamCombo->currentText().trimmed()
                                : "param";
      const QString problem = tab->problemShort.isEmpty() ? "problem" : tab->problemShort;
      const QString defName = QString("sensitivity_%1_%2.xlsx").arg(param, problem);

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

    resTop->addWidget(sensMetricLbl);
    resTop->addStretch(1);
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
    sensThemeDarkRadio  = new QRadioButton("Dark",  sensChartTab);
    sensThemeLightRadio = new QRadioButton("Light", sensChartTab);
    sensThemeDarkRadio->setChecked(true);
    auto* sensThemeGroup = new QButtonGroup(sensChartTab);
    sensThemeGroup->setExclusive(true);
    sensThemeGroup->addButton(sensThemeDarkRadio);
    sensThemeGroup->addButton(sensThemeLightRadio);
    auto* sensGridLbl = new QLabel("Grid", sensChartTab);
    sensGridCombo = new QComboBox(sensChartTab);
    sensGridCombo->addItem("Sparse");
    sensGridCombo->addItem("Medium");
    sensGridCombo->addItem("Dense");
    sensGridCombo->setCurrentIndex(1);

    connect(sensThemeDarkRadio, &QRadioButton::toggled, this,
            [sp = sensPlot](bool checked) {
              if (!checked) return;
              if (auto* w = dynamic_cast<SensitivityBarWidget*>(sp))
                w->setThemeMode(SensitivityBarWidget::ThemeMode::Dark);
            });
    connect(sensThemeLightRadio, &QRadioButton::toggled, this,
            [sp = sensPlot](bool checked) {
              if (!checked) return;
              if (auto* w = dynamic_cast<SensitivityBarWidget*>(sp))
                w->setThemeMode(SensitivityBarWidget::ThemeMode::Light);
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
    sensDispRow->addWidget(sensThemeDarkRadio);
    sensDispRow->addSpacing(10);
    sensDispRow->addWidget(sensThemeLightRadio);
    sensDispRow->addSpacing(22);
    sensDispRow->addWidget(sensGridLbl);
    sensDispRow->addSpacing(8);
    sensDispRow->addWidget(sensGridCombo);
    sensDispRow->addStretch(1);
    sensChartLay->addLayout(sensDispRow);

    // Assemble inner tabs for Sensitivity run
    inner->addTab(resTab,       "Results table");
    inner->addTab(sensChartTab, "Parameter Sensitivity");
    inner->addTab(logTab,       "Run Log");

  } else {
    // Single run: Convergence + Final Distribution + Run Log (no sensitivity tab)
    inner->addTab(convTab,  "Convergence");
    inner->addTab(distTab,  "Final Distribution");
    inner->addTab(logTab,   "Run Log");
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
  run.convLoaded = false;
  outputRuns_.push_back(std::move(run));

  return tabIndex;
}

MainWindow::OutputRunTab* MainWindow::outputTab(int index) {
  if (index < 0 || index >= int(outputRuns_.size())) return nullptr;
  return &outputRuns_[index];
}


void MainWindow::renderSensitivityForParam(int tabIndex, const QString& paramName) {
  if (tabIndex < 0 || tabIndex >= static_cast<int>(outputRuns_.size())) return;
  OutputRunTab& tab = outputRuns_[tabIndex];

  if (!tab.sensitivityLog || !tab.sensitivityParamCombo || !tab.sensitivityPlot || !tab.sensitivitySummaryTable) return;

  auto* plot = dynamic_cast<SensitivityBarWidget*>(tab.sensitivityPlot);
  if (plot) plot->clear();
  tab.sensitivitySummaryTable->setRowCount(0);

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

  // ---- Populate extended summary table (8 columns) ----
  // Determine which optional columns actually have data.
  const bool hasSd  = std::any_of(pts.begin(), pts.end(), [](const auto& p){ return p.sd > 0.0; });
  const bool hasMin = std::any_of(pts.begin(), pts.end(), [](const auto& p){ return !std::isnan(p.minV); });
  const bool hasMax = std::any_of(pts.begin(), pts.end(), [](const auto& p){ return !std::isnan(p.maxV); });

  // Build column descriptor list — only include columns with data.
  struct ColDef { QString header; };
  QVector<ColDef> cols;
  cols.push_back({"Value"});
  cols.push_back({"Mean best_f"});
  if (hasSd)  cols.push_back({"SD"});
  if (hasMin) cols.push_back({"Min best_f"});
  if (hasMax) cols.push_back({"Max best_f"});
  cols.push_back({"Main effect"});
  cols.push_back({"Rank"});
  cols.push_back({"N (runs)"});

  tab.sensitivitySummaryTable->setColumnCount(cols.size());
  QStringList colHeaders;
  for (const auto& c : cols) colHeaders << c.header;
  tab.sensitivitySummaryTable->setHorizontalHeaderLabels(colHeaders);
  // Pre-compute global stats.
  double globalMinMean =  std::numeric_limits<double>::max();
  double globalMaxMean = -std::numeric_limits<double>::max();
  for (const auto& p : pts) {
    globalMinMean = std::min(globalMinMean, p.mean);
    globalMaxMean = std::max(globalMaxMean, p.mean);
  }
  const double mainEffectRange = (pts.size() > 1) ? (globalMaxMean - globalMinMean) : 0.0;

  // Rank: 1 = smallest mean (best for minimisation).
  QVector<int> order(pts.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b){ return pts[a].mean < pts[b].mean; });
  QVector<int> rank(pts.size());
  for (int r = 0; r < order.size(); ++r) rank[order[r]] = r + 1;

  auto fmtVal = [](double v) -> QString { return QString::number(v, 'g', 8); };

  // Map column header -> index for safe lookup.
  auto colIdx = [&](const QString& name) -> int {
    for (int i = 0; i < cols.size(); ++i) if (cols[i].header == name) return i;
    return -1;
  };
  const int cValue  = colIdx("Value");
  const int cMean   = colIdx("Mean best_f");
  const int cSd     = colIdx("SD");
  const int cMin    = colIdx("Min best_f");
  const int cMax    = colIdx("Max best_f");
  const int cEffect = colIdx("Main effect");
  const int cRank   = colIdx("Rank");
  const int cN      = colIdx("N (runs)");

  tab.sensitivitySummaryTable->setRowCount(pts.size() + 1); // +1 summary footer
  for (int i = 0; i < pts.size(); ++i) {
    const auto& sp = pts[i];
    const bool isBest = (rank[i] == 1);

    auto makeItem = [&](const QString& text, bool bold = false) {
      auto* item = new QTableWidgetItem(text);
      item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
      if (bold) { QFont f = item->font(); f.setBold(true); item->setFont(f); }
      return item;
    };
    auto setCol = [&](int c, const QString& text) {
      if (c >= 0) tab.sensitivitySummaryTable->setItem(i, c, makeItem(text, isBest));
    };

    setCol(cValue,  fmtVal(sp.x));
    setCol(cMean,   fmtVal(sp.mean));
    if (cSd  >= 0) setCol(cSd,  fmtVal(sp.sd));
    if (cMin >= 0) setCol(cMin, fmtVal(sp.minV));
    if (cMax >= 0) setCol(cMax, fmtVal(sp.maxV));
    setCol(cEffect, fmtVal(mainEffectRange));
    setCol(cRank,   QString::number(rank[i]));
    setCol(cN,      QString::number(sp.n));
  }

  // Footer row: summary statistics.
  {
    const int fi = pts.size();
    auto summaryItem = [&](const QString& text, bool leftAlign = false) {
      auto* item = new QTableWidgetItem(text);
      item->setTextAlignment((leftAlign ? Qt::AlignLeft : Qt::AlignRight) | Qt::AlignVCenter);
      QFont f = item->font(); f.setItalic(true); item->setFont(f);
      return item;
    };
    tab.sensitivitySummaryTable->setItem(fi, 0, summaryItem("↓ Summary", true));
    if (cMean >= 0)
      tab.sensitivitySummaryTable->setItem(fi, cMean,
        summaryItem(QString("min=%1  max=%2").arg(fmtVal(globalMinMean), fmtVal(globalMaxMean))));
    if (cEffect >= 0)
      tab.sensitivitySummaryTable->setItem(fi, cEffect, summaryItem(fmtVal(mainEffectRange)));
    if (cRank >= 0)
      tab.sensitivitySummaryTable->setItem(fi, cRank,
        summaryItem(QString("%1 values").arg(pts.size())));
  }

  tab.sensitivitySummaryTable->resizeColumnsToContents();

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
          }
          if (batchAggCombo_) {
            const int aggIdx = batchAggCombo_->findText(snap.aggText);
            if (aggIdx >= 0 && batchAggCombo_->currentIndex() != aggIdx) {
              QSignalBlocker blocker(batchAggCombo_);
              batchAggCombo_->setCurrentIndex(aggIdx);
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

void MainWindow::tryLoadConvergenceForTab(int index) {
  auto* tab = outputTab(index);
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

void MainWindow::onRunClicked() {
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
        const QString prob = probItem->text().trimmed();
        if (prob.isEmpty()) continue;
        QVector<int> dims;
        const int fixed = fixedDimForProblem(prob);
        if (fixed > 0) {
          dims = {fixed};
        } else if (auto* w = sensProblemDimsTable_->cellWidget(r, 1)) {
          if (auto* le = qobject_cast<QLineEdit*>(w))
            dims = parseBatchDimensionList(le->text(), defaultDim, 2, maxDim);
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
    runBtn_->setText(mode == 1 ? "Stop batch" : mode == 2 || mode == 3 ? "Stop" : "Stop");
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
    if (batchRunsSpin_) batchRunsSpin_->setValue(globalRunsFromSettings());
    onBatchMetricUiChanged();
  }
  if (mode == 2 || mode == 3) {
    populateBatchLists(); // also populates sensProblemsList_
  }

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

  updateBatchPanelVisibility();
}

void MainWindow::updateBatchPanelVisibility() {
  const int mode = runModeBox_ ? runModeBox_->currentData().toInt() : 0;
  const bool isBatch = (mode == 1);
  const bool isSensitivity = (mode == 2);
  const bool isProblemSensitivity = (mode == 3);
  const bool isAnySensitivity = (isSensitivity || isProblemSensitivity);

  if (batchPanel_) batchPanel_->setVisible(isBatch);
  if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setVisible(isBatch);
  if (sensProblemPanel_) sensProblemPanel_->setVisible(isSensitivity);  // only mode 2; mode 3 uses main dropdown
  if (settingsBox_) {
    if (auto* row = settingsBox_->findChild<QWidget*>("batchMethodSettingsRow")) {
      row->setVisible(isBatch);
    }
  }
  if (auto* btn = findChild<QPushButton*>("saveBatchSelectionBtn")) {
    btn->setVisible(isBatch);
  }
  if (auto* btn = findChild<QPushButton*>("loadBatchSelectionBtn")) {
    btn->setVisible(isBatch);
  }

  // In sensitivity mode: method stays visible, problem/dim are hidden (panel used instead).
  // In problem sensitivity mode: both method and problem stay visible (sweep is on problem params).
  // In batch mode: all single-run dropdowns are hidden.
  const bool showMethod   = !isBatch;
  const bool showProbDim  = !isBatch && !isSensitivity;  // mode 3 keeps problem/dim visible

  if (methodBox_) methodBox_->setVisible(showMethod);
  if (problemBox_) problemBox_->setVisible(showProbDim);
  if (dimSpin_) dimSpin_->setVisible(showProbDim);

  if (selectionForm_) {
    if (auto* lbl = selectionForm_->labelForField(methodBox_)) lbl->setVisible(showMethod);
    if (auto* lbl = selectionForm_->labelForField(problemBox_)) lbl->setVisible(showProbDim);
    if (auto* lbl = selectionForm_->labelForField(dimSpin_)) lbl->setVisible(showProbDim);
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
    const QString display = shortName.isEmpty() ? problemBox_->itemText(i) : shortName;
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
      const QString display   = shortName.isEmpty() ? problemBox_->itemText(i) : shortName;
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

    auto* probItem = new QTableWidgetItem(p);
    probItem->setFlags(probItem->flags() & ~Qt::ItemIsEditable);
    batchProblemDimsTable_->setItem(r, 0, probItem);

    const int fixed = fixedDimForProblem(p);
    if (fixed > 0) {
      auto* dimItem = new QTableWidgetItem(QString::number(fixed));
      dimItem->setFlags(dimItem->flags() & ~Qt::ItemIsEditable);
      batchProblemDimsTable_->setItem(r, 1, dimItem);
      continue;
    }

    QVector<int> dims;
    if (batchProblemDimOverride_.contains(p)) {
      dims = parseBatchDimensionList(batchProblemDimOverride_.value(p).join(", "), defaultDim, 2, maxDim);
    } else {
      dims = QVector<int>{defaultDim};
    }

    auto* line = new QLineEdit(batchProblemDimsTable_);
    line->setPlaceholderText("e.g. 10, 30");
    line->setMinimumWidth(120);
    line->setText(batchDimensionListToText(dims));

    connect(line, &QLineEdit::editingFinished, this, [this, line, p, defaultDim, maxDim]() {
      const QVector<int> dims = parseBatchDimensionList(line->text(), defaultDim, 2, maxDim);
      const QString canonicalText = batchDimensionListToText(dims);
      {
        QSignalBlocker blocker(line);
        line->setText(canonicalText);
      }

      if (dims.size() == 1 && dims.front() == defaultDim) {
        batchProblemDimOverride_.remove(p);
      } else {
        QStringList parts;
        for (int d : dims) parts << QString::number(d);
        batchProblemDimOverride_[p] = parts;
      }

      invalidateBatchProblemCache(p);
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
    auto* probItem = new QTableWidgetItem(p);
    probItem->setFlags(probItem->flags() & ~Qt::ItemIsEditable);
    sensProblemDimsTable_->setItem(r, 0, probItem);

    const int fixed = fixedDimForProblem(p);
    if (fixed > 0) {
      auto* dimItem = new QTableWidgetItem(QString::number(fixed));
      dimItem->setFlags(dimItem->flags() & ~Qt::ItemIsEditable);
      sensProblemDimsTable_->setItem(r, 1, dimItem);
      continue;
    }

    auto* line = new QLineEdit(sensProblemDimsTable_);
    line->setPlaceholderText("e.g. 10, 30");
    line->setMinimumWidth(120);
    line->setText(QString::number(defaultDim));
    connect(line, &QLineEdit::editingFinished, this, [this, line, defaultDim, maxDim]() {
      const QVector<int> dims = parseBatchDimensionList(line->text(), defaultDim, 2, maxDim);
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
  wilcoxonPairsCombo_ = ui.wilcoxonPairsCombo;
  statsAlphaCombo_ = ui.statsAlphaCombo;
  exportWilcoxonPlotBtn_ = ui.exportWilcoxonPlotBtn;
  wilcoxonPlot_ = ui.wilcoxonPlot;
  wilcoxonSummaryLbl_ = ui.wilcoxonSummaryLbl;
  exportRankPlotBtn_ = ui.exportRankPlotBtn;
  rankPlot_ = ui.rankPlot;
  statsSummaryLbl_ = ui.statsSummaryLbl;
  statsPairwiseTable_ = ui.statsPairwiseTable;
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

  QStringList methods = selectedBatchMethodShortNames();
  QStringList problems = selectedBatchProblemShortNames();

  if (g_activeBatchSummaryPage && g_batchSummarySnapshots.contains(g_activeBatchSummaryPage)) {
    const BatchSummarySnapshot existing = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
    if (methods.isEmpty()) methods = existing.methods;
    if (problems.isEmpty()) problems = existing.problems;
    batchCells_ = batchSummaryCellsByPage_.value(g_activeBatchSummaryPage);
    batchCachedProblemDims_ = existing.cachedProblemDims;
    batchProblemDimOverride_ = existing.problemDimOverrides;
    g_liveBatchCsvPaths = existing.csvPaths;

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
  }

  if (g_activeBatchSummaryPage && !internalUpdate_) {
    BatchSummarySnapshot snap = g_batchSummarySnapshots.value(g_activeBatchSummaryPage);
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

  // Reload settings from disk to pick up any external edits (e.g. runs value).
  if (cfg_ && !settingsPath_.isEmpty()) cfg_->load(settingsPath_);

  // Runs are taken from [global].runs (mirrored read-only in the batch panel).
  const int runs = globalRunsFromSettings();
  if (batchRunsSpin_) {
    batchRunsSpin_->setValue(runs);
  }

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

  if (summaryPage) {
    g_activeBatchSummaryPage = summaryPage;
    bindBatchSummaryUiForPage(summaryPage);
    BatchSummarySnapshot snap;
    snap.methods = methods;
    snap.problems = problems;
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

  initStatsTableForBatch(methods, problems);
  for (const QString& prob : problems) {
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
  if (auto* w = findChild<QGroupBox*>("wizardBox")) w->setEnabled(true);
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

  updateDimUiForProblem(currentProblemShort());

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
  // Used by the Wilcoxon box plot.
  const BatchMetricMode mode = currentBatchMetricMode();
  const QString agg = (batchAggCombo_ ? batchAggCombo_->currentText() : QString());
  if (mode == BatchMetricMode::IterationAtBest || mode == BatchMetricMode::EvalsAtBest) {
    // Aggregation matters only for time-to-best metrics.
    const QString base = (mode == BatchMetricMode::IterationAtBest) ? "Iterations to best" : "Function evaluations to best";
    if (!agg.isEmpty()) {
      return base + QString(" (%1)").arg(agg);
    }
    return base;
  }
  return "final best_f";
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
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    if (err) *err = QString("Cannot open file '%1'.").arg(csvPath);
    return false;
  }

  QTextStream ts(&f);
  const QString headerLine = ts.readLine();
  if (headerLine.isEmpty()) {
    if (err) *err = "Empty CSV file.";
    return false;
  }

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

  while (!ts.atEnd()) {
    const QString line = ts.readLine().trimmed();
    if (line.isEmpty()) continue;
    const QStringList parts = line.split(',', Qt::KeepEmptyParts);
    if (parts.size() <= std::max(std::max(cRun, cIter), std::max(cEvals, cBest))) continue;

    bool okRun = false, okIter = false, okEval = false, okBest = false;
    const int run = parts[cRun].toInt(&okRun);
    const int itv = parts[cIter].toInt(&okIter);
    const qint64 ev = parts[cEvals].toLongLong(&okEval);
    const double bf = parts[cBest].toDouble(&okBest);
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

  const QString valueText = hasComputedCell ? fmt(value) : QString();
  const QString meanText = (hasComputedCell && std::isfinite(mean)) ? QString::number(mean, 'g', 15) : QString();
  const QString rateText = hasComputedCell ? QString::number(int(std::llround(rate))) : QString();
  const QString sdText = hasComputedCell ? fmtSd(sd) : QString();
  const QString timeText = (hasComputedCell && std::isfinite(timePerRun))
                             ? QString::number(timePerRun, 'f', 3)
                             : QString();

  statsTable_->setItem(row, colBase, mkItem(valueText));

  int c = colBase + 1;
  if (statsShowMean) {
    statsTable_->setItem(row, c++, mkItem(meanText));
  }
  if (statsShowRate_) {
    statsTable_->setItem(row, c++, mkItem(rateText));
  }
  if (statsShowSd_) {
    statsTable_->setItem(row, c++, mkItem(sdText));
  }
  if (statsShowTime_) {
    statsTable_->setItem(row, c++, mkItem(timeText));
  }

  exportStatsBtn_->setEnabled(!batchCells_.isEmpty());
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

      statsTable_->setItem(r, c++, mkItem(valueText));
      if (showMean) {
        statsTable_->setItem(r, c++, mkItem(meanText));
      }
      if (showRate) {
        statsTable_->setItem(r, c++, mkItem(rateText));
      }
      if (showSd) {
        statsTable_->setItem(r, c++, mkItem(sdText));
      }
      if (showTime) {
        statsTable_->setItem(r, c++, mkItem(timeText));
      }
    }
  }

  statsTable_->resizeColumnsToContents();
  exportStatsBtn_->setEnabled(!batchCells_.isEmpty());
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
  const double chi2 = (12.0 * double(N) / (double(k) * double(k + 1))) * sumR2 - 3.0 * double(N) * double(k + 1);
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

  // Fill pairwise table.
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
    double bestMean = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
      double m = 0.0;
      for (double v : groups[j]) m += v;
      m /= double(groups[j].size());
      if (m < bestMean) { bestMean = m; bestIdx = j; }
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

    if (cliSupportsConfigArg_ == 1) {
      runtimeCfgPath = QDir(baseWd).filePath("optimsolution_gui_merged.cfg");
      applyForceCsv();
      if (!isSensitivityMode) writeSensitivityConfigFromUi();
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
    // Disable entire Selection and Settings areas.
    if (selectionBox_) selectionBox_->setEnabled(false);
    if (auto* w = findChild<QGroupBox*>("wizardBox")) w->setEnabled(false);
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
      singleRunsPerJob_ = (batchRunsSpin_ && batchRunsSpin_->value() > 0)
                          ? batchRunsSpin_->value()
                          : globalRunsFromSettings();
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
  if (auto* w = findChild<QGroupBox*>("wizardBox")) w->setEnabled(true);
    if (settingsBox_) settingsBox_->setEnabled(true);
    if (loadExperimentCsvBtn_) loadExperimentCsvBtn_->setEnabled(true);
    if (reloadConvergenceBtn_) reloadConvergenceBtn_->setEnabled(true);
    if (clearCsvBtn_) clearCsvBtn_->setEnabled(true);
    updateDimUiForProblem(problemBox_->currentText().trimmed());

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
    if (!inSensQueue) tryLoadSensitivityForTab(activeOutputRunIndex_);
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
  if (auto* splot = dynamic_cast<SensitivityBarWidget*>(tab->sensitivityPlot)) {
    splot->setThemeMode(darkTheme ? SensitivityBarWidget::ThemeMode::Dark
                                  : SensitivityBarWidget::ThemeMode::Light);
    if (tab->gridDensityCombo) {
      const int idx = tab->gridDensityCombo->currentIndex();
      const auto gd = (idx <= 0) ? SensitivityBarWidget::GridDensity::Sparse
                                 : (idx == 1 ? SensitivityBarWidget::GridDensity::Medium
                                             : SensitivityBarWidget::GridDensity::Dense);
      splot->setGridDensity(gd);
    }
  }

  if (auto* dplot = dynamic_cast<DistributionPlotWidget*>(tab->distributionPlot)) {
    dplot->setThemeMode(darkTheme ? DistributionPlotWidget::ThemeMode::Dark
                                  : DistributionPlotWidget::ThemeMode::Light);
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
  plot->setXAxisTitle(useEvalX ? "Function Evaluations" : "Iterations");

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
      // Build an X grid that is dense for iterations and capped for evaluations.
      double maxX = 0.0;
      for (const auto& r : runs) if (!r.x.isEmpty()) maxX = std::max(maxX, r.x.back());
      if (maxX <= 0.0) maxX = 1.0;

      const int maxPoints = 2500;
      long long maxStep = (long long)std::llround(maxX);
      long long stride = 1;
      if (maxStep > maxPoints) stride = (long long)std::ceil(double(maxStep) / double(maxPoints));
      if (stride < 1) stride = 1;

      QVector<double> xGrid;
      xGrid.reserve(int(maxStep / stride) + 2);
      for (long long s = 0; s <= maxStep; s += stride) xGrid.push_back(double(s));
      if (xGrid.isEmpty() || xGrid.back() < double(maxStep)) xGrid.push_back(double(maxStep));

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

      plot->setXAxisTitle(useEvalX ? "Function Evaluations" : "Iterations");
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

  // Apply display options (Theme + Grid density), matching the Convergence tab controls.
  const bool darkTheme = (!tab->themeCombo || tab->themeCombo->currentIndex() != 1);
  plot->setThemeMode(darkTheme ? DistributionPlotWidget::ThemeMode::Dark
                               : DistributionPlotWidget::ThemeMode::Light);
  if (tab->gridDensityCombo) {
    const int idx = tab->gridDensityCombo->currentIndex();
    const auto gd = (idx <= 0) ? DistributionPlotWidget::GridDensity::Sparse
                               : (idx == 1 ? DistributionPlotWidget::GridDensity::Medium
                                           : DistributionPlotWidget::GridDensity::Dense);
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
    if (outputRuns_[i].themeCombo == s || outputRuns_[i].gridDensityCombo == s) {
      updateConvergencePlotForTab(i);
      break;
    }
  }
}

void MainWindow::onConvergenceGridChanged(int /*index*/) {
  QObject* s = sender();
  for (int i = 0; i < int(outputRuns_.size()); ++i) {
    if (outputRuns_[i].gridDensityCombo == s) {
      updateConvergencePlotForTab(i);
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

  QStringList files;
  QSet<QString> seen;
  for (const QString& root : rootList) {
    QDir d(root);
    if (!d.exists()) continue;
    QDirIterator it(root, QStringList() << "*.csv", QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
      const QString path = it.next();
      const QString ap = QFileInfo(path).absoluteFilePath();
      if (seen.contains(ap)) continue;
      seen.insert(ap);
      files.push_back(ap);
    }
  }

  if (files.isEmpty()) {
    QMessageBox::information(this, "Delete CSV files", "No CSV files were found in the optimsolution_gui_run folder.");
    return;
  }

  const QString prompt = QString(
    "This will delete %1 CSV file(s) under the following optimsolution_gui_run folder(s):\n\n%2\n\nContinue?"
  ).arg(files.size()).arg(rootList.join("\n"));

  const auto btn = QMessageBox::question(this, "Delete CSV files", prompt, QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (btn != QMessageBox::Yes) return;

  int removed = 0;
  int failed = 0;
  for (const QString& fp : files) {
    if (QFile::remove(fp)) removed++;
    else failed++;
  }

  const QString msg = failed > 0
    ? QString("Deleted %1 CSV file(s) from optimsolution_gui_run (%2 failed).")
        .arg(removed)
        .arg(failed)
    : QString("Deleted %1 CSV file(s) from optimsolution_gui_run.")
        .arg(removed);

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
    if (run.convergenceInfo) run.convergenceInfo->setText("CSV files deleted — run again to reload.");
    updateDistributionPlotForTab(i);
  }

  // Refresh plots for the active output tab.
  if (outputTabs_) {
    const int idx = outputTabs_->currentIndex();
    tryLoadConvergenceForTab(idx);
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

  const QString base = tab->title.isEmpty() ? QString("distribution") : tab->title;
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

  const QString base = tab->title.isEmpty() ? QString("sensitivity") : tab->title;
  const QString defName = base + "_sensitivity.png";
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


static bool parseConvergenceCsvFile(const QString& path,
                                   QVector<double>& outIterX,
                                   QVector<double>& outEvalX,
                                   QVector<double>& outY,
                                   QString& outInfo,
                                   bool* outHasEvalX) {
  if (outHasEvalX) *outHasEvalX = false;

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
  const QStringList fields = splitLine(firstNonEmpty, delim);

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

  int iterCol = -1;
  int evalCol = -1;
  int yCol = 1;
  QStringList header;

  auto findCol = [&](const QStringList& h, const QStringList& needles) -> int {
    for (int i = 0; i < h.size(); ++i) {
      const QString key = h[i].trimmed().toLower();
      for (const QString& n : needles) {
        if (key.contains(n)) return i;
      }
    }
    return -1;
  };

  bool hasEval = false;

  if (hasHeader) {
    header = fields;
    iterCol = findCol(header, {"iter", "iteration", "gen", "step"});
    evalCol = findCol(header, {"eval", "fe", "nfe", "calls"});
    const int bestCol = findCol(header, {"best", "fbest", "bestf", "best_f", "gbest"});

    hasEval = (evalCol >= 0);
    // Choose Y column.
    if (bestCol >= 0) {
      yCol = bestCol;
    } else {
      // Fallback: first column that is not iter/eval.
      yCol = -1;
      for (int i = 0; i < header.size(); ++i) {
        if (i == iterCol || i == evalCol) continue;
        yCol = i;
        break;
      }
      if (yCol < 0) yCol = std::min(1, int(header.size()) - 1);
    }
  } else {
    // No header: assume x,y in first two columns. Treat x as iterations.
    iterCol = 0;
    evalCol = -1;
    hasEval = false;
    yCol = std::min(1, int(fields.size()) - 1);
  }

  if (outHasEvalX) *outHasEvalX = hasEval;

  // Parse helper
  auto parseDataLine = [&](const QString& ln) {
    const QString t = ln.trimmed();
    if (t.isEmpty()) return;
    if (t.startsWith('#') || t.startsWith(';')) return;
    const QStringList cols = splitLine(ln, delim);
    if (cols.size() <= yCol) return;

    bool okY = false;
    const double y = cols[yCol].trimmed().toDouble(&okY);
    if (!okY) return;

    double iterX = double(outY.size() + 1);
    double evalX = 0.0;

    if (iterCol >= 0) {
      if (cols.size() <= iterCol) return;
      bool okI = false;
      iterX = cols[iterCol].trimmed().toDouble(&okI);
      if (!okI) return;
    }

    if (hasEval) {
      if (cols.size() <= evalCol) return;
      bool okE = false;
      evalX = cols[evalCol].trimmed().toDouble(&okE);
      if (!okE) return;
    }

    outIterX.push_back(iterX);
    if (hasEval) outEvalX.push_back(evalX);
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

  const int n = outY.size();
  if (n < 2 || outIterX.size() < 2) {
    outIterX.clear();
    outEvalX.clear();
    outY.clear();
    outInfo = QString("No usable convergence data in: %1").arg(QFileInfo(path).fileName());
    return false;
  }

  // Downsample for rendering if extremely large.
  const int maxPts = 6000;
  if (n > maxPts) {
    QVector<double> di;
    QVector<double> de;
    QVector<double> dy;
    di.reserve(maxPts);
    if (hasEval) de.reserve(maxPts);
    dy.reserve(maxPts);
    const int stride = int(std::ceil(double(n) / maxPts));
    for (int i = 0; i < n; i += stride) {
      di.push_back(outIterX[i]);
      if (hasEval) de.push_back(outEvalX[i]);
      dy.push_back(outY[i]);
    }
    outIterX = std::move(di);
    if (hasEval) outEvalX = std::move(de);
    else outEvalX.clear();
    outY = std::move(dy);
  }

  const QString xInfo = hasEval ? "Iterations + Function Evaluations" : "Iterations";
  outInfo = QString("Loaded %1 points from %2 (x: %3)")
                .arg(outY.size())
                .arg(QFileInfo(path).fileName())
                .arg(xInfo);
  return true;
}

} // namespace optimsolution_gui
