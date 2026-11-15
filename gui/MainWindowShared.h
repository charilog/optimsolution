// MainWindowShared.h
// Shared includes, file-scope helper functions, small widget classes and
// global UI state used by the MainWindow implementation, which is split
// across several translation units:
//   MainWindow_Ui.cpp              - constructor, buildUi(), styling/regions
//   MainWindow_Tabs.cpp            - factory lists, parameter tables, output tabs
//   MainWindow_RunControl.cpp      - run/stop control, batch selection, sensitivity jobs
//   MainWindow_BatchStats.cpp      - batch execution, statistics tables, Wilcoxon/Friedman, XLSX export
//   MainWindow_ProcessSettings.cpp - CLI process I/O, logging, settings tables, sensitivity config
//   MainWindow_Plots.cpp           - convergence/distribution plots and PNG exports
//
// Former file-scope `static` helpers are now `inline` so a single shared
// definition serves all translation units; mutable globals use C++17
// inline variables (one instance program-wide).
#pragma once

#include "MainWindow.h"
#include "PathUtils.h"
#include "ConfigFile.h"
#include "CodeGenDialog.h"
#include "factory.h"
#include "fixed_dims.h"
#include "allowed_dims.h"
#include "AnsiStrip.h"
#include "CrashLog.h"
#include "BusySpinner.h"

#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QApplication>
#include <QComboBox>
#include <QSlider>
#include <QSpinBox>
#include <QTextEdit>
#include <QTextDocument>
#include <QColor>
#include <QBrush>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextDocumentWriter>
#include <QBuffer>
#include <QPrinter>
#include <QPageSize>
#include <QPageLayout>
#include <QMargins>
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
#include <QLineEdit>
#include <QDesktopServices>
#include <QUrl>
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
  QPushButton* generateReportBtn = nullptr;
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
  // Critical Difference (Nemenyi) tab
  QPushButton* exportCdPlotBtn = nullptr;
  QWidget* cdPlot = nullptr;
  QLabel* cdSummaryLbl = nullptr;
  // Metric + Aggregate combos (primary = Wilcoxon panel, secondary = Friedman panel)
  QComboBox* metricCombo  = nullptr;  // primary → bound to batchMetricCombo_
  QComboBox* aggCombo     = nullptr;  // primary → bound to batchAggCombo_
  QComboBox* metricCombo2 = nullptr;  // secondary (Friedman), kept in sync
  QComboBox* aggCombo2    = nullptr;  // secondary (Friedman), kept in sync
};

struct BatchSummarySnapshot {
  // methods/problems = what the Output tabs of this batch summary currently
  // SHOW, i.e. the Selection-area choice. Deselecting a method or problem in
  // the Selection area after a batch hides it from every Output tab
  // (ablation study); selecting it again shows it again.
  QStringList methods;
  QStringList problems;
  // allMethods/allProblems = EVERY method/problem this experiment contains,
  // in the experiment's own order. Never shrunk by deselection, so that
  //  - a re-selected method/problem returns to its original column/row
  //    position instead of being appended at the end, and
  //  - the experiment manifest (*.osexp) keeps the hidden cells too
  //    (see finishBatchPostProcess()).
  // Empty in snapshots that never got them set: callers fall back to
  // methods/problems (see batchExperimentMethods()/batchExperimentProblems()).
  QStringList allMethods;
  QStringList allProblems;
  QMap<QString, QMap<QString, QString>> csvPaths;
  QMap<QString, int> cachedProblemDims;
  QMap<QString, QStringList> problemDimOverrides;
  int metricComboIndex = -1;
  QString aggText;
  bool showMean = true;
  bool showRate = true;
  bool showSd = true;
  bool showTime = true;
  // Experiment manifest (*.osexp) of this batch summary page: the exact list
  // of (method, problem, dim, convergence CSV) cells that make up the
  // experiment. Empty = not yet written (see writeBatchExperimentManifest()).
  QString manifestPath;
};

// Full experiment lists of a snapshot (see BatchSummarySnapshot::allMethods).
inline QStringList batchExperimentMethods(const BatchSummarySnapshot& s) {
  return s.allMethods.isEmpty() ? s.methods : s.allMethods;
}
inline QStringList batchExperimentProblems(const BatchSummarySnapshot& s) {
  return s.allProblems.isEmpty() ? s.problems : s.allProblems;
}

// Ordered union: every entry of 'base' (kept in its order), then the entries
// of 'extra' that 'base' does not have yet (in extra's order). No duplicates.
inline QStringList batchOrderedUnion(const QStringList& base, const QStringList& extra) {
  QStringList out;
  QSet<QString> seen;
  for (const QString& s : base) {
    if (!seen.contains(s)) { seen.insert(s); out << s; }
  }
  for (const QString& s : extra) {
    if (!seen.contains(s)) { seen.insert(s); out << s; }
  }
  return out;
}

// Puts 'wanted' in the order of 'reference': first the entries of 'wanted'
// that appear in 'reference' (in reference's order), then the remaining
// entries of 'wanted' in their own order. Used so that a method/problem that
// is hidden and then shown again returns to its original column/row.
inline QStringList batchOrderLike(const QStringList& wanted, const QStringList& reference) {
  if (reference.isEmpty() || wanted.isEmpty()) return wanted;
  const QSet<QString> wantedSet(wanted.begin(), wanted.end());
  QStringList out;
  QSet<QString> used;
  for (const QString& r : reference) {
    if (wantedSet.contains(r) && !used.contains(r)) { used.insert(r); out << r; }
  }
  for (const QString& w : wanted) {
    if (!used.contains(w)) { used.insert(w); out << w; }
  }
  return out;
}

inline QHash<QWidget*, BatchSummaryUiBundle> g_batchSummaryUi;
inline QHash<QWidget*, BatchSummarySnapshot> g_batchSummarySnapshots;
inline QWidget* g_activeBatchSummaryPage = nullptr;
inline QMap<QString, QMap<QString, QString>> g_liveBatchCsvPaths;

// Forward declaration — defined later in this file.
// (definition inline below; formerly forward-declared and defined late in MainWindow.cpp)
QString lastSummaryField(const QString& text, const QString& prefix);

inline QString sanitizeExportFileStem(const QString& text) {
  QString s = text.trimmed();
  if (s.isEmpty()) s = "batch";
  s.replace(QRegularExpression(QStringLiteral("[\\/:*?\"<>|]+")), "_");
  s.replace(QRegularExpression(QStringLiteral(R"(\s+)")), "_");
  s.replace(QRegularExpression(QStringLiteral("_+")), "_");
  s.remove(QRegularExpression(QStringLiteral("^_+|_+$")));
  if (s.isEmpty()) s = "batch";
  return s;
}

inline QString sanitizeSheetName(QString s) {
  s = s.trimmed();
  if (s.isEmpty()) s = "sheet";
  s.replace(QRegularExpression(QStringLiteral(R"([:\/?*\[\]])")), " ");
  s.replace(QRegularExpression(QStringLiteral(R"(\s+)")), " ");
  s = s.trimmed();
  if (s.isEmpty()) s = "sheet";
  if (s.size() > 31) s = s.left(31);
  return s;
}

inline QTableWidgetItem* makeReadOnlyItem(const QString& text) {
  auto* it = new QTableWidgetItem(text);
  it->setFlags(it->flags() & ~Qt::ItemIsEditable);
  return it;
}

inline void setupDerivedAnalysisTable(QTableWidget* t) {
  if (!t) return;
  t->setEditTriggers(QAbstractItemView::NoEditTriggers);
  t->setSelectionBehavior(QAbstractItemView::SelectItems);
  t->setSelectionMode(QAbstractItemView::ExtendedSelection);
  t->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  t->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  if (t->horizontalHeader()) t->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  if (t->verticalHeader()) t->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
}

// Bulk refill of a table whose header sections are in ResizeToContents mode.
// In that mode QHeaderView re-measures the columns after EVERY header label
// and every single cell that changes, so refilling e.g. the Best/Mean/Ranking
// tables of a 30-method x 20-problem experiment took ~1-2 s on every click in
// the Selection area. While this guard lives the sections are Interactive
// (no re-measuring) and painting is off; on destruction the given modes are
// restored, which measures every column/row exactly once.
class TableBulkFillGuard {
 public:
  TableBulkFillGuard(QTableWidget* t, QHeaderView::ResizeMode restoreH, QHeaderView::ResizeMode restoreV)
      : t_(t), restoreH_(restoreH), restoreV_(restoreV) {
    if (!t_) return;
    t_->setUpdatesEnabled(false);
    if (auto* h = t_->horizontalHeader()) h->setSectionResizeMode(QHeaderView::Interactive);
    if (auto* v = t_->verticalHeader()) v->setSectionResizeMode(QHeaderView::Interactive);
  }
  ~TableBulkFillGuard() {
    if (!t_) return;
    if (auto* h = t_->horizontalHeader()) h->setSectionResizeMode(restoreH_);
    if (auto* v = t_->verticalHeader()) v->setSectionResizeMode(restoreV_);
    t_->setUpdatesEnabled(true);
  }
  TableBulkFillGuard(const TableBulkFillGuard&) = delete;
  TableBulkFillGuard& operator=(const TableBulkFillGuard&) = delete;

 private:
  QTableWidget* t_;
  QHeaderView::ResizeMode restoreH_;
  QHeaderView::ResizeMode restoreV_;
};

inline void applyTopRankHighlight(QTableWidgetItem* it, int rank) {
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

inline QVector<int> competitionRanksForMinimization(const QVector<double>& vals) {
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
inline QIcon makeTintedIcon(const QIcon& base, const QColor& color, const QSize& size, double alphaMul = 1.0) {
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
inline QIcon makeTintedIconPreserveDetails(const QIcon& base, const QColor& color, const QSize& size, double alphaMul = 1.0) {
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

inline void openPersistentValueEditors(QTableWidget* t, int valueCol) {
  if (!t) return;
  for (int r = 0; r < t->rowCount(); ++r) {
    if (auto* it = t->item(r, valueCol)) {
      t->openPersistentEditor(it);
    }
  }
}










bool parseConvergenceCsvFile(const QString& path,
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

    // Compute overall bounds across all series (skip NaN/Inf values).
    bool any = false;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;

    for (const auto& s : series_) {
      const int n = (int(s.x.size()) < int(s.y.size()) ? int(s.x.size()) : int(s.y.size()));
      if (n < 2) continue;
      for (int i = 0; i < n; ++i) {
        if (!std::isfinite(s.x[i]) || !std::isfinite(s.y[i])) continue;
        if (!any) {
          xmin = s.x[i]; xmax = s.x[i];
          ymin = s.y[i]; ymax = s.y[i];
          any = true;
        } else {
          xmin = std::min(xmin, s.x[i]);
          xmax = std::max(xmax, s.x[i]);
          ymin = std::min(ymin, s.y[i]);
          ymax = std::max(ymax, s.y[i]);
        }
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
      if (!std::isfinite(v)) return QStringLiteral("-");
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

inline void setupTable(QTableWidget* t, const QStringList& headers) {
  t->setColumnCount(headers.size());
  t->setHorizontalHeaderLabels(headers);
  t->horizontalHeader()->setStretchLastSection(true);
  t->verticalHeader()->setVisible(false);
  t->setSelectionBehavior(QAbstractItemView::SelectRows);
  t->setSelectionMode(QAbstractItemView::SingleSelection);
  t->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::EditKeyPressed);
}


inline QStringList splitCsvLineRespectingQuotes(const QString& line) {
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


inline QVector<int> parseBatchDimensionList(const QString& text, int defaultDim, int minDim, int maxDim) {
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

inline QString batchDimensionListToText(const QVector<int>& dims) {
  QStringList parts;
  parts.reserve(dims.size());
  for (int v : dims) {
    if (v > 0) parts << QString::number(v);
  }
  return parts.join(", ");
}

inline bool parseBatchSummaryCsvMetadata(const QString& summaryPath,
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

inline QString siblingConvergenceCsvForSummary(const QString& summaryPath) {
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


// ---------------------------------------------------------------------------
// Batch experiment manifest (*.osexp)
//
// BUGFIX: Single runs and batch runs write their CSV files into the SAME
// folder (optimsolution_gui_run). "Load experiment CSV" used to rebuild a
// batch from EVERY *_summary.csv / *_convergence.csv in the chosen folder, so
// any unrelated single run that happened to be in the folder (e.g. a
// Single-run "weierstrass D=60") was silently added to the experiment as an
// extra problem row -- and, because the loader also selects that problem in
// the Batch "Problems" list, the next batch run (e.g. after adding a method)
// computed jobs for it too.
//
// Every batch run now records the exact cells of its experiment in a small
// text manifest next to its CSV files; loading uses the manifest when one
// exists. Format (UTF-8, one cell per line, '#' = comment):
//   method,problem,dim,convergence_csv
// convergence_csv is relative to the manifest folder (absolute if the file
// lives elsewhere).
// ---------------------------------------------------------------------------
struct BatchManifestEntry {
  QString method;
  QString problem;      // base problem short name (without the dimension)
  int     dim = 0;
  QString csvPath;      // absolute path of the convergence CSV
};

inline QString batchManifestFileSuffix() { return QStringLiteral("osexp"); }

inline bool readBatchManifest(const QString& manifestPath,
                              QList<BatchManifestEntry>& out,
                              QString* err = nullptr) {
  out.clear();
  QFile f(manifestPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    if (err) *err = QString("Cannot open manifest '%1'.").arg(manifestPath);
    return false;
  }
  const QDir base = QFileInfo(manifestPath).absoluteDir();
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);
  while (!ts.atEnd()) {
    const QString line = ts.readLine().trimmed();
    if (line.isEmpty() || line.startsWith('#')) continue;
    const QStringList parts = splitCsvLineRespectingQuotes(line);
    if (parts.size() < 4) continue;
    if (parts[0].trimmed().compare("method", Qt::CaseInsensitive) == 0) continue;   // header
    BatchManifestEntry e;
    e.method  = parts[0].trimmed();
    e.problem = parts[1].trimmed();
    bool ok = false;
    e.dim = parts[2].trimmed().toInt(&ok);
    QString p = parts[3].trimmed();
    if (e.method.isEmpty() || e.problem.isEmpty() || !ok || e.dim <= 0 || p.isEmpty()) continue;
    if (QFileInfo(p).isRelative()) p = base.absoluteFilePath(p);
    e.csvPath = QDir::cleanPath(p);
    out.push_back(e);
  }
  if (out.isEmpty()) {
    if (err) *err = QString("Manifest '%1' does not list any results.").arg(manifestPath);
    return false;
  }
  return true;
}

inline bool writeBatchManifest(const QString& manifestPath,
                               const QList<BatchManifestEntry>& entries,
                               QString* err = nullptr) {
  QSaveFile f(manifestPath);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    if (err) *err = QString("Cannot write manifest '%1'.").arg(manifestPath);
    return false;
  }
  const QDir base = QFileInfo(manifestPath).absoluteDir();
  QTextStream ts(&f);
  ts.setEncoding(QStringConverter::Utf8);
  ts << "# OptimSolution batch experiment manifest v1\n";
  ts << "# Written by the GUI after every batch job; \"Load experiment CSV\" loads exactly these cells.\n";
  ts << "method,problem,dim,convergence_csv\n";
  for (const BatchManifestEntry& e : entries) {
    QString rel = base.relativeFilePath(e.csvPath);
    if (rel.startsWith("..")) rel = QDir::cleanPath(e.csvPath);       // outside the folder: keep absolute
    auto q = [](QString v) {
      if (v.contains(',') || v.contains('"')) { v.replace("\"", "\"\""); v = "\"" + v + "\""; }
      return v;
    };
    ts << q(e.method) << ',' << q(e.problem) << ',' << e.dim << ',' << q(rel) << '\n';
  }
  ts.flush();
  if (!f.commit()) {
    if (err) *err = QString("Cannot write manifest '%1'.").arg(manifestPath);
    return false;
  }
  return true;
}

// Normalised path key for comparisons (case-insensitive on Windows).
inline QString batchManifestPathKey(const QString& path) {
  QString p = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
#ifdef Q_OS_WIN
  p = p.toLower();
#endif
  return p;
}

// Newest manifest in 'dirPath' (and its sub-folders) that lists 'csvPath'
// (either as convergence CSV or as the convergence sibling of a summary CSV).
inline QString findBatchManifestForCsv(const QString& dirPath, const QString& csvPath) {
  QString target = csvPath;
  if (target.endsWith("_summary.csv", Qt::CaseInsensitive)) {
    const QString conv = siblingConvergenceCsvForSummary(target);
    if (!conv.isEmpty()) target = conv;
  }
  const QString key = batchManifestPathKey(target);
  QString best;
  QDateTime bestTime;
  QDirIterator it(dirPath, QStringList() << ("*." + batchManifestFileSuffix()), QDir::Files,
                  QDirIterator::Subdirectories);
  while (it.hasNext()) {
    it.next();
    QList<BatchManifestEntry> entries;
    if (!readBatchManifest(it.filePath(), entries)) continue;
    bool hit = false;
    for (const BatchManifestEntry& e : entries) {
      if (batchManifestPathKey(e.csvPath) == key) { hit = true; break; }
    }
    if (!hit) continue;
    const QDateTime t = it.fileInfo().lastModified();
    if (best.isEmpty() || t > bestTime) { best = it.filePath(); bestTime = t; }
  }
  return best;
}

inline int parseDimFromConvergenceCsvFilename(const QString& path) {
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
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
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
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);

    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

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
    // Config parameter names (paramName_) are raw cfg keys and may contain
    // underscores (e.g. "max_iters"); prettify for display only, same
    // no-underscore/capitalized-first-word convention used everywhere else.
    QString niceParam = paramName_;
    niceParam.replace('_', ' ');
    if (!niceParam.isEmpty()) niceParam[0] = niceParam[0].toUpper();
    const QString title = paramName_.isEmpty()
      ? "Sensitivity (mean best objective value)"
      : QString("Sensitivity: %1 (mean best objective value)").arg(niceParam);
    p.drawText(QRect(plot.left(), r.top() + 2, plot.width(), 16), Qt::AlignLeft | Qt::AlignVCenter, title);

    // X axis label.
    if (!paramName_.isEmpty()) {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(std::max(9, tf.pointSize()));
      p.setFont(tf);
      p.drawText(QRect(plot.left(), r.bottom() - 30, plot.width(), 22),
                 Qt::AlignHCenter | Qt::AlignVCenter, niceParam);
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
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
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
    const bool transp = (themeMode_ == ThemeMode::Transparent);

    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

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

      // Draw X-axis title with f_best (subscript): "Runs (final f" + sub "best" + ")"
      {
        const QString xPrefix = "Runs (final f";
        const QString xSub    = "best";
        const QString xSuffix = ")";

        QFont normF = tf;
        QFont subF  = tf;
        subF.setPointSize(std::max(7, tf.pointSize() - 2));
        subF.setBold(false);

        const QFontMetrics fmN(normF);
        const QFontMetrics fmS(subF);

        const int wPre = fmN.horizontalAdvance(xPrefix);
        const int wSub = fmS.horizontalAdvance(xSub);
        const int wSuf = fmN.horizontalAdvance(xSuffix);
        const int totW = wPre + wSub + wSuf;

        const int baselineY = plot.bottom() + 28 + fmN.ascent();
        const int startX    = plot.left() + (plot.width() - totW) / 2;
        const int subShift  = fmN.ascent() / 3;

        p.setFont(normF);
        p.drawText(startX, baselineY, xPrefix);

        p.setFont(subF);
        p.drawText(startX + wPre, baselineY + subShift, xSub);

        p.setFont(normF);
        p.drawText(startX + wPre + wSub, baselineY, xSuffix);
      }

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
  Q_OBJECT
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  void setThemeMode(ThemeMode mode) {
    themeMode_ = mode;
    // Without this, QWidget::render() (used by the 300 DPI PNG export)
    // paints an implicit opaque QPalette::Window background before this
    // widget's own paintEvent runs -- regardless of paintEvent skipping its
    // own fillRect() for Transparent mode -- so exported PNGs never came
    // out actually transparent. ConvergencePlotWidget already had this;
    // it was simply missing here.
    setAttribute(Qt::WA_TranslucentBackground, mode == ThemeMode::Transparent);
    update();
  }
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
    setAttribute(Qt::WA_TranslucentBackground, false);
  }

  void clear() {
    groups_.clear();
    ann_.clear();
    title_.clear();
    subtitle_.clear();
    yLabel_.clear();
    update();
  }

  // Overrides the "No Wilcoxon data" placeholder shown (and exported to
  // PNG) whenever clear()/an empty setData() leaves nothing to plot.
  // Resets back to the default "No Wilcoxon data" if called with an empty
  // string. Persists across clear() so a caller can set a specific reason
  // (e.g. "this metric isn't applicable to a regression-only pool") right
  // before clearing.
  void setEmptyMessage(const QString& msg) {
    emptyMessage_ = msg.isEmpty() ? QStringLiteral("No Wilcoxon data") : msg;
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
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

    if (groups_.isEmpty()) {
      p.setPen(fg);
      p.drawText(r, Qt::AlignCenter, emptyMessage_.isEmpty() ? QStringLiteral("No Wilcoxon data") : emptyMessage_);
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
      p.drawText(r, Qt::AlignCenter, emptyMessage_.isEmpty() ? QStringLiteral("No Wilcoxon data") : emptyMessage_);
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
    const int bottom = 74;
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
      lf.setPointSize(std::max(10, lf.pointSize() + 1));
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
      const QString yLab = yLabel_.isEmpty() ? QStringLiteral("Metric") : yLabel_;
      if (yLab == QLatin1String("Best objective value")) {
        QFont baseF = tf;
        QFont subF = tf; subF.setBold(false);
        subF.setPointSize(std::max(7, tf.pointSize() - 2));
        QFontMetrics fmBase(baseF), fmSub(subF);
        const int wF    = fmBase.horizontalAdvance("f");
        const int wBest = fmSub.horizontalAdvance("best");
        const int startX = -(wF + wBest) / 2;
        const int baselineY = fmBase.ascent() / 2;
        p.setFont(baseF); p.drawText(startX,      baselineY,                     "f");
        p.setFont(subF);  p.drawText(startX + wF, baselineY + fmBase.ascent()/3, "best");
      } else {
        p.drawText(QRect(-plot.height()/2, -18, plot.height(), 36),
                   Qt::AlignHCenter | Qt::AlignVCenter, yLab);
      }
      p.restore();
    }

    // Annotations (brackets + p text).
    if (!ann_.isEmpty()) {
      QFont af = font();
      af.setPointSize(std::max(8, af.pointSize() - 1));

      const double y0 = globalMax;
      for (int a = 0; a < ann_.size(); ++a) {
        const Annotation& an = ann_[a];
        if (an.i < 0 || an.j < 0) continue;
        if (an.i >= k || an.j >= k) continue;
        const int i = an.i, j = an.j;
        const int l = std::min(i,j), rr = std::max(i,j);
        const int xL = int(std::llround(plot.left() + (l  + 0.5) * dx));
        const int xR = int(std::llround(plot.left() + (rr + 0.5) * dx));
        const int lev = (a < levels.size()) ? levels[a] : 0;
        const int yP = yToPix(y0 + annStep * double(lev + 1));

        QColor penC = an.significant ? (dark ? QColor(255,210,0) : QColor(160,100,0))
                                     : (dark ? QColor(190,190,190) : QColor(90,90,90));
        penC.setAlpha(230);
        p.setPen(QPen(penC, 1.5));
        p.drawLine(xL, yP, xR, yP);
        p.drawLine(xL, yP, xL, yP + 6);
        p.drawLine(xR, yP, xR, yP + 6);

        p.setPen(fg);
        const int mid = (xL + xR) / 2;
        const QLatin1String pAdjPfx("p_adj=");
        const QLatin1String pPfx("p=");
        // Both "p_adj=<value>" (Holm-corrected, existing convention) and
        // plain "p=<value>" now share the same italic rendering, and both
        // additionally recognize scientific notation in <value> (e.g.
        // "2.6e-06") and render it as "2.6" + "x10" + a raised, smaller
        // superscript exponent ("-6"), instead of the raw "e-06" text --
        // any trailing " *" (significance marker) is kept outside the
        // scientific-notation part, still italic, at the normal baseline.
        if (an.text.startsWith(pAdjPfx) || an.text.startsWith(pPfx)) {
          const bool isAdj = an.text.startsWith(pAdjPfx);
          QString val = an.text.mid(isAdj ? pAdjPfx.size() : pPfx.size());
          QString star;
          if (val.endsWith(" *")) { star = " *"; val.chop(2); }

          QString mantissa = val;
          QString exponent; // empty => no scientific notation in this value
          static const QRegularExpression sciRe("^([0-9.]+)[eE]([+-]?[0-9]+)$");
          const QRegularExpressionMatch m = sciRe.match(val);
          if (m.hasMatch()) {
            mantissa = m.captured(1);
            exponent = QString::number(m.captured(2).toInt()); // drop leading zeros/plus
          }

          QFont baseF = af; baseF.setItalic(true);
          QFont subF  = af; subF.setItalic(true);
          subF.setPointSize(std::max(5, af.pointSize() - 2));
          QFontMetrics fmBase(baseF), fmSub(subF);

          const int wP    = fmBase.horizontalAdvance("p");
          const int wAdj  = isAdj ? fmSub.horizontalAdvance("adj") : 0;
          const int wEq   = fmBase.horizontalAdvance("=");
          const int wMant = fmBase.horizontalAdvance(mantissa);
          const int wX10  = exponent.isEmpty() ? 0 : fmBase.horizontalAdvance(QString::fromUtf8("\xC3\x97" "10"));
          const int wExp  = exponent.isEmpty() ? 0 : fmSub.horizontalAdvance(exponent);
          const int wStar = star.isEmpty() ? 0 : fmBase.horizontalAdvance(star);
          const int totalW = wP + wAdj + wEq + wMant + wX10 + wExp + wStar;

          int x = mid - totalW / 2;
          const int baseY = yP - 5, subOff = fmBase.ascent() / 3, supOff = -fmBase.ascent() * 2 / 5;
          p.setFont(baseF); p.drawText(x, baseY, "p"); x += wP;
          if (isAdj) { p.setFont(subF); p.drawText(x, baseY + subOff, "adj"); x += wAdj; }
          p.setFont(baseF); p.drawText(x, baseY, "=" + mantissa); x += wEq + wMant;
          if (!exponent.isEmpty()) {
            p.setFont(baseF); p.drawText(x, baseY, QString::fromUtf8("\xC3\x97" "10")); x += wX10;
            p.setFont(subF);  p.drawText(x, baseY + supOff, exponent); x += wExp;
          }
          if (!star.isEmpty()) { p.setFont(baseF); p.drawText(x, baseY, star); }
        } else {
          p.setFont(af);
          p.drawText(QRect(mid - 120, yP - 18, 240, 16),
                     Qt::AlignHCenter | Qt::AlignVCenter, an.text);
        }
      }
    }
  }

private:
  QVector<Group> groups_;
  QVector<Annotation> ann_;
  QString title_;
  QString subtitle_;
  QString yLabel_;
  QString emptyMessage_;

  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
};
// Rank plot for batch statistics (average ranks across problems; lower rank is better).
class RankPlotWidget final : public QWidget {
  Q_OBJECT
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  struct Annotation {
    QString a;
    QString b;
    QString text;
    bool significant = false;
  };

  void setThemeMode(ThemeMode mode) {
    themeMode_ = mode;
    // See WilcoxonBoxPlotWidget::setThemeMode() for why this is needed --
    // without it, the 300 DPI PNG export never actually comes out
    // transparent regardless of paintEvent's own transparent-mode handling.
    setAttribute(Qt::WA_TranslucentBackground, mode == ThemeMode::Transparent);
    update();
  }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }
  void setAnnotations(QVector<Annotation> ann) { ann_ = std::move(ann); update(); }

  explicit RankPlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TranslucentBackground, false);

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

  // Temporarily suppress the info panel (for PNG export).
  void setInfoVisible(bool visible) {
    if (!infoBox_) return;
    if (visible) {
      const bool hasText = !infoText_.trimmed().isEmpty();
      infoBox_->setVisible(hasText);
      if (hasText) updateInfoBoxGeometry();
    } else {
      infoBox_->setVisible(false);
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
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

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
    // Reviewer 2, comment 3: the average-rank chart (else) now uses a
    // CONVENTIONAL vertical axis -- values increase upward from a zero
    // baseline, exactly like an ordinary bar chart -- rather than the
    // former inverted mapping where a taller bar meant a lower (better)
    // rank. Because lower rank is still what "better" means here, the
    // chart draws its own explicit "(lower = better)" annotation directly
    // (see below) rather than relying on a caller-supplied subtitle.
    QFont tickFont = font();
    tickFont.setPointSize(std::max(9, tickFont.pointSize()));
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
        maxTickW = std::max(maxTickW, fmTick.horizontalAdvance(QString::number(t, 'f', 1)));
      }
    }

    // else: extra left room for the rotated "Rank" axis title, placed to
    // the left of the tick numbers rather than inside the plot area.
    const int left = hasBoxes ? std::max(60, maxTickW + 16) : std::max(60, maxTickW + 16) + 22;
    int right = 18;
    // else: extra top room for the title + subtitle row + the larger
    // value labels now drawn above the tallest bar.
    const int top = hasBoxes ? 34 : 54;
    // else: a little extra headroom above the tallest bar for its value
    // label, and extra room below for a larger, more legible method name.
    const int bottom = hasBoxes ? 76 : 54;

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

    // else-only: a plain, conventional numeric scale for average rank,
    // zero at the bottom, increasing upward -- no sign flip of any kind.
    const int n = items.size();
    double vMax = 1.0;
    if (!hasBoxes && n > 0) {
      for (const Item& it : items) {
        if (std::isfinite(it.label)) vMax = std::max(vMax, it.label);
      }
      vMax *= 1.15; // headroom so the tallest bar's value label isn't clipped
    }
    auto mapYBar = [&](double v) {
      return plot.bottom() - (v / std::max(1e-9, vMax)) * plot.height();
    };

    // Grid (mirrors Convergence) -- same, proven grid for both chart types.
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
        const double v = t * vMax;
        const int yy = int(mapYBar(v));
        p.drawLine(QPoint(plot.left() - 4, yy), QPoint(plot.left(), yy));
        p.drawText(QRect(2, yy - 10, left - 8, 20), Qt::AlignRight | Qt::AlignVCenter, QString::number(v, 'f', 1));
      }
      // The "(lower = better)" note is drawn later, in the Title+subtitle
      // section below, at the subtitle's own row -- see there. Drawing it
      // here as well used to collide with the title text.
    }

    // Plot elements.
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

        // Restore pen (was NoPen for jittered points) before text drawing.
        p.setPen(fg);
        p.setBrush(Qt::NoBrush);

        // Median label.
        p.drawText(QRectF(xx, std::min(yLo, yHi) - 18, boxW, 16), Qt::AlignHCenter | Qt::AlignVCenter,
                   std::isfinite(it.label) ? QString::number(it.label, 'f', 3) : "-");

        // Method label - larger readable font.
        {
          QFont labF = p.font();
          labF.setPointSize(std::max(10, labF.pointSize() + 1));
          labF.setBold(false);
          p.setFont(labF);
          const QString lab = QFontMetrics(labF).elidedText(it.m, Qt::ElideRight, int(boxW) + 20);
          p.drawText(QRectF(xx - 10, plot.bottom() + 6, boxW + 20, 36), Qt::AlignHCenter | Qt::AlignTop, lab);
          p.setFont(tickFont);
        }
      }
    } else {
      // Conventional vertical bars (Reviewer 2, comment 3): the bar for
      // method i runs from the zero baseline (plot.bottom()) up to its
      // average-rank value on the plain scale defined by mapYBar() above,
      // so a LOWER (better) rank now draws a SHORTER bar and a HIGHER
      // (worse) rank draws a TALLER one -- ordinary bar-chart semantics,
      // with the "(lower = better)" annotation above making the direction
      // that is good explicit rather than left for the reader to infer.
      const double barW = boxW;
      QFont valueFont = font();
      valueFont.setBold(true);
      valueFont.setPointSize(std::max(12, tickFont.pointSize() + 3));
      QFont nameFont = font();
      nameFont.setPointSize(std::max(10, nameFont.pointSize() + 1));

      for (int i = 0; i < n; ++i) {
        const double rv = items[i].label;
        const double xx = x0 + i * slotW + 0.5 * (slotW - barW);
        const double yTop = mapYBar(rv);
        const QRectF bar(xx, yTop, barW, plot.bottom() - yTop);

        QColor col = dark ? QColor(70, 130, 255) : QColor(50, 110, 235);
        col.setAlpha(dark ? 150 : 150);
        if (i == 0) {
          col = dark ? QColor(0, 200, 120) : QColor(0, 160, 80);
          col.setAlpha(dark ? 200 : 200);
        }
        p.fillRect(bar, col);
        p.setPen(QPen(fg, 1));
        p.drawRect(bar);

        // Value label, above the bar.
        p.setFont(valueFont);
        p.setPen(fg);
        p.drawText(QRectF(bar.left() - 14, bar.top() - 24, bar.width() + 28, 20),
                   Qt::AlignHCenter | Qt::AlignVCenter, QString::number(rv, 'f', 2));

        // Method name, below the axis -- larger, more legible font.
        p.setFont(nameFont);
        const QString lab = QFontMetrics(nameFont).elidedText(items[i].m, Qt::ElideRight, int(barW) + 14);
        p.drawText(QRectF(bar.left() - 7, plot.bottom() + 6, bar.width() + 14, 34),
                   Qt::AlignHCenter | Qt::AlignTop, lab);
        p.setFont(tickFont);
      }
    }

    // Pairwise annotations (all pairs).
    if (!annDraw.isEmpty()) {
      QFont af = font();
      af.setPointSize(std::max(8, af.pointSize() - 1));

      for (const auto& ad : annDraw) {
        const double cxL = x0 + ad.l * slotW + 0.5 * slotW;
        const double cxR = x0 + ad.r * slotW + 0.5 * slotW;
        const double yRv = 0.5 - annStep * double(ad.level + 1);
        const int yP = int(std::llround(mapY(yRv)));

        QColor penCol = ad.significant ? (dark ? QColor(255,210,0) : QColor(160,100,0))
                                       : (dark ? QColor(190,190,190) : QColor(90,90,90));
        penCol.setAlpha(230);
        QPen pen(penCol, 1.5); pen.setCosmetic(true);
        p.setPen(pen);
        const int xL = int(std::llround(cxL)), xR = int(std::llround(cxR));
        p.drawLine(xL, yP, xR, yP);
        p.drawLine(xL, yP, xL, yP + 6);
        p.drawLine(xR, yP, xR, yP + 6);

        p.setPen(fg);
        const int mid = (xL + xR) / 2;
        const QLatin1String pAdjPfx("p_adj=");
        if (ad.text.startsWith(pAdjPfx)) {
          const QString val = ad.text.mid(pAdjPfx.size());
          QFont baseF = af; baseF.setItalic(true);
          QFont subF  = af; subF.setItalic(true);
          subF.setPointSize(std::max(5, af.pointSize() - 2));
          QFontMetrics fmBase(baseF), fmSub(subF);
          const int wP    = fmBase.horizontalAdvance("p");
          const int wAdj  = fmSub.horizontalAdvance("adj");
          const int wRest = fmBase.horizontalAdvance("=" + val);
          const int sx    = mid - (wP + wAdj + wRest) / 2;
          const int baseY = yP - 5, subOff = fmBase.ascent() / 3;
          p.setFont(baseF); p.drawText(sx,             baseY,          "p");
          p.setFont(subF);  p.drawText(sx + wP,        baseY + subOff, "adj");
          p.setFont(baseF); p.drawText(sx + wP + wAdj, baseY,          "=" + val);
        } else {
          p.setFont(af);
          p.drawText(QRect(mid - 120, yP - 18, 240, 16),
                     Qt::AlignHCenter | Qt::AlignVCenter, ad.text);
        }
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
    } else if (!hasBoxes) {
      // Self-contained "lower is better" note (Reviewer 2, comment 3),
      // drawn at the subtitle's own row so it never collides with the
      // title above it, regardless of what the caller passes in.
      QFont noteFont = sf;
      noteFont.setBold(true);
      noteFont.setItalic(true);
      p.setFont(noteFont);
      p.drawText(QRect(plot.left(), r.top() + 20, plot.width(), 14), Qt::AlignLeft | Qt::AlignVCenter,
                 QStringLiteral("Lower bar is better"));
    }

    // Y axis label ("Rank"). hasBoxes keeps its original placement, just
    // inside the plot next to the axis. For the average-rank chart (else),
    // Reviewer 2 asked for this label to sit to the LEFT of the vertical
    // axis (in the margin, not overlapping the plot) and to be bolder --
    // the extra left margin reserved above makes room for it there.
    p.save();
    if (hasBoxes) {
      QColor yLab = fg;
      yLab.setAlpha(200);
      p.setPen(yLab);
      p.translate(plot.left() + 14, plot.center().y());
    } else {
      QFont rankFont = font();
      rankFont.setBold(true);
      rankFont.setPointSize(std::max(10, rankFont.pointSize() + 1));
      p.setFont(rankFont);
      p.setPen(fg);
      p.translate(r.left() + 14, plot.center().y());
    }
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


// -----------------------------------------------------------------------------
// CDDiagramWidget — classic Nemenyi "critical difference" diagram (Demsar 2006).
// Plots average ranks of each method on a single axis and draws thick bars
// ("cliques") connecting groups of methods whose average ranks differ by no
// more than the critical difference (CD) — i.e., groups that are not
// significantly different according to the post-hoc Nemenyi test.
// -----------------------------------------------------------------------------
class CDDiagramWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }

  explicit CDDiagramWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(240);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void clear() {
    methods_.clear();
    avgRanks_.clear();
    k_ = 0;
    n_ = 0;
    cd_ = std::numeric_limits<double>::quiet_NaN();
    alpha_ = 0.05;
    title_.clear();
    subtitle_.clear();
    approx_ = false;
    update();
  }

  // methods/avgRanks must be the same size (k). n = number of problems, alpha = significance level.
  void setData(QStringList methods, QVector<double> avgRanks, double cd, double alpha,
               int n, bool approx, const QString& title, const QString& subtitle) {
    methods_ = std::move(methods);
    avgRanks_ = std::move(avgRanks);
    k_ = methods_.size();
    n_ = n;
    cd_ = cd;
    alpha_ = alpha;
    approx_ = approx;
    title_ = title;
    subtitle_ = subtitle;
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    const QColor axisColor = (dark || transp) ? QColor(200, 200, 200) : QColor(60, 60, 60);
    const QColor cliqueColor = (dark || transp) ? QColor(255, 170, 60) : QColor(200, 110, 0);
    const QColor tickColor = (dark || transp) ? QColor(120, 190, 255) : QColor(20, 90, 170);
    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

    if (methods_.isEmpty() || k_ < 2 || !std::isfinite(cd_)) {
      p.setPen(fg);
      QFont f = font();
      f.setBold(true);
      p.setFont(f);
      p.drawText(r.adjusted(12, 12, -12, -12), Qt::AlignCenter,
                 "No CD diagram data.\nRun a batch with at least 2 methods and 2 problems.");
      return;
    }

    // ---- Header: title / subtitle ----
    int y = 8;
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(tf.pointSize() + 1);
      p.setFont(tf);
      p.setPen(fg);
      p.drawText(QRect(12, y, r.width() - 24, 20), Qt::AlignLeft | Qt::AlignVCenter, title_);
      y += 20;
      QFont sf = font();
      sf.setPointSize(std::max(7, sf.pointSize() - 1));
      p.setFont(sf);
      QColor subFg = fg; subFg.setAlpha(200);
      p.setPen(subFg);
      p.drawText(QRect(12, y, r.width() - 24, 16), Qt::AlignLeft | Qt::AlignVCenter, subtitle_);
      y += 22;
    }

    // ---- Layout geometry ----
    const int leftMargin = 60;
    const int rightMargin = 60;
    const int topArea = y + 46;      // room for above-axis labels
    const int bottomArea = 46;       // room for below-axis labels + CD ruler
    const int axisY = topArea;
    const int axisLeft = leftMargin;
    const int axisRight = r.width() - rightMargin;
    if (axisRight <= axisLeft || r.height() <= topArea + bottomArea) {
      return;
    }

    // Rank scale: 1 (best) .. k (worst), left-to-right.
    const double rankMin = 1.0;
    const double rankMax = double(k_);
    auto xOf = [&](double rank) -> double {
      if (rankMax <= rankMin) return double(axisLeft);
      const double t = (rank - rankMin) / (rankMax - rankMin);
      return double(axisLeft) + t * double(axisRight - axisLeft);
    };

    // ---- Grid (mirrors Convergence / Friedman / Wilcoxon plots) ----
    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      const int gridTop = topArea - 30;
      const int gridBottom = axisY + bottomArea - 6;
      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int gi = 0; gi <= div; ++gi) {
        const int xx = axisLeft + int(double(gi) * double(axisRight - axisLeft) / double(div));
        p.drawLine(QPoint(xx, gridTop), QPoint(xx, gridBottom));
      }
    }

    // ---- Draw axis with integer rank ticks ----
    QPen axisPen(axisColor, 1.4);
    p.setPen(axisPen);
    p.drawLine(QPointF(axisLeft, axisY), QPointF(axisRight, axisY));
    QFont tickFont = font();
    tickFont.setPointSize(std::max(7, tickFont.pointSize() - 1));
    p.setFont(tickFont);
    for (int rk = 1; rk <= k_; ++rk) {
      const double xx = xOf(double(rk));
      p.drawLine(QPointF(xx, axisY - 4), QPointF(xx, axisY + 4));
      p.drawText(QRectF(xx - 14, axisY + 6, 28, 14), Qt::AlignCenter, QString::number(rk));
    }

    // ---- Sort methods by average rank for label placement + clique detection ----
    QVector<int> order(k_);
    for (int i = 0; i < k_; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (avgRanks_[a] == avgRanks_[b]) return methods_[a].toLower() < methods_[b].toLower();
      return avgRanks_[a] < avgRanks_[b];
    });

    // ---- Method ticks + alternating labels (above/below) to reduce overlap ----
    QFont labelFont = font();
    labelFont.setPointSize(std::max(7, labelFont.pointSize()));
    p.setFont(labelFont);
    const QFontMetrics fm(labelFont);

    for (int pos = 0; pos < order.size(); ++pos) {
      const int mi = order[pos];
      const double xx = xOf(avgRanks_[mi]);
      const bool above = (pos % 2 == 0);

      QPen tp(tickColor, 1.6);
      p.setPen(tp);
      p.drawEllipse(QPointF(xx, axisY), 3.2, 3.2);

      const QString label = QString("%1 (%2)").arg(methods_[mi]).arg(avgRanks_[mi], 0, 'f', 2);
      const int textW = fm.horizontalAdvance(label) + 6;
      const int lx = int(xx) - textW / 2;

      if (above) {
        const int labelY = axisY - 12 - 14;
        p.drawLine(QPointF(xx, axisY - 4), QPointF(xx, labelY + 14));
        p.setPen(fg);
        p.drawText(QRect(lx, labelY, textW, 14), Qt::AlignCenter, label);
      } else {
        const int labelY = axisY + 12;
        p.setPen(tp);
        p.drawLine(QPointF(xx, axisY + 4), QPointF(xx, labelY));
        p.setPen(fg);
        p.drawText(QRect(lx, labelY, textW, 14), Qt::AlignCenter, label);
      }
    }

    // ---- Clique bars: maximal runs of consecutive (by rank) methods within CD of each other ----
    QVector<QPair<int,int>> cliques; // indices into 'order'
    {
      QVector<QPair<int,int>> raw;
      for (int i = 0; i < order.size(); ++i) {
        int j = i;
        while (j + 1 < order.size() && (avgRanks_[order[j + 1]] - avgRanks_[order[i]]) <= cd_ + 1e-9) ++j;
        if (j > i) raw.push_back({i, j});
      }
      for (const auto& c : raw) {
        bool subsumed = false;
        for (const auto& d : raw) {
          if (c == d) continue;
          if (d.first <= c.first && d.second >= c.second && (d.first < c.first || d.second > c.second)) {
            subsumed = true; break;
          }
        }
        if (!subsumed && !cliques.contains(c)) cliques.push_back(c);
      }
    }

    // Stack clique bars just below the axis (levels to avoid overlap).
    QVector<int> levelOfClique(cliques.size(), 0);
    {
      QVector<QVector<QPair<int,int>>> levels;
      for (int ci = 0; ci < cliques.size(); ++ci) {
        int lev = 0;
        for (; lev < levels.size(); ++lev) {
          bool overlap = false;
          for (const auto& other : levels[lev]) {
            if (!(cliques[ci].second < other.first || cliques[ci].first > other.second)) { overlap = true; break; }
          }
          if (!overlap) break;
        }
        if (lev == levels.size()) levels.push_back({});
        levels[lev].push_back(cliques[ci]);
        levelOfClique[ci] = lev;
      }
    }

    QPen cliquePen(cliqueColor, 4.0, Qt::SolidLine, Qt::RoundCap);
    for (int ci = 0; ci < cliques.size(); ++ci) {
      const int i0 = cliques[ci].first;
      const int i1 = cliques[ci].second;
      const double x0 = xOf(avgRanks_[order[i0]]);
      const double x1 = xOf(avgRanks_[order[i1]]);
      const double barY = axisY - 8.0 - double(levelOfClique[ci]) * 6.0;
      p.setPen(cliquePen);
      p.drawLine(QPointF(x0, barY), QPointF(x1, barY));
    }

    // ---- CD reference ruler (top-right corner) ----
    {
      const double rulerLen = xOf(rankMin + cd_) - xOf(rankMin);
      const double rx1 = axisRight - std::max(20.0, rulerLen);
      const double rx0 = rx1 - rulerLen;
      const double ry = topArea - 26;
      QPen rp(axisColor, 1.6);
      p.setPen(rp);
      p.drawLine(QPointF(rx0, ry), QPointF(rx1, ry));
      p.drawLine(QPointF(rx0, ry - 4), QPointF(rx0, ry + 4));
      p.drawLine(QPointF(rx1, ry - 4), QPointF(rx1, ry + 4));
      QFont cdFont = font();
      cdFont.setPointSize(std::max(7, cdFont.pointSize() - 1));
      p.setFont(cdFont);
      p.setPen(fg);
      const QString cdLabel = QString("CD = %1%2").arg(cd_, 0, 'f', 3).arg(approx_ ? " (approx.)" : "");
      p.drawText(QRectF(rx0 - 40, ry - 20, (rx1 - rx0) + 80, 14), Qt::AlignCenter, cdLabel);
    }
  }

private:
  QStringList methods_;
  QVector<double> avgRanks_;
  int k_ = 0;
  int n_ = 0;
  double cd_ = std::numeric_limits<double>::quiet_NaN();
  double alpha_ = 0.05;
  bool approx_ = false;
  QString title_;
  QString subtitle_;
  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
};


// -----------------------------------------------------------------------------
// LandscapeWidget — 2D objective-function contour with an animated overlay of
// the best-so-far trajectory across iterations (exploration -> exploitation).
//
// Note: this shows the trajectory of the current *best* candidate, not the
// full population of every individual at every iteration. Exposing the full
// swarm would require a per-method hook into each of the ~78 optimizer
// implementations; the best-so-far trajectory is generic across all
// population-based methods and requires no per-method changes.
// -----------------------------------------------------------------------------
class LandscapeWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };

  struct Point {
    double x1 = 0.0;
    double x2 = 0.0;
    double f = 0.0;
    int iter = 0;
  };

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }

  explicit LandscapeWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(260);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void clear() {
    gridReady_ = false;
    gridZ_.clear();
    traj_.clear();
    frame_ = 0;
    title_.clear();
    subtitle_.clear();
    update();
  }

  // z is row-major (ny rows outer, nx columns inner), sampled over [xmin,xmax] x [ymin,ymax].
  void setGrid(int nx, int ny, QVector<double> z, double xmin, double xmax, double ymin, double ymax) {
    nx_ = nx;
    ny_ = ny;
    gridZ_ = std::move(z);
    xmin_ = xmin; xmax_ = xmax; ymin_ = ymin; ymax_ = ymax;
    gridReady_ = (nx_ > 0 && ny_ > 0 && gridZ_.size() == nx_ * ny_);
    update();
  }

  void setTrajectory(QVector<Point> traj) {
    traj_ = std::move(traj);
    frame_ = traj_.isEmpty() ? 0 : std::min(frame_, int(traj_.size()) - 1);
    update();
  }

  void setTitleSubtitle(const QString& title, const QString& subtitle) {
    title_ = title;
    subtitle_ = subtitle;
    update();
  }

  int frameCount() const { return traj_.size(); }

  void setFrame(int f) {
    if (traj_.isEmpty()) { frame_ = 0; update(); return; }
    frame_ = std::max(0, std::min(f, int(traj_.size()) - 1));
    update();
  }
  int frame() const { return frame_; }

  Point currentPoint() const {
    if (traj_.isEmpty()) return Point{};
    return traj_[std::max(0, std::min(frame_, int(traj_.size()) - 1))];
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);

    if (!transp) p.fillRect(r, bg);

    if (!gridReady_) {
      p.setPen(fg);
      QFont f = font();
      f.setBold(true);
      p.setFont(f);
      p.drawText(r.adjusted(12, 12, -12, -12), Qt::AlignCenter,
                 "No landscape data.\nRequires a completed run on a 2D problem.");
      return;
    }

    int y = 8;
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(tf.pointSize() + 1);
      p.setFont(tf);
      p.setPen(fg);
      p.drawText(QRect(12, y, r.width() - 24, 20), Qt::AlignLeft | Qt::AlignVCenter, title_);
      y += 20;
      QFont sf = font();
      sf.setPointSize(std::max(7, sf.pointSize() - 1));
      p.setFont(sf);
      QColor subFg = fg; subFg.setAlpha(200);
      p.setPen(subFg);
      p.drawText(QRect(12, y, r.width() - 24, 16), Qt::AlignLeft | Qt::AlignVCenter, subtitle_);
      y += 22;
    }

    const int leftM = 46, rightM = 16, topM = y + 6, botM = 26;
    const QRect plotRect(leftM, topM, r.width() - leftM - rightM, r.height() - topM - botM);
    if (plotRect.width() < 40 || plotRect.height() < 40) return;

    double zmin = std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();
    for (double v : gridZ_) {
      if (std::isfinite(v)) { zmin = std::min(zmin, v); zmax = std::max(zmax, v); }
    }
    if (!(zmax > zmin)) zmax = zmin + 1.0;

    auto colorFor = [&](double v) -> QColor {
      double t = (v - zmin) / (zmax - zmin);
      t = std::max(0.0, std::min(1.0, t));
      static const QVector<QColor> stops = {
        QColor(40, 60, 180), QColor(40, 160, 90), QColor(230, 200, 40), QColor(200, 40, 40)
      };
      const double seg = t * double(stops.size() - 1);
      const int i0 = std::min(int(seg), int(stops.size()) - 2);
      const double frac = seg - double(i0);
      const QColor& c0 = stops[i0];
      const QColor& c1 = stops[i0 + 1];
      return QColor(int(c0.red()   + (c1.red()   - c0.red())   * frac),
                    int(c0.green() + (c1.green() - c0.green()) * frac),
                    int(c0.blue()  + (c1.blue()  - c0.blue())  * frac));
    };

    if (nx_ > 0 && ny_ > 0) {
      const double cellW = double(plotRect.width()) / double(nx_);
      const double cellH = double(plotRect.height()) / double(ny_);
      for (int j = 0; j < ny_; ++j) {
        const int py = plotRect.bottom() - int((j + 1) * cellH);
        for (int i = 0; i < nx_; ++i) {
          const double v = gridZ_[j * nx_ + i];
          if (!std::isfinite(v)) continue;
          const int px = plotRect.left() + int(i * cellW);
          p.fillRect(QRect(px, py, int(std::ceil(cellW)) + 1, int(std::ceil(cellH)) + 1), colorFor(v));
        }
      }
    }

    p.setPen(QPen(fg, 1));
    p.drawRect(plotRect.adjusted(0, 0, -1, -1));

    auto mapPt = [&](double x1, double x2) -> QPointF {
      const double tx = (x1 - xmin_) / std::max(1e-12, (xmax_ - xmin_));
      const double ty = (x2 - ymin_) / std::max(1e-12, (ymax_ - ymin_));
      return QPointF(plotRect.left() + tx * plotRect.width(), plotRect.bottom() - ty * plotRect.height());
    };

    QFont tickFont = font();
    tickFont.setPointSize(std::max(7, tickFont.pointSize() - 2));
    p.setFont(tickFont);
    p.setPen(fg);
    p.drawText(QRect(plotRect.left() - 4, plotRect.bottom() + 4, 70, 14), Qt::AlignLeft, QString::number(xmin_, 'g', 3));
    p.drawText(QRect(plotRect.right() - 66, plotRect.bottom() + 4, 70, 14), Qt::AlignRight, QString::number(xmax_, 'g', 3));

    if (!traj_.isEmpty()) {
      const int upto = std::max(0, std::min(frame_, int(traj_.size()) - 1));
      QPen trailPen(QColor(255, 255, 255, 190), 1.6);
      p.setPen(trailPen);
      for (int i = 1; i <= upto; ++i) {
        p.drawLine(mapPt(traj_[i - 1].x1, traj_[i - 1].x2), mapPt(traj_[i].x1, traj_[i].x2));
      }
      const QPointF cur = mapPt(traj_[upto].x1, traj_[upto].x2);
      p.setPen(QPen(Qt::black, 1.4));
      p.setBrush(QColor(255, 255, 255));
      p.drawEllipse(cur, 6.0, 6.0);
      p.setBrush(QColor(255, 60, 60));
      p.drawEllipse(cur, 3.2, 3.2);
    }
  }

private:
  ThemeMode themeMode_ = ThemeMode::Dark;
  bool gridReady_ = false;
  int nx_ = 0, ny_ = 0;
  QVector<double> gridZ_;
  double xmin_ = 0.0, xmax_ = 1.0, ymin_ = 0.0, ymax_ = 1.0;
  QVector<Point> traj_;
  int frame_ = 0;
  QString title_;
  QString subtitle_;
};


// -----------------------------------------------------------------------------
// ParetoFrontWidget -- 2D scatter of a multi-objective run's Pareto-optimal
// solutions (objective 1 vs objective 2). For problems with more than two
// objectives, only the first two are plotted; a note says so in the subtitle.
// -----------------------------------------------------------------------------
class ParetoFrontWidget final : public QWidget {
public:
  enum class ThemeMode { Light = 0, Dark = 1, Transparent = 2 };
  enum class GridDensity { Sparse = 0, Medium = 1, Dense = 2 };

  void setThemeMode(ThemeMode mode) { themeMode_ = mode; update(); }
  void setGridDensity(GridDensity density) { gridDensity_ = density; update(); }

  explicit ParetoFrontWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMinimumHeight(260);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void clear() {
    f1_.clear();
    f2_.clear();
    title_.clear();
    subtitle_.clear();
    update();
  }

  // f1/f2 are the first two objective values of each Pareto-optimal point.
  void setPoints(QVector<double> f1, QVector<double> f2, const QString& obj1Name, const QString& obj2Name) {
    f1_ = std::move(f1);
    f2_ = std::move(f2);
    obj1Name_ = obj1Name;
    obj2Name_ = obj2Name;
    update();
  }

  void setTitleSubtitle(const QString& title, const QString& subtitle) {
    title_ = title;
    subtitle_ = subtitle;
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect();
    const bool dark = (themeMode_ == ThemeMode::Dark);
    const bool transp = (themeMode_ == ThemeMode::Transparent);
    const QColor bg = dark ? QColor(24, 24, 24) : QColor(255, 255, 255);
    const QColor fg = (dark || transp) ? QColor(230, 230, 230) : QColor(24, 24, 24);
    const QColor axisColor = (dark || transp) ? QColor(200, 200, 200) : QColor(60, 60, 60);
    const QColor dotColor = (dark || transp) ? QColor(255, 170, 60) : QColor(200, 110, 0);
    QColor grid = (dark || transp) ? QColor(120, 120, 120) : QColor(140, 140, 140);
    grid.setAlpha(dark ? 55 : transp ? 80 : 50);
    if (gridDensity_ == GridDensity::Dense) grid.setAlpha(dark ? 42 : transp ? 65 : 38);

    if (!transp) p.fillRect(r, bg);

    if (f1_.isEmpty()) {
      p.setPen(fg);
      QFont f = font();
      f.setBold(true);
      p.setFont(f);
      p.drawText(r.adjusted(12, 12, -12, -12), Qt::AlignCenter,
                 "No Pareto front data.\nRun a multi-objective optimization to see it here.");
      return;
    }

    int y = 8;
    {
      QFont tf = font();
      tf.setBold(true);
      tf.setPointSize(tf.pointSize() + 1);
      p.setFont(tf);
      p.setPen(fg);
      p.drawText(QRect(12, y, r.width() - 24, 20), Qt::AlignLeft | Qt::AlignVCenter, title_);
      y += 20;
      QFont sf = font();
      sf.setPointSize(std::max(7, sf.pointSize() - 1));
      p.setFont(sf);
      QColor subFg = fg; subFg.setAlpha(200);
      p.setPen(subFg);
      p.drawText(QRect(12, y, r.width() - 24, 16), Qt::AlignLeft | Qt::AlignVCenter, subtitle_);
      y += 22;
    }

    const int leftM = 56, rightM = 16, topM = y + 6, botM = 36;
    const QRect plotRect(leftM, topM, r.width() - leftM - rightM, r.height() - topM - botM);
    if (plotRect.width() < 40 || plotRect.height() < 40) return;

    double xmin = f1_[0], xmax = f1_[0], ymin = f2_[0], ymax = f2_[0];
    for (int i = 0; i < f1_.size(); ++i) {
      xmin = std::min(xmin, f1_[i]); xmax = std::max(xmax, f1_[i]);
      ymin = std::min(ymin, f2_[i]); ymax = std::max(ymax, f2_[i]);
    }
    if (!(xmax > xmin)) xmax = xmin + 1.0;
    if (!(ymax > ymin)) ymax = ymin + 1.0;
    const double padX = (xmax - xmin) * 0.08;
    const double padY = (ymax - ymin) * 0.08;
    xmin -= padX; xmax += padX;
    ymin -= padY; ymax += padY;

    auto mapPt = [&](double x, double yv) -> QPointF {
      const double tx = (x - xmin) / std::max(1e-300, (xmax - xmin));
      const double ty = (yv - ymin) / std::max(1e-300, (ymax - ymin));
      return QPointF(plotRect.left() + tx * plotRect.width(), plotRect.bottom() - ty * plotRect.height());
    };

    p.setPen(QPen(axisColor, 1.2));
    p.drawRect(plotRect.adjusted(0, 0, -1, -1));

    {
      int div = 10;
      if (gridDensity_ == GridDensity::Sparse) div = 5;
      else if (gridDensity_ == GridDensity::Medium) div = 10;
      else div = 20;

      p.setPen(QPen(grid, 1, Qt::DotLine));
      for (int gi = 1; gi < div; ++gi) {
        const int gx = plotRect.left() + int(double(gi) * double(plotRect.width()) / double(div));
        p.drawLine(QPoint(gx, plotRect.top()), QPoint(gx, plotRect.bottom()));
        const int gy = plotRect.top() + int(double(gi) * double(plotRect.height()) / double(div));
        p.drawLine(QPoint(plotRect.left(), gy), QPoint(plotRect.right(), gy));
      }
    }

    QFont tickFont = font();
    tickFont.setPointSize(std::max(7, tickFont.pointSize() - 2));
    p.setFont(tickFont);
    p.setPen(fg);
    p.drawText(QRect(plotRect.left() - 6, plotRect.bottom() + 4, 90, 14), Qt::AlignLeft, QString::number(xmin, 'g', 3));
    p.drawText(QRect(plotRect.right() - 84, plotRect.bottom() + 4, 90, 14), Qt::AlignRight, QString::number(xmax, 'g', 3));
    p.save();
    p.translate(leftM - 40, plotRect.center().y() + 30);
    p.rotate(-90);
    p.drawText(QRect(-60, -8, 120, 16), Qt::AlignCenter, QString::number(ymax, 'g', 3));
    p.restore();

    QFont axisLabelFont = font();
    axisLabelFont.setPointSize(std::max(7, axisLabelFont.pointSize() - 1));
    p.setFont(axisLabelFont);
    auto niceObjName = [](const QString& raw, const QString& fallback) -> QString {
      if (raw.isEmpty()) return fallback;
      QString s = raw;
      s.replace('_', ' ');
      if (!s.isEmpty()) s[0] = s[0].toUpper();
      return s;
    };
    p.drawText(QRect(plotRect.left(), plotRect.bottom() + 18, plotRect.width(), 14),
               Qt::AlignCenter, niceObjName(obj1Name_, "Objective 1"));

    // Y axis title (rotated 90 degrees) -- was missing entirely before;
    // mirrors the same rotated-text convention every other chart here uses.
    p.save();
    p.translate(14, plotRect.center().y());
    p.rotate(-90.0);
    p.drawText(QRect(-plotRect.height() / 2, -8, plotRect.height(), 16),
               Qt::AlignCenter, niceObjName(obj2Name_, "Objective 2"));
    p.restore();
    p.setFont(axisLabelFont);

    p.setPen(QPen(dotColor, 1.4));
    p.setBrush(dotColor);
    for (int i = 0; i < f1_.size(); ++i) {
      const QPointF pt = mapPt(f1_[i], f2_[i]);
      p.drawEllipse(pt, 3.6, 3.6);
    }
  }

private:
  QVector<double> f1_, f2_;
  QString obj1Name_, obj2Name_;
  QString title_, subtitle_;
  ThemeMode themeMode_ = ThemeMode::Dark;
  GridDensity gridDensity_ = GridDensity::Medium;
};


inline QString lastSummaryField(const QString& text, const QString& prefix) {
  const QStringList lines = text.split(QRegularExpression(R"(\r\n|\n|\r)"));
  for (int i = lines.size() - 1; i >= 0; --i) {
    const QString ln = lines[i].trimmed();
    if (ln.startsWith(prefix, Qt::CaseInsensitive)) {
      return ln.mid(prefix.size()).trimmed().simplified();
    }
  }
  return QString();
}

inline bool parseConvergenceCsvFile(const QString& path,
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
