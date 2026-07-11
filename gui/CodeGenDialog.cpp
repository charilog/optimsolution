#include "CodeGenDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QSplitter>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QFont>
#include <QRegularExpression>
#include <QKeySequence>
#include <QShortcut>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDateTime>

namespace optimsolution {

// ──────────────────────────────────────────────────────────────────────
// Pending-rebuild flag helpers
// ──────────────────────────────────────────────────────────────────────
static QString pendingFlagPath(const QString& root) {
    return QDir(root).filePath(".rebuild_pending");
}
static void writePendingFlag(const QString& root) {
    if (root.isEmpty()) return;
    QFile f(pendingFlagPath(root));
    if (f.open(QIODevice::WriteOnly | QIODevice::Text))
        QTextStream(&f) << QDateTime::currentDateTime().toString(Qt::ISODate) << "\n";
}
// Saves the .h/.cpp paths of the generated file so that the startup
// error dialog can offer "Open in Code Editor" directly.
static void writePendingSources(const QString& root,
                                 const QString& hPath,
                                 const QString& cppPath) {
    if (root.isEmpty()) return;
    QFile f(QDir(root).filePath(".rebuild_sources.txt"));
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream ts(&f);
        ts << hPath   << "\n";
        ts << cppPath << "\n";
    }
}
static void clearPendingFlag(const QString& root) {
    if (root.isEmpty()) return;
    QFile::remove(pendingFlagPath(root));
}



// ──────────────────────────────────────────────────────────────────────
// CodeEditorDialog
// ──────────────────────────────────────────────────────────────────────
CodeEditorDialog::CodeEditorDialog(const QString& hPath, const QString& cppPath,
                                    const QString& projectRoot, QWidget* parent)
    : QDialog(parent), hPath_(hPath), cppPath_(cppPath), projectRoot_(projectRoot)
{
    const QString title = QFileInfo(cppPath).baseName();
    setWindowTitle(QString("Code Editor — %1").arg(title));
    setMinimumSize(900, 680);
    setSizeGripEnabled(true);

    QFont mono;
    mono.setFamily("Courier New");
    mono.setPointSize(10);
    mono.setStyleHint(QFont::TypeWriter);
    mono.setFixedPitch(true);

    // ── Tabs: header | source | build output ──
    tabs_ = new QTabWidget(this);

    hEditor_   = new QPlainTextEdit(tabs_);
    cppEditor_ = new QPlainTextEdit(tabs_);

    hEditor_->setFont(mono);
    cppEditor_->setFont(mono);

    // Line numbers feel — monospace + tab stop 4 spaces
    const int tabStop = 4 * QFontMetrics(mono).horizontalAdvance(' ');
    hEditor_->setTabStopDistance(tabStop);
    cppEditor_->setTabStopDistance(tabStop);

    loadFile(hPath_,   hEditor_);
    loadFile(cppPath_, cppEditor_);

    tabs_->addTab(hEditor_,   QFileInfo(hPath_).fileName());
    tabs_->addTab(cppEditor_, QFileInfo(cppPath_).fileName());

    // ── Toolbar ──
    saveBtn_  = new QPushButton("💾  Save",  this);
    buildBtn_ = new QPushButton("🔄  Save & Restart", this);
    saveBtn_->setToolTip("Save current file  (Ctrl+S)");
    buildBtn_->setToolTip("Save all files and restart — the app will rebuild automatically on startup");
    buildBtn_->setDefault(false);
    statusLbl_ = new QLabel("", this);
    statusLbl_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    auto* closeBtn = new QPushButton("Close", this);
    auto* toolbar  = new QHBoxLayout();
    toolbar->addWidget(saveBtn_);
    toolbar->addWidget(buildBtn_);
    toolbar->addWidget(statusLbl_, 1);
    toolbar->addWidget(closeBtn);

    auto* root = new QVBoxLayout(this);
    root->addWidget(tabs_, 1);
    root->addLayout(toolbar);

    // Ctrl+S saves
    auto* saveShortcut = new QShortcut(QKeySequence::Save, this);
    connect(saveShortcut, &QShortcut::activated, this, &CodeEditorDialog::onSave);

    connect(saveBtn_,  &QPushButton::clicked, this, &CodeEditorDialog::onSave);
    connect(buildBtn_, &QPushButton::clicked, this, &CodeEditorDialog::onBuild);
    connect(closeBtn,  &QPushButton::clicked, this, &QDialog::accept);
    connect(tabs_,     &QTabWidget::currentChanged, this, &CodeEditorDialog::onTabChanged);

    // Track unsaved changes
    connect(hEditor_,   &QPlainTextEdit::textChanged, this, [this]{ hDirty_   = true; updateTitle(); });
    connect(cppEditor_, &QPlainTextEdit::textChanged, this, [this]{ cppDirty_ = true; updateTitle(); });
}

void CodeEditorDialog::updateTitle() {
    const QString h   = hDirty_   ? "* " + QFileInfo(hPath_).fileName()   : QFileInfo(hPath_).fileName();
    const QString cpp = cppDirty_ ? "* " + QFileInfo(cppPath_).fileName() : QFileInfo(cppPath_).fileName();
    tabs_->setTabText(0, h);
    tabs_->setTabText(1, cpp);
}

void CodeEditorDialog::loadFile(const QString& path, QPlainTextEdit* editor) {
    QFile f(path);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text))
        editor->setPlainText(QTextStream(&f).readAll());
    else
        editor->setPlainText(QString("// Could not open: %1").arg(path));
}

bool CodeEditorDialog::saveFile(const QString& path, QPlainTextEdit* editor) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        statusLbl_->setText("⚠ Cannot save: " + QFileInfo(path).fileName());
        return false;
    }
    QTextStream(&f) << editor->toPlainText();
    return true;
}

void CodeEditorDialog::onSave() {
    bool ok = true;
    if (hDirty_   && saveFile(hPath_,   hEditor_))   { hDirty_   = false; }
    else if (hDirty_) ok = false;
    if (cppDirty_ && saveFile(cppPath_, cppEditor_)) { cppDirty_ = false; }
    else if (cppDirty_) ok = false;
    updateTitle();
    statusLbl_->setText(ok ? "✓ Saved." : "⚠ Save failed.");
}

void CodeEditorDialog::onBuild()
{
    // ── "Save & Restart" logic ────────────────────────────────────────
    // cmake --build cannot replace the running exe on Windows (LNK1104),
    // so we save the source files, mark the rebuild flag and restart.
    // The startup auto-rebuild picks it up and runs cmake after the old
    // process has fully exited.
    onSave();

    // Make sure the rebuild flag is set (generation may have set it already).
    writePendingFlag(projectRoot_);

    const auto ans = QMessageBox::question(this,
        "Save & Restart",
        "Files saved.\n\n"
        "The application will restart and rebuild automatically on startup.\n\n"
        "Restart now?",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

    if (ans == QMessageBox::Yes) {
        QProcess::startDetached(QCoreApplication::applicationFilePath(),
                                QCoreApplication::arguments());
        QCoreApplication::quit();
    } else {
        statusLbl_->setText("✓ Saved. Restart when ready to rebuild.");
    }
}

// Kept for MOC — not used in the new flow but declared in the header.
void CodeEditorDialog::onBuildOutput() {}
void CodeEditorDialog::onBuildFinished(int, QProcess::ExitStatus) {}


void CodeEditorDialog::onTabChanged(int) {
    // nothing needed — tabs are self-managing
}

void CodeEditorDialog::markDirty() {}  // kept for MOC

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────
static QString toClassName(const QString& s) {
    // "my_method" → "MyMethod"
    QString out;
    bool cap = true;
    for (QChar c : s) {
        if (c == '_' || c == '-' || c == ' ') { cap = true; continue; }
        out += cap ? c.toUpper() : c;
        cap = false;
    }
    return out.isEmpty() ? "MyClass" : out;
}

static QString toSnake(const QString& s) {
    return s.trimmed().toLower().replace(QRegularExpression("[^a-z0-9_]"), "_");
}

// ──────────────────────────────────────────────────────────────────────
// DeleteItemDialog
// ──────────────────────────────────────────────────────────────────────
DeleteItemDialog::DeleteItemDialog(bool isMethod, const QString& projectRoot,
                                    QWidget* parent)
    : QDialog(parent), isMethod_(isMethod), projectRoot_(projectRoot)
{
    setWindowTitle(isMethod_ ? "Delete method" : "Delete problem");
    setMinimumSize(420, 380);

    const QString factoryPath = QDir(projectRoot_).filePath("src/factory.cpp");
    const QStringList names   = readNamesFromFactory(factoryPath);

    auto* lbl = new QLabel(
        isMethod_
          ? "Select the method to delete.\nThis will remove its .h/.cpp files,\n"
            "its entry in factory.cpp and CMakeLists.txt."
          : "Select the problem to delete.\nThis will remove its .h/.cpp files,\n"
            "its entry in factory.cpp and CMakeLists.txt.",
        this);
    lbl->setWordWrap(true);

    list_ = new QListWidget(this);
    list_->setAlternatingRowColors(true);
    for (const QString& n : names)
        list_->addItem(n);

    statusLbl_ = new QLabel("", this);
    statusLbl_->setWordWrap(true);

    deleteBtn_ = new QPushButton("Delete selected", this);
    deleteBtn_->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
    auto* cancelBtn = new QPushButton("Cancel", this);

    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(statusLbl_, 1);
    btnRow->addWidget(deleteBtn_);
    btnRow->addWidget(cancelBtn);

    auto* root = new QVBoxLayout(this);
    root->addWidget(lbl);
    root->addWidget(list_, 1);
    root->addLayout(btnRow);

    connect(deleteBtn_, &QPushButton::clicked, this, &DeleteItemDialog::onDelete);
    connect(cancelBtn,  &QPushButton::clicked, this, &QDialog::reject);
}

QStringList DeleteItemDialog::readNamesFromFactory(const QString& factoryPath) const {
    // ── Primary: parse factory.cpp source ────────────────────────────────────
    QStringList names;
    QFile f(factoryPath);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QString src = QTextStream(&f).readAll();
        const QString fnName = isMethod_ ? "listMethodNames()" : "listProblemNames()";
        int fnPos = src.indexOf(fnName);
        if (fnPos >= 0) {
            // Accept "return {" or "return{" with optional whitespace
            static const QRegularExpression reRet(R"(return\s*\{)");
            auto m = reRet.match(src, fnPos);
            if (m.hasMatch()) {
                int retEnd   = m.capturedEnd();
                int closePos = src.indexOf("};", retEnd);
                if (closePos > retEnd) {
                    const QString block = src.mid(retEnd, closePos - retEnd);
                    for (const QString& tok : block.split(',', Qt::SkipEmptyParts)) {
                        QString n = tok;
                        n.remove(QRegularExpression(R"([\s"'\n\r])"));
                        if (!n.isEmpty()) names.append(n);
                    }
                }
            }
        }
    }

    // ── Fallback: scan filesystem if factory parsing gave nothing ─────────────
    if (names.isEmpty() && !projectRoot_.isEmpty()) {
        const QString subdir = isMethod_ ? "src/methods" : "src/problems";
        QDir dir(QDir(projectRoot_).filePath(subdir));
        const QStringList headers = dir.entryList({"*.h"}, QDir::Files);
        for (const QString& h : headers)
            names.append(QFileInfo(h).baseName());
    }

    names.sort(Qt::CaseInsensitive);
    names.removeDuplicates();
    return names;
}

void DeleteItemDialog::onDelete() {
    const auto sel = list_->selectedItems();
    if (sel.isEmpty()) { statusLbl_->setText("⚠ Select an item first."); return; }
    const QString name = sel.first()->text().trimmed();

    const auto answer = QMessageBox::question(this,
        "Confirm deletion",
        QString("Delete \"%1\"?\n\nThis will:\n"
                "• Delete %2/%3.h and %3.cpp\n"
                "• Remove entry from factory.cpp\n"
                "• Remove entry from CMakeLists.txt\n\n"
                "⚠ This cannot be undone.")
            .arg(name, isMethod_ ? "src/methods" : "src/problems", name),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) return;

    QStringList log;
    bool ok = true;

    // 1. Delete source files
    {
        QString err;
        if (deleteSourceFiles(name, &err)) log << "✓ Source files deleted.";
        else { log << "⚠ Files: " + err; ok = false; }
    }

    // 2. Unpatch factory.cpp
    {
        const QString factoryPath = QDir(projectRoot_).filePath("src/factory.cpp");
        QString err;
        if (unpatchFactory(factoryPath, name, &err)) log << "✓ factory.cpp updated.";
        else { log << "⚠ factory.cpp: " + err; ok = false; }
    }

    // 3. Unpatch CMakeLists.txt
    {
        const QString cmakePath = QDir(projectRoot_).filePath("CMakeLists.txt");
        QString err;
        if (unpatchCMake(cmakePath, name, &err)) log << "✓ CMakeLists.txt updated.";
        else { log << "⚠ CMakeLists.txt: " + err; ok = false; }
    }

    QMessageBox::information(this, "Deletion complete",
        log.join("\n") + "\n\n⚠ REBUILD REQUIRED\ncmake --build build --config Release");

    if (ok) {
        writePendingFlag(projectRoot_);   // rebuild needed — detected at next startup
        delete list_->takeItem(list_->row(sel.first()));

        const auto ans = QMessageBox::question(this,
            "Restart required",
            QString("\"%1\" deleted.\n\n"
                    "The application must restart and rebuild to reflect the change.\n\n"
                    "Restart now? (Build will run automatically on startup)")
                .arg(name),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        if (ans == QMessageBox::Yes) {
            QProcess::startDetached(QCoreApplication::applicationFilePath(),
                                    QCoreApplication::arguments());
            QCoreApplication::quit();
        }
    }
    statusLbl_->setText(ok ? "✓ Done." : "⚠ Partial — check log.");
}

bool DeleteItemDialog::deleteSourceFiles(const QString& name, QString* err) const {
    const QString subdir = isMethod_ ? "src/methods" : "src/problems";
    const QString base   = QDir(projectRoot_).filePath(subdir + "/" + name);
    const QStringList paths = { base + ".h", base + ".cpp" };
    QStringList missing;
    for (const QString& p : paths) {
        if (QFileInfo::exists(p)) {
            if (!QFile::remove(p)) { if (err) *err = "Cannot delete " + p; return false; }
        } else {
            missing << QFileInfo(p).fileName();
        }
    }
    if (!missing.isEmpty() && err)
        *err = "Not found (already deleted?): " + missing.join(", ");
    return true;  // treat missing files as non-fatal
}

bool DeleteItemDialog::unpatchFactory(const QString& factoryPath,
                                       const QString& name, QString* err) const {
    QFile f(factoryPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (err) *err = "Cannot open " + factoryPath;
        return false;
    }
    QString code = QTextStream(&f).readAll();
    f.close();

    const QString subdir = isMethod_ ? "methods" : "problems";

    // Remove #include line
    code.remove(QRegularExpression(
        QString(R"(\n?#include\s*"%1/%2\.h"\n?)").arg(subdir, name)));

    // Remove if-line in make function
    code.remove(QRegularExpression(
        QString(R"(\n?\tif\s*\(name\s*==\s*"%1"\)[^\n]*\n?)").arg(name)));

    // Remove name from list (handles "name", "name" with various spacing)
    code.remove(QRegularExpression(
        QString(R"(\s*,?\s*"%1"\s*,?)").arg(name)));

    QFile fw(factoryPath);
    if (!fw.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (err) *err = "Cannot write " + factoryPath;
        return false;
    }
    QTextStream(&fw) << code;
    return true;
}

bool DeleteItemDialog::unpatchCMake(const QString& cmakePath,
                                     const QString& name, QString* err) const {
    QFile f(cmakePath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (err) *err = "Cannot open " + cmakePath;
        return false;
    }
    QString code = QTextStream(&f).readAll();
    f.close();

    const QString subdir = isMethod_ ? "methods" : "problems";

    // Line-by-line removal: immune to \r\n / \n mixed endings.
    const QString sep = code.contains("\r\n") ? "\r\n" : "\n";
    QStringList lines = code.split(sep);
    const QString target = QString("src/%1/%2.cpp").arg(subdir, name);
    lines.removeIf([&](const QString& line) {
        return line.trimmed() == target;
    });
    code = lines.join(sep);

    QFile fw(cmakePath);
    if (!fw.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (err) *err = "Cannot write " + cmakePath;
        return false;
    }
    QTextStream(&fw) << code;
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// NewMethodDialog
// ──────────────────────────────────────────────────────────────────────
NewMethodDialog::NewMethodDialog(const QString& projectRoot, QWidget* parent)
    : QDialog(parent), projectRoot_(projectRoot)
{
    setWindowTitle("New Method Wizard");
    setMinimumSize(820, 620);

    // ── Identity fields ──
    auto* idBox  = new QGroupBox("Identity", this);
    auto* idForm = new QFormLayout(idBox);
    shortNameEdit_ = new QLineEdit("mymethod", idBox);
    shortNameEdit_->setPlaceholderText("e.g. de, pso, ga  (short, no spaces)");
    fullNameEdit_  = new QLineEdit("My Optimisation Method", idBox);
    classNameEdit_ = new QLineEdit("MyMethod", idBox);
    classNameEdit_->setPlaceholderText("C++ class name (auto-filled)");
    idForm->addRow("Short name:", shortNameEdit_);
    idForm->addRow("Full name:",  fullNameEdit_);
    idForm->addRow("Class name:", classNameEdit_);

    // ── Parameter table ──
    auto* paramBox = new QGroupBox("Parameters (configure() will read these from .cfg)", this);
    auto* paramLay = new QVBoxLayout(paramBox);

    paramTable_ = new QTableWidget(0, 4, paramBox);
    paramTable_->setHorizontalHeaderLabels({"Name", "Type", "Default", "Comment"});
    paramTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    paramTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    paramTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    paramTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    paramTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    paramTable_->setAlternatingRowColors(true);

    // Pre-populate with DE-like defaults
    auto addDefaultRow = [&](const QString& n, const QString& t,
                              const QString& def, const QString& cmt) {
        const int r = paramTable_->rowCount();
        paramTable_->insertRow(r);
        paramTable_->setItem(r, 0, new QTableWidgetItem(n));
        auto* typeCombo = new QComboBox(paramTable_);
        typeCombo->addItems({"double","int","bool","string"});
        typeCombo->setCurrentText(t);
        paramTable_->setCellWidget(r, 1, typeCombo);
        paramTable_->setItem(r, 2, new QTableWidgetItem(def));
        paramTable_->setItem(r, 3, new QTableWidgetItem(cmt));
    };
    addDefaultRow("F",  "double", "0.6",    "Scale factor");
    addDefaultRow("CR", "double", "0.9",    "Crossover rate");
    addDefaultRow("population", "int", "50", "Population size");

    auto* pBtnRow = new QHBoxLayout();
    auto* addBtn = new QPushButton("+ Add parameter", paramBox);
    auto* remBtn = new QPushButton("- Remove selected", paramBox);
    pBtnRow->addWidget(addBtn);
    pBtnRow->addWidget(remBtn);
    pBtnRow->addStretch();
    paramLay->addWidget(paramTable_);
    paramLay->addLayout(pBtnRow);

    // ── Preview ──
    auto* previewBox = new QGroupBox("Code preview (.h)", this);
    auto* previewLay = new QVBoxLayout(previewBox);
    previewEdit_ = new QTextEdit(previewBox);
    previewEdit_->setReadOnly(true);
    QFont mono("Courier New", 9);
    mono.setStyleHint(QFont::TypeWriter);
    previewEdit_->setFont(mono);
    previewLay->addWidget(previewEdit_);

    // ── Status + buttons ──
    statusLbl_   = new QLabel("", this);
    previewBtn_  = new QPushButton("Preview code", this);
    generateBtn_ = new QPushButton("Generate files", this);
    generateBtn_->setDefault(true);
    auto* cancelBtn = new QPushButton("Cancel", this);

    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(statusLbl_, 1);
    btnRow->addWidget(previewBtn_);
    btnRow->addWidget(generateBtn_);
    btnRow->addWidget(cancelBtn);

    // ── Splitter: top (id+params) / bottom (preview) ──
    auto* topWidget = new QWidget(this);
    auto* topLay    = new QVBoxLayout(topWidget);
    topLay->setContentsMargins(0,0,0,0);
    topLay->addWidget(idBox);
    topLay->addWidget(paramBox, 1);

    auto* splitter = new QSplitter(Qt::Vertical, this);
    splitter->addWidget(topWidget);
    splitter->addWidget(previewBox);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 1);

    auto* root = new QVBoxLayout(this);
    root->addWidget(splitter, 1);
    root->addLayout(btnRow);

    connect(shortNameEdit_, &QLineEdit::textChanged, this, &NewMethodDialog::onShortNameChanged);
    connect(addBtn,         &QPushButton::clicked,   this, &NewMethodDialog::onAddParam);
    connect(remBtn,         &QPushButton::clicked,   this, &NewMethodDialog::onRemoveParam);
    connect(previewBtn_,    &QPushButton::clicked,   this, &NewMethodDialog::onPreview);
    connect(generateBtn_,   &QPushButton::clicked,   this, &NewMethodDialog::onGenerate);
    connect(cancelBtn,      &QPushButton::clicked,   this, &QDialog::reject);
}

void NewMethodDialog::onShortNameChanged(const QString& s) {
    classNameEdit_->setText(toClassName(s));
}

void NewMethodDialog::onAddParam() {
    const int r = paramTable_->rowCount();
    paramTable_->insertRow(r);
    paramTable_->setItem(r, 0, new QTableWidgetItem("param" + QString::number(r+1)));
    auto* typeCombo = new QComboBox(paramTable_);
    typeCombo->addItems({"double","int","bool","string"});
    paramTable_->setCellWidget(r, 1, typeCombo);
    paramTable_->setItem(r, 2, new QTableWidgetItem("0.0"));
    paramTable_->setItem(r, 3, new QTableWidgetItem(""));
    paramTable_->scrollToBottom();
}

void NewMethodDialog::onRemoveParam() {
    const QList<QTableWidgetSelectionRange> sel = paramTable_->selectedRanges();
    QList<int> rows;
    for (const auto& r : sel)
        for (int i = r.topRow(); i <= r.bottomRow(); ++i)
            if (!rows.contains(i)) rows.append(i);
    std::sort(rows.rbegin(), rows.rend());
    for (int r : rows) paramTable_->removeRow(r);
}

QList<ParamDef> NewMethodDialog::collectParams() const {
    QList<ParamDef> out;
    for (int r = 0; r < paramTable_->rowCount(); ++r) {
        ParamDef p;
        p.name       = paramTable_->item(r,0) ? paramTable_->item(r,0)->text().trimmed() : "";
        auto* combo  = qobject_cast<QComboBox*>(paramTable_->cellWidget(r,1));
        p.type       = combo ? combo->currentText() : "double";
        p.defaultVal = paramTable_->item(r,2) ? paramTable_->item(r,2)->text().trimmed() : "0";
        p.comment    = paramTable_->item(r,3) ? paramTable_->item(r,3)->text().trimmed() : "";
        if (!p.name.isEmpty()) out.append(p);
    }
    return out;
}

QString NewMethodDialog::buildHeader(const QString& shortName,
                                      const QString& fullName,
                                      const QString& className,
                                      const QList<ParamDef>& params) const {
    QString s;
    QTextStream ts(&s);
    ts << "#pragma once\n";
    ts << "#include \"optimizer.h\"\n";
    ts << "#include <vector>\n#include <random>\n#include <string>\n#include <cmath>\n\n";
    ts << "namespace optimsolution {\n\n";
    ts << "class " << className << " : public Optimizer {\n";
    ts << "public:\n";
    ts << "    " << className << "() = default;\n";
    ts << "    ~" << className << "() override = default;\n\n";
    ts << "    std::string methodShortName() const override { return \"" << shortName << "\"; }\n";
    ts << "    std::string methodFullName()  const override { return \"" << fullName  << "\"; }\n\n";
    ts << "    void configure(const MethodConfig& mc) override;\n\n";
    ts << "protected:\n";
    ts << "    void init() override;\n";
    ts << "    void one_iteration() override;\n";
    ts << "    void end() override;\n\n";
    ts << "private:\n";
    ts << "    using Vec = std::vector<double>;\n\n";
    ts << "    std::vector<Vec>    X_;\n";
    ts << "    std::vector<double> FX_;\n\n";
    // params
    for (const ParamDef& p : params) {
        QString cppType = p.type;
        if (cppType == "string") cppType = "std::string";
        QString defVal = p.defaultVal;
        if (cppType == "std::string") defVal = "\"" + defVal + "\"";
        ts << "    " << cppType << " " << toSnake(p.name) << "_{" << defVal << "}";
        if (!p.comment.isEmpty()) ts << ";  // " << p.comment;
        ts << ";\n";
    }
    ts << "};\n\n} // namespace optimsolution\n";
    return s;
}

QString NewMethodDialog::buildSource(const QString& shortName,
                                      const QString& className,
                                      const QList<ParamDef>& params) const {
    QString s;
    QTextStream ts(&s);
    const QString lower = shortName.toLower();
    ts << "#include \"" << lower << ".h\"\n";
    ts << "#include \"init.h\"\n";
    ts << "#include <cstdio>\n#include <cmath>\n#include <limits>\n\n";
    ts << "namespace optimsolution {\n\n";

    // configure()
    ts << "void " << className << "::configure(const MethodConfig& mc) {\n";
    ts << "    auto trim = [](std::string s){\n";
    ts << "        size_t a=0,b=s.size();\n";
    ts << "        while(a<b&&std::isspace((unsigned char)s[a]))++a;\n";
    ts << "        while(b>a&&std::isspace((unsigned char)s[b-1]))--b;\n";
    ts << "        return s.substr(a,b-a);\n    };\n\n";
    for (const ParamDef& p : params) {
        const QString sn = toSnake(p.name);
        if (p.type == "double") {
            ts << "    { bool ok=false; double v=mc.getDbl(\"" << p.name << "\"," << sn
               << "_); if(std::isfinite(v)) " << sn << "_=v; }\n";
        } else if (p.type == "int") {
            ts << "    { int v=mc.getInt(\"" << p.name << "\"," << sn << "_); "
               << sn << "_=v; }\n";
        } else if (p.type == "bool") {
            ts << "    { std::string bs=trim(mc.getStr(\"" << p.name << "\",\"\")); "
               << "if(bs==\"1\"||bs==\"true\"||bs==\"on\"||bs==\"yes\") " << sn << "_=true;\n"
               << "      else if(bs==\"0\"||bs==\"false\"||bs==\"off\"||bs==\"no\") " << sn << "_=false; }\n";
        } else {
            ts << "    { std::string v=trim(mc.getStr(\"" << p.name << "\",\"\")); "
               << "if(!v.empty()) " << sn << "_=v; }\n";
        }
    }
    ts << "}\n\n";

    // init()
    ts << "void " << className << "::init() {\n";
    ts << "    if (!prob_) return;\n";
    ts << "    const int D = prob_->dimension();\n";
    ts << "    const int N = std::max(4, population());\n\n";
    ts << "    Initializer initSampler;\n";
    ts << "    initSampler.configure(initopt_);\n";
    ts << "    X_ = initSampler.samplePopulation(*prob_, rng_, N);\n\n";
    ts << "    FX_.assign(N, std::numeric_limits<double>::infinity());\n";
    ts << "    best_f_ = std::numeric_limits<double>::infinity();\n";
    ts << "    best_x_.assign(D, 0.0);\n\n";
    ts << "    for (int i=0; i<N; ++i) {\n";
    ts << "        FX_[i] = prob_->evaluate(X_[i]);\n";
    ts << "        if (FX_[i] < best_f_) { best_f_=FX_[i]; best_x_=X_[i]; }\n";
    ts << "        if (prob_->calls() >= max_evals_) break;\n";
    ts << "    }\n";
    ts << "    printBest();\n";
    ts << "}\n\n";

    // one_iteration()
    ts << "void " << className << "::one_iteration() {\n";
    ts << "    if (!prob_) return;\n";
    ts << "    // TODO: implement one iteration of " << className << "\n";
    ts << "    //       Update X_, FX_, best_f_, best_x_ as needed.\n";
    ts << "    printBest();\n";
    ts << "    updateStop(FX_);\n";
    ts << "}\n\n";

    // end()
    ts << "void " << className << "::end() {\n";
    ts << "    // TODO: optional end-of-run refinement\n";
    ts << "    printBest();\n";
    ts << "}\n\n";

    ts << "} // namespace optimsolution\n";
    return s;
}

void NewMethodDialog::onPreview() {
    const QString sn   = toSnake(shortNameEdit_->text().trimmed());
    const QString fn   = fullNameEdit_->text().trimmed();
    const QString cn   = classNameEdit_->text().trimmed();
    const auto params  = collectParams();
    if (sn.isEmpty() || cn.isEmpty()) {
        statusLbl_->setText("⚠ Fill in short name and class name first.");
        return;
    }
    const QString hdr = buildHeader(sn, fn, cn, params);
    const QString src = buildSource(sn, cn, params);
    previewEdit_->setPlainText("// ── " + cn + ".h ────────────────\n" + hdr +
                                "\n\n// ── " + cn + ".cpp ──────────────\n" + src);
    statusLbl_->setText("Preview updated.");
}

// ──────────────────────────────────────────────────────────────────────
// Shared file-patching helpers
// ──────────────────────────────────────────────────────────────────────

// Patch factory.cpp: add #include, if-line in makeProblem/makeMethod, and name in list.
static bool patchFactory(const QString& factoryPath,
                          const QString& shortName,   // e.g. "mymethod"
                          const QString& className,   // e.g. "MyMethod"
                          bool isMethod,
                          QString* errOut)
{
    QFile f(factoryPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (errOut) *errOut = "Cannot open " + factoryPath;
        return false;
    }
    QString code = QTextStream(&f).readAll();
    f.close();

    const QString subdir   = isMethod ? "methods" : "problems";
    const QString include  = QString("#include \"%1/%2.h\"").arg(subdir, shortName);
    const QString listFn   = isMethod ? "listMethodNames()" : "listProblemNames()";
    const QString makeFn   = isMethod ? "makeMethod" : "makeProblem";
    const QString ifLine   = QString("\tif (name == \"%1\") return std::make_unique<optimsolution::%2>();")
                              .arg(shortName, className);

    // 1. Add #include after the last include of that subdir.
    if (!code.contains(include)) {
        const QString marker = isMethod ? "// methods" : "// problems";
        // Find last #include "methods/" or "problems/" line
        const QRegularExpression reInc(
            QString(R"(#include "%1/[^"]+\.h")").arg(subdir));
        int lastPos = -1;
        QRegularExpressionMatchIterator it = reInc.globalMatch(code);
        while (it.hasNext()) { auto m = it.next(); lastPos = m.capturedEnd(); }
        if (lastPos > 0) {
            code.insert(lastPos, "\n" + include);
        } else {
            // Fallback: insert before namespace
            int nsPos = code.indexOf("namespace optimsolution {");
            if (nsPos > 0) code.insert(nsPos, include + "\n");
        }
    }

    // 2. Add if-line in make function before "return nullptr;"
    if (!code.contains(ifLine)) {
        // Find the correct make function's "return nullptr;"
        const QString makeMarker = QString("std::unique_ptr<%1> %2(")
                                    .arg(isMethod ? "Optimizer" : "Problem", makeFn);
        int fnPos = code.indexOf(makeMarker);
        if (fnPos >= 0) {
            int retPos = code.indexOf("    return nullptr;", fnPos);
            if (retPos > 0)
                code.insert(retPos, ifLine + "\n");
        }
    }

    // 3. Add name in listMethodNames()/listProblemNames() function.
    // Search specifically within the list function body to avoid false positives.
    {
        const QString listMarker = QString("std::vector<std::string> %1").arg(listFn);
        int fnPos = code.indexOf(listMarker);
        if (fnPos >= 0) {
            // Find closing }; of this function
            int closePos = code.indexOf("};", fnPos);
            if (closePos > 0) {
                // Check if name is already in the list body (not just anywhere in file)
                const QString listBody = code.mid(fnPos, closePos - fnPos);
                // Look for the name as a standalone string in the list
                const bool alreadyInList = listBody.contains("\"" + shortName + "\"");
                if (!alreadyInList) {
                    // Insert before closing }; — find last '"' before it
                    int lastQuote = code.lastIndexOf('"', closePos - 1);
                    if (lastQuote > fnPos)
                        code.insert(lastQuote + 1, ",\"" + shortName + "\"");
                    else
                        code.insert(closePos, "\t\"" + shortName + "\",\n");
                }
            }
        }
    }

    QFile fw(factoryPath);
    if (!fw.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errOut) *errOut = "Cannot write " + factoryPath;
        return false;
    }
    QTextStream(&fw) << code;
    return true;
}

// Patch CMakeLists.txt: add src/methods/<name>.cpp or src/problems/<name>.cpp.
static bool patchCMake(const QString& cmakePath,
                        const QString& shortName,
                        bool isMethod,
                        QString* errOut)
{
    QFile f(cmakePath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (errOut) *errOut = "Cannot open " + cmakePath;
        return false;
    }
    QString code = QTextStream(&f).readAll();
    f.close();

    const QString subdir  = isMethod ? "methods" : "problems";
    const QString newLine = QString("    src/%1/%2.cpp").arg(subdir, shortName);

    if (code.contains(newLine)) return true;  // already there

    // Line-by-line insertion: find last entry of the target subdir,
    // insert the new line immediately after it.
    // This approach is immune to \r\n / \n mixed endings.
    const QString sep = code.contains("\r\n") ? "\r\n" : "\n";
    QStringList lines = code.split(sep);
    const QString subdirPrefix = "src/" + subdir + "/";
    int lastIdx = -1;
    for (int i = 0; i < lines.size(); ++i) {
        const QString trimmed = lines[i].trimmed();
        if (trimmed.startsWith(subdirPrefix) && trimmed.endsWith(".cpp"))
            lastIdx = i;
    }

    if (lastIdx >= 0) {
        lines.insert(lastIdx + 1, newLine);
        code = lines.join(sep);
    } else {
        if (errOut) *errOut = "Could not find insertion point in CMakeLists.txt";
        return false;
    }

    QFile fw(cmakePath);
    if (!fw.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errOut) *errOut = "Cannot write " + cmakePath;
        return false;
    }
    QTextStream(&fw) << code;
    return true;
}

bool NewMethodDialog::writeFiles(const QString& header, const QString& source,
                                  const QString& className) {
    const QString lower = className.toLower();
    QString dir = projectRoot_.isEmpty()
                  ? QDir::currentPath()
                  : QDir(projectRoot_).filePath("src/methods");

    dir = QFileDialog::getExistingDirectory(this,
            "Choose output directory for " + className, dir);
    if (dir.isEmpty()) return false;

    const QString hPath = QDir(dir).filePath(lower + ".h");
    const QString cPath = QDir(dir).filePath(lower + ".cpp");

    for (const auto& [path, content] : std::initializer_list<std::pair<QString,QString>>{
            {hPath, header}, {cPath, source}}) {
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::critical(this, "Write error",
                "Cannot write: " + path + "\n" + f.errorString());
            return false;
        }
        QTextStream ts(&f);
        ts << content;
    }
    generatedH_   = hPath;
    generatedCpp_ = cPath;
    return true;
}

void NewMethodDialog::onGenerate() {
    const QString sn = toSnake(shortNameEdit_->text().trimmed());
    const QString fn = fullNameEdit_->text().trimmed();
    const QString cn = classNameEdit_->text().trimmed();
    if (sn.isEmpty()) { statusLbl_->setText("⚠ Short name is required."); return; }
    if (cn.isEmpty()) { statusLbl_->setText("⚠ Class name is required."); return; }

    const auto params = collectParams();
    const QString hdr = buildHeader(sn, fn, cn, params);
    const QString src = buildSource(sn, cn, params);

    if (!writeFiles(hdr, src, cn)) return;

    // Patch factory.cpp
    QStringList patchLog;
    const QString factoryPath = projectRoot_.isEmpty() ? QString()
                                : QDir(projectRoot_).filePath("src/factory.cpp");
    if (!factoryPath.isEmpty() && QFileInfo::exists(factoryPath)) {
        QString err;
        if (patchFactory(factoryPath, sn, cn, true, &err))
            patchLog << "✓ factory.cpp updated";
        else
            patchLog << "⚠ factory.cpp: " + err;
    } else {
        patchLog << "⚠ factory.cpp not found (patch manually)";
    }

    // Patch CMakeLists.txt
    const QString cmakePath = projectRoot_.isEmpty() ? QString()
                              : QDir(projectRoot_).filePath("CMakeLists.txt");
    if (!cmakePath.isEmpty() && QFileInfo::exists(cmakePath)) {
        QString err;
        if (patchCMake(cmakePath, sn, true, &err))
            patchLog << "✓ CMakeLists.txt updated";
        else
            patchLog << "⚠ CMakeLists.txt: " + err;
    } else {
        patchLog << "⚠ CMakeLists.txt not found (patch manually)";
    }

    statusLbl_->setText("✓ Done.");
    writePendingSources(projectRoot_, generatedH_, generatedCpp_);  // for error-recovery dialog
    writePendingFlag(projectRoot_);   // signal rebuild needed
    QMessageBox::information(this, "Method generated",
        QString("Files written:\n  %1\n  %2\n\n"
                "%3\n\n"
                "⚠ REBUILD REQUIRED\n"
                "Run: cmake --build build --config Release\n\n"
                "The method will appear in the GUI only after rebuilding.\n\n"
                "Remaining manual step:\n"
                "→ Implement one_iteration() in %4.cpp")
        .arg(generatedH_, generatedCpp_, patchLog.join("\n"), sn));

    // Open code editor so user can implement one_iteration() immediately.
    accept();
    auto* editor = new CodeEditorDialog(generatedH_, generatedCpp_, projectRoot_, parentWidget());
    editor->setAttribute(Qt::WA_DeleteOnClose);
    editor->show();
}

// ──────────────────────────────────────────────────────────────────────
// NewProblemDialog
// ──────────────────────────────────────────────────────────────────────
NewProblemDialog::NewProblemDialog(const QString& projectRoot, QWidget* parent)
    : QDialog(parent), projectRoot_(projectRoot)
{
    setWindowTitle("New Problem Wizard");
    setMinimumSize(820, 580);

    // ── Identity ──
    auto* idBox  = new QGroupBox("Identity", this);
    auto* idForm = new QFormLayout(idBox);
    shortNameEdit_ = new QLineEdit("myproblem", idBox);
    shortNameEdit_->setPlaceholderText("e.g. rastrigin2, sphere5 (short, no spaces)");
    fullNameEdit_  = new QLineEdit("My Benchmark Problem", idBox);
    classNameEdit_ = new QLineEdit("MyProblem", idBox);
    idForm->addRow("Short name:", shortNameEdit_);
    idForm->addRow("Full name:",  fullNameEdit_);
    idForm->addRow("Class name:", classNameEdit_);

    // ── Geometry ──
    auto* geomBox  = new QGroupBox("Geometry", this);
    auto* geomForm = new QFormLayout(geomBox);
    fixedDimChk_ = new QCheckBox("Fixed dimension", geomBox);
    dimSpin_     = new QSpinBox(geomBox);
    dimSpin_->setRange(1, 10000);
    dimSpin_->setValue(2);
    loBoundSpin_ = new QDoubleSpinBox(geomBox);
    loBoundSpin_->setRange(-1e9, 0);
    loBoundSpin_->setValue(-5.12);
    loBoundSpin_->setDecimals(4);
    hiBoundSpin_ = new QDoubleSpinBox(geomBox);
    hiBoundSpin_->setRange(0, 1e9);
    hiBoundSpin_->setValue(5.12);
    hiBoundSpin_->setDecimals(4);
    geomForm->addRow(fixedDimChk_, dimSpin_);
    geomForm->addRow("Lower bound (per dim):", loBoundSpin_);
    geomForm->addRow("Upper bound (per dim):", hiBoundSpin_);

    // ── Optimum ──
    auto* optBox  = new QGroupBox("Known global optimum", this);
    auto* optForm = new QFormLayout(optBox);
    hasOptimumChk_ = new QCheckBox("Known", optBox);
    hasOptimumChk_->setChecked(true);
    optValSpin_ = new QDoubleSpinBox(optBox);
    optValSpin_->setRange(-1e15, 1e15);
    optValSpin_->setValue(-2.0);
    optValSpin_->setDecimals(6);
    optValSpin_->setStepType(QAbstractSpinBox::AdaptiveDecimalStepType);
    optForm->addRow("Global minimum value:", hasOptimumChk_);
    optForm->addRow("f* =", optValSpin_);

    // ── Preview ──
    auto* previewBox = new QGroupBox("Code preview (.h)", this);
    auto* previewLay = new QVBoxLayout(previewBox);
    previewEdit_ = new QTextEdit(previewBox);
    previewEdit_->setReadOnly(true);
    QFont mono("Courier New", 9);
    mono.setStyleHint(QFont::TypeWriter);
    previewEdit_->setFont(mono);
    previewLay->addWidget(previewEdit_);

    statusLbl_   = new QLabel("", this);
    previewBtn_  = new QPushButton("Preview code", this);
    generateBtn_ = new QPushButton("Generate files", this);
    generateBtn_->setDefault(true);
    auto* cancelBtn = new QPushButton("Cancel", this);

    auto* btnRow = new QHBoxLayout();
    btnRow->addWidget(statusLbl_, 1);
    btnRow->addWidget(previewBtn_);
    btnRow->addWidget(generateBtn_);
    btnRow->addWidget(cancelBtn);

    auto* topWidget = new QWidget(this);
    auto* topLay    = new QVBoxLayout(topWidget);
    topLay->setContentsMargins(0,0,0,0);
    topLay->addWidget(idBox);
    topLay->addWidget(geomBox);
    topLay->addWidget(optBox);

    auto* splitter = new QSplitter(Qt::Vertical, this);
    splitter->addWidget(topWidget);
    splitter->addWidget(previewBox);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 1);

    auto* root = new QVBoxLayout(this);
    root->addWidget(splitter, 1);
    root->addLayout(btnRow);

    connect(shortNameEdit_, &QLineEdit::textChanged, this, &NewProblemDialog::onShortNameChanged);
    connect(fixedDimChk_,  &QCheckBox::toggled,     this, &NewProblemDialog::onFixedDimToggled);
    connect(previewBtn_,   &QPushButton::clicked,   this, &NewProblemDialog::onPreview);
    connect(generateBtn_,  &QPushButton::clicked,   this, &NewProblemDialog::onGenerate);
    connect(cancelBtn,     &QPushButton::clicked,   this, &QDialog::reject);

    onFixedDimToggled(false);
}

void NewProblemDialog::onShortNameChanged(const QString& s) {
    classNameEdit_->setText(toClassName(s));
}

void NewProblemDialog::onFixedDimToggled(bool fixed) {
    dimSpin_->setEnabled(fixed);
}

QString NewProblemDialog::buildHeader(const QString& shortName,
                                       const QString& fullName,
                                       const QString& className,
                                       bool fixedDim, int dim) const {
    Q_UNUSED(shortName); Q_UNUSED(fullName);
    QString s;
    QTextStream ts(&s);
    ts << "#pragma once\n#include \"problem.h\"\n\n";
    ts << "namespace optimsolution {\n\n";
    ts << "/**\n * " << fullName << "\n";
    if (fixedDim) ts << " * Fixed dimension: " << dim << "\n";
    ts << " * Generated by OptimSolution Code Wizard\n */\n";
    ts << "class " << className << " : public Problem {\n";
    ts << "public:\n";
    ts << "    " << className << "();\n";
    ts << "    void init(int dim) override;\n\n";
    ts << "protected:\n";
    ts << "    double evaluate_core(const Vec& x) override;\n";
    ts << "    void   gradient_core(const Vec& x, Vec& g) override;\n";
    ts << "};\n\n} // namespace optimsolution\n";
    return s;
}

QString NewProblemDialog::buildSource(const QString& shortName,
                                       const QString& fullName,
                                       const QString& className,
                                       bool fixedDim, int dim,
                                       double loBound, double hiBound,
                                       bool hasOptimum, double optVal) const {
    QString s;
    QTextStream ts(&s);
    const QString lower = shortName.toLower();
    ts << "#include \"" << lower << ".h\"\n#include <cmath>\n\n";
    ts << "namespace optimsolution {\n\n";

    // Constructor
    ts << className << "::" << className << "() {\n";
    ts << "    setName(\"" << shortName << "\");\n";
    ts << "    setFullName(\"" << fullName << "\");\n";
    ts << "    setModality(\"multimodal\");\n";
    ts << "    setSeparability(\"separable\");\n";
    ts << "    setCategory(\"continuous benchmark test function\");\n";
    if (hasOptimum) {
        if (fixedDim) {
            ts << "    Vec xopt(" << dim << ", 0.0);\n";
            ts << "    setKnownGlobalOptimum(" << optVal << ", xopt);\n";
        } else {
            ts << "    // Global optimum depends on dimension; set in init() if needed.\n";
        }
    }
    ts << "}\n\n";

    // init()
    ts << "void " << className << "::init(int dim) {\n";
    if (fixedDim) {
        ts << "    // Always " << dim << "D\n";
        ts << "    Problem::init(" << dim << ");\n";
        ts << "    Vec lo(" << dim << ", " << loBound << "), hi(" << dim << ", " << hiBound << ");\n";
    } else {
        ts << "    Problem::init(dim);\n";
        ts << "    Vec lo(dim, " << loBound << "), hi(dim, " << hiBound << ");\n";
        if (hasOptimum) {
            ts << "    Vec xopt(dim, 0.0);\n";
            ts << "    setKnownGlobalOptimum(" << optVal << ", xopt);\n";
        }
    }
    ts << "    setBounds(lo, hi);\n";
    ts << "}\n\n";

    // evaluate_core()
    ts << "double " << className << "::evaluate_core(const Vec& x) {\n";
    ts << "    // TODO: implement f(x)\n";
    ts << "    // Example: simple sphere\n";
    ts << "    double sum = 0.0;\n";
    ts << "    for (double xi : x) sum += xi * xi;\n";
    ts << "    return sum";
    if (hasOptimum && optVal != 0.0)
        ts << " + " << optVal;
    ts << ";\n}\n\n";

    // gradient_core()
    ts << "void " << className << "::gradient_core(const Vec& x, Vec& g) {\n";
    ts << "    g.assign(x.size(), 0.0);\n";
    ts << "    // TODO: implement gradient of f(x)\n";
    ts << "    for (size_t i=0; i<x.size(); ++i) g[i] = 2.0 * x[i];\n";
    ts << "}\n\n";
    ts << "} // namespace optimsolution\n";
    return s;
}

void NewProblemDialog::onPreview() {
    const QString sn = toSnake(shortNameEdit_->text().trimmed());
    const QString fn = fullNameEdit_->text().trimmed();
    const QString cn = classNameEdit_->text().trimmed();
    if (sn.isEmpty() || cn.isEmpty()) {
        statusLbl_->setText("⚠ Fill in short name and class name first.");
        return;
    }
    const bool fixedDim = fixedDimChk_->isChecked();
    const int dim       = dimSpin_->value();
    const double lo     = loBoundSpin_->value();
    const double hi     = hiBoundSpin_->value();
    const bool hasOpt   = hasOptimumChk_->isChecked();
    const double optVal = optValSpin_->value();

    const QString hdr = buildHeader(sn, fn, cn, fixedDim, dim);
    const QString src = buildSource(sn, fn, cn, fixedDim, dim, lo, hi, hasOpt, optVal);
    previewEdit_->setPlainText("// ── " + cn + ".h ────────────────\n" + hdr +
                                "\n\n// ── " + cn + ".cpp ──────────────\n" + src);
    statusLbl_->setText("Preview updated.");
}

bool NewProblemDialog::writeFiles(const QString& header, const QString& source,
                                   const QString& className) {
    const QString lower = className.toLower();
    QString dir = projectRoot_.isEmpty()
                  ? QDir::currentPath()
                  : QDir(projectRoot_).filePath("src/problems");

    dir = QFileDialog::getExistingDirectory(this,
            "Choose output directory for " + className, dir);
    if (dir.isEmpty()) return false;

    const QString hPath = QDir(dir).filePath(lower + ".h");
    const QString cPath = QDir(dir).filePath(lower + ".cpp");

    for (const auto& [path, content] : std::initializer_list<std::pair<QString,QString>>{
            {hPath, header}, {cPath, source}}) {
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::critical(this, "Write error",
                "Cannot write: " + path + "\n" + f.errorString());
            return false;
        }
        QTextStream ts(&f);
        ts << content;
    }
    generatedH_   = hPath;
    generatedCpp_ = cPath;
    return true;
}

void NewProblemDialog::onGenerate() {
    const QString sn = toSnake(shortNameEdit_->text().trimmed());
    const QString fn = fullNameEdit_->text().trimmed();
    const QString cn = classNameEdit_->text().trimmed();
    if (sn.isEmpty()) { statusLbl_->setText("⚠ Short name is required."); return; }
    if (cn.isEmpty()) { statusLbl_->setText("⚠ Class name is required."); return; }

    const bool fixedDim = fixedDimChk_->isChecked();
    const int  dim      = dimSpin_->value();
    const double lo     = loBoundSpin_->value();
    const double hi     = hiBoundSpin_->value();
    const bool hasOpt   = hasOptimumChk_->isChecked();
    const double optVal = optValSpin_->value();

    const QString hdr = buildHeader(sn, fn, cn, fixedDim, dim);
    const QString src = buildSource(sn, fn, cn, fixedDim, dim, lo, hi, hasOpt, optVal);

    if (!writeFiles(hdr, src, cn)) return;

    // Patch factory.cpp
    QStringList patchLog;
    const QString factoryPath = projectRoot_.isEmpty() ? QString()
                                : QDir(projectRoot_).filePath("src/factory.cpp");
    if (!factoryPath.isEmpty() && QFileInfo::exists(factoryPath)) {
        QString err;
        if (patchFactory(factoryPath, sn, cn, false, &err))
            patchLog << "✓ factory.cpp updated";
        else
            patchLog << "⚠ factory.cpp: " + err;
    } else {
        patchLog << "⚠ factory.cpp not found (patch manually)";
    }

    // Patch CMakeLists.txt
    const QString cmakePath = projectRoot_.isEmpty() ? QString()
                              : QDir(projectRoot_).filePath("CMakeLists.txt");
    if (!cmakePath.isEmpty() && QFileInfo::exists(cmakePath)) {
        QString err;
        if (patchCMake(cmakePath, sn, false, &err))
            patchLog << "✓ CMakeLists.txt updated";
        else
            patchLog << "⚠ CMakeLists.txt: " + err;
    } else {
        patchLog << "⚠ CMakeLists.txt not found (patch manually)";
    }

    statusLbl_->setText("✓ Done.");
    writePendingSources(projectRoot_, generatedH_, generatedCpp_);  // for error-recovery dialog
    writePendingFlag(projectRoot_);   // signal rebuild needed
    QMessageBox::information(this, "Problem generated",
        QString("Files written:\n  %1\n  %2\n\n"
                "%3\n\n"
                "⚠ REBUILD REQUIRED\n"
                "Run: cmake --build build --config Release\n\n"
                "The problem will appear in the GUI only after rebuilding.\n\n"
                "Remaining manual step:\n"
                "→ Implement evaluate_core() in %4.cpp")
        .arg(generatedH_, generatedCpp_, patchLog.join("\n"), sn));

    // Open code editor so user can implement evaluate_core() immediately.
    accept();
    auto* editor = new CodeEditorDialog(generatedH_, generatedCpp_, projectRoot_, parentWidget());
    editor->setAttribute(Qt::WA_DeleteOnClose);
    editor->show();
}

} // namespace optimsolution
