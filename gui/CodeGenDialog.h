#pragma once
#include <QDialog>
#include <QTableWidget>
#include <QLineEdit>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QPushButton>
#include <QTabWidget>
#include <QListWidget>
#include <QProcess>
#include <QString>
#include <QStringList>

namespace optimsolution {

struct ParamDef {
    QString name;
    QString type;
    QString defaultVal;
    QString comment;
};

// ── Code editor dialog ────────────────────────────────────────────────
class CodeEditorDialog : public QDialog {
    Q_OBJECT
public:
    explicit CodeEditorDialog(const QString& hPath, const QString& cppPath,
                               const QString& projectRoot, QWidget* parent = nullptr);

private slots:
    void onSave();
    void onBuild();
    void onBuildOutput();
    void onBuildFinished(int exitCode, QProcess::ExitStatus);
    void onTabChanged(int index);
    void markDirty();

private:
    bool saveFile(const QString& path, QPlainTextEdit* editor);
    void loadFile(const QString& path, QPlainTextEdit* editor);
    void updateTitle();

    QTabWidget*     tabs_      = nullptr;
    QPlainTextEdit* hEditor_   = nullptr;
    QPlainTextEdit* cppEditor_ = nullptr;
    QPlainTextEdit* buildLog_  = nullptr;
    QPushButton*    saveBtn_   = nullptr;
    QPushButton*    buildBtn_  = nullptr;
    QLabel*         statusLbl_ = nullptr;

    QString   hPath_;
    QString   cppPath_;
    QString   projectRoot_;
    QProcess* buildProc_ = nullptr;
    bool      hDirty_    = false;
    bool      cppDirty_  = false;
};

// ── Delete method/problem dialog ─────────────────────────────────────
class DeleteItemDialog : public QDialog {
    Q_OBJECT
public:
    // isMethod=true → lists methods, false → lists problems
    explicit DeleteItemDialog(bool isMethod, const QString& projectRoot,
                               QWidget* parent = nullptr);

private slots:
    void onDelete();

private:
    // Parse names from factory.cpp's listMethodNames()/listProblemNames()
    QStringList readNamesFromFactory(const QString& factoryPath) const;

    // Unpatch factory.cpp: remove include, if-line, list entry
    bool unpatchFactory(const QString& factoryPath, const QString& name,
                         QString* err) const;
    // Unpatch CMakeLists.txt: remove src/methods/<name>.cpp line
    bool unpatchCMake(const QString& cmakePath, const QString& name,
                       QString* err) const;
    // Delete the .h and .cpp files
    bool deleteSourceFiles(const QString& name, QString* err) const;

    bool        isMethod_;
    QString     projectRoot_;
    QListWidget* list_      = nullptr;
    QPushButton* deleteBtn_ = nullptr;
    QLabel*      statusLbl_ = nullptr;
};


class NewMethodDialog : public QDialog {
    Q_OBJECT
public:
    explicit NewMethodDialog(const QString& projectRoot, QWidget* parent = nullptr);
    QString generatedHeaderPath() const { return generatedH_; }
    QString generatedSourcePath() const { return generatedCpp_; }

private slots:
    void onAddParam();
    void onRemoveParam();
    void onShortNameChanged(const QString& s);
    void onPreview();
    void onGenerate();

private:
    QList<ParamDef> collectParams() const;
    QString buildHeader(const QString& sn, const QString& fn,
                        const QString& cn, const QList<ParamDef>& p) const;
    QString buildSource(const QString& sn, const QString& cn,
                        const QList<ParamDef>& p) const;
    bool writeFiles(const QString& header, const QString& source,
                    const QString& className);

    QString       projectRoot_;
    QString       generatedH_;
    QString       generatedCpp_;
    QLineEdit*    shortNameEdit_  = nullptr;
    QLineEdit*    fullNameEdit_   = nullptr;
    QLineEdit*    classNameEdit_  = nullptr;
    QTableWidget* paramTable_     = nullptr;
    QTextEdit*    previewEdit_    = nullptr;
    QLabel*       statusLbl_      = nullptr;
    QPushButton*  previewBtn_     = nullptr;
    QPushButton*  generateBtn_    = nullptr;
};

// ── New Problem wizard ────────────────────────────────────────────────
class NewProblemDialog : public QDialog {
    Q_OBJECT
public:
    explicit NewProblemDialog(const QString& projectRoot, QWidget* parent = nullptr);
    QString generatedHeaderPath() const { return generatedH_; }
    QString generatedSourcePath() const { return generatedCpp_; }

private slots:
    void onFixedDimToggled(bool fixed);
    void onShortNameChanged(const QString& s);
    void onPreview();
    void onGenerate();

private:
    QString buildHeader(const QString& sn, const QString& fn,
                        const QString& cn, bool fixedDim, int dim) const;
    QString buildSource(const QString& sn, const QString& fn,
                        const QString& cn, bool fixedDim, int dim,
                        double lo, double hi, bool hasOpt, double optVal) const;
    bool writeFiles(const QString& header, const QString& source,
                    const QString& className);

    QString         projectRoot_;
    QString         generatedH_;
    QString         generatedCpp_;
    QLineEdit*      shortNameEdit_  = nullptr;
    QLineEdit*      fullNameEdit_   = nullptr;
    QLineEdit*      classNameEdit_  = nullptr;
    QCheckBox*      fixedDimChk_    = nullptr;
    QSpinBox*       dimSpin_        = nullptr;
    QDoubleSpinBox* loBoundSpin_    = nullptr;
    QDoubleSpinBox* hiBoundSpin_    = nullptr;
    QCheckBox*      hasOptimumChk_  = nullptr;
    QDoubleSpinBox* optValSpin_     = nullptr;
    QTextEdit*      previewEdit_    = nullptr;
    QLabel*         statusLbl_      = nullptr;
    QPushButton*    previewBtn_     = nullptr;
    QPushButton*    generateBtn_    = nullptr;
};

} // namespace optimsolution
