#include "settings_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QCoreApplication>
#include <QDir>

namespace ui {

// ============================================================
// AppSettings
// ============================================================
QString AppSettings::filePath()
{
    return QCoreApplication::applicationDirPath()
           + QStringLiteral("/settings.ini");
}

AppSettings AppSettings::load()
{
    AppSettings s;
    QSettings ini(filePath(), QSettings::IniFormat);
    ini.setIniCodec("UTF-8");

    ini.beginGroup(QStringLiteral("General"));
    s.defaultOutputDir    = ini.value(QStringLiteral("defaultOutputDir")).toString();
    s.defaultOutputSuffix = ini.value(QStringLiteral("defaultOutputSuffix")).toString();
    s.defaultJpegQuality  = ini.value(QStringLiteral("defaultJpegQuality"), 98).toInt();
    s.defaultDpi          = ini.value(QStringLiteral("defaultDpi"), 300.0).toDouble();
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Tesseract"));
    s.tesseractPath         = ini.value(QStringLiteral("path")).toString();
    s.errPageTopLeft        = ini.value(QStringLiteral("topLeft"), true).toBool();
    s.errPageTopRight       = ini.value(QStringLiteral("topRight"), true).toBool();
    s.errPageBottomRight    = ini.value(QStringLiteral("bottomRight"), true).toBool();
    s.errPageMinDigitHeight = ini.value(QStringLiteral("minDigitHeight"), 20).toInt();
    s.errPageMaxDigitHeight = ini.value(QStringLiteral("maxDigitHeight"), 100).toInt();
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Batch"));
    s.defaultExtraDirs   = ini.value(QStringLiteral("extraDirs")).toStringList();
    s.defaultShardIndex  = ini.value(QStringLiteral("shardIndex"), 0).toInt();
    s.defaultTotalShards = ini.value(QStringLiteral("totalShards"), 1).toInt();
    s.reportDir          = ini.value(QStringLiteral("reportDir")).toString();
    ini.endGroup();

    // ★ 备份（与 batch_dialog.cpp 里读的键名保持一致）
    ini.beginGroup(QStringLiteral("Backup"));
    s.backupDir     = ini.value(QStringLiteral("backupDir")).toString();
    s.backupEnabled = ini.value(QStringLiteral("backupEnabled"), false).toBool();
    ini.endGroup();

    return s;
}

void AppSettings::save() const
{
    QSettings ini(filePath(), QSettings::IniFormat);
    ini.setIniCodec("UTF-8");

    ini.beginGroup(QStringLiteral("General"));
    ini.setValue(QStringLiteral("defaultOutputDir"),    defaultOutputDir);
    ini.setValue(QStringLiteral("defaultOutputSuffix"), defaultOutputSuffix);
    ini.setValue(QStringLiteral("defaultJpegQuality"),  defaultJpegQuality);
    ini.setValue(QStringLiteral("defaultDpi"),          defaultDpi);
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Tesseract"));
    ini.setValue(QStringLiteral("path"),           tesseractPath);
    ini.setValue(QStringLiteral("topLeft"),        errPageTopLeft);
    ini.setValue(QStringLiteral("topRight"),       errPageTopRight);
    ini.setValue(QStringLiteral("bottomRight"),    errPageBottomRight);
    ini.setValue(QStringLiteral("minDigitHeight"), errPageMinDigitHeight);
    ini.setValue(QStringLiteral("maxDigitHeight"), errPageMaxDigitHeight);
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Batch"));
    ini.setValue(QStringLiteral("extraDirs"),   defaultExtraDirs);
    ini.setValue(QStringLiteral("shardIndex"),  defaultShardIndex);
    ini.setValue(QStringLiteral("totalShards"), defaultTotalShards);
    ini.setValue(QStringLiteral("reportDir"),   reportDir);
    ini.endGroup();

    // ★ 备份
    ini.beginGroup(QStringLiteral("Backup"));
    ini.setValue(QStringLiteral("backupDir"),     backupDir);
    ini.setValue(QStringLiteral("backupEnabled"), backupEnabled);
    ini.endGroup();
}

// ============================================================
// SettingsDialog
// ============================================================
SettingsDialog::SettingsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QString::fromUtf8("设置"));
    resize(660, 860);
    setupUi();
    loadToUi();
}

SettingsDialog::~SettingsDialog() = default;

void SettingsDialog::setupUi()
{
    // ---------- ① 通用 ----------
    QGroupBox *generalGroup = new QGroupBox(QString::fromUtf8("通用"), this);

    m_outputDirEdit = new QLineEdit(generalGroup);
    m_outputDirEdit->setPlaceholderText(
        QString::fromUtf8("留空则使用「图片所在目录/output」"));
    m_outputDirBtn = new QPushButton(QString::fromUtf8("选择..."), generalGroup);
    connect(m_outputDirBtn, &QPushButton::clicked,
            this, &SettingsDialog::onBrowseOutputDir);

    QHBoxLayout *outputDirLayout = new QHBoxLayout();
    outputDirLayout->addWidget(m_outputDirEdit, 1);
    outputDirLayout->addWidget(m_outputDirBtn);

    m_suffixEdit = new QLineEdit(generalGroup);
    m_suffixEdit->setPlaceholderText(
        QString::fromUtf8("例如 _clean（留空表示保持原名）"));

    m_jpegQualitySpin = new QSpinBox(generalGroup);
    m_jpegQualitySpin->setRange(1, 100);
    m_jpegQualitySpin->setValue(98);

    m_dpiSpin = new QDoubleSpinBox(generalGroup);
    m_dpiSpin->setRange(50.0, 1200.0);
    m_dpiSpin->setDecimals(0);
    m_dpiSpin->setSingleStep(50.0);
    m_dpiSpin->setValue(300.0);
    m_dpiSpin->setSuffix(QString::fromUtf8(" DPI"));

    QFormLayout *generalForm = new QFormLayout(generalGroup);
    generalForm->addRow(QString::fromUtf8("默认输出目录："), outputDirLayout);
    generalForm->addRow(QString::fromUtf8("默认输出后缀："), m_suffixEdit);
    generalForm->addRow(QString::fromUtf8("默认 JPEG 质量："), m_jpegQualitySpin);
    generalForm->addRow(QString::fromUtf8("默认 DPI："), m_dpiSpin);

    // ---------- ② Tesseract ----------
    QGroupBox *tessGroup = new QGroupBox(
        QString::fromUtf8("Tesseract（错误页码处理）"), this);

    m_tesseractEdit = new QLineEdit(tessGroup);
    m_tesseractEdit->setPlaceholderText(
        QString::fromUtf8("tesseract.exe 完整路径，或留空自动查找"));
    m_tesseractBtn = new QPushButton(QString::fromUtf8("浏览..."), tessGroup);
    connect(m_tesseractBtn, &QPushButton::clicked,
            this, &SettingsDialog::onBrowseTesseract);

    QHBoxLayout *tessLayout = new QHBoxLayout();
    tessLayout->addWidget(m_tesseractEdit, 1);
    tessLayout->addWidget(m_tesseractBtn);

    m_topLeftCheck     = new QCheckBox(QString::fromUtf8("左上"), tessGroup);
    m_topRightCheck    = new QCheckBox(QString::fromUtf8("右上"), tessGroup);
    m_bottomRightCheck = new QCheckBox(QString::fromUtf8("右下"), tessGroup);

    QHBoxLayout *cornerLayout = new QHBoxLayout();
    cornerLayout->addWidget(m_topLeftCheck);
    cornerLayout->addWidget(m_topRightCheck);
    cornerLayout->addWidget(m_bottomRightCheck);
    cornerLayout->addStretch();

    m_minDigitHeightSpin = new QSpinBox(tessGroup);
    m_minDigitHeightSpin->setRange(1, 9999);
    m_minDigitHeightSpin->setValue(20);

    m_maxDigitHeightSpin = new QSpinBox(tessGroup);
    m_maxDigitHeightSpin->setRange(1, 9999);
    m_maxDigitHeightSpin->setValue(100);

    QHBoxLayout *digitLayout = new QHBoxLayout();
    digitLayout->addWidget(new QLabel(QString::fromUtf8("最小："), tessGroup));
    digitLayout->addWidget(m_minDigitHeightSpin);
    digitLayout->addSpacing(20);
    digitLayout->addWidget(new QLabel(QString::fromUtf8("最大："), tessGroup));
    digitLayout->addWidget(m_maxDigitHeightSpin);
    digitLayout->addStretch();

    QFormLayout *tessForm = new QFormLayout(tessGroup);
    tessForm->addRow(QString::fromUtf8("tesseract 路径："), tessLayout);
    tessForm->addRow(QString::fromUtf8("检测区域："), cornerLayout);
    tessForm->addRow(QString::fromUtf8("数字高度（像素）："), digitLayout);

    // ---------- ③ 批处理 ----------
    QGroupBox *batchGroup = new QGroupBox(QString::fromUtf8("批处理"), this);

    m_extraDirsList = new QListWidget(batchGroup);
    m_extraDirsList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_extraDirsList->setMinimumHeight(80);

    m_addExtraDirBtn = new QPushButton(QString::fromUtf8("添加..."), batchGroup);
    m_removeExtraDirBtn = new QPushButton(QString::fromUtf8("删除"), batchGroup);
    connect(m_addExtraDirBtn, &QPushButton::clicked,
            this, &SettingsDialog::onAddExtraDir);
    connect(m_removeExtraDirBtn, &QPushButton::clicked,
            this, &SettingsDialog::onRemoveExtraDir);

    QVBoxLayout *extraBtnLayout = new QVBoxLayout();
    extraBtnLayout->addWidget(m_addExtraDirBtn);
    extraBtnLayout->addWidget(m_removeExtraDirBtn);
    extraBtnLayout->addStretch();

    QHBoxLayout *extraListLayout = new QHBoxLayout();
    extraListLayout->addWidget(m_extraDirsList, 1);
    extraListLayout->addLayout(extraBtnLayout);

    m_shardIndexSpin = new QSpinBox(batchGroup);
    m_shardIndexSpin->setRange(0, 63);
    m_shardIndexSpin->setValue(0);

    m_totalShardsSpin = new QSpinBox(batchGroup);
    m_totalShardsSpin->setRange(1, 64);
    m_totalShardsSpin->setValue(1);

    QHBoxLayout *shardLayout = new QHBoxLayout();
    shardLayout->addWidget(new QLabel(QString::fromUtf8("本机编号："), batchGroup));
    shardLayout->addWidget(m_shardIndexSpin);
    shardLayout->addSpacing(20);
    shardLayout->addWidget(new QLabel(QString::fromUtf8("总机器数："), batchGroup));
    shardLayout->addWidget(m_totalShardsSpin);
    shardLayout->addStretch();

    m_reportDirEdit = new QLineEdit(batchGroup);
    m_reportDirEdit->setPlaceholderText(
        QString::fromUtf8("留空则跟输出目录相同"));
    m_reportDirBtn = new QPushButton(QString::fromUtf8("选择..."), batchGroup);
    connect(m_reportDirBtn, &QPushButton::clicked,
            this, &SettingsDialog::onBrowseReportDir);

    QHBoxLayout *reportDirLayout = new QHBoxLayout();
    reportDirLayout->addWidget(m_reportDirEdit, 1);
    reportDirLayout->addWidget(m_reportDirBtn);

    QFormLayout *batchForm = new QFormLayout(batchGroup);
    batchForm->addRow(QString::fromUtf8("默认副本目录："), extraListLayout);
    batchForm->addRow(QString::fromUtf8("默认分片："), shardLayout);
    batchForm->addRow(QString::fromUtf8("报告目录："), reportDirLayout);

    // ---------- ★ ④ 备份 ----------
    QGroupBox *backupGroup = new QGroupBox(
        QString::fromUtf8("备份（处理前把原图复制到备份目录）"), this);

    m_backupEnabledCheck = new QCheckBox(
        QString::fromUtf8("启用自动备份（批处理默认勾选）"), backupGroup);
    m_backupEnabledCheck->setChecked(false);

    QLabel *backupHint = new QLabel(
        QString::fromUtf8("备份会放到：<备份目录>/YYYY-MM-DD/原文件名。\n"
                          "支持移动硬盘、网络共享盘（如 //nas/share/backup）。"),
        backupGroup);
    backupHint->setWordWrap(true);

    m_backupDirEdit = new QLineEdit(backupGroup);
    m_backupDirEdit->setPlaceholderText(
        QString::fromUtf8("例如 D:/backup 或 //nas/share/backup"));

    m_backupDirBtn = new QPushButton(QString::fromUtf8("选择..."), backupGroup);
    connect(m_backupDirBtn, &QPushButton::clicked,
            this, &SettingsDialog::onBrowseBackupDir);

    QHBoxLayout *backupDirLayout = new QHBoxLayout();
    backupDirLayout->addWidget(new QLabel(QString::fromUtf8("备份目录："), backupGroup));
    backupDirLayout->addWidget(m_backupDirEdit, 1);
    backupDirLayout->addWidget(m_backupDirBtn);

    QVBoxLayout *backupLayout = new QVBoxLayout(backupGroup);
    backupLayout->addWidget(m_backupEnabledCheck);
    backupLayout->addWidget(backupHint);
    backupLayout->addLayout(backupDirLayout);

    // ---------- 底部按钮 ----------
    m_resetBtn  = new QPushButton(QString::fromUtf8("恢复默认"), this);
    m_okBtn     = new QPushButton(QString::fromUtf8("确定"), this);
    m_cancelBtn = new QPushButton(QString::fromUtf8("取消"), this);
    m_applyBtn  = new QPushButton(QString::fromUtf8("应用"), this);

    connect(m_resetBtn, &QPushButton::clicked,
            this, &SettingsDialog::onResetDefaults);
    connect(m_okBtn, &QPushButton::clicked,
            this, &SettingsDialog::onOk);
    connect(m_cancelBtn, &QPushButton::clicked,
            this, &QDialog::reject);
    connect(m_applyBtn, &QPushButton::clicked,
            this, &SettingsDialog::onApply);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->addWidget(m_resetBtn);
    btnLayout->addStretch();
    btnLayout->addWidget(m_okBtn);
    btnLayout->addWidget(m_cancelBtn);
    btnLayout->addWidget(m_applyBtn);

    // ---------- 主布局 ----------
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(generalGroup);
    mainLayout->addWidget(tessGroup);
    mainLayout->addWidget(batchGroup, 1);
    mainLayout->addWidget(backupGroup);
    mainLayout->addLayout(btnLayout);
}

void SettingsDialog::loadToUi()
{
    const AppSettings s = AppSettings::load();

    m_outputDirEdit->setText(s.defaultOutputDir);
    m_suffixEdit->setText(s.defaultOutputSuffix);
    m_jpegQualitySpin->setValue(qBound(1, s.defaultJpegQuality, 100));

    double dpi = s.defaultDpi;
    if (dpi < 50.0)   dpi = 50.0;
    if (dpi > 1200.0) dpi = 1200.0;
    m_dpiSpin->setValue(dpi);

    m_tesseractEdit->setText(s.tesseractPath);
    m_topLeftCheck->setChecked(s.errPageTopLeft);
    m_topRightCheck->setChecked(s.errPageTopRight);
    m_bottomRightCheck->setChecked(s.errPageBottomRight);
    m_minDigitHeightSpin->setValue(qBound(1, s.errPageMinDigitHeight, 9999));
    m_maxDigitHeightSpin->setValue(qBound(1, s.errPageMaxDigitHeight, 9999));

    m_extraDirsList->clear();
    for (const QString &d : s.defaultExtraDirs) {
        if (!d.trimmed().isEmpty()) {
            m_extraDirsList->addItem(d);
        }
    }
    m_shardIndexSpin->setValue(qBound(0, s.defaultShardIndex, 63));
    m_totalShardsSpin->setValue(qBound(1, s.defaultTotalShards, 64));
    m_reportDirEdit->setText(s.reportDir);

    // ★ 备份
    m_backupEnabledCheck->setChecked(s.backupEnabled);
    m_backupDirEdit->setText(s.backupDir);
}

void SettingsDialog::uiToSettings(AppSettings &s) const
{
    s.defaultOutputDir    = m_outputDirEdit->text().trimmed();
    s.defaultOutputSuffix = m_suffixEdit->text().trimmed();
    s.defaultJpegQuality  = m_jpegQualitySpin->value();
    s.defaultDpi          = m_dpiSpin->value();

    s.tesseractPath         = m_tesseractEdit->text().trimmed();
    s.errPageTopLeft        = m_topLeftCheck->isChecked();
    s.errPageTopRight       = m_topRightCheck->isChecked();
    s.errPageBottomRight    = m_bottomRightCheck->isChecked();
    s.errPageMinDigitHeight = m_minDigitHeightSpin->value();
    s.errPageMaxDigitHeight = m_maxDigitHeightSpin->value();

    s.defaultExtraDirs.clear();
    for (int i = 0; i < m_extraDirsList->count(); ++i) {
        const QString d = m_extraDirsList->item(i)->text().trimmed();
        if (!d.isEmpty()) s.defaultExtraDirs.append(d);
    }
    s.defaultShardIndex  = m_shardIndexSpin->value();
    s.defaultTotalShards = m_totalShardsSpin->value();
    s.reportDir          = m_reportDirEdit->text().trimmed();

    // ★ 备份
    s.backupEnabled = m_backupEnabledCheck->isChecked();
    s.backupDir     = m_backupDirEdit->text().trimmed();
}

void SettingsDialog::applySettings()
{
    AppSettings s = AppSettings::load();
    uiToSettings(s);
    s.save();
}

// ---- 槽 ----

void SettingsDialog::onBrowseOutputDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择默认输出目录"),
        m_outputDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) m_outputDirEdit->setText(dir);
}

void SettingsDialog::onBrowseTesseract()
{
    const QString file = QFileDialog::getOpenFileName(
        this,
        QString::fromUtf8("选择 tesseract.exe"),
        m_tesseractEdit->text(),
        QString::fromUtf8("可执行文件 (*.exe);;所有文件 (*)"));
    if (!file.isEmpty()) m_tesseractEdit->setText(file);
}

void SettingsDialog::onBrowseReportDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择报告目录"),
        m_reportDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) m_reportDirEdit->setText(dir);
}

// ★ 备份目录选择
void SettingsDialog::onBrowseBackupDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择备份目录"),
        m_backupDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) m_backupDirEdit->setText(dir);
}

void SettingsDialog::onAddExtraDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择副本目录"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (dir.isEmpty()) return;
    for (int i = 0; i < m_extraDirsList->count(); ++i) {
        if (QDir(m_extraDirsList->item(i)->text()) == QDir(dir)) return;
    }
    m_extraDirsList->addItem(dir);
}

void SettingsDialog::onRemoveExtraDir()
{
    const int row = m_extraDirsList->currentRow();
    if (row < 0) return;
    delete m_extraDirsList->takeItem(row);
}

void SettingsDialog::onResetDefaults()
{
    const auto ret = QMessageBox::question(
        this,
        QString::fromUtf8("确认"),
        QString::fromUtf8("确定要恢复所有设置到默认值吗？"));
    if (ret != QMessageBox::Yes) return;

    m_outputDirEdit->clear();
    m_suffixEdit->clear();
    m_jpegQualitySpin->setValue(98);
    m_dpiSpin->setValue(300.0);

    m_tesseractEdit->clear();
    m_topLeftCheck->setChecked(true);
    m_topRightCheck->setChecked(true);
    m_bottomRightCheck->setChecked(true);
    m_minDigitHeightSpin->setValue(20);
    m_maxDigitHeightSpin->setValue(100);

    m_extraDirsList->clear();
    m_shardIndexSpin->setValue(0);
    m_totalShardsSpin->setValue(1);
    m_reportDirEdit->clear();

    // ★ 备份
    m_backupEnabledCheck->setChecked(false);
    m_backupDirEdit->clear();
}

void SettingsDialog::onApply()
{
    applySettings();
}

void SettingsDialog::onOk()
{
    applySettings();
    accept();
}

} // namespace ui