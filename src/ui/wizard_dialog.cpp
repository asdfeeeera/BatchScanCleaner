#include "wizard_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QSettings>
#include <QCoreApplication>
#include <QDir>

namespace ui {

// ============================================================
// 工具：settings.ini 路径
// ============================================================
static QString settingsPath()
{
    return QCoreApplication::applicationDirPath()
           + QStringLiteral("/settings.ini");
}

// ============================================================
// 构造
// ============================================================
WizardDialog::WizardDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QString::fromUtf8("欢迎使用 — 首次启动向导"));
    resize(620, 560);
    setupUi();
    loadCurrentSettings();
}

WizardDialog::~WizardDialog() = default;

// ============================================================
// UI
// ============================================================
void WizardDialog::setupUi()
{
    // ---- 欢迎文字 ----
    QLabel *title = new QLabel(
        QString::fromUtf8("欢迎使用 批量扫描图片净化增强软件"), this);
    QFont tf = title->font();
    tf.setPointSize(tf.pointSize() + 3);
    tf.setBold(true);
    title->setFont(tf);

    QLabel *intro = new QLabel(
        QString::fromUtf8("下面几步可以帮您快速配置常用目录和参数，"
                          "配置会保存到程序目录下的 settings.ini。\n"
                          "以后随时可以在【工具 → 设置】里修改。\n\n"
                          "如果暂时不确定，直接点“跳过”也可以。"),
        this);
    intro->setWordWrap(true);

    // ---- 输出目录 ----
    QGroupBox *dirGroup = new QGroupBox(
        QString::fromUtf8("① 常用目录"), this);

    m_outputDirEdit = new QLineEdit(dirGroup);
    m_outputDirEdit->setPlaceholderText(
        QString::fromUtf8("留空则使用「图片所在目录/output」"));
    m_outputDirBtn = new QPushButton(QString::fromUtf8("选择..."), dirGroup);
    connect(m_outputDirBtn, &QPushButton::clicked,
            this, &WizardDialog::onBrowseOutputDir);

    QHBoxLayout *outputLayout = new QHBoxLayout();
    outputLayout->addWidget(m_outputDirEdit, 1);
    outputLayout->addWidget(m_outputDirBtn);

    m_backupCheck = new QCheckBox(
        QString::fromUtf8("启用自动备份（处理前把原图复制到备份目录）"),
        dirGroup);
    m_backupCheck->setChecked(false);

    m_backupDirEdit = new QLineEdit(dirGroup);
    m_backupDirEdit->setPlaceholderText(
        QString::fromUtf8("例如 D:/backup 或 //nas/share/backup"));
    m_backupDirBtn = new QPushButton(QString::fromUtf8("选择..."), dirGroup);
    connect(m_backupDirBtn, &QPushButton::clicked,
            this, &WizardDialog::onBrowseBackupDir);

    QHBoxLayout *backupLayout = new QHBoxLayout();
    backupLayout->addWidget(m_backupDirEdit, 1);
    backupLayout->addWidget(m_backupDirBtn);

    QFormLayout *dirForm = new QFormLayout(dirGroup);
    dirForm->addRow(QString::fromUtf8("默认输出目录："), outputLayout);
    dirForm->addRow(QString(), m_backupCheck);
    dirForm->addRow(QString::fromUtf8("备份目录："), backupLayout);

    // ---- Tesseract ----
    QGroupBox *tessGroup = new QGroupBox(
        QString::fromUtf8("② Tesseract（错误页码处理，可留空）"), this);

    m_tesseractEdit = new QLineEdit(tessGroup);
    m_tesseractEdit->setPlaceholderText(
        QString::fromUtf8("tesseract.exe 完整路径，或留空自动查找"));
    m_tesseractBtn = new QPushButton(QString::fromUtf8("浏览..."), tessGroup);
    connect(m_tesseractBtn, &QPushButton::clicked,
            this, &WizardDialog::onBrowseTesseract);

    QHBoxLayout *tessLayout = new QHBoxLayout();
    tessLayout->addWidget(m_tesseractEdit, 1);
    tessLayout->addWidget(m_tesseractBtn);

    QFormLayout *tessForm = new QFormLayout(tessGroup);
    tessForm->addRow(QString::fromUtf8("tesseract 路径："), tessLayout);

    // ---- 分片 ----
    QGroupBox *shardGroup = new QGroupBox(
        QString::fromUtf8("③ 多机分片（单机跑保持总机器数 = 1）"), this);

    m_shardIndexSpin = new QSpinBox(shardGroup);
    m_shardIndexSpin->setRange(0, 63);
    m_shardIndexSpin->setValue(0);

    m_totalShardsSpin = new QSpinBox(shardGroup);
    m_totalShardsSpin->setRange(1, 64);
    m_totalShardsSpin->setValue(1);

    QHBoxLayout *shardLayout = new QHBoxLayout();
    shardLayout->addWidget(new QLabel(QString::fromUtf8("本机编号："), shardGroup));
    shardLayout->addWidget(m_shardIndexSpin);
    shardLayout->addSpacing(20);
    shardLayout->addWidget(new QLabel(QString::fromUtf8("总机器数："), shardGroup));
    shardLayout->addWidget(m_totalShardsSpin);
    shardLayout->addStretch();

    QFormLayout *shardForm = new QFormLayout(shardGroup);
    shardForm->addRow(QString(), shardLayout);

    // ---- 底部按钮 ----
    m_skipBtn   = new QPushButton(QString::fromUtf8("跳过"), this);
    m_finishBtn = new QPushButton(QString::fromUtf8("完成并保存"), this);

    connect(m_skipBtn, &QPushButton::clicked,
            this, &WizardDialog::onSkip);
    connect(m_finishBtn, &QPushButton::clicked,
            this, &WizardDialog::onFinish);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    btnLayout->addWidget(m_skipBtn);
    btnLayout->addWidget(m_finishBtn);

    // ---- 主布局 ----
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(title);
    mainLayout->addWidget(intro);
    mainLayout->addSpacing(6);
    mainLayout->addWidget(dirGroup);
    mainLayout->addWidget(tessGroup);
    mainLayout->addWidget(shardGroup);
    mainLayout->addStretch();
    mainLayout->addLayout(btnLayout);
}

// ============================================================
// 载入现有设置
// ============================================================
void WizardDialog::loadCurrentSettings()
{
    QSettings ini(settingsPath(), QSettings::IniFormat);
    ini.setIniCodec("UTF-8");

    ini.beginGroup(QStringLiteral("General"));
    m_outputDirEdit->setText(
        ini.value(QStringLiteral("defaultOutputDir")).toString());
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Tesseract"));
    m_tesseractEdit->setText(
        ini.value(QStringLiteral("path")).toString());
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Batch"));
    m_shardIndexSpin->setValue(
        qBound(0, ini.value(QStringLiteral("shardIndex"), 0).toInt(), 63));
    m_totalShardsSpin->setValue(
        qBound(1, ini.value(QStringLiteral("totalShards"), 1).toInt(), 64));
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Backup"));
    m_backupCheck->setChecked(
        ini.value(QStringLiteral("backupEnabled"), false).toBool());
    m_backupDirEdit->setText(
        ini.value(QStringLiteral("backupDir")).toString());
    ini.endGroup();
}

// ============================================================
// 保存设置（保留未在向导中展示的字段，只写这几项）
// ============================================================
void WizardDialog::saveCurrentSettings()
{
    QSettings ini(settingsPath(), QSettings::IniFormat);
    ini.setIniCodec("UTF-8");

    ini.beginGroup(QStringLiteral("General"));
    ini.setValue(QStringLiteral("defaultOutputDir"),
                 m_outputDirEdit->text().trimmed());
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Tesseract"));
    ini.setValue(QStringLiteral("path"),
                 m_tesseractEdit->text().trimmed());
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Batch"));
    ini.setValue(QStringLiteral("shardIndex"),  m_shardIndexSpin->value());
    ini.setValue(QStringLiteral("totalShards"), m_totalShardsSpin->value());
    ini.endGroup();

    ini.beginGroup(QStringLiteral("Backup"));
    ini.setValue(QStringLiteral("backupEnabled"), m_backupCheck->isChecked());
    ini.setValue(QStringLiteral("backupDir"),
                 m_backupDirEdit->text().trimmed());
    ini.endGroup();
}

// ============================================================
// 首次启动标记
// ============================================================
void WizardDialog::markFirstRunDone()
{
    QSettings ini(settingsPath(), QSettings::IniFormat);
    ini.setIniCodec("UTF-8");
    ini.beginGroup(QStringLiteral("FirstRun"));
    ini.setValue(QStringLiteral("completed"), 1);
    ini.endGroup();
}

// ============================================================
// 槽：选择目录
// ============================================================
void WizardDialog::onBrowseOutputDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择默认输出目录"),
        m_outputDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) m_outputDirEdit->setText(dir);
}

void WizardDialog::onBrowseBackupDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择备份目录"),
        m_backupDirEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!dir.isEmpty()) m_backupDirEdit->setText(dir);
}

void WizardDialog::onBrowseTesseract()
{
    const QString file = QFileDialog::getOpenFileName(
        this,
        QString::fromUtf8("选择 tesseract.exe"),
        m_tesseractEdit->text(),
        QString::fromUtf8("可执行文件 (*.exe);;所有文件 (*)"));
    if (!file.isEmpty()) m_tesseractEdit->setText(file);
}

// ============================================================
// 槽：完成 / 跳过
// ============================================================
void WizardDialog::onFinish()
{
    saveCurrentSettings();
    markFirstRunDone();
    accept();
}

void WizardDialog::onSkip()
{
    markFirstRunDone();
    reject();
}

} // namespace ui