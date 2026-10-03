#include "batch_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QListWidget>
#include <QListWidgetItem>
#include <QProgressBar>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QCloseEvent>
#include <QScreen>
#include <QGuiApplication>

namespace batch {

// ============================================================
// 构造
// ============================================================
BatchDialog::BatchDialog(const QStringList &inputFiles,
                         const QString &sourceDir,
                         QWidget *parent)
    : QDialog(parent)
    , m_inputFiles(inputFiles)
    , m_sourceDir(sourceDir)
{
    setWindowTitle(QString::fromUtf8("批量处理"));
    resize(820, 680);

    m_steps = defaultSteps();

    m_processor = new BatchProcessor(this);

    connect(m_processor, &BatchProcessor::progressChanged,
            this, &BatchDialog::onProgress);
    connect(m_processor, &BatchProcessor::fileStarted,
            this, &BatchDialog::onFileStarted);
    connect(m_processor, &BatchProcessor::fileFinished,
            this, &BatchDialog::onFileFinished);
    connect(m_processor, &BatchProcessor::previewImageReady,
            this, &BatchDialog::previewImageReady);

    connect(m_processor, &BatchProcessor::finished,
            this, &BatchDialog::onFinished);

    setupUi();
    rebuildStepList();
    updateButtonsState();
}

BatchDialog::~BatchDialog() = default;

// ============================================================
// UI
// ============================================================
void BatchDialog::setupUi()
{
    // ---------- 输入信息 ----------
    m_inputLabel = new QLabel(
        QString::fromUtf8("输入：%1 张图片（来自 %2）")
            .arg(m_inputFiles.size())
            .arg(m_sourceDir),
        this);

    // ---------- 输出目录 ----------
    QLabel *outLabel = new QLabel(QString::fromUtf8("输出目录："), this);
    m_outputEdit = new QLineEdit(this);
    m_outputEdit->setReadOnly(true);
    m_outputEdit->setText(QDir(m_sourceDir).filePath(QStringLiteral("output")));
    m_outputBtn = new QPushButton(QString::fromUtf8("选择..."), this);
    connect(m_outputBtn, &QPushButton::clicked,
            this, &BatchDialog::onSelectOutputDir);

    QHBoxLayout *outLayout = new QHBoxLayout();
    outLayout->addWidget(outLabel);
    outLayout->addWidget(m_outputEdit, 1);
    outLayout->addWidget(m_outputBtn);

    // ---------- 递归 ----------
    m_recurseCheck = new QCheckBox(QString::fromUtf8("递归子文件夹"), this);
    m_recurseCheck->setChecked(true);
    connect(m_recurseCheck, &QCheckBox::toggled,
            this, &BatchDialog::onToggleRecurse);

    // ---------- 步骤列表 ----------
    m_stepGroup = new QGroupBox(QString::fromUtf8("处理步骤（勾选并排序）"), this);
    m_stepList = new QListWidget(m_stepGroup);
    m_stepList->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(m_stepList, &QListWidget::itemChanged,
            this, &BatchDialog::onStepItemChanged);

    m_upBtn = new QPushButton(QString::fromUtf8("上移"), m_stepGroup);
    m_downBtn = new QPushButton(QString::fromUtf8("下移"), m_stepGroup);
    connect(m_upBtn, &QPushButton::clicked, this, &BatchDialog::onMoveStepUp);
    connect(m_downBtn, &QPushButton::clicked, this, &BatchDialog::onMoveStepDown);

    QVBoxLayout *stepBtnLayout = new QVBoxLayout();
    stepBtnLayout->addWidget(m_upBtn);
    stepBtnLayout->addWidget(m_downBtn);
    stepBtnLayout->addStretch();

    QHBoxLayout *stepLayout = new QHBoxLayout(m_stepGroup);
    stepLayout->addWidget(m_stepList, 1);
    stepLayout->addLayout(stepBtnLayout);

    // ---------- 进度 ----------
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);

    m_statusLabel = new QLabel(QString::fromUtf8("就绪"), this);
    m_remainLabel = new QLabel(QString(), this);

    // ---------- 日志 ----------
    m_logEdit = new QPlainTextEdit(this);
    m_logEdit->setReadOnly(true);
    m_logEdit->setMaximumBlockCount(5000);

    // ---------- 按钮 ----------
    m_startBtn = new QPushButton(QString::fromUtf8("开始"), this);
    m_pauseBtn = new QPushButton(QString::fromUtf8("暂停"), this);
    m_resumeBtn = new QPushButton(QString::fromUtf8("继续"), this);
    m_cancelBtn = new QPushButton(QString::fromUtf8("取消"), this);
    m_closeBtn = new QPushButton(QString::fromUtf8("关闭"), this);

    connect(m_startBtn, &QPushButton::clicked, this, &BatchDialog::onStart);
    connect(m_pauseBtn, &QPushButton::clicked, this, &BatchDialog::onPause);
    connect(m_resumeBtn, &QPushButton::clicked, this, &BatchDialog::onResume);
    connect(m_cancelBtn, &QPushButton::clicked, this, &BatchDialog::onCancel);
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::close);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->addWidget(m_startBtn);
    btnLayout->addWidget(m_pauseBtn);
    btnLayout->addWidget(m_resumeBtn);
    btnLayout->addWidget(m_cancelBtn);
    btnLayout->addStretch();
    btnLayout->addWidget(m_closeBtn);

    // ---------- 主布局 ----------
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(m_inputLabel);
    mainLayout->addLayout(outLayout);
    mainLayout->addWidget(m_recurseCheck);
    mainLayout->addWidget(m_stepGroup, 1);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_remainLabel);
    mainLayout->addWidget(new QLabel(QString::fromUtf8("日志："), this));
    mainLayout->addWidget(m_logEdit, 1);
    mainLayout->addLayout(btnLayout);
}

// ============================================================
// 重建步骤列表
// ============================================================
void BatchDialog::rebuildStepList()
{
    // 先断开信号，避免重填时触发 itemChanged
    disconnect(m_stepList, &QListWidget::itemChanged,
               this, &BatchDialog::onStepItemChanged);

    m_stepList->clear();
    for (const StepItem &s : m_steps) {
        QListWidgetItem *item = new QListWidgetItem(
            QString::fromUtf8("%1  —  %2")
                .arg(stepName(s.type))
                .arg(stepDescription(s.type)),
            m_stepList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(s.enabled ? Qt::Checked : Qt::Unchecked);
        item->setData(Qt::UserRole, static_cast<int>(s.type));
    }

    connect(m_stepList, &QListWidget::itemChanged,
            this, &BatchDialog::onStepItemChanged);
}

// ============================================================
// 槽：选择输出目录
// ============================================================
void BatchDialog::onSelectOutputDir()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择输出目录"),
        m_outputEdit->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (!dir.isEmpty()) {
        m_outputEdit->setText(dir);
    }
}

// ============================================================
// 槽：递归选项
// ============================================================
void BatchDialog::onToggleRecurse()
{
    // 暂时只做提示，不实际改变输入列表
    // 如果用户需要重新扫描，可以再点"添加文件夹"
}

// ============================================================
// 槽：勾选/取消步骤
// ============================================================
void BatchDialog::onStepItemChanged(QListWidgetItem *item)
{
    if (!item) return;
    const int idx = m_stepList->row(item);
    if (idx < 0 || idx >= m_steps.size()) return;

    m_steps[idx].enabled = (item->checkState() == Qt::Checked);
}

// ============================================================
// 槽：上移
// ============================================================
void BatchDialog::onMoveStepUp()
{
    const int row = m_stepList->currentRow();
    if (row <= 0) return;

    std::swap(m_steps[row], m_steps[row - 1]);
    rebuildStepList();
    m_stepList->setCurrentRow(row - 1);
}

// ============================================================
// 槽：下移
// ============================================================
void BatchDialog::onMoveStepDown()
{
    const int row = m_stepList->currentRow();
    if (row < 0 || row >= m_steps.size() - 1) return;

    std::swap(m_steps[row], m_steps[row + 1]);
    rebuildStepList();
    m_stepList->setCurrentRow(row + 1);
}

// ============================================================
// 槽：开始
// ============================================================
void BatchDialog::onStart()
{
    if (m_inputFiles.isEmpty()) {
        QMessageBox::warning(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("没有输入文件。"));
        return;
    }

    if (m_outputEdit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("请先选择输出目录。"));
        return;
    }

    // 至少勾选一个步骤
    bool anyEnabled = false;
    for (const StepItem &s : m_steps) {
        if (s.enabled) { anyEnabled = true; break; }
    }
    if (!anyEnabled) {
        QMessageBox::warning(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("请至少勾选一个处理步骤。"));
        return;
    }

    BatchOptions opt = buildOptions();
    m_logEdit->clear();
    appendLog(QString::fromUtf8("开始批处理，共 %1 张").arg(m_inputFiles.size()));

    m_processor->start(opt);
    updateButtonsState();
}

// ============================================================
// 槽：暂停 / 继续 / 取消
// ============================================================
void BatchDialog::onPause()
{
    m_processor->pause();
    appendLog(QString::fromUtf8("已暂停"));
    updateButtonsState();
}

void BatchDialog::onResume()
{
    m_processor->resume();
    appendLog(QString::fromUtf8("已继续"));
    updateButtonsState();
}

void BatchDialog::onCancel()
{
    m_processor->cancel();
    appendLog(QString::fromUtf8("正在取消..."));
    updateButtonsState();
}

// ============================================================
// 引擎信号
// ============================================================
void BatchDialog::onProgress(const BatchProgress &progress)
{
    if (progress.total <= 0) return;

    const int percent = progress.current * 100 / progress.total;
    m_progressBar->setValue(percent);

    m_statusLabel->setText(
        QString::fromUtf8("当前：%1  （成功 %2 / 失败 %3）")
            .arg(progress.currentFile)
            .arg(progress.succeeded)
            .arg(progress.failed));

    if (progress.remainingSeconds > 0.5) {
        const int sec = static_cast<int>(progress.remainingSeconds);
        const int min = sec / 60;
        const int ss = sec % 60;
        m_remainLabel->setText(
            QString::fromUtf8("已用 %1 秒，剩余约 %2 分 %3 秒")
                .arg(progress.elapsedSeconds, 0, 'f', 1)
                .arg(min).arg(ss));
    } else {
        m_remainLabel->setText(
            QString::fromUtf8("已用 %1 秒")
                .arg(progress.elapsedSeconds, 0, 'f', 1));
    }

    updateButtonsState();
}

void BatchDialog::onFileStarted(const QString &path)
{
    const QFileInfo fi(path);
    appendLog(QString::fromUtf8("处理：%1").arg(fi.fileName()));
}

void BatchDialog::onFileFinished(const QString &path, bool success)
{
    const QFileInfo fi(path);
    if (success) {
        appendLog(QString::fromUtf8("  ✓ %1").arg(fi.fileName()));
    } else {
        appendLog(QString::fromUtf8("  ✗ %1（失败）").arg(fi.fileName()));
    }
}

void BatchDialog::onFinished(const BatchResult &result)
{
    QString msg;
    if (result.cancelled) {
        msg = QString::fromUtf8("已取消。完成 %1 / %2，失败 %3，耗时 %4 秒")
                  .arg(result.succeeded)
                  .arg(result.total)
                  .arg(result.failed)
                  .arg(result.totalSeconds, 0, 'f', 1);
    } else {
        msg = QString::fromUtf8("全部完成！成功 %1 / %2，失败 %3，耗时 %4 秒")
                  .arg(result.succeeded)
                  .arg(result.total)
                  .arg(result.failed)
                  .arg(result.totalSeconds, 0, 'f', 1);
    }

    appendLog(msg);
    m_statusLabel->setText(msg);

    if (result.failed > 0) {
        appendLog(QString::fromUtf8("失败文件："));
        for (const QString &f : result.failedFiles) {
            appendLog(QString::fromUtf8("  %1").arg(f));
        }
    }

    updateButtonsState();
}

// ============================================================
// 构建 BatchOptions
// ============================================================
BatchOptions BatchDialog::buildOptions() const
{
    BatchOptions opt;

    opt.inputFiles = m_inputFiles;
    opt.outputDir = m_outputEdit->text().trimmed();

    // 步骤排序：按 order 排（其实已经是列表顺序）
    opt.steps = m_steps;

    {
        process::BlackEdgeOptions b;
        b.paperSampleRatio   = 0.6;
        b.paperPercentile    = 0.9;
        b.darkRatio          = 0.75;
        b.maxScanRatio       = 0.30;
        b.darkPixelRatio     = 0.50;
        b.gapTolerance       = 3;
        b.smoothKernelSize   = 5;
        b.expandPixels       = 5;
        b.fillWhite          = true;
        opt.blackEdgeOpt = b;
    }

    {
        process::DenoiseOptions d;
        d.maxSpotArea   = 200;
        d.maxSpotWidth  = 30;
        d.maxSpotHeight = 30;
        d.darkRatio     = 0.60;
        d.protectRadius = 2;
        d.strengthLevel = 1;
        d.useInpaint    = true;
        opt.denoiseOpt = d;
    }

    {
        process::EnhanceOptions e;
        e.gamma = 1.6;
        e.protectColor = true;
        e.colorSaturationThreshold = 40;
        e.targetDarkGray = 0;
        e.targetPaperGray = 255;
        e.minDarkGrayToEnhance = 110;
        opt.enhanceOpt = e;
    }

    {
        process::ErrPageOptions ep;
        ep.detectTopLeft          = true;
        ep.topLeftWidthRatio      = 0.12;
        ep.topLeftHeightRatio     = 0.08;
        ep.detectTopRight         = true;
        ep.topRightWidthRatio     = 0.25;
        ep.topRightHeightRatio    = 0.15;
        ep.detectBottomRight      = true;
        ep.bottomRightWidthRatio  = 0.25;
        ep.bottomRightHeightRatio = 0.15;
        ep.detectBottomLeft       = false;
        ep.minDigitHeight         = 20;
        ep.maxDigitHeight         = 100;
        ep.minDigitWidth          = 10;
        ep.maxDigitWidth          = 150;
        ep.crossLineRatio         = 0.5;
        ep.fillWhite              = true;
        ep.tesseractPath          = QString();
        opt.errPageOpt = ep;
    }

    {
        process::BackgroundOptions bg;
        bg.paperSampleRatio = 0.6;
        bg.paperPercentile  = 0.9;
        bg.contentRatio     = 0.70;
        bg.targetPaperGray  = 255;
        bg.protectColor     = true;
        bg.colorSatMin      = 40;
        opt.backgroundOpt = bg;
    }

    {
        process::ColorLineOptions cl;
        cl.channelDiffThreshold = 8;
        cl.valueThreshold       = 180;
        cl.colorRatioThreshold  = 0.40;
        cl.maxThickness         = 8;
        cl.edgeMarginRatio      = 0.02;
        opt.colorLineOpt = cl;
    }

    opt.stampOpt = protect::StampProtectOptions();

    // 输出
    opt.outputSuffix = QString();   // 保持原名
    opt.jpegOpt.quality = 98;
    opt.jpegOpt.dpiX = 300.0;
    opt.jpegOpt.dpiY = 300.0;
    opt.jpegOpt.use444Sampling = true;

    return opt;
}

// ============================================================
// 按钮状态
// ============================================================
void BatchDialog::updateButtonsState()
{
    const bool running = m_processor->isRunning();
    const bool paused = m_processor->isPaused();

    m_startBtn->setEnabled(!running);
    m_pauseBtn->setEnabled(running && !paused);
    m_resumeBtn->setEnabled(running && paused);
    m_cancelBtn->setEnabled(running);

    m_outputBtn->setEnabled(!running);
    m_recurseCheck->setEnabled(!running);
    m_stepList->setEnabled(!running);
    m_upBtn->setEnabled(!running);
    m_downBtn->setEnabled(!running);
}

// ============================================================
// 日志
// ============================================================
void BatchDialog::appendLog(const QString &line)
{
    const QString ts = QDateTime::currentDateTime().toString(
        QStringLiteral("HH:mm:ss"));
    m_logEdit->appendPlainText(
        QString::fromUtf8("[%1] %2").arg(ts).arg(line));
}
// ============================================================
// 移到屏幕右上角
// ============================================================
void BatchDialog::moveToTopRight()
{
    QScreen *screen = QGuiApplication::primaryScreen();
    if (!screen) return;

    const QRect avail = screen->availableGeometry();
    const int margin = 20;

    const int w = width();
    const int h = height();

    const int x = avail.right() - w - margin;
    const int y = avail.top() + margin;

    move(x, y);
}

} // namespace batch