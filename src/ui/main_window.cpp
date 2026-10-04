#include "main_window.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QSplitter>
#include <QTreeView>
#include <QTableView>
#include <QStandardItemModel>
#include <QHeaderView>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QToolButton>
#include <QWidget>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QFileInfo>
#include <QResizeEvent>
#include <QTimer>
#include <QSlider>
#include <QSpinBox>
#include <QDir>
#include <QDebug>
#include <QMenu>
#include <QWidgetAction>
#include <QCheckBox>
#include <QPushButton>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_io.h"
#include "jpeg_writer.h"
#include "file_scanner.h"
#include "deskew.h"
#include "blackedge.h"
#include "denoise.h"
#include "enhance.h"
#include "colorline.h"
#include "errpage.h"
#include "background.h"

#include "../protect/stamp_protect.h"
#include "../analyze/pending_center.h"
#include "../analyze/pending_dialog.h"
#include "../db/batch_dialog.h"

// 3.2 设置窗口
#include "settings_dialog.h"

// ★ 3.3 备份管理窗口
#include "../backup/backup_dialog.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QString::fromUtf8("批量扫描图片净化增强软件"));
    resize(1280, 800);

    m_enhanceDebounceTimer = new QTimer(this);
    m_enhanceDebounceTimer->setSingleShot(true);
    connect(m_enhanceDebounceTimer, &QTimer::timeout,
            this, &MainWindow::onEnhanceDebounceTimeout);

    setupMenuBar();
    setupToolBar();
    setupCentralWidget();
    setupStatusBar();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupMenuBar()
{
    QMenu *fileMenu = menuBar()->addMenu(QString::fromUtf8("文件"));

    QAction *openImageAction = fileMenu->addAction(QString::fromUtf8("打开图片..."));
    connect(openImageAction, &QAction::triggered, this, &MainWindow::onOpenImage);

    QAction *addFolderAction = fileMenu->addAction(QString::fromUtf8("添加文件夹..."));
    connect(addFolderAction, &QAction::triggered, this, &MainWindow::onAddFolder);

    QAction *saveAsAction = fileMenu->addAction(QString::fromUtf8("另存为..."));
    connect(saveAsAction, &QAction::triggered, this, &MainWindow::onSaveAs);

    fileMenu->addSeparator();
    fileMenu->addAction(QString::fromUtf8("退出"));

    QMenu *editMenu = menuBar()->addMenu(QString::fromUtf8("编辑"));
    editMenu->addAction(QString::fromUtf8("全选"));
    editMenu->addAction(QString::fromUtf8("反选"));

    QMenu *viewMenu = menuBar()->addMenu(QString::fromUtf8("视图"));
    viewMenu->addAction(QString::fromUtf8("显示/隐藏文件区"));
    viewMenu->addAction(QString::fromUtf8("显示/隐藏预览区"));
    viewMenu->addAction(QString::fromUtf8("显示/隐藏参数区"));

    QMenu *toolMenu = menuBar()->addMenu(QString::fromUtf8("工具"));

    QAction *pendingAct = toolMenu->addAction(QString::fromUtf8("待确认中心"));
    connect(pendingAct, &QAction::triggered, this, &MainWindow::onPendingCenter);

    QAction *batchAct = toolMenu->addAction(QString::fromUtf8("批量处理"));
    connect(batchAct, &QAction::triggered, this, &MainWindow::onBatchProcess);

    toolMenu->addAction(QString::fromUtf8("任务队列"));
    toolMenu->addAction(QString::fromUtf8("日志与报告"));

    // ★ 3.3 备份管理（接上槽）
    QAction *backupAct = toolMenu->addAction(QString::fromUtf8("备份管理"));
    connect(backupAct, &QAction::triggered,
            this, &MainWindow::onOpenBackupManager);

    // 3.2 设置（接上槽）
    QAction *settingsAct = toolMenu->addAction(QString::fromUtf8("设置"));
    connect(settingsAct, &QAction::triggered, this, &MainWindow::onOpenSettings);

    QMenu *helpMenu = menuBar()->addMenu(QString::fromUtf8("帮助"));
    helpMenu->addAction(QString::fromUtf8("使用手册"));
    helpMenu->addAction(QString::fromUtf8("快捷键"));
    helpMenu->addAction(QString::fromUtf8("关于"));
}

void MainWindow::setupToolBar()
{
    QToolBar *toolBar = addToolBar(QString::fromUtf8("工具栏"));
    toolBar->setMovable(false);

    QAction *openAction = toolBar->addAction(QString::fromUtf8("打开图片"));
    connect(openAction, &QAction::triggered, this, &MainWindow::onOpenImage);

    QAction *addFolderAction = toolBar->addAction(QString::fromUtf8("添加文件夹"));
    connect(addFolderAction, &QAction::triggered, this, &MainWindow::onAddFolder);

    QAction *saveAsAction = toolBar->addAction(QString::fromUtf8("另存为"));
    connect(saveAsAction, &QAction::triggered, this, &MainWindow::onSaveAs);

    toolBar->addSeparator();

    // 一键处理：下拉菜单 + 勾选框
    {
        QToolButton *oneClickBtn = new QToolButton(toolBar);
        oneClickBtn->setText(QString::fromUtf8("一键处理"));
        oneClickBtn->setPopupMode(QToolButton::InstantPopup);

        QMenu *menu = new QMenu(oneClickBtn);

        auto makeCheckItem = [&](const QString &text, bool checked) -> QCheckBox* {
            QWidget *w = new QWidget(menu);
            QHBoxLayout *lay = new QHBoxLayout(w);
            lay->setContentsMargins(20, 2, 20, 2);
            QCheckBox *cb = new QCheckBox(text, w);
            cb->setChecked(checked);
            lay->addWidget(cb);
            QWidgetAction *wa = new QWidgetAction(menu);
            wa->setDefaultWidget(w);
            menu->addAction(wa);
            return cb;
        };

        m_oneClickDeskewCheck     = makeCheckItem(QString::fromUtf8("自动扶正"), true);
        m_oneClickBlackEdgeCheck  = makeCheckItem(QString::fromUtf8("黑边去除"), true);
        m_oneClickErrPageCheck    = makeCheckItem(QString::fromUtf8("错误页码处理"), false);
        m_oneClickDenoiseCheck    = makeCheckItem(QString::fromUtf8("污点去除"), true);
        m_oneClickEnhanceCheck    = makeCheckItem(QString::fromUtf8("文字加深"), true);
        m_oneClickBackgroundCheck = makeCheckItem(QString::fromUtf8("底色处理"), false);
        m_oneClickColorLineCheck  = makeCheckItem(QString::fromUtf8("彩色细线清除"), false);

        menu->addSeparator();

        QWidget *execWidget = new QWidget(menu);
        QHBoxLayout *execLayout = new QHBoxLayout(execWidget);
        execLayout->setContentsMargins(4, 4, 4, 4);
        QPushButton *execBtn = new QPushButton(QString::fromUtf8("执行"), execWidget);
        execLayout->addWidget(execBtn);
        QWidgetAction *execAction = new QWidgetAction(menu);
        execAction->setDefaultWidget(execWidget);
        menu->addAction(execAction);

        connect(execBtn, &QPushButton::clicked, this, [this, menu]() {
            menu->close();
            onOneClickProcess();
        });

        oneClickBtn->setMenu(menu);
        toolBar->addWidget(oneClickBtn);
    }

    QAction *deskewAction = toolBar->addAction(QString::fromUtf8("自动扶正"));
    connect(deskewAction, &QAction::triggered, this, &MainWindow::onDeskew);

    QAction *blackEdgeAction = toolBar->addAction(QString::fromUtf8("黑边去除"));
    connect(blackEdgeAction, &QAction::triggered, this, &MainWindow::onRemoveBlackEdge);

    QAction *denoiseAction = toolBar->addAction(QString::fromUtf8("污点去除"));
    connect(denoiseAction, &QAction::triggered, this, &MainWindow::onDenoise);

    QAction *enhanceAction = toolBar->addAction(QString::fromUtf8("文字加深"));
    connect(enhanceAction, &QAction::triggered, this, &MainWindow::onEnhance);

    QAction *detectColorAction = toolBar->addAction(QString::fromUtf8("检测彩色细线"));
    connect(detectColorAction, &QAction::triggered, this, &MainWindow::onDetectColorLine);

    QAction *clearColorAction = toolBar->addAction(QString::fromUtf8("清除彩色细线"));
    connect(clearColorAction, &QAction::triggered, this, &MainWindow::onClearColorLine);

    QAction *errPageAction = toolBar->addAction(QString::fromUtf8("错误页码处理"));
    connect(errPageAction, &QAction::triggered, this, &MainWindow::onProcessErrPage);

    QAction *stampTestAction = toolBar->addAction(QString::fromUtf8("印章保护测试"));
    connect(stampTestAction, &QAction::triggered, this, &MainWindow::onStampProtectTest);

    QAction *pendingAction = toolBar->addAction(QString::fromUtf8("待确认中心"));
    connect(pendingAction, &QAction::triggered, this, &MainWindow::onPendingCenter);

    QAction *backgroundAction = toolBar->addAction(QString::fromUtf8("底色处理"));
    connect(backgroundAction, &QAction::triggered, this, &MainWindow::onBackground);

    QAction *batchAction = toolBar->addAction(QString::fromUtf8("批量处理"));
    connect(batchAction, &QAction::triggered, this, &MainWindow::onBatchProcess);

    toolBar->addSeparator();
    toolBar->addAction(QString::fromUtf8("输出设置"));

    QAction *presetAction = toolBar->addAction(QString::fromUtf8("预设"));
    Q_UNUSED(presetAction);

    // 3.2 设置按钮
    QAction *settingsAction = toolBar->addAction(QString::fromUtf8("设置"));
    connect(settingsAction, &QAction::triggered, this, &MainWindow::onOpenSettings);

    toolBar->addSeparator();
    toolBar->addAction(QString::fromUtf8("开始"));
    toolBar->addAction(QString::fromUtf8("暂停"));
    toolBar->addAction(QString::fromUtf8("继续"));
    toolBar->addAction(QString::fromUtf8("取消"));
}

void MainWindow::setupCentralWidget()
{
    QSplitter *mainSplitter = new QSplitter(Qt::Horizontal, this);

    QWidget *leftWidget = new QWidget(this);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    m_folderTree = new QTreeView(leftWidget);
    m_folderTree->setHeaderHidden(true);
    leftLayout->addWidget(m_folderTree);

    m_fileModel = new QStandardItemModel(this);
    m_fileModel->setHorizontalHeaderLabels({
        QString::fromUtf8("文件名"),
        QString::fromUtf8("大小"),
        QString::fromUtf8("路径")
    });

    m_fileTable = new QTableView(leftWidget);
    m_fileTable->setModel(m_fileModel);
    m_fileTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_fileTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_fileTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_fileTable->horizontalHeader()->setStretchLastSection(true);
    m_fileTable->verticalHeader()->setVisible(false);
    connect(m_fileTable, &QTableView::doubleClicked,
            this, &MainWindow::onFileDoubleClicked);
    leftLayout->addWidget(m_fileTable);

    m_previewScene = new QGraphicsScene(this);
    m_previewView = new QGraphicsView(m_previewScene, this);
    m_previewView->setRenderHint(QPainter::SmoothPixmapTransform);
    m_previewView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_previewView->setAlignment(Qt::AlignCenter);

    QScrollArea *paramScroll = new QScrollArea(this);
    QWidget *paramWidget = new QWidget(paramScroll);
    QVBoxLayout *paramLayout = new QVBoxLayout(paramWidget);

    paramLayout->addWidget(new QLabel(QString::fromUtf8("预设")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("污点去除")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("黑边去除")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("自动扶正")));

    buildEnhancePanel(paramLayout);

    paramLayout->addWidget(new QLabel(QString::fromUtf8("彩色故障细线")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("签名印章保护")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("底色处理")));
    paramLayout->addWidget(new QLabel(QString::fromUtf8("输出设置")));
    paramLayout->addStretch();

    paramScroll->setWidget(paramWidget);
    paramScroll->setWidgetResizable(true);

    mainSplitter->addWidget(leftWidget);
    mainSplitter->addWidget(m_previewView);
    mainSplitter->addWidget(paramScroll);
    mainSplitter->setStretchFactor(0, 2);
    mainSplitter->setStretchFactor(1, 3);
    mainSplitter->setStretchFactor(2, 1);

    setCentralWidget(mainSplitter);
}
void MainWindow::buildEnhancePanel(QVBoxLayout *paramLayout)
{
    QToolButton *toggleBtn = new QToolButton(this);
    toggleBtn->setText(QString::fromUtf8("浅色文字加深"));
    toggleBtn->setCheckable(true);
    toggleBtn->setChecked(false);
    toggleBtn->setArrowType(Qt::RightArrow);
    toggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleBtn->setAutoRaise(true);

    QWidget *body = new QWidget(this);
    QGridLayout *grid = new QGridLayout(body);
    grid->setContentsMargins(10, 4, 4, 4);
    grid->setHorizontalSpacing(6);
    grid->setVerticalSpacing(4);

    int row = 0;

    grid->addWidget(new QLabel(QString::fromUtf8("强度"), body), row, 0);
    m_enhanceStrengthSlider = new QSlider(Qt::Horizontal, body);
    m_enhanceStrengthSlider->setRange(1, 30);
    m_enhanceStrengthSlider->setValue(16);
    m_enhanceStrengthSpin = new QSpinBox(body);
    m_enhanceStrengthSpin->setRange(1, 30);
    m_enhanceStrengthSpin->setValue(16);
    m_enhanceStrengthSpin->setSuffix(QString::fromUtf8(" ×0.1"));
    grid->addWidget(m_enhanceStrengthSlider, row, 1);
    grid->addWidget(m_enhanceStrengthSpin, row, 2);
    ++row;

    grid->addWidget(new QLabel(QString::fromUtf8("彩色保护"), body), row, 0);
    m_enhanceColorSatSlider = new QSlider(Qt::Horizontal, body);
    m_enhanceColorSatSlider->setRange(0, 255);
    m_enhanceColorSatSlider->setValue(40);
    m_enhanceColorSatSpin = new QSpinBox(body);
    m_enhanceColorSatSpin->setRange(0, 255);
    m_enhanceColorSatSpin->setValue(40);
    grid->addWidget(m_enhanceColorSatSlider, row, 1);
    grid->addWidget(m_enhanceColorSatSpin, row, 2);
    ++row;

    grid->addWidget(new QLabel(QString::fromUtf8("暗部目标"), body), row, 0);
    m_enhanceDarkTargetSlider = new QSlider(Qt::Horizontal, body);
    m_enhanceDarkTargetSlider->setRange(0, 100);
    m_enhanceDarkTargetSlider->setValue(0);
    m_enhanceDarkTargetSpin = new QSpinBox(body);
    m_enhanceDarkTargetSpin->setRange(0, 100);
    m_enhanceDarkTargetSpin->setValue(0);
    grid->addWidget(m_enhanceDarkTargetSlider, row, 1);
    grid->addWidget(m_enhanceDarkTargetSpin, row, 2);
    ++row;

    grid->addWidget(new QLabel(QString::fromUtf8("纸张目标"), body), row, 0);
    m_enhancePaperTargetSlider = new QSlider(Qt::Horizontal, body);
    m_enhancePaperTargetSlider->setRange(200, 255);
    m_enhancePaperTargetSlider->setValue(255);
    m_enhancePaperTargetSpin = new QSpinBox(body);
    m_enhancePaperTargetSpin->setRange(200, 255);
    m_enhancePaperTargetSpin->setValue(255);
    grid->addWidget(m_enhancePaperTargetSlider, row, 1);
    grid->addWidget(m_enhancePaperTargetSpin, row, 2);
    ++row;

    body->setVisible(false);

    connect(m_enhanceStrengthSlider, &QSlider::valueChanged,
            m_enhanceStrengthSpin, &QSpinBox::setValue);
    connect(m_enhanceStrengthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_enhanceStrengthSlider, &QSlider::setValue);

    connect(m_enhanceColorSatSlider, &QSlider::valueChanged,
            m_enhanceColorSatSpin, &QSpinBox::setValue);
    connect(m_enhanceColorSatSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_enhanceColorSatSlider, &QSlider::setValue);

    connect(m_enhanceDarkTargetSlider, &QSlider::valueChanged,
            m_enhanceDarkTargetSpin, &QSpinBox::setValue);
    connect(m_enhanceDarkTargetSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_enhanceDarkTargetSlider, &QSlider::setValue);

    connect(m_enhancePaperTargetSlider, &QSlider::valueChanged,
            m_enhancePaperTargetSpin, &QSpinBox::setValue);
    connect(m_enhancePaperTargetSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_enhancePaperTargetSlider, &QSlider::setValue);

    connect(m_enhanceStrengthSlider, &QSlider::valueChanged,
            this, &MainWindow::onEnhanceParamChanged);
    connect(m_enhanceColorSatSlider, &QSlider::valueChanged,
            this, &MainWindow::onEnhanceParamChanged);
    connect(m_enhanceDarkTargetSlider, &QSlider::valueChanged,
            this, &MainWindow::onEnhanceParamChanged);
    connect(m_enhancePaperTargetSlider, &QSlider::valueChanged,
            this, &MainWindow::onEnhanceParamChanged);

    connect(toggleBtn, &QToolButton::toggled, this,
            [body, toggleBtn](bool on) {
                body->setVisible(on);
                toggleBtn->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
            });

    paramLayout->addWidget(toggleBtn);
    paramLayout->addWidget(body);
}

process::EnhanceOptions MainWindow::currentEnhanceOptions() const
{
    process::EnhanceOptions opt;

    double raw = m_enhanceStrengthSlider->value() / 10.0;
    if (raw < 0.5) raw = 0.5;
    opt.gamma = raw;

    if (raw <= 1.3) opt.strengthLevel = 0;
    else if (raw <= 1.8) opt.strengthLevel = 1;
    else opt.strengthLevel = 2;

    opt.protectColor = true;
    opt.colorSaturationThreshold = m_enhanceColorSatSlider->value();
    opt.targetDarkGray = m_enhanceDarkTargetSlider->value();
    opt.targetPaperGray = m_enhancePaperTargetSlider->value();

    return opt;
}

void MainWindow::onEnhanceParamChanged()
{
    if (!m_hasImage || m_enhancePreviewBase.empty()) return;
    m_enhanceDebounceTimer->start(300);
}

void MainWindow::onEnhanceDebounceTimeout()
{
    if (m_enhancePreviewBase.empty()) return;

    process::EnhanceOptions opt = currentEnhanceOptions();
    const process::EnhanceResult result =
        process::Enhance::enhanceText(m_enhancePreviewBase, opt);

    if (!result.ok) return;

    showMatOnPreview(result.image);

    if (result.skipped) {
        statusBar()->showMessage(
            QString::fromUtf8("预览：文字已足够黑（文字灰度 %1），跳过")
                .arg(result.darkGray, 0, 'f', 1));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("预览：纸张 %1，文字 %2，加深 %3 像素")
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.darkGray, 0, 'f', 1)
                .arg(result.enhancedPixels));
    }
}

void MainWindow::setupStatusBar()
{
    statusBar()->showMessage(QString::fromUtf8("就绪"));
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    if (m_hasImage) {
        fitPreviewToWindow();
    }
}

void MainWindow::fitPreviewToWindow()
{
    if (!m_previewScene || !m_previewView) {
        return;
    }
    const QRectF rect = m_previewScene->itemsBoundingRect();
    if (rect.isEmpty()) {
        return;
    }
    m_previewView->resetTransform();
    m_previewView->fitInView(rect, Qt::KeepAspectRatio);
}

void MainWindow::onAddFolder()
{
    const QString folder = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择图片文件夹"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (folder.isEmpty()) {
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在扫描：%1 ...").arg(folder));

    const QStringList files = core::FileScanner::scanFolder(folder);
    m_currentFiles = files;
    m_currentFolder = folder;
    fillFileTable(files);

    statusBar()->showMessage(
        QString::fromUtf8("扫描完成：%1 个文件，来自 %2")
            .arg(files.size())
            .arg(folder));
}

void MainWindow::fillFileTable(const QStringList &files)
{
    m_fileModel->removeRows(0, m_fileModel->rowCount());

    for (const QString &path : files) {
        const QFileInfo info(path);

        QList<QStandardItem *> row;
        row << new QStandardItem(info.fileName());
        row << new QStandardItem(QString::number(info.size() / 1024) + QString::fromUtf8(" KB"));
        row << new QStandardItem(info.absolutePath());

        row[0]->setData(path, Qt::UserRole);
        row[0]->setToolTip(path);

        m_fileModel->appendRow(row);
    }

    m_fileTable->resizeColumnsToContents();
}

void MainWindow::onFileDoubleClicked(const QModelIndex &index)
{
    if (!index.isValid()) {
        return;
    }

    const QModelIndex firstColIndex = m_fileModel->index(index.row(), 0);
    const QString path = m_fileModel->data(firstColIndex, Qt::UserRole).toString();
    if (path.isEmpty()) {
        return;
    }

    showImageOnPreview(path);
}

void MainWindow::onOpenImage()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QString::fromUtf8("选择图片"),
        QString(),
        QString::fromUtf8("图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"));

    if (path.isEmpty()) {
        return;
    }

    showImageOnPreview(path);
}

void MainWindow::onSaveAs()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    QString defaultName = QStringLiteral("processed.jpg");
    if (!m_currentImagePath.isEmpty()) {
        const QFileInfo info(m_currentImagePath);
        defaultName = info.completeBaseName() + QStringLiteral("_processed.jpg");
    }

    const QString savePath = QFileDialog::getSaveFileName(
        this,
        QString::fromUtf8("另存为"),
        defaultName,
        QString::fromUtf8("JPG 图片 (*.jpg *.jpeg)"));

    if (savePath.isEmpty()) {
        return;
    }

    image::JpegSaveOptions options;
    options.quality = 98;
    options.dpiX = 300.0;
    options.dpiY = 300.0;
    options.use444Sampling = true;

    if (!image::JpegWriter::write(savePath, m_currentMat, options)) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("保存失败：%1").arg(savePath));
        return;
    }

    statusBar()->showMessage(
        QString::fromUtf8("已保存：%1").arg(savePath));
}
void MainWindow::onDeskew()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在自动扶正..."));

    const process::DeskewResult result = process::Deskew::autoDeskew(m_currentMat);

    if (!result.ok) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("无法判断倾斜角，或倾斜角过大，已跳过。"));
        statusBar()->showMessage(QString::fromUtf8("扶正已跳过"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    statusBar()->showMessage(
        QString::fromUtf8("自动扶正完成，旋转 %1 度")
            .arg(result.angle, 0, 'f', 2));
}

void MainWindow::onRemoveBlackEdge()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在去除黑边..."));

    process::BlackEdgeOptions options;
    options.paperSampleRatio   = 0.6;
    options.paperPercentile    = 0.9;
    options.darkRatio          = 0.75;
    options.maxScanRatio       = 0.30;
    options.darkPixelRatio     = 0.50;
    options.gapTolerance       = 3;
    options.smoothKernelSize   = 5;
    options.expandPixels       = 5;
    options.fillWhite          = true;

    const process::BlackEdgeResult result =
        process::BlackEdge::removeBlackEdge(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("黑边去除失败。"));
        statusBar()->showMessage(QString::fromUtf8("黑边去除失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    if (result.skipped) {
        statusBar()->showMessage(
            QString::fromUtf8("黑边去除：未检测到明显黑边（纸张灰度 %1，阈值 %2）")
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.darkThreshold, 0, 'f', 1));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("黑边去除完成：上 %1 / 下 %2 / 左 %3 / 右 %4 像素")
                .arg(result.topPixels)
                .arg(result.bottomPixels)
                .arg(result.leftPixels)
                .arg(result.rightPixels));
    }
}

void MainWindow::onDenoise()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    const cv::Mat srcBackup = m_currentMat.clone();

    statusBar()->showMessage(QString::fromUtf8("正在去除污点..."));

    process::DenoiseOptions options;
    options.maxSpotArea   = 200;
    options.maxSpotWidth  = 30;
    options.maxSpotHeight = 30;
    options.darkRatio     = 0.60;
    options.protectRadius = 2;
    options.strengthLevel = 1;
    options.useInpaint    = true;

    {
        protect::StampProtectOptions protectOpt;
        const protect::StampProtectResult pr =
            protect::StampProtect::detect(m_currentMat, protectOpt);
        if (pr.ok) {
            options.protectMask = pr.mask;
        }
    }

    const process::DenoiseResult result =
        process::Denoise::removeSpots(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("污点去除失败。"));
        statusBar()->showMessage(QString::fromUtf8("污点去除失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    if (result.skipped) {
        statusBar()->showMessage(
            QString::fromUtf8("污点去除：未检测到明显污点，已跳过"));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("污点去除完成：检测到 %1 处污点，修补 %2 像素")
                .arg(result.spotCount)
                .arg(result.cleanedPixels));
    }

    {
        const QList<analyze::PendingItem> allOld =
            analyze::PendingCenter::instance().allItems();
        QList<int> idsToDrop;
        for (const analyze::PendingItem &x : allOld) {
            if (x.sourceImagePath == m_currentImagePath &&
                x.type == analyze::PendingType::YellowBlob) {
                idsToDrop.append(x.id);
            }
        }
        if (!idsToDrop.isEmpty()) {
            analyze::PendingCenter::instance().setDecisionByIds(
                idsToDrop, analyze::PendingDecision::Rejected);
            analyze::PendingCenter::instance().clearDecided();
        }
    }

    // 暂时禁用黄色污渍检测（检测不准，会乱加项）
    if (false && !result.yellowBlobs.empty()) {
        const QFileInfo fi(m_currentImagePath);
        for (const cv::Rect &r : result.yellowBlobs) {
            analyze::PendingItem p;
            p.type = analyze::PendingType::YellowBlob;
            p.suggestedAction = analyze::PendingAction::Remove;
            p.sourceImagePath = m_currentImagePath;
            p.fileName = fi.fileName();
            p.boundingBox = r;
            p.confidence = 0.7;
            p.reason = QString::fromUtf8("检测到黄色污渍");
            p.detail = QString::fromUtf8("位置：(%1,%2) 大小：%3x%4")
                           .arg(r.x).arg(r.y).arg(r.width).arg(r.height);
            cv::Rect safe = r & cv::Rect(0, 0, srcBackup.cols, srcBackup.rows);
            if (safe.width > 0 && safe.height > 0) {
                p.thumbnail = srcBackup(safe).clone();
            }
            analyze::PendingCenter::instance().addItem(p);
        }
    }
}

void MainWindow::onEnhance()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在加深浅色文字..."));

    process::EnhanceOptions options = currentEnhanceOptions();

    const process::EnhanceResult result =
        process::Enhance::enhanceText(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("文字加深失败。"));
        statusBar()->showMessage(QString::fromUtf8("文字加深失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    if (result.skipped) {
        if (result.paperGray < 150.0) {
            statusBar()->showMessage(
                QString::fromUtf8("文字加深：纸张过暗，已跳过（纸张灰度 %1）")
                    .arg(result.paperGray, 0, 'f', 1));
        } else {
            statusBar()->showMessage(
                QString::fromUtf8("文字加深：文字已足够黑（文字灰度 %1），跳过")
                    .arg(result.darkGray, 0, 'f', 1));
        }
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("文字加深完成：纸张灰度 %1，文字灰度 %2，加深 %3 像素")
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.darkGray, 0, 'f', 1)
                .arg(result.enhancedPixels));
    }
}

void MainWindow::onBackground()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在处理底色..."));

    process::BackgroundOptions options;
    options.paperSampleRatio = 0.6;
    options.paperPercentile  = 0.9;
    options.contentRatio     = 0.70;
    options.targetPaperGray  = 255;
    options.protectColor     = true;
    options.colorSatMin      = 40;

    {
        protect::StampProtectOptions protectOpt;
        const protect::StampProtectResult pr =
            protect::StampProtect::detect(m_currentMat, protectOpt);
        if (pr.ok) {
            options.protectMask = pr.mask;
        }
    }

    const process::BackgroundResult result =
        process::Background::whiten(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("底色处理失败。"));
        statusBar()->showMessage(QString::fromUtf8("底色处理失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    if (result.skipped) {
        statusBar()->showMessage(
            QString::fromUtf8("底色处理：纸张已经很白（灰度 %1），已跳过")
                .arg(result.paperGray, 0, 'f', 1));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("底色处理完成：纸张灰度 %1，白化 %2 像素")
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.whitenedPixels));
    }
}

void MainWindow::onOneClickProcess()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("一键处理中，请稍候..."));

    QStringList doneSteps;
    int pendingCount = 0;

    // 1. 自动扶正
    if (m_oneClickDeskewCheck && m_oneClickDeskewCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在自动扶正..."));
        const process::DeskewResult r = process::Deskew::autoDeskew(m_currentMat);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("扶正");
        }
    }

    // 2. 黑边去除
    if (m_oneClickBlackEdgeCheck && m_oneClickBlackEdgeCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在去除黑边..."));
        process::BlackEdgeOptions opt;
        opt.paperSampleRatio = 0.6;
        opt.paperPercentile  = 0.9;
        opt.darkRatio        = 0.75;
        opt.maxScanRatio     = 0.30;
        opt.darkPixelRatio   = 0.50;
        opt.gapTolerance     = 3;
        opt.smoothKernelSize = 5;
        opt.expandPixels     = 5;
        opt.fillWhite        = true;

        const process::BlackEdgeResult r =
            process::BlackEdge::removeBlackEdge(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("黑边");
        }
    }

    // 3. 错误页码处理
    if (m_oneClickErrPageCheck && m_oneClickErrPageCheck->isChecked()
        && !m_currentImagePath.isEmpty()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在处理错误页码..."));
        process::ErrPageOptions opt;
        opt.detectTopLeft          = true;
        opt.topLeftWidthRatio      = 0.12;
        opt.topLeftHeightRatio     = 0.08;
        opt.detectTopRight         = true;
        opt.topRightWidthRatio     = 0.25;
        opt.topRightHeightRatio    = 0.15;
        opt.detectBottomRight      = true;
        opt.bottomRightWidthRatio  = 0.25;
        opt.bottomRightHeightRatio = 0.15;
        opt.detectBottomLeft       = false;
        opt.minDigitHeight         = 20;
        opt.maxDigitHeight         = 100;
        opt.minDigitWidth          = 10;
        opt.maxDigitWidth          = 150;
        opt.crossLineRatio         = 0.5;
        opt.fillWhite              = true;
        opt.tesseractPath          = QString();

        const process::ErrPageResult r =
            process::ErrPage::process(m_currentMat, m_currentImagePath, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("错误页码");
        }
    }

    // 4. 污点去除（含印章保护）
    if (m_oneClickDenoiseCheck && m_oneClickDenoiseCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在去除污点..."));

        process::DenoiseOptions opt;
        opt.maxSpotArea   = 200;
        opt.maxSpotWidth  = 30;
        opt.maxSpotHeight = 30;
        opt.darkRatio     = 0.60;
        opt.protectRadius = 2;
        opt.strengthLevel = 1;
        opt.useInpaint    = true;

        protect::StampProtectOptions protectOpt;
        const protect::StampProtectResult pr =
            protect::StampProtect::detect(m_currentMat, protectOpt);
        if (pr.ok) {
            opt.protectMask = pr.mask;
        }

        const process::DenoiseResult r =
            process::Denoise::removeSpots(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("污点");
        }
    }

    // 5. 文字加深
    if (m_oneClickEnhanceCheck && m_oneClickEnhanceCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在加深文字..."));
        process::EnhanceOptions opt = currentEnhanceOptions();
        const process::EnhanceResult r =
            process::Enhance::enhanceText(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("文字");
        }
    }

    // 6. 底色处理
    if (m_oneClickBackgroundCheck && m_oneClickBackgroundCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在处理底色..."));
        process::BackgroundOptions opt;
        opt.paperSampleRatio = 0.6;
        opt.paperPercentile  = 0.9;
        opt.contentRatio     = 0.70;
        opt.targetPaperGray  = 255;
        opt.protectColor     = true;
        opt.colorSatMin      = 40;

        protect::StampProtectOptions protectOpt;
        const protect::StampProtectResult pr =
            protect::StampProtect::detect(m_currentMat, protectOpt);
        if (pr.ok) {
            opt.protectMask = pr.mask;
        }

        const process::BackgroundResult r =
            process::Background::whiten(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("底色");
        }
    }

    // 7. 彩色细线清除
    if (m_oneClickColorLineCheck && m_oneClickColorLineCheck->isChecked()) {
        statusBar()->showMessage(QString::fromUtf8("一键处理：正在清除彩色细线..."));
        process::ColorLineOptions opt;
        opt.channelDiffThreshold = 8;
        opt.valueThreshold       = 180;
        opt.colorRatioThreshold  = 0.40;
        opt.maxThickness         = 8;
        opt.edgeMarginRatio      = 0.02;

        const process::ColorLineResult r =
            process::ColorLine::clear(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            doneSteps << QString::fromUtf8("彩色细线");
        }
    }

    showMatOnPreview(m_currentMat);

    pendingCount = analyze::PendingCenter::instance().pendingCount();

    QString msg;
    if (doneSteps.isEmpty()) {
        msg = QString::fromUtf8("一键处理：未勾选任何步骤");
    } else {
        msg = QString::fromUtf8("一键处理完成：%1")
                  .arg(doneSteps.join(QString::fromUtf8(" → ")));
    }

    if (pendingCount > 0) {
        msg += QString::fromUtf8("。待确认 %1 项，请打开待确认中心处理。")
                   .arg(pendingCount);
    } else {
        msg += QString::fromUtf8("。请点另存为保存。");
    }

    statusBar()->showMessage(msg);
}

void MainWindow::onDetectColorLine()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在检测彩色细线..."));

    m_colorLineSource = m_currentMat.clone();

    process::ColorLineOptions options;
    options.channelDiffThreshold = 8;
    options.valueThreshold       = 180;
    options.colorRatioThreshold  = 0.40;
    options.maxThickness         = 8;
    options.edgeMarginRatio      = 0.02;

    const process::ColorLineResult result =
        process::ColorLine::detect(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("彩色细线检测失败。"));
        statusBar()->showMessage(QString::fromUtf8("检测失败"));
        return;
    }

    showMatOnPreview(result.markedImage);

    if (result.items.empty()) {
        statusBar()->showMessage(
            QString::fromUtf8("彩色细线检测：未检测到彩色故障细线"));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("彩色细线检测完成：检测到 %1 条。点清除彩色细线可清除。")
                .arg(static_cast<int>(result.items.size())));
    }
}

void MainWindow::onClearColorLine()
{
    if (m_colorLineSource.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先点检测彩色细线。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在清除彩色细线..."));

    process::ColorLineOptions options;
    options.channelDiffThreshold = 8;
    options.valueThreshold       = 180;
    options.colorRatioThreshold  = 0.40;
    options.maxThickness         = 8;
    options.edgeMarginRatio      = 0.02;

    const process::ColorLineResult result =
        process::ColorLine::clear(m_colorLineSource, options);

    if (!result.ok || result.items.empty()) {
        statusBar()->showMessage(QString::fromUtf8("没有需要清除的彩色细线"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    statusBar()->showMessage(
        QString::fromUtf8("彩色细线清除完成：清除 %1 条，填白 %2 像素")
            .arg(static_cast<int>(result.items.size()))
            .arg(result.clearedPixels));
}

void MainWindow::onProcessErrPage()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    if (m_currentImagePath.isEmpty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请通过打开图片或双击列表打开，才能解析文件名页码。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在处理错误页码..."));

    process::ErrPageOptions options;
    options.detectTopLeft          = true;
    options.topLeftWidthRatio      = 0.12;
    options.topLeftHeightRatio     = 0.08;
    options.detectTopRight         = true;
    options.topRightWidthRatio     = 0.25;
    options.topRightHeightRatio    = 0.15;
    options.detectBottomRight      = true;
    options.bottomRightWidthRatio  = 0.25;
    options.bottomRightHeightRatio = 0.15;
    options.detectBottomLeft       = false;
    options.minDigitHeight         = 20;
    options.maxDigitHeight         = 100;
    options.minDigitWidth          = 10;
    options.maxDigitWidth          = 150;
    options.crossLineRatio         = 0.5;
    options.fillWhite              = true;
    options.tesseractPath          = QString();

    const cv::Mat srcForThumb = m_currentMat.clone();

    const process::ErrPageResult result =
        process::ErrPage::process(m_currentMat, m_currentImagePath, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("错误页码处理失败。"));
        statusBar()->showMessage(QString::fromUtf8("错误页码处理失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(result.image);

    {
        const QList<analyze::PendingItem> allItems =
            analyze::PendingCenter::instance().allItems();
        QList<int> idsToDrop;
        for (const analyze::PendingItem &x : allItems) {
            if (x.sourceImagePath == m_currentImagePath) {
                idsToDrop.append(x.id);
            }
        }
        if (!idsToDrop.isEmpty()) {
            analyze::PendingCenter::instance().setDecisionByIds(
                idsToDrop, analyze::PendingDecision::Rejected);
            analyze::PendingCenter::instance().clearDecided();
        }
    }

    const QFileInfo fi(m_currentImagePath);
    int addedCount = 0;

    for (const auto &it : result.items) {
        const bool matchesCorrect =
            (result.correctPage >= 0 &&
             it.recognizedNumber == result.correctPage);
        if (matchesCorrect) continue;
        if (!it.isCrossed) continue;
        if (it.recognizedNumber < 10) continue;

        analyze::PendingItem p;
        p.type = analyze::PendingType::WrongPageNumber;
        p.suggestedAction = analyze::PendingAction::Remove;
        p.sourceImagePath = m_currentImagePath;
        p.fileName = fi.fileName();
        p.boundingBox = it.boundingBox;
        p.confidence = it.confidence;
        p.reason = QString::fromUtf8("检测到划线数字，与正确页码不符");
        p.detail = QString::fromUtf8("识别结果：%1，正确页码：%2")
                       .arg(it.recognizedNumber)
                       .arg(result.correctPage);

        cv::Rect safe = it.boundingBox &
                        cv::Rect(0, 0, srcForThumb.cols, srcForThumb.rows);
        if (safe.width > 0 && safe.height > 0) {
            p.thumbnail = srcForThumb(safe).clone();
        }

        analyze::PendingCenter::instance().addItem(p);
        ++addedCount;
    }

    QString pageInfo;
    if (result.correctPage >= 0) {
        pageInfo = QString::fromUtf8("正确页码 %1").arg(result.correctPage);
    } else {
        pageInfo = QString::fromUtf8("文件名无页码");
    }

    QString msg;
    if (result.skipped) {
        msg = QString::fromUtf8("错误页码处理：角落未检测到数字。%1")
                  .arg(pageInfo);
    } else {
        msg = QString::fromUtf8("错误页码处理完成：%1，检测到 %2 个数字块，自动删除划线 %3 个，待确认 %4 个")
                  .arg(pageInfo)
                  .arg(static_cast<int>(result.items.size()))
                  .arg(result.crossedRemoved)
                  .arg(result.pendingCount);
    }
    if (addedCount > 0) {
        msg += QString::fromUtf8("（已加入待确认中心 %1 项）").arg(addedCount);
    }

    statusBar()->showMessage(msg);
}

void MainWindow::onStampProtectTest()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在检测签名印章保护区域..."));

    protect::StampProtectOptions options;
    const protect::StampProtectResult result =
        protect::StampProtect::detect(m_currentMat, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("签名印章保护检测失败。"));
        statusBar()->showMessage(QString::fromUtf8("签名印章检测失败"));
        return;
    }

    const QString desktop = QDir::homePath() + QStringLiteral("/Desktop");

    const QString maskPath = desktop + QStringLiteral("/stamp_mask.png");
    cv::imwrite(maskPath.toStdString(), result.mask);

    const QString colorMaskPath = desktop + QStringLiteral("/stamp_colormask.png");
    cv::imwrite(colorMaskPath.toStdString(), result.colorMask);

    const QString hwMaskPath = desktop + QStringLiteral("/stamp_handwriting.png");
    cv::imwrite(hwMaskPath.toStdString(), result.handwritingMask);

    cv::Mat bgr;
    if (m_currentMat.channels() == 3) {
        bgr = m_currentMat.clone();
    } else if (m_currentMat.channels() == 4) {
        cv::cvtColor(m_currentMat, bgr, cv::COLOR_BGRA2BGR);
    } else {
        cv::cvtColor(m_currentMat, bgr, cv::COLOR_GRAY2BGR);
    }

    cv::Mat overlay = bgr.clone();
    overlay.setTo(cv::Scalar(0, 255, 0), result.mask);

    cv::Mat blended;
    cv::addWeighted(bgr, 0.6, overlay, 0.4, 0, blended);

    const QString overlayPath = desktop + QStringLiteral("/stamp_overlay.png");
    cv::imwrite(overlayPath.toStdString(), blended);

    const int totalPixels = m_currentMat.cols * m_currentMat.rows;
    const double ratio = totalPixels > 0
        ? (100.0 * result.protectedPixels / totalPixels)
        : 0.0;
    const double hwRatio = totalPixels > 0
        ? (100.0 * result.handwritingPixels / totalPixels)
        : 0.0;

    statusBar()->showMessage(
        QString::fromUtf8("签名印章检测完成：保护 %1 像素（占 %2%），其中手写签名 %3% 。"
                          "已保存 4 张图到桌面")
            .arg(result.protectedPixels)
            .arg(ratio, 0, 'f', 2)
            .arg(hwRatio, 0, 'f', 2));
}

void MainWindow::onBatchProcess()
{
    if (m_currentFiles.isEmpty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先点\"添加文件夹\"扫描图片。"));
        return;
    }

    if (m_batchDialog) {
        m_batchDialog->raise();
        m_batchDialog->activateWindow();
        return;
    }

    m_batchDialog = new batch::BatchDialog(m_currentFiles, m_currentFolder, this);
    m_batchDialog->setAttribute(Qt::WA_DeleteOnClose);

    connect(m_batchDialog, &batch::BatchDialog::previewImageReady,
            this, &MainWindow::onBatchPreview);

    connect(m_batchDialog, &QObject::destroyed, this, [this]() {
        m_batchDialog = nullptr;
    });

    m_batchDialog->show();
    m_batchDialog->moveToTopRight();
}

// 3.2 打开设置窗口
void MainWindow::onOpenSettings()
{
    if (m_settingsDialog) {
        m_settingsDialog->raise();
        m_settingsDialog->activateWindow();
        return;
    }

    m_settingsDialog = new ui::SettingsDialog(this);
    m_settingsDialog->setAttribute(Qt::WA_DeleteOnClose);

    connect(m_settingsDialog, &QObject::destroyed, this, [this]() {
        m_settingsDialog = nullptr;
    });

    m_settingsDialog->show();
}

// ★ 3.3 打开备份管理窗口
void MainWindow::onOpenBackupManager()
{
    if (m_backupDialog) {
        m_backupDialog->raise();
        m_backupDialog->activateWindow();
        return;
    }

    m_backupDialog = new backup::BackupDialog(this);
    m_backupDialog->setAttribute(Qt::WA_DeleteOnClose);

    connect(m_backupDialog, &QObject::destroyed, this, [this]() {
        m_backupDialog = nullptr;
    });

    m_backupDialog->show();
}

void MainWindow::onBatchPreview(const cv::Mat &mat, bool before)
{
    Q_UNUSED(before);
    if (mat.empty()) return;
    showMatOnPreview(mat);
}

void MainWindow::onPendingCenter()
{
    analyze::PendingDialog dlg(this);
    dlg.exec();

    applyAcceptedYellowBlobs();
}

void MainWindow::applyAcceptedYellowBlobs()
{
    if (m_currentMat.empty()) return;

    const QList<analyze::PendingItem> allItems =
        analyze::PendingCenter::instance().allItems();

    int applied = 0;
    cv::Scalar white;
    if (m_currentMat.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    for (const analyze::PendingItem &item : allItems) {
        if (item.type != analyze::PendingType::YellowBlob) continue;
        if (item.decision != analyze::PendingDecision::Accepted) continue;
        if (item.sourceImagePath != m_currentImagePath) continue;

        cv::Rect r = item.boundingBox &
            cv::Rect(0, 0, m_currentMat.cols, m_currentMat.rows);
        if (r.width <= 0 || r.height <= 0) continue;

        cv::rectangle(m_currentMat, r, white, cv::FILLED);
        ++applied;
    }

    if (applied > 0) {
        showMatOnPreview(m_currentMat);
        statusBar()->showMessage(
            QString::fromUtf8("已处理 %1 处黄色污渍").arg(applied));
    }
}

void MainWindow::showMatOnPreview(const cv::Mat &mat)
{
    if (mat.empty()) {
        return;
    }

    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    }

    QImage qimg(rgb.data, rgb.cols, rgb.rows,
                static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());

    m_previewScene->clear();
    m_previewScene->addPixmap(pix);
    m_previewScene->setSceneRect(pix.rect());

    m_hasImage = true;

    QTimer::singleShot(0, this, [this]() {
        fitPreviewToWindow();
    });
}

void MainWindow::showImageOnPreview(const QString &path)
{
    cv::Mat mat;
    image::ImageMeta meta;
    if (!image::ImageIO::read(path, mat, meta)) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("无法读取图片：%1").arg(path));
        return;
    }

    m_originalMat = mat.clone();
    m_currentMat = mat;
    m_currentImagePath = path;

    m_enhancePreviewBase = mat.clone();

    showMatOnPreview(m_currentMat);

    statusBar()->showMessage(
        QString::fromUtf8("%1  |  %2 x %3  |  %4 通道")
            .arg(QFileInfo(path).fileName())
            .arg(meta.width)
            .arg(meta.height)
            .arg(meta.channels));
}