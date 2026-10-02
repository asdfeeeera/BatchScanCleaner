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
#include <QWidget>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QFileInfo>
#include <QResizeEvent>
#include <QTimer>
#include <QDir>
#include <QDebug>

#include <opencv2/imgproc.hpp>

#include "image_io.h"
#include "jpeg_writer.h"
#include "file_scanner.h"
#include "deskew.h"
#include "blackedge.h"
#include "denoise.h"
#include "enhance.h"
#include "colorline.h"
#include "errpage.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QString::fromUtf8("批量扫描图片净化增强软件"));
    resize(1280, 800);

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
    toolMenu->addAction(QString::fromUtf8("待确认中心"));
    toolMenu->addAction(QString::fromUtf8("任务队列"));
    toolMenu->addAction(QString::fromUtf8("日志与报告"));
    toolMenu->addAction(QString::fromUtf8("备份管理"));
    toolMenu->addAction(QString::fromUtf8("设置"));

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

    QAction *oneClickAction = toolBar->addAction(QString::fromUtf8("一键处理"));
    connect(oneClickAction, &QAction::triggered, this, &MainWindow::onOneClickProcess);

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

    toolBar->addSeparator();
    toolBar->addAction(QString::fromUtf8("输出设置"));
    toolBar->addAction(QString::fromUtf8("预设"));
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
    paramLayout->addWidget(new QLabel(QString::fromUtf8("浅色文字加深")));
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
    showMatOnPreview(result.markedImage);

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

    statusBar()->showMessage(QString::fromUtf8("正在去除污点..."));

    process::DenoiseOptions options;
    options.maxSpotArea   = 200;
    options.maxSpotWidth  = 30;
    options.maxSpotHeight = 30;
    options.darkRatio     = 0.60;
    options.protectRadius = 2;
    options.strengthLevel = 1;
    options.useInpaint    = true;

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
}

void MainWindow::onEnhance()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在加深浅色文字..."));

    process::EnhanceOptions options;
    options.strengthLevel = 1;
    options.protectColor = true;
    options.targetDarkGray = 0;
    options.targetPaperGray = 255;
    options.colorSaturationThreshold = 40;

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
        statusBar()->showMessage(
            QString::fromUtf8("文字加深：纸张过暗，已跳过（纸张灰度 %1）")
                .arg(result.paperGray, 0, 'f', 1));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("文字加深完成：纸张灰度 %1，文字灰度 %2，加深 %3 像素")
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.darkGray, 0, 'f', 1)
                .arg(result.enhancedPixels));
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

    statusBar()->showMessage(QString::fromUtf8("一键处理：正在自动扶正..."));
    double deskewAngle = 0.0;
    {
        const process::DeskewResult r = process::Deskew::autoDeskew(m_currentMat);
        if (r.ok) {
            m_currentMat = r.image;
            deskewAngle = r.angle;
        }
    }

    statusBar()->showMessage(QString::fromUtf8("一键处理：正在去除黑边..."));
    int blackEdgeTotal = 0;
    {
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
            blackEdgeTotal = r.topPixels + r.bottomPixels
                             + r.leftPixels + r.rightPixels;
        }
    }

    statusBar()->showMessage(QString::fromUtf8("一键处理：正在去除污点..."));
    int spotCount = 0;
    {
        process::DenoiseOptions opt;
        opt.maxSpotArea   = 200;
        opt.maxSpotWidth  = 30;
        opt.maxSpotHeight = 30;
        opt.darkRatio     = 0.60;
        opt.protectRadius = 2;
        opt.strengthLevel = 1;
        opt.useInpaint    = true;

        const process::DenoiseResult r =
            process::Denoise::removeSpots(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
            spotCount = r.spotCount;
        }
    }

    statusBar()->showMessage(QString::fromUtf8("一键处理：正在加深文字..."));
    {
        process::EnhanceOptions opt;
        opt.strengthLevel = 1;
        opt.protectColor = true;
        opt.targetDarkGray = 0;
        opt.targetPaperGray = 255;
        opt.colorSaturationThreshold = 40;

        const process::EnhanceResult r =
            process::Enhance::enhanceText(m_currentMat, opt);
        if (r.ok) {
            m_currentMat = r.image;
        }
    }

    showMatOnPreview(m_currentMat);

    statusBar()->showMessage(
        QString::fromUtf8("一键处理完成：扶正 %1 度，黑边 %2 像素，污点 %3 处。请点\"另存为\"保存。")
            .arg(deskewAngle, 0, 'f', 2)
            .arg(blackEdgeTotal)
            .arg(spotCount));
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
            QString::fromUtf8("彩色细线检测完成：检测到 %1 条。点\"清除彩色细线\"可清除。")
                .arg(static_cast<int>(result.items.size())));
    }
}

void MainWindow::onClearColorLine()
{
    if (m_colorLineSource.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先点\"检测彩色细线\"。"));
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
                                 QString::fromUtf8("请通过\"打开图片\"或双击列表打开，才能解析文件名页码。"));
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
    options.detectBottomRight      = true;    // ★ 新增
    options.bottomRightWidthRatio  = 0.25;    // ★ 新增
    options.bottomRightHeightRatio = 0.15;    // ★ 新增
    options.detectBottomLeft       = false;
    options.minDigitHeight         = 20;
    options.maxDigitHeight         = 100;
    options.minDigitWidth          = 10;
    options.maxDigitWidth          = 100;
    options.crossLineRatio         = 0.5;
    options.fillWhite              = true;
    options.tesseractPath          = QString();   // 空 = 自动查找 exe 同目录

    const process::ErrPageResult result =
        process::ErrPage::process(m_currentMat, m_currentImagePath, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("错误页码处理失败。"));
        statusBar()->showMessage(QString::fromUtf8("错误页码处理失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(m_currentMat);

    QString pageInfo;
    if (result.correctPage >= 0) {
        pageInfo = QString::fromUtf8("正确页码 %1").arg(result.correctPage);
    } else {
        pageInfo = QString::fromUtf8("文件名无页码");
    }

    if (result.skipped) {
        statusBar()->showMessage(
            QString::fromUtf8("错误页码处理：角落未检测到数字。%1").arg(pageInfo));
    } else {
        statusBar()->showMessage(
            QString::fromUtf8("错误页码处理完成：%1，检测到 %2 个数字块，自动删除划线 %3 个，待确认 %4 个")
                .arg(pageInfo)
                .arg(static_cast<int>(result.items.size()))
                .arg(result.crossedRemoved)
                .arg(result.pendingCount));
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

    showMatOnPreview(m_currentMat);

    statusBar()->showMessage(
        QString::fromUtf8("%1  |  %2 x %3  |  %4 通道")
            .arg(QFileInfo(path).fileName())
            .arg(meta.width)
            .arg(meta.height)
            .arg(meta.channels));
}