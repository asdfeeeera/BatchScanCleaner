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

    QAction *deskewAction = toolBar->addAction(QString::fromUtf8("自动扶正"));
    connect(deskewAction, &QAction::triggered, this, &MainWindow::onDeskew);

    QAction *blackEdgeAction = toolBar->addAction(QString::fromUtf8("黑边去除"));
    connect(blackEdgeAction, &QAction::triggered, this, &MainWindow::onRemoveBlackEdge);

    QAction *denoiseAction = toolBar->addAction(QString::fromUtf8("污点去除"));
    connect(denoiseAction, &QAction::triggered, this, &MainWindow::onDenoise);

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
            QString::fromUtf8("黑边去除完成：上 %1 / 下 %2 / 左 %3 / 右 %4 像素，纸张灰度 %5，阈值 %6")
                .arg(result.topPixels)
                .arg(result.bottomPixels)
                .arg(result.leftPixels)
                .arg(result.rightPixels)
                .arg(result.paperGray, 0, 'f', 1)
                .arg(result.darkThreshold, 0, 'f', 1));
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
    options.strengthLevel = 1;   // 标准
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