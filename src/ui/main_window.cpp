#include "main_window.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QSplitter>
#include <QTreeWidget>
#include <QTableView>
#include <QListView>
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
#include <QApplication>
#include <QStandardPaths>

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
#include "settings_dialog.h"
#include "../backup/backup_dialog.h"
#include "../annotate/annotation_dialog.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QString::fromUtf8("批量扫描图片净化增强软件"));
    resize(1366, 850);

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
    fileMenu->addAction(QString::fromUtf8("打开图片..."), this, &MainWindow::onOpenImage);
    fileMenu->addAction(QString::fromUtf8("添加文件夹..."), this, &MainWindow::onAddFolder);
    fileMenu->addAction(QString::fromUtf8("另存为..."), this, &MainWindow::onSaveAs);
    fileMenu->addSeparator();
    fileMenu->addAction(QString::fromUtf8("退出"), this, &QWidget::close);

    QMenu *toolMenu = menuBar()->addMenu(QString::fromUtf8("工具"));
    toolMenu->addAction(QString::fromUtf8("待确认中心"), this, &MainWindow::onPendingCenter);
    toolMenu->addAction(QString::fromUtf8("批量处理"), this, &MainWindow::onBatchProcess);
    toolMenu->addAction(QString::fromUtf8("备份管理"), this, &MainWindow::onOpenBackupManager);
    toolMenu->addAction(QString::fromUtf8("标注工具"), this, &MainWindow::onOpenAnnotator);
    toolMenu->addAction(QString::fromUtf8("设置"), this, &MainWindow::onOpenSettings);
}

void MainWindow::setupToolBar()
{
    QToolBar *toolBar = addToolBar(QString::fromUtf8("工具栏"));
    toolBar->setMovable(false);

    toolBar->addAction(QString::fromUtf8("打开图片"), this, &MainWindow::onOpenImage);
    toolBar->addAction(QString::fromUtf8("添加文件夹"), this, &MainWindow::onAddFolder);
    toolBar->addAction(QString::fromUtf8("另存为"), this, &MainWindow::onSaveAs);
    toolBar->addSeparator();

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

    toolBar->addAction(QString::fromUtf8("自动扶正"), this, &MainWindow::onDeskew);
    toolBar->addAction(QString::fromUtf8("黑边去除"), this, &MainWindow::onRemoveBlackEdge);
    toolBar->addAction(QString::fromUtf8("污点去除"), this, &MainWindow::onDenoise);
    toolBar->addAction(QString::fromUtf8("文字加深"), this, &MainWindow::onEnhance);
    toolBar->addAction(QString::fromUtf8("错误页码"), this, &MainWindow::onProcessErrPage);
    toolBar->addAction(QString::fromUtf8("印章测试"), this, &MainWindow::onStampProtectTest);
    toolBar->addAction(QString::fromUtf8("待确认中心"), this, &MainWindow::onPendingCenter);
    toolBar->addAction(QString::fromUtf8("批量处理"), this, &MainWindow::onBatchProcess);
}

void MainWindow::setupCentralWidget()
{
    QSplitter *mainSplitter = new QSplitter(Qt::Horizontal, this);

    // ==========================================
    // 左侧区域（目录树 + 缩略图列表）
    // ==========================================
    QWidget *leftWidget = new QWidget(this);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    m_folderTree = new QTreeWidget(leftWidget);
    m_folderTree->setHeaderHidden(true);
    m_folderTree->setIconSize(QSize(18, 18));

    // 1. 桌面
    QTreeWidgetItem *desktopItem = new QTreeWidgetItem(m_folderTree);
    desktopItem->setText(0, QString::fromUtf8("桌面"));
    desktopItem->setData(0, Qt::UserRole, QStandardPaths::writableLocation(QStandardPaths::DesktopLocation));

    // 2. 此电脑
    QTreeWidgetItem *computerItem = new QTreeWidgetItem(m_folderTree);
    computerItem->setText(0, QString::fromUtf8("此电脑"));
    computerItem->setData(0, Qt::UserRole, "COMPUTER");
    
    // 自动添加所有盘符
    QFileInfoList drives = QDir::drives();
    for (const QFileInfo &drive : drives) {
        QTreeWidgetItem *driveItem = new QTreeWidgetItem(computerItem);
        driveItem->setText(0, drive.filePath());
        driveItem->setData(0, Qt::UserRole, drive.filePath());
        // 添加一个占位节点，让前面显示展开的 + 号
        new QTreeWidgetItem(driveItem); 
    }

    // 3. 网络
    QTreeWidgetItem *networkItem = new QTreeWidgetItem(m_folderTree);
    networkItem->setText(0, QString::fromUtf8("网络"));
    networkItem->setData(0, Qt::UserRole, "NETWORK");

    // 展开“此电脑”
    computerItem->setExpanded(true);

    connect(m_folderTree, &QTreeWidget::itemDoubleClicked, this, &MainWindow::onFolderTreeDoubleClicked);
    connect(m_folderTree, &QTreeWidget::itemExpanded, this, &MainWindow::onFolderTreeExpanded);

    m_fileModel = new QStandardItemModel(this);
    
    m_fileList = new QListView(leftWidget);
    m_fileList->setModel(m_fileModel);
    m_fileList->setViewMode(QListView::IconMode); 
    m_fileList->setIconSize(QSize(100, 100));       
    m_fileList->setGridSize(QSize(120, 140));      
    m_fileList->setResizeMode(QListView::Adjust);
    m_fileList->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_fileList->setSpacing(5);
    connect(m_fileList, &QListView::doubleClicked, this, &MainWindow::onFileDoubleClicked);

    leftLayout->addWidget(m_folderTree);
    leftLayout->addWidget(m_fileList, 1);

    // ==========================================
    // 中间区域（预览区）
    // ==========================================
    m_previewScene = new QGraphicsScene(this);
    m_previewView = new QGraphicsView(m_previewScene, this);
    m_previewView->setRenderHint(QPainter::SmoothPixmapTransform);
    m_previewView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_previewView->setAlignment(Qt::AlignCenter);
    m_previewView->setStyleSheet(QStringLiteral("QGraphicsView { background-color: #dcdcdc; border: none; }"));

    // ==========================================
    // 右侧区域（参数分组）
    // ==========================================
    QScrollArea *paramScroll = new QScrollArea(this);
    paramScroll->setWidgetResizable(true);
    paramScroll->setMinimumWidth(320); 
    paramScroll->setStyleSheet(QStringLiteral("QScrollArea { border: none; }"));

    QWidget *paramWidget = new QWidget(paramScroll);
    paramWidget->setStyleSheet(QStringLiteral("QWidget { background-color: #f5f5f5; }"));
    QVBoxLayout *paramLayout = new QVBoxLayout(paramWidget);
    paramLayout->setSpacing(10);

    // 组 1：基础处理
    QGroupBox *basicGroup = new QGroupBox(QString::fromUtf8("基础图像处理"), paramWidget);
    QVBoxLayout *basicLayout = new QVBoxLayout(basicGroup);
    QPushButton *btnDenoise = new QPushButton(QString::fromUtf8("执行 污点去除"), basicGroup);
    QPushButton *btnBlackEdge = new QPushButton(QString::fromUtf8("执行 黑边去除"), basicGroup);
    QPushButton *btnDeskew = new QPushButton(QString::fromUtf8("执行 自动扶正"), basicGroup);
    basicLayout->addWidget(btnDenoise);
    basicLayout->addWidget(btnBlackEdge);
    basicLayout->addWidget(btnDeskew);
    connect(btnDenoise, &QPushButton::clicked, this, &MainWindow::onDenoise);
    connect(btnBlackEdge, &QPushButton::clicked, this, &MainWindow::onRemoveBlackEdge);
    connect(btnDeskew, &QPushButton::clicked, this, &MainWindow::onDeskew);

    // 组 2：增强处理
    QGroupBox *enhanceGroup = new QGroupBox(QString::fromUtf8("文字与底色增强"), paramWidget);
    QVBoxLayout *enhanceLayout = new QVBoxLayout(enhanceGroup);
    buildEnhancePanel(enhanceLayout); 

    // 组 3：智能分析
    QGroupBox *aiGroup = new QGroupBox(QString::fromUtf8("智能分析"), paramWidget);
    QVBoxLayout *aiLayout = new QVBoxLayout(aiGroup);
    QPushButton *btnErrPage = new QPushButton(QString::fromUtf8("处理 错误页码"), aiGroup);
    QPushButton *btnStamp = new QPushButton(QString::fromUtf8("印章保护测试"), aiGroup);
    QPushButton *btnPending = new QPushButton(QString::fromUtf8("打开待确认中心"), aiGroup);
    aiLayout->addWidget(btnErrPage);
    aiLayout->addWidget(btnStamp);
    aiLayout->addWidget(btnPending);
    connect(btnErrPage, &QPushButton::clicked, this, &MainWindow::onProcessErrPage);
    connect(btnStamp, &QPushButton::clicked, this, &MainWindow::onStampProtectTest);
    connect(btnPending, &QPushButton::clicked, this, &MainWindow::onPendingCenter);

    paramLayout->addWidget(basicGroup);
    paramLayout->addWidget(enhanceGroup);
    paramLayout->addWidget(aiGroup);
    paramLayout->addStretch(); 

    paramScroll->setWidget(paramWidget);

    mainSplitter->addWidget(leftWidget);
    mainSplitter->addWidget(m_previewView);
    mainSplitter->addWidget(paramScroll);
    
    mainSplitter->setStretchFactor(0, 2);
    mainSplitter->setStretchFactor(1, 4);
    mainSplitter->setStretchFactor(2, 2);
    mainSplitter->setSizes({300, 700, 340});

    setCentralWidget(mainSplitter);
}

// ★ 辅助：填充子节点（防止初始化时卡死）
void MainWindow::populateTreeItem(QTreeWidgetItem *item, const QString &path)
{
    if (item->childCount() > 1 || (item->childCount() == 1 && item->child(0)->data(0, Qt::UserRole).toString() != "PLACEHOLDER")) {
        return; // 已经加载过了
    }
    item->takeChildren(); // 删除占位符

    QDir dir(path);
    dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
    QFileInfoList list = dir.entryInfoList();
    for (const QFileInfo &info : list) {
        QTreeWidgetItem *child = new QTreeWidgetItem(item);
        child->setText(0, info.fileName());
        child->setData(0, Qt::UserRole, info.absoluteFilePath());
        // 判断是不是有子目录，如果有，放个占位符
        QDir childDir(info.absoluteFilePath());
        childDir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
        if (!childDir.entryList().isEmpty()) {
            new QTreeWidgetItem(child); 
        }
    }
}

void MainWindow::onFolderTreeExpanded(QTreeWidgetItem *item)
{
    if (!item) return;
    QString path = item->data(0, Qt::UserRole).toString();
    if (path.isEmpty() || path == "COMPUTER" || path == "NETWORK") return;
    populateTreeItem(item, path);
}

void MainWindow::onFolderTreeDoubleClicked(QTreeWidgetItem *item, int column)
{
    Q_UNUSED(column);
    if (!item) return;
    QString path = item->data(0, Qt::UserRole).toString();
    
    if (path.isEmpty() || path == "COMPUTER" || path == "NETWORK") return;

    m_currentFolder = path;
    statusBar()->showMessage(QString::fromUtf8("正在加载：%1 ...").arg(path));
    
    // 只加载当前目录的图片
    QStringList files;
    QDir dir(path);
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tif" << "*.tiff";
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files);
    for (const QFileInfo &fileInfo : fileList) {
        files.append(fileInfo.absoluteFilePath());
    }
    
    m_currentFiles = files;
    fillFileTable(files);
    statusBar()->showMessage(QString::fromUtf8("加载完成：%1 个文件").arg(files.size()));
}

void MainWindow::buildEnhancePanel(QVBoxLayout *paramLayout)
{
    QToolButton *toggleBtn = new QToolButton(this);
    toggleBtn->setText(QString::fromUtf8("浅色文字加深 (展开)"));
    toggleBtn->setCheckable(true);
    toggleBtn->setChecked(true);
    toggleBtn->setArrowType(Qt::DownArrow);
    toggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleBtn->setAutoRaise(true);

    QWidget *body = new QWidget(paramLayout->parentWidget());
    QGridLayout *grid = new QGridLayout(body);
    grid->setContentsMargins(5, 5, 5, 5);
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

    connect(m_enhanceStrengthSlider, &QSlider::valueChanged, m_enhanceStrengthSpin, &QSpinBox::setValue);
    connect(m_enhanceStrengthSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_enhanceStrengthSlider, &QSlider::setValue);
    connect(m_enhanceColorSatSlider, &QSlider::valueChanged, m_enhanceColorSatSpin, &QSpinBox::setValue);
    connect(m_enhanceColorSatSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_enhanceColorSatSlider, &QSlider::setValue);
    connect(m_enhanceDarkTargetSlider, &QSlider::valueChanged, m_enhanceDarkTargetSpin, &QSpinBox::setValue);
    connect(m_enhanceDarkTargetSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_enhanceDarkTargetSlider, &QSlider::setValue);
    connect(m_enhancePaperTargetSlider, &QSlider::valueChanged, m_enhancePaperTargetSpin, &QSpinBox::setValue);
    connect(m_enhancePaperTargetSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_enhancePaperTargetSlider, &QSlider::setValue);

    connect(toggleBtn, &QToolButton::toggled, this, [body, toggleBtn](bool on) {
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

void MainWindow::onEnhanceParamChanged() {}
void MainWindow::onEnhanceDebounceTimeout() {}

void MainWindow::setupStatusBar()
{
    statusBar()->showMessage(QString::fromUtf8("就绪"));
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    if (m_hasImage) fitPreviewToWindow();
}

void MainWindow::fitPreviewToWindow()
{
    if (!m_previewScene || !m_previewView) return;
    const QRectF rect = m_previewScene->itemsBoundingRect();
    if (rect.isEmpty()) return;
    m_previewView->resetTransform();
    m_previewView->fitInView(rect, Qt::KeepAspectRatio);
}

void MainWindow::onAddFolder()
{
    const QString folder = QFileDialog::getExistingDirectory(this, QString::fromUtf8("选择图片文件夹"), QString(), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (folder.isEmpty()) return;
    m_currentFolder = folder;
    const QStringList files = core::FileScanner::scanFolder(folder);
    m_currentFiles = files;
    fillFileTable(files);
}

void MainWindow::fillFileTable(const QStringList &files)
{
    m_fileModel->removeRows(0, m_fileModel->rowCount());
    for (const QString &path : files) {
        const QFileInfo info(path);
        QStandardItem *item = new QStandardItem(info.fileName());
        
        cv::Mat mat = cv::imread(path.toStdString());
        if (!mat.empty()) {
            cv::Mat thumb;
            cv::resize(mat, thumb, cv::Size(100, 100), 0, 0, cv::INTER_AREA);
            cv::cvtColor(thumb, thumb, cv::COLOR_BGR2RGB);
            QImage img(thumb.data, thumb.cols, thumb.rows, (int)thumb.step, QImage::Format_RGB888);
            item->setIcon(QIcon(QPixmap::fromImage(img)));
        }

        item->setData(path, Qt::UserRole);
        item->setToolTip(path);
        item->setEditable(false);
        m_fileModel->appendRow(item);
    }
}

void MainWindow::onFileDoubleClicked(const QModelIndex &index)
{
    if (!index.isValid()) return;
    const QString path = m_fileModel->data(index, Qt::UserRole).toString();
    if (path.isEmpty()) return;
    showImageOnPreview(path);
}

void MainWindow::onOpenImage()
{
    const QString path = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择图片"), QString(), QString::fromUtf8("图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"));
    if (path.isEmpty()) return;
    showImageOnPreview(path);
}

void MainWindow::onSaveAs()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    QString defaultName = QStringLiteral("processed.jpg");
    if (!m_currentImagePath.isEmpty()) {
        const QFileInfo info(m_currentImagePath);
        defaultName = info.completeBaseName() + QStringLiteral("_processed.jpg");
    }
    const QString savePath = QFileDialog::getSaveFileName(this, QString::fromUtf8("另存为"), defaultName, QString::fromUtf8("JPG 图片 (*.jpg *.jpeg)"));
    if (savePath.isEmpty()) return;

    image::JpegSaveOptions options;
    options.quality = 98;
    options.dpiX = 300.0;
    options.dpiY = 300.0;
    options.use444Sampling = true;
    image::JpegWriter::write(savePath, m_currentMat, options);
}

void MainWindow::onDeskew()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    const process::DeskewResult result = process::Deskew::autoDeskew(m_currentMat);
    if (result.ok) {
        m_currentMat = result.image;
        showMatOnPreview(m_currentMat);
    }
}

void MainWindow::onRemoveBlackEdge()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    process::BlackEdgeOptions options;
    options.fillWhite = true;
    const process::BlackEdgeResult result = process::BlackEdge::removeBlackEdge(m_currentMat, options);
    if (result.ok) {
        m_currentMat = result.image;
        showMatOnPreview(m_currentMat);
    }
}

void MainWindow::onDenoise()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    process::DenoiseOptions options;
    protect::StampProtectOptions protectOpt;
    const protect::StampProtectResult pr = protect::StampProtect::detect(m_currentMat, protectOpt);
    if (pr.ok) options.protectMask = pr.mask;
    const process::DenoiseResult result = process::Denoise::removeSpots(m_currentMat, options);
    if (result.ok) {
        m_currentMat = result.image;
        showMatOnPreview(m_currentMat);
    }
}

void MainWindow::onEnhance()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    process::EnhanceOptions options = currentEnhanceOptions();
    const process::EnhanceResult result = process::Enhance::enhanceText(m_currentMat, options);
    if (result.ok) {
        m_currentMat = result.image;
        showMatOnPreview(m_currentMat);
    }
}

void MainWindow::onBackground()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    process::BackgroundOptions options;
    protect::StampProtectOptions protectOpt;
    const protect::StampProtectResult pr = protect::StampProtect::detect(m_currentMat, protectOpt);
    if (pr.ok) options.protectMask = pr.mask;
    const process::BackgroundResult result = process::Background::whiten(m_currentMat, options);
    if (result.ok) {
        m_currentMat = result.image;
        showMatOnPreview(m_currentMat);
    }
}

void MainWindow::onOneClickProcess()
{
    if (!m_hasImage || m_currentMat.empty()) return;
    if (m_oneClickDeskewCheck && m_oneClickDeskewCheck->isChecked()) {
        const process::DeskewResult r = process::Deskew::autoDeskew(m_currentMat);
        if (r.ok) m_currentMat = r.image;
    }
    if (m_oneClickBlackEdgeCheck && m_oneClickBlackEdgeCheck->isChecked()) {
        const process::BlackEdgeResult r = process::BlackEdge::removeBlackEdge(m_currentMat, process::BlackEdgeOptions());
        if (r.ok) m_currentMat = r.image;
    }
    if (m_oneClickDenoiseCheck && m_oneClickDenoiseCheck->isChecked()) {
        const process::DenoiseResult r = process::Denoise::removeSpots(m_currentMat, process::DenoiseOptions());
        if (r.ok) m_currentMat = r.image;
    }
    if (m_oneClickEnhanceCheck && m_oneClickEnhanceCheck->isChecked()) {
        const process::EnhanceResult r = process::Enhance::enhanceText(m_currentMat, currentEnhanceOptions());
        if (r.ok) m_currentMat = r.image;
    }
    showMatOnPreview(m_currentMat);
}

void MainWindow::onDetectColorLine() {}
void MainWindow::onClearColorLine() {}
void MainWindow::onProcessErrPage()
{
    if (!m_hasImage || m_currentMat.empty() || m_currentImagePath.isEmpty()) return;
    process::ErrPageOptions opt;
    opt.enablePaddleOcr = ui::AppSettings::load().enablePaddleOcr;
    const process::ErrPageResult r = process::ErrPage::process(m_currentMat, m_currentImagePath, opt);
    if (r.ok) {
        m_currentMat = r.image;
        showMatOnPreview(m_currentMat);
    }
}
void MainWindow::onStampProtectTest() {}
void MainWindow::onBatchProcess()
{
    if (m_currentFiles.isEmpty()) return;
    if (!m_batchDialog) {
        m_batchDialog = new batch::BatchDialog(m_currentFiles, m_currentFolder, this);
        m_batchDialog->setAttribute(Qt::WA_DeleteOnClose);
        connect(m_batchDialog, &QObject::destroyed, this, [this]() { m_batchDialog = nullptr; });
        m_batchDialog->show();
    }
}
void MainWindow::onOpenSettings()
{
    if (!m_settingsDialog) {
        m_settingsDialog = new ui::SettingsDialog(this);
        m_settingsDialog->setAttribute(Qt::WA_DeleteOnClose);
        connect(m_settingsDialog, &QObject::destroyed, this, [this]() { m_settingsDialog = nullptr; });
        m_settingsDialog->show();
    }
}
void MainWindow::onOpenBackupManager() {}
void MainWindow::onOpenAnnotator() {}
void MainWindow::onBatchPreview(const cv::Mat &mat, bool before) {}
void MainWindow::onPendingCenter()
{
    analyze::PendingDialog dlg(this);
    dlg.exec();
    applyAcceptedYellowBlobs();
}
void MainWindow::applyAcceptedYellowBlobs() {}

void MainWindow::showMatOnPreview(const cv::Mat &mat)
{
    if (mat.empty()) return;
    cv::Mat rgb;
    if (mat.channels() == 3) cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4) cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    else cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);

    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());
    m_previewScene->clear();
    m_previewScene->addPixmap(pix);
    m_previewScene->setSceneRect(pix.rect());
    m_hasImage = true;
    QTimer::singleShot(0, this, [this]() { fitPreviewToWindow(); });
}

void MainWindow::showImageOnPreview(const QString &path)
{
    cv::Mat mat;
    image::ImageMeta meta;
    if (image::ImageIO::read(path, mat, meta)) {
        m_currentMat = mat;
        m_currentImagePath = path;
        m_enhancePreviewBase = mat.clone();
        showMatOnPreview(m_currentMat);
    }
}