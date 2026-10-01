#pragma execution_character_set("utf-8")
#include "main_window.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QSplitter>
#include <QTreeView>
#include <QTableView>
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
#include <QDebug>

#include "image_io.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("批量扫描图片净化增强软件"));
    resize(1280, 800);

    setupMenuBar();
    setupToolBar();
    setupCentralWidget();
    setupStatusBar();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupMenuBar()
{
    QMenu *fileMenu = menuBar()->addMenu(QStringLiteral("文件"));
    QAction *openImageAction = fileMenu->addAction(QStringLiteral("打开图片..."));
    connect(openImageAction, &QAction::triggered, this, &MainWindow::onOpenImage);
    fileMenu->addAction(QStringLiteral("添加文件夹"));
    fileMenu->addSeparator();
    fileMenu->addAction(QStringLiteral("退出"));

    QMenu *editMenu = menuBar()->addMenu(QStringLiteral("编辑"));
    editMenu->addAction(QStringLiteral("全选"));
    editMenu->addAction(QStringLiteral("反选"));

    QMenu *viewMenu = menuBar()->addMenu(QStringLiteral("视图"));
    viewMenu->addAction(QStringLiteral("显示/隐藏文件区"));
    viewMenu->addAction(QStringLiteral("显示/隐藏预览区"));
    viewMenu->addAction(QStringLiteral("显示/隐藏参数区"));

    QMenu *toolMenu = menuBar()->addMenu(QStringLiteral("工具"));
    toolMenu->addAction(QStringLiteral("待确认中心"));
    toolMenu->addAction(QStringLiteral("任务队列"));
    toolMenu->addAction(QStringLiteral("日志与报告"));
    toolMenu->addAction(QStringLiteral("备份管理"));
    toolMenu->addAction(QStringLiteral("设置"));

    QMenu *helpMenu = menuBar()->addMenu(QStringLiteral("帮助"));
    helpMenu->addAction(QStringLiteral("使用手册"));
    helpMenu->addAction(QStringLiteral("快捷键"));
    helpMenu->addAction(QStringLiteral("关于"));
}

void MainWindow::setupToolBar()
{
    QToolBar *toolBar = addToolBar(QStringLiteral("工具栏"));
    toolBar->setMovable(false);

    QAction *openAction = toolBar->addAction(QStringLiteral("打开图片"));
    connect(openAction, &QAction::triggered, this, &MainWindow::onOpenImage);

    toolBar->addAction(QStringLiteral("添加文件夹"));
    toolBar->addAction(QStringLiteral("移除"));
    toolBar->addSeparator();
    toolBar->addAction(QStringLiteral("输出设置"));
    toolBar->addAction(QStringLiteral("预设"));
    toolBar->addSeparator();
    toolBar->addAction(QStringLiteral("开始"));
    toolBar->addAction(QStringLiteral("暂停"));
    toolBar->addAction(QStringLiteral("继续"));
    toolBar->addAction(QStringLiteral("取消"));
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

    m_fileTable = new QTableView(leftWidget);
    leftLayout->addWidget(m_fileTable);

    m_previewScene = new QGraphicsScene(this);
    m_previewView = new QGraphicsView(m_previewScene, this);
    m_previewView->setRenderHint(QPainter::SmoothPixmapTransform);
    m_previewView->setDragMode(QGraphicsView::ScrollHandDrag);

    QScrollArea *paramScroll = new QScrollArea(this);
    QWidget *paramWidget = new QWidget(paramScroll);
    QVBoxLayout *paramLayout = new QVBoxLayout(paramWidget);
    paramLayout->addWidget(new QLabel(QStringLiteral("预设")));
    paramLayout->addWidget(new QLabel(QStringLiteral("污点去除")));
    paramLayout->addWidget(new QLabel(QStringLiteral("黑边去除")));
    paramLayout->addWidget(new QLabel(QStringLiteral("自动扶正")));
    paramLayout->addWidget(new QLabel(QStringLiteral("浅色文字加深")));
    paramLayout->addWidget(new QLabel(QStringLiteral("彩色故障细线")));
    paramLayout->addWidget(new QLabel(QStringLiteral("签名印章保护")));
    paramLayout->addWidget(new QLabel(QStringLiteral("底色处理")));
    paramLayout->addWidget(new QLabel(QStringLiteral("输出设置")));
    paramLayout->addStretch();
    paramScroll->setWidget(paramWidget);
    paramScroll->setWidgetResizable(true);

    mainSplitter->addWidget(leftWidget);
    mainSplitter->addWidget(m_previewView);
    mainSplitter->addWidget(paramScroll);
    mainSplitter->setStretchFactor(0, 1);
    mainSplitter->setStretchFactor(1, 3);
    mainSplitter->setStretchFactor(2, 1);

    setCentralWidget(mainSplitter);
}

void MainWindow::setupStatusBar()
{
    statusBar()->showMessage(QStringLiteral("就绪"));
}

void MainWindow::onOpenImage()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("选择图片"),
        QString(),
        QStringLiteral("图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"));

    if (path.isEmpty()) {
        return;
    }

    showImageOnPreview(path);
}

void MainWindow::showImageOnPreview(const QString &path)
{
    cv::Mat mat;
    image::ImageMeta meta;
    if (!image::ImageIO::read(path, mat, meta)) {
        QMessageBox::warning(this, QStringLiteral("错误"),
                             QStringLiteral("无法读取图片：%1").arg(path));
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
    m_previewView->fitInView(pix.rect(), Qt::KeepAspectRatio);

    statusBar()->showMessage(
        QStringLiteral("%1  |  %2 × %3  |  %4 通道")
            .arg(QFileInfo(path).fileName())
            .arg(meta.width)
            .arg(meta.height)
            .arg(meta.channels));
}