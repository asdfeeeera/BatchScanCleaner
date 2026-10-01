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
    setWindowTitle(QStringLiteral("����ɨ��ͼƬ������ǿ���"));
    resize(1280, 800);

    setupMenuBar();
    setupToolBar();
    setupCentralWidget();
    setupStatusBar();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupMenuBar()
{
    QMenu *fileMenu = menuBar()->addMenu(QStringLiteral("�ļ�"));
    QAction *openImageAction = fileMenu->addAction(QStringLiteral("��ͼƬ..."));
    connect(openImageAction, &QAction::triggered, this, &MainWindow::onOpenImage);
    fileMenu->addAction(QStringLiteral("����ļ���"));
    fileMenu->addSeparator();
    fileMenu->addAction(QStringLiteral("�˳�"));

    QMenu *editMenu = menuBar()->addMenu(QStringLiteral("�༭"));
    editMenu->addAction(QStringLiteral("ȫѡ"));
    editMenu->addAction(QStringLiteral("��ѡ"));

    QMenu *viewMenu = menuBar()->addMenu(QStringLiteral("��ͼ"));
    viewMenu->addAction(QStringLiteral("��ʾ/�����ļ���"));
    viewMenu->addAction(QStringLiteral("��ʾ/����Ԥ����"));
    viewMenu->addAction(QStringLiteral("��ʾ/���ز�����"));

    QMenu *toolMenu = menuBar()->addMenu(QStringLiteral("����"));
    toolMenu->addAction(QStringLiteral("��ȷ������"));
    toolMenu->addAction(QStringLiteral("�������"));
    toolMenu->addAction(QStringLiteral("��־�뱨��"));
    toolMenu->addAction(QStringLiteral("���ݹ���"));
    toolMenu->addAction(QStringLiteral("����"));

    QMenu *helpMenu = menuBar()->addMenu(QStringLiteral("����"));
    helpMenu->addAction(QStringLiteral("ʹ���ֲ�"));
    helpMenu->addAction(QStringLiteral("��ݼ�"));
    helpMenu->addAction(QStringLiteral("����"));
}

void MainWindow::setupToolBar()
{
    QToolBar *toolBar = addToolBar(QStringLiteral("������"));
    toolBar->setMovable(false);

    QAction *openAction = toolBar->addAction(QStringLiteral("��ͼƬ"));
    connect(openAction, &QAction::triggered, this, &MainWindow::onOpenImage);

    toolBar->addAction(QStringLiteral("����ļ���"));
    toolBar->addAction(QStringLiteral("�Ƴ�"));
    toolBar->addSeparator();
    toolBar->addAction(QStringLiteral("�������"));
    toolBar->addAction(QStringLiteral("Ԥ��"));
    toolBar->addSeparator();
    toolBar->addAction(QStringLiteral("��ʼ"));
    toolBar->addAction(QStringLiteral("��ͣ"));
    toolBar->addAction(QStringLiteral("����"));
    toolBar->addAction(QStringLiteral("ȡ��"));
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
    paramLayout->addWidget(new QLabel(QStringLiteral("Ԥ��")));
    paramLayout->addWidget(new QLabel(QStringLiteral("�۵�ȥ��")));
    paramLayout->addWidget(new QLabel(QStringLiteral("�ڱ�ȥ��")));
    paramLayout->addWidget(new QLabel(QStringLiteral("�Զ�����")));
    paramLayout->addWidget(new QLabel(QStringLiteral("ǳɫ���ּ���")));
    paramLayout->addWidget(new QLabel(QStringLiteral("��ɫ����ϸ��")));
    paramLayout->addWidget(new QLabel(QStringLiteral("ǩ��ӡ�±���")));
    paramLayout->addWidget(new QLabel(QStringLiteral("��ɫ����")));
    paramLayout->addWidget(new QLabel(QStringLiteral("�������")));
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
    statusBar()->showMessage(QStringLiteral("����"));
}

void MainWindow::onOpenImage()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("ѡ��ͼƬ"),
        QString(),
        QStringLiteral("ͼƬ�ļ� (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"));

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
        QMessageBox::warning(this, QStringLiteral("����"),
                             QStringLiteral("�޷���ȡͼƬ��%1").arg(path));
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
        QStringLiteral("%1  |  %2 �� %3  |  %4 ͨ��")
            .arg(QFileInfo(path).fileName())
            .arg(meta.width)
            .arg(meta.height)
            .arg(meta.channels));
}