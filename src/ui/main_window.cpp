#include "main_window.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QSplitter>
#include <QTreeView>
#include <QTableView>
#include <QGraphicsView>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QWidget>
#include <QAction>

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
    fileMenu->addAction(QStringLiteral("添加文件夹"));
    fileMenu->addAction(QStringLiteral("移除文件夹"));
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

    // 左侧文件区
    QWidget *leftWidget = new QWidget(this);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    QTreeView *folderTree = new QTreeView(leftWidget);
    folderTree->setHeaderHidden(true);
    leftLayout->addWidget(folderTree);

    QTableView *fileTable = new QTableView(leftWidget);
    leftLayout->addWidget(fileTable);

    // 中间预览区
    QGraphicsView *previewView = new QGraphicsView(this);

    // 右侧参数区
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
    mainSplitter->addWidget(previewView);
    mainSplitter->addWidget(paramScroll);
    mainSplitter->setStretchFactor(0, 1);
    mainSplitter->setStretchFactor(1, 2);
    mainSplitter->setStretchFactor(2, 1);

    setCentralWidget(mainSplitter);
}

void MainWindow::setupStatusBar()
{
    statusBar()->showMessage(QStringLiteral("就绪"));
}