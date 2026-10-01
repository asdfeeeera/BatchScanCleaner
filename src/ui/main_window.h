#pragma once

#include <QMainWindow>

class QGraphicsScene;
class QGraphicsView;
class QTreeView;
class QTableView;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void onOpenImage();

private:
    void setupMenuBar();
    void setupToolBar();
    void setupCentralWidget();
    void setupStatusBar();
    void showImageOnPreview(const QString &path);
    void fitPreviewToWindow();

    QTreeView *m_folderTree = nullptr;
    QTableView *m_fileTable = nullptr;
    QGraphicsView *m_previewView = nullptr;
    QGraphicsScene *m_previewScene = nullptr;
    bool m_hasImage = false;
};