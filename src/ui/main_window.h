#pragma once

#include <QMainWindow>
#include <QStringList>

class QGraphicsScene;
class QGraphicsView;
class QTreeView;
class QTableView;
class QStandardItemModel;

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
    void onAddFolder();
    void onFileDoubleClicked(const QModelIndex &index);

private:
    void setupMenuBar();
    void setupToolBar();
    void setupCentralWidget();
    void setupStatusBar();
    void showImageOnPreview(const QString &path);
    void fitPreviewToWindow();
    void fillFileTable(const QStringList &files);

    QTreeView *m_folderTree = nullptr;
    QTableView *m_fileTable = nullptr;
    QStandardItemModel *m_fileModel = nullptr;
    QGraphicsView *m_previewView = nullptr;
    QGraphicsScene *m_previewScene = nullptr;
    bool m_hasImage = false;
    QStringList m_currentFiles;
};