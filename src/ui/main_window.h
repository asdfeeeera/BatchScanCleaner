#pragma once

#include <QMainWindow>
#include <QStringList>
#include <opencv2/core.hpp>

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
    void onSaveAs();
    void onDeskew();
    void onRemoveBlackEdge();
    void onDenoise();
    void onEnhance();
    void onOneClickProcess();
    void onDetectColorLine();
    void onClearColorLine();
    void onProcessErrPage();
    void onStampProtectTest();
    void onPendingCenter();      // ★ 新增

private:
    void setupMenuBar();
    void setupToolBar();
    void setupCentralWidget();
    void setupStatusBar();
    void showImageOnPreview(const QString &path);
    void showMatOnPreview(const cv::Mat &mat);
    void fitPreviewToWindow();
    void fillFileTable(const QStringList &files);

    QTreeView *m_folderTree = nullptr;
    QTableView *m_fileTable = nullptr;
    QStandardItemModel *m_fileModel = nullptr;
    QGraphicsView *m_previewView = nullptr;
    QGraphicsScene *m_previewScene = nullptr;
    bool m_hasImage = false;
    QStringList m_currentFiles;
    QString m_currentImagePath;
    cv::Mat m_currentMat;
    cv::Mat m_originalMat;

    // 彩色细线：检测时的原图（供清除用）
    cv::Mat m_colorLineSource;
};