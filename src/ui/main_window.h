#pragma once

#include <QMainWindow>
#include <QStringList>
#include <opencv2/core.hpp>

#include "enhance.h"

class QGraphicsScene;
class QGraphicsView;
class QTreeView;
class QTableView;
class QStandardItemModel;
class QVBoxLayout;
class QSlider;
class QSpinBox;
class QTimer;
class QCheckBox;

namespace batch { class BatchDialog; }

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
    void onPendingCenter();
    void onBackground();
    void onBatchProcess();

    void onBatchPreview(const cv::Mat &mat, bool before);

    void onEnhanceParamChanged();
    void onEnhanceDebounceTimeout();

private:
    void setupMenuBar();
    void setupToolBar();
    void setupCentralWidget();
    void setupStatusBar();
    void showImageOnPreview(const QString &path);
    void showMatOnPreview(const cv::Mat &mat);
    void fitPreviewToWindow();
    void fillFileTable(const QStringList &files);

    void buildEnhancePanel(QVBoxLayout *paramLayout);
    process::EnhanceOptions currentEnhanceOptions() const;

    void applyAcceptedYellowBlobs();

    QTreeView *m_folderTree = nullptr;
    QTableView *m_fileTable = nullptr;
    QStandardItemModel *m_fileModel = nullptr;
    QGraphicsView *m_previewView = nullptr;
    QGraphicsScene *m_previewScene = nullptr;
    bool m_hasImage = false;
    QStringList m_currentFiles;
    QString m_currentImagePath;
    QString m_currentFolder;
    cv::Mat m_currentMat;
    cv::Mat m_originalMat;

    cv::Mat m_colorLineSource;

    QSlider *m_enhanceStrengthSlider = nullptr;
    QSpinBox *m_enhanceStrengthSpin = nullptr;

    QSlider *m_enhanceColorSatSlider = nullptr;
    QSpinBox *m_enhanceColorSatSpin = nullptr;

    QSlider *m_enhanceDarkTargetSlider = nullptr;
    QSpinBox *m_enhanceDarkTargetSpin = nullptr;

    QSlider *m_enhancePaperTargetSlider = nullptr;
    QSpinBox *m_enhancePaperTargetSpin = nullptr;

    QTimer *m_enhanceDebounceTimer = nullptr;
    cv::Mat m_enhancePreviewBase;

    batch::BatchDialog *m_batchDialog = nullptr;

    // ★ 一键处理下拉菜单的勾选框
    QCheckBox *m_oneClickDeskewCheck = nullptr;
    QCheckBox *m_oneClickBlackEdgeCheck = nullptr;
    QCheckBox *m_oneClickErrPageCheck = nullptr;
    QCheckBox *m_oneClickDenoiseCheck = nullptr;
    QCheckBox *m_oneClickEnhanceCheck = nullptr;
    QCheckBox *m_oneClickBackgroundCheck = nullptr;
    QCheckBox *m_oneClickColorLineCheck = nullptr;
};