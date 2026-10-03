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

    // ★ 文字加深参数
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

    // ★ 增强参数面板
    void buildEnhancePanel(QVBoxLayout *paramLayout);
    process::EnhanceOptions currentEnhanceOptions() const;

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

    // ★ 文字加深参数控件
    QSlider *m_enhanceStrengthSlider = nullptr;
    QSpinBox *m_enhanceStrengthSpin = nullptr;

    QSlider *m_enhanceColorSatSlider = nullptr;
    QSpinBox *m_enhanceColorSatSpin = nullptr;

    QSlider *m_enhanceDarkTargetSlider = nullptr;
    QSpinBox *m_enhanceDarkTargetSpin = nullptr;

    QSlider *m_enhancePaperTargetSlider = nullptr;
    QSpinBox *m_enhancePaperTargetSpin = nullptr;

    // ★ 实时预览防抖
    QTimer *m_enhanceDebounceTimer = nullptr;
    cv::Mat m_enhancePreviewBase;   // 实时预览的原图（首次打开图时保存）
};