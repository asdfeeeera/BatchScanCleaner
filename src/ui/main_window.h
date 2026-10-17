#pragma once

#include <QMainWindow>
#include <QStringList>
#include <QFileSystemModel>
#include <QListView>
#include <opencv2/core.hpp>    // ★ 加上这一行，解决 cv::Mat 不识别的问题

class QTreeView;
class QTableView;
class QStandardItemModel;
class QGraphicsView;
class QGraphicsScene;
class QLabel;
class QTimer;
class QSlider;
class QSpinBox;
class QCheckBox;
class QToolButton;
class QModelIndex;

namespace batch { class BatchDialog; }
namespace ui { class SettingsDialog; }
namespace backup { class BackupDialog; }
namespace annotate { class AnnotationDialog; }
namespace process { struct EnhanceOptions; }

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
    void onSaveAs();
    void onDeskew();
    void onRemoveBlackEdge();
    void onDenoise();
    void onEnhance();
    void onEnhanceParamChanged();
    void onEnhanceDebounceTimeout();
    void onDetectColorLine();
    void onClearColorLine();
    void onProcessErrPage();
    void onStampProtectTest();
    void onBatchProcess();
    void onOpenSettings();
    void onOpenBackupManager();
    void onOpenAnnotator();
    void onBatchPreview(const cv::Mat &mat, bool before);
    void onPendingCenter();
    void onBackground();
    void onOneClickProcess();
    void onFolderTreeClicked(const QModelIndex &index);

private:
    void setupMenuBar();
    void setupToolBar();
    void setupCentralWidget();
    void setupStatusBar();
    void fitPreviewToWindow();
    void fillFileTable(const QStringList &files);
    void showMatOnPreview(const cv::Mat &mat);
    void showImageOnPreview(const QString &path);
    void applyAcceptedYellowBlobs();
    
    void buildEnhancePanel(QVBoxLayout *paramLayout);
    process::EnhanceOptions currentEnhanceOptions() const;

    // 文件与图像状态
    QStringList m_currentFiles;
    QString m_currentFolder;
    QString m_currentImagePath;
    bool m_hasImage = false;
    
    cv::Mat m_originalMat;
    cv::Mat m_currentMat;
    cv::Mat m_enhancePreviewBase;
    cv::Mat m_colorLineSource;

    // 核心控件
    QTreeView *m_folderTree = nullptr;
    QFileSystemModel *m_dirModel = nullptr;
    
    QStandardItemModel *m_fileModel = nullptr;
    QTableView *m_fileTable = nullptr; 
    QListView *m_fileList = nullptr;   
    
    QGraphicsView *m_previewView = nullptr;
    QGraphicsScene *m_previewScene = nullptr;

    // 增强参数控件
    QSlider *m_enhanceStrengthSlider = nullptr;
    QSpinBox *m_enhanceStrengthSpin = nullptr;
    QSlider *m_enhanceColorSatSlider = nullptr;
    QSpinBox *m_enhanceColorSatSpin = nullptr;
    QSlider *m_enhanceDarkTargetSlider = nullptr;
    QSpinBox *m_enhanceDarkTargetSpin = nullptr;
    QSlider *m_enhancePaperTargetSlider = nullptr;
    QSpinBox *m_enhancePaperTargetSpin = nullptr;

    // 一键处理参数
    QCheckBox *m_oneClickDeskewCheck = nullptr;
    QCheckBox *m_oneClickBlackEdgeCheck = nullptr;
    QCheckBox *m_oneClickDenoiseCheck = nullptr;
    QCheckBox *m_oneClickEnhanceCheck = nullptr;
    QCheckBox *m_oneClickErrPageCheck = nullptr;
    QCheckBox *m_oneClickBackgroundCheck = nullptr;
    QCheckBox *m_oneClickColorLineCheck = nullptr;

    // 其他窗口
    batch::BatchDialog *m_batchDialog = nullptr;
    ui::SettingsDialog *m_settingsDialog = nullptr;
    backup::BackupDialog *m_backupDialog = nullptr;
    annotate::AnnotationDialog *m_annotateDialog = nullptr;

    QTimer *m_enhanceDebounceTimer = nullptr;
};