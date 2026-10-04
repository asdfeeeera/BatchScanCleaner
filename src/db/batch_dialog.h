#pragma once

#include "batch_step.h"
#include "batch_processor.h"

#include <QDialog>
#include <QStringList>
#include <opencv2/core.hpp>

class QLabel;
class QLineEdit;
class QPushButton;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QCheckBox;
class QPlainTextEdit;
class QGroupBox;
class QSpinBox;

namespace batch {

// ============================================================
// 批量处理对话框
// ============================================================
class BatchDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BatchDialog(const QStringList &inputFiles,
                         const QString &sourceDir,
                         QWidget *parent = nullptr);
    ~BatchDialog() override;

    // 父窗口调用：把对话框移到屏幕右上角
    void moveToTopRight();

signals:
    // ★ 转发处理器的预览信号给 MainWindow
    void previewImageReady(const cv::Mat &mat, bool before);

private slots:
    void onSelectOutputDir();
    void onToggleRecurse();
    void onStepItemChanged(QListWidgetItem *item);
    void onMoveStepUp();
    void onMoveStepDown();
    void onStart();
    void onPause();
    void onResume();
    void onCancel();

    // ★ 新增：多存储目录的添加/删除
    void onAddExtraDir();
    void onRemoveExtraDir();

    void onProgress(const BatchProgress &progress);
    void onFileStarted(const QString &path);
    void onFileFinished(const QString &path, bool success);
    void onFinished(const BatchResult &result);

private:
    void setupUi();
    void rebuildStepList();
    void updateButtonsState();
    void appendLog(const QString &line);

    BatchOptions buildOptions() const;

    QStringList m_inputFiles;
    QString m_sourceDir;

    QLabel *m_inputLabel = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QPushButton *m_outputBtn = nullptr;
    QCheckBox *m_recurseCheck = nullptr;

    // ★ 新增：多存储目录（副本目录，可添加多个）
    QGroupBox    *m_extraDirsGroup = nullptr;
    QListWidget  *m_extraDirsList  = nullptr;
    QPushButton  *m_addExtraDirBtn = nullptr;
    QPushButton  *m_removeExtraDirBtn = nullptr;

    // ★ 新增：多机分片设置
    QGroupBox *m_shardGroup = nullptr;
    QSpinBox  *m_shardIndexSpin = nullptr;   // 本机编号（0 开始）
    QSpinBox  *m_totalShardsSpin = nullptr;  // 总机器数（>=1）

    QGroupBox *m_stepGroup = nullptr;
    QListWidget *m_stepList = nullptr;
    QPushButton *m_upBtn = nullptr;
    QPushButton *m_downBtn = nullptr;

    QProgressBar *m_progressBar = nullptr;
    QLabel *m_statusLabel = nullptr;
    QLabel *m_remainLabel = nullptr;

    QPlainTextEdit *m_logEdit = nullptr;

    QPushButton *m_startBtn = nullptr;
    QPushButton *m_pauseBtn = nullptr;
    QPushButton *m_resumeBtn = nullptr;
    QPushButton *m_cancelBtn = nullptr;
    QPushButton *m_closeBtn = nullptr;

    QList<StepItem> m_steps;

    BatchProcessor *m_processor = nullptr;
};

} // namespace batch