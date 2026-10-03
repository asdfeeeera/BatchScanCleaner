#pragma once

#include "batch_step.h"
#include "batch_processor.h"

#include <QDialog>
#include <QStringList>

class QLabel;
class QLineEdit;
class QPushButton;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QCheckBox;
class QPlainTextEdit;
class QGroupBox;

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

    // 引擎信号
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

    // 输入
    QStringList m_inputFiles;
    QString m_sourceDir;

    // UI 控件
    QLabel *m_inputLabel = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QPushButton *m_outputBtn = nullptr;
    QCheckBox *m_recurseCheck = nullptr;

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

    // 步骤列表（与 UI 同步）
    QList<StepItem> m_steps;

    // 引擎
    BatchProcessor *m_processor = nullptr;
};

} // namespace batch