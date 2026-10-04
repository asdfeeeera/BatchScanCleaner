#pragma once

#include "batch_step.h"

#include <QThread>
#include <QString>
#include <QStringList>
#include <QList>
#include <QMutex>
#include <QMetaType>

#include <opencv2/core.hpp>

#include "deskew.h"
#include "blackedge.h"
#include "denoise.h"
#include "enhance.h"
#include "errpage.h"
#include "background.h"
#include "colorline.h"
#include "image_io.h"
#include "jpeg_writer.h"

#include "../protect/stamp_protect.h"

// ★ cv::Mat 跨线程信号需要 Q_DECLARE_METATYPE
Q_DECLARE_METATYPE(cv::Mat)

namespace batch {

// ============================================================
// 批处理选项
// ============================================================
struct BatchOptions
{
    QStringList inputFiles;
    QString outputDir;
    QStringList outputDirs;
    QString reportDir;

    // ★ 3.3 备份：处理前把原图复制到 backupDir/YYYY-MM-DD/ 下
    QString backupDir;
    bool    backupEnabled = false;

    int shardIndex = 0;
    int totalShards = 1;

    QList<StepItem> steps;

    process::BlackEdgeOptions   blackEdgeOpt;
    process::DenoiseOptions     denoiseOpt;
    process::EnhanceOptions     enhanceOpt;
    process::ErrPageOptions     errPageOpt;
    process::BackgroundOptions  backgroundOpt;
    process::ColorLineOptions   colorLineOpt;
    protect::StampProtectOptions stampOpt;

    QString outputSuffix = QString();
    image::JpegSaveOptions jpegOpt;
};

struct BatchProgress
{
    int current = 0;
    int total = 0;
    QString currentFile;
    int succeeded = 0;
    int failed = 0;
    double elapsedSeconds = 0;
    double remainingSeconds = 0;
};

struct BatchResult
{
    bool ok = false;
    bool cancelled = false;
    int total = 0;
    int succeeded = 0;
    int failed = 0;
    QStringList failedFiles;
    double totalSeconds = 0;
    QString reportPath;

    // ★ 3.3 备份统计
    int backedUp     = 0;
    int backupFailed = 0;
};

class BatchProcessor : public QThread
{
    Q_OBJECT

public:
    explicit BatchProcessor(QObject *parent = nullptr);
    ~BatchProcessor() override;

    void startBatch(const BatchOptions &options);
    void pause();
    void resume();
    void cancel();

    bool isRunning() const { return m_running; }
    bool isPaused() const { return m_paused; }
    bool isCancelled() const { return m_cancelled; }

signals:
    void progressChanged(const BatchProgress &progress);
    void fileStarted(const QString &filePath);
    void fileFinished(const QString &filePath, bool success);
    void finished(const BatchResult &result);
    void previewImageReady(const cv::Mat &mat, bool before);

protected:
    void run() override;

private:
    bool processOneFile(const QString &inputPath, QString &outError);

    bool runDeskew(cv::Mat &img);
    bool runBlackEdge(cv::Mat &img);
    bool runDenoise(cv::Mat &img);
    bool runEnhance(cv::Mat &img);
    bool runErrPage(cv::Mat &img, const QString &sourcePath);
    bool runBackground(cv::Mat &img);
    bool runColorLine(cv::Mat &img);

    QString makeOutputPath(const QString &inputPath) const;
    cv::Mat makePreview(const cv::Mat &img, int maxSize = 1000) const;
    QString writeReport(bool cancelled) const;

    // ★ 3.3 备份单张（处理前调用）
    bool backupOneFile(const QString &inputPath, QString &outError) const;

    BatchOptions m_options;
    volatile bool m_running = false;
    volatile bool m_paused = false;
    volatile bool m_cancelled = false;

    int m_index = 0;
    int m_succeeded = 0;
    int m_failed = 0;
    QStringList m_failedFiles;

    qint64 m_startMs = 0;
    QList<qint64> m_fileDurations;

    int m_processedTotal = 0;
    int m_globalTotal = 0;

    // ★ 3.3 备份统计
    int m_backedUp     = 0;
    int m_backupFailed = 0;
};

} // namespace batch

// ★ 让 BatchProgress / BatchResult 能用于跨线程队列信号
Q_DECLARE_METATYPE(batch::BatchProgress)
Q_DECLARE_METATYPE(batch::BatchResult)