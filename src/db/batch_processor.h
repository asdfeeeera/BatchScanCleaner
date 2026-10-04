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

namespace batch {

// ============================================================
// 批处理选项
// ============================================================
struct BatchOptions
{
    QStringList inputFiles;

    // 主输出目录（兼容旧代码，等于 outputDirs[0]）
    QString outputDir;

    // 多存储：输出目录列表（第一个是主目录，其余是副本目录）
    QStringList outputDirs;

    // ★ 3.1 日志报告：报告文件输出目录
    QString reportDir;

    // 多机分片：把文件按 totalShards 台机器分片
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

    // ★ 3.1 日志报告：本次运行生成的报告文件路径（可能为空）
    QString reportPath;
};

// ============================================================
// 批处理引擎（独立线程）
// ============================================================
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
};

} // namespace batch

// ============================================================
// ★ 关键：让 BatchProgress / BatchResult 能用于跨线程队列信号
//   （Q_DECLARE_METATYPE 必须写在 namespace batch 之外，参数带完整命名空间）
// ============================================================
Q_DECLARE_METATYPE(batch::BatchProgress)
Q_DECLARE_METATYPE(batch::BatchResult)