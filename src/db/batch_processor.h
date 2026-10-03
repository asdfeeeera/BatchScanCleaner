#pragma once

#include "batch_step.h"

#include <QThread>
#include <QString>
#include <QStringList>
#include <QList>
#include <QMutex>

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
    QString outputDir;
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

    // 开始处理（会启动线程）
    void startBatch(const BatchOptions &options);

    // 控制（线程安全）
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
    // QThread 入口
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
};

} // namespace batch