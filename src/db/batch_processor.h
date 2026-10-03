#pragma once

#include "batch_step.h"

#include <QObject>
#include <QString>
#include <QStringList>
#include <QList>

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
// 批处理引擎
// ============================================================
class BatchProcessor : public QObject
{
    Q_OBJECT

public:
    explicit BatchProcessor(QObject *parent = nullptr);
    ~BatchProcessor() override;

    void start(const BatchOptions &options);
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

    // ★ 预览图就绪：before=true 表示处理前，false 表示处理后
    void previewImageReady(const cv::Mat &mat, bool before);

private slots:
    void processNext();

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

    // ★ 缩放为预览用（最长边 <= maxSize）
    cv::Mat makePreview(const cv::Mat &img, int maxSize = 1000) const;

    BatchOptions m_options;
    bool m_running = false;
    bool m_paused = false;
    bool m_cancelled = false;

    int m_index = 0;
    int m_succeeded = 0;
    int m_failed = 0;
    QStringList m_failedFiles;

    qint64 m_startMs = 0;
    QList<qint64> m_fileDurations;
};

} // namespace batch