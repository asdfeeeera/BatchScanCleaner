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
    QStringList inputFiles;    // 待处理的图片列表
    QString outputDir;         // 输出目录
    QList<StepItem> steps;     // 勾选的步骤（已排序）

    // 各步骤参数
    process::DeskewOptions      deskewOpt;
    process::BlackEdgeOptions   blackEdgeOpt;
    process::DenoiseOptions     denoiseOpt;
    process::EnhanceOptions     enhanceOpt;
    process::ErrPageOptions     errPageOpt;
    process::BackgroundOptions  backgroundOpt;
    process::ColorLineOptions   colorLineOpt;
    protect::StampProtectOptions stampOpt;

    // 输出
    QString outputSuffix = QString();   // 空 = 保持原名
    image::JpegSaveOptions jpegOpt;     // Jpeg 保存参数
};

// ============================================================
// 批处理进度
// ============================================================
struct BatchProgress
{
    int current = 0;           // 当前第几张（从 1 开始）
    int total = 0;             // 总数
    QString currentFile;       // 当前文件名
    int succeeded = 0;         // 成功数
    int failed = 0;            // 失败数
    double elapsedSeconds = 0; // 已用时间
    double remainingSeconds = 0; // 预估剩余时间
};

// ============================================================
// 批处理最终结果
// ============================================================
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
//   在主线程运行，每处理一张图让出控制权（避免 UI 卡顿）
// ============================================================
class BatchProcessor : public QObject
{
    Q_OBJECT

public:
    explicit BatchProcessor(QObject *parent = nullptr);
    ~BatchProcessor() override;

    // 开始处理（异步；通过信号反馈进度和结果）
    void start(const BatchOptions &options);

    // 控制
    void pause();
    void resume();
    void cancel();

    // 状态查询
    bool isRunning() const { return m_running; }
    bool isPaused() const { return m_paused; }
    bool isCancelled() const { return m_cancelled; }

signals:
    // 进度变化
    void progressChanged(const BatchProgress &progress);
    // 单张开始
    void fileStarted(const QString &filePath);
    // 单张完成（success = 是否成功）
    void fileFinished(const QString &filePath, bool success);
    // 全部完成
    void finished(const BatchResult &result);

private slots:
    void processNext();

private:
    // 处理单张图（返回 true 表示成功）
    bool processOneFile(const QString &inputPath, QString &outError);

    // 执行单个步骤
    bool runDeskew(cv::Mat &img);
    bool runBlackEdge(cv::Mat &img);
    bool runDenoise(cv::Mat &img);
    bool runEnhance(cv::Mat &img);
    bool runErrPage(cv::Mat &img, const QString &sourcePath);
    bool runBackground(cv::Mat &img);
    bool runColorLine(cv::Mat &img);

    // 生成输出路径
    QString makeOutputPath(const QString &inputPath) const;

    BatchOptions m_options;
    bool m_running = false;
    bool m_paused = false;
    bool m_cancelled = false;

    int m_index = 0;
    int m_succeeded = 0;
    int m_failed = 0;
    QStringList m_failedFiles;

    qint64 m_startMs = 0;
    QList<qint64> m_fileDurations;   // 每张图耗时（毫秒）
};

} // namespace batch