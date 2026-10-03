#include "batch_processor.h"

#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QDebug>

#include <opencv2/imgproc.hpp>

namespace batch {

BatchProcessor::BatchProcessor(QObject *parent)
    : QThread(parent)
{
}

BatchProcessor::~BatchProcessor()
{
    cancel();
    wait(5000);
}

// ============================================================
// 开始（启动线程）
// ============================================================
void BatchProcessor::startBatch(const BatchOptions &options)
{
    if (m_running) return;

    m_options = options;
    m_running = true;
    m_paused = false;
    m_cancelled = false;

    m_index = 0;
    m_succeeded = 0;
    m_failed = 0;
    m_failedFiles.clear();
    m_fileDurations.clear();

    m_startMs = QDateTime::currentMSecsSinceEpoch();

    QDir().mkpath(m_options.outputDir);

    // 启动线程（会调用 run()）
    QThread::start();
}

// ============================================================
// 控制（线程安全）
// ============================================================
void BatchProcessor::pause()
{
    if (!m_running) return;
    m_paused = true;
}

void BatchProcessor::resume()
{
    if (!m_running) return;
    m_paused = false;
}

void BatchProcessor::cancel()
{
    if (!m_running) return;
    m_cancelled = true;
}

// ============================================================
// QThread 入口
// ============================================================
void BatchProcessor::run()
{
    const int total = m_options.inputFiles.size();

    for (m_index = 0; m_index < total; ++m_index) {

        // 取消
        if (m_cancelled) {
            m_running = false;
            BatchResult result;
            result.ok = true;
            result.cancelled = true;
            result.total = total;
            result.succeeded = m_succeeded;
            result.failed = m_failed;
            result.failedFiles = m_failedFiles;
            result.totalSeconds =
                (QDateTime::currentMSecsSinceEpoch() - m_startMs) / 1000.0;
            emit finished(result);
            return;
        }

        // 暂停：循环等待
        while (m_paused && !m_cancelled) {
            QThread::msleep(100);
        }
        if (m_cancelled) {
            m_running = false;
            BatchResult result;
            result.ok = true;
            result.cancelled = true;
            result.total = total;
            result.succeeded = m_succeeded;
            result.failed = m_failed;
            result.failedFiles = m_failedFiles;
            result.totalSeconds =
                (QDateTime::currentMSecsSinceEpoch() - m_startMs) / 1000.0;
            emit finished(result);
            return;
        }

        const QString inputPath = m_options.inputFiles[m_index];
        const QFileInfo fi(inputPath);

        emit fileStarted(inputPath);

        // 读取原图，发预览
        {
            cv::Mat img;
            image::ImageMeta meta;
            if (image::ImageIO::read(inputPath, img, meta)) {
                emit previewImageReady(makePreview(img), true);
            }
        }

        const qint64 t0 = QDateTime::currentMSecsSinceEpoch();
        QString errMsg;
        const bool ok = processOneFile(inputPath, errMsg);
        const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
        const qint64 duration = t1 - t0;

        m_fileDurations.append(duration);

        if (ok) {
            ++m_succeeded;
        } else {
            ++m_failed;
            m_failedFiles.append(inputPath);
        }

        emit fileFinished(inputPath, ok);

        // 结果预览
        {
            cv::Mat preview;
            if (ok) {
                const QString outPath = makeOutputPath(inputPath);
                image::ImageMeta m2;
                if (image::ImageIO::read(outPath, preview, m2)) {
                    emit previewImageReady(makePreview(preview), false);
                }
            } else {
                cv::Mat img2;
                image::ImageMeta m3;
                if (image::ImageIO::read(inputPath, img2, m3)) {
                    emit previewImageReady(makePreview(img2), false);
                }
            }
        }

        // 进度
        BatchProgress prog;
        prog.current = m_index + 1;
        prog.total = total;
        prog.currentFile = fi.fileName();
        prog.succeeded = m_succeeded;
        prog.failed = m_failed;
        prog.elapsedSeconds =
            (QDateTime::currentMSecsSinceEpoch() - m_startMs) / 1000.0;

        if (!m_fileDurations.isEmpty()) {
            double sum = 0.0;
            for (qint64 d : m_fileDurations) sum += d;
            const double avg = sum / m_fileDurations.size();
            const int remain = total - (m_index + 1);
            prog.remainingSeconds = avg * remain / 1000.0;
        }

        emit progressChanged(prog);
    }

    // 全部完成
    m_running = false;
    BatchResult result;
    result.ok = true;
    result.cancelled = false;
    result.total = total;
    result.succeeded = m_succeeded;
    result.failed = m_failed;
    result.failedFiles = m_failedFiles;
    result.totalSeconds =
        (QDateTime::currentMSecsSinceEpoch() - m_startMs) / 1000.0;
    emit finished(result);
}

// ============================================================
// 处理单张
// ============================================================
bool BatchProcessor::processOneFile(const QString &inputPath, QString &outError)
{
    cv::Mat img;
    image::ImageMeta meta;
    if (!image::ImageIO::read(inputPath, img, meta)) {
        outError = QString::fromUtf8("读取失败");
        return false;
    }

    for (const StepItem &step : m_options.steps) {
        if (!step.enabled) continue;

        switch (step.type) {
        case StepType::Deskew:      runDeskew(img); break;
        case StepType::BlackEdge:   runBlackEdge(img); break;
        case StepType::Denoise:     runDenoise(img); break;
        case StepType::Enhance:     runEnhance(img); break;
        case StepType::ErrPage:     runErrPage(img, inputPath); break;
        case StepType::Background:  runBackground(img); break;
        case StepType::ColorLine:   runColorLine(img); break;
        }
    }

    const QString outputPath = makeOutputPath(inputPath);

    if (!image::JpegWriter::write(outputPath, img, m_options.jpegOpt)) {
        outError = QString::fromUtf8("保存失败：%1").arg(outputPath);
        return false;
    }

    return true;
}

// ============================================================
// 各步骤
// ============================================================
bool BatchProcessor::runDeskew(cv::Mat &img)
{
    const process::DeskewResult r = process::Deskew::autoDeskew(img);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runBlackEdge(cv::Mat &img)
{
    const process::BlackEdgeResult r =
        process::BlackEdge::removeBlackEdge(img, m_options.blackEdgeOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runDenoise(cv::Mat &img)
{
    protect::StampProtectOptions stampOpt = m_options.stampOpt;
    const protect::StampProtectResult pr =
        protect::StampProtect::detect(img, stampOpt);
    if (pr.ok) {
        m_options.denoiseOpt.protectMask = pr.mask;
    }

    const process::DenoiseResult r =
        process::Denoise::removeSpots(img, m_options.denoiseOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runEnhance(cv::Mat &img)
{
    const process::EnhanceResult r =
        process::Enhance::enhanceText(img, m_options.enhanceOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runErrPage(cv::Mat &img, const QString &sourcePath)
{
    const process::ErrPageResult r =
        process::ErrPage::process(img, sourcePath, m_options.errPageOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runBackground(cv::Mat &img)
{
    protect::StampProtectOptions stampOpt = m_options.stampOpt;
    const protect::StampProtectResult pr =
        protect::StampProtect::detect(img, stampOpt);
    process::BackgroundOptions bgOpt = m_options.backgroundOpt;
    if (pr.ok) {
        bgOpt.protectMask = pr.mask;
    }

    const process::BackgroundResult r =
        process::Background::whiten(img, bgOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runColorLine(cv::Mat &img)
{
    const process::ColorLineResult r =
        process::ColorLine::clear(img, m_options.colorLineOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

// ============================================================
// 输出路径
// ============================================================
QString BatchProcessor::makeOutputPath(const QString &inputPath) const
{
    const QFileInfo fi(inputPath);
    QString baseName = fi.completeBaseName();

    if (!m_options.outputSuffix.isEmpty()) {
        baseName += m_options.outputSuffix;
    }

    const QString outName = baseName + QStringLiteral(".jpg");
    return QDir(m_options.outputDir).filePath(outName);
}

// ============================================================
// 缩放为预览图
// ============================================================
cv::Mat BatchProcessor::makePreview(const cv::Mat &img, int maxSize) const
{
    if (img.empty()) return img;

    const int w = img.cols;
    const int h = img.rows;
    const int longest = std::max(w, h);

    if (longest <= maxSize) return img.clone();

    const double scale = static_cast<double>(maxSize) / longest;
    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

} // namespace batch