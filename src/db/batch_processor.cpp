#include "batch_processor.h"

#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
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

    // ---- 规范化输出目录列表 ----
    {
        QStringList dirs;
        dirs.append(m_options.outputDir);
        for (const QString &d : m_options.outputDirs) {
            if (d.isEmpty()) continue;
            if (d == m_options.outputDir) continue;
            if (dirs.contains(d)) continue;
            dirs.append(d);
        }
        m_options.outputDirs = dirs;
    }

    // ---- 规范化报告目录 ----
    if (m_options.reportDir.trimmed().isEmpty()) {
        m_options.reportDir = m_options.outputDir;
    }

    // ---- 规范化分片参数 ----
    if (m_options.totalShards < 1) m_options.totalShards = 1;
    if (m_options.shardIndex < 0) m_options.shardIndex = 0;
    if (m_options.totalShards > 0) {
        m_options.shardIndex = m_options.shardIndex % m_options.totalShards;
        if (m_options.shardIndex < 0) {
            m_options.shardIndex += m_options.totalShards;
        }
    }

    // ---- 规范化备份目录 ----
    if (m_options.backupEnabled && m_options.backupDir.trimmed().isEmpty()) {
        // 启用备份但没填目录：自动关掉，避免误操作
        m_options.backupEnabled = false;
    }

    m_running = true;
    m_paused = false;
    m_cancelled = false;

    m_index = 0;
    m_succeeded = 0;
    m_failed = 0;
    m_failedFiles.clear();
    m_fileDurations.clear();
    m_processedTotal = 0;
    m_globalTotal = 0;
    m_backedUp = 0;
    m_backupFailed = 0;

    m_startMs = QDateTime::currentMSecsSinceEpoch();

    for (const QString &d : m_options.outputDirs) {
        QDir().mkpath(d);
    }
    QDir().mkpath(m_options.reportDir);

    // 备份目录也提前创建（按日期子目录在处理时再建）
    if (m_options.backupEnabled) {
        QDir().mkpath(m_options.backupDir);
    }

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
    const int totalAll = m_options.inputFiles.size();
    m_globalTotal = totalAll;

    QList<int> myIndices;
    myIndices.reserve(totalAll);
    for (int i = 0; i < totalAll; ++i) {
        if (m_options.totalShards > 1) {
            if ((i % m_options.totalShards) != m_options.shardIndex) {
                continue;
            }
        }
        myIndices.append(i);
    }

    const int total = myIndices.size();
    m_processedTotal = total;

    for (m_index = 0; m_index < total; ++m_index) {

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
            result.reportPath = writeReport(true);
            result.backedUp     = m_backedUp;
            result.backupFailed = m_backupFailed;
            emit finished(result);
            return;
        }

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
            result.reportPath = writeReport(true);
            result.backedUp     = m_backedUp;
            result.backupFailed = m_backupFailed;
            emit finished(result);
            return;
        }

        const int globalIdx = myIndices[m_index];
        const QString inputPath = m_options.inputFiles[globalIdx];
        const QFileInfo fi(inputPath);

        emit fileStarted(inputPath);

        {
            cv::Mat img;
            image::ImageMeta meta;
            if (image::ImageIO::read(inputPath, img, meta)) {
                emit previewImageReady(makePreview(img), true);
            }
        }

        // ★ 3.3 备份：处理前先复制原图
        if (m_options.backupEnabled) {
            QString backupErr;
            if (backupOneFile(inputPath, backupErr)) {
                ++m_backedUp;
            } else {
                ++m_backupFailed;
                qWarning() << "[BatchProcessor] backup failed:"
                           << inputPath << "err:" << backupErr;
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
    result.reportPath = writeReport(false);
    result.backedUp     = m_backedUp;
    result.backupFailed = m_backupFailed;
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

    if (m_options.outputDirs.size() > 1) {
        const QString fileName = QFileInfo(outputPath).fileName();

        for (int i = 1; i < m_options.outputDirs.size(); ++i) {
            const QString dstDir = m_options.outputDirs[i];
            if (dstDir.isEmpty()) continue;

            if (!QDir().mkpath(dstDir)) {
                qWarning() << "[BatchProcessor] mkpath failed:" << dstDir;
                continue;
            }

            const QString dstPath = QDir(dstDir).filePath(fileName);

            if (QFile::exists(dstPath)) {
                QFile::remove(dstPath);
            }

            if (!QFile::copy(outputPath, dstPath)) {
                qWarning() << "[BatchProcessor] copy to backup failed:"
                           << outputPath << "->" << dstPath;
            }
        }
    }

    return true;
}

// ============================================================
// ★ 3.3 备份单张（处理前调用）
//   目标：<backupDir>/YYYY-MM-DD/原文件名
// ============================================================
bool BatchProcessor::backupOneFile(const QString &inputPath,
                                   QString &outError) const
{
    if (m_options.backupDir.trimmed().isEmpty()) {
        outError = QString::fromUtf8("备份目录为空");
        return false;
    }

    const QFileInfo fi(inputPath);
    if (!fi.exists()) {
        outError = QString::fromUtf8("源文件不存在");
        return false;
    }

    const QString dateDir =
        m_options.backupDir + QStringLiteral("/")
        + QDate::currentDate().toString(QStringLiteral("yyyy-MM-dd"));

    if (!QDir().mkpath(dateDir)) {
        outError = QString::fromUtf8("无法创建备份子目录：%1").arg(dateDir);
        return false;
    }

    const QString dstPath = QDir(dateDir).filePath(fi.fileName());

    // 同名（同一天同一文件名）已存在：先删再复制
    if (QFile::exists(dstPath)) {
        QFile::remove(dstPath);
    }

    if (!QFile::copy(inputPath, dstPath)) {
        outError = QString::fromUtf8("复制失败：%1 -> %2")
                       .arg(inputPath, dstPath);
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

// ============================================================
// 写日志报告
// ============================================================
QString BatchProcessor::writeReport(bool cancelled) const
{
    if (m_options.reportDir.trimmed().isEmpty()) {
        return QString();
    }

    QDir().mkpath(m_options.reportDir);

    const QDateTime now = QDateTime::currentDateTime();
    const QString ts = now.toString(QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
    const QString fileName = QStringLiteral("batch_report_%1.txt").arg(ts);
    const QString path = QDir(m_options.reportDir).filePath(fileName);

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        qWarning() << "[BatchProcessor] open report failed:" << path;
        return QString();
    }

    f.write("\xEF\xBB\xBF");

    QTextStream out(&f);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    out.setCodec("UTF-8");
#endif

    const QDateTime startTime = QDateTime::fromMSecsSinceEpoch(m_startMs);

    double avgMs = 0.0;
    if (!m_fileDurations.isEmpty()) {
        double sum = 0.0;
        for (qint64 d : m_fileDurations) sum += d;
        avgMs = sum / m_fileDurations.size();
    }

    const double elapsedSec =
        (QDateTime::currentMSecsSinceEpoch() - m_startMs) / 1000.0;

    const QString stateStr = cancelled
        ? QString::fromUtf8("已取消")
        : QString::fromUtf8("全部完成");

    out << QString::fromUtf8("批量扫描图片净化增强软件 — 批处理报告\n");
    out << QString::fromUtf8("========================================\n");
    out << QString::fromUtf8("开始时间：%1\n")
           .arg(startTime.toString(QStringLiteral("yyyy-MM-dd HH:mm:ss")));
    out << QString::fromUtf8("结束时间：%1\n")
           .arg(now.toString(QStringLiteral("yyyy-MM-dd HH:mm:ss")));
    out << QString::fromUtf8("结束状态：%1\n").arg(stateStr);
    out << QString::fromUtf8("\n");

    out << QString::fromUtf8("【输入】\n");
    out << QString::fromUtf8("  输入文件总数（分片前）：%1\n").arg(m_globalTotal);
    out << QString::fromUtf8("  本机编号 / 总机器数：%1 / %2\n")
           .arg(m_options.shardIndex)
           .arg(m_options.totalShards);
    out << QString::fromUtf8("  本机实际处理数：%1\n").arg(m_processedTotal);
    out << QString::fromUtf8("\n");

    out << QString::fromUtf8("【输出】\n");
    out << QString::fromUtf8("  主输出目录：%1\n").arg(m_options.outputDir);
    for (int i = 1; i < m_options.outputDirs.size(); ++i) {
        out << QString::fromUtf8("  副本目录 %1：%2\n")
               .arg(i)
               .arg(m_options.outputDirs[i]);
    }
    out << QString::fromUtf8("\n");

    // ★ 3.3 备份段
    if (m_options.backupEnabled) {
        out << QString::fromUtf8("【备份】\n");
        out << QString::fromUtf8("  备份目录：%1\n").arg(m_options.backupDir);
        out << QString::fromUtf8("  成功：%1\n").arg(m_backedUp);
        out << QString::fromUtf8("  失败：%1\n").arg(m_backupFailed);
        out << QString::fromUtf8("\n");
    }

    out << QString::fromUtf8("【结果】\n");
    out << QString::fromUtf8("  成功：%1\n").arg(m_succeeded);
    out << QString::fromUtf8("  失败：%1\n").arg(m_failed);
    out << QString::fromUtf8("  总耗时：%1 秒\n")
           .arg(QString::number(elapsedSec, 'f', 2));
    out << QString::fromUtf8("  平均每张：%1 毫秒\n")
           .arg(QString::number(avgMs, 'f', 1));
    out << QString::fromUtf8("\n");

    if (m_failed > 0) {
        out << QString::fromUtf8("【失败文件】\n");
        for (const QString &ff : m_failedFiles) {
            out << QString::fromUtf8("  %1\n").arg(ff);
        }
        out << QString::fromUtf8("\n");
    }

    out.flush();
    f.close();

    return path;
}

} // namespace batch