#include "batch_processor.h"

#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
#include <QDebug>
#include <QMutexLocker>
#include <QElapsedTimer>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../analyze/pending_center.h"
#include "../ui/settings_dialog.h"

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
    m_processedTotal = 0;
    m_globalTotal = options.inputFiles.size();
    
    m_startMs = QDateTime::currentMSecsSinceEpoch();
    
    start();
}

void BatchProcessor::pause()
{
    m_paused = true;
}

void BatchProcessor::resume()
{
    m_paused = false;
}

void BatchProcessor::cancel()
{
    m_cancelled = true;
    m_running = false;
}

void BatchProcessor::run()
{
    QElapsedTimer timer;
    timer.start();

    for (m_index = 0; m_index < m_options.inputFiles.size(); ++m_index) {
        if (m_cancelled) break;

        while (m_paused && !m_cancelled) {
            QThread::msleep(100);
        }
        if (m_cancelled) break;

        if (m_options.totalShards > 1) {
            if ((m_index % m_options.totalShards) != m_options.shardIndex) {
                continue;
            }
        }

        const QString inputPath = m_options.inputFiles[m_index];

        if (m_options.backupEnabled) {
            QString err;
            if (!backupOneFile(inputPath, err)) {
                qWarning() << "备份失败:" << inputPath << err;
                m_backupFailed++;
            } else {
                m_backedUp++;
            }
        }

        emit fileStarted(inputPath);

        qint64 fileStartMs = QDateTime::currentMSecsSinceEpoch();
        QString errMsg;
        bool success = processOneFile(inputPath, errMsg);
        qint64 fileEndMs = QDateTime::currentMSecsSinceEpoch();

        m_fileDurations.append(fileEndMs - fileStartMs);
        m_processedTotal++;

        if (success) {
            m_succeeded++;
            emit fileFinished(inputPath, true);
        } else {
            m_failed++;
            m_failedFiles.append(inputPath);
            emit fileFinished(inputPath, false);
        }

        BatchProgress progress;
        progress.current = m_processedTotal;
        progress.total = m_options.totalShards > 1 ? (m_globalTotal / m_options.totalShards) : m_globalTotal;
        progress.currentFile = QFileInfo(inputPath).fileName();
        progress.succeeded = m_succeeded;
        progress.failed = m_failed;
        progress.elapsedSeconds = timer.elapsed() / 1000.0;
        
        if (m_processedTotal > 0) {
            double avgMs = static_cast<double>(timer.elapsed()) / m_processedTotal;
            progress.remainingSeconds = (progress.total - m_processedTotal) * avgMs / 1000.0;
        }
        emit progressChanged(progress);
    }

    BatchResult res;
    res.cancelled = m_cancelled;
    res.total = m_options.totalShards > 1 ? (m_globalTotal / m_options.totalShards) : m_globalTotal;
    res.succeeded = m_succeeded;
    res.failed = m_failed;
    res.failedFiles = m_failedFiles;
    res.totalSeconds = timer.elapsed() / 1000.0;
    res.backedUp = m_backedUp;
    res.backupFailed = m_backupFailed;
    res.ok = true;

    res.reportPath = writeReport(m_cancelled);

    m_running = false;
    emit finished(res);
}

bool BatchProcessor::processOneFile(const QString &inputPath, QString &outError)
{
    // ★ 核心修复：使用 image::ImageIO::read 读取，完美解决中文路径问题
    cv::Mat img;
    image::ImageMeta meta;
    if (!image::ImageIO::read(inputPath, img, meta)) {
        outError = "读取图片失败(可能包含中文路径或文件损坏)";
        return false;
    }

    cv::Mat stampMask;
    protect::StampProtectResult spr = protect::StampProtect::detect(img, m_options.stampOpt);
    if (spr.ok) stampMask = spr.mask;

    for (const StepItem &step : m_options.steps) {
        if (!step.enabled) continue;

        switch (step.type) {
            case StepType::Deskew:     runDeskew(img); break;
            case StepType::BlackEdge:  runBlackEdge(img); break;
            case StepType::Denoise:    runDenoise(img, stampMask); break;
            case StepType::Enhance:    runEnhance(img); break;
            case StepType::ErrPage:    runErrPage(img, inputPath); break;
            case StepType::Background: runBackground(img, stampMask); break;
            case StepType::ColorLine:  runColorLine(img); break;
            default: break;
        }
    }

    QString outPath = makeOutputPath(inputPath);
    QDir().mkpath(QFileInfo(outPath).absolutePath());

    if (!image::JpegWriter::write(outPath, img, m_options.jpegOpt)) {
        outError = "保存失败";
        return false;
    }

    cv::Mat preview = makePreview(img);
    if (!preview.empty()) {
        emit previewImageReady(preview, false);
    }

    return true;
}

bool BatchProcessor::runDeskew(cv::Mat &img)
{
    process::DeskewResult r = process::Deskew::autoDeskew(img);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runBlackEdge(cv::Mat &img)
{
    process::BlackEdgeResult r = process::BlackEdge::removeBlackEdge(img, m_options.blackEdgeOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runDenoise(cv::Mat &img, const cv::Mat &protectMask)
{
    process::DenoiseOptions opt = m_options.denoiseOpt;
    opt.protectMask = protectMask;
    process::DenoiseResult r = process::Denoise::removeSpots(img, opt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runEnhance(cv::Mat &img)
{
    process::EnhanceResult r = process::Enhance::enhanceText(img, m_options.enhanceOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runBackground(cv::Mat &img, const cv::Mat &protectMask)
{
    process::BackgroundOptions opt = m_options.backgroundOpt;
    opt.protectMask = protectMask;
    process::BackgroundResult r = process::Background::whiten(img, opt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runColorLine(cv::Mat &img)
{
    process::ColorLineResult r = process::ColorLine::clear(img, m_options.colorLineOpt);
    if (r.ok && !r.image.empty()) { img = r.image; return true; }
    return false;
}

bool BatchProcessor::runErrPage(cv::Mat &img, const QString &sourcePath)
{
    process::ErrPageOptions opt = m_options.errPageOpt;
    opt.enablePaddleOcr = ui::AppSettings::load().enablePaddleOcr;
    opt.tesseractPath   = QString();
    opt.fillWhite       = true;

    const process::ErrPageResult r = process::ErrPage::process(img, sourcePath, opt);

    if (r.ok && !r.image.empty()) {
        const QFileInfo fi(sourcePath);
        
        for (const auto &it : r.items) {
            analyze::PendingItem p;
            p.type = analyze::PendingType::WrongPageNumber;
            p.suggestedAction = analyze::PendingAction::Remove;
            p.sourceImagePath = sourcePath;
            p.fileName = fi.fileName();
            p.boundingBox = it.boundingBox;
            p.confidence = it.confidence;
            p.reason = QString::fromUtf8("检测到疑似错误页码（需人工确认）");
            p.detail = QString::fromUtf8("识别结果：%1，正确页码：%2")
                           .arg(it.recognizedNumber).arg(r.correctPage);

            cv::Rect safe = it.boundingBox & cv::Rect(0, 0, r.image.cols, r.image.rows);
            if (safe.width > 0 && safe.height > 0) {
                p.thumbnail = r.image(safe).clone();
            }

            analyze::PendingCenter::instance().addItem(p);
        }

        img = r.image;
        return true;
    }
    return false;
}

QString BatchProcessor::makeOutputPath(const QString &inputPath) const
{
    const QFileInfo fi(inputPath);
    QString baseName = fi.completeBaseName() + m_options.outputSuffix + "." + fi.suffix();
    
    QString subDir = relativeSubDir(inputPath);
    QString outDir = m_options.outputDir;
    if (!subDir.isEmpty()) {
        outDir = QDir(outDir).filePath(subDir);
    }
    return QDir(outDir).filePath(baseName);
}

QString BatchProcessor::relativeSubDir(const QString &inputPath) const
{
    if (m_options.inputRootDir.isEmpty()) return QString();
    
    QDir rootDir(m_options.inputRootDir);
    QString relPath = rootDir.relativeFilePath(inputPath);
    
    if (relPath.startsWith("..")) return QString();
    
    QFileInfo fi(relPath);
    return fi.path() == "." ? QString() : fi.path();
}

cv::Mat BatchProcessor::makePreview(const cv::Mat &img, int maxSize) const
{
    if (img.empty()) return cv::Mat();
    int w = img.cols;
    int h = img.rows;
    if (w <= maxSize && h <= maxSize) return img.clone();

    double scale = static_cast<double>(maxSize) / std::max(w, h);
    cv::Mat res;
    cv::resize(img, res, cv::Size(), scale, scale, cv::INTER_AREA);
    return res;
}

QString BatchProcessor::writeReport(bool cancelled) const
{
    QString dir = m_options.reportDir;
    if (dir.isEmpty()) dir = m_options.outputDir;
    
    QDir().mkpath(dir);
    QString path = QDir(dir).filePath(
        QStringLiteral("batch_report_%1.txt")
            .arg(QDateTime::currentDateTime().toString("yyyy-MM-dd_HH-mm-ss")));

    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream ts(&f);
        ts.setCodec("UTF-8");
        ts << "批量扫描图片净化增强软件 — 批处理报告\n";
        ts << "====================================\n";
        ts << QString("开始时间：%1\n").arg(QDateTime::fromMSecsSinceEpoch(m_startMs).toString("yyyy-MM-dd HH:mm:ss"));
        ts << QString("结束时间：%1\n").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
        ts << QString("结束状态：%1\n\n").arg(cancelled ? "已取消" : "全部完成");

        ts << "【结果】\n";
        ts << QString("成功：%1\n").arg(m_succeeded);
        ts << QString("失败：%1\n").arg(m_failed);
        
        if (!m_failedFiles.isEmpty()) {
            ts << "失败文件列表：\n";
            for (const QString &failedFile : m_failedFiles) {
                ts << "  " << failedFile << "\n";
            }
        }
    }
    f.close();
    return path;
}

bool BatchProcessor::backupOneFile(const QString &inputPath, QString &outError) const
{
    const QFileInfo fi(inputPath);
    const QString dateStr = QDate::currentDate().toString("yyyy-MM-dd");
    
    QString subDir = relativeSubDir(inputPath);
    QString backupDir = QDir(m_options.backupDir).filePath(dateStr);
    if (!subDir.isEmpty()) {
        backupDir = QDir(backupDir).filePath(subDir);
    }
    
    QDir().mkpath(backupDir);
    QString dstPath = QDir(backupDir).filePath(fi.fileName());
    
    if (QFile::exists(dstPath)) {
        QFile::remove(dstPath);
    }
    
    if (!QFile::copy(inputPath, dstPath)) {
        outError = "复制失败";
        return false;
    }
    return true;
}

} // namespace batch