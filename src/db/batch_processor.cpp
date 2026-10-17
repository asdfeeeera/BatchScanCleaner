#include "batch_processor.h"

#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
#include <QDebug>
#include <QMutexLocker>

#include <opencv2/imgproc.hpp>

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
    start();
}

void BatchProcessor::pause()
{
    QMutexLocker locker(&m_mutex);
    m_paused = true;
}

void BatchProcessor::resume()
{
    QMutexLocker locker(&m_mutex);
    m_paused = false;
    m_pauseCond.wakeAll();
}

void BatchProcessor::cancel()
{
    QMutexLocker locker(&m_mutex);
    m_running = false;
    m_paused = false;
    m_pauseCond.wakeAll();
}

void BatchProcessor::run()
{
    int total = m_options.files.size();
    int success = 0;
    int fail = 0;
    QDateTime startTime = QDateTime::currentDateTime();

    for (int i = 0; i < total; ++i) {
        {
            QMutexLocker locker(&m_mutex);
            if (!m_running) break;
            while (m_paused && m_running) {
                m_pauseCond.wait(&m_mutex);
            }
            if (!m_running) break;
        }

        QString filePath = m_options.files[i];
        QFileInfo fi(filePath);
        
        emit progressChanged(i + 1, total, fi.fileName());

        cv::Mat img = cv::imread(filePath.toStdString(), cv::IMREAD_COLOR);
        if (img.empty()) {
            fail++;
            continue;
        }

        bool ok = true;
        QStringList doneSteps;

        // 1. 自动扶正
        if (m_options.enableDeskew) {
            process::DeskewResult r = process::Deskew::autoDeskew(img);
            if (r.ok) { img = r.image; doneSteps << QString::fromUtf8("扶正"); }
        }

        // 2. 黑边去除
        if (m_options.enableBlackEdge) {
            process::BlackEdgeOptions opt;
            opt.fillWhite = true;
            process::BlackEdgeResult r = process::BlackEdge::removeBlackEdge(img, opt);
            if (r.ok) { img = r.image; doneSteps << QString::fromUtf8("黑边"); }
        }

        // 3. 错误页码处理 (核心)
        if (m_options.enableErrPage) {
            if (runErrPage(img, filePath)) {
                doneSteps << QString::fromUtf8("错误页码");
            }
        }

        // 4. 污点去除
        if (m_options.enableDenoise) {
            process::DenoiseOptions opt;
            protect::StampProtectOptions protectOpt;
            protect::StampProtectResult pr = protect::StampProtect::detect(img, protectOpt);
            if (pr.ok) opt.protectMask = pr.mask;
            process::DenoiseResult r = process::Denoise::removeSpots(img, opt);
            if (r.ok) { img = r.image; doneSteps << QString::fromUtf8("污点"); }
        }

        // 5. 文字加深
        if (m_options.enableEnhance) {
            process::EnhanceOptions opt;
            process::EnhanceResult r = process::Enhance::enhanceText(img, opt);
            if (r.ok) { img = r.image; doneSteps << QString::fromUtf8("文字"); }
        }

        // 6. 底色处理
        if (m_options.enableBackground) {
            process::BackgroundOptions opt;
            protect::StampProtectOptions protectOpt;
            protect::StampProtectResult pr = protect::StampProtect::detect(img, protectOpt);
            if (pr.ok) opt.protectMask = pr.mask;
            process::BackgroundResult r = process::Background::whiten(img, opt);
            if (r.ok) { img = r.image; doneSteps << QString::fromUtf8("底色"); }
        }

        // 保存
        if (ok) {
            QString outDir = m_options.outputDir;
            if (outDir.isEmpty()) outDir = fi.absolutePath() + "/output";
            QDir().mkpath(outDir);
            
            QString outName = fi.completeBaseName() + "_clean." + fi.suffix();
            QString outPath = outDir + "/" + outName;
            
            cv::imwrite(outPath.toStdString(), img);
            success++;
            
            emit previewImageReady(img, false);
        } else {
            fail++;
        }
    }

    emit batchFinished(success, fail, startTime.secsTo(QDateTime::currentDateTime()));
}

bool BatchProcessor::runErrPage(cv::Mat &img, const QString &sourcePath)
{
    // 强制读取最新设置，确保单张和批量的参数完全一致！
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

} // namespace batch