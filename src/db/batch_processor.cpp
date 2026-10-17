#include "batch_processor.h"

#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
#include <QDebug>

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
    start();
}

void BatchProcessor::cancel()
{
    m_running = false;
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

// 这里省略你原有的其他函数（runBackground, runColorLine 等）... 
// 它们在原文件中保持不变即可，因为这次只改了 runErrPage 的塞入逻辑。
// 如果你替换时发现下面缺少你原来的函数，请不要删除，保留你原有的即可。

} // namespace batch