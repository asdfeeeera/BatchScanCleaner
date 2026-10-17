#include "errpage.h"
#include "onnx_ocr.h"
#include "paddle_ocr.h"
#include "yolo_detector.h"
#include <QFileInfo>
#include <QCoreApplication>
#include <QMutex>
#include <QMutexLocker>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QImage>
#include <QProcess>
#include <QTemporaryDir>
#include <QStandardPaths>
#include <algorithm>
#include <cmath>
#include <vector>

namespace process {

int ErrPage::parseCorrectPage(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QString baseName = info.completeBaseName().trimmed();

    QString digitsOnly;
    for (const QChar &c : baseName) {
        if (c.isDigit()) digitsOnly += c;
    }

    if (digitsOnly.isEmpty()) return -1;

    bool ok = false;
    const int page = digitsOnly.toInt(&ok);
    if (ok && page >= 0) return page;
    return -1;
}

namespace {

static process::PaddleOcr *g_paddleOcr = nullptr;
static QMutex             g_paddleMutex;
static int                g_paddleFailCount = 0;
static const int          kPaddleMaxFails   = 3;
static bool               g_paddleDisabled  = false;

static process::PaddleOcr *getPaddleOcr()
{
    QMutexLocker locker(&g_paddleMutex);
    if (g_paddleDisabled) return nullptr;
    if (g_paddleOcr && g_paddleOcr->isReady()) return g_paddleOcr;
    if (g_paddleOcr) {
        delete g_paddleOcr;
        g_paddleOcr = nullptr;
    }

    const QString exePath = QCoreApplication::applicationDirPath()
        + QStringLiteral("/PaddleOCR-json/PaddleOCR-json.exe");
    if (!QFileInfo::exists(exePath)) {
        g_paddleDisabled = true;
        return nullptr;
    }

    process::PaddleOcr *ocr = new process::PaddleOcr();
    if (!ocr->start(exePath)) {
        delete ocr;
        ++g_paddleFailCount;
        if (g_paddleFailCount >= kPaddleMaxFails) g_paddleDisabled = true;
        return nullptr;
    }
    g_paddleOcr = ocr;
    return g_paddleOcr;
}

static void notePaddleFailure()
{
    QMutexLocker locker(&g_paddleMutex);
    ++g_paddleFailCount;
    if (g_paddleFailCount >= kPaddleMaxFails) g_paddleDisabled = true;
}

static void notePaddleSuccess()
{
    QMutexLocker locker(&g_paddleMutex);
    g_paddleFailCount = 0;
}

// ★ 放宽尺寸限制，避免 215 和 264 连在一起时被当成异常块过滤
std::vector<cv::Rect> detectAllDigits(const cv::Mat &gray)
{
    std::vector<cv::Rect> result;
    const int W = gray.cols;
    const int H = gray.rows;

    // 只在右上角和左上角查找，范围严格限制（宽 20%，高 15%）
    const int marginX = static_cast<int>(W * 0.20);
    const int marginY = static_cast<int>(H * 0.15);

    std::vector<cv::Rect> rois;
    rois.push_back(cv::Rect(W - marginX, 0, marginX, marginY)); // 右上
    rois.push_back(cv::Rect(0, 0, marginX, marginY));           // 左上

    for (const auto &roi : rois) {
        cv::Rect safeRoi = roi & cv::Rect(0, 0, W, H);
        if (safeRoi.width <= 0 || safeRoi.height <= 0) continue;

        cv::Mat roiImg = gray(safeRoi);
        cv::Mat binary;
        cv::threshold(roiImg, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        // 仅用非常轻微的闭运算连接笔画，绝不横向膨胀
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat closed;
        cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

        cv::Mat labels, stats, centroids;
        const int nLabels = cv::connectedComponentsWithStats(closed, labels, stats, centroids, 8, CV_32S);

        // 收集候选框
        std::vector<cv::Rect> boxes;
        for (int i = 1; i < nLabels; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            // 过滤明显的噪点，但允许数字笔画较小
            if (h < 10 || h > 120) continue;
            if (w < 5 || w > 250) continue;
            if (area < 30) continue;

            boxes.push_back(cv::Rect(
                safeRoi.x + stats.at<int>(i, cv::CC_STAT_LEFT),
                safeRoi.y + stats.at<int>(i, cv::CC_STAT_TOP),
                w, h));
        }

        if (boxes.empty()) continue;

        // 排序并合并
        std::sort(boxes.begin(), boxes.end(), [](const cv::Rect &a, const cv::Rect &b) {
            if (std::abs(a.y - b.y) < a.height / 2) return a.x < b.x;
            return a.y < b.y;
        });

        cv::Rect current = boxes[0];
        for (size_t i = 1; i < boxes.size(); ++i) {
            const cv::Rect &next = boxes[i];
            
            // 计算垂直重叠度
            const int top = std::max(current.y, next.y);
            const int bot = std::min(current.y + current.height, next.y + next.height);
            const int minH = std::min(current.height, next.height);
            const double vOverlap = (minH <= 0) ? 0.0 : static_cast<double>(bot - top) / minH;
            
            // 间距
            const int gap = next.x - (current.x + current.width);

            // 合并逻辑：同一行，且间距小于当前字高（太远的绝不合并）
            if (vOverlap > 0.5 && gap >= -5 && gap < current.height) {
                current = cv::Rect(
                    std::min(current.x, next.x), std::min(current.y, next.y),
                    std::max(current.x + current.width, next.x + next.width) - std::min(current.x, next.x),
                    std::max(current.y + current.height, next.y + next.height) - std::min(current.y, next.y)
                );
            } else {
                result.push_back(current);
                current = next;
            }
        }
        result.push_back(current);
    }

    return result;
}

bool fuzzyMatchPage(int recognized, int correctPage)
{
    if (recognized < 0 || correctPage < 0) return false;
    if (recognized == correctPage) return true;

    const QString recStr = QString::number(recognized);
    const QString corrStr = QString::number(correctPage);

    {
        QString recNoZero = recStr;
        QString corrNoZero = corrStr;
        while (recNoZero.startsWith(QLatin1Char('0')) && recNoZero.length() > 1) {
            recNoZero.remove(0, 1);
        }
        while (corrNoZero.startsWith(QLatin1Char('0')) && corrNoZero.length() > 1) {
            corrNoZero.remove(0, 1);
        }
        if (recNoZero.length() == corrNoZero.length()) {
            QString a = recNoZero;
            QString b = corrNoZero;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a == b) return true;
        }
    }

    if (std::abs(recStr.length() - corrStr.length()) > 1) return false;
    if (corrStr.endsWith(recStr) || recStr.endsWith(corrStr)) return true;
    if (corrStr.startsWith(recStr) || recStr.startsWith(corrStr)) return true;
    return false;
}

cv::Mat removeHorizontalLinesByMask(const cv::Mat &gray)
{
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int kernelW = std::max(25, static_cast<int>(gray.cols * 0.4));
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelW, 1));
    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    cv::Mat dKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::Mat hLinesDilated;
    cv::dilate(hLines, hLinesDilated, dKernel);

    cv::Mat result = gray.clone();
    result.setTo(255, hLinesDilated);
    return result;
}

QString locateTesseract(const QString &hint)
{
    if (!hint.isEmpty() && QFile::exists(hint)) return hint;

    const QString appDir = QCoreApplication::applicationDirPath();
    const QString c1 = appDir + QStringLiteral("/tesseract/tesseract.exe");
    if (QFile::exists(c1)) return c1;
    const QString c2 = appDir + QStringLiteral("/tesseract.exe");
    if (QFile::exists(c2)) return c2;

    const QString inPath = QStandardPaths::findExecutable(QStringLiteral("tesseract"));
    if (!inPath.isEmpty()) return inPath;
    return QString();
}

struct OcrAttempt
{
    int number = -1;
    QString text;
};

OcrAttempt tryTesseract(const cv::Mat &image, const QString &tessExe,
                         const QString &tessdataDir, const QString &psmMode)
{
    OcrAttempt result;

    QTemporaryDir tempDir;
    if (!tempDir.isValid()) return result;

    const QString tmpPng = tempDir.path() + QStringLiteral("/digit.png");
    const QString tmpOutBase = tempDir.path() + QStringLiteral("/out");

    cv::Mat enlarged;
    cv::resize(image, enlarged, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);

    QImage qimg(enlarged.data, enlarged.cols, enlarged.rows,
                static_cast<int>(enlarged.step),
                (enlarged.channels() == 1) ? QImage::Format_Grayscale8
                                            : QImage::Format_RGB888);
    QImage copy = qimg.copy();
    if (!copy.save(tmpPng, "PNG")) return result;

    QProcess proc;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("TESSDATA_PREFIX"), tessdataDir);
    proc.setProcessEnvironment(env);
    proc.setWorkingDirectory(QFileInfo(tessExe).absolutePath());

    QStringList args;
    args << tmpPng << tmpOutBase
         << QStringLiteral("-l") << QStringLiteral("eng")
         << QStringLiteral("--psm") << psmMode
         << QStringLiteral("-c")
         << QStringLiteral("tessedit_char_whitelist=0123456789");

    proc.start(tessExe, args);
    if (!proc.waitForStarted(5000)) return result;
    if (!proc.waitForFinished(10000)) { proc.kill(); return result; }
    if (proc.exitCode() != 0) return result;

    QFile outFile(tmpOutBase + QStringLiteral(".txt"));
    if (!outFile.open(QIODevice::ReadOnly | QIODevice::Text)) return result;
    const QString rawText = QString::fromUtf8(outFile.readAll()).trimmed();
    outFile.close();

    QString digitsOnly;
    for (const QChar &c : rawText) {
        if (c.isDigit()) digitsOnly += c;
    }
    if (digitsOnly.isEmpty()) return result;

    bool ok = false;
    const int num = digitsOnly.toInt(&ok);
    if (!ok) return result;

    result.number = num;
    result.text = digitsOnly;
    return result;
}

void eraseBlock(cv::Mat &dst,
                const cv::Mat &srcGray,
                const cv::Rect &box,
                const cv::Scalar &white)
{
    cv::Rect r = box;
    r.x -= 2;
    r.y -= 2;
    r.width += 4;
    r.height += 4;
    r &= cv::Rect(0, 0, dst.cols, dst.rows);
    if (r.width <= 0 || r.height <= 0) return;

    cv::Mat roiGray = srcGray(r);
    cv::Mat bin;
    cv::threshold(roiGray, bin, 200, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats,
                                                     centroids, 8, CV_32S);

    const int boxLeft = 2;
    const int boxTop = 2;
    const int boxRight = boxLeft + box.width;
    const int boxBottom = boxTop + box.height;

    cv::Mat keep = cv::Mat::zeros(bin.size(), CV_8UC1);
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 10) continue;

        const int cLeft = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int cTop = stats.at<int>(i, cv::CC_STAT_TOP);
        const int cWidth = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int cHeight = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int cRight = cLeft + cWidth;
        const int cBottom = cTop + cHeight;

        if (cLeft < boxLeft - 2) continue;
        if (cRight > boxRight + 2) continue;
        if (cTop < boxTop - 2) continue;
        if (cBottom > boxBottom + 2) continue;

        keep.setTo(255, labels == i);
    }

    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(keep, keep, k);

    cv::Mat roiDst = dst(r);
    roiDst.setTo(white, keep);
}

} // namespace

int ErrPage::recognizeWithTesseract(const cv::Mat &digitImage,
                                     const QString &tesseractPath,
                                     QString &outText,
                                     double &outConfidence,
                                     bool enablePaddleOcr)
{
    if (digitImage.empty()) {
        outText = QString::fromUtf8("空图");
        outConfidence = 0.0;
        return -1;
    }

    // 1. ONNX 识别
    {
        process::OnnxOcr &ocr = process::OnnxOcr::instance();
        process::OnnxOcr::Result r = ocr.recognize(digitImage);

        // 只要 ONNX 有结果就返回（不再强求 >= 0.9）
        if (r.ok) {
            outText = r.text;
            outConfidence = r.confidence;
            return r.number;
        }

        // 2. PaddleOCR 兜底
        if (enablePaddleOcr && !g_paddleDisabled) {
            if (process::PaddleOcr *pocr = getPaddleOcr()) {
                const process::PaddleOcrResult pr = pocr->recognize(digitImage);
                if (pr.ok && !pr.text.isEmpty()) {
                    outText = pr.text;
                    outConfidence = pr.confidence;
                    notePaddleSuccess();
                    return pr.number;
                }
                notePaddleFailure();
            }
        }

        outText.clear();
        outConfidence = 0.0;
        return -1;
    }
}

bool ErrPage::detectCrossLine(const cv::Mat &gray,
                               const cv::Rect &digitBox,
                               double crossLineRatio)
{
    cv::Rect expanded = digitBox;
    expanded.y -= 15;
    expanded.height += 30;
    expanded.x -= 5;
    expanded.width += 10;
    expanded &= cv::Rect(0, 0, gray.cols, gray.rows);

    if (expanded.width <= 0 || expanded.height <= 0) return false;

    cv::Mat roi = gray(expanded);
    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int digitW = digitBox.width;
    const int minCount = digitW * 5 / 10;
    const int maxCount = digitW * 15 / 10;

    int hitRows = 0;
    for (int y = 0; y < binary.rows; ++y) {
        const uchar *row = binary.ptr<uchar>(y);
        int count = 0;
        for (int x = 0; x < binary.cols; ++x) {
            if (row[x] > 0) ++count;
        }
        if (count >= minCount && count <= maxCount) {
            ++hitRows;
            if (hitRows >= 2) return true;
        }
    }
    return false;
}

ErrPageResult ErrPage::process(const cv::Mat &src,
                                const QString &sourcePath,
                                const ErrPageOptions &options)
{
    ErrPageResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    result.correctPage = parseCorrectPage(sourcePath);

    const QString diagPath = QDir::homePath() +
                             QStringLiteral("/Desktop/ocr_debug.txt");
    QFile diagFile(diagPath);
    diagFile.open(QIODevice::Append | QIODevice::Text);
    QTextStream diag(&diagFile);
    diag.setCodec("UTF-8");
    auto writeDiag = [&](const QString &msg) {
        if (diagFile.isOpen()) {
            diag << msg << "\n";
            diag.flush();
        }
    };

    writeDiag(QString::fromUtf8("----- process() -----"));
    writeDiag(QString::fromUtf8("文件名：%1").arg(sourcePath));
    writeDiag(QString::fromUtf8("正确页码：%1").arg(result.correctPage));
    writeDiag(QString::fromUtf8("页面尺寸：%1 x %2").arg(W).arg(H));
    writeDiag(QString::fromUtf8("PaddleOCR 精识别：%1")
                  .arg(options.enablePaddleOcr ? QStringLiteral("开") : QStringLiteral("关")));

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    std::vector<cv::Rect> boxes = detectAllDigits(gray);
    writeDiag(QString::fromUtf8("全页扫描得到 %1 个边缘候选")
                  .arg(static_cast<int>(boxes.size())));

    std::vector<PageNumberItem> allItems;

    for (const auto &r : boxes) {
        cv::Rect expanded = r;
        expanded.x -= 10;
        expanded.y -= 10;
        expanded.width += 20;
        expanded.height += 20;
        expanded &= cv::Rect(0, 0, W, H);
        if (expanded.width <= 0 || expanded.height <= 0) continue;

        PageNumberItem item;
        item.boundingBox = r;
        item.isCrossed = detectCrossLine(gray, r, options.crossLineRatio);

        cv::Mat digitImg = gray(expanded).clone();
        QString text;
        double conf = 0.0;
        const int num = recognizeWithTesseract(
            digitImg, options.tesseractPath, text, conf,
            options.enablePaddleOcr);

        item.recognizedNumber = num;
        item.recognizedText = text;
        item.confidence = conf;

        writeDiag(QString::fromUtf8("  候选(%1,%2,%3x%4) 识别=%5 conf=%6 划线=%7")
                      .arg(r.x).arg(r.y).arg(r.width).arg(r.height)
                      .arg(num).arg(conf)
                      .arg(item.isCrossed ? QStringLiteral("是") : QStringLiteral("否")));

        allItems.push_back(item);
    }

    std::vector<PageNumberItem> candidates = allItems;

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    std::vector<PageNumberItem> pendingItems;

    for (auto &item : candidates) {
        // 1. 必须识别为 3 位数字
        const int textLen = item.recognizedText.length();
        if (item.recognizedNumber >= 0 && textLen != 3) {
            writeDiag(QString::fromUtf8("  → 识别=%1 位数=%2≠3，跳过")
                          .arg(item.recognizedNumber).arg(textLen));
            continue;
        }

        const bool matchesCorrect = fuzzyMatchPage(item.recognizedNumber, result.correctPage);
        const bool ocrFailed = (item.recognizedNumber < 0);

        // 2. 是正确页码，直接跳过保留
        if (matchesCorrect) {
            continue;
        }

        // 3. 识别失败，加入待确认
        if (ocrFailed) {
            ++result.pendingCount;
            pendingItems.push_back(item);
            writeDiag(QString::fromUtf8("  → OCR 失败，加待确认"));
            continue;
        }

        // 4. ★ 绝对安全机制：必须在数字中间检测到“手画线”，才自动涂白！
        //   如果没画线，哪怕认成了别的数字，也绝不自动删，统统扔进待确认中心！
        const double kMinConf = 0.5;
        if (item.isCrossed && item.confidence >= kMinConf) {
            eraseBlock(dst, gray, item.boundingBox, white);
            ++result.crossedRemoved;
            writeDiag(QString::fromUtf8("  → 识别=%1，划线=是，自动涂白")
                          .arg(item.recognizedNumber));
        } else {
            ++result.pendingCount;
            pendingItems.push_back(item);
            writeDiag(QString::fromUtf8("  → 识别=%1，划线=否，加待确认")
                          .arg(item.recognizedNumber));
        }
    }
            QStringLiteral("/Desktop/ocr_debug_dst.png");
        cv::imwrite(dbgPath.toStdString(), dst);
    }

    cv::Mat marked = src.clone();
    int idx = 0;
    for (const auto &item : candidates) {
        const bool matchesCorrect = fuzzyMatchPage(item.recognizedNumber,
                                                     result.correctPage);

        cv::Scalar color;
        if (matchesCorrect)      color = cv::Scalar(0, 255, 0);
        else if (item.isCrossed) color = cv::Scalar(0, 0, 255);
        else                     color = cv::Scalar(0, 165, 255);

        cv::rectangle(marked, item.boundingBox, color, 2);

        QString info = QString::fromUtf8("块%1:").arg(idx++);
        if (item.recognizedNumber >= 0) {
            info += QString::number(item.recognizedNumber);
        } else {
            info += item.recognizedText.left(25);
        }

        cv::putText(marked, info.toStdString(),
                    cv::Point(item.boundingBox.x,
                              item.boundingBox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 0, 255), 1);
    }

    result.image = dst;
    result.markedImage = marked;
    result.items = pendingItems;
    result.ok = true;

    if (candidates.empty()) {
        result.skipped = true;
    }

    diagFile.close();
    return result;
}

} // namespace process