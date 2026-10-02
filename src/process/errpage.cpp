#include "errpage.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QFileInfo>
#include <QRegularExpression>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QImage>
#include <QProcess>
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QStandardPaths>
#include <algorithm>
#include <cmath>

namespace process {

// ============================================================
// 从文件名解析正确页码
// ============================================================
int ErrPage::parseCorrectPage(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QString baseName = info.completeBaseName().trimmed();

    QString digitsOnly;
    for (const QChar &c : baseName) {
        if (c.isDigit()) {
            digitsOnly += c;
        }
    }

    if (digitsOnly.isEmpty()) return -1;

    bool ok = false;
    const int page = digitsOnly.toInt(&ok);
    if (ok && page >= 0) {
        return page;
    }
    return -1;
}

// ============================================================
// 匿名命名空间
// ============================================================
namespace {

std::vector<cv::Rect> findDigitBoxes(const cv::Mat &binary,
                                      const ErrPageOptions &options)
{
    std::vector<cv::Rect> result;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
    cv::Mat dilated;
    cv::dilate(binary, dilated, kernel);

    cv::Mat labels, stats, centroids;
    const int nLabels = cv::connectedComponentsWithStats(
        dilated, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (h < options.minDigitHeight || h > options.maxDigitHeight) continue;
        if (w < options.minDigitWidth || w > options.maxDigitWidth) continue;
        if (area < 50) continue;

        const double aspect = static_cast<double>(h) / std::max(1, w);
        if (aspect < 0.15) continue;
        if (aspect > 4.0) continue;

        cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                   stats.at<int>(i, cv::CC_STAT_TOP),
                   w, h);
        result.push_back(r);
    }
    return result;
}

double verticalOverlap(const cv::Rect &a, const cv::Rect &b)
{
    const int top = std::max(a.y, b.y);
    const int bot = std::min(a.y + a.height, b.y + b.height);
    if (bot <= top) return 0.0;
    const int minH = std::min(a.height, b.height);
    if (minH <= 0) return 0.0;
    return static_cast<double>(bot - top) / minH;
}

std::vector<cv::Rect> mergeAdjacentDigits(std::vector<cv::Rect> boxes)
{
    if (boxes.size() < 2) return boxes;

    std::sort(boxes.begin(), boxes.end(),
              [](const cv::Rect &a, const cv::Rect &b) {
                  if (std::abs(a.y - b.y) < a.height / 2) return a.x < b.x;
                  return a.y < b.y;
              });

    std::vector<cv::Rect> merged;
    cv::Rect current = boxes[0];

    for (size_t i = 1; i < boxes.size(); ++i) {
        const cv::Rect &next = boxes[i];

        const double vOverlap = verticalOverlap(current, next);
        const int gap = next.x - (current.x + current.width);
        const bool closeGap = gap >= 0 && gap < current.height * 0.8;

        if (vOverlap >= 0.6 && closeGap) {
            const int x1 = std::min(current.x, next.x);
            const int y1 = std::min(current.y, next.y);
            const int x2 = std::max(current.x + current.width,
                                     next.x + next.width);
            const int y2 = std::max(current.y + current.height,
                                     next.y + next.height);
            current = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        } else {
            merged.push_back(current);
            current = next;
        }
    }
    merged.push_back(current);
    return merged;
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

bool isPageMatch(int recognized, int correctPage)
{
    if (recognized < 0 || correctPage < 0) return false;
    return recognized == correctPage;
}

// ★ 新增：去掉数字块内的长横线，只保留数字
// 输入：digitImage（灰度图，黑字白底或黑底白字）
// 输出：处理后的图，横线被去掉
cv::Mat removeHorizontalLines(const cv::Mat &digitImage)
{
    // 二值化：文字/线条为白前景
    cv::Mat binary;
    cv::threshold(digitImage, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 用长横核开运算提取横线
    const int kernelW = std::max(15, static_cast<int>(digitImage.cols * 0.7));
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelW, 1));

    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    // 从二值图中减去横线
    cv::Mat digitsOnly;
    cv::subtract(binary, hLines, digitsOnly);

    // 反色回白底黑字（Tesseract 更容易识别）
    cv::Mat result;
    cv::bitwise_not(digitsOnly, result);

    return result;
}

} // namespace

// ============================================================
// Tesseract OCR 识别（含诊断）
// ============================================================
int ErrPage::recognizeWithTesseract(const cv::Mat &digitImage,
                                     const QString &tesseractPath,
                                     QString &outText,
                                     double &outConfidence)
{
    outText.clear();
    outConfidence = 0.0;

    const QString diagPath = QDir::homePath() +
                             QStringLiteral("/Desktop/ocr_debug.txt");
    QFile diagFile(diagPath);
    diagFile.open(QIODevice::Append | QIODevice::Text);
    QTextStream diag(&diagFile);

    auto writeDiag = [&](const QString &msg) {
        if (diagFile.isOpen()) {
            diag << msg << "\n";
            diag.flush();
        }
    };

    static int callIndex = 0;
    const int myIndex = ++callIndex;
    writeDiag(QString::fromUtf8("========== 第 %1 次调用 ==========").arg(myIndex));

    if (digitImage.empty()) {
        writeDiag(QString::fromUtf8("错误：输入图像为空"));
        outText = QString::fromUtf8("空图");
        diagFile.close();
        return -1;
    }
    writeDiag(QString::fromUtf8("输入尺寸：%1 x %2，通道 %3")
                  .arg(digitImage.cols).arg(digitImage.rows).arg(digitImage.channels()));

    const QString tessExe = locateTesseract(tesseractPath);
    if (tessExe.isEmpty()) {
        writeDiag(QString::fromUtf8("错误：未找到 tesseract.exe"));
        outText = QString::fromUtf8("未找到tesseract");
        diagFile.close();
        return -1;
    }
    writeDiag(QString::fromUtf8("tesseract 路径：") + tessExe);

    QTemporaryDir tempDir;
    if (!tempDir.isValid()) {
        writeDiag(QString::fromUtf8("错误：临时目录创建失败"));
        outText = QString::fromUtf8("临时目录失败");
        diagFile.close();
        return -1;
    }

    const QString tmpPng = tempDir.path() + QStringLiteral("/digit.png");
    const QString tmpOutBase = tempDir.path() + QStringLiteral("/out");

    // 放大 3 倍
    cv::Mat enlarged;
    cv::resize(digitImage, enlarged, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);

    // 用 QImage 保存
    QImage qimg(enlarged.data, enlarged.cols, enlarged.rows,
                static_cast<int>(enlarged.step),
                (enlarged.channels() == 1) ? QImage::Format_Grayscale8
                                            : QImage::Format_RGB888);
    QImage copy = qimg.copy();
    if (!copy.save(tmpPng, "PNG")) {
        writeDiag(QString::fromUtf8("错误：PNG 保存失败（Qt）"));
        outText = QString::fromUtf8("PNG保存失败");
        diagFile.close();
        return -1;
    }

    const QString tessDir = QFileInfo(tessExe).absolutePath();
    const QString tessdataDir = tessDir + QStringLiteral("/tessdata");

    QProcess proc;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("TESSDATA_PREFIX"), tessdataDir);
    proc.setProcessEnvironment(env);
    proc.setWorkingDirectory(tessDir);

    QStringList args;
    args << tmpPng << tmpOutBase
         << QStringLiteral("-l") << QStringLiteral("eng")
         << QStringLiteral("--psm") << QStringLiteral("7")
         << QStringLiteral("-c")
         << QStringLiteral("tessedit_char_whitelist=0123456789");

    proc.start(tessExe, args);
    if (!proc.waitForStarted(5000)) {
        writeDiag(QString::fromUtf8("错误：进程启动失败"));
        outText = QString::fromUtf8("启动失败");
        diagFile.close();
        return -1;
    }
    if (!proc.waitForFinished(10000)) {
        proc.kill();
        writeDiag(QString::fromUtf8("错误：超时"));
        outText = QString::fromUtf8("超时");
        diagFile.close();
        return -1;
    }

    const int exitCode = proc.exitCode();
    const QString stdErr = QString::fromUtf8(proc.readAllStandardError());
    writeDiag(QString::fromUtf8("退出码：%1").arg(exitCode));

    if (exitCode != 0) {
        outText = QString::fromUtf8("码%1:%2").arg(exitCode).arg(stdErr.left(30));
        diagFile.close();
        return -1;
    }

    QFile outFile(tmpOutBase + QStringLiteral(".txt"));
    if (!outFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        writeDiag(QString::fromUtf8("错误：输出文件打不开"));
        outText = QString::fromUtf8("输出打不开");
        diagFile.close();
        return -1;
    }
    const QString rawText = QString::fromUtf8(outFile.readAll()).trimmed();
    outFile.close();
    writeDiag(QString::fromUtf8("原始文本：[") + rawText + QString::fromUtf8("]"));

    QString digitsOnly;
    for (const QChar &c : rawText) {
        if (c.isDigit()) digitsOnly += c;
    }

    if (digitsOnly.isEmpty()) {
        writeDiag(QString::fromUtf8("结果：无数字"));
        outText = QString::fromUtf8("无数字");
        diagFile.close();
        return -1;
    }

    writeDiag(QString::fromUtf8("提取数字：") + digitsOnly);

    outText = digitsOnly;
    bool ok = false;
    const int num = outText.toInt(&ok);
    if (!ok) {
        writeDiag(QString::fromUtf8("错误：转换为 int 失败"));
        outText = QString::fromUtf8("转换失败");
        diagFile.close();
        return -1;
    }

    writeDiag(QString::fromUtf8("成功识别：%1").arg(num));
    diagFile.close();

    outConfidence = 0.9;
    return num;
}

// ============================================================
// 在指定区域检测数字块（★ 关键改动：OCR 前先去横线）
// ============================================================
void ErrPage::detectDigitsInRegion(const cv::Mat &gray,
                                    const cv::Rect &region,
                                    const ErrPageOptions &options,
                                    std::vector<PageNumberItem> &outItems)
{
    cv::Rect r = region & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width <= 0 || r.height <= 0) return;

    cv::Mat roi = gray(r);

    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::vector<cv::Rect> digitBoxes = findDigitBoxes(binary, options);
    std::vector<cv::Rect> pageBoxes = mergeAdjacentDigits(digitBoxes);

    for (const auto &box : pageBoxes) {
        // 上边扩大一点，把划线也含进来
        cv::Rect expandedBox = box;
        expandedBox.y -= 5;
        expandedBox.height += 10;
        expandedBox &= cv::Rect(0, 0, roi.cols, roi.rows);

        const cv::Rect globalBox(expandedBox.x + r.x, expandedBox.y + r.y,
                                  expandedBox.width, expandedBox.height);

        PageNumberItem item;
        item.boundingBox = cv::Rect(box.x + r.x, box.y + r.y,
                                     box.width, box.height);
        item.isCrossed = detectCrossLine(gray, item.boundingBox,
                                          options.crossLineRatio);

        cv::Rect safeBox = expandedBox & cv::Rect(0, 0, roi.cols, roi.rows);
        if (safeBox.width > 0 && safeBox.height > 0) {
            cv::Mat digitImg = roi(safeBox).clone();

            // ★ 关键：OCR 前先把横线去掉
            cv::Mat cleaned = removeHorizontalLines(digitImg);

            QString text;
            double conf = 0.0;
            const int num = recognizeWithTesseract(
                cleaned, options.tesseractPath, text, conf);

            item.recognizedNumber = num;
            item.recognizedText = text;
            item.confidence = conf;
        }

        outItems.push_back(item);
    }
}

// ============================================================
// 划线检测
// ============================================================
bool ErrPage::detectCrossLine(const cv::Mat &gray,
                               const cv::Rect &digitBox,
                               double crossLineRatio)
{
    cv::Rect expanded = digitBox;
    expanded.x -= 8;
    expanded.y -= 15;
    expanded.width += 16;
    expanded.height += 30;
    expanded &= cv::Rect(0, 0, gray.cols, gray.rows);

    if (expanded.width <= 0 || expanded.height <= 0) return false;

    cv::Mat roi = gray(expanded);
    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int kernelWidth = std::max(15, static_cast<int>(digitBox.width * 0.7));
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelWidth, 1));

    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    cv::Mat labels, stats, centroids;
    const int nLabels = cv::connectedComponentsWithStats(
        hLines, labels, stats, centroids, 8, CV_32S);

    int longLineCount = 0;
    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (w >= static_cast<int>(digitBox.width * 0.6) && h <= 6 && area >= 20) {
            ++longLineCount;
        }
    }

    return longLineCount >= 1;
}

// ============================================================
// 主流程
// ============================================================
ErrPageResult ErrPage::process(const cv::Mat &src,
                                const QString &sourcePath,
                                const ErrPageOptions &options)
{
    ErrPageResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    result.correctPage = parseCorrectPage(sourcePath);

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    std::vector<PageNumberItem> allItems;

    if (options.detectTopLeft) {
        const int w = static_cast<int>(W * options.topLeftWidthRatio);
        const int h = static_cast<int>(H * options.topLeftHeightRatio);
        cv::Rect region(0, 0, w, h);
        detectDigitsInRegion(gray, region, options, allItems);
    }

    if (options.detectTopRight) {
        const int w = static_cast<int>(W * options.topRightWidthRatio);
        const int h = static_cast<int>(H * options.topRightHeightRatio);
        cv::Rect region(W - w, 0, w, h);
        detectDigitsInRegion(gray, region, options, allItems);
    }

    if (options.detectBottomRight) {
        const int w = static_cast<int>(W * options.bottomRightWidthRatio);
        const int h = static_cast<int>(H * options.bottomRightHeightRatio);
        cv::Rect region(W - w, H - h, w, h);
        detectDigitsInRegion(gray, region, options, allItems);
    }

    if (options.detectBottomLeft) {
        const int w = static_cast<int>(W * options.bottomLeftWidthRatio);
        const int h = static_cast<int>(H * options.bottomLeftHeightRatio);
        cv::Rect region(0, H - h, w, h);
        detectDigitsInRegion(gray, region, options, allItems);
    }

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    for (auto &item : allItems) {
        const bool ocrFailed = (item.recognizedNumber < 0);
        const bool matchesCorrect = isPageMatch(item.recognizedNumber,
                                                 result.correctPage);

        if (matchesCorrect) continue;
        if (ocrFailed) {
            ++result.pendingCount;
            continue;
        }

        if (item.isCrossed) {
            cv::Rect r = item.boundingBox;
            r.x -= 3;
            r.y -= 3;
            r.width += 6;
            r.height += 6;
            r &= cv::Rect(0, 0, W, H);
            cv::rectangle(dst, r, white, cv::FILLED);
            ++result.crossedRemoved;
        } else {
            ++result.pendingCount;
        }
    }

    // 标记图
    cv::Mat marked = src.clone();
    int idx = 0;
    for (const auto &item : allItems) {
        const bool ocrFailed = (item.recognizedNumber < 0);
        const bool matchesCorrect = isPageMatch(item.recognizedNumber,
                                                 result.correctPage);

        cv::Scalar color;
        if (matchesCorrect)      color = cv::Scalar(0, 255, 0);
        else if (ocrFailed)      color = cv::Scalar(255, 0, 255);
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
    result.items = allItems;
    result.ok = true;

    if (allItems.empty()) {
        result.skipped = true;
    }
    return result;
}

} // namespace process