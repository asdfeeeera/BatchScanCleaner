#include "errpage.h"

#include <opencv2/core.hpp>
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
#include <vector>

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
        if (c.isDigit()) digitsOnly += c;
    }

    if (digitsOnly.isEmpty()) return -1;

    bool ok = false;
    const int page = digitsOnly.toInt(&ok);
    if (ok && page >= 0) return page;
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

// ============================================================
// ★ 核心：OCR 前的图像预处理
// 1. 二值化（黑字白底 → 白字黑底）
// 2. 水平开运算提取横线
// 3. 减去横线
// 4. 3x3 小闭运算修复细小缺口
// 5. 1x7 垂直闭运算，把被横线切开的数字重新拼起来
// 6. 反色回白底黑字，给 Tesseract
// ============================================================
cv::Mat preprocessForOcr(const cv::Mat &digitImage)
{
    if (digitImage.empty()) return digitImage;

    cv::Mat gray;
    if (digitImage.channels() == 3) {
        cv::cvtColor(digitImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = digitImage.clone();
    }

    // 二值化：白字黑底
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 提取长横线：核宽度用图宽的 0.35 倍
    const int kernelW = std::max(10, static_cast<int>(gray.cols * 0.35));
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelW, 1));
    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    // 从二值图中减去横线
    cv::Mat noLines;
    cv::subtract(binary, hLines, noLines);

    // 修复 1：小的全方向闭运算，修复细小缺口
    cv::Mat kernel3 = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat repaired1;
    cv::morphologyEx(noLines, repaired1, cv::MORPH_CLOSE, kernel3);

    // 修复 2：垂直闭运算，连接被横线切开的上下部分（核高 7，比较温和）
    cv::Mat vKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(1, 7));
    cv::Mat repaired2;
    cv::morphologyEx(repaired1, repaired2, cv::MORPH_CLOSE, vKernel);

    // 反色回白底黑字
    cv::Mat result;
    cv::bitwise_not(repaired2, result);

    return result;
}

// ============================================================
// ★ 新增：用垂直投影把数字块切成单字符
// 输入是白底黑字的图像，内部会反色成白字黑底来做投影
// 返回的矩形基于原图尺寸，可以直接在原图上裁剪
// ============================================================
std::vector<cv::Rect> splitByVerticalProjection(const cv::Mat &image)
{
    std::vector<cv::Rect> boxes;
    if (image.empty()) return boxes;

    // 转成灰度
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // 反色成白字黑底
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 垂直投影：每列的像素和
    cv::Mat proj;
    cv::reduce(binary, proj, 0, cv::REDUCE_SUM, CV_32S);

    bool inChar = false;
    int start = 0;
    for (int x = 0; x < binary.cols; ++x) {
        int colSum = proj.at<int>(0, x);
        if (!inChar && colSum > 0) {
            inChar = true;
            start = x;
        } else if (inChar && colSum == 0) {
            inChar = false;
            if (x - start >= 3) {
                boxes.push_back(cv::Rect(start, 0, x - start, binary.rows));
            }
        }
    }
    if (inChar && binary.cols - start >= 3) {
        boxes.push_back(cv::Rect(start, 0, binary.cols - start, binary.rows));
    }

    return boxes;
}

// ============================================================
// 用 Tesseract 识别（单次）
// ============================================================
struct OcrAttempt
{
    int number = -1;
    QString text;
    QString psmUsed;
};

OcrAttempt tryRecognize(const cv::Mat &image, const QString &tessExe,
                         const QString &tessdataDir, const QString &psmMode)
{
    OcrAttempt result;

    QTemporaryDir tempDir;
    if (!tempDir.isValid()) return result;

    const QString tmpPng = tempDir.path() + QStringLiteral("/digit.png");
    const QString tmpOutBase = tempDir.path() + QStringLiteral("/out");

    // 放大 4 倍
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
    result.psmUsed = psmMode;
    return result;
}

} // namespace

// ============================================================
// Tesseract OCR 识别（多次尝试）
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

    const QString tessDir = QFileInfo(tessExe).absolutePath();
    const QString tessdataDir = tessDir + QStringLiteral("/tessdata");

    // 方案 1：先按原图直接识别
    {
        OcrAttempt r = tryRecognize(digitImage, tessExe, tessdataDir,
                                     QStringLiteral("7"));
        writeDiag(QString::fromUtf8("原图+psm7：") +
                  (r.number >= 0 ? QString::fromUtf8("成功 %1").arg(r.number)
                                  : QString::fromUtf8("失败")));
        if (r.number >= 0) {
            outText = r.text;
            outConfidence = 0.9;
            writeDiag(QString::fromUtf8("→ 采用原图识别结果"));
            diagFile.close();
            return r.number;
        }
    }

    // 方案 2：预处理后识别
    cv::Mat cleaned = preprocessForOcr(digitImage);

    const QStringList psmModes = {
        QStringLiteral("7"),
        QStringLiteral("8"),
        QStringLiteral("13"),
        QStringLiteral("6")
    };

    for (const QString &psm : psmModes) {
        OcrAttempt r = tryRecognize(cleaned, tessExe, tessdataDir, psm);
        if (r.number >= 0) {
            outText = r.text;
            outConfidence = 0.9;
            writeDiag(QString::fromUtf8("预处理+psm%1：成功 %2（采用）")
                          .arg(psm).arg(r.number));
            diagFile.close();
            return r.number;
        } else {
            writeDiag(QString::fromUtf8("预处理+psm%1：失败").arg(psm));
        }
    }

    // ★ 方案 3：垂直分割后逐个字符识别
    {
        std::vector<cv::Rect> charBoxes = splitByVerticalProjection(cleaned);
        writeDiag(QString::fromUtf8("垂直分割得到 %1 个字符块")
                      .arg(static_cast<int>(charBoxes.size())));

        if (charBoxes.size() >= 2 && charBoxes.size() <= 6) {
            QString combined;
            bool allOk = true;

            for (size_t i = 0; i < charBoxes.size(); ++i) {
                const cv::Rect &cb = charBoxes[i];
                if (cb.width <= 0 || cb.height <= 0) { allOk = false; break; }
                cv::Mat charImg = cleaned(cb).clone();

                OcrAttempt cr = tryRecognize(charImg, tessExe, tessdataDir,
                                              QStringLiteral("8"));
                if (cr.number < 0) {
                    cr = tryRecognize(charImg, tessExe, tessdataDir,
                                       QStringLiteral("7"));
                }

                if (cr.number >= 0) {
                    combined += QString::number(cr.number);
                    writeDiag(QString::fromUtf8("  字符%1：成功 %2")
                                  .arg(static_cast<int>(i))
                                  .arg(cr.number));
                } else {
                    allOk = false;
                    writeDiag(QString::fromUtf8("  字符%1：失败")
                                  .arg(static_cast<int>(i)));
                    break;
                }
            }

            if (allOk && !combined.isEmpty()) {
                bool ok = false;
                const int num = combined.toInt(&ok);
                if (ok) {
                    outText = combined;
                    outConfidence = 0.85;
                    writeDiag(QString::fromUtf8("垂直分割识别成功：%1（采用）")
                                  .arg(combined));
                    diagFile.close();
                    return num;
                }
            }
        }
    }

    writeDiag(QString::fromUtf8("所有方案均失败"));
    outText = QString::fromUtf8("无数字");
    diagFile.close();
    return -1;
}

// ============================================================
// 在指定区域检测数字块
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
        cv::Rect expandedBox = box;
        expandedBox.y -= 8;
        expandedBox.height += 16;
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

            QString text;
            double conf = 0.0;
            const int num = recognizeWithTesseract(
                digitImg, options.tesseractPath, text, conf);

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

} // namespace processs