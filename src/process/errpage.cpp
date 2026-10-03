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

        if (h < 15 || h > 250) continue;
        if (w < 8 || w > 350) continue;
        if (area < 40) continue;

        const double aspect = static_cast<double>(h) / std::max(1, w);
        if (aspect < 0.1) continue;
        if (aspect > 6.0) continue;

        const double fillRatio = static_cast<double>(area) /
                                  std::max(1, w * h);
        if (fillRatio > 0.7) continue;

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

// 合并阈值 0.25，避免 0009 和 005 被粘连
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
        const bool closeGap = gap >= 0 && gap < current.height * 0.25;

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

// 模糊匹配：识别值是否可能是正确页码
bool fuzzyMatchPage(int recognized, int correctPage)
{
    if (recognized < 0 || correctPage < 0) return false;
    if (recognized == correctPage) return true;

    const QString recStr = QString::number(recognized);
    const QString corrStr = QString::number(correctPage);

    // 长度差 > 1 -> 不匹配
    if (std::abs(recStr.length() - corrStr.length()) > 1) return false;

    if (corrStr.contains(recStr) || recStr.contains(corrStr)) return true;

    for (const QChar &c : recStr) {
        if (corrStr.contains(c)) return true;
    }
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

    cv::Mat dKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 1));
    cv::Mat hLinesDilated;
    cv::dilate(hLines, hLinesDilated, dKernel);

    cv::Mat result = gray.clone();
    result.setTo(255, hLinesDilated);
    return result;
}

std::vector<cv::Rect> splitByVerticalProjection(const cv::Mat &binary)
{
    std::vector<cv::Rect> boxes;
    if (binary.empty()) return boxes;

    cv::Mat proj;
    cv::reduce(binary, proj, 0, cv::REDUCE_SUM, CV_32S);

    int maxSum = 0;
    for (int x = 0; x < binary.cols; ++x) {
        const int s = proj.at<int>(0, x);
        if (s > maxSum) maxSum = s;
    }
    const int threshold = std::max(1, maxSum / 20);

    bool inChar = false;
    int start = 0;
    for (int x = 0; x < binary.cols; ++x) {
        const int colSum = proj.at<int>(0, x);
        if (!inChar && colSum > threshold) {
            inChar = true;
            start = x;
        } else if (inChar && colSum <= threshold) {
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

bool recognizeSingleChar(const cv::Mat &charImage,
                          const QString &tessExe,
                          const QString &tessdataDir,
                          int &outDigit)
{
    if (charImage.empty()) return false;

    cv::Mat gray;
    if (charImage.channels() == 3) {
        cv::cvtColor(charImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = charImage.clone();
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const double ratios[] = { 0.15, 0.25, 0.35 };

    for (double ratio : ratios) {
        const int vH = std::max(3, static_cast<int>(binary.rows * ratio));
        cv::Mat vKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(1, vH));
        cv::Mat repaired1;
        cv::morphologyEx(binary, repaired1, cv::MORPH_CLOSE, vKernel);

        cv::Mat k3 = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat repaired2;
        cv::morphologyEx(repaired1, repaired2, cv::MORPH_CLOSE, k3);

        cv::Mat out;
        cv::bitwise_not(repaired2, out);

        const QStringList modes = {
            QStringLiteral("10"),
            QStringLiteral("8"),
            QStringLiteral("7")
        };

        for (const QString &m : modes) {
            OcrAttempt r = tryRecognize(out, tessExe, tessdataDir, m);
            if (r.number >= 0 && r.text.length() == 1) {
                outDigit = r.number;
                return true;
            }
        }
    }
    return false;
}

} // namespace

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
    diag.setCodec("UTF-8");
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

    cv::Mat grayOrig;
    if (digitImage.channels() == 3) {
        cv::cvtColor(digitImage, grayOrig, cv::COLOR_BGR2GRAY);
    } else {
        grayOrig = digitImage.clone();
    }

    cv::Mat noLinesGray = removeHorizontalLinesByMask(grayOrig);

    cv::Mat cleaned;
    cv::threshold(noLinesGray, cleaned, 0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat cleanedBinary;
    cv::threshold(noLinesGray, cleanedBinary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    {
        OcrAttempt r = tryRecognize(digitImage, tessExe, tessdataDir,
                                     QStringLiteral("7"));
        writeDiag(QString::fromUtf8("原图+psm7：") +
                  (r.number >= 0 ? QString::fromUtf8("成功 %1").arg(r.number)
                                  : QString::fromUtf8("失败")));
        if (r.number >= 0) {
            outText = r.text;
            outConfidence = 0.95;
            writeDiag(QString::fromUtf8("→ 采用原图识别结果"));
            diagFile.close();
            return r.number;
        }
    }

    {
        const QString dbgPath = QDir::homePath() +
            QStringLiteral("/Desktop/ocr_debug_input.png");
        cv::imwrite(dbgPath.toStdString(), cleaned);
    }

    int bestNumber = -1;
    QString bestText;
    int bestDigits = 0;

    {
        const QStringList psmModes = {
            QStringLiteral("7"),
            QStringLiteral("8"),
            QStringLiteral("13"),
            QStringLiteral("6")
        };

        for (const QString &psm : psmModes) {
            OcrAttempt r = tryRecognize(cleaned, tessExe, tessdataDir, psm);
            if (r.number >= 0) {
                writeDiag(QString::fromUtf8("预处理+psm%1：成功 %2")
                              .arg(psm).arg(r.text));
                const int d = r.text.length();
                if (d > bestDigits) {
                    bestNumber = r.number;
                    bestText = r.text;
                    bestDigits = d;
                }
            } else {
                writeDiag(QString::fromUtf8("预处理+psm%1：失败").arg(psm));
            }
        }
    }

    {
        std::vector<cv::Rect> charBoxes = splitByVerticalProjection(cleanedBinary);
        const int rawCount = static_cast<int>(charBoxes.size());

        writeDiag(QString::fromUtf8("分割得到 %1 个块").arg(rawCount));

        {
            cv::Mat dbg;
            cv::cvtColor(cleaned, dbg, cv::COLOR_GRAY2BGR);
            for (size_t i = 0; i < charBoxes.size(); ++i) {
                cv::rectangle(dbg, charBoxes[i], cv::Scalar(0, 0, 255), 1);
            }
            const QString dbgPath = QDir::homePath() +
                QStringLiteral("/Desktop/ocr_debug_split.png");
            cv::imwrite(dbgPath.toStdString(), dbg);
        }

        if (charBoxes.size() >= 2 && charBoxes.size() <= 6) {
            QString combined;
            bool allOk = true;

            for (size_t i = 0; i < charBoxes.size(); ++i) {
                cv::Rect safe = charBoxes[i] &
                    cv::Rect(0, 0, cleaned.cols, cleaned.rows);
                if (safe.width <= 0 || safe.height <= 0) {
                    allOk = false;
                    break;
                }

                cv::Mat cbCleaned = cleaned(safe).clone();
                cv::Mat cbOrig = grayOrig(safe).clone();

                int digit = -1;
                bool ok = false;

                if (recognizeSingleChar(cbCleaned, tessExe, tessdataDir, digit)) {
                    ok = true;
                    writeDiag(QString::fromUtf8("  字符%1：成功 %2（涂白图）")
                                  .arg(static_cast<int>(i)).arg(digit));
                }
                else if (recognizeSingleChar(cbOrig, tessExe, tessdataDir, digit)) {
                    ok = true;
                    writeDiag(QString::fromUtf8("  字符%1：成功 %2（原图）")
                                  .arg(static_cast<int>(i)).arg(digit));
                }

                if (ok) {
                    combined += QString::number(digit);
                } else {
                    allOk = false;
                    writeDiag(QString::fromUtf8("  字符%1：失败")
                                  .arg(static_cast<int>(i)));
                    break;
                }
            }

            if (allOk && !combined.isEmpty()) {
                const int d = combined.length();
                if (d > bestDigits) {
                    bool ok = false;
                    const int num = combined.toInt(&ok);
                    if (ok) {
                        bestNumber = num;
                        bestText = combined;
                        bestDigits = d;
                        writeDiag(QString::fromUtf8(
                            "→ 分割识别胜出：%1").arg(combined));
                    }
                }
            }
        }
    }

    if (bestNumber >= 0) {
        outText = bestText;
        outConfidence = 0.9;
        writeDiag(QString::fromUtf8("最终采用：%1（位数 %2）")
                      .arg(bestText).arg(bestDigits));
        diagFile.close();
        return bestNumber;
    }

    writeDiag(QString::fromUtf8("所有方案均失败"));
    outText = QString::fromUtf8("无数字");
    diagFile.close();
    return -1;
}

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

    {
        cv::Mat dbg;
        cv::cvtColor(roi, dbg, cv::COLOR_GRAY2BGR);
        for (const auto &b : digitBoxes) {
            cv::rectangle(dbg, b, cv::Scalar(255, 0, 0), 1);
        }
        for (const auto &b : pageBoxes) {
            cv::rectangle(dbg, b, cv::Scalar(0, 0, 255), 1);
        }
        static int regionIdx = 0;
        const int idxMod = regionIdx % 4;
        ++regionIdx;
        const QString dbgPath = QDir::homePath() +
            QStringLiteral("/Desktop/ocr_debug_candidates_%1.png").arg(idxMod);
        cv::imwrite(dbgPath.toStdString(), dbg);
    }

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

// 划线检测：只往上扩 5 像素（横线在数字上方）
bool ErrPage::detectCrossLine(const cv::Mat &gray,
                               const cv::Rect &digitBox,
                               double crossLineRatio)
{
    cv::Rect expanded = digitBox;
    expanded.x -= 3;
    expanded.width += 6;
    expanded.y -= 5;
    expanded.height += 5;
    expanded &= cv::Rect(0, 0, gray.cols, gray.rows);

    if (expanded.width <= 0 || expanded.height <= 0) return false;

    cv::Mat roi = gray(expanded);
    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int yStart = 3;
    const int yEnd = binary.rows - 2;
    if (yEnd <= yStart) return false;

    double ratio = 0.7;
    if (crossLineRatio > 0.1 && crossLineRatio < 0.95) {
        ratio = crossLineRatio;
    }
    const int minSpan = static_cast<int>(binary.cols * ratio);

    for (int y = yStart; y < yEnd; ++y) {
        const uchar *row = binary.ptr<uchar>(y);
        int firstX = -1;
        int lastX = -1;
        int count = 0;
        for (int x = 0; x < binary.cols; ++x) {
            if (row[x] > 0) {
                if (firstX < 0) firstX = x;
                lastX = x;
                ++count;
            }
        }
        if (firstX < 0) continue;
        const int span = lastX - firstX + 1;
        if (span >= minSpan && count >= span / 4) {
            return true;
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

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    std::vector<PageNumberItem> allItems;

    const int minRegionH = 600;

    if (options.detectTopLeft) {
        const int w = static_cast<int>(W * options.topLeftWidthRatio);
        int h = static_cast<int>(H * options.topLeftHeightRatio);
        if (h < minRegionH) h = std::min(minRegionH, H);
        cv::Rect region(0, 0, w, h);
        detectDigitsInRegion(gray, region, options, allItems);
    }

    if (options.detectTopRight) {
        const int w = static_cast<int>(W * options.topRightWidthRatio);
        int h = static_cast<int>(H * options.topRightHeightRatio);
        if (h < minRegionH) h = std::min(minRegionH, H);
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

    const double CORNER_RATIO = 0.25;
    const double MIN_H_RATIO  = 0.013;

    std::vector<PageNumberItem> candidates;
    candidates.reserve(allItems.size());

    for (const auto &it : allItems) {
        const cv::Rect &b = it.boundingBox;
        const double cx = b.x + b.width / 2.0;
        const double cy = b.y + b.height / 2.0;

        const bool inLeft   = (cx < W * CORNER_RATIO);
        const bool inRight  = (cx > W * (1.0 - CORNER_RATIO));
        const bool inTop    = (cy < H * CORNER_RATIO);
        const bool inBottom = (cy > H * (1.0 - CORNER_RATIO));

        const bool inTopLeft     = (inLeft  && inTop);
        const bool inTopRight    = (inRight && inTop);
        const bool inBottomRight = (inRight && inBottom);

        const bool inCorner = (inTopLeft || inTopRight || inBottomRight);
        const bool bigEnough = (b.height >= H * MIN_H_RATIO);
        const bool keep = (inCorner && bigEnough);

        writeDiag(QString::fromUtf8("  [过滤] bbox=(%1,%2,%3x%4) 中心=(%5,%6) "
                                     "角=%7 大=%8 → %9")
                      .arg(b.x).arg(b.y).arg(b.width).arg(b.height)
                      .arg(static_cast<int>(cx)).arg(static_cast<int>(cy))
                      .arg(inCorner ? QStringLiteral("是") : QStringLiteral("否"))
                      .arg(bigEnough ? QStringLiteral("是") : QStringLiteral("否"))
                      .arg(keep ? QStringLiteral("保留") : QStringLiteral("丢弃")));

        if (keep) candidates.push_back(it);
    }

    writeDiag(QString::fromUtf8("过滤后剩余 %1 个候选块")
                  .arg(static_cast<int>(candidates.size())));

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto &it = candidates[i];
        writeDiag(QString::fromUtf8("  候选%1：识别=%2 划线=%3 bbox=(%4,%5,%6x%7)")
                      .arg(static_cast<int>(i))
                      .arg(it.recognizedNumber)
                      .arg(it.isCrossed ? QStringLiteral("是") : QStringLiteral("否"))
                      .arg(it.boundingBox.x).arg(it.boundingBox.y)
                      .arg(it.boundingBox.width).arg(it.boundingBox.height));
    }

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    // 只按 isCrossed 涂白，不再依赖 hasCorrectMatch
    for (auto &item : candidates) {
        const bool matchesCorrect = fuzzyMatchPage(item.recognizedNumber,
                                                     result.correctPage);

        if (matchesCorrect) {
            writeDiag(QString::fromUtf8("  → 识别=%1 匹配正确页码，保留")
                          .arg(item.recognizedNumber));
            continue;
        }

        if (item.recognizedNumber < 0) {
            ++result.pendingCount;
            writeDiag(QString::fromUtf8("  → OCR 失败，加待确认"));
            continue;
        }

        if (item.isCrossed) {
            cv::Rect r = item.boundingBox;
            r.x -= 8;
            r.y -= 8;
            r.width += 16;
            r.height += 16;
            r &= cv::Rect(0, 0, W, H);
            cv::rectangle(dst, r, white, cv::FILLED);
            ++result.crossedRemoved;
            writeDiag(QString::fromUtf8("  → 识别=%1，划线=是，涂白")
                          .arg(item.recognizedNumber));
        } else {
            ++result.pendingCount;
            writeDiag(QString::fromUtf8("  → 识别=%1，划线=否，加待确认")
                          .arg(item.recognizedNumber));
        }
    }

    {
        const QString dbgPath = QDir::homePath() +
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
    result.items = candidates;
    result.ok = true;

    if (candidates.empty()) {
        result.skipped = true;
    }

    diagFile.close();
    return result;
}

} // namespace process