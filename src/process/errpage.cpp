#include "errpage.h"

#include <opencv2/imgproc.hpp>
#include <QFileInfo>
#include <QRegularExpression>
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QStandardPaths>
#include <algorithm>
#include <cmath>

namespace process {

int ErrPage::parseCorrectPage(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QString baseName = info.completeBaseName();

    static const QRegularExpression re(QStringLiteral("^\\d+$"));
    if (re.match(baseName).hasMatch()) {
        bool ok = false;
        const int page = baseName.toInt(&ok);
        if (ok && page >= 0) {
            return page;
        }
    }
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

        if (h < options.minDigitHeight || h > options.maxDigitHeight) continue;
        if (w < options.minDigitWidth || w > options.maxDigitWidth) continue;
        if (area < 30) continue;

        const double aspect = static_cast<double>(h) / std::max(1, w);
        if (aspect < 0.5) continue;

        cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                   stats.at<int>(i, cv::CC_STAT_TOP),
                   w, h);
        result.push_back(r);
    }
    return result;
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

        const bool sameRow = std::abs(current.y - next.y) < current.height / 2;
        const int gap = next.x - (current.x + current.width);
        const bool closeGap = gap >= 0 && gap < current.height * 2;

        if (sameRow && closeGap) {
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

// 找到 tesseract.exe：优先用 options.tesseractPath，
// 否则查找 exe 同目录下的 tesseract\tesseract.exe
QString locateTesseract(const QString &hint)
{
    if (!hint.isEmpty() && QFile::exists(hint)) {
        return hint;
    }

    const QString appDir = QCoreApplication::applicationDirPath();

    // 候选 1：<appDir>/tesseract/tesseract.exe
    const QString candidate1 = appDir + QStringLiteral("/tesseract/tesseract.exe");
    if (QFile::exists(candidate1)) return candidate1;

    // 候选 2：<appDir>/tesseract.exe
    const QString candidate2 = appDir + QStringLiteral("/tesseract.exe");
    if (QFile::exists(candidate2)) return candidate2;

    // 候选 3：系统 PATH
    const QString inPath = QStandardPaths::findExecutable(QStringLiteral("tesseract"));
    if (!inPath.isEmpty()) return inPath;

    return QString();
}

} // namespace

int ErrPage::recognizeWithTesseract(const cv::Mat &digitImage,
                                     const QString &tesseractPath,
                                     QString &outText,
                                     double &outConfidence)
{
    outText.clear();
    outConfidence = 0.0;

    if (digitImage.empty()) return -1;

    const QString tessExe = locateTesseract(tesseractPath);
    if (tessExe.isEmpty()) {
        return -1;
    }

    // 创建临时目录
    QTemporaryDir tempDir;
    if (!tempDir.isValid()) return -1;

    const QString tmpPng = tempDir.path() + QStringLiteral("/digit.png");
    const QString tmpOutBase = tempDir.path() + QStringLiteral("/out");

    // 保存数字块（放大 3 倍提高 OCR 准确率）
    cv::Mat enlarged;
    cv::resize(digitImage, enlarged, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);
    if (!cv::imwrite(tmpPng.toStdString(), enlarged)) {
        return -1;
    }

    // 设置 tessdata 路径为 exe 同目录
    const QString tessDir = QFileInfo(tessExe).absolutePath();
    const QString tessdataDir = tessDir + QStringLiteral("/tessdata");

    QProcess proc;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("TESSDATA_PREFIX"), tessdataDir);
    proc.setProcessEnvironment(env);
    proc.setWorkingDirectory(tessDir);

    QStringList args;
    args << tmpPng
         << tmpOutBase
         << QStringLiteral("-l") << QStringLiteral("eng")
         << QStringLiteral("--psm") << QStringLiteral("7")  // 单行文本
         << QStringLiteral("-c")
         << QStringLiteral("tessedit_char_whitelist=0123456789");

    proc.start(tessExe, args);
    if (!proc.waitForStarted(5000)) {
        return -1;
    }
    if (!proc.waitForFinished(10000)) {
        proc.kill();
        return -1;
    }

    if (proc.exitCode() != 0) {
        return -1;
    }

    // 读取输出文本
    QFile outFile(tmpOutBase + QStringLiteral(".txt"));
    if (!outFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return -1;
    }
    outText = QString::fromUtf8(outFile.readAll()).trimmed();
    outFile.close();

    // 只保留数字
    QString digitsOnly;
    for (const QChar &c : outText) {
        if (c.isDigit()) digitsOnly += c;
    }
    outText = digitsOnly;

    if (outText.isEmpty()) return -1;

    bool ok = false;
    const int num = outText.toInt(&ok);
    if (!ok) return -1;

    outConfidence = 0.9;   // Tesseract 不输出置信度时给个默认值
    return num;
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

    for (const auto &box : pageBoxes) {
        const cv::Rect globalBox(box.x + r.x, box.y + r.y,
                                  box.width, box.height);

        PageNumberItem item;
        item.boundingBox = globalBox;
        item.isCrossed = detectCrossLine(gray, globalBox, options.crossLineRatio);

        // 提取数字块图像，供 OCR 识别
        cv::Rect safeBox = globalBox & cv::Rect(0, 0, gray.cols, gray.rows);
        if (safeBox.width > 0 && safeBox.height > 0) {
            cv::Mat digitImg = gray(safeBox).clone();

            // 反色 + 二值化，让 Tesseract 更容易识别（白底黑字）
            cv::Mat digitProc;
            cv::threshold(digitImg, digitProc, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);

            QString text;
            double conf = 0.0;
            const int num = recognizeWithTesseract(
                digitProc, options.tesseractPath, text, conf);

            item.recognizedNumber = num;
            item.recognizedText = text;
            item.confidence = conf;
        }

        outItems.push_back(item);
    }
}

bool ErrPage::detectCrossLine(const cv::Mat &gray,
                               const cv::Rect &digitBox,
                               double crossLineRatio)
{
    cv::Rect expanded = digitBox;
    expanded.x -= 5;
    expanded.y -= 5;
    expanded.width += 10;
    expanded.height += 10;
    expanded &= cv::Rect(0, 0, gray.cols, gray.rows);

    if (expanded.width <= 0 || expanded.height <= 0) return false;

    cv::Mat roi = gray(expanded);

    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int kernelWidth = std::max(15, expanded.width * 2 / 3);
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelWidth, 1));

    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    int maxRowCount = 0;
    const int binaryRowMax = expanded.width;
    for (int y = 0; y < hLines.rows; ++y) {
        const uchar *row = hLines.ptr<uchar>(y);
        int count = 0;
        for (int x = 0; x < hLines.cols; ++x) {
            if (row[x] > 0) ++count;
        }
        if (count > maxRowCount) maxRowCount = count;
    }

    const double ratio = static_cast<double>(maxRowCount) / binaryRowMax;
    return ratio >= crossLineRatio;
}

ErrPageResult ErrPage::process(const cv::Mat &src,
                                const QString &sourcePath,
                                const ErrPageOptions &options)
{
    ErrPageResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    // 1. 从文件名解析正确页码
    result.correctPage = parseCorrectPage(sourcePath);

    // 2. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 3. 检测区域
    const int regionW = static_cast<int>(W * options.regionWidthRatio);
    const int regionH = static_cast<int>(H * options.regionHeightRatio);

    std::vector<PageNumberItem> allItems;

    if (options.detectTopLeft) {
        cv::Rect topLeft(0, 0, regionW, regionH);
        detectDigitsInRegion(gray, topLeft, options, allItems);
    }
    if (options.detectTopRight) {
        cv::Rect topRight(W - regionW, 0, regionW, regionH);
        detectDigitsInRegion(gray, topRight, options, allItems);
    }

    // 4. 应用规则
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    for (auto &item : allItems) {
        const bool matchesCorrectPage =
            (result.correctPage >= 0 &&
             item.recognizedNumber == result.correctPage);

        if (matchesCorrectPage) {
            // 正确页码 → 保留
            continue;
        }

        // 非正确页码
        if (item.isCrossed) {
            // 有划线 → 自动删除
            cv::Rect r = item.boundingBox;
            r.x -= 3;
            r.y -= 3;
            r.width += 6;
            r.height += 6;
            r &= cv::Rect(0, 0, W, H);
            cv::rectangle(dst, r, white, cv::FILLED);
            ++result.crossedRemoved;
        } else {
            // 无划线 → 待确认
            ++result.pendingCount;
        }
    }

    // 5. 标记图
    cv::Mat marked = src.clone();
    for (const auto &item : allItems) {
        const bool matchesCorrectPage =
            (result.correctPage >= 0 &&
             item.recognizedNumber == result.correctPage);

        cv::Scalar color;
        if (matchesCorrectPage) {
            color = cv::Scalar(0, 255, 0);       // 绿 = 正确页码
        } else if (item.isCrossed) {
            color = cv::Scalar(0, 0, 255);       // 红 = 自动删除
        } else {
            color = cv::Scalar(0, 165, 255);     // 橙 = 待确认
        }
        cv::rectangle(marked, item.boundingBox, color, 2);

        // 在框下方显示识别结果
        QString label = QString::fromUtf8("?");
        if (item.recognizedNumber >= 0) {
            label = QString::number(item.recognizedNumber);
        }
        cv::putText(marked, label.toStdString(),
                    cv::Point(item.boundingBox.x,
                              item.boundingBox.y + item.boundingBox.height + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
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