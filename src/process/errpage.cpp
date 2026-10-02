#include "errpage.h"

#include <opencv2/imgproc.hpp>
#include <QFileInfo>
#include <QRegularExpression>
#include <algorithm>
#include <cmath>

namespace process {

int ErrPage::parseCorrectPage(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QString baseName = info.completeBaseName();

    // 只提取纯数字文件名
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

// 提取数字块（用形态学膨胀让数字笔画连通）
std::vector<cv::Rect> findDigitBoxes(const cv::Mat &binary,
                                      const ErrPageOptions &options)
{
    std::vector<cv::Rect> result;

    // 水平方向膨胀，让同一数字的笔画连通
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

        // 长宽比过滤：数字通常是窄的
        const double aspect = static_cast<double>(h) / std::max(1, w);
        if (aspect < 0.5) continue;   // 太扁，可能是横线

        cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                   stats.at<int>(i, cv::CC_STAT_TOP),
                   w, h);
        result.push_back(r);
    }
    return result;
}

// 把距离近的数字块合并成一个页码（例如 "278" 由 3 个数字组成）
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

        // 同一行 + 水平距离近 → 合并
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

} // namespace

void ErrPage::detectDigitsInRegion(const cv::Mat &gray,
                                    const cv::Rect &region,
                                    const ErrPageOptions &options,
                                    std::vector<PageNumberItem> &outItems)
{
    // 裁剪区域
    cv::Rect r = region & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width <= 0 || r.height <= 0) return;

    cv::Mat roi = gray(r);

    // 二值化：文字=白（前景），背景=黑
    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 找数字块
    std::vector<cv::Rect> digitBoxes = findDigitBoxes(binary, options);

    // 合并相邻数字
    std::vector<cv::Rect> pageBoxes = mergeAdjacentDigits(digitBoxes);

    // 每个页码块判断是否被划线
    for (const auto &box : pageBoxes) {
        // 转回原图坐标
        const cv::Rect globalBox(box.x + r.x, box.y + r.y,
                                  box.width, box.height);

        PageNumberItem item;
        item.boundingBox = globalBox;
        item.isCrossed = detectCrossLine(gray, globalBox, options.crossLineRatio);

        // 置信度：面积比例大 + 划线特征明显
        const double areaRatio = static_cast<double>(box.area()) /
                                 (options.maxDigitWidth * options.maxDigitHeight);
        item.confidence = std::min(1.0, areaRatio * 3.0);

        outItems.push_back(item);
    }
}

bool ErrPage::detectCrossLine(const cv::Mat &gray,
                               const cv::Rect &digitBox,
                               double crossLineRatio)
{
    // 稍微扩大区域，检查数字周围是否有横线穿过
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

    // 用纯水平方向的长核进行开运算，只保留长横线
    const int kernelWidth = std::max(15, expanded.width * 2 / 3);
    cv::Mat hKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelWidth, 1));

    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    // 统计每一行的横线像素
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

    // 如果某一行横线像素超过 60%，认为有横线穿过
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

    // 4. 判断每一个页码：
    //    - 划线 → 自动删除
    //    - 无划线 → 待确认
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }

    for (const auto &item : allItems) {
        if (item.isCrossed) {
            // 自动删除（填白）
            cv::Rect r = item.boundingBox;
            r.x -= 3;
            r.y -= 3;
            r.width += 6;
            r.height += 6;
            r &= cv::Rect(0, 0, W, H);
            cv::rectangle(dst, r, white, cv::FILLED);
            ++result.crossedRemoved;
        } else {
            // 进入待确认
            ++result.pendingCount;
        }
    }

    // 5. 标记图
    cv::Mat marked = src.clone();
    for (const auto &item : allItems) {
        cv::Scalar color;
        if (item.isCrossed) {
            color = cv::Scalar(0, 0, 255);   // 红色 = 自动删除
        } else {
            color = cv::Scalar(0, 165, 255); // 橙色 = 待确认
        }
        cv::rectangle(marked, item.boundingBox, color, 2);
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