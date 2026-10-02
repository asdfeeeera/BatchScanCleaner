#include "colorline.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace process {

void ColorLine::buildColorMask(const cv::Mat &bgr,
                                cv::Mat &colorMask,
                                const ColorLineOptions &options)
{
    colorMask = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);
    if (bgr.channels() < 3) return;

    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);
    cv::Mat &B = ch[0];
    cv::Mat &G = ch[1];
    cv::Mat &R = ch[2];

    // 弱彩色：任一通道比其他两通道高 channelDiffThreshold
    cv::Mat rgMax, rbMax, gbMax;
    cv::max(R, G, rgMax);   // max(R,G)
    cv::max(R, B, rbMax);   // max(R,B)
    cv::max(G, B, gbMax);   // max(G,B)

    cv::Mat bDiff, gDiff, rDiff;
    cv::subtract(B, rgMax, bDiff);
    cv::subtract(G, rbMax, gDiff);
    cv::subtract(R, gbMax, rDiff);

    cv::Mat bMask, gMask, rMask;
    cv::threshold(bDiff, bMask, options.channelDiffThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(gDiff, gMask, options.channelDiffThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(rDiff, rMask, options.channelDiffThreshold, 255, cv::THRESH_BINARY);

    cv::Mat weakColor;
    cv::bitwise_or(bMask, gMask, weakColor);
    cv::bitwise_or(weakColor, rMask, weakColor);

    // 排除暗色
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat bright;
    cv::threshold(gray, bright, options.valueThreshold, 255, cv::THRESH_BINARY);

    cv::bitwise_and(weakColor, bright, colorMask);
}

namespace {

// 判断一行是否算彩色细线
bool isColorRow(const cv::Mat &colorMask, int y, double threshold,
                int x0, int x1, double &ratio)
{
    const uchar *row = colorMask.ptr<uchar>(y);
    const int width = x1 - x0;
    if (width <= 0) { ratio = 0; return false; }

    int count = 0;
    for (int x = x0; x < x1; ++x) {
        if (row[x] > 0) ++count;
    }
    ratio = static_cast<double>(count) / width;
    return ratio >= threshold;
}

bool isColorCol(const cv::Mat &colorMask, int x, double threshold,
                int y0, int y1, double &ratio)
{
    const int height = y1 - y0;
    if (height <= 0) { ratio = 0; return false; }

    int count = 0;
    for (int y = y0; y < y1; ++y) {
        if (colorMask.at<uchar>(y, x) > 0) ++count;
    }
    ratio = static_cast<double>(count) / height;
    return ratio >= threshold;
}

cv::Scalar avgRowColor(const cv::Mat &bgr, const cv::Mat &colorMask,
                       int y, int x0, int x1)
{
    long b = 0, g = 0, r = 0;
    int count = 0;
    const cv::Vec3b *bgrRow = bgr.ptr<cv::Vec3b>(y);
    const uchar *maskRow = colorMask.ptr<uchar>(y);
    for (int x = x0; x < x1; ++x) {
        if (maskRow[x] > 0) {
            b += bgrRow[x][0];
            g += bgrRow[x][1];
            r += bgrRow[x][2];
            ++count;
        }
    }
    if (count == 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / count, g / count, r / count);
}

cv::Scalar avgColColor(const cv::Mat &bgr, const cv::Mat &colorMask,
                       int x, int y0, int y1)
{
    long b = 0, g = 0, r = 0;
    int count = 0;
    for (int y = y0; y < y1; ++y) {
        if (colorMask.at<uchar>(y, x) > 0) {
            const cv::Vec3b &px = bgr.at<cv::Vec3b>(y, x);
            b += px[0];
            g += px[1];
            r += px[2];
            ++count;
        }
    }
    if (count == 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / count, g / count, r / count);
}

} // namespace

ColorLineResult ColorLine::detect(const cv::Mat &src,
                                   const ColorLineOptions &options)
{
    ColorLineResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat bgr;
    if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = src.clone();
    }

    cv::Mat colorMask;
    buildColorMask(bgr, colorMask, options);

    // 边缘排除
    const int mx = static_cast<int>(W * options.edgeMarginRatio);
    const int my = static_cast<int>(H * options.edgeMarginRatio);
    const int x0 = mx;
    const int x1 = W - mx;
    const int y0 = my;
    const int y1 = H - my;

    if (x1 <= x0 || y1 <= y0) {
        result.image = bgr;
        result.markedImage = bgr.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // ---- 检测水平细线（逐行）----
    std::vector<bool> rowIsLine(H, false);
    std::vector<double> rowRatio(H, 0.0);
    for (int y = y0; y < y1; ++y) {
        double ratio = 0.0;
        if (isColorRow(colorMask, y, options.colorRatioThreshold, x0, x1, ratio)) {
            rowIsLine[y] = true;
            rowRatio[y] = ratio;
        }
    }

    // 合并相邻行
    int runStart = -1;
    for (int y = y0; y <= y1; ++y) {
        const bool isLine = (y < y1) && rowIsLine[y];
        if (isLine && runStart < 0) {
            runStart = y;
        } else if (!isLine && runStart >= 0) {
            const int thickness = y - runStart;
            if (thickness <= options.maxThickness) {
                ColorLineItem item;
                item.horizontal = true;
                item.position = runStart;
                item.thickness = thickness;
                item.colorRatio = rowRatio[runStart];
                item.color = avgRowColor(bgr, colorMask, runStart + thickness / 2, x0, x1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

    // ---- 检测垂直细线（逐列）----
    std::vector<bool> colIsLine(W, false);
    std::vector<double> colRatio(W, 0.0);
    for (int x = x0; x < x1; ++x) {
        double ratio = 0.0;
        if (isColorCol(colorMask, x, options.colorRatioThreshold, y0, y1, ratio)) {
            colIsLine[x] = true;
            colRatio[x] = ratio;
        }
    }

    runStart = -1;
    for (int x = x0; x <= x1; ++x) {
        const bool isLine = (x < x1) && colIsLine[x];
        if (isLine && runStart < 0) {
            runStart = x;
        } else if (!isLine && runStart >= 0) {
            const int thickness = x - runStart;
            if (thickness <= options.maxThickness) {
                ColorLineItem item;
                item.horizontal = false;
                item.position = runStart;
                item.thickness = thickness;
                item.colorRatio = colRatio[runStart];
                item.color = avgColColor(bgr, colorMask, runStart + thickness / 2, y0, y1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

    // ---- 生成标记图 ----
    cv::Mat marked = bgr.clone();
    for (const auto &item : result.items) {
        if (item.horizontal) {
            cv::Rect r(0, item.position, W, item.thickness);
            cv::rectangle(marked, r, cv::Scalar(0, 255, 255), 2);
        } else {
            cv::Rect r(item.position, 0, item.thickness, H);
            cv::rectangle(marked, r, cv::Scalar(0, 255, 255), 2);
        }
    }

    result.image = bgr;
    result.markedImage = marked;
    result.ok = true;

    if (result.items.empty()) {
        result.skipped = true;
    }
    return result;
}

ColorLineResult ColorLine::clear(const cv::Mat &src,
                                  const ColorLineOptions &options)
{
    ColorLineResult result = detect(src, options);

    if (!result.ok || result.items.empty()) {
        return result;
    }

    cv::Mat bgr = result.image.clone();
    const int W = bgr.cols;
    const int H = bgr.rows;

    // 构建清除掩膜
    cv::Mat clearMask = cv::Mat::zeros(H, W, CV_8UC1);
    for (const auto &item : result.items) {
        if (item.horizontal) {
            const int y = std::max(0, item.position - 2);
            const int h = std::min(H - y, item.thickness + 4);
            cv::rectangle(clearMask, cv::Rect(0, y, W, h),
                          cv::Scalar(255), cv::FILLED);
        } else {
            const int x = std::max(0, item.position - 2);
            const int w = std::min(W - x, item.thickness + 4);
            cv::rectangle(clearMask, cv::Rect(x, 0, w, H),
                          cv::Scalar(255), cv::FILLED);
        }
    }

    // 只清除彩色像素
    cv::Mat colorMask;
    buildColorMask(bgr, colorMask, options);
    cv::bitwise_and(clearMask, colorMask, clearMask);

    // 膨胀，边缘平滑
    cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(clearMask, clearMask, dilateKernel);

    cv::Mat dst = bgr.clone();
    dst.setTo(cv::Scalar(255, 255, 255), clearMask);

    result.image = dst;
    result.clearedPixels = cv::countNonZero(clearMask);
    return result;
}

} // namespace process