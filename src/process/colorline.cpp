#include "colorline.h"

#include <opencv2/imgproc.hpp>
#include <vector>
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

    cv::Mat maxRG, maxRGB, minRG, minRGB;
    cv::max(R, G, maxRG);
    cv::max(maxRG, B, maxRGB);
    cv::min(R, G, minRG);
    cv::min(minRG, B, minRGB);

    cv::Mat colorfulness;
    cv::subtract(maxRGB, minRGB, colorfulness);

    cv::Mat brightness;
    cv::add(B, G, brightness);
    cv::add(brightness, R, brightness);
    cv::divide(brightness, 3.0, brightness);

    cv::Mat c1, c2;
    cv::threshold(colorfulness, c1, options.channelDiffThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(brightness, c2, options.valueThreshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(c1, c2, colorMask);
}

namespace {

const int kMinColorDiff = 6;    // 像素级色度差阈值
const int kMinBrightness = 120; // 彩色像素的最小亮度（防阴影误判）
const int kMaxDarkBrightness = 100;  // 黑色像素的最大亮度

// ★ 判断一个像素是否属于"故障线候选"：明显彩色 或 明显暗色
inline bool isPixelInk(int b, int g, int r)
{
    const int mx = std::max({b, g, r});
    const int mn = std::min({b, g, r});
    const int bright = (b + g + r) / 3;

    // 1. 彩色像素（色度差大 + 亮度够）
    if (mx - mn >= kMinColorDiff && bright >= kMinBrightness) {
        return true;
    }
    // 2. 黑色像素（亮度低）
    if (bright < kMaxDarkBrightness) {
        return true;
    }
    return false;
}

// 分析一行：返回 (故障像素占比, 跨度占页面宽度比例)
struct RowAnalysis
{
    double inkRatio = 0.0;
    double spanRatio = 0.0;
    int inkCount = 0;
};

RowAnalysis analyzeRow(const cv::Mat &B, const cv::Mat &G, const cv::Mat &R,
                       int y, int x0, int x1)
{
    RowAnalysis ra;
    const int width = x1 - x0;
    if (width <= 0) return ra;

    const uchar *bRow = B.ptr<uchar>(y);
    const uchar *gRow = G.ptr<uchar>(y);
    const uchar *rRow = R.ptr<uchar>(y);

    int firstX = -1;
    int lastX = -1;

    for (int x = x0; x < x1; ++x) {
        if (isPixelInk(bRow[x], gRow[x], rRow[x])) {
            if (firstX < 0) firstX = x;
            lastX = x;
            ++ra.inkCount;
        }
    }

    ra.inkRatio = static_cast<double>(ra.inkCount) / width;
    if (firstX >= 0 && lastX >= firstX) {
        const int span = lastX - firstX + 1;
        ra.spanRatio = static_cast<double>(span) / width;
    } else {
        ra.spanRatio = 0.0;
    }
    return ra;
}

struct ColAnalysis
{
    double inkRatio = 0.0;
    double spanRatio = 0.0;
    int inkCount = 0;
};

ColAnalysis analyzeCol(const cv::Mat &B, const cv::Mat &G, const cv::Mat &R,
                       int x, int y0, int y1)
{
    ColAnalysis ca;
    const int height = y1 - y0;
    if (height <= 0) return ca;

    int firstY = -1;
    int lastY = -1;

    for (int y = y0; y < y1; ++y) {
        const int b = B.at<uchar>(y, x);
        const int g = G.at<uchar>(y, x);
        const int r = R.at<uchar>(y, x);
        if (isPixelInk(b, g, r)) {
            if (firstY < 0) firstY = y;
            lastY = y;
            ++ca.inkCount;
        }
    }

    ca.inkRatio = static_cast<double>(ca.inkCount) / height;
    if (firstY >= 0 && lastY >= firstY) {
        const int span = lastY - firstY + 1;
        ca.spanRatio = static_cast<double>(span) / height;
    } else {
        ca.spanRatio = 0.0;
    }
    return ca;
}

cv::Scalar avgRowColor(const cv::Mat &bgr, int y, int x0, int x1)
{
    long b = 0, g = 0, r = 0;
    const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
    for (int x = x0; x < x1; ++x) {
        b += row[x][0];
        g += row[x][1];
        r += row[x][2];
    }
    const int n = x1 - x0;
    if (n <= 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / n, g / n, r / n);
}

cv::Scalar avgColColor(const cv::Mat &bgr, int x, int y0, int y1)
{
    long b = 0, g = 0, r = 0;
    for (int y = y0; y < y1; ++y) {
        const cv::Vec3b &px = bgr.at<cv::Vec3b>(y, x);
        b += px[0];
        g += px[1];
        r += px[2];
    }
    const int n = y1 - y0;
    if (n <= 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / n, g / n, r / n);
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

    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);

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

    // ============ 逐行分析 ============
    std::vector<bool> rowIsLine(H, false);
    std::vector<double> rowBias(H, 0.0);

    for (int y = y0; y < y1; ++y) {
        const RowAnalysis ra = analyzeRow(ch[0], ch[1], ch[2], y, x0, x1);

        // ★ 判定：跨度 >= 95%（贯穿整页）
        //   故障线像素占比 >= 20%（避免空白行被误判）
        if (ra.inkRatio >= 0.20 && ra.spanRatio >= 0.95) {
            rowIsLine[y] = true;
            rowBias[y] = ra.inkRatio;
        }
    }

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
                item.colorRatio = rowBias[runStart];
                item.color = avgRowColor(bgr, runStart + thickness / 2, x0, x1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

    // ============ 逐列分析 ============
    std::vector<bool> colIsLine(W, false);
    std::vector<double> colBias(W, 0.0);

    for (int x = x0; x < x1; ++x) {
        const ColAnalysis ca = analyzeCol(ch[0], ch[1], ch[2], x, y0, y1);

        if (ca.inkRatio >= 0.20 && ca.spanRatio >= 0.95) {
            colIsLine[x] = true;
            colBias[x] = ca.inkRatio;
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
                item.colorRatio = colBias[runStart];
                item.color = avgColColor(bgr, runStart + thickness / 2, y0, y1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

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

    cv::Mat lineMask = cv::Mat::zeros(H, W, CV_8UC1);
    for (const auto &item : result.items) {
        if (item.horizontal) {
            const int y = std::max(0, item.position - 2);
            const int h = std::min(H - y, item.thickness + 4);
            cv::rectangle(lineMask, cv::Rect(0, y, W, h),
                          cv::Scalar(255), cv::FILLED);
        } else {
            const int x = std::max(0, item.position - 2);
            const int w = std::min(W - x, item.thickness + 4);
            cv::rectangle(lineMask, cv::Rect(x, 0, w, H),
                          cv::Scalar(255), cv::FILLED);
        }
    }

    // ★ 像素级"墨"判定：彩色 或 暗色
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);

    cv::Mat pixelInkMask = cv::Mat::zeros(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        const uchar *bRow = ch[0].ptr<uchar>(y);
        const uchar *gRow = ch[1].ptr<uchar>(y);
        const uchar *rRow = ch[2].ptr<uchar>(y);
        uchar *mRow = pixelInkMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            if (isPixelInk(bRow[x], gRow[x], rRow[x])) {
                mRow[x] = 255;
            }
        }
    }

    cv::Mat clearMask;
    cv::bitwise_and(pixelInkMask, lineMask, clearMask);

    cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(clearMask, clearMask, dilateKernel);

    cv::Mat dst = bgr.clone();
    dst.setTo(cv::Scalar(255, 255, 255), clearMask);

    result.image = dst;
    result.clearedPixels = cv::countNonZero(clearMask);
    return result;
}

} // namespace process