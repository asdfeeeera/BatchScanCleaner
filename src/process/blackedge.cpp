#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace process {

double BlackEdge::estimatePaperGray(const cv::Mat &gray,
                                     const BlackEdgeOptions &options)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * 0.25);
    const int my = static_cast<int>(H * 0.25);

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(W) * H / 8);
    for (int y = my; y < H - my; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = mx; x < W - mx; x += 2) {
            pixels.push_back(row[x]);
        }
    }
    if (pixels.empty()) {
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }
    const size_t idx = pixels.size() * 9 / 10;
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

namespace {

// ============================================================
// ★ 投影法：按行算"平均灰度"，找"脏 → 干净"的跳变点
// ============================================================
int scanRowsByProjection(const cv::Mat &gray, int paperGray,
                         bool fromTop, int maxScan)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int step = 4;
    const int dirtyThr = paperGray - 20;    // 低于 = 脏
    const int cleanThr = paperGray - 8;     // 高于 = 干净
    const int minCleanRun = 3;

    int lastDirty = -1;
    int cleanRun = 0;

    for (int i = 0; i < maxScan; ++i) {
        int y;
        if (fromTop) y = i;
        else         y = H - 1 - i;
        if (y < 0 || y >= H) break;

        const uchar *row = gray.ptr<uchar>(y);
        long sum = 0;
        int cnt = 0;
        for (int x = 0; x < W; x += step) {
            sum += row[x];
            ++cnt;
        }
        const double avg = (cnt > 0) ? (static_cast<double>(sum) / cnt) : 0.0;

        if (avg < dirtyThr) {
            lastDirty = i;
            cleanRun = 0;
        } else if (avg >= cleanThr) {
            ++cleanRun;
            if (lastDirty >= 0 && cleanRun >= minCleanRun) {
                return lastDirty + 1;
            }
        } else {
            lastDirty = i;
            cleanRun = 0;
        }
    }

    return (lastDirty < 0) ? 0 : (lastDirty + 1);
}

int scanColsByProjection(const cv::Mat &gray, int paperGray,
                         bool fromLeft, int maxScan)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int step = 4;
    const int dirtyThr = paperGray - 20;
    const int cleanThr = paperGray - 8;
    const int minCleanRun = 3;

    int lastDirty = -1;
    int cleanRun = 0;

    for (int i = 0; i < maxScan; ++i) {
        int x;
        if (fromLeft) x = i;
        else          x = W - 1 - i;
        if (x < 0 || x >= W) break;

        long sum = 0;
        int cnt = 0;
        for (int y = 0; y < H; y += step) {
            sum += gray.at<uchar>(y, x);
            ++cnt;
        }
        const double avg = (cnt > 0) ? (static_cast<double>(sum) / cnt) : 0.0;

        if (avg < dirtyThr) {
            lastDirty = i;
            cleanRun = 0;
        } else if (avg >= cleanThr) {
            ++cleanRun;
            if (lastDirty >= 0 && cleanRun >= minCleanRun) {
                return lastDirty + 1;
            }
        } else {
            lastDirty = i;
            cleanRun = 0;
        }
    }

    return (lastDirty < 0) ? 0 : (lastDirty + 1);
}

} // namespace

int BlackEdge::scanTop(const cv::Mat &gray, double darkThreshold,
                       const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.rows * 0.05),
                                 static_cast<int>(gray.rows * options.maxScanRatio));
    return scanRowsByProjection(gray, static_cast<int>(paperGray), true, maxScan);
}

int BlackEdge::scanBottom(const cv::Mat &gray, double darkThreshold,
                          const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.rows * 0.05),
                                 static_cast<int>(gray.rows * options.maxScanRatio));
    return scanRowsByProjection(gray, static_cast<int>(paperGray), false, maxScan);
}

int BlackEdge::scanLeft(const cv::Mat &gray, double darkThreshold,
                        const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.cols * 0.05),
                                 static_cast<int>(gray.cols * options.maxScanRatio));
    return scanColsByProjection(gray, static_cast<int>(paperGray), true, maxScan);
}

int BlackEdge::scanRight(const cv::Mat &gray, double darkThreshold,
                         const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.cols * 0.05),
                                 static_cast<int>(gray.cols * options.maxScanRatio));
    return scanColsByProjection(gray, static_cast<int>(paperGray), false, maxScan);
}

void BlackEdge::smoothMask(cv::Mat &mask, int kernelSize)
{
    if (kernelSize <= 1) return;
    if (kernelSize % 2 == 0) kernelSize += 1;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options)
{
    BlackEdgeResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const double paperGray = estimatePaperGray(gray, options);
    result.paperGray = paperGray;
    result.darkThreshold = paperGray - 20;

    int top    = scanTop(gray, paperGray - 20, options);
    int bottom = scanBottom(gray, paperGray - 20, options);
    int left   = scanLeft(gray, paperGray - 20, options);
    int right  = scanRight(gray, paperGray - 20, options);

    const int maxTopBottom = static_cast<int>(H * 0.05);
    const int maxLeftRight = static_cast<int>(W * 0.05);
    if (top > maxTopBottom)       top = 0;
    if (bottom > maxTopBottom)    bottom = 0;
    if (left > maxLeftRight)      left = 0;
    if (right > maxLeftRight)     right = 0;

    result.topPixels = top;
    result.bottomPixels = bottom;
    result.leftPixels = left;
    result.rightPixels = right;

    const int minEdge = 3;
    if (top <= minEdge && bottom <= minEdge &&
        left <= minEdge && right <= minEdge) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);

    if (top > 0) {
        cv::rectangle(mask, cv::Rect(0, 0, W, top),
                      cv::Scalar(255), cv::FILLED);
    }
    if (bottom > 0) {
        cv::rectangle(mask, cv::Rect(0, H - bottom, W, bottom),
                      cv::Scalar(255), cv::FILLED);
    }
    if (left > 0) {
        cv::rectangle(mask, cv::Rect(0, 0, left, H),
                      cv::Scalar(255), cv::FILLED);
    }
    if (right > 0) {
        cv::rectangle(mask, cv::Rect(W - right, 0, right, H),
                      cv::Scalar(255), cv::FILLED);
    }

    if (options.smoothKernelSize > 1) {
        smoothMask(mask, options.smoothKernelSize);
    }

    cv::Mat dst = src.clone();
    cv::Scalar white = (dst.channels() == 4)
                           ? cv::Scalar(255, 255, 255, 255)
                           : cv::Scalar(255, 255, 255);
    dst.setTo(white, mask);

    result.filledPixels = cv::countNonZero(mask);
    result.image = dst;
    result.ok = true;
    return result;
}

} // namespace process