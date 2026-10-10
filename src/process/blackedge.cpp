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
// ★ 核心：基于"行/列灰度中位数"判定脏边
//   返回：脏边的行数/列数（0 表示干净）
// ============================================================
int countDirtyLines(const cv::Mat &gray,
                    bool fromTop, bool fromLeft,
                    int paperGray, int maxScan)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int whiteThr = paperGray - 15;   // 认为"纯白"的下界
    const int minWhiteRun = 3;             // 连续 3 行/列纯白才算"干净区开始"

    int whiteRun = 0;
    int dirtyEnd = 0;

    for (int i = 0; i < maxScan; ++i) {
        // 采样：每 4 个像素取 1 个
        std::vector<uchar> vals;

        if (fromTop || !fromTop) {
            // 上下方向：按行采样
            int y;
            if (fromTop) y = i;
            else         y = H - 1 - i;
            if (y < 0 || y >= H) break;

            const uchar *row = gray.ptr<uchar>(y);
            for (int x = 0; x < W; x += 4) {
                vals.push_back(row[x]);
            }
        }

        if (fromLeft) {
            // 左右方向：按列采样
            int x = i;
            if (x < 0 || x >= W) break;

            for (int y = 0; y < H; y += 4) {
                vals.push_back(gray.at<uchar>(y, x));
            }
        } else {
            // 从右
            int x = W - 1 - i;
            if (x < 0 || x >= W) break;

            for (int y = 0; y < H; y += 4) {
                vals.push_back(gray.at<uchar>(y, x));
            }
        }

        if (vals.empty()) break;

        // 中位数
        const size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
        const uchar median = vals[mid];

        if (median >= whiteThr) {
            ++whiteRun;
            if (whiteRun >= minWhiteRun) {
                // 连续 3 行/列都白 → 干净区从这里开始
                dirtyEnd = i - minWhiteRun + 1;
                return dirtyEnd;
            }
        } else {
            whiteRun = 0;
            dirtyEnd = i + 1;
        }
    }

    return dirtyEnd;
}

} // namespace

int BlackEdge::scanTop(const cv::Mat &gray, double darkThreshold,
                       const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.rows * 0.05),
                                 static_cast<int>(gray.rows * options.maxScanRatio));
    return countDirtyLines(gray, true, true, static_cast<int>(paperGray), maxScan);
}

int BlackEdge::scanBottom(const cv::Mat &gray, double darkThreshold,
                          const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.rows * 0.05),
                                 static_cast<int>(gray.rows * options.maxScanRatio));
    return countDirtyLines(gray, false, true, static_cast<int>(paperGray), maxScan);
}

int BlackEdge::scanLeft(const cv::Mat &gray, double darkThreshold,
                        const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.cols * 0.05),
                                 static_cast<int>(gray.cols * options.maxScanRatio));
    return countDirtyLines(gray, true, true, static_cast<int>(paperGray), maxScan);
}

int BlackEdge::scanRight(const cv::Mat &gray, double darkThreshold,
                         const BlackEdgeOptions &options)
{
    (void)darkThreshold;
    const double paperGray = estimatePaperGray(gray, options);
    const int maxScan = std::min(static_cast<int>(gray.cols * 0.05),
                                 static_cast<int>(gray.cols * options.maxScanRatio));
    return countDirtyLines(gray, true, false, static_cast<int>(paperGray), maxScan);
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
    result.darkThreshold = paperGray - 15;

    int top    = scanTop(gray, paperGray - 15, options);
    int bottom = scanBottom(gray, paperGray - 15, options);
    int left   = scanLeft(gray, paperGray - 15, options);
    int right  = scanRight(gray, paperGray - 15, options);

    // 安全上限（5%，已在 scan 函数里限制，但再保一次）
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