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

    // 取中间区域
    const int marginX = static_cast<int>(W * (1.0 - options.paperSampleRatio) / 2.0);
    const int marginY = static_cast<int>(H * (1.0 - options.paperSampleRatio) / 2.0);

    const int x0 = std::max(0, marginX);
    const int y0 = std::max(0, marginY);
    const int x1 = std::min(W, W - marginX);
    const int y1 = std::min(H, H - marginY);

    if (x1 <= x0 || y1 <= y0) {
        // 退化情况，直接用整图
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }

    // 收集中间区域像素
    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(x1 - x0) * (y1 - y0) / 4);

    // 采样：每隔 2 像素取一个，加速
    for (int y = y0; y < y1; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = x0; x < x1; x += 2) {
            pixels.push_back(row[x]);
        }
    }

    if (pixels.empty()) {
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }

    // 排序取百分位
    const size_t idx = static_cast<size_t>(
        std::min<double>(pixels.size() - 1,
                         pixels.size() * options.paperPercentile));
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

namespace {

// 判断一行是否够"黑"（暗像素比例 >= 阈值）
bool isDarkRow(const cv::Mat &gray, int y, double darkThreshold, double darkPixelRatio)
{
    if (y < 0 || y >= gray.rows) {
        return false;
    }
    const uchar *row = gray.ptr<uchar>(y);
    const int W = gray.cols;
    const int need = static_cast<int>(W * darkPixelRatio);

    int darkCount = 0;
    for (int x = 0; x < W; ++x) {
        if (row[x] < darkThreshold) {
            ++darkCount;
            if (darkCount >= need) {
                return true;
            }
        }
    }
    return false;
}

// 判断一列是否够"黑"
bool isDarkCol(const cv::Mat &gray, int x, double darkThreshold, double darkPixelRatio)
{
    if (x < 0 || x >= gray.cols) {
        return false;
    }
    const int H = gray.rows;
    const int need = static_cast<int>(H * darkPixelRatio);

    int darkCount = 0;
    for (int y = 0; y < H; ++y) {
        if (gray.at<uchar>(y, x) < darkThreshold) {
            ++darkCount;
            if (darkCount >= need) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

int BlackEdge::scanTop(const cv::Mat &gray,
                       double darkThreshold,
                       const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    int count = 0;
    int gap = 0;

    for (int y = 0; y < maxScan && y < gray.rows; ++y) {
        if (isDarkRow(gray, y, darkThreshold, options.darkPixelRatio)) {
            ++count;
            gap = 0;
        } else {
            ++gap;
            if (gap > options.gapTolerance) {
                break;
            }
            // 允许间隙：不计数，但继续
        }
    }
    return count;
}

int BlackEdge::scanBottom(const cv::Mat &gray,
                          double darkThreshold,
                          const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    int count = 0;
    int gap = 0;

    for (int i = 0; i < maxScan && i < gray.rows; ++i) {
        const int y = gray.rows - 1 - i;
        if (isDarkRow(gray, y, darkThreshold, options.darkPixelRatio)) {
            ++count;
            gap = 0;
        } else {
            ++gap;
            if (gap > options.gapTolerance) {
                break;
            }
        }
    }
    return count;
}

int BlackEdge::scanLeft(const cv::Mat &gray,
                        double darkThreshold,
                        const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    int count = 0;
    int gap = 0;

    for (int x = 0; x < maxScan && x < gray.cols; ++x) {
        if (isDarkCol(gray, x, darkThreshold, options.darkPixelRatio)) {
            ++count;
            gap = 0;
        } else {
            ++gap;
            if (gap > options.gapTolerance) {
                break;
            }
        }
    }
    return count;
}

int BlackEdge::scanRight(const cv::Mat &gray,
                         double darkThreshold,
                         const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    int count = 0;
    int gap = 0;

    for (int i = 0; i < maxScan && i < gray.cols; ++i) {
        const int x = gray.cols - 1 - i;
        if (isDarkCol(gray, x, darkThreshold, options.darkPixelRatio)) {
            ++count;
            gap = 0;
        } else {
            ++gap;
            if (gap > options.gapTolerance) {
                break;
            }
        }
    }
    return count;
}

void BlackEdge::smoothMask(cv::Mat &mask, int kernelSize)
{
    if (kernelSize <= 1) {
        return;
    }
    if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options)
{
    BlackEdgeResult result;

    if (src.empty()) {
        return result;
    }

    const int W = src.cols;
    const int H = src.rows;

    // 1. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 2. 估算纸张灰度
    const double paperGray = estimatePaperGray(gray, options);
    result.paperGray = paperGray;

    // 3. 黑边阈值 = 纸张灰度 × darkRatio
    const double darkThreshold = paperGray * options.darkRatio;
    result.darkThreshold = darkThreshold;

    // 4. 四边独立扫描
    const int top    = scanTop(gray, darkThreshold, options);
    const int bottom = scanBottom(gray, darkThreshold, options);
    const int left   = scanLeft(gray, darkThreshold, options);
    const int right  = scanRight(gray, darkThreshold, options);

    result.topPixels = top;
    result.bottomPixels = bottom;
    result.leftPixels = left;
    result.rightPixels = right;

    // 5. 四边都很小 → 跳过
    const int minEdge = 2;
    if (top <= minEdge && bottom <= minEdge &&
        left <= minEdge && right <= minEdge) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 6. 构建黑边掩膜（黑边=255，纸张=0）
    cv::Mat edgeMask = cv::Mat::zeros(H, W, CV_8UC1);

    if (top > 0) {
        cv::rectangle(edgeMask, cv::Rect(0, 0, W, top),
                      cv::Scalar(255), cv::FILLED);
    }
    if (bottom > 0) {
        cv::rectangle(edgeMask, cv::Rect(0, H - bottom, W, bottom),
                      cv::Scalar(255), cv::FILLED);
    }
    if (left > 0) {
        cv::rectangle(edgeMask, cv::Rect(0, 0, left, H),
                      cv::Scalar(255), cv::FILLED);
    }
    if (right > 0) {
        cv::rectangle(edgeMask, cv::Rect(W - right, 0, right, H),
                      cv::Scalar(255), cv::FILLED);
    }

    // 7. 平滑掩膜（去斜边锯齿）
    if (options.smoothKernelSize > 1) {
        smoothMask(edgeMask, options.smoothKernelSize);
    }

    // 8. 处理
    if (options.fillWhite) {
        // 填白
        cv::Mat dst = src.clone();
        cv::Scalar white;
        if (dst.channels() == 4) {
            white = cv::Scalar(255, 255, 255, 255);
        } else {
            white = cv::Scalar(255, 255, 255);
        }
        dst.setTo(white, edgeMask);
        result.image = dst;
    } else {
        // 裁切：只裁掉四边黑边区域
        int x0 = (left > 0) ? left : 0;
        int y0 = (top > 0) ? top : 0;
        int x1 = W - ((right > 0) ? right : 0);
        int y1 = H - ((bottom > 0) ? bottom : 0);

        int w = x1 - x0;
        int h = y1 - y0;
        if (w < 1) w = 1;
        if (h < 1) h = 1;

        cv::Rect roi(x0, y0, w, h);
        roi &= cv::Rect(0, 0, W, H);
        result.image = src(roi).clone();
    }

    result.ok = true;
    return result;
}

} // namespace process