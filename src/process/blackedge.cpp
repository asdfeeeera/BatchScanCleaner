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

    const int marginX = static_cast<int>(W * (1.0 - options.paperSampleRatio) / 2.0);
    const int marginY = static_cast<int>(H * (1.0 - options.paperSampleRatio) / 2.0);

    const int x0 = std::max(0, marginX);
    const int y0 = std::max(0, marginY);
    const int x1 = std::min(W, W - marginX);
    const int y1 = std::min(H, H - marginY);

    if (x1 <= x0 || y1 <= y0) {
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(x1 - x0) * (y1 - y0) / 4);

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

    const size_t idx = static_cast<size_t>(
        std::min<double>(pixels.size() - 1,
                         pixels.size() * options.paperPercentile));
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

namespace {

int scanDepth(const cv::Mat &gray, int x, int y, int dx, int dy,
              int maxScan, double darkThreshold, int gapTolerance)
{
    const int W = gray.cols;
    const int H = gray.rows;
    int lastDark = -1;
    int gapCount = 0;

    for (int i = 0; i < maxScan; ++i) {
        const int px = x + dx * i;
        const int py = y + dy * i;
        if (px < 0 || px >= W || py < 0 || py >= H) break;

        if (gray.at<uchar>(py, px) < darkThreshold) {
            lastDark = i;
            gapCount = 0;
        } else {
            ++gapCount;
            if (gapCount > gapTolerance) break;
        }
    }

    return (lastDark >= 0) ? (lastDark + 1) : 0;
}

int percentileOf(std::vector<int> &values, double p)
{
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t idx = static_cast<size_t>(
        std::min<double>(values.size() - 1, values.size() * p));
    return values[idx];
}

} // namespace

int BlackEdge::scanTop(const cv::Mat &gray,
                       double darkThreshold,
                       const BlackEdgeOptions &options)
{
    const int H = gray.rows;
    const int W = gray.cols;
    const int maxScan = static_cast<int>(H * options.maxScanRatio);

    std::vector<int> depths;
    depths.reserve(W);
    for (int x = 0; x < W; ++x) {
        const int d = scanDepth(gray, x, 0, 0, 1, maxScan,
                                darkThreshold, options.gapTolerance);
        depths.push_back(d);
    }
    return percentileOf(depths, 0.95) + 5;
}

int BlackEdge::scanBottom(const cv::Mat &gray,
                          double darkThreshold,
                          const BlackEdgeOptions &options)
{
    const int H = gray.rows;
    const int W = gray.cols;
    const int maxScan = static_cast<int>(H * options.maxScanRatio);

    std::vector<int> depths;
    depths.reserve(W);
    for (int x = 0; x < W; ++x) {
        const int d = scanDepth(gray, x, H - 1, 0, -1, maxScan,
                                darkThreshold, options.gapTolerance);
        depths.push_back(d);
    }
    return percentileOf(depths, 0.95) + 5;
}

int BlackEdge::scanLeft(const cv::Mat &gray,
                        double darkThreshold,
                        const BlackEdgeOptions &options)
{
    const int H = gray.rows;
    const int W = gray.cols;
    const int maxScan = static_cast<int>(W * options.maxScanRatio);

    std::vector<int> depths;
    depths.reserve(H);
    for (int y = 0; y < H; ++y) {
        const int d = scanDepth(gray, 0, y, 1, 0, maxScan,
                                darkThreshold, options.gapTolerance);
        depths.push_back(d);
    }
    return percentileOf(depths, 0.95) + 5;
}

int BlackEdge::scanRight(const cv::Mat &gray,
                         double darkThreshold,
                         const BlackEdgeOptions &options)
{
    const int H = gray.rows;
    const int W = gray.cols;
    const int maxScan = static_cast<int>(W * options.maxScanRatio);

    std::vector<int> depths;
    depths.reserve(H);
    for (int y = 0; y < H; ++y) {
        const int d = scanDepth(gray, W - 1, y, -1, 0, maxScan,
                                darkThreshold, options.gapTolerance);
        depths.push_back(d);
    }
    return percentileOf(depths, 0.95) + 5;
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

    if (src.empty()) {
        return result;
    }

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

    const double darkThreshold = paperGray * options.darkRatio;
    result.darkThreshold = darkThreshold;

    const int top    = scanTop(gray, darkThreshold, options);
    const int bottom = scanBottom(gray, darkThreshold, options);
    const int left   = scanLeft(gray, darkThreshold, options);
    const int right  = scanRight(gray, darkThreshold, options);

    result.topPixels = top;
    result.bottomPixels = bottom;
    result.leftPixels = left;
    result.rightPixels = right;

    const int minEdge = 5;
    if (top <= minEdge && bottom <= minEdge &&
        left <= minEdge && right <= minEdge) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

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

    if (options.smoothKernelSize > 1) {
        smoothMask(edgeMask, options.smoothKernelSize);
    }

    if (options.fillWhite) {
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