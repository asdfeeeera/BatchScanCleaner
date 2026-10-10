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

int scanEdgeDepth(const cv::Mat &gray, int x, int y, int dx, int dy,
                  int maxScan, double darkThreshold,
                  int whiteTolerance, int gapTolerance)
{
    const int W = gray.cols;
    const int H = gray.rows;

    int firstDark = -1;
    int lastDark = -1;
    int whiteRun = 0;

    for (int i = 0; i < maxScan; ++i) {
        const int px = x + dx * i;
        const int py = y + dy * i;
        if (px < 0 || px >= W || py < 0 || py >= H) break;

        if (gray.at<uchar>(py, px) < darkThreshold) {
            if (firstDark < 0) firstDark = i;
            lastDark = i;
            whiteRun = 0;
        } else {
            ++whiteRun;
            if (firstDark >= 0) {
                if (whiteRun > gapTolerance) break;
            } else {
                if (whiteRun > whiteTolerance) break;
            }
        }
    }

    return (lastDark < 0) ? 0 : (lastDark + 1);
}

int robustMax(std::vector<int> &v, int maxAllowed)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());

    for (int i = static_cast<int>(v.size()) - 1; i >= 0; --i) {
        if (v[i] <= maxAllowed) {
            return v[i];
        }
    }
    return 0;
}

} // namespace

int BlackEdge::scanTop(const cv::Mat &gray, double darkThreshold,
                       const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    std::vector<int> depths;
    depths.reserve(gray.cols);
    for (int x = 0; x < gray.cols; ++x) {
        depths.push_back(scanEdgeDepth(gray, x, 0, 0, 1, maxScan,
                                       darkThreshold, 30, options.gapTolerance));
    }
    const int maxAllowed = static_cast<int>(gray.rows * options.maxScanRatio * 0.8);
    return robustMax(depths, maxAllowed);
}

int BlackEdge::scanBottom(const cv::Mat &gray, double darkThreshold,
                          const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    std::vector<int> depths;
    depths.reserve(gray.cols);
    for (int x = 0; x < gray.cols; ++x) {
        depths.push_back(scanEdgeDepth(gray, x, gray.rows - 1, 0, -1, maxScan,
                                       darkThreshold, 30, options.gapTolerance));
    }
    const int maxAllowed = static_cast<int>(gray.rows * options.maxScanRatio * 0.8);
    return robustMax(depths, maxAllowed);
}

int BlackEdge::scanLeft(const cv::Mat &gray, double darkThreshold,
                        const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    std::vector<int> depths;
    depths.reserve(gray.rows);
    for (int y = 0; y < gray.rows; ++y) {
        depths.push_back(scanEdgeDepth(gray, 0, y, 1, 0, maxScan,
                                       darkThreshold, 30, options.gapTolerance));
    }
    const int maxAllowed = static_cast<int>(gray.cols * options.maxScanRatio * 0.8);
    return robustMax(depths, maxAllowed);
}

int BlackEdge::scanRight(const cv::Mat &gray, double darkThreshold,
                         const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    std::vector<int> depths;
    depths.reserve(gray.rows);
    for (int y = 0; y < gray.rows; ++y) {
        depths.push_back(scanEdgeDepth(gray, gray.cols - 1, y, -1, 0, maxScan,
                                       darkThreshold, 30, options.gapTolerance));
    }
    const int maxAllowed = static_cast<int>(gray.cols * options.maxScanRatio * 0.8);
    return robustMax(depths, maxAllowed);
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

    // ★ 黑边必须是"绝对黑"：只用 0.55，且上限 60
    double darkThreshold = paperGray * 0.55;
    if (darkThreshold > 60.0) darkThreshold = 60.0;
    result.darkThreshold = darkThreshold;

    int top    = scanTop(gray, darkThreshold, options);
    int bottom = scanBottom(gray, darkThreshold, options);
    int left   = scanLeft(gray, darkThreshold, options);
    int right  = scanRight(gray, darkThreshold, options);

    // ★ 黑边不可能超过页面尺寸的 5%，超过一定是误判
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