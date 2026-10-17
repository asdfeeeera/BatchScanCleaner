#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

// 辅助：估算纸张灰度（取中间区域的高分位数）
double estimatePaperGray(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * 0.2);
    const int my = static_cast<int>(H * 0.2);

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(W) * H / 10);

    for (int y = my; y < H - my; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = mx; x < W - mx; x += 2) {
            pixels.push_back(row[x]);
        }
    }

    if (pixels.empty()) return 255.0;

    const size_t idx = static_cast<size_t>(pixels.size() * 0.9);
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

// 辅助：从一端向另一端扫描，容忍断点
// 返回需要被切掉的像素数量
int scanLineForEdge(const uchar *data, int length, double darkThr, int maxGap, int maxScan)
{
    int cutLength = 0;
    int whiteRun = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        if (data[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0; // 遇到黑色，重置白色连续计数
        } else {
            whiteRun++;
            if (whiteRun > maxGap) {
                // 连续白了超过 maxGap 个像素，认定黑边结束
                break;
            }
        }
    }
    return cutLength;
}

} // namespace

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src, const BlackEdgeOptions &options)
{
    BlackEdgeResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src.clone();

    const double paperGray = estimatePaperGray(gray);
    result.paperGray = paperGray;

    // 黑度阈值 = 纸张灰度 * darkRatio (默认0.75)
    const double darkThr = paperGray * options.darkRatio;
    result.darkThreshold = darkThr;

    // 最大扫描深度 = 页面宽高的 maxScanRatio (默认30%)
    const int maxScanY = static_cast<int>(H * options.maxScanRatio);
    const int maxScanX = static_cast<int>(W * options.maxScanRatio);

    // 容差（允许连续白点不视为断开）
    // 选用页面宽高的 2% 作为容差，至少 20 像素
    const int maxGapY = std::max(20, static_cast<int>(H * 0.02));
    const int maxGapX = std::max(20, static_cast<int>(W * 0.02));

    // 1. 扫描上边缘
    int topCut = 0;
    for (int x = 0; x < W; ++x) {
        const int cut = scanLineForEdge(gray.ptr<uchar>(0) + x, H, darkThr, maxGapY, maxScanY);
        if (cut > topCut) topCut = cut;
    }
    
    // 2. 扫描下边缘
    int bottomCut = 0;
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> col(H);
        for (int y = 0; y < H; ++y) col[y] = gray.at<uchar>(H - 1 - y, x);
        const int cut = scanLineForEdge(col.data(), H, darkThr, maxGapY, maxScanY);
        if (cut > bottomCut) bottomCut = cut;
    }

    // 3. 扫描左边缘
    int leftCut = 0;
    for (int y = 0; y < H; ++y) {
        const int cut = scanLineForEdge(gray.ptr<uchar>(y), W, darkThr, maxGapX, maxScanX);
        if (cut > leftCut) leftCut = cut;
    }

    // 4. 扫描右边缘
    int rightCut = 0;
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> row(W);
        for (int x = 0; x < W; ++x) row[x] = gray.at<uchar>(y, W - 1 - x);
        const int cut = scanLineForEdge(row.data(), W, darkThr, maxGapX, maxScanX);
        if (cut > rightCut) rightCut = cut;
    }

    // 增加一点扩张像素，确保切干净
    topCut = std::min(H, topCut + options.expandPixels);
    bottomCut = std::min(H, bottomCut + options.expandPixels);
    leftCut = std::min(W, leftCut + options.expandPixels);
    rightCut = std::min(W, rightCut + options.expandPixels);

    // 判断是否真的需要切
    if (topCut <= options.expandPixels && 
        bottomCut <= options.expandPixels &&
        leftCut <= options.expandPixels && 
        rightCut <= options.expandPixels) {
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    result.topPixels = topCut;
    result.bottomPixels = bottomCut;
    result.leftPixels = leftCut;
    result.rightPixels = rightCut;

    // 开始裁剪并涂白
    cv::Rect roi(leftCut, topCut, W - leftCut - rightCut, H - topCut - bottomCut);
    if (roi.width <= 0 || roi.height <= 0) {
        result.image = src.clone();
        result.ok = false;
        return result;
    }

    cv::Mat cropped = src(roi);
    
    if (options.fillWhite) {
        result.image = cv::Mat::zeros(cropped.size(), cropped.type());
        if (src.channels() == 4) {
            result.image.setTo(cv::Scalar(255, 255, 255, 255));
        } else if (src.channels() == 3) {
            result.image.setTo(cv::Scalar(255, 255, 255));
        } else {
            result.image.setTo(cv::Scalar(255));
        }
        cropped.copyTo(result.image);
    } else {
        result.image = cropped.clone();
    }

    result.ok = true;
    return result;
}

} // namespace process