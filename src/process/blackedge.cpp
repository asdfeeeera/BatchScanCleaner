#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

// 全图采样估算纸张灰度，取 85% 分位数，避免大面积黑图拉低
double calcPaperGray(const cv::Mat &gray)
{
    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(gray.cols) * gray.rows / 4);

    for (int y = 0; y < gray.rows; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; x += 2) {
            pixels.push_back(row[x]);
        }
    }

    if (pixels.empty()) return 255.0;

    const size_t idx = static_cast<size_t>(pixels.size() * 0.85);
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

// 从边缘向内扫描
// 返回 0 表示该行/列没有黑边；否则返回需要切掉的像素数
int scanLineForEdge(const uchar *data, int length, double darkThr, int maxGap, int maxScan, int edgeMinThick)
{
    // 核心防呆：前 edgeMinThick 个像素必须都是暗的，否则说明边缘本身是白的，不是黑边
    for (int i = 0; i < edgeMinThick && i < length; ++i) {
        if (data[i] >= darkThr) {
            return 0;
        }
    }

    int cutLength = 0;
    int whiteRun = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        if (data[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0;
        } else {
            whiteRun++;
            if (whiteRun > maxGap) {
                break;
            }
        }
    }
    return cutLength;
}

// 取数组的中位数，排除离群点（单行划痕导致的误判）
int getMedian(const std::vector<int> &v)
{
    if (v.empty()) return 0;
    std::vector<int> copy = v;
    std::sort(copy.begin(), copy.end());
    return copy[copy.size() / 2];
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

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 黑度阈值：限制上限为 100，防止把深灰文字或阴影当黑边
    double darkThr = std::min(paperGray * options.darkRatio, 100.0);
    // 防止纸张极暗时阈值太低
    if (darkThr < 30.0) darkThr = 30.0;
    result.darkThreshold = darkThr;

    const int maxScanY = static_cast<int>(H * options.maxScanRatio);
    const int maxScanX = static_cast<int>(W * options.maxScanRatio);

    // 容差设为 1.5% 宽高，至少 15 像素（用于跨越白斑）
    const int maxGapY = std::max(15, static_cast<int>(H * 0.015));
    const int maxGapX = std::max(15, static_cast<int>(W * 0.015));

    // 定义最小连续黑边厚度（至少3像素），过滤单像素噪点/阴影
    const int edgeMinThick = 3;

    // --- 1. 扫描上边缘 ---
    std::vector<int> topCuts;
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> col(H);
        for (int y = 0; y < H; ++y) col[y] = gray.at<uchar>(y, x);
        int cut = scanLineForEdge(col.data(), H, darkThr, maxGapY, maxScanY, edgeMinThick);
        if (cut > 0) topCuts.push_back(cut);
    }
    int topCut = 0;
    // 至少 10% 的列有黑边，才认为上边存在黑边
    if (topCuts.size() > W * 0.1) {
        topCut = getMedian(topCuts);
    }

    // --- 2. 扫描下边缘 ---
    std::vector<int> bottomCuts;
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> col(H);
        for (int y = 0; y < H; ++y) col[y] = gray.at<uchar>(H - 1 - y, x);
        int cut = scanLineForEdge(col.data(), H, darkThr, maxGapY, maxScanY, edgeMinThick);
        if (cut > 0) bottomCuts.push_back(cut);
    }
    int bottomCut = 0;
    if (bottomCuts.size() > W * 0.1) {
        bottomCut = getMedian(bottomCuts);
    }

    // --- 3. 扫描左边缘 ---
    std::vector<int> leftCuts;
    for (int y = 0; y < H; ++y) {
        int cut = scanLineForEdge(gray.ptr<uchar>(y), W, darkThr, maxGapX, maxScanX, edgeMinThick);
        if (cut > 0) leftCuts.push_back(cut);
    }
    int leftCut = 0;
    if (leftCuts.size() > H * 0.1) {
        leftCut = getMedian(leftCuts);
    }

    // --- 4. 扫描右边缘 ---
    std::vector<int> rightCuts;
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> row(W);
        for (int x = 0; x < W; ++x) row[x] = gray.at<uchar>(y, W - 1 - x);
        int cut = scanLineForEdge(row.data(), W, darkThr, maxGapX, maxScanX, edgeMinThick);
        if (cut > 0) rightCuts.push_back(cut);
    }
    int rightCut = 0;
    if (rightCuts.size() > H * 0.1) {
        rightCut = getMedian(rightCuts);
    }

    // 增加一点扩张像素，确保切干净
    topCut = std::min(H, topCut + options.expandPixels);
    bottomCut = std::min(H, bottomCut + options.expandPixels);
    leftCut = std::min(W, leftCut + options.expandPixels);
    rightCut = std::min(W, rightCut + options.expandPixels);

    // 判断是否真的需要切
    if (topCut <= options.expandPixels + 1 && 
        bottomCut <= options.expandPixels + 1 &&
        leftCut <= options.expandPixels + 1 && 
        rightCut <= options.expandPixels + 1) {
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