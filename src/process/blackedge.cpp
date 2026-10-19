#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

// 全图采样估算纸张灰度
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

// 扫描函数：增加颜色掩膜保护（遇到印章就停止）
int scanLineForEdge(const uchar *grayData, const uchar *colorData, int length, double darkThr, int maxGap, int maxScan, int edgeMinThick)
{
    // 前 edgeMinThick 个像素必须都是暗的，否则说明边缘本身是白的，不是黑边
    for (int i = 0; i < edgeMinThick && i < length; ++i) {
        if (grayData[i] >= darkThr) {
            return 0;
        }
    }

    int cutLength = 0;
    int whiteRun = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        // 关键保护：如果发现当前像素是红色印章（或明显彩色），立刻视为黑边结束
        if (colorData && colorData[i] == 255) {
            break;
        }

        if (grayData[i] < darkThr) {
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

    // 1. 提取高饱和度区域作为保护掩膜（彩色印章）
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 饱和度 > 40 视为彩色（印章、红章）
        cv::inRange(hsvCh[1], 40, 255, colorMask);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 黑度阈值：限制上限为 100
    double darkThr = std::min(paperGray * options.darkRatio, 100.0);
    if (darkThr < 30.0) darkThr = 30.0;
    result.darkThreshold = darkThr;

    const int maxScanY = static_cast<int>(H * options.maxScanRatio);
    const int maxScanX = static_cast<int>(W * options.maxScanRatio);

    const int maxGapY = std::max(15, static_cast<int>(H * 0.015));
    const int maxGapX = std::max(15, static_cast<int>(W * 0.015));
    const int edgeMinThick = 3;

    // --- 1. 扫描上边缘 ---
    std::vector<int> topCuts;
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGapY, maxScanY, edgeMinThick);
        if (cut > 0) topCuts.push_back(cut);
    }
    int topCut = 0;
    if (topCuts.size() > W * 0.1) topCut = getMedian(topCuts);

    // --- 2. 扫描下边缘 ---
    std::vector<int> bottomCuts;
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(H - 1 - y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(H - 1 - y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGapY, maxScanY, edgeMinThick);
        if (cut > 0) bottomCuts.push_back(cut);
    }
    int bottomCut = 0;
    if (bottomCuts.size() > W * 0.1) bottomCut = getMedian(bottomCuts);

    // --- 3. 扫描左边缘 ---
    std::vector<int> leftCuts;
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[x];
            rowC[x] = cPtr ? cPtr[x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGapX, maxScanX, edgeMinThick);
        if (cut > 0) leftCuts.push_back(cut);
    }
    int leftCut = 0;
    if (leftCuts.size() > H * 0.1) leftCut = getMedian(leftCuts);

    // --- 4. 扫描右边缘 ---
    std::vector<int> rightCuts;
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[W - 1 - x];
            rowC[x] = cPtr ? cPtr[W - 1 - x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGapX, maxScanX, edgeMinThick);
        if (cut > 0) rightCuts.push_back(cut);
    }
    int rightCut = 0;
    if (rightCuts.size() > H * 0.1) rightCut = getMedian(rightCuts);

    // 增加一点扩张像素，确保切干净
    topCut = std::min(H, topCut + options.expandPixels);
    bottomCut = std::min(H, bottomCut + options.expandPixels);
    leftCut = std::min(W, leftCut + options.expandPixels);
    rightCut = std::min(W, rightCut + options.expandPixels);

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