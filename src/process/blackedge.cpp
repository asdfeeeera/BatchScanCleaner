#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

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

int scanLineForEdge(const uchar *grayData, const uchar *colorData, int length, double darkThr, int maxGap, int maxScan, int edgeMinThick)
{
    // 放宽前几个像素的检测，只要连续暗像素达到 edgeMinThick，就认为碰上了扫描黑边
    int initThick = 0;
    for (int i = 0; i < edgeMinThick && i < length; ++i) {
        if (grayData[i] < darkThr) initThick++;
    }
    if (initThick < edgeMinThick) return 0;

    int cutLength = 0;
    int whiteRun = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        // 遇到彩色印章，立刻停止
        if (colorData && colorData[i] == 255) break;

        if (grayData[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0;
        } else {
            whiteRun++;
            if (whiteRun > maxGap) break;
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

    // 提取印章保护区域
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        cv::inRange(hsvCh[1], 35, 255, colorMask);
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    double darkThr = std::min(paperGray * options.darkRatio, 100.0);
    if (darkThr < 30.0) darkThr = 30.0;
    result.darkThreshold = darkThr;

    const int maxScanY = static_cast<int>(H * 0.20); // 恢复到 20% 扫描深度
    const int maxScanX = static_cast<int>(W * 0.20);
    const int maxGap = 15;
    const int edgeMinThick = 5; // 必须连续5像素暗，防止表格线

    cv::Mat dst = src.clone();
    int paintedCount = 0;

    // ★ 逐行、逐列精确涂白
    auto setWhite = [&](int y, int x) {
        if (y < 0 || y >= dst.rows || x < 0 || x >= dst.cols) return;
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return; // 保护印章
        
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    // 1. 扫描上边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY, edgeMinThick);
        for (int y = 0; y < cut; ++y) setWhite(y, x);
    }

    // 2. 扫描下边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(H - 1 - y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(H - 1 - y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY, edgeMinThick);
        for (int y = 0; y < cut; ++y) setWhite(H - 1 - y, x);
    }

    // 3. 扫描左边缘
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[x];
            rowC[x] = cPtr ? cPtr[x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX, edgeMinThick);
        for (int x = 0; x < cut; ++x) setWhite(y, x);
    }

    // 4. 扫描右边缘
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[W - 1 - x];
            rowC[x] = cPtr ? cPtr[W - 1 - x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX, edgeMinThick);
        for (int x = 0; x < cut; ++x) setWhite(y, W - 1 - x);
    }

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process