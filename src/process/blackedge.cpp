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

// 扫描函数：返回应该被涂白的深度
int scanLineForEdge(const uchar *grayData, const uchar *colorData, int length, double darkThr, int maxGap, int maxScan)
{
    int cutLength = 0;
    int whiteRun = 0;
    int darkCount = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        // 遇到彩色印章，立刻停止，保护印章
        if (colorData && colorData[i] == 255) break;

        if (grayData[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0;
            darkCount++;
        } else {
            whiteRun++;
            // 遇到连续白点，说明黑边断了，立刻停止
            if (whiteRun > maxGap) break;
        }
    }

    // 如果这段区域的深度超过10个像素，且黑点占比大于5%，就判定为黑边
    if (cutLength > 10 && (static_cast<double>(darkCount) / cutLength) > 0.05) {
        return cutLength;
    }
    return 0;
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

    // 印章保护掩膜
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        cv::inRange(hsvCh[1], 30, 255, colorMask);
        
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 比较宽容的黑度阈值
    double darkThr = std::min(paperGray * options.darkRatio, 140.0);
    if (darkThr < 40.0) darkThr = 40.0;
    result.darkThreshold = darkThr;

    const int maxScanY = static_cast<int>(H * 0.20);
    const int maxScanX = static_cast<int>(W * 0.20);
    const int maxGap = 15; // 容忍 15 像素的反光断点

    cv::Mat dst = src.clone();
    int paintedCount = 0;

    auto setWhite = [&](int y, int x) {
        if (y < 0 || y >= dst.rows || x < 0 || x >= dst.cols) return;
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return; // 保护印章
        
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    // 1. 涂白上边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
        for (int y = 0; y < cut; ++y) setWhite(y, x);
    }

    // 2. 涂白下边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(H - 1 - y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(H - 1 - y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
        for (int y = 0; y < cut; ++y) setWhite(H - 1 - y, x);
    }

    // 3. 涂白左边缘
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[x];
            rowC[x] = cPtr ? cPtr[x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX);
        for (int x = 0; x < cut; ++x) setWhite(y, x);
    }

    // 4. 涂白右边缘
    for (int y = 0; y < H; ++y) {
        std::vector<uchar> rowG(W), rowC(W);
        const uchar *gPtr = gray.ptr<uchar>(y);
        const uchar *cPtr = colorMask.empty() ? nullptr : colorMask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            rowG[x] = gPtr[W - 1 - x];
            rowC[x] = cPtr ? cPtr[W - 1 - x] : 0;
        }
        int cut = scanLineForEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX);
        for (int x = 0; x < cut; ++x) setWhite(y, W - 1 - x);
    }

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process