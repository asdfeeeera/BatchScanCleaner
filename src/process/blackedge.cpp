#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

// 采样估算纸张灰度（放宽极暗的干扰）
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

// ★ 重新设计的扫描函数：利用黑点比例和数量来确认真实黑边
int scanLineForEdge(const uchar *grayData, const uchar *colorData, int length, double darkThr, int maxGap, int maxScan)
{
    int cutLength = 0;
    int whiteRun = 0;
    int darkCount = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        // 遇到彩色印章，立刻停止，保护印章不被涂白
        if (colorData && colorData[i] == 255) {
            break;
        }

        if (grayData[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0;
            darkCount++;
        } else {
            whiteRun++;
            // 如果连续白点超过 maxGap（比如遇到了纸张真实的白色区域），说明黑边结束了
            if (whiteRun > maxGap) {
                break;
            }
        }
    }

    // 判定条件：这段区域的深度必须有一定厚度，且黑点数量不能太少（过滤单像素噪点）
    // 同时，黑点在这个区域内的占比不能太低，防止误伤边缘的文字
    if (cutLength > 5 && darkCount > 5 && (static_cast<double>(darkCount) / cutLength) > 0.1) {
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

    // 提取高饱和度区域（彩色印章），用于绝对保护
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
        
        // 稍微膨胀一下印章保护区域
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 黑度阈值：稍微放宽到 120，为了抓住发灰的扫描黑边
    double darkThr = std::min(paperGray * options.darkRatio, 120.0);
    if (darkThr < 40.0) darkThr = 40.0;
    result.darkThreshold = darkThr;

    const int maxScanY = static_cast<int>(H * 0.20);
    const int maxScanX = static_cast<int>(W * 0.20);
    const int maxGap = 30; // 允许黑边中有 30 像素的断点（比如反光）

    cv::Mat dst = src.clone();
    int paintedCount = 0;

    // 精确涂白函数，带印章保护
    auto setWhite = [&](int y, int x) {
        if (y < 0 || y >= dst.rows || x < 0 || x >= dst.cols) return;
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return; // 避开印章
        
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    // 1. 逐列扫描上边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
        for (int y = 0; y < cut; ++y) setWhite(y, x);
    }

    // 2. 逐列扫描下边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(H - 1 - y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(H - 1 - y, x);
        }
        int cut = scanLineForEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
        for (int y = 0; y < cut; ++y) setWhite(H - 1 - y, x);
    }

    // 3. 逐行扫描左边缘
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

    // 4. 逐行扫描右边缘
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