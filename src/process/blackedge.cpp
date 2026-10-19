#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

// 边缘单向扫描：碰到黑边就涂，碰到红色刹车，碰到连续白纸停止
int scanEdge(const uchar *grayData, const uchar *colorData, int length, double darkThr, int maxGap, int maxScan)
{
    int cutLength = 0;
    int whiteRun = 0;

    for (int i = 0; i < length && i < maxScan; ++i) {
        // 1. 遇到红色印章，立刻停止扫描，绝不伤害印章
        if (colorData && colorData[i] == 255) {
            break;
        }

        // 2. 碰到黑边（或者发灰的阴影），记录要涂白的深度
        if (grayData[i] < darkThr) {
            cutLength = i + 1;
            whiteRun = 0;
        } else {
            // 3. 碰到白点，记录连续白点数量
            whiteRun++;
            // 4. 如果连续白点超过 maxGap（比如 30 个像素的纯白），说明黑边彻底结束了，停止扫描
            if (whiteRun > maxGap) {
                break;
            }
        }
    }
    
    // 确保切掉的是真正有内容的黑边，而不是零星几个像素
    if (cutLength > 5) {
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

    // 提取极高饱和度的红色印章掩膜，并做极大的膨胀，确保印章附近绝不涂白
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 提取红色区域（S > 50）
        cv::inRange(hsvCh[1], 50, 255, colorMask);
        
        // ★ 超级膨胀，把印章周围 15 个像素都保护起来！
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    // 判定黑边的灰度阈值（上限放宽到 120，防止发灰的扫描黑边漏掉）
    double darkThr = 120.0;
    
    // 扫描深度上限设为 20%，防止大黑图被整张涂白
    const int maxScanY = static_cast<int>(H * 0.20);
    const int maxScanX = static_cast<int>(W * 0.20);
    
    // 连续遇到 30 个白像素就停止（防止误伤文字）
    const int maxGap = 30;

    cv::Mat dst = src.clone();
    int paintedCount = 0;

    // 辅助涂白函数
    auto setWhite = [&](int y, int x) {
        if (y < 0 || y >= dst.rows || x < 0 || x >= dst.cols) return;
        // 再次防御：绝不涂白印章保护区域
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return;
        
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
        int cut = scanEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
        for (int y = 0; y < cut; ++y) setWhite(y, x);
    }

    // 2. 涂白下边缘
    for (int x = 0; x < W; ++x) {
        std::vector<uchar> colG(H), colC(H);
        for (int y = 0; y < H; ++y) {
            colG[y] = gray.at<uchar>(H - 1 - y, x);
            colC[y] = colorMask.empty() ? 0 : colorMask.at<uchar>(H - 1 - y, x);
        }
        int cut = scanEdge(colG.data(), colC.data(), H, darkThr, maxGap, maxScanY);
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
        int cut = scanEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX);
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
        int cut = scanEdge(rowG.data(), rowC.data(), W, darkThr, maxGap, maxScanX);
        for (int x = 0; x < cut; ++x) setWhite(y, W - 1 - x);
    }

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process