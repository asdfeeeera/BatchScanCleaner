#include "blackedge.h"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

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

    // 1. 融合保护掩膜
    cv::Mat colorMask;
    if (!options.protectMask.empty()) {
        // 优先使用批处理或外部传入的印章掩膜
        colorMask = options.protectMask.clone();
    } else if (src.channels() >= 3) {
        // 没传的话，自己用红色检测找
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        cv::Mat mask1, mask2;
        cv::inRange(hsvCh[0], 0, 15, mask1);
        cv::inRange(hsvCh[0], 150, 180, mask2);
        cv::bitwise_or(mask1, mask2, colorMask);
        
        cv::Mat satMask;
        cv::inRange(hsvCh[1], 40, 255, satMask);
        cv::bitwise_and(colorMask, satMask, colorMask);
    }

    // ★ 保护掩膜必须膨胀！印章边缘也要保
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    }

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (src.channels() == 4) white = cv::Scalar(255, 255, 255, 255);
    else if (src.channels() == 3) white = cv::Scalar(255, 255, 255);
    else white = cv::Scalar(255);

    int paintedCount = 0;
    auto setWhite = [&](int y, int x) {
        if (y < 0 || y >= dst.rows || x < 0 || x >= dst.cols) return;
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return; // 绝对不碰保护掩膜
        
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    const double darkThr = 150.0; // 容忍发灰的黑边
    const int maxScanX = static_cast<int>(W * 0.30); // 修复警告
    const int maxScanY = static_cast<int>(H * 0.30); // 修复警告
    const int maxGap = 40; // 允许 40 像素的反光断点

    // 1. 扫描上边缘
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        int darkRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) break; // 遇印章直接停
            if (gray.at<uchar>(y, x) < darkThr) {
                cut = y + 1;
                whiteRun = 0;
                darkRun++;
            } else {
                whiteRun++;
                if (whiteRun > maxGap) break;
            }
        }
        if (darkRun > 10) { // 确认有连续黑边才涂白
            for (int y = 0; y < cut; ++y) setWhite(y, x);
        }
    }

    // 2. 扫描下边缘
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        int darkRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            if (!colorMask.empty() && colorMask.at<uchar>(H - 1 - y, x) > 0) break;
            if (gray.at<uchar>(H - 1 - y, x) < darkThr) {
                cut = y + 1;
                whiteRun = 0;
                darkRun++;
            } else {
                whiteRun++;
                if (whiteRun > maxGap) break;
            }
        }
        if (darkRun > 10) {
            for (int y = 0; y < cut; ++y) setWhite(H - 1 - y, x);
        }
    }

    // 3. 扫描左边缘
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        int darkRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) break;
            if (gray.at<uchar>(y, x) < darkThr) {
                cut = x + 1;
                whiteRun = 0;
                darkRun++;
            } else {
                whiteRun++;
                if (whiteRun > maxGap) break;
            }
        }
        if (darkRun > 10) {
            for (int x = 0; x < cut; ++x) setWhite(y, x);
        }
    }

    // 4. 扫描右边缘
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        int darkRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, W - 1 - x) > 0) break;
            if (gray.at<uchar>(y, W - 1 - x) < darkThr) {
                cut = x + 1;
                whiteRun = 0;
                darkRun++;
            } else {
                whiteRun++;
                if (whiteRun > maxGap) break;
            }
        }
        if (darkRun > 10) {
            for (int x = 0; x < cut; ++x) setWhite(y, W - 1 - x);
        }
    }

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process