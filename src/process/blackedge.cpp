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

    // 1. 提取印章保护掩膜（极度灵敏）
    cv::Mat colorMask;
    if (!options.protectMask.empty()) {
        colorMask = options.protectMask.clone();
    } else if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 饱和度 > 25，亮度 > 30 就算印章（红章、蓝签等）
        cv::inRange(hsvCh[1], 25, 255, colorMask);
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 30, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);
    }

    // 膨胀印章掩膜，绝对保护边缘
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    }

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (src.channels() == 4) white = cv::Scalar(255, 255, 255, 255);
    else if (src.channels() == 3) white = cv::Scalar(255, 255, 255);
    else white = cv::Scalar(255);

    int paintedCount = 0;

    // 执行涂白的 lambda
    auto setWhite = [&](int y, int x) {
        // 不碰印章
        if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) return;
        
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    const double darkThr = 180.0; // 把发灰的暗边也算上
    const int maxScanX = static_cast<int>(W * 0.35); // 允许往深处扫
    const int maxScanY = static_cast<int>(H * 0.35);
    const int maxGap = 50; // 容忍 50 像素的反光/白斑断点

    // --- 1. 扫描上边缘 ---
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            // 遇到印章，跳过，重置连续白点
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) {
                whiteRun = 0;
                continue;
            }
            
            if (gray.at<uchar>(y, x) < darkThr) {
                cut = y + 1;
                whiteRun = 0;
            } else if (gray.at<uchar>(y, x) > 230) {
                whiteRun++;
                if (whiteRun > maxGap) break;
            } else {
                // 介于 180~230 的浅灰过渡带，将其囊括在 cut 内
                cut = y + 1;
                whiteRun = 0;
            }
        }
        for (int y = 0; y < cut; ++y) setWhite(y, x);
    }

    // --- 2. 扫描下边缘 ---
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            if (!colorMask.empty() && colorMask.at<uchar>(H - 1 - y, x) > 0) {
                whiteRun = 0;
                continue;
            }
            
            if (gray.at<uchar>(H - 1 - y, x) < darkThr) {
                cut = y + 1;
                whiteRun = 0;
            } else if (gray.at<uchar>(H - 1 - y, x) > 230) {
                whiteRun++;
                if (whiteRun > maxGap) break;
            } else {
                cut = y + 1;
                whiteRun = 0;
            }
        }
        for (int y = 0; y < cut; ++y) setWhite(H - 1 - y, x);
    }

    // --- 3. 扫描左边缘 ---
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) {
                whiteRun = 0;
                continue;
            }
            
            if (gray.at<uchar>(y, x) < darkThr) {
                cut = x + 1;
                whiteRun = 0;
            } else if (gray.at<uchar>(y, x) > 230) {
                whiteRun++;
                if (whiteRun > maxGap) break;
            } else {
                cut = x + 1;
                whiteRun = 0;
            }
        }
        for (int x = 0; x < cut; ++x) setWhite(y, x);
    }

    // --- 4. 扫描右边缘 ---
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, W - 1 - x) > 0) {
                whiteRun = 0;
                continue;
            }
            
            if (gray.at<uchar>(y, W - 1 - x) < darkThr) {
                cut = x + 1;
                whiteRun = 0;
            } else if (gray.at<uchar>(y, W - 1 - x) > 230) {
                whiteRun++;
                if (whiteRun > maxGap) break;
            } else {
                cut = x + 1;
                whiteRun = 0;
            }
        }
        for (int x = 0; x < cut; ++x) setWhite(y, W - 1 - x);
    }

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process