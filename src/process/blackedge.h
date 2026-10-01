#pragma once

#include <opencv2/core.hpp>

namespace process {

struct BlackEdgeResult
{
    cv::Mat image;                      // 处理后的图像
    int topPixels = 0;                  // 上边去除像素
    int bottomPixels = 0;               // 下边去除像素
    int leftPixels = 0;                 // 左边去除像素
    int rightPixels = 0;                // 右边去除像素
    double paperGray = 0.0;             // 估算的纸张灰度
    double darkThreshold = 0.0;         // 黑边判定阈值
    bool ok = false;
    bool skipped = false;
};

struct BlackEdgeOptions
{
    // 纸张灰度估算：中间区域比例（0.6 = 中间 60% 宽高）
    double paperSampleRatio = 0.6;

    // 纸张灰度估算：百分位（0.9 = 90 百分位）
    double paperPercentile = 0.9;

    // 黑边阈值：纸张灰度 × 此比例
    // 0.75 表示暗于纸张 75% 才算黑边
    double darkRatio = 0.75;

    // 单边最多扫描深度（占该边长度比例）
    double maxScanRatio = 0.30;

    // 一行/一列中，暗像素占比达到多少算黑边
    double darkPixelRatio = 0.50;

    // 间隙容忍：连续 N 行/列不满足就停
    int gapTolerance = 3;

    // 斜边平滑核大小（0 = 不平滑）
    int smoothKernelSize = 5;

    // 填白还是裁切
    bool fillWhite = true;
};

class BlackEdge
{
public:
    static BlackEdgeResult removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options = BlackEdgeOptions());

private:
    static double estimatePaperGray(const cv::Mat &gray, const BlackEdgeOptions &options);

    static int scanTop(const cv::Mat &gray,
                       double darkThreshold,
                       const BlackEdgeOptions &options);

    static int scanBottom(const cv::Mat &gray,
                          double darkThreshold,
                          const BlackEdgeOptions &options);

    static int scanLeft(const cv::Mat &gray,
                        double darkThreshold,
                        const BlackEdgeOptions &options);

    static int scanRight(const cv::Mat &gray,
                         double darkThreshold,
                         const BlackEdgeOptions &options);

    // 用形态学平滑掩膜边缘（去斜边锯齿）
    static void smoothMask(cv::Mat &mask, int kernelSize);
};

} // namespace process