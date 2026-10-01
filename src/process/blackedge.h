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
    bool ok = false;
};

struct BlackEdgeOptions
{
    // 暗度阈值（0-255），像素灰度低于此值视为暗
    int darkThreshold = 120;

    // 一行中暗像素占比超过此值，视为黑边行
    double darkRatio = 0.6;

    // 最多扫描的深度（占页面的百分比），避免误伤正文
    double maxScanRatio = 0.15;

    // 填白还是裁掉，默认填白不裁切
    bool fillWhite = true;
};

class BlackEdge
{
public:
    // 去除四边黑边，默认只填白不裁切
    static BlackEdgeResult removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options = BlackEdgeOptions());

private:
    // 检测上边黑边宽度
    static int detectTop(const cv::Mat &gray, const BlackEdgeOptions &options);
    // 检测下边黑边宽度
    static int detectBottom(const cv::Mat &gray, const BlackEdgeOptions &options);
    // 检测左边黑边宽度
    static int detectLeft(const cv::Mat &gray, const BlackEdgeOptions &options);
    // 检测右边黑边宽度
    static int detectRight(const cv::Mat &gray, const BlackEdgeOptions &options);
};

} // namespace process