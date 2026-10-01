#pragma once

#include <opencv2/core.hpp>

namespace process {

struct BlackEdgeResult
{
    cv::Mat image;
    int topPixels = 0;
    int bottomPixels = 0;
    int leftPixels = 0;
    int rightPixels = 0;
    bool ok = false;
};

struct BlackEdgeOptions
{
    // 边缘行/列的平均灰度低于整图平均灰度的多少倍，才算黑边
    // 0.75 表示比整图平均暗 25% 以上
    double relativeDarkRatio = 0.75;

    // 最多扫描的深度（占页面比例），避免误伤正文
    double maxScanRatio = 0.25;

    // 填白还是裁切
    bool fillWhite = true;
};

class BlackEdge
{
public:
    static BlackEdgeResult removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options = BlackEdgeOptions());

private:
    static int detectTop(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean);
    static int detectBottom(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean);
    static int detectLeft(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean);
    static int detectRight(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean);
};

} // namespace process