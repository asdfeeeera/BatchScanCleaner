#pragma once

#include <opencv2/core.hpp>

namespace process {

struct BlackEdgeResult
{
    cv::Mat image;
    int filledPixels = 0;
    double detectedAreaRatio = 0.0;
    bool ok = false;
    bool skipped = false;
};

struct BlackEdgeOptions
{
    int blockSize = 51;              // 自适应二值化窗口
    double adaptiveC = 10.0;         // 自适应二值化常数
    int morphSize = 5;               // 形态学核大小
    double minAreaRatio = 0.3;       // 纸张最小面积占比
    double edgeMarginRatio = 0.02;   // 边缘贴合判定
    double approxEpsilonRatio = 0.02;// 多边形近似精度
    int expandPixels = 3;            // 掩膜外扩像素
};

class BlackEdge
{
public:
    static BlackEdgeResult removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options = BlackEdgeOptions());
};

} // namespace process