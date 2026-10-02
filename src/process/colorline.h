#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace process {

// 检测到的彩色细线（一行或一列）
struct ColorLineItem
{
    bool horizontal = true;      // true=水平线, false=垂直线
    int position = 0;            // 起始坐标（水平线=y, 垂直线=x）
    int thickness = 1;           // 厚度（像素）
    double colorRatio = 0.0;     // 彩色像素占比
    cv::Scalar color;            // 平均颜色
};

struct ColorLineResult
{
    cv::Mat image;
    cv::Mat markedImage;
    std::vector<ColorLineItem> items;
    int clearedPixels = 0;
    bool ok = false;
    bool skipped = false;
};

struct ColorLineOptions
{
    // 弱彩色阈值
    int channelDiffThreshold = 15;

    // 明度阈值
    int valueThreshold = 80;

    // 行/列彩色占比阈值
    double colorRatioThreshold = 0.60;

    // 细线最大厚度（像素）
    int maxThickness = 20;

    // 排除边缘比例
    double edgeMarginRatio = 0.02;

    bool detectOnly = true;
};

class ColorLine
{
public:
    static ColorLineResult detect(const cv::Mat &src,
                                   const ColorLineOptions &options = ColorLineOptions());

    static ColorLineResult clear(const cv::Mat &src,
                                  const ColorLineOptions &options = ColorLineOptions());

private:
    static void buildColorMask(const cv::Mat &bgr,
                                cv::Mat &colorMask,
                                const ColorLineOptions &options);
};

} // namespace process