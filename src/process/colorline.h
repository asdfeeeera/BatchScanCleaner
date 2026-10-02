#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace process {

// 检测到的彩色细线
struct ColorLineItem
{
    cv::Rect boundingBox;   // 外接矩形
    double length = 0.0;    // 长度（像素）
    double aspectRatio = 0.0; // 长宽比
    cv::Scalar color;       // 平均 BGR 颜色
    int area = 0;           // 面积
};

struct ColorLineResult
{
    cv::Mat image;                          // 处理后的图像
    cv::Mat markedImage;                    // 带标记的图像（黄框）
    std::vector<ColorLineItem> items;       // 检测到的细线列表
    int clearedPixels = 0;                  // 清除的像素数
    bool ok = false;
    bool skipped = false;
};

struct ColorLineOptions
{
    // 彩色饱和度阈值（HSV 的 S 通道）
    // 高于此值视为彩色像素
    int saturationThreshold = 50;

    // 明度阈值（HSV 的 V 通道）
    // 高于此值才算"明显的彩色"
    int valueThreshold = 60;

    // 最小长度（像素）
    int minLength = 50;

    // 最小长宽比
    double minAspectRatio = 5.0;

    // 最大面积（像素），超过此值认为不是细线
    int maxArea = 2000;

    // 检测模式：true=只检测，false=检测并清除
    bool detectOnly = true;

    // 排除边缘区域的比例（避免把黑边误判为彩色细线）
    double edgeMarginRatio = 0.02;
};

class ColorLine
{
public:
    // 检测彩色细线
    static ColorLineResult detect(const cv::Mat &src,
                                   const ColorLineOptions &options = ColorLineOptions());

    // 检测并清除
    static ColorLineResult clear(const cv::Mat &src,
                                  const ColorLineOptions &options = ColorLineOptions());

private:
    // 构建彩色掩膜
    static void buildColorMask(const cv::Mat &bgr,
                                cv::Mat &colorMask,
                                int satThreshold,
                                int valThreshold);
};

} // namespace process