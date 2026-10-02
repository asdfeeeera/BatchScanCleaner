#pragma once

#include <opencv2/core.hpp>

namespace process {

struct EnhanceResult
{
    cv::Mat image;
    double paperGray = 0.0;      // 估算的纸张灰度
    double darkGray = 0.0;       // 估算的文字暗部灰度
    int enhancedPixels = 0;      // 被加深的像素数
    bool ok = false;
    bool skipped = false;
};

struct EnhanceOptions
{
    // 强度：0=轻微, 1=标准, 2=强力
    int strengthLevel = 1;

    // 保护彩色内容：彩色像素不做加深
    bool protectColor = true;

    // 目标文字灰度（越小越黑）
    int targetDarkGray = 0;

    // 背景保持的灰度（越大越白）
    int targetPaperGray = 255;

    // 彩色饱和度阈值：超过此值认为彩色
    int colorSaturationThreshold = 40;
};

class Enhance
{
public:
    static EnhanceResult enhanceText(const cv::Mat &src,
                                      const EnhanceOptions &options = EnhanceOptions());

private:
    // 估算纸张灰度（亮部 90 百分位）
    static double estimatePaperGray(const cv::Mat &gray);

    // 估算文字暗部灰度（暗部 5 百分位）
    static double estimateDarkGray(const cv::Mat &gray);

    // 构建彩色保护掩膜
    static void buildColorProtectMask(const cv::Mat &src,
                                       cv::Mat &colorMask,
                                       int saturationThreshold);
};

} // namespace process