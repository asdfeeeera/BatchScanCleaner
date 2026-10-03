#pragma once

#include <opencv2/core.hpp>

namespace process {

struct EnhanceResult
{
    cv::Mat image;
    double paperGray = 0.0;
    double darkGray = 0.0;
    int enhancedPixels = 0;
    bool ok = false;
    bool skipped = false;
};

struct EnhanceOptions
{
    int strengthLevel = 1;
    double gamma = 1.6;
    bool protectColor = true;
    int targetDarkGray = 0;
    int targetPaperGray = 255;
    int colorSaturationThreshold = 40;

    // ★ 自动跳过阈值：
    //   文字灰度 <= 此值 → 文字已够黑，跳过加深
    //   文字灰度 >  此值 → 文字偏淡，执行加深
    int minDarkGrayToEnhance = 110;
};

class Enhance
{
public:
    static EnhanceResult enhanceText(const cv::Mat &src,
                                      const EnhanceOptions &options = EnhanceOptions());

private:
    static double estimatePaperGray(const cv::Mat &gray);
    static double estimateDarkGray(const cv::Mat &gray);
    static void buildColorProtectMask(const cv::Mat &src,
                                       cv::Mat &colorMask,
                                       int saturationThreshold);
};

} // namespace process